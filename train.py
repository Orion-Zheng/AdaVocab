import os
import sys
import math
import torch
import numpy as np
import random
from transformers import (Trainer, AutoModelForCausalLM,
                          PreTrainedTokenizer,
                          DataCollatorForLanguageModeling,
                          AutoConfig, TrainerCallback,
                          LlamaForCausalLM)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator, init_empty_weights
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Sequence
from transformers.integrations.integration_utils import WandbCallback
from datetime import datetime, timedelta

from codebase.monkey_patch import new_wandb_on_train_end, SafeSavingCallback_NSCC
from codebase.utils import print_trainable_parameters, load_tokenizer, prepare_for_train, enable_flash_attn
from codebase.args_parser import parse_args
from codebase.dist_logging import get_dist_logger
from codebase.model.ada_vocab_llama import AdaVocabLlamaForCausalLM

# can skip if you have already logged in at console by 'wandb login'
# import wandb
# wandb.login(key="")

logger = get_dist_logger()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['WANDB_LOG_MODEL'] = 'true'
IGNORE_INDEX = -100
SAFE_MINUTES = 5

SafeSavingCallback_NSCC.safe_minutes = SAFE_MINUTES
    
@dataclass
class PaddToMaxLenCollator(object):
    # Adapt from: https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/6e8c6c23e51ec8f0cf8a2b1f1633e52edb768e9c/scripts/training/build_dataset.py
    tokenizer: PreTrainedTokenizer
    max_len: int
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Truncate each sequence to `max_len`. If dataset is already packed to max_len, no truncation will be done.
        input_ids, labels = tuple([torch.tensor(instance[key])[:self.max_len] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def load_model(model_args, quant_config=None, peft_config=None, model_class=AutoModelForCausalLM):
    kwargs = {
        "torch_dtype": eval(f"torch.{model_args.load_dtype}")
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    
    if model_args.use_flash:
        enable_flash_attn(model_args.model_dir)

    model = model_class.from_pretrained(model_args.model_dir, **kwargs)
    if model_args.do_train:
        model = prepare_for_train(model, model_args)  # e.g. disable kv cache, freeze some modules(specified in the model_args)
    if quant_config:
        # 1- Cast the layernorm in fp32; 2- making output embedding layer require grads; 3- Add the upcasting of the lm_head to fp32
        model = prepare_model_for_kbit_training(model)
    if peft_config:
        model = get_peft_model(model, peft_config)

    return model

def enable_monkey_patch():
    logger.info('New `on_train_end` is applied to `WandbCallback`')
    WandbCallback.on_train_end = new_wandb_on_train_end

def main():
    enable_monkey_patch()
    
    model_args, data_args, trainer_config, peft_config, quant_config, log_args = parse_args()
    # TODO: add option to not using eval data(considering training arguments)
    logger.info("Load Training Dataset ...")
    train_data = load_from_disk(data_args.train_data_dir)
    logger.info("Load Evaluation Dataset ...")
    eval_data = load_from_disk(data_args.eval_data_dir) if data_args.eval_data_dir else None
    if data_args.input_column:
        train_data = train_data.rename_column(data_args.input_column, "input_ids")
        eval_data = eval_data.rename_column(data_args.input_column, "input_ids") if data_args else None
        
    logger.info(f"Training Data:\n{train_data}")
    logger.info(f"Evaluation Data:\n{eval_data}")
    
    tokenizer = load_tokenizer(model_args.tokenizer_dir, train_mode=model_args.do_train)
    model = load_model(model_args, quant_config, peft_config, AdaVocabLlamaForCausalLM)
    logger.info(f"Model Architecture:\n{model}")
    print_trainable_parameters(model)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data, 
        args=trainer_config,
        data_collator=PaddToMaxLenCollator(tokenizer, model_args.max_length), 
        # callbacks=[SafeSavingCallback_NSCC]  # only for PBS Pro Cluster(e.g. NSCC)
    )

    # Training
    if model_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=model_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)  # e.g {'train_runtime': 112.9526, 'train_samples_per_second': 0.142, 'train_steps_per_second': 0.035, 'train_loss': 9.430782318115234, 'epoch': 0.0, 'num_input_tokens_seen': 14166}
        trainer.save_metrics("train", train_result.metrics)  # all_results.json

    # Evaluation
    if model_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")  # e.g {'eval_loss': 8.76622486114502, 'eval_runtime': 9.493, 'eval_samples_per_second': 10.534, 'eval_steps_per_second': 5.267, 'epoch': 0.0, 'num_input_tokens_seen': 14166}
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)  # eval_results.json

if __name__ == "__main__":
    main()