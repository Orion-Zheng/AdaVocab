import math
import json
import os
import sys
import numpy as np
from transformers import BitsAndBytesConfig, TrainingArguments, HfArgumentParser
from peft import LoraConfig
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Sequence
from codebase.dist_logging import get_dist_logger

logger = get_dist_logger()
excluded_keys = ['world_gpu_size', 'target_token_per_batch', 'max_token_per_seq']
@dataclass
class LogArgs():
    report_to: str = field(default='wandb')  # enable logging to W&B
    logging_strategy: str = field(default='steps')  
    logging_steps: int = field(default=1)
    run_name: Optional[str] = None  # name of the WandB run
    include_num_input_tokens_seen: Optional[bool] = None  # =True --> raise error when perform multi-gpu training by accelerate (in `gather` operation)

@dataclass
class TrainArgs():
    # ref: https://github.com/huggingface/transformers/blob/831bc25d8fdb85768402f772cf65cc3d7872b211/src/transformers/training_args.py#L155
    optim: str = field(default='paged_adamw_32bit')  # paged_adamw_32bit, paged_adamw_8bit  
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    world_gpu_size: Optional[int] = None  # Auxiliary Parameter, to infer `grad_acc_steps`
    target_token_per_batch: Optional[float] = None # Auxiliary Parameter, to infer `grad_acc_steps` 
    gradient_accumulation_steps: Optional[Union[int, str]] = None
    per_device_train_batch_size: Optional[int] = None
    max_token_per_seq: Optional[int] = None
    num_train_epochs: float = field(default=1.0) 
    max_steps: Optional[int] = None  # Override `num_epochs`
    seed: int = field(default=42)  # `TrainingArguments.seed` and `TrainingArguments.data_seed`(Random seed to be used with data samplers)
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None  # fp16 tends to cause overflow or underflow, so don't use it when context is long
    learning_rate: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    lr_scheduler_kwargs: Optional[Dict] = None  # arguments passed to lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts
    warmup_ratio: Optional[float] = None
    warmup_steps: Optional[int] = None  # Overrides any effect of `warmup_ratio`.
    weight_decay: Optional[float] = None
    max_grad_norm: Optional[float] = 1.0
    gradient_checkpointing: Optional[bool] = None
    gradient_checkpointing_kwargs: Optional[dict] = None
    ddp_backend: Optional[str] = None
    
    def __post_init__(self):
        if self.target_token_per_batch:
            self.target_token_per_batch = int(self.target_token_per_batch)
        if 'adam' not in self.optim:
            setattr(self, 'adam_beta1', None)
            setattr(self, 'adam_beta2', None)
        # target_token_per_batch = world_gpu_size * per_device_train_batch_size * gradient_accumulation_steps * max_token_per_seq
        # Situation 1: 
        # Given `per_device_train_batch_size`, `max_token_per_seq`, `gradient_accumulation_steps`
        if self.gradient_accumulation_steps != 'auto':
            self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
            assert all((self.per_device_train_batch_size,
                        self.gradient_accumulation_steps,
                        self.max_token_per_seq)), 'Please provide complete info about `per_device_train_batch_size`, `gradient_accumulation_steps`, `max_token_per_seq`'
        # Situation 2:
        # Given `per_device_train_batch_size`, `max_token_per_seq`, `world_gpu_size`, `target_token_per_batch` --> infer minimum gradient_accumulation_steps
        else:
            assert all((self.target_token_per_batch,
                        self.world_gpu_size,
                        self.max_token_per_seq,
                        self.per_device_train_batch_size)), '''Please Revisit Hyperparameters Compatibility in the Formula:
                             target_token_per_batch = world_gpu_size * per_device_train_batch_size * gradient_accumulation_steps * max_token_per_seq`'''
            self.gradient_accumulation_steps = math.ceil(
                    self.target_token_per_batch /
                    (self.world_gpu_size * self.per_device_train_batch_size * self.max_token_per_seq)
                    )

@dataclass
class EvalIOArgs():
    output_dir: str
    save_strategy: str = field(default='steps')
    evaluation_strategy: str = field(default='steps')
    dataloader_num_workers: int = field(default=0)  # 0 -> main process
    load_best_model_at_end: Optional[bool] = None  # this will let the Trainer save the best checkpoint(with the best eval result) at the end (used in conjunction with save_total_limit: int, metric_for_best_model: Optional[str])
    save_total_limit: Optional[int] = None  # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`.
    per_device_eval_batch_size: Optional[int] = None
    eval_and_save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    def __post_init__(self):
        if self.eval_and_save_steps:
            self.eval_steps = self.eval_and_save_steps
            self.save_steps = self.eval_and_save_steps
            self.eval_and_save_steps = None  # will be ignored later so that won't be pass to TrainingArguments
        if self.load_best_model_at_end:
            assert self.save_steps % self.eval_steps == 0, '`load_best_model_at_end` requires the `saving_steps` to be a round multiple of the `eval_steps`(make sure each saved checkpoint has a eval result)'
            assert self.save_strategy == self.evaluation_strategy, '`load_best_model_at_end` requires the `save_strategy` needs to be the same as `evaluation_strategy`'

@dataclass
class ModelArgs():
    model_dir: str 
    tokenizer_dir: str
    load_dtype: str = field(default="bfloat16")  # dtype to load model
    use_flash: Optional[bool] = None
    do_train: Optional[bool] = None
    do_eval: Optional[bool] = None
    resume_from_checkpoint: Optional[str] = None  # the path to model checkpoint
    freeze_non_embed: Optional[bool] = None  # freeze output and embed layer
    max_length: Optional[int] = None
    

@dataclass
class DataArgs():
    train_data_dir: str
    eval_data_dir: Optional[str] = None
    input_column: Optional[str] = None

@dataclass
class QuantArgs():
    quant_config_path: Optional[str] = None

@dataclass
class PeftArgs():
    lora_config_path: Optional[str] = None


def get_quant_config(quant_args: QuantArgs) -> Optional[dict]:
    if quant_args.quant_config_path:
        with open(quant_args.quant_config_path, 'r') as f:
            quant_config = json.load(f)
            quant_config = BitsAndBytesConfig(**quant_config)
            return quant_config
    else:
        return None

def get_peft_config(peft_args: PeftArgs) -> Optional[dict]:
    # TODO: Switch to DoRA
    if peft_args.lora_config_path:
        with open(peft_args.lora_config_path, 'r') as f:
            lora_config = json.load(f)
            lora_config = LoraConfig(**lora_config)
            return lora_config
    else:
        return None

def keep_valid_arguments(*args: List[Union[LogArgs, TrainArgs, EvalIOArgs]]) -> List[Dict[str, Any]]:
    # Ignore `None` field in the Dataclass (will let `TrainingArguments` use its default values)
    valid_args = []
    for arg in args:
        arg_dict = asdict(arg)
        arg_valid_dict = {k: v for k, v in arg_dict.items() if (v is not None) and (k not in excluded_keys)}
        valid_args.append(arg_valid_dict)
    return valid_args

def create_train_config(log_args: LogArgs, train_args: TrainArgs, eval_io_args: EvalIOArgs):
    log_args, train_args, eval_io_args = keep_valid_arguments(log_args, train_args, eval_io_args)
    hf_train_args = TrainingArguments(
            # ---- Logging Setting ----
            **log_args,
            # ---- Training Setting ----
            **train_args,
            # ---- Eval and IO(save/load)Setting ----
            **eval_io_args
        )
    return hf_train_args

def check_config_sanity(log_args, train_args, eval_io_args, model_args, data_args, peft_args, quant_args):
    # Deprecated:
    # assert (eval_io_args.eval_steps % train_args.gradient_accumulation_steps == 0) \
    #        and (log_args.logging_steps % train_args.gradient_accumulation_steps == 0), \
    #        '`eval_steps` and `logging_steps` have to be a round multiple of your `gradient_accumulation_steps` since those are tested only when you actually do an update'
    pass

def parse_args():
    parser = HfArgumentParser((LogArgs, TrainArgs, EvalIOArgs, ModelArgs, DataArgs, PeftArgs, QuantArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        log_args, train_args, eval_io_args, model_args, data_args, peft_args, quant_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else: # when using accelerate, 'train.py' | '--args' ... 
        log_args, train_args, eval_io_args, model_args, data_args, peft_args, quant_args = parser.parse_args_into_dataclasses(sys.argv[1:])
    
    check_config_sanity(log_args, train_args, eval_io_args, model_args, data_args, peft_args, quant_args)
    
    peft_config = get_peft_config(peft_args)
    quant_config = get_quant_config(quant_args)
    trainer_config = create_train_config(log_args, train_args, eval_io_args)
    
    model_args.max_length = train_args.max_token_per_seq
    
    logger.info(f"log_args:\n {log_args}")
    logger.info(f"train_args:\n {train_args}")
    logger.info(f"eval_io_args:\n {eval_io_args}")
    logger.info(f"model_args:\n {model_args}")
    logger.info(f"data_args:\n {data_args}")
    logger.info(f"peft_args:\n {peft_args}")
    logger.info(f"quant_args:\n {quant_args}")

    return model_args, data_args, trainer_config, peft_config, quant_config, log_args

if __name__ == '__main__':
    parse_args()