import torch 
import random
import functools
import torch.nn as nn
import numpy as np
import bitsandbytes as bnb
from transformers import AutoTokenizer, LlamaPreTrainedModel, LlamaTokenizer, LlamaTokenizerFast, AutoConfig
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Sequence
import os
import subprocess
from datetime import datetime, timedelta
import yaml


from codebase.dist_logging import get_dist_logger

logger = get_dist_logger()

def get_model_type_from_config(model_dir):
    return AutoConfig.from_pretrained(model_dir).architectures

def load_tokenizer(tokenizer_dir, train_mode=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
        tokenizer = setup_llama_tokenizer(tokenizer, train_mode)
    return tokenizer

def enable_flash_attn(model_dir):
    model_type = get_model_type_from_config(model_dir)  # such as `['LlamaForCausalLM']`
    if 'LlamaForCausalLM' in model_type:
        setup_llama_flash_attn()

def prepare_for_train(model, model_args):
    if isinstance(model, LlamaPreTrainedModel):
        model = setup_llama_train(model, model_args)
    return model 

def setup_llama_tokenizer(tokenizer: Union[LlamaTokenizer, LlamaTokenizerFast], train_mode):
    # In training, right/left padding side are both OK. But in inference, we need left padding.
    if train_mode:
        tokenizer.add_bos_token = True  
        tokenizer.add_eos_token = True  # 这里好像加了也不会加eos token
        # should not use <eos> as pad token, because this will cause model ignore all <eos> during training and can't stop generating
        tokenizer.pad_token = tokenizer.unk_token  
    else:
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = False
        tokenizer.padding_side = 'left'
    return tokenizer

def setup_llama_train(model: LlamaPreTrainedModel, model_args):
    model.config.use_cache = False 
    if model_args.freeze_non_embed:
        # freeze layers (disable gradients)
        for param in model.parameters(): param.requires_grad = False
        for param in model.lm_head.parameters(): param.requires_grad = True
        for param in model.model.embed_tokens.parameters(): param.requires_grad = True
    return model 

def setup_llama_flash_attn():
    from transformers.models.llama.configuration_llama import LlamaConfig
    original_init = LlamaConfig.__init__
    def new_init(self, *args, **kwargs):
        kwargs.setdefault('_attn_implementation', 'flash_attention_2')
        original_init(self, *args, **kwargs)
    functools.update_wrapper(new_init, original_init)
    LlamaConfig.__init__ = new_init

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_trainable_layers(model):
    # Accessing layers and weights
    for name, module in model.named_modules():
        logger.info(f"Layer: {name}")
        for param_name, param in module.named_parameters():
            logger.info(f"\tWeight: {param_name} | Size: {param.size()} | Trainable: {param.requires_grad}")  
    
# Adapted from https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model, bits):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """

    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)

    return freezed_parameters

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")
    return max_length

def set_model_config(config: Any, args: dict):
    for k, v in args.items():
        setattr(config, k, v)
    return config

class GlobalConfig:
    _instance = None
    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            if config_path is not None:
                try:
                    with open(config_path, 'r') as file:
                        config = yaml.safe_load(file)
                        for key, value in config.items():
                            if isinstance(value, str) and value.lower() == 'none':
                                config[key] = None
                        cls._instance.config = config
                except Exception as e:
                    print(f"Error reading the config file: {e}")
                    cls._instance.config = {}
        return cls._instance

    @staticmethod
    def get_config(config_path=None):
        return GlobalConfig(config_path)._instance.config