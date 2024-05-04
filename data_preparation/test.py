from train import load_model
from train_utils.args_parser import ModelArgs, QuantArgs, PeftArgs, get_peft_config, get_quant_config
import torch
from datasets import load_from_disk
# quant_args = get_quant_config(QuantArgs(quant_config_path="config/quant_config/4bit_quant.json"))
# lora_args = get_peft_config(PeftArgs(lora_config_path="config/peft_config/lora.json"))
# model_args = ModelArgs(model_dir="experiment_models/llama_empty_expand",
#                        tokenizer_dir="experiment_models/llama_empty_expand")
# model = load_model(model_args, quant_args, lora_args)
# print(model)
# print(model.config.use_cache)

print(len(load_from_disk('tokenized_datasets/skypile_2022_sampled_5000M_colossal_ft')))