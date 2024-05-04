from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import cpu_count
import numpy as np
from typing import Tuple
import math

def tokenize_function(example):
    tokens = tokenizer(example['text'], add_special_tokens=False)['input_ids']
    return {'tokens': tokens}

def count_dataset(dataset:Dataset, max_token:int, num_proc: int=1)->Tuple[Dataset, int]:
    # Input:
    # - dataset: containing key 'text'
    # - max_token: max tokens to be sampled
    # Output:
    # - sampled_dataset: containing key 'text' and 'tokens'
    # - sampled_tokens: tokens in sampled_dataset
    total_token_count = 0
    selected_indices = []
    dataset = dataset.map(tokenize_function, num_proc=num_proc,
                          # batched=True, batch_size=10
                          )
    selected_indices = []
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        token_num = len(sample['tokens'])
        total_token_count += token_num
        selected_indices.append(idx)
        if total_token_count >= max_token:
            break

    return dataset.select(selected_indices), total_token_count

def sample_once(dataset, max_token, start_idx, estim_sample_num, curr_token, num_proc: int=1):
    # Output: sampled_ds, end_idx, curr_token, remain_token,
    assert start_idx < len(dataset)
    end_idx = start_idx + estim_sample_num
    if end_idx >= len(dataset):
        end_idx = len(dataset)
    
    estim_ds = dataset.select(range(start_idx, end_idx))
    remain_token = max_token - curr_token
    # Try to get samples containing `remain_token` from `estim_ds`, but won't exceed `remain_token` too much
    new_ds, new_token = count_dataset(estim_ds, remain_token, num_proc=num_proc)
    start_idx += len(new_ds)
    curr_token += new_token
    remain_token = max_token - curr_token
    return new_ds, start_idx, curr_token, remain_token

if __name__ == "__main__":
    tokenizer_path = "base_models/colossal_llama_2_7b"
    dataset_path = "original_datasets/skypile_2022"
    max_token = 5e9
    probe_len = 100
    num_proc = cpu_count() - 2
    seed = 24

    print("Loading Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Loading Dataset ...")
    dataset = load_from_disk(dataset_path)['train']
    print("Shuffling Dataset ...")
    ds_shuffled = dataset.shuffle(seed=seed).select_columns(['text'])

    result_ds = Dataset.from_dict({"text": [],
                                    "tokens": []})

    print(f'Tokenizing first {probe_len} samples to estimate `tokens per sample`...')
    start_idx = 0
    curr_token = 0
    estim_sample_num = None
    sampled_ds, start_idx, curr_token, remain_token = sample_once(ds_shuffled, max_token,
                                                                start_idx, probe_len, curr_token, num_proc)
    result_ds = concatenate_datasets([result_ds, sampled_ds])
    print('The estimation finished.')

    while remain_token > 0:
        if start_idx >= len(ds_shuffled):
            print('The dataset is exhausted')
            break
        token_per_sample = curr_token/(start_idx+1)
        estim_sample_num = math.ceil(remain_token/token_per_sample)
        print(f"Estimated `tokens per sample` = {token_per_sample:.2f}.")
        print(f"Approximate {estim_sample_num} samples are needed for {(max_token/1e6):.0f}M tokens total.")
        sampled_ds, start_idx, curr_token, remain_token = sample_once(ds_shuffled, max_token,
                                                                    start_idx, estim_sample_num,
                                                                    curr_token, num_proc)
        result_ds = concatenate_datasets([result_ds, sampled_ds])
        print(f"Current Data Points: {start_idx+1}", 
              f"Current Tokens {(curr_token/1e6):.0f}M",
              f"Remaining Tokens: {(remain_token/1e6):.0f}M", 
              f"Estimated Data Points To Be Tokenized: {estim_sample_num}")

    result_ds.save_to_disk(f"./original_datasets/skypile_2022_sampled_{(curr_token/1e6):.0f}M")