from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import cpu_count
import numpy as np
from typing import Tuple
import math

if __name__ == "__main__":
    dataset_path = "original_datasets/skypile_2023"
    num_samples = 100
    num_proc = cpu_count() - 2
    seed = 24
    print("Loading Dataset ...")
    dataset = load_from_disk(dataset_path)['train']
    print("Shuffling Dataset ...")
    ds_shuffled = dataset.shuffle(seed=seed).select_columns(['text']).select(range(num_samples))
    ds_shuffled.save_to_disk(f"./original_datasets/skypile_2023_sampled_100_eval")


