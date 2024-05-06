import argparse
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import cpu_count
import numpy as np
from typing import Tuple
import math

def main(args):
    print("Loading Dataset ...")
    dataset = load_from_disk(args.dataset_path)
    print("Shuffling Dataset ...")
    if args.columns:  # Select columns if specified
        dataset = dataset.select_columns(args.columns)
    ds_shuffled = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    ds_shuffled.save_to_disk(args.output_path, num_proc=args.num_proc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output dataset directory')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples to select from the dataset')
    parser.add_argument('--num_proc', type=int, default=1, help='Number of processors to use')
    parser.add_argument('--seed', type=int, default=24, help='Seed for shuffling the dataset')
    parser.add_argument('--columns', nargs='*', default=None, help='Columns to select from the dataset(splitted by space)')
    args = parser.parse_args()
    main(args)
