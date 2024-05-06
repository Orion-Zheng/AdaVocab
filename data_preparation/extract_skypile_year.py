from datasets import load_dataset, DatasetDict, load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocessing import cpu_count
if __name__ == "__main__":
    year = 2023
    num_proc = cpu_count() - 2
    print(f"original_datasets/SkyPile-150B/{year}-*.jsonl")
    dataset = load_dataset("json", 
                            data_files=f"original_datasets/SkyPile-150B/{year}-*.jsonl",
                            cache_dir="./cache_dir",
                            num_proc=num_proc)
    dataset.save_to_disk(f'./original_datasets/skypile_{year}',
                              num_proc=num_proc)