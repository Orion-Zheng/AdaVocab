from codebase.dataset.constant_tokens_batch_dataset import ConstantTokenLengthDataset
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from multiprocessing import cpu_count
from functools import partial
from datasets import Dataset

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


if __name__ == "__main__":
    max_len = 2048
    num_proc = cpu_count() - 3
    tokenizer_path = "experiment_models/tinyllama_expanded_empty"
    dataset_path = "original_datasets/skypile_2022_sampled_50M"
    output_dir = f"tokenized_datasets/skypile_2022_sampled_50M_{max_len}_ft"
    dataset = load_from_disk(dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    wrap_data_func = lambda input_ids: [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    const_dataset = ConstantTokenLengthDataset(
                        dataset, 
                        dataset_token_field="tokens",
                        seq_num_token=max_len,
                        buffer_sequences=100000,
                        wrap_special_token=True,
                        wrap_special_token_func=wrap_data_func,
                    )
    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, const_dataset), num_proc=num_proc)
    dataset.save_to_disk(output_dir)