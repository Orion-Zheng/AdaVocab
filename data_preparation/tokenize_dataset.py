from huggingface_hub import snapshot_download
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from typing import Tuple, Dict, Union, List
from copy import deepcopy
from multiprocessing import cpu_count

def supervised_tokenize_pretrain(
    data_point: Dict[str, str], tokenizer: LlamaTokenizer
) -> Dict[str, Union[int, str, List[int]]]:
    assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
        "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
        "add <bos> and <eos> manually later"
    )
    full_text = tokenizer.bos_token + data_point['text'] + tokenizer.eos_token
    tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = deepcopy(tokens)
    return {'colossal_tokens': tokens, 'labels': labels}


if __name__ == '__main__':
    tokenizer_path = "base_models/colossal_llama_2_7b"
    # dataset_dir = "original_datasets/skypile_2022_sampled_5000M"
    dataset_dir = "original_datasets/skypile_2023_sampled_100_eval"
    # _ft: tokenized data for fine-tuning. Loss is computed for each token.
    output_dir = f'tokenized_datasets/{dataset_dir.split("/")[-1]}_colossal_ft'
    num_proc = cpu_count() - 2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    dataset = load_from_disk(dataset_dir).select_columns(["text"])
    dataset = dataset.map(
            function=supervised_tokenize_pretrain,
            fn_kwargs={"tokenizer": tokenizer},
            keep_in_memory=False,
            num_proc=num_proc,
        )
    
    dataset.save_to_disk(output_dir, num_proc=num_proc)