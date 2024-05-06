import argparse
from huggingface_hub import snapshot_download
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from typing import Tuple, Dict, Union, List
from copy import deepcopy
from multiprocessing import cpu_count

def aggregate_wildchat_convs(data_point):
    return '\n'.join([conv['content'] for conv in data_point['conversation']])

def supervised_tokenize_pretrain(
    data_point: Dict[str, str], tokenizer: LlamaTokenizer, aggr_func
) -> Dict[str, Union[int, str, List[int]]]:
    assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
        "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
        "add <bos> and <eos> manually later"
    )
    if aggr_func:
        text = aggregate_wildchat_convs(data_point)
    else:
        text = data_point['text']
    full_text = tokenizer.bos_token + text + tokenizer.eos_token
    tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = deepcopy(tokens)
    return {'input_ids': tokens, 'labels': labels}

# TODO: add tokenization for SFT Dataset

def main(args):
    tokenizer_path = args.tokenizer
    dataset_dir = args.source_dir
    target_dir = args.output_dir
    # _ft: tokenized data for fine-tuning. Loss is computed for each token.
    output_dir = f'{target_dir}/{dataset_dir.split("/")[-1]}_{tokenizer_path.split("/")[-1]}_ft'
    num_proc = cpu_count() // 2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    dataset = load_dataset(dataset_dir, split=args.split)
    # dataset = load_from_disk(dataset_dir)
    # dataset = dataset.select_columns(["text"])
    dataset = dataset.map(
            function=supervised_tokenize_pretrain,
            fn_kwargs={"tokenizer": tokenizer, "aggr_func": aggregate_wildchat_convs},
            keep_in_memory=False,
            num_proc=num_proc,
        )
    
    dataset.save_to_disk(output_dir, num_proc=num_proc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize Given Dataset')
    parser.add_argument('--tokenizer', type=str, required=True, help="Path to Hugging Face's Model Repo/Local Tokenizer")
    parser.add_argument('--source_dir', type=str, required=True, help='Local directory to the source raw dataset')
    parser.add_argument('--split', type=str, default='train', help='Split of the dataset to tokenize')
    parser.add_argument('--output_dir', type=str, default='./tokenized_datasets', help='Output directory for tokenized dataset')
    args = parser.parse_args()
    main(args)
    