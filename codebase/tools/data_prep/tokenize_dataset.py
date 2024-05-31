import argparse
from huggingface_hub import snapshot_download
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from typing import Tuple, Dict, Union, List
from copy import deepcopy
from multiprocessing import cpu_count

def aggregate_wildchat_convs(data_point):
    return '\n'.join([conv['content'] for conv in data_point['conversation']])

def gen_multi_turn_text(chat, tokenizer, split_role='assistant', split_each_turns=True):
    prev_idx = 0
    for i in range(len(chat)):
        if chat[i]['role'] == split_role:
            prompt = chat[prev_idx:i]
            answer = chat[i]
            full_conv = prompt + [answer]
            if split_each_turns:
                prev_idx = i + 1
            sft_full = tokenizer.apply_chat_template(full_conv, tokenize=False)
            sft_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            input_text = sft_full[:len(sft_prompt)]
            output_text = sft_full[len(sft_prompt):]
            yield {'input': input_text, 'output': output_text}

def tokenize_multi_turn_conv(chat, tokenizer, split_role='assistant', split_each_turns=True, IGNORE_INDEX=-100):
    all_turns_data = gen_multi_turn_text(chat, tokenizer, split_role=split_role, split_each_turns=split_each_turns)
    input_ids, labels = [], []
    for sample in all_turns_data:
        query_tokens = tokenizer(sample['input'], add_special_tokens=False)['input_ids'] 
        response_tokens = tokenizer(sample['output'], add_special_tokens=False)['input_ids']
        input_ids.extend(query_tokens + response_tokens)
        labels.extend([IGNORE_INDEX]*len(query_tokens) + response_tokens)
    return {'input_ids': input_ids, 'labels': labels}

def tokenize_sft(
    data_point: Dict[str, str], tokenizer: LlamaTokenizer, conv_field='conversation', split_role='assistant', split_each_turns=True, IGNORE_INDEX=-100
) -> Dict[str, Union[int, str, List[int]]]:
    chat = data_point[conv_field]  # extract conversation from the data point
    return tokenize_multi_turn_conv(chat, tokenizer, split_role=split_role, split_each_turns=split_each_turns, IGNORE_INDEX=IGNORE_INDEX)

def tokenize_pretrain(
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


def main(args):
    tokenizer_path = args.tokenizer
    dataset_dir = args.source_dir
    target_dir = args.output_dir
    # _ft: tokenized data for fine-tuning. Loss is computed for each token.
    output_dir = f'{target_dir}/{dataset_dir.split("/")[-1]}_{tokenizer_path.split("/")[-1]}'
    if args.sft:
        output_dir += '_sft'
    else:
        output_dir += '_pretrain'
    num_proc = cpu_count() - 2 

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    dataset = load_dataset(dataset_dir, split=args.split)
    # dataset = load_from_disk(dataset_dir)
    # dataset = dataset.select_columns(["text"])
    if args.sft:
        # make sure tokenizer_config.json has correct `chat_template` field
        dataset = dataset.map(
            function=tokenize_sft,
            fn_kwargs={"tokenizer": tokenizer, "split_role": 'assistant', "split_each_turns": True},
            keep_in_memory=False,
            num_proc=num_proc,
        )
    else:
        dataset = dataset.map(
            function=tokenize_pretrain,
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
    parser.add_argument('--sft', action='store_true', help='Tokenize for SFT Dataset')
    args = parser.parse_args()
    main(args)
    