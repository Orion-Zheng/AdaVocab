import warnings
import random
import torch
from datasets import load_from_disk, Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Sequence
from functools import partial

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

@dataclass
class ConstantTokensCollator(object):
    tokenizer: PreTrainedTokenizer
    batch_size: int
    def __call__(self, tokenized_instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_len = len(tokenized_instances['input_ids']) // self.batch_size
        all_input_ids = tokenized_instances['input_ids'][:max_len*self.batch_size]
        all_labels = tokenized_instances['labels'][:max_len*self.batch_size]
        batch_input_ids = torch.stack([torch.tensor(all_input_ids[i:i+max_len]) for i in range(0, len(all_input_ids), max_len)], dim=0)
        batch_labels = torch.stack([torch.tensor(all_labels[i:i+max_len]) for i in range(0, len(all_labels), max_len)], dim=0)

        return dict(
            input_ids=batch_input_ids,
            labels=batch_labels,
            attention_mask=batch_labels.ne(self.tokenizer.pad_token_id),
        )


class ConstantTokenLengthDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        dataset_token_field='tokens',
        infinite=False,
        seq_num_token=1024,
        buffer_sequences=128,
        wrap_special_token=True,
        wrap_special_token_func=None,
        shuffle=False,
    ):
        self.dataset = dataset
        self.seq_num_token = seq_num_token
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_num_token * buffer_sequences  # max tokens in buffer
        self.dataset_token_field = dataset_token_field
        self.wrap_special_token = wrap_special_token
        self.wrap_special_token_func = wrap_special_token_func
        self.shuffle = shuffle  # 这个真的不会影响性能吗？从curriculum的角度来讲，连续的内容应该连着学才对，打乱也应该在Dataset Example层面打乱，不应该在token之后的层面shuffle

    def __len__(self):
        # TODO: count real length
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        token_buffer = [] 
        while more_examples:
            while True:
                if len(token_buffer) >= self.max_buffer_size:
                    break
                try:
                    one_tokenized_sample = next(iterator)[self.dataset_token_field]
                    if self.wrap_special_token:
                        one_tokenized_sample = self.wrap_special_token_func(one_tokenized_sample)
                    token_buffer.extend(one_tokenized_sample)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            examples = []
            for i in range(0, len(token_buffer), self.seq_num_token):
                input_ids = token_buffer[i : i + self.seq_num_token]
                if len(input_ids) == self.seq_num_token:
                    examples.append(input_ids)
                else:
                    token_buffer = input_ids.copy()
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


if __name__ == "__main__":
    tokenizer_path = "base_models/colossal_llama_2_7b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = Dataset.from_dict({"tokens": [[11]*13, [22]*15, [33]*5, [44]*13, [55]*5]})
    wrap_data_func = lambda input_ids: [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    const_dataset = ConstantTokenLengthDataset(
                        dataset, 
                        dataset_token_field="tokens",
                        seq_num_token=10,
                        buffer_sequences=100,
                        wrap_special_token=True,
                        wrap_special_token_func=wrap_data_func,
                    )
    for idx, data in enumerate(const_dataset):
        print(data)
        if idx >= 10:
            break

    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset,
                                             const_dataset), num_proc=10)
    dataset.save_to_disk('test_dataset')
    