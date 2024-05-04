from datasets import Dataset, concatenate_datasets
from typing import Dict, List
import numpy as np
import random


class DataMixture:
    def __init__(self, dataset_dict: Dict[str, Dataset], batch_schedules: Dict = None):
        self.datasets = dataset_dict  
        self.batch_schedules = batch_schedules
        self.dataset_random_indices = {name: np.random.permutation(len(dataset)) for name, dataset in dataset_dict.items()}
        self.dataset_remain_size = {name: 0 for name, _ in dataset_dict.items()}
    
    # Generate a batch's indices according to a given data_ratio
    def gen_batch_indices(self, data_ratios: Dict[str, float], batch_size: int) -> Dict[str, List[int]]:
        """"
        data_ratios: {"bs1": 0.5, "bs2": 0.5, "bs": 0}
        """
        
        sample_counts = {key: int(np.floor(ratio * batch_size)) for key, ratio in data_ratios.items()}
        
        # If allocated_samples are less than batch_size due to np.floor
        allocated_samples = sum(sample_counts.values())
        num_remaining_samples = batch_size - allocated_samples
        if num_remaining_samples > 0: 
            remaining_samples = random.choices(list(data_ratios.keys()), 
                                               weights=[prob for _, prob in data_ratios.items()], 
                                               k=num_remaining_samples)
            for key in remaining_samples:
                sample_counts[key] += 1

        batch_indices = {}
        for ds_name, num_samples in sample_counts.items():
            start_index = self.dataset_remain_size[ds_name]
            end_index = start_index + num_samples
            assert end_index <= len(self.dataset_random_indices[ds_name]), f"The samples of the dataset {ds_name} are exhausted!"
            batch_indices[ds_name] = self.dataset_random_indices[ds_name][start_index:end_index]
            self.dataset_remain_size[ds_name] += num_samples
    
        return batch_indices

    def gen_batch_samples(self, batch_indices):
        batch_samples = []
        for name, indices in batch_indices.items():
            if len(indices) == 0:
                continue
            samples = self.datasets[name].select(indices)
            batch_samples.append(samples)
        return concatenate_datasets(batch_samples).shuffle(keep_in_memory=True)

if __name__ == "__main__":
    ds1 = Dataset.from_dict({"a": [f"ds1_{i}" for i in range(100)]})
    ds2 = Dataset.from_dict({"a": [f"ds2_{i}" for i in range(100)]})
    ds3 = Dataset.from_dict({"a": [f"ds3_{i}" for i in range(100)]})
    batch_size = 10
    datasets = {
        "dataset_1": ds1,
        "dataset_2": ds2,
        "dataset_3": ds3,
    }

    batch_schedules = [
        (10, {'dataset_1': 0.5, 'dataset_2': 0.5, 'dataset_3':0.0}),
        (5, {'dataset_1': 0.0, 'dataset_2': 0.5, 'dataset_3':0.5})
    ]
    data_mixture = DataMixture(datasets, batch_schedules)
    for batch_num, data_ratios in batch_schedules:
        for _ in range(batch_num):
            batch_indices = data_mixture.gen_batch_indices(data_ratios, batch_size)
            batch_samples = data_mixture.gen_batch_samples(batch_indices)
            print(batch_samples['a'])