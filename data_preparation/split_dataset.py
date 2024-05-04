from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocessing import cpu_count

def split_dataset_by_cat(dataset, classify_func, cat_key, num_proc=1):
    # classify_func: classify a sample to a certain category (in `int` or `str`)
    dataset = dataset.map(classify_func, keep_in_memory=False, num_proc=num_proc)
    categories = dataset.unique(cat_key)
    
    ds_dict = DatasetDict({})
    for cat in categories:
        filter_func = lambda x, cat=cat: x[cat_key] == cat
        ds_dict[cat] = dataset.filter(filter_func, num_proc=num_proc)
        
    return ds_dict


if __name__ == '__main__':
    dataset = load_dataset("./original_datasets/SlimPajama_sampled",
                           cache_dir="./cache_dir")
    test_set = dataset['test']
    cat_name = "category"
    classify_func = lambda x: {cat_name: x["meta"]["redpajama_set_name"]}
    subset_dict = split_dataset_by_cat(test_set, classify_func, cat_name)
    subset_dict.save_to_disk('./original_datasets/splitted_slimpajama_6b_test')