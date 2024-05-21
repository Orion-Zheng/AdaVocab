import argparse
from datasets import load_from_disk, Dataset, DatasetDict

def parse_test_size(value):
    value = float(value)
    if value < 1:
        return value  # Return as float
    else:
        return int(value)  # Return as int
    
def split_dataset(dataset_path, test_size=0.2, seed=42):
    """
    Load a dataset from a CSV file, shuffle it, and split it into training and evaluation sets.
    
    Args:
        dataset_path (str): Path to the input dataset file (CSV format).
        test_size (float/int): float -> Proportion of the dataset to include in the evaluation split. int -> number of eval samples
        seed (int): Random seed for shuffling the dataset.
    
    Returns:
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
    """
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    # Shuffle and split the dataset
    shuffled_dataset = dataset.shuffle(seed=seed)
    split_dataset = shuffled_dataset.train_test_split(test_size=test_size)
    
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    return train_dataset, eval_dataset

def save_datasets(train_dataset, eval_dataset, output_dir):
    """
    Save the training and evaluation datasets to disk.
    
    Args:
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        output_dir (str): Directory to save the output datasets.
    """
    train_output_path = f"{output_dir}/train"
    eval_output_path = f"{output_dir}/eval"
    
    train_dataset.save_to_disk(train_output_path)
    eval_dataset.save_to_disk(eval_output_path)
    
    print(f"Train dataset saved to: {train_output_path}")
    print(f"Eval dataset saved to: {eval_output_path}")

def main():
    """
    Main function to parse command-line arguments and split the dataset.
    """
    parser = argparse.ArgumentParser(description="Split a dataset into train and eval sets.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input dataset file (CSV format).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output datasets.")
    parser.add_argument("--test_size", type=parse_test_size, default=0.1, help="Proportion of the dataset to include in the eval split (if float) or number of samples (if int).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset.")

    args = parser.parse_args()
    
    # Split the dataset into training and evaluation sets
    train_dataset, eval_dataset = split_dataset(args.dataset_path, test_size=args.test_size, seed=args.seed)
    
    # Save the split datasets to disk
    save_datasets(train_dataset, eval_dataset, args.output_dir)

if __name__ == "__main__":
    main()
