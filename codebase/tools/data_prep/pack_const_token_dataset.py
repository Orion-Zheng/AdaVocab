import argparse
from codebase.dataset.constant_tokens_batch_dataset import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Packing Given Dataset to Constant Length')
    parser.add_argument('--tokenizer', type=str, required=True, help="Path to Hugging Face's Model Repo/Local Tokenizer")
    parser.add_argument('--token_field', type=str, default="input_ids", help="Columns in the dataset that contains tokenized data")
    parser.add_argument('--seq_num_token', type=int, default=2048, help="Number of tokens in each example")
    parser.add_argument('--buffer_size', type=int, default=128, help="Number of examples in buffer when packing")
    parser.add_argument('--source_dir', type=str, required=True, help='Local directory to the source raw dataset')
    parser.add_argument('--output_dir', type=str, default='./tokenized_datasets', help='Output directory for tokenized dataset')
    parser.add_argument('--add_bos', action='store_true', help='Add BOS token to each example')
    parser.add_argument('--add_eos', action='store_true', help='Add EOS token to each example')
    args = parser.parse_args()
    main(args)
    