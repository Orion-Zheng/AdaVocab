import argparse

from datasets import load_from_disk
from huggingface_hub import login, snapshot_download

def main(args):
    login(token=args.hf_token)  
    
    # ref: https://huggingface.co/docs/huggingface_hub/v0.13.4/en/guides/upload
    dataset = load_from_disk(args.dataset_path)
    # dataset = dataset.map(...)  # do all your processing here
    dataset.push_to_hub(args.repo_path,
                        private=args.is_private)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_path', type=str, required=True)
    parser.add_argument('--repo_path', type=str, required=True)
    parser.add_argument('--hf_token', type=str, required=True)
    parser.add_argument('--is_private', action='store_false')
    args = parser.parse_args()

    main(args)