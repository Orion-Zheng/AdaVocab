import argparse
from huggingface_hub import login, snapshot_download

def main(args):
    login(token=args.hf_token)  

    snapshot_download(repo_id=args.repo_id,
                      repo_type=args.repo_type,
                      local_dir=args.local_dir,
                      local_dir_use_symlinks=args.use_symlinks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a dataset from Hugging Face Hub.')
    parser.add_argument('--repo_id', type=str, required=True, help='Repository ID for the dataset')
    parser.add_argument('--repo_type', type=str, default='dataset', help='Type of the repository (dataset/model)')
    parser.add_argument('--local_dir', type=str, default='./original_datasets', help='Local directory to download the dataset')
    parser.add_argument('--use_symlinks', action='store_true', help='Use symbolic links for the local directory')
    parser.add_argument('--hf_token', type=str, help='Hugging Face login token')

    args = parser.parse_args()
    main(args)
