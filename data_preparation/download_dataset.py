
from huggingface_hub import login, snapshot_download
login(token="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")

snapshot_download(repo_id="OrionZheng/skypile_2022_sampled_5000M",
                  repo_type='dataset',
                  local_dir='./original_datasets',
                  local_dir_use_symlinks=False
                  )