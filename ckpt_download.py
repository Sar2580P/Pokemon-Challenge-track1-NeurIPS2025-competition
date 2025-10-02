from huggingface_hub import snapshot_download
from metamon import METAMON_CACHE_DIR

print(METAMON_CACHE_DIR)

import torch
model_path="abra/ckpts/policy_weights/policy_epoch_40.pt"
path = snapshot_download(
    repo_id="jakegrigsby/metamon",
    allow_patterns=model_path,  # Replace 'large-il' with your desired model
    local_dir=f"{METAMON_CACHE_DIR}/metamon_models",  # Local directory to save files
    local_dir_use_symlinks=False
)

ckpt_path=f'{METAMON_CACHE_DIR}/metamon_models/{model_path}'
state_dict = torch.load(ckpt_path, map_location="cpu")

print(state_dict.keys())