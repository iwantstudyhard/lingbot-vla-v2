import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingbotvla.utils.arguments import workspace_path

from huggingface_hub import snapshot_download


"""
python3 scripts/download_hf_model.py --repo_id deepseek-ai/Janus-1.3B --local_dir models
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="deepseek-ai/Janus-1.3B")
    parser.add_argument("--local_dir", type=str, default="models")
    parser.add_argument("--local_dir_use_symlinks", type=bool, default=False)
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = str(workspace_path(args.local_dir))
    local_dir_use_symlinks = args.local_dir_use_symlinks

    snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(local_dir, repo_id.split("/")[1]),
        local_dir_use_symlinks=local_dir_use_symlinks,
    )
