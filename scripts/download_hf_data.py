import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingbotvla.utils.arguments import workspace_path

from huggingface_hub import snapshot_download


"""
python3 scripts/download_hf_data.py --repo_id HuggingFaceFW/fineweb --local_dir datasets/fineweb/ --allow_patterns sample/10BT/*
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--local_dir", type=str, default="datasets/fineweb/")
    parser.add_argument("--allow_patterns", type=str, default=None)
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = str(workspace_path(args.local_dir))
    allow_patterns = args.allow_patterns

    folder = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
