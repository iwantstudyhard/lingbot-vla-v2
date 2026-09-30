"""Read-only preflight shared by independent clean-training launchers."""

from datetime import datetime
import json
import os
from pathlib import Path
import uuid

import yaml

from lingbotvla.utils.episode_boundaries import bounded_timestamps
from lingbotvla.utils.normalization_contract import semantic_hash
from tools.verify_clean_norm import file_hash
from lingbotvla.utils.arguments import workspace_path

REPO_ROOT = workspace_path(".")


def validate_hf(weights):
    weights = resolve_path(weights, REPO_ROOT)
    if not (weights / "config.json").is_file():
        raise ValueError(f"Missing HF config: {weights}")
    if not any(weights.glob("*.safetensors")) and not (weights / "pytorch_model.bin").is_file():
        raise ValueError(f"Missing HF weights: {weights}")
    for index in weights.glob("*.index.json"):
        manifest = json.loads(index.read_text(encoding="utf-8"))
        for name in set(manifest.get("weight_map", {}).values()):
            shard = (weights / name).resolve()
            if weights not in shard.parents or not shard.is_file():
                raise ValueError(f"Missing/unsafe HF shard: {name}")
    return weights


def resolve_path(value, repo_root):
    return workspace_path(value, repo_root)


def preflight(config, repo_root=REPO_ROOT):
    import pyarrow.parquet as pq
    data = config["data"]
    manifest = resolve_path(data["train_path"], repo_root)
    entries = [line.split() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != 1 or len(entries[0]) != 2 or entries[0][0] != "robotwin":
        raise ValueError("Exactly one local official-clean robotwin dataset is required")
    root = resolve_path(entries[0][1], repo_root)
    report = json.loads((repo_root / "docs/clean_training/norm_verification.json").read_text(encoding="utf-8"))
    stats_path = resolve_path(data["norm_stats_file"], repo_root)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not report["comparison_passed"] or stats.get("verification", {}).get("source_fingerprint") != report["source_fingerprint"]:
        raise ValueError("Normalization verification provenance mismatch")
    # Cross-platform semantic hashes deliberately ignore CRLF / JSON whitespace.
    canonical = json.loads((repo_root / "assets/norm_stats/robotwin_clean_verified.json").read_bytes())
    if semantic_hash(stats) != semantic_hash(canonical):
        raise ValueError("This launch requires the verified clean normalization, not an edited copy")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["codebase_version"] != "v3.0" or info["total_frames"] != 548893 or info["total_episodes"] != 2500:
        raise ValueError("Expected clean v3: 548893 frames / 2500 episodes")
    if stats["count"] != info["total_frames"] or data["expected_num_frames"] != info["total_frames"]:
        raise ValueError("Actual normalization / dataset count mismatch")
    if not data.get("require_normalization_contract") or not data.get("video_episode_boundary") or data.get("image_augment", False):
        raise ValueError("Use the audited clean configuration (contract, boundaries, no legacy augmentation)")
    expected_paths = {row["path"] for row in report["data_sources"]}
    actual_paths = {"meta/info.json"} | {
        path.relative_to(root).as_posix()
        for directory in ("data", "meta/episodes")
        for path in (root / directory).rglob("*.parquet")
    }
    if actual_paths != expected_paths:
        raise ValueError("Clean source file inventory differs from the independently audited copy")
    for row in report["data_sources"]:
        path = (root / row["path"]).resolve()
        if root not in path.parents or file_hash(path) != row["sha256"]:
            raise ValueError(f"Clean source checksum mismatch: {row['path']}")
    checked = 0
    for file in sorted((root / "meta/episodes").rglob("*.parquet")):
        for episode in pq.read_table(file).to_pylist():
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                prefix = f"videos/observation.images.{camera}"
                bounded_timestamps([0, (episode["length"] - 1) / info["fps"]],
                    episode[f"{prefix}/from_timestamp"], episode[f"{prefix}/to_timestamp"], info["fps"])
                video = root / info["video_path"].format(video_key=f"observation.images.{camera}",
                    chunk_index=episode[f"{prefix}/chunk_index"], file_index=episode[f"{prefix}/file_index"])
                if not video.is_file():
                    raise ValueError(f"Missing clean video: {video}")
                checked += 1
    if checked != 7500:
        raise ValueError("Incomplete episode-camera metadata")
    depth, video = config["train"]["align_params"]["depth"], config["train"]["align_params"]["video"]
    for value in (config["model"]["tokenizer_path"], depth["moge_path"], depth["morgbd_path"],
                  video["ckpt_path"], video["config_path"]):
        if not resolve_path(value, repo_root).exists():
            raise ValueError(f"Missing teacher/tokenizer dependency: {value}")
    print(f"Preflight OK: clean source hashes + {checked} camera segments + normalization + dependency paths")
    return stats_path


def environment(gpus, port):
    devices = [value.strip() for value in gpus.split(",")]
    if not all(value.isascii() and value.isdigit() for value in devices):
        raise ValueError("--gpus requires a non-empty comma-separated list of non-negative integer GPU IDs")
    devices = [str(int(value)) for value in devices]
    if len(set(devices)) != len(devices):
        raise ValueError("--gpus requires distinct GPU IDs")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid master port")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=",".join(devices), MASTER_PORT=str(port),
               NNODES="1", NODE_RANK="0", MASTER_ADDR="127.0.0.1",
               WORKSPACE=str(REPO_ROOT),
               LINGBOT_TRAIN_RUN_ID=os.environ.get("LINGBOT_TRAIN_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    env.pop("LINGBOT_TRAIN_RUN_DIR", None)
    return env


def load_config(path):
    config = yaml.safe_load(resolve_path(path, REPO_ROOT).read_text(encoding="utf-8"))
    tokenizer = os.environ.get("QWEN3VL_DIR") or os.environ.get("QWEN3VL_PATH")
    if tokenizer:
        config.setdefault("model", {})["tokenizer_path"] = str(resolve_path(tokenizer, REPO_ROOT))
    return config
