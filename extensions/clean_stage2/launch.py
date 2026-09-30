"""Preflight an independent, fresh stage-two visual-adaptation run."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from tools.clean_training_common import REPO_ROOT, environment, load_config, preflight, resolve_path, validate_hf
from lingbotvla.utils.normalization_contract import resolve_inference_normalization, semantic_hash
from extensions.clean_stage2.augmentation import load_settings


def prepare_command(args, repo_root=REPO_ROOT):
    env = environment(args.gpus, args.master_port)
    weights = validate_hf(args.init_hf)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    config_path = repo_root / "extensions/clean_stage2/config.yaml"
    config = load_config(config_path)
    verified_path = preflight(config, repo_root)
    load_settings(repo_root / config["data"]["stage2_augmentation_config"])
    source_run = weights.parent.parent.parent
    if not (source_run / "normalization/manifest.json").is_file():
        raise ValueError("Stage2 requires NEW audited clean stage1 weights, not historical mixed-stat runs")
    saved_config = load_config(source_run / "lingbotvla_cli.yaml")
    if saved_config.get("data", {}).get("stage2_augmentation_config"):
        raise ValueError("Select a stage1 baseline, not a previous augmented stage2 experiment")
    source_norm = resolve_inference_normalization(weights, {"data": {"require_normalization_contract": True}})
    if semantic_hash(json.loads(Path(source_norm).read_bytes())) != semantic_hash(json.loads(verified_path.read_bytes())):
        raise ValueError("Source model has incompatible normalization scales")
    output = resolve_path(args.output_dir, repo_root) if args.output_dir else (
        resolve_path(env.get("OUTPUT_DIR") or "outputs", repo_root) / "train_outputs"
        / f"robotwin_clean_stage2_{env['LINGBOT_TRAIN_RUN_ID']}")
    if output.exists():
        raise ValueError(f"Choose a NEW stage2 output directory: {output}")
    if output == weights or weights in output.parents or output in weights.parents:
        raise ValueError("Output must not overlap source weights")
    command = ["bash", "train.sh", "extensions/clean_stage2/train.py", str(config_path),
               "--model.model_path", str(weights), "--train.output_dir", str(output),
               "--train.max_steps", str(args.steps),
               "--train.lr_warmup_ratio", str(min(150, max(1, args.steps // 3)) / args.steps),
               "--train.training_visualization_output_dir", str(output / "visualizations")]
    print("Stage2: HF WEIGHTS ONLY; fresh optimizer/scheduler; same audited normalization")
    print(f"Output: {output}")
    print(shlex.join(command))
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-hf", required=True, help="Selected new audited stage1 checkpoints/global_step_N/hf_ckpt")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--master-port", type=int, default=62510)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, env = prepare_command(args)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
