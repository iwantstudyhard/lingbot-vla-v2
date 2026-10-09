"""Launch a fresh audited clean baseline, or resume only its own audited run."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from tools.clean_training_common import REPO_ROOT, environment, load_config, preflight, resolve_path, validate_hf
from lingbotvla.utils.normalization_contract import resolve_inference_normalization, semantic_hash
from lingbotvla.utils.checkpoint_layout import resolve_resume_checkpoint
import json


def prepare_command(args, *, announce=True):
    env = environment(args.gpus, args.master_port)
    resume_checkpoint = None
    resume_step = getattr(args, "resume_step", None)
    if resume_step is not None and not args.resume_run:
        raise ValueError("--resume-step requires --resume-run")
    if args.resume_run and getattr(args, "smoke", False):
        raise ValueError("Smoke overrides are for a NEW run, not resume")
    if args.resume_run:
        if args.init_hf or args.output_dir:
            raise ValueError("Resume cannot be combined with new weights/output")
        output = resolve_path(args.resume_run, REPO_ROOT)
        config_path = output / "lingbotvla_cli.yaml"
        config = load_config(config_path)
        if config.get("data", {}).get("stage2_augmentation_config"):
            raise ValueError("Stage1 launcher cannot resume an augmented stage2 run")
        if Path(config["train"]["output_dir"]).resolve() != output:
            raise ValueError("Moved runs require deliberate path migration before resume")
        checkpoints = output / "checkpoints"
        configured_checkpoint = config["train"].get("load_checkpoint_path")
        if configured_checkpoint and resume_step is None:
            raise ValueError("Saved config pins load_checkpoint_path; choose --resume-step explicitly")
        resume_checkpoint = resolve_resume_checkpoint(
            checkpoints, len(env["CUDA_VISIBLE_DEVICES"].split(",")), resume_step)
        # Structural placeholder is sufficient; no weights are read here.
        resolve_inference_normalization(checkpoints / "global_step_0/hf_ckpt", config)
        configured = json.loads(Path(config["data"]["norm_stats_file"]).read_bytes())
        verified = json.loads((REPO_ROOT / "assets/norm_stats/robotwin_clean_verified.json").read_bytes())
        if semantic_hash(configured) != semantic_hash(verified):
            raise ValueError("Resume must preserve exactly the audited clean normalization")
    else:
        config_path = REPO_ROOT / "configs/vla/robotwin/robotwin_clean_stage1.yaml"
        config = load_config(config_path)
        output = resolve_path(args.output_dir, REPO_ROOT) if args.output_dir else (
            resolve_path(env.get("OUTPUT_DIR") or "outputs", REPO_ROOT) / "train_outputs"
            / f"robotwin_clean_stage1_{env['LINGBOT_TRAIN_RUN_ID']}")
        if output.exists():
            raise ValueError(f"Fresh training needs a NEW output directory: {output}")
    weights = validate_hf(args.init_hf or (None if args.resume_run else os.environ.get("MODEL_DIR") or os.environ.get("MODEL_PATH"))
                          or config["model"]["model_path"])
    if output == weights or output in weights.parents or weights in output.parents:
        raise ValueError("Output must not overlap initial weights")
    preflight(config)
    command = ["bash", "train.sh", "tasks/vla/train_lingbotvla.py", str(config_path),
               "--model.model_path", str(weights), "--train.output_dir", str(output),
               "--train.enable_resume", "true" if args.resume_run else "false",
               "--train.training_visualization_output_dir", str(output / "visualizations")]
    if resume_checkpoint is not None:
        command += ["--train.load_checkpoint_path", str(resume_checkpoint)]
        if announce:
            print(f"Pinned split DCP resume checkpoint: {resume_checkpoint}")
    if getattr(args, "smoke", False):
        command += ["--train.max_steps", "5", "--train.save_steps", "5"]
    if announce:
        schedule = 'persisted continuation LR plan' if config['train'].get('continuation_start_step') is not None else 'same optimizer/scheduler'
        print(f"Stage1 {'RESUME (' + schedule + ')' if args.resume_run else 'FRESH (official pretrained initialization)'}")
        print(f"Output: {output}")
        print(shlex.join(command))
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-hf", help="Optional relocated OFFICIAL pretrained initialization")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-run", help="Only a new audited stage1 run, never the historical mixed-stat run")
    parser.add_argument("--resume-step", type=int, help="Pin an exact distributed checkpoint step; requires --resume-run")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs; manually match batch settings in YAML (default: 0)")
    parser.add_argument("--master-port", type=int, default=62500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="NEW 5-step/5-save verification run; never a formal baseline")
    args = parser.parse_args()
    command, env = prepare_command(args)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
