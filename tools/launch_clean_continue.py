"""Deadline-budgeted clean continuation: full-state resume, explicit LR plan."""

import argparse
import math
from pathlib import Path
import shlex
import subprocess

from lingbotvla.utils.checkpoint_layout import validate_training_checkpoint
from lingbotvla.utils.continuation_schedule import validate_continuation_plan
from tools.clean_training_common import REPO_ROOT, load_config, resolve_path
from tools.launch_clean_stage1 import prepare_command as prepare_stage1


def prepare_command(args):
    run = resolve_path(args.resume_run, REPO_ROOT)
    config = load_config(run / "lingbotvla_cli.yaml")
    train = config["train"]
    if train.get("allow_partial_checkpoint") or not train.get("resume_dataloader_state", True):
        raise ValueError("Continuation requires strict optimizer and dataloader restoration")
    # Reuse exactly the same clean-only, norm-contract and dependency checks.
    args.init_hf = args.output_dir = None
    args.smoke = False
    command, env = prepare_stage1(args, announce=False)
    checkpoint = Path(command[command.index("--train.load_checkpoint_path") + 1])
    step = int(checkpoint.name.removeprefix("global_step_"))
    world_size = len(env["CUDA_VISIBLE_DEVICES"].split(","))
    expected_batch = train["micro_batch_size"] * world_size * train["gradient_accumulation_steps"]
    if expected_batch != train["global_batch_size"]:
        raise ValueError("Saved batch/accumulation settings do not match --gpus; no automatic batch changes")
    previous_start = train.get("continuation_start_step")
    start = previous_start if previous_start is not None else step
    end = args.until if args.until is not None else (train["max_steps"] if previous_start is not None else 48000)
    warmup = args.warmup_steps if args.warmup_steps is not None else train.get("continuation_warmup_steps", 200)
    peak = args.peak_lr if args.peak_lr is not None else train.get("continuation_peak_lr", 1e-5)
    minimum = args.min_lr if args.min_lr is not None else train.get("continuation_min_lr", 5e-6)
    validate_continuation_plan(start, end, warmup, peak, minimum)
    if not start <= step < end:
        raise ValueError("Checkpoint is outside the requested continuation plan")
    if previous_start is not None:
        requested = {"max_steps": end, "continuation_warmup_steps": warmup,
                     "continuation_peak_lr": peak, "continuation_min_lr": minimum}
        if any(train.get(key) != value for key, value in requested.items()):
            raise ValueError("Continuation plan already exists; resume it without silently changing its LR schedule")
    if peak > train["lr"]:
        raise ValueError("Conservative continuation peak cannot exceed the original configured peak LR")
    pinned_step = train.get("checkpoint_pinned_step")
    pinned_step = start if pinned_step is None else pinned_step
    validate_training_checkpoint(run / "checkpoints" / f"global_step_{pinned_step}", world_size)
    overrides = {
        "max_steps": end, "num_train_epochs": 999999999,
        "continuation_start_step": start, "continuation_warmup_steps": warmup,
        "continuation_peak_lr": peak, "continuation_min_lr": minimum,
        "save_steps": 500, "max_checkpoints_to_keep": 3,
        "keep_best_checkpoint": True, "checkpoint_pinned_step": pinned_step,
    }
    for key, value in overrides.items():
        command += [f"--train.{key}", str(value).lower() if isinstance(value, bool) else str(value)]
    if not math.isfinite(args.seconds_per_step) or args.seconds_per_step <= 0:
        raise ValueError("--seconds-per-step must be positive and finite")
    days = (end - step) * args.seconds_per_step / 86400
    count = (end - step + 499) // 500
    print(f"CLEAN CONTINUATION: full optimizer/data/RNG restore from {checkpoint}")
    print(f"LR plan: origin={start}, warmup={warmup} steps from ACTUAL saved LR, "
          f"peak={peak:g}, cosine floor={minimum:g}, stop={end}")
    print(f"Budget: {end-step} further optimizer steps, approximately {days:.2f} days "
          f"at {args.seconds_per_step:g}s/step, PLUS {count} checkpoint exports and evaluation overhead")
    print(f"Retention: baseline={pinned_step} + best TRAINING-loss checkpoint + latest, within 3 completed checkpoints")
    print("Unchanged: clean data, normalization, BF16 settings, batch, vision training, model and loss")
    print(f"Output: {run}")
    print(shlex.join(command))
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-run", required=True)
    parser.add_argument("--resume-step", type=int, help="Exact full-state checkpoint; later resumes must choose their newest saved step")
    parser.add_argument("--until", type=int, help="TOTAL optimizer steps, not additional steps; first plan defaults to 48000")
    parser.add_argument("--peak-lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--seconds-per-step", type=float, default=34.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--master-port", type=int, default=62500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, env = prepare_command(args)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
