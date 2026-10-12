"""Read-only validation of this repository's split DCP training checkpoints.

DistributedCheckpointer writes separate model/optimizer DCP stores, plus one
extra_state file per rank. HF exports and root-level .metadata are NOT resume
checkpoints. These are structural checks; tensor/state readability is verified
by the actual distributed loader, which must not swallow optimizer failures.
"""

import argparse
import json
from pathlib import Path
import re


def validate_training_checkpoint(path, world_size):
    """Require model, optimizer and every requested rank's training state."""
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("Resume world_size must be a positive integer")
    checkpoint = Path(path).resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Training checkpoint directory missing: {checkpoint}")
    missing = []
    for component in ("model", "optimizer"):
        directory = checkpoint / component
        metadata = directory / ".metadata"
        if not metadata.is_file() or metadata.stat().st_size == 0:
            missing.append(f"{component}/.metadata (missing or empty)")
        if not any(p.is_file() and p.stat().st_size > 0 for p in directory.glob("*.distcp")):
            missing.append(f"{component}/*.distcp (missing or empty)")
    for rank in range(world_size):
        extra = checkpoint / "extra_state" / f"extra_state_rank_{rank}.pt"
        if not extra.is_file() or extra.stat().st_size == 0:
            missing.append(f"extra_state/{extra.name} (missing or empty)")
    if missing:
        raise ValueError(
            f"Incomplete split DCP training checkpoint {checkpoint}: " + "; ".join(missing)
            + ". HF weights alone cannot restore optimizer/scheduler/dataloader. "
              "Do not create fake metadata or restart silently."
        )
    return checkpoint


def resolve_resume_checkpoint(checkpoint_dir, world_size, step=None):
    """Pin one checkpoint; an incomplete latest save requires explicit choice."""
    directory = Path(checkpoint_dir)
    if step is not None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("--resume-step must be a nonnegative integer")
        candidate = directory / f"global_step_{step}"
    else:
        candidates = []
        for path in directory.glob("global_step_*"):
            match = re.fullmatch(r"global_step_(\d+)", path.name)
            if path.is_dir() and match:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise ValueError(f"No distributed checkpoint available for resume: {directory}")
        candidate = max(candidates, key=lambda pair: pair[0])[1]
    try:
        return validate_training_checkpoint(candidate, world_size)
    except ValueError as exc:
        if step is None:
            raise ValueError(f"{exc} Select a known complete save with --resume-step N.") from exc
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="global_step_N directory, NOT hf_ckpt")
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    checkpoint = validate_training_checkpoint(args.checkpoint, args.world_size)
    print(json.dumps({"checkpoint": str(checkpoint), "world_size": args.world_size,
                      "structural_check": "passed", "tensor_load_verified": False}, indent=2))


if __name__ == "__main__":
    main()
