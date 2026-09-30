"""Safe rolling retention for step-numbered training checkpoints."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


_CHECKPOINT_PATTERN = re.compile(r"global_step_(\d+)")
_BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"


def list_step_checkpoints(checkpoint_root: str | Path) -> list[tuple[int, Path]]:
    """Return valid ``global_step_<N>`` directories sorted from oldest to newest."""
    root = Path(checkpoint_root)
    if not root.is_dir():
        return []

    checkpoints: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _CHECKPOINT_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            checkpoints.append((int(match.group(1)), child))
    return sorted(checkpoints, key=lambda item: item[0])


def prune_old_checkpoints(
    checkpoint_root: str | Path,
    max_to_keep: int,
    *,
    protected_paths: Iterable[str | Path] = (),
    preferred_paths: Iterable[str | Path] = (),
) -> list[Path]:
    """Delete old completed checkpoints while keeping at most ``max_to_keep``.

    ``preferred_paths`` count toward the retention quota before remaining slots
    are filled by the newest checkpoints. ``protected_paths`` are transiently
    undeletable (for example while an asynchronous HF conversion is reading
    them), so they may temporarily make the directory exceed the quota. A later
    pruning pass restores the requested limit. ``max_to_keep=0`` disables
    pruning.
    """
    max_to_keep = int(max_to_keep)
    if max_to_keep < 0:
        raise ValueError("max_to_keep must be non-negative")
    if max_to_keep == 0:
        return []

    root = Path(checkpoint_root).resolve()
    checkpoints = list_step_checkpoints(root)
    if len(checkpoints) <= max_to_keep:
        return []

    available_paths = {checkpoint.resolve() for _, checkpoint in checkpoints}
    transiently_protected = {
        Path(path).resolve()
        for path in protected_paths
        if Path(path).resolve() in available_paths
    }
    preferred = {
        Path(path).resolve()
        for path in preferred_paths
        if Path(path).resolve() in available_paths
    }
    keep = set(preferred)
    for _, checkpoint in reversed(checkpoints):
        resolved = checkpoint.resolve()
        if resolved in keep:
            continue
        if len(keep) >= max_to_keep:
            break
        keep.add(resolved)

    removed: list[Path] = []
    for _, checkpoint in checkpoints:
        resolved = checkpoint.resolve()
        if resolved.parent != root or not _CHECKPOINT_PATTERN.fullmatch(resolved.name):
            raise RuntimeError(f"Refusing to prune unexpected checkpoint path: {resolved}")
        if resolved in keep or resolved in transiently_protected:
            continue
        shutil.rmtree(resolved)
        removed.append(resolved)
    return removed


def load_best_checkpoint_record(checkpoint_root: str | Path) -> dict[str, Any] | None:
    """Load a valid best-checkpoint record, or return ``None`` if stale/missing."""
    root = Path(checkpoint_root).resolve()
    record_path = root / _BEST_CHECKPOINT_FILENAME
    if not record_path.is_file():
        return None

    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        step = int(record["step"])
        metric_value = float(record["metric_value"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    checkpoint_path = (root / f"global_step_{step}").resolve()
    if checkpoint_path.parent != root or not checkpoint_path.is_dir() or not math.isfinite(metric_value):
        return None
    record["step"] = step
    record["checkpoint"] = checkpoint_path.name
    record["metric_value"] = metric_value
    return record


def update_best_checkpoint_record(
    checkpoint_root: str | Path,
    *,
    step: int,
    metric_value: float,
    window_start_step: int,
    window_end_step: int,
) -> tuple[dict[str, Any], bool]:
    """Atomically persist a lower-is-better mean-training-loss checkpoint."""
    root = Path(checkpoint_root).resolve()
    metric_value = float(metric_value)
    if not math.isfinite(metric_value):
        raise ValueError(f"Best-checkpoint metric must be finite, got {metric_value!r}")
    checkpoint_path = (root / f"global_step_{int(step)}").resolve()
    if checkpoint_path.parent != root or not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Cannot mark missing checkpoint as best: {checkpoint_path}")

    previous = load_best_checkpoint_record(root)
    if previous is not None and float(previous["metric_value"]) <= metric_value:
        return previous, False

    record: dict[str, Any] = {
        "step": int(step),
        "checkpoint": checkpoint_path.name,
        "metric": "mean_training_loss_since_previous_checkpoint",
        "metric_value": metric_value,
        "window_start_step": int(window_start_step),
        "window_end_step": int(window_end_step),
    }
    record_path = root / _BEST_CHECKPOINT_FILENAME
    tmp_path = root / f".{_BEST_CHECKPOINT_FILENAME}.tmp"
    tmp_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, record_path)
    return record, True


def best_checkpoint_path(checkpoint_root: str | Path) -> Path | None:
    """Return the currently valid best checkpoint directory."""
    record = load_best_checkpoint_record(checkpoint_root)
    if record is None:
        return None
    return Path(checkpoint_root).resolve() / str(record["checkpoint"])
