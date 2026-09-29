"""Safe rolling retention for step-numbered training checkpoints."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable


_CHECKPOINT_PATTERN = re.compile(r"global_step_(\d+)")


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
) -> list[Path]:
    """Delete old completed checkpoints while always preserving the newest N.

    ``max_to_keep=0`` disables pruning. Paths used by an asynchronous HF
    conversion can be supplied via ``protected_paths``; they are deferred until
    a later pruning pass.
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

    protected = {Path(path).resolve() for path in protected_paths}
    removed: list[Path] = []
    for _, checkpoint in checkpoints[:-max_to_keep]:
        resolved = checkpoint.resolve()
        if resolved.parent != root or not _CHECKPOINT_PATTERN.fullmatch(resolved.name):
            raise RuntimeError(f"Refusing to prune unexpected checkpoint path: {resolved}")
        if resolved in protected:
            continue
        shutil.rmtree(resolved)
        removed.append(resolved)
    return removed
