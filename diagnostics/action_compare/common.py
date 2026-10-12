"""Small, CPU-only helpers shared by the opt-in action diagnosis tools."""
import hashlib
import json
from pathlib import Path

import numpy as np

ARM = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
JOINTS = [f"Left J{i + 1}" for i in range(6)] + ["Left grip"] + [
    f"Right J{i + 1}" for i in range(6)
] + ["Right grip"]
COLORS = {"official": "#2563EB", "ours": "#D97706"}


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def observation_hash(obs):
    """Hash the exact input arrays, their shapes/dtypes, and instruction."""
    digest = hashlib.sha256()
    for key in sorted(obs):
        digest.update(key.encode())
        value = obs[key]
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def checked_actions(value):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 14 or array.shape[0] < 2:
        raise ValueError(f"Expected H x 14 RoboTwin qpos, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Model returned NaN/Inf actions; do not execute them")
    return array


def differences(actions):
    """Discrete increments in rad/action, NOT physical velocity/acceleration."""
    values = checked_actions(actions)[:, ARM]
    first = np.diff(values, axis=0)
    second = np.diff(values, n=2, axis=0)
    return {
        "mean_abs_delta_rad_per_action": float(np.abs(first).mean()),
        "max_abs_delta_rad_per_action": float(np.abs(first).max()),
        "mean_abs_second_difference_rad": (
            float(np.abs(second).mean()) if second.size else None
        ),
    }


def boundary_summary(actions, chunks):
    values = checked_actions(actions)[:, ARM]
    delta = np.abs(np.diff(values, axis=0)).mean(axis=1)
    chunks = np.asarray(chunks)
    if chunks.shape != (len(values),):
        raise ValueError("Chunk ids must correspond to executed actions")
    boundary = chunks[1:] != chunks[:-1]
    return {
        "inside": float(delta[~boundary].mean()) if (~boundary).any() else None,
        "boundary": float(delta[boundary].mean()) if boundary.any() else None,
        "inside_count": int((~boundary).sum()),
        "boundary_count": int(boundary.sum()),
    }
