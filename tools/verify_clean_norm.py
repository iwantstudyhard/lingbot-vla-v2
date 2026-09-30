"""Deterministically recompute clean RoboTwin statistics from raw Parquet.

Matches the official absolute-action, merged 50-step chunk convention,
including repeated terminal actions. No videos, augmented data or rollouts.
Quantiles are exact weighted empirical quantiles rather than order-dependent
running histograms. Reports that algorithm difference explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_weights(length, horizon):
    if length < 1 or horizon < 1:
        raise ValueError("Positive episode length and horizon required")
    weights = np.minimum(np.arange(1, length + 1, dtype=np.int64), horizon)
    weights[-1] = length * horizon - int(weights[:-1].sum())
    assert int(weights.sum()) == length * horizon
    return weights


def weighted_stats(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(weights) or not np.isfinite(values).all():
        raise ValueError("Invalid/non-finite values or weights")
    if np.any(weights <= 0):
        raise ValueError("Weights must be positive")
    total = int(weights.sum())
    mean = np.sum(values * weights[:, None], axis=0) / total
    variance = np.sum((values - mean) ** 2 * weights[:, None], axis=0) / total
    result = {"mean": mean, "std": np.sqrt(variance), "min": values.min(axis=0), "max": values.max(axis=0)}
    for probability, key in ((0.01, "q01"), (0.99, "q99"), (0.02, "q02"), (0.98, "q98")):
        quantiles = []
        for dimension in range(values.shape[1]):
            order = np.argsort(values[:, dimension], kind="stable")
            cumulative = np.cumsum(weights[order])
            index = np.searchsorted(cumulative, probability * total, side="left")
            quantiles.append(values[order[index], dimension])
        result[key] = np.asarray(quantiles)
    return {key: value.tolist() for key, value in result.items()}


def verify(root, reference, horizon=50):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["total_frames"] != 548893 or info["total_episodes"] != 2500:
        raise ValueError("Not the expected full clean-50 dataset (548893 frames / 2500 episodes)")
    episodes = []
    metadata_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    for path in metadata_files:
        episodes.extend(pq.read_table(path, columns=["episode_index", "length", "dataset_from_index", "dataset_to_index"]).to_pylist())
    episodes.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Episode metadata incomplete")
    states, actions, indices, episode_ids, frames = [], [], [], [], []
    data_files = sorted((root / "data").rglob("*.parquet"))
    for path in data_files:
        table = pq.read_table(path, columns=["observation.state", "action", "index", "episode_index", "frame_index"])
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        indices.append(table["index"].to_numpy())
        episode_ids.append(table["episode_index"].to_numpy())
        frames.append(table["frame_index"].to_numpy())
    states, actions = np.concatenate(states), np.concatenate(actions)
    indices, episode_ids, frames = np.concatenate(indices), np.concatenate(episode_ids), np.concatenate(frames)
    if states.shape != (info["total_frames"], 14) or actions.shape != states.shape:
        raise ValueError("Unexpected state/action shape")
    if not np.array_equal(indices, np.arange(info["total_frames"])):
        raise ValueError("Data rows not contiguous")
    weights = np.empty(len(indices), dtype=np.int64)
    previous_end = 0
    for episode in episodes:
        start, end, length = episode["dataset_from_index"], episode["dataset_to_index"], episode["length"]
        if start != previous_end or end - start != length:
            raise ValueError("Non-contiguous episode data range")
        if not np.all(episode_ids[start:end] == episode["episode_index"]) or not np.array_equal(frames[start:end], np.arange(length)):
            raise ValueError("Episode/frame data disagreement")
        weights[start:end] = chunk_weights(length, horizon)
        previous_end = end
    if previous_end != len(indices):
        raise ValueError("Incomplete episode coverage")
    arm = list(range(6)) + list(range(7, 13))
    gripper = [6, 13]
    result = {"count": len(indices), "norm_stats": {}}
    for name, values, weighting in (("observation.state", states, np.ones(len(indices), dtype=np.int64)), ("action", actions, weights)):
        result["norm_stats"][name + ".arm.position"] = weighted_stats(values[:, arm], weighting)
        result["norm_stats"][name + ".effector.position"] = weighted_stats(values[:, gripper], weighting)
    original = json.loads(Path(reference).read_text(encoding="utf-8"))
    if original.get("count") != len(indices) or set(original["norm_stats"]) != set(result["norm_stats"]):
        raise ValueError("Reference count/keys mismatch")
    comparisons = {}
    passed = True
    for key, stats in result["norm_stats"].items():
        low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
        span = np.maximum(high - low, 1)
        comparisons[key] = {}
        for field, values in stats.items():
            old, new = np.asarray(original["norm_stats"][key][field]), np.asarray(values)
            if old.shape != new.shape or not np.isfinite(old).all():
                raise ValueError(f"Bad reference dimensions/values: {key}.{field}")
            tolerance = 0.005 * span if field.startswith("q") else 0.0001 * span
            difference = np.abs(old - new)
            okay = bool(np.all(difference <= tolerance))
            passed &= okay
            comparisons[key][field] = {"max_abs_diff": float(difference.max()), "within_tolerance": okay}
    sources = [{"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": file_hash(path)}
               for path in [root / "meta/info.json", *metadata_files, *data_files]]
    provenance = {
        "dataset_root": str(root), "count": len(indices), "episodes": len(episodes), "horizon": horizon,
        "state_rows": len(indices), "action_chunk_elements": int(weights.sum()),
        "method": "float64 weighted moments + exact weighted empirical quantiles; official merged absolute-action chunk semantics",
        "quantile_difference": "Reference uses adaptive 5000-bin histogram; exact empirical quantiles can differ",
        "comparison_passed": passed, "comparison": comparisons,
        "reference_sha256": file_hash(reference), "data_sources": sources,
        "source_fingerprint": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "origin_claim": "User-provided official clean copy; no random directory read. Dataset provenance still requires official source confirmation.",
    }
    result["verification"] = {key: provenance[key] for key in ("count", "episodes", "horizon", "method", "source_fingerprint")}
    return result, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reference", default="assets/norm_stats/robotwin_clean_only.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    for path in (args.output, args.report):
        if Path(path).exists():
            raise ValueError(f"Refusing to overwrite: {path}")
    stats, report = verify(args.dataset_root, args.reference)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["comparison_passed"]:
        raise ValueError(f"Normalization comparison failed; inspect {args.report}; no new stats written")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": stats["count"], "comparison_passed": True, "output": args.output, "report": args.report}))


if __name__ == "__main__":
    main()
