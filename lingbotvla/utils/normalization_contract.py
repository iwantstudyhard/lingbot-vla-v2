"""Audited normalization snapshots shared by training and deployment."""

import hashlib
import json
import math
import os
from pathlib import Path


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_clean_stats(stats):
    expected = {f"{prefix}.{joint}": dimension
                for prefix in ("action", "observation.state")
                for joint, dimension in (("arm.position", 12), ("effector.position", 2))}
    if set(stats.get("norm_stats", {})) != set(expected):
        raise ValueError("Unexpected clean state/action normalization keys")
    for key, dimension in expected.items():
        for name in ("mean", "std", "min", "max", "q01", "q99", "q02", "q98"):
            values = stats["norm_stats"][key].get(name, [])
            if len(values) != dimension or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"Invalid normalization shape/values: {key}.{name}")
        row = stats["norm_stats"][key]
        if any(low > high for low, high in zip(row["q01"], row["q99"])):
            raise ValueError("Reversed normalization bounds")


def freeze_normalization(source, output, expected_count, allow_existing=False):
    source = Path(source).resolve()
    raw = source.read_bytes()
    stats = json.loads(raw)
    validate_clean_stats(stats)
    if stats.get("count") != expected_count or expected_count is None:
        raise ValueError("Actual loaded normalization count disagrees with the clean dataset")
    if stats.get("verification", {}).get("count") != expected_count:
        raise ValueError("Use independently verified clean statistics, not a legacy unchecked file")
    if stats["verification"].get("horizon") != 50 or stats["verification"].get("episodes") != 2500:
        raise ValueError("Unexpected clean verification horizon/episode count")
    fingerprint = semantic_hash(stats)
    directory = Path(output).resolve() / "normalization"
    snapshot, manifest_path = directory / "norm_stats.json", directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not allow_existing or manifest["semantic_sha256"] != fingerprint:
            raise ValueError("Existing run cannot be reused with different normalization or a fresh launch")
        if not snapshot.exists() or semantic_hash(json.loads(snapshot.read_bytes())) != fingerprint:
            raise ValueError("Saved normalization snapshot missing or corrupted")
        return {**manifest, "path": str(snapshot)}
    output_path = Path(output).resolve()
    if allow_existing and (output_path / "checkpoints").exists():
        raise ValueError("Cannot resume an old run without an audited normalization contract")
    # train.sh exclusively creates the fresh run and its log before torchrun.
    launch_directory = os.environ.get("LINGBOT_TRAIN_RUN_DIR")
    launch_log = os.environ.get("TRAIN_LOG_FILE")
    launcher_owned = (
        launch_directory and Path(launch_directory).resolve() == output_path
        and launch_log and os.environ.get("LINGBOT_TRAIN_RUN_ID")
        and output_path.is_dir()
        and all(path == Path(launch_log).resolve() and path.is_file() for path in output_path.iterdir())
    )
    if (output_path.exists() and any(output_path.iterdir()) and not launcher_owned
            and not (output_path / "stage2_run.json").exists()):
        raise ValueError("Fresh clean training requires a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(raw)
    manifest = {
        "file": "norm_stats.json", "source": str(source), "count": expected_count,
        "semantic_sha256": fingerprint, "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_fingerprint": stats["verification"]["source_fingerprint"],
        "normalization_type": "bounds_99_woclip", "action_horizon": stats["verification"]["horizon"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "path": str(snapshot)}


def resolve_inference_normalization(hf_checkpoint, training_config, override=None):
    """New runs require their own snapshot; legacy runs retain explicit paths."""
    run = Path(hf_checkpoint).resolve().parent.parent.parent
    manifest_path = run / "normalization/manifest.json"
    required = training_config.get("data", {}).get("require_normalization_contract", False)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["file"] != "norm_stats.json":
            raise ValueError("Unexpected normalization snapshot path")
        snapshot = manifest_path.parent / manifest["file"]
        value = json.loads(snapshot.read_bytes())
        if semantic_hash(value) != manifest["semantic_sha256"]:
            raise ValueError("Inference normalization snapshot checksum mismatch")
        if override and semantic_hash(json.loads(Path(override).read_bytes())) != manifest["semantic_sha256"]:
            raise ValueError("Inference override disagrees with this model's training statistics")
        return str(snapshot)
    if required:
        raise ValueError("Audited training run has no normalization snapshot; refusing silent fallback")
    value = override or training_config["data"]["norm_stats_file"]
    if not value or not Path(value).exists():
        raise ValueError(f"Legacy normalization file missing: {value}; provide explicit robot_norm_path")
    return str(Path(value).resolve())
