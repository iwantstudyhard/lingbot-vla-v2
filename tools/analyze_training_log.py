#!/usr/bin/env python3
"""Extract LingBot-VLA training metrics and build checkpoint-safe diagnostics.

The script is intentionally read-only with respect to ``checkpoints/``.  All
generated files live below ``<run>/analysis`` (or ``--output``).  If checkpoint
directories are present, their step numbers are used only to choose snapshot
cut-offs; no file is ever written into a model checkpoint directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import yaml


METRIC_MAP = {
    "Loss": "loss",
    "VLA_Loss": "vla_loss",
    "Depth_Loss": "depth_loss",
    "Future_Depth_Loss": "future_depth_loss",
    "FutureVideo_Loss": "future_video_loss",
    "SeqWise_Loss": "seq_wise_loss",
    "RouterZ_Loss": "router_z_loss",
    "MaxVio": "maxvio",
    "AvgSigmoid": "avg_sigmoid",
    "GradNorm": "grad_norm",
    "LR": "lr",
    "Expert_LR": "expert_lr",
    "StepTime": "step_time_s",
    "Depth_Forward_Time": "depth_forward_time_s",
    "Ignore_Batch_Num": "ignore_batch_num",
}

NUMERIC_FIELDS = list(METRIC_MAP.values())
CSV_FIELDS = ["timestamp", "step", "train_steps_per_epoch", "epoch", *NUMERIC_FIELDS]
LOSS_FIELDS = [
    "loss",
    "vla_loss",
    "depth_loss",
    "future_depth_loss",
    "future_video_loss",
    "seq_wise_loss",
    "router_z_loss",
]

BLUE = "#2563eb"
BLUE_DARK = "#1e3a8a"
BLUE_LIGHT = "#93c5fd"
GOLD = "#ca8a04"
ORANGE = "#ea580c"
OLIVE = "#718135"
PINK = "#be5678"
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#d1d5db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--alternate-config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rolling-window", type=int, default=200)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_logged_config(text: str) -> dict[str, Any] | None:
    marker = " - INFO - __main__ - {"
    pos = text.find(marker)
    if pos < 0:
        return None
    start = text.find("{", pos)
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_metric_rows(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    header = re.compile(
        r"(?P<timestamp>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})"
        r" - INFO - __main__ - Step (?P<step>\d+)/(?P<denom>\d+), "
        r"Epoch (?P<epoch>\d+), (?P<body>[^\r\n]+)"
    )
    value_re = re.compile(r"^(?P<name>[A-Za-z_]+)\s+(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)s?$")
    by_step: dict[int, dict[str, Any]] = {}
    duplicates = Counter()
    parse_failures: list[str] = []
    for match in header.finditer(text):
        step = int(match.group("step"))
        row: dict[str, Any] = {
            "timestamp": match.group("timestamp"),
            "step": step,
            "train_steps_per_epoch": int(match.group("denom")),
            "epoch": int(match.group("epoch")),
        }
        for field in NUMERIC_FIELDS:
            row[field] = math.nan
        for token in match.group("body").split(","):
            token = token.strip()
            vm = value_re.match(token)
            if not vm:
                parse_failures.append(token)
                continue
            source_name = vm.group("name")
            if source_name not in METRIC_MAP:
                continue
            target = METRIC_MAP[source_name]
            row[target] = float(vm.group("value"))
        if step in by_step:
            duplicates[step] += 1
        by_step[step] = row

    rows = [by_step[key] for key in sorted(by_step)]
    if not rows:
        raise RuntimeError("No training metric rows were parsed from the log.")
    expected = set(range(rows[0]["step"], rows[-1]["step"] + 1))
    gaps = sorted(expected.difference(by_step))
    metadata = {
        "duplicate_steps": dict(duplicates),
        "missing_steps": gaps,
        "parse_failure_examples": parse_failures[:20],
        "parse_failure_count": len(parse_failures),
    }
    return rows, metadata


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = yaml.safe_load(read_text(path))
    return value if isinstance(value, dict) else {}


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(child, path))
    elif isinstance(value, list):
        result[prefix] = value
    else:
        result[prefix] = value
    return result


def config_diff(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    lf, rf = flatten(left), flatten(right)
    rows = []
    for key in sorted(set(lf) | set(rf)):
        lv, rv = lf.get(key, "<missing>"), rf.get(key, "<missing>")
        if lv != rv:
            rows.append({"parameter": key, "left": lv, "right": rv})
    return rows


def values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row.get(field, math.nan)) for row in rows], dtype=float)


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(arr)
    vals = np.where(valid, arr, 0.0)
    csum = np.cumsum(vals)
    count = np.cumsum(valid.astype(int))
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        j = max(0, i - window + 1)
        total = csum[i] - (csum[j - 1] if j else 0.0)
        n = count[i] - (count[j - 1] if j else 0)
        out[i] = total / n if n else math.nan
    return out


def finite_stats(arr: np.ndarray) -> dict[str, float | int | None]:
    clean = arr[np.isfinite(arr)]
    if clean.size == 0:
        return {key: None for key in ["count", "mean", "std", "min", "p05", "median", "p95", "p99", "max"]}
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "p05": float(np.quantile(clean, 0.05)),
        "median": float(np.median(clean)),
        "p95": float(np.quantile(clean, 0.95)),
        "p99": float(np.quantile(clean, 0.99)),
        "max": float(np.max(clean)),
    }


def window_stats(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    subset = [row for row in rows if start <= row["step"] <= end]
    result: dict[str, Any] = {"start_step": start, "end_step": end, "count": len(subset)}
    for field in LOSS_FIELDS + ["grad_norm", "step_time_s", "depth_forward_time_s", "maxvio", "avg_sigmoid", "lr", "expert_lr"]:
        stat = finite_stats(values(subset, field))
        result[f"{field}_mean"] = stat["mean"]
        result[f"{field}_p95"] = stat["p95"]
    return result


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def add_derived_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    depth_weight = float(get_nested(config, "train", "align_params", "depth_loss_weight", default=1.0))
    future_depth_weight = float(get_nested(config, "train", "align_params", "future_depth_loss_weight", default=1.0))
    video_weight = float(
        get_nested(
            config,
            "train",
            "align_params",
            "video",
            "future_video_loss_weight",
            default=get_nested(config, "train", "align_params", "depth_loss_weight", default=1.0),
        )
    )
    for row in rows:
        row["weighted_depth"] = row["depth_loss"] * depth_weight
        row["weighted_future_depth"] = row["future_depth_loss"] * future_depth_weight
        row["weighted_future_video"] = row["future_video_loss"] * video_weight
        terms = [
            row["vla_loss"],
            row["weighted_depth"],
            row["weighted_future_depth"],
            row["weighted_future_video"],
            row["seq_wise_loss"],
            row["router_z_loss"],
        ]
        row["reconstructed_loss"] = sum(terms) if all(math.isfinite(float(x)) for x in terms) else math.nan
        row["loss_residual"] = row["loss"] - row["reconstructed_loss"]


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def annotate_checkpoints(ax: plt.Axes, checkpoint_steps: Iterable[int]) -> None:
    first = True
    for step in checkpoint_steps:
        ax.axvline(step, color=INK, linewidth=1.0, linestyle="--", alpha=0.45, label="checkpoint" if first else None)
        first = False


def plot_loss_overview(rows: list[dict[str, Any]], path: Path, window: int, checkpoint_steps: list[int]) -> None:
    steps = values(rows, "step")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for field, label, color in [("loss", "Total loss", BLUE), ("vla_loss", "VLA loss", ORANGE)]:
        arr = values(rows, field)
        ax.plot(steps, arr, color=color, alpha=0.09, linewidth=0.55)
        ax.plot(steps, rolling_mean(arr, window), color=color, linewidth=2.1, label=f"{label} ({window}-step mean)")
    annotate_checkpoints(ax, checkpoint_steps)
    ax.set_title("Training and VLA Loss")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", ncol=2)
    fig.text(
        0.01,
        0.01,
        "Raw values are shown faintly; solid lines are rolling means.",
        color=MUTED,
        fontsize=8,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_aux_losses(rows: list[dict[str, Any]], path: Path, window: int, checkpoint_steps: list[int]) -> None:
    steps = values(rows, "step")
    specs = [
        ("depth_loss", "Depth loss (unweighted)", BLUE),
        ("future_depth_loss", "Future depth loss (unweighted)", GOLD),
        ("future_video_loss", "Future video loss (unweighted)", ORANGE),
        ("seq_wise_loss", "Sequence-wise balance loss (weighted)", OLIVE),
        ("router_z_loss", "Router z-loss (weighted)", PINK),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for ax, (field, title, color) in zip(axes, specs):
        arr = values(rows, field)
        ax.plot(steps, arr, color=color, alpha=0.11, linewidth=0.5)
        ax.plot(steps, rolling_mean(arr, window), color=color, linewidth=2)
        annotate_checkpoints(ax, checkpoint_steps)
        ax.set_title(title)
        ax.grid(True, axis="y")
    ax = axes[-1]
    aux = values(rows, "loss") - values(rows, "vla_loss")
    ax.plot(steps, aux, color=BLUE_DARK, alpha=0.12, linewidth=0.5)
    ax.plot(steps, rolling_mean(aux, window), color=BLUE_DARK, linewidth=2)
    annotate_checkpoints(ax, checkpoint_steps)
    ax.set_title("Total auxiliary contribution (Total - VLA)")
    ax.grid(True, axis="y")
    for ax in axes[-2:]:
        ax.set_xlabel("Optimizer step")
    fig.suptitle("Auxiliary Training Losses", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.01, "Depth/video panels show de-weighted logged losses; sequence/router panels are already coefficient-weighted.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_weighted_contributions(rows: list[dict[str, Any]], path: Path, window: int, checkpoint_steps: list[int]) -> None:
    steps = values(rows, "step")
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    total = rolling_mean(values(rows, "loss"), window)
    reconstructed = rolling_mean(values(rows, "reconstructed_loss"), window)
    vla = rolling_mean(values(rows, "vla_loss"), window)
    axes[0].plot(steps, total, color=INK, linewidth=2.2, label="Logged total")
    axes[0].plot(steps, reconstructed, color=BLUE, linewidth=1.6, linestyle="--", label="Reconstructed total")
    axes[0].plot(steps, vla, color=ORANGE, linewidth=1.4, label="VLA component")
    annotate_checkpoints(axes[0], checkpoint_steps)
    axes[0].set_title("Logged vs. Reconstructed Weighted Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, axis="y")
    axes[0].legend(loc="upper right", ncol=3)

    fields = [
        ("weighted_depth", "Depth × 0.004", BLUE),
        ("weighted_future_depth", "Future depth × 0.004", GOLD),
        ("weighted_future_video", "Future video × 0.004", ORANGE),
        ("seq_wise_loss", "Sequence-wise", OLIVE),
        ("router_z_loss", "Router z", PINK),
    ]
    for field, label, color in fields:
        axes[1].plot(steps, rolling_mean(values(rows, field), window), label=label, color=color, linewidth=1.8)
    annotate_checkpoints(axes[1], checkpoint_steps)
    axes[1].set_title("Weighted Auxiliary Contributions")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Contribution to total loss")
    axes[1].grid(True, axis="y")
    axes[1].legend(loc="upper right", ncol=3)
    fig.text(0.01, 0.01, "All curves are rolling means. The reconstruction residual mostly reflects 4-decimal log rounding.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_learning_rates(rows: list[dict[str, Any]], path: Path, checkpoint_steps: list[int]) -> None:
    steps = values(rows, "step")
    base = values(rows, "lr")
    expert = values(rows, "expert_lr")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(steps, base, color=BLUE, linewidth=2, label="Base LR")
    axes[0].plot(steps, expert, color=ORANGE, linewidth=2, label="Expert LR")
    annotate_checkpoints(axes[0], checkpoint_steps)
    axes[0].set_title("Learning-rate Schedule")
    axes[0].set_ylabel("Learning rate")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[0].grid(True, axis="y")
    axes[0].legend()
    ratio = np.divide(expert, base, out=np.full_like(expert, np.nan), where=np.isfinite(base) & (base != 0))
    axes[1].plot(steps, ratio, color=INK, linewidth=1.5)
    annotate_checkpoints(axes[1], checkpoint_steps)
    axes[1].set_title("Expert/base LR ratio")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlabel("Optimizer step")
    axes[1].grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_optimization_health(rows: list[dict[str, Any]], path: Path, window: int, checkpoint_steps: list[int]) -> None:
    steps = values(rows, "step")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    specs = [
        ("grad_norm", "Gradient norm", BLUE, "Norm"),
        ("step_time_s", "Step time", ORANGE, "Seconds"),
        ("depth_forward_time_s", "Depth/teacher forward time", OLIVE, "Seconds"),
    ]
    for ax, (field, title, color, ylabel) in zip(axes, specs):
        arr = values(rows, field)
        ax.plot(steps, arr, color=color, linewidth=0.5, alpha=0.14)
        ax.plot(steps, rolling_mean(arr, window), color=color, linewidth=2)
        annotate_checkpoints(ax, checkpoint_steps)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
    axes[-1].set_xlabel("Optimizer step")
    fig.text(0.01, 0.01, "Raw values are faint; solid lines are rolling means. Ignore_Batch_Num is checked separately in the summary.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_moe_health(rows: list[dict[str, Any]], path: Path, window: int, checkpoint_steps: list[int]) -> None:
    steps = values(rows, "step")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    specs = [
        ("maxvio", "MoE MaxVio", BLUE),
        ("avg_sigmoid", "Average top-k sigmoid score", ORANGE),
        ("seq_wise_loss", "Sequence-wise balance loss", OLIVE),
        ("router_z_loss", "Router z-loss", PINK),
    ]
    for ax, (field, title, color) in zip(axes.ravel(), specs):
        arr = values(rows, field)
        ax.plot(steps, arr, color=color, linewidth=0.5, alpha=0.13)
        ax.plot(steps, rolling_mean(arr, window), color=color, linewidth=2)
        annotate_checkpoints(ax, checkpoint_steps)
        ax.set_title(title)
        ax.grid(True, axis="y")
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer step")
    fig.suptitle("MoE Routing Health", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phase_distribution(rows: list[dict[str, Any]], path: Path) -> None:
    steps = values(rows, "step")
    max_step = int(np.nanmax(steps))
    edges = np.linspace(int(np.nanmin(steps)), max_step + 1, 6, dtype=int)
    labels, series = [], []
    loss = values(rows, "loss")
    for i in range(5):
        mask = (steps >= edges[i]) & (steps < edges[i + 1]) & np.isfinite(loss)
        series.append(loss[mask])
        labels.append(f"{edges[i]}–{edges[i + 1]-1}")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bp = ax.boxplot(series, tick_labels=labels, showfliers=False, patch_artist=True, medianprops={"color": INK, "linewidth": 1.5})
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE_LIGHT)
        patch.set_edgecolor(BLUE_DARK)
    ax.set_title("Total-loss Distribution by Training Phase")
    ax.set_xlabel("Step interval")
    ax.set_ylabel("Total loss")
    ax.grid(True, axis="y")
    fig.text(0.01, 0.01, "Boxes show the middle 50%; whiskers use the standard 1.5×IQR rule. Outliers are omitted from the drawing, not from analysis.", color=MUTED, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_correlation(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["loss", "vla_loss", "depth_loss", "future_depth_loss", "future_video_loss", "seq_wise_loss", "maxvio", "avg_sigmoid", "grad_norm", "step_time_s"]
    labels = ["Total", "VLA", "Depth", "Future depth", "Video", "Seq-wise", "MaxVio", "Avg sigmoid", "Grad norm", "Step time"]
    matrix = np.vstack([values(rows, field) for field in fields]).T
    keep = np.all(np.isfinite(matrix), axis=1)
    corr = np.corrcoef(matrix[keep], rowvar=False)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 0.55 else INK)
    ax.set_title("Metric Correlation (Per-step Values)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")


def robust_outliers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for field in ["loss", "grad_norm", "step_time_s", "depth_forward_time_s", "maxvio"]:
        arr = values(rows, field)
        clean = arr[np.isfinite(arr)]
        if clean.size < 10:
            continue
        median = float(np.median(clean))
        mad = float(np.median(np.abs(clean - median)))
        threshold = median + 6 * 1.4826 * mad if mad > 0 else float(np.quantile(clean, 0.999))
        indices = np.where(np.isfinite(arr) & (arr > threshold))[0]
        for idx in indices:
            output.append(
                {
                    "metric": field,
                    "step": rows[int(idx)]["step"],
                    "timestamp": rows[int(idx)]["timestamp"],
                    "value": float(arr[int(idx)]),
                    "median": median,
                    "threshold": threshold,
                }
            )
    output.sort(key=lambda item: (item["metric"], -item["value"]))
    return output


def extract_log_context(text: str) -> dict[str, Any]:
    command = next((line[2:] for line in text.splitlines() if line.startswith("+ torchrun ")), None)
    progress_totals = sorted({int(x) for x in re.findall(r"Step:\s*\d+/(\d+)", text)})
    warnings = {
        "pydantic_unsupported_field_attribute": text.count("UnsupportedFieldAttributeWarning"),
        "flash_attention_dtype_warning": text.count("attempting to use Flash Attention 2 without specifying a torch dtype"),
        "traceback_count": text.count("Traceback (most recent call last)"),
        "cuda_oom_count": len(re.findall(r"CUDA out of memory|OutOfMemoryError", text, flags=re.IGNORECASE)),
        "nan_token_count": len(re.findall(r"(?<![A-Za-z])nan(?![A-Za-z])", text, flags=re.IGNORECASE)),
    }
    param_match = re.search(
        r"VLM params:\s+([\d.]+)M.*?Action expert params \(total\):\s+([\d.]+)M.*?"
        r"Action expert params \(active\):\s+([\d.]+)M.*?Total params:\s+([\d.]+)M.*?Activated total params:\s+([\d.]+)M",
        text,
        flags=re.DOTALL,
    )
    params = None
    if param_match:
        params = {
            "vlm_million": float(param_match.group(1)),
            "action_expert_total_million": float(param_match.group(2)),
            "action_expert_active_million": float(param_match.group(3)),
            "total_million": float(param_match.group(4)),
            "activated_total_million": float(param_match.group(5)),
        }
    return {"command": command, "progress_totals": progress_totals, "warning_counts": warnings, "parameter_counts": params}


def checkpoint_steps(run_dir: Path, save_steps: int, max_logged_step: int) -> tuple[list[int], list[int]]:
    actual: list[int] = []
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        for child in checkpoint_dir.iterdir():
            match = re.fullmatch(r"global_step_(\d+)", child.name)
            if child.is_dir() and match:
                actual.append(int(match.group(1)))
    expected = list(range(save_steps, max_logged_step + 1, save_steps)) if save_steps > 0 else []
    return sorted(set(actual)), expected


def create_snapshot(rows: list[dict[str, Any]], target_step: int, out_dir: Path, window: int, status: str) -> None:
    subset = [row for row in rows if row["step"] <= target_step]
    if not subset:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    final_step = subset[-1]["step"]
    checkpoint_marker = [target_step] if target_step <= final_step else []
    plot_loss_overview(subset, out_dir / "01_loss_overview.png", window, checkpoint_marker)
    plot_aux_losses(subset, out_dir / "02_auxiliary_losses.png", window, checkpoint_marker)
    plot_weighted_contributions(subset, out_dir / "03_weighted_loss_contributions.png", window, checkpoint_marker)
    plot_learning_rates(subset, out_dir / "04_learning_rates.png", checkpoint_marker)
    plot_optimization_health(subset, out_dir / "05_optimization_health.png", window, checkpoint_marker)
    plot_moe_health(subset, out_dir / "06_moe_health.png", window, checkpoint_marker)
    plot_phase_distribution(subset, out_dir / "07_loss_distribution_by_phase.png")
    plot_correlation(subset, out_dir / "08_metric_correlation.png")
    summary = {
        "status": status,
        "requested_checkpoint_step": target_step,
        "last_logged_step_in_snapshot": final_step,
        "metric_rows": len(subset),
        "loss": finite_stats(values(subset[-min(500, len(subset)):], "loss")),
        "vla_loss": finite_stats(values(subset[-min(500, len(subset)):], "vla_loss")),
        "grad_norm": finite_stats(values(subset[-min(500, len(subset)):], "grad_norm")),
    }
    write_json(out_dir / "summary.json", summary)


def render_config_markdown(
    official_diff: list[dict[str, Any]],
    alternate_diff: list[dict[str, Any]],
    config_path: Path,
    official_path: Path | None,
    alternate_path: Path | None,
) -> str:
    def table(rows: list[dict[str, Any]], left_label: str, right_label: str) -> str:
        lines = [f"| 参数 | {left_label} | {right_label} |", "|---|---|---|"]
        for row in rows:
            left = str(row["left"]).replace("|", "\\|")
            right = str(row["right"]).replace("|", "\\|")
            lines.append(f"| `{row['parameter']}` | `{left}` | `{right}` |")
        return "\n".join(lines)

    sections = [
        "# 训练配置对照",
        "",
        f"正式配置：`{config_path.as_posix()}`",
        "",
        "## 官方配置 → 正式训练配置",
        "",
        table(official_diff, official_path.name if official_path else "official", config_path.name) if official_diff else "未提供官方配置或没有语义差异。",
        "",
        "## 试运行配置 → 正式训练配置",
        "",
        table(alternate_diff, alternate_path.name if alternate_path else "trial", config_path.name) if alternate_diff else "未提供试运行配置或没有语义差异。",
        "",
        "说明：列表和字典按解析后的 YAML 值比较，换行风格、注释和键顺序不会造成伪差异。",
    ]
    return "\n".join(sections) + "\n"


def write_readme(
    path: Path,
    summary: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
    rolling_window: int,
) -> None:
    first = summary["windows"]["first_500"]["loss"]
    last = summary["windows"]["last_500"]["loss"]
    change = (last / first - 1) * 100 if first else math.nan
    lines = [
        "# RobotWin BF16 4-GPU 训练分析",
        "",
        "## 结论摘要",
        "",
        f"- 日志完整解析出 **{summary['row_count']:,}** 个逐步记录，覆盖 step **{summary['first_step']}–{summary['last_step']}**。",
        f"- 前 500 步平均总损失为 **{first:.4f}**，最后 500 步为 **{last:.4f}**，相对变化 **{change:+.1f}%**。",
        f"- 日志最后更新时间为 **{summary['last_timestamp']}**；末尾没有正常训练结束或 checkpoint 保存记录，因此这是截断日志，不代表训练在该步失败。",
        f"- 配置 `save_steps={summary['save_steps']}`；当前日志未覆盖 step {summary['save_steps']}，故不能从这份日志恢复该检查点区间的指标。",
        "- 模型 checkpoint 目录保持只读；所有图和表仅保存在本 `analysis/` 目录。",
        "",
        "## 汇总报告",
        "",
        "- `report.html`：可离线打开的交互式技术报告，包含关键指标、趋势图、窗口统计和配置差异。",
        "- `artifact.json`：报告的可复现数据与版式定义。",
        "",
        "## 图表索引",
        "",
        f"- `figures/01_loss_overview.png`：总损失与 VLA 损失，含 {rolling_window} 步移动平均。",
        "- `figures/02_auxiliary_losses.png`：深度、未来深度、未来视频、MoE 辅助损失。",
        "- `figures/03_weighted_loss_contributions.png`：按训练权重还原的损失贡献与总损失核对。",
        "- `figures/04_learning_rates.png`：基础参数与 MoE 专家参数学习率。",
        "- `figures/05_optimization_health.png`：梯度范数、单步耗时、教师前向耗时。",
        "- `figures/06_moe_health.png`：MaxVio、路由 sigmoid、平衡损失与 z-loss。",
        "- `figures/07_loss_distribution_by_phase.png`：五个训练阶段的损失分布。",
        "- `figures/08_metric_correlation.png`：逐步指标相关性（仅描述相关，不代表因果）。",
        "",
        "## 数据与配置",
        "",
        "- `data/training_metrics.csv`：逐步原始指标与还原后的加权项。",
        f"- `data/training_metrics_rolling_{rolling_window}.csv`：移动平均指标。",
        "- `data/window_summary.csv`：每 1000 步窗口统计。",
        "- `data/outliers.csv`：基于 median + 6×MAD 的稳健异常候选。",
        "- `data/summary.json`：机器可读汇总、完整性检查和告警计数。",
        "- `configs/effective_logged_config.json`：日志启动时打印的最终有效参数。",
        "- `configs/config_comparison.md`：官方、试运行、正式配置的语义差异。",
        "",
        "## Checkpoint 可视化快照",
        "",
        "快照仅写入 `analysis/by_checkpoint/`，绝不写入 `checkpoints/`。",
        "",
        "| 名称 | 状态 | 日志覆盖到 |",
        "|---|---|---:|",
    ]
    for row in checkpoint_rows:
        lines.append(f"| `{row['name']}` | {row['status']} | {row['last_logged_step']} |")
    lines += [
        "",
        "## 重跑命令",
        "",
        "```powershell",
        "python tools/analyze_training_log.py `",
        "  --log bf16_4gpu_formal.log `",
        "  --config configs/vla/robotwin/robotwin_clean_freeze_vision_bf16_formal.yaml `",
        "  --official-config configs/vla/robotwin/robotwin.yaml `",
        "  --alternate-config configs/vla/robotwin/robotwin_clean_freeze_vision_bf16.yaml `",
        "  --run-dir train_outputs/robotwin_clean_freeze_vision_bf16",
        "```",
        "",
        "把更完整的日志放回同一路径后重跑，脚本会自动识别 `checkpoints/global_step_*`，并在 `analysis/by_checkpoint/` 创建对应的只读分析快照。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    log_path = args.log.resolve()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    output = (args.output or (run_dir / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    fig_dir = output / "figures"
    config_dir = output / "configs"
    by_checkpoint_dir = output / "by_checkpoint"
    for directory in [data_dir, fig_dir, config_dir, by_checkpoint_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    text = read_text(log_path)
    rows, parse_meta = parse_metric_rows(text)
    config = load_yaml(config_path)
    official = load_yaml(args.official_config.resolve() if args.official_config else None)
    alternate = load_yaml(args.alternate_config.resolve() if args.alternate_config else None)
    logged_config = extract_logged_config(text)
    effective = logged_config or config
    add_derived_metrics(rows, effective)

    save_steps = int(get_nested(effective, "train", "save_steps", default=0) or 0)
    actual_checkpoints, expected_checkpoints = checkpoint_steps(run_dir, save_steps, rows[-1]["step"])
    covered_actual = [step for step in actual_checkpoints if step <= rows[-1]["step"]]
    chart_checkpoints = sorted(set(expected_checkpoints + covered_actual))

    setup_plot_style()
    plot_loss_overview(rows, fig_dir / "01_loss_overview.png", args.rolling_window, chart_checkpoints)
    plot_aux_losses(rows, fig_dir / "02_auxiliary_losses.png", args.rolling_window, chart_checkpoints)
    plot_weighted_contributions(rows, fig_dir / "03_weighted_loss_contributions.png", args.rolling_window, chart_checkpoints)
    plot_learning_rates(rows, fig_dir / "04_learning_rates.png", chart_checkpoints)
    plot_optimization_health(rows, fig_dir / "05_optimization_health.png", args.rolling_window, chart_checkpoints)
    plot_moe_health(rows, fig_dir / "06_moe_health.png", args.rolling_window, chart_checkpoints)
    plot_phase_distribution(rows, fig_dir / "07_loss_distribution_by_phase.png")
    plot_correlation(rows, fig_dir / "08_metric_correlation.png")

    all_fields = CSV_FIELDS + ["weighted_depth", "weighted_future_depth", "weighted_future_video", "reconstructed_loss", "loss_residual"]
    save_csv(data_dir / "training_metrics.csv", rows, all_fields)
    rolling_fields = NUMERIC_FIELDS + [
        "weighted_depth",
        "weighted_future_depth",
        "weighted_future_video",
        "reconstructed_loss",
        "loss_residual",
    ]
    rolling_cache = {
        field: rolling_mean(values(rows, field), args.rolling_window)
        for field in rolling_fields
    }
    rolling_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = {"step": row["step"], "timestamp": row["timestamp"]}
        for field in rolling_fields:
            item[field] = float(rolling_cache[field][idx])
        rolling_rows.append(item)
    save_csv(data_dir / f"training_metrics_rolling_{args.rolling_window}.csv", rolling_rows, list(rolling_rows[0].keys()))

    windows = []
    for start in range(rows[0]["step"], rows[-1]["step"] + 1, 1000):
        windows.append(window_stats(rows, start, min(start + 999, rows[-1]["step"])))
    save_csv(data_dir / "window_summary.csv", windows, list(windows[0].keys()))
    outliers = robust_outliers(rows)
    if outliers:
        save_csv(data_dir / "outliers.csv", outliers, list(outliers[0].keys()))
    else:
        save_csv(data_dir / "outliers.csv", [], ["metric", "step", "timestamp", "value", "median", "threshold"])

    first_ts = datetime.strptime(rows[0]["timestamp"], "%m/%d/%Y %H:%M:%S")
    last_ts = datetime.strptime(rows[-1]["timestamp"], "%m/%d/%Y %H:%M:%S")
    first_500 = rows[: min(500, len(rows))]
    last_500 = rows[-min(500, len(rows)) :]
    log_context = extract_log_context(text)
    summary = {
        "source_log": {"name": log_path.name, "bytes": log_path.stat().st_size, "sha256": sha256(log_path)},
        "source_config": {"name": config_path.name, "sha256": sha256(config_path)},
        "row_count": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "wall_clock_hours_between_first_and_last_metric": (last_ts - first_ts).total_seconds() / 3600,
        "train_steps_per_epoch_logged": sorted({row["train_steps_per_epoch"] for row in rows}),
        "max_steps_configured": get_nested(effective, "train", "max_steps"),
        "save_steps": save_steps,
        "expected_checkpoint_steps_covered_by_log": expected_checkpoints,
        "checkpoint_directories_observed": actual_checkpoints,
        "metric_statistics": {field: finite_stats(values(rows, field)) for field in NUMERIC_FIELDS + ["reconstructed_loss", "loss_residual"]},
        "windows": {
            "first_500": {field: finite_stats(values(first_500, field))["mean"] for field in LOSS_FIELDS + ["grad_norm", "step_time_s"]},
            "last_500": {field: finite_stats(values(last_500, field))["mean"] for field in LOSS_FIELDS + ["grad_norm", "step_time_s"]},
        },
        "ignored_batch_total": int(np.nansum(values(rows, "ignore_batch_num"))),
        "parse_integrity": parse_meta,
        "log_context": log_context,
        "log_ended_cleanly": "Reached max_steps=" in text or "Distributed checkpoint saved" in text[-20000:],
        "notes": [
            "Depth, future-depth and future-video values in the text log are de-weighted after distributed averaging.",
            "Sequence-wise and router z-loss values are already coefficient-weighted in model code.",
            "Total loss is reconstructed as VLA + 0.004*depth + 0.004*future_depth + 0.004*future_video + seq_wise + router_z.",
            f"The supplied log ends at step {rows[-1]['step']} without a traceback or normal completion/checkpoint footer.",
        ],
    }
    write_json(data_dir / "summary.json", summary)

    official_diff = config_diff(official, config) if official else []
    alternate_diff = config_diff(alternate, config) if alternate else []
    write_json(data_dir / "config_diff_official_to_formal.json", official_diff)
    write_json(data_dir / "config_diff_trial_to_formal.json", alternate_diff)
    (config_dir / "config_comparison.md").write_text(
        render_config_markdown(official_diff, alternate_diff, config_path, args.official_config, args.alternate_config),
        encoding="utf-8",
    )
    if logged_config:
        write_json(config_dir / "effective_logged_config.json", logged_config)
    shutil.copy2(config_path, config_dir / "formal_training_config.yaml")
    if args.official_config and args.official_config.exists():
        shutil.copy2(args.official_config, config_dir / "official_robotwin_config.yaml")
    if args.alternate_config and args.alternate_config.exists():
        shutil.copy2(args.alternate_config, config_dir / "trial_training_config.yaml")

    checkpoint_index: list[dict[str, Any]] = []
    for step in sorted(set(actual_checkpoints + expected_checkpoints)):
        if step <= rows[-1]["step"]:
            status = "checkpoint_observed" if step in actual_checkpoints else "checkpoint_interval_covered"
            name = f"global_step_{step}"
            create_snapshot(rows, step, by_checkpoint_dir / name, args.rolling_window, status)
            checkpoint_index.append({"name": name, "status": status, "last_logged_step": step})
        else:
            name = f"global_step_{step}"
            status = "checkpoint_exists_but_log_missing"
            create_snapshot(rows, step, by_checkpoint_dir / name, args.rolling_window, status)
            checkpoint_index.append({"name": name, "status": status, "last_logged_step": rows[-1]["step"]})

    if rows[-1]["step"] not in expected_checkpoints:
        name = f"latest_partial_step_{rows[-1]['step']}"
        create_snapshot(rows, rows[-1]["step"], by_checkpoint_dir / name, args.rolling_window, "partial_log")
        checkpoint_index.append({"name": name, "status": "partial_log", "last_logged_step": rows[-1]["step"]})
    save_csv(by_checkpoint_dir / "index.csv", checkpoint_index, ["name", "status", "last_logged_step"])

    write_readme(output / "README.md", summary, checkpoint_index, args.rolling_window)
    print(json.dumps({"output": str(output), "rows": len(rows), "last_step": rows[-1]["step"], "checkpoints": checkpoint_index}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
