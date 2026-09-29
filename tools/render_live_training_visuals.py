#!/usr/bin/env python3
"""Render checkpoint-safe training diagnostics from the live JSONL metric log.

This script reads ``<analysis-run>/analysis/data/training_metrics_live.jsonl``
and only writes below ``<analysis-run>/analysis``.  A separate checkpoint run
directory is inspected by name but is never modified.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import analyze_training_log as analysis


FIGURE_DESCRIPTIONS = [
    ("01_loss_overview.png", "总损失与 VLA 损失"),
    ("02_auxiliary_losses.png", "辅助损失"),
    ("03_weighted_loss_contributions.png", "加权损失贡献"),
    ("04_learning_rates.png", "学习率"),
    ("05_optimization_health.png", "优化健康"),
    ("06_moe_health.png", "MoE 路由健康"),
    ("07_loss_distribution_by_phase.png", "分阶段损失分布"),
    ("08_metric_correlation.png", "指标相关性"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-run-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--rolling-window", type=int, default=200)
    return parser.parse_args()


def load_live_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    duplicate_steps: Counter[int] = Counter()
    invalid_lines: list[int] = []
    restart_events: list[dict[str, int]] = []
    last_seen_step: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                step = int(raw["step"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                invalid_lines.append(line_number)
                continue
            row: dict[str, Any] = {
                "timestamp": str(raw.get("timestamp", "")),
                "step": step,
                "train_steps_per_epoch": int(raw.get("train_steps_per_epoch", 0) or 0),
                "epoch": int(raw.get("epoch", 0) or 0),
            }
            for field in analysis.NUMERIC_FIELDS:
                try:
                    row[field] = float(raw.get(field, math.nan))
                except (TypeError, ValueError):
                    row[field] = math.nan
            if step in by_step:
                duplicate_steps[step] += 1
            if last_seen_step is not None and step <= last_seen_step:
                # A resumed/restarted process may append from an earlier checkpoint.
                # Discard the now-superseded tail while retaining the earlier history.
                superseded = [old_step for old_step in by_step if old_step >= step]
                for old_step in superseded:
                    del by_step[old_step]
                restart_events.append(
                    {
                        "line": line_number,
                        "restart_step": step,
                        "previous_last_step": last_seen_step,
                        "superseded_rows": len(superseded),
                    }
                )
            by_step[step] = row
            last_seen_step = step

    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise RuntimeError(f"No valid metric rows found in {path}")
    expected = set(range(rows[0]["step"], rows[-1]["step"] + 1))
    return rows, {
        "duplicate_steps": dict(duplicate_steps),
        "missing_steps": sorted(expected.difference(by_step)),
        "invalid_jsonl_line_count": len(invalid_lines),
        "invalid_jsonl_line_examples": invalid_lines[:20],
        "restart_events": restart_events,
    }


def render_figure_set(
    rows: list[dict[str, Any]],
    output_dir: Path,
    rolling_window: int,
    checkpoint_steps: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis.plot_loss_overview(rows, output_dir / "01_loss_overview.png", rolling_window, checkpoint_steps)
    analysis.plot_aux_losses(rows, output_dir / "02_auxiliary_losses.png", rolling_window, checkpoint_steps)
    analysis.plot_weighted_contributions(
        rows, output_dir / "03_weighted_loss_contributions.png", rolling_window, checkpoint_steps
    )
    analysis.plot_learning_rates(rows, output_dir / "04_learning_rates.png", checkpoint_steps)
    analysis.plot_optimization_health(
        rows, output_dir / "05_optimization_health.png", rolling_window, checkpoint_steps
    )
    analysis.plot_moe_health(rows, output_dir / "06_moe_health.png", rolling_window, checkpoint_steps)
    analysis.plot_phase_distribution(rows, output_dir / "07_loss_distribution_by_phase.png")
    analysis.plot_correlation(rows, output_dir / "08_metric_correlation.png")


def safe_timestamp(value: str) -> datetime | None:
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def save_tabular_outputs(
    rows: list[dict[str, Any]],
    data_dir: Path,
    rolling_window: int,
) -> None:
    all_fields = analysis.CSV_FIELDS + [
        "weighted_depth",
        "weighted_future_depth",
        "weighted_future_video",
        "reconstructed_loss",
        "loss_residual",
    ]
    analysis.save_csv(data_dir / "training_metrics.csv", rows, all_fields)

    rolling_fields = analysis.NUMERIC_FIELDS + [
        "weighted_depth",
        "weighted_future_depth",
        "weighted_future_video",
        "reconstructed_loss",
        "loss_residual",
    ]
    rolling_cache = {
        field: analysis.rolling_mean(analysis.values(rows, field), rolling_window)
        for field in rolling_fields
    }
    rolling_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = {"step": row["step"], "timestamp": row["timestamp"]}
        for field in rolling_fields:
            item[field] = float(rolling_cache[field][index])
        rolling_rows.append(item)
    analysis.save_csv(
        data_dir / f"training_metrics_rolling_{rolling_window}.csv",
        rolling_rows,
        list(rolling_rows[0]),
    )

    windows = [
        analysis.window_stats(rows, start, min(start + 999, rows[-1]["step"]))
        for start in range(rows[0]["step"], rows[-1]["step"] + 1, 1000)
    ]
    analysis.save_csv(data_dir / "window_summary.csv", windows, list(windows[0]))
    outliers = analysis.robust_outliers(rows)
    analysis.save_csv(
        data_dir / "outliers.csv",
        outliers,
        list(outliers[0]) if outliers else ["metric", "step", "timestamp", "value", "median", "threshold"],
    )


def build_summary(
    rows: list[dict[str, Any]],
    parse_meta: dict[str, Any],
    metrics_path: Path,
    config_path: Path,
    config: dict[str, Any],
    checkpoint_step: int,
    actual_checkpoints: list[int],
) -> dict[str, Any]:
    first_ts = safe_timestamp(rows[0]["timestamp"])
    last_ts = safe_timestamp(rows[-1]["timestamp"])
    first_window = rows[: min(500, len(rows))]
    last_window = rows[-min(500, len(rows)) :]
    return {
        "source_metrics": {
            "name": metrics_path.name,
            "bytes": metrics_path.stat().st_size,
            "sha256": analysis.sha256(metrics_path),
        },
        "source_config": {
            "name": config_path.name,
            "sha256": analysis.sha256(config_path),
        },
        "generated_for_checkpoint_step": checkpoint_step,
        "row_count": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "wall_clock_hours_between_first_and_last_metric": (
            (last_ts - first_ts).total_seconds() / 3600 if first_ts and last_ts else None
        ),
        "train_steps_per_epoch_logged": sorted({row["train_steps_per_epoch"] for row in rows}),
        "max_steps_configured": analysis.get_nested(config, "train", "max_steps"),
        "save_steps": int(analysis.get_nested(config, "train", "save_steps", default=0) or 0),
        "checkpoint_directories_observed": actual_checkpoints,
        "metric_statistics": {
            field: analysis.finite_stats(analysis.values(rows, field))
            for field in analysis.NUMERIC_FIELDS + ["reconstructed_loss", "loss_residual"]
        },
        "windows": {
            "first_500": {
                field: analysis.finite_stats(analysis.values(first_window, field))["mean"]
                for field in analysis.LOSS_FIELDS + ["grad_norm", "step_time_s"]
            },
            "last_500": {
                field: analysis.finite_stats(analysis.values(last_window, field))["mean"]
                for field in analysis.LOSS_FIELDS + ["grad_norm", "step_time_s"]
            },
        },
        "ignored_batch_total": int(analysis.np.nansum(analysis.values(rows, "ignore_batch_num"))),
        "parse_integrity": parse_meta,
        "notes": [
            "Metrics are written asynchronously by rank 0 and flushed before checkpoint visualization.",
            "All generated files are outside checkpoints/; model checkpoint contents are never modified.",
            "A resumed run can contain duplicate steps in JSONL; the last record for each step is used.",
        ],
    }


def write_live_readme(path: Path, summary: dict[str, Any], checkpoint_step: int) -> None:
    first = summary["windows"]["first_500"]["loss"]
    last = summary["windows"]["last_500"]["loss"]
    change = (last / first - 1) * 100 if first else math.nan
    lines = [
        "# 自动训练可视化",
        "",
        f"- 最新 checkpoint：`global_step_{checkpoint_step}`",
        f"- 指标覆盖：step {summary['first_step']}–{summary['last_step']}（{summary['row_count']:,} 行）",
        f"- 前 500 步平均损失：{first:.6f}",
        f"- 最近 500 步平均损失：{last:.6f}（相对变化 {change:+.1f}%）",
        "- 总览图位于 `figures/`；逐 checkpoint 快照位于 `by_checkpoint/`。",
        "- 原始结构化指标保存在 `data/training_metrics_live.jsonl`，可用于后续离线重画。",
        "- 本目录与 `checkpoints/` 完全分离，不会改动模型权重。",
        "",
        "每次 checkpoint 保存成功后，训练主进程会自动刷新这里的内容。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_gallery(
    path: Path,
    summary: dict[str, Any],
    checkpoint_step: int,
    image_prefix: str,
    checkpoint_links: list[tuple[int, str]] | None = None,
) -> None:
    first = summary["windows"]["first_500"]["loss"]
    last = summary["windows"]["last_500"]["loss"]
    change = (last / first - 1) * 100 if first else math.nan
    cards = [
        ("Checkpoint", f"{checkpoint_step:,}"),
        ("Metric rows", f"{summary['row_count']:,}"),
        ("First 500 loss", f"{first:.6f}"),
        ("Last 500 loss", f"{last:.6f}"),
        ("Loss change", f"{change:+.1f}%"),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    figures_html = "".join(
        (
            '<figure><a href="{src}"><img src="{src}" alt="{label}"></a>'
            '<figcaption>{label}</figcaption></figure>'
        ).format(src=html.escape(f"{image_prefix}{filename}"), label=html.escape(label))
        for filename, label in FIGURE_DESCRIPTIONS
    )
    checkpoints_html = ""
    if checkpoint_links:
        links = "".join(
            f'<li><a href="{html.escape(href)}">global_step_{step}</a></li>'
            for step, href in checkpoint_links
        )
        checkpoints_html = f"<section><h2>Checkpoint snapshots</h2><ul>{links}</ul></section>"
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RobotWin training visualization — step {checkpoint_step}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; color: #172033; background: #f4f7fb; }}
body {{ margin: 0 auto; max-width: 1500px; padding: 32px; }}
h1 {{ margin-bottom: 8px; }}
.subtitle {{ color: #607089; margin-top: 0; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap: 14px; margin: 24px 0; }}
.card, figure, section {{ background: white; border: 1px solid #dce3ed; border-radius: 12px; box-shadow: 0 2px 8px #14213d0d; }}
.card {{ padding: 16px; }} .card span {{ display:block; color:#607089; font-size:13px; }} .card strong {{ font-size:24px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:18px; }}
figure {{ margin:0; padding:12px; }} img {{ width:100%; height:auto; display:block; }} figcaption {{ padding:10px 4px 2px; font-weight:600; }}
section {{ margin-top:24px; padding:16px 22px; }} a {{ color:#2457c5; }}
@media (max-width:600px) {{ body {{ padding:16px; }} .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>RobotWin 自动训练可视化</h1>
<p class="subtitle">指标覆盖 step {summary['first_step']}–{summary['last_step']}；模型 checkpoint 目录保持只读。</p>
<div class="cards">{card_html}</div>
<main class="grid">{figures_html}</main>
{checkpoints_html}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_run_dir = args.analysis_run_dir.resolve()
    checkpoint_run_dir = args.checkpoint_run_dir.resolve()
    analysis_dir = analysis_run_dir / "analysis"
    data_dir = analysis_dir / "data"
    figures_dir = analysis_dir / "figures"
    configs_dir = analysis_dir / "configs"
    snapshot_dir = analysis_dir / "by_checkpoint" / f"global_step_{args.checkpoint_step}"
    metrics_path = (args.metrics_path or (data_dir / "training_metrics_live.jsonl")).resolve()
    config_path = (args.config_path or (analysis_run_dir / "lingbotvla_cli.yaml")).resolve()
    if not metrics_path.exists():
        raise FileNotFoundError(f"Live metrics file not found: {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Saved effective config not found: {config_path}")

    for directory in (data_dir, figures_dir, configs_dir, snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows, parse_meta = load_live_rows(metrics_path)
    config = analysis.load_yaml(config_path)
    analysis.add_derived_metrics(rows, config)
    actual_checkpoints, _ = analysis.checkpoint_steps(checkpoint_run_dir, 0, rows[-1]["step"])
    chart_checkpoints = [step for step in actual_checkpoints if step <= rows[-1]["step"]]
    if args.checkpoint_step <= rows[-1]["step"] and args.checkpoint_step not in chart_checkpoints:
        chart_checkpoints.append(args.checkpoint_step)
    chart_checkpoints.sort()

    analysis.setup_plot_style()
    render_figure_set(rows, figures_dir, args.rolling_window, chart_checkpoints)
    snapshot_rows = [row for row in rows if row["step"] <= args.checkpoint_step]
    if not snapshot_rows:
        raise RuntimeError(f"No metrics at or before checkpoint step {args.checkpoint_step}")
    render_figure_set(snapshot_rows, snapshot_dir, args.rolling_window, [args.checkpoint_step])
    save_tabular_outputs(rows, data_dir, args.rolling_window)

    summary = build_summary(
        rows,
        parse_meta,
        metrics_path,
        config_path,
        config,
        args.checkpoint_step,
        actual_checkpoints,
    )
    analysis.write_json(data_dir / "summary.json", summary)
    checkpoint_summary = {
        "status": "checkpoint_observed" if args.checkpoint_step in actual_checkpoints else "checkpoint_requested",
        "requested_checkpoint_step": args.checkpoint_step,
        "last_logged_step_in_snapshot": snapshot_rows[-1]["step"],
        "metric_rows": len(snapshot_rows),
        "loss": analysis.finite_stats(analysis.values(snapshot_rows[-min(500, len(snapshot_rows)) :], "loss")),
        "vla_loss": analysis.finite_stats(
            analysis.values(snapshot_rows[-min(500, len(snapshot_rows)) :], "vla_loss")
        ),
        "grad_norm": analysis.finite_stats(
            analysis.values(snapshot_rows[-min(500, len(snapshot_rows)) :], "grad_norm")
        ),
    }
    analysis.write_json(snapshot_dir / "summary.json", checkpoint_summary)

    index_rows = []
    for directory in sorted((analysis_dir / "by_checkpoint").glob("global_step_*")):
        if not directory.is_dir():
            continue
        prefix = "global_step_"
        if not directory.name.startswith(prefix):
            continue
        try:
            step = int(directory.name[len(prefix) :])
        except ValueError:
            continue
        summary_path = directory / "summary.json"
        status = "snapshot_complete" if summary_path.exists() else "snapshot_incomplete"
        index_rows.append({"name": directory.name, "step": step, "status": status})
    index_rows.sort(key=lambda item: item["step"])
    analysis.save_csv(analysis_dir / "by_checkpoint" / "index.csv", index_rows, ["name", "step", "status"])

    shutil.copy2(config_path, configs_dir / "effective_training_config.yaml")
    write_live_readme(analysis_dir / "README.md", summary, args.checkpoint_step)
    write_html_gallery(snapshot_dir / "report.html", summary, args.checkpoint_step, "")
    checkpoint_links = [
        (item["step"], f"by_checkpoint/{item['name']}/report.html")
        for item in index_rows
        if item["status"] == "snapshot_complete"
    ]
    write_html_gallery(
        analysis_dir / "report.html",
        summary,
        args.checkpoint_step,
        "figures/",
        checkpoint_links=checkpoint_links,
    )
    print(
        json.dumps(
            {
                "analysis_run_dir": str(analysis_run_dir),
                "checkpoint_run_dir": str(checkpoint_run_dir),
                "checkpoint_step": args.checkpoint_step,
                "metric_rows": len(rows),
                "snapshot": str(snapshot_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
