#!/usr/bin/env python3
"""Build an offline evidence report from RoboTwin evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.eval_logging import read_events, read_json, summarize_run


ARM_INDICES = list(range(6)) + list(range(7, 13))
JUMP_RAD = 0.5
TARGET_CHANGE_RAD = 0.05
SMALL_MOTION_RAD = 0.001


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def legacy_tasks(run: Path) -> list[dict]:
    rows = []
    for root in sorted((run / "eval_results").glob("*")):
        if not root.is_dir():
            continue
        text = run / "eval_logs" / f"{root.name}.log"
        clean = (
            re.sub(r"\x1b\[[0-9;]*m", "", text.read_text(encoding="utf-8", errors="replace")) if text.exists() else ""
        )
        matches = re.findall(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([\d.]+)%", clean)
        row = {
            "task": root.name,
            "status": "legacy: trace unavailable",
            "selected_attempt": None,
            "successes": None,
            "episodes": None,
            "errors": None,
            "attempts": [],
        }
        if matches:
            row.update(successes=int(matches[-1][0]), episodes=int(matches[-1][1]))
        result = root / "_result.txt"
        if result.exists():
            lines = result.read_text(encoding="utf-8").strip().splitlines()
            try:
                row["legacy_rate"] = float(lines[-1])
            except (ValueError, IndexError):
                pass
        rows.append(row)
    return rows


def episode_anomalies(run: Path, info: dict, inference: list, execution: list, warnings: list) -> tuple[list, list]:
    rows, predictions = [], []
    open_requests = {}
    identity = {key: info.get(key) for key in ("task", "attempt", "episode_id", "seed")}

    def add(kind, request=None, step=None, evidence=None):
        rows.append({**identity, "kind": kind, "request_id": request, "step": step, "evidence": evidence})

    for item in inference:
        if item.get("event") == "request_start":
            open_requests[item.get("request_id")] = item
        if item.get("event") != "request_end":
            continue
        open_requests.pop(item.get("request_id"), None)
        diagnostic = item.get("diagnostic") or {}
        relative = diagnostic.get("artifact")
        if not relative:
            continue  # Reset has metadata rather than an action artifact.
        try:
            path = (run / relative).resolve()
            path.relative_to(run)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            predictions.append((item, arrays))
            for key in ("normalized_actions", "full_action", "returned_action"):
                if key in arrays and not np.isfinite(arrays[key]).all():
                    add(
                        "nonfinite_prediction",
                        item.get("request_id"),
                        evidence={"array": key, "indices": np.argwhere(~np.isfinite(arrays[key])).tolist()},
                    )
            if "normalized_actions" in arrays and "action_joint_mask" in arrays:
                normalized = arrays["normalized_actions"][0]
                mask = arrays["action_joint_mask"][0].astype(bool)
                # Only bounds-normalized dimensions have a [-1, 1] interpretation.
                metadata = read_json(run / info.get("model_metadata", "__missing__"), {}, warnings)
                offset = 0
                types = metadata.get("normalization_types", {})
                features = metadata.get("features", {})
                for joint in features.get("joints", []):
                    width = features.get("joints_max_dim", {}).get(joint, 0)
                    valid = mask[offset : offset + width]
                    values = normalized[:, offset : offset + width][:, valid]
                    if types.get(f"action.{joint}", "").startswith("bounds") and np.any(np.abs(values) > 1):
                        add(
                            "normalization_bounds_excursion",
                            item.get("request_id"),
                            evidence={
                                "feature": joint,
                                "max_abs": float(np.nanmax(np.abs(values))),
                                "note": "Outside training quantiles; this alone does not establish a failure cause",
                            },
                        )
                    offset += width
        except (OSError, ValueError, EOFError, KeyError, IndexError, zipfile.BadZipFile) as exc:
            message = f"{relative}: {exc}"
            warnings.append(message)
            add("prediction_unavailable", item.get("request_id"), evidence=message)

    for request_id, item in open_requests.items():
        add("unclosed_request", request_id, item.get("executed_actions"), "Request has no response event")
    open_actions = {}
    previous_action = None
    limits = np.asarray(info.get("robot", {}).get("arm_joint_limits", []), dtype=float)
    for item in execution:
        key = (item.get("request_id"), item.get("chunk_index"))
        if item.get("event") == "action_start":
            open_actions[key] = item
            continue
        if item.get("event") not in ("action_end", "action_error"):
            if item.get("event") in ("cleanup_error", "diagnostic_unavailable"):
                add(item["event"], item.get("request_id"), evidence=item)
            continue
        open_actions.pop(key, None)
        request, step = item.get("request_id"), item.get("take_action_cnt")
        for plan in item.get("planning", []):
            if plan.get("status") != "ok":
                add("planning_failure", request, step, plan)
        if item.get("event") == "action_error":
            add("execution_exception", request, step, item)
            continue
        action = np.asarray(item.get("action"), dtype=float)
        if action.shape != (14,):
            add("action_mapping_unavailable", request, step, action.shape)
            continue
        if limits.shape == (12, 2):
            outside = (action[ARM_INDICES] < limits[:, 0]) | (action[ARM_INDICES] > limits[:, 1])
            if outside.any():
                add("joint_limit_excursion", request, step, {"arm_joint_indices": np.flatnonzero(outside).tolist()})
        if previous_action is not None:
            delta = np.abs(action[ARM_INDICES] - previous_action[ARM_INDICES])
            if np.any(delta > JUMP_RAD):
                add("action_jump", request, step, {"max_rad": float(delta.max()), "threshold_rad": JUMP_RAD})
        previous_action = action
        before, after = item.get("before") or {}, item.get("after") or {}
        target_before = np.asarray(before.get("control_target", []), dtype=float)
        actual_before = np.asarray(before.get("actual_arm_qpos", []), dtype=float)
        actual_after = np.asarray(after.get("actual_arm_qpos", []), dtype=float)
        if target_before.shape == (14,) and actual_before.shape == actual_after.shape == (12,):
            change = np.abs(action[ARM_INDICES] - target_before[ARM_INDICES])
            motion = np.abs(actual_after - actual_before)
            indices = np.flatnonzero((change > TARGET_CHANGE_RAD) & (motion < SMALL_MOTION_RAD))
            if len(indices):
                add(
                    "target_changed_small_actual_motion",
                    request,
                    step,
                    {
                        "arm_joint_indices": indices.tolist(),
                        "command_change_rad": change[indices].tolist(),
                        "actual_motion_rad": motion[indices].tolist(),
                        "thresholds_rad": [TARGET_CHANGE_RAD, SMALL_MOTION_RAD],
                    },
                )
    for item in open_actions.values():
        add(
            "unclosed_action",
            item.get("request_id"),
            item.get("take_action_cnt"),
            "Start event has no end/error event",
        )
    if info.get("status") == "running":
        add("unclosed_episode", evidence="No episode terminal record; consult scheduler exit/signal events")
    if info.get("reason") not in ("success", "step_limit", None):
        add(
            "episode_error",
            info.get("request_id"),
            evidence={k: info.get(k) for k in ("reason", "stage", "message", "traceback")},
        )
    if info.get("eval_trace") == "off":
        add("trace_disabled", evidence="Detailed inference and execution diagnostics were disabled")
    return rows, predictions


def plot_episode(path: Path, info: dict, execution: list, predictions: list) -> bool:
    completed = [item for item in execution if item.get("event") == "action_end" and item.get("count_delta", 0) > 0]
    if not completed:
        return False
    steps = np.asarray([item["take_action_cnt"] for item in completed])
    commands = np.asarray([item["action"] for item in completed], dtype=float)
    target = np.asarray([item["after"]["control_target"] for item in completed], dtype=float)
    actual = np.asarray([item["after"]["actual_arm_qpos"] for item in completed], dtype=float)
    if commands.shape[1:] != (14,) or target.shape[1:] != (14,) or actual.shape[1:] != (12,):
        return False
    starts = {
        item["request_id"]: item["take_action_cnt"]
        for item in execution
        if item.get("event") == "action_start" and item.get("chunk_index") == 0
    }
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), constrained_layout=True, sharex=True)
    colors = plt.get_cmap("tab10").colors
    for axis, command_indices, actual_indices, title in (
        (axes[0], range(6), range(6), "Left arm / radians"),
        (axes[1], range(7, 13), range(6, 12), "Right arm / radians"),
    ):
        for color_index, (command_index, actual_index) in enumerate(zip(command_indices, actual_indices)):
            color = colors[color_index]
            axis.plot(steps, commands[:, command_index], color=color, label=f"J{color_index + 1} command")
            axis.plot(steps, target[:, command_index], color=color, linestyle=":", alpha=0.7)
            axis.plot(steps, actual[:, actual_index], color=color, linestyle="--", alpha=0.8)
        axis.set_title(title + " — solid: sent; dotted: drive target; dashed: physical position", loc="left")
        axis.legend(ncol=6, fontsize=8)
    axes[2].plot(steps, commands[:, 6], label="Left sent", color=colors[0])
    axes[2].plot(steps, commands[:, 13], label="Right sent", color=colors[1])
    axes[2].plot(steps, target[:, 6], ":", label="Left command feedback", color=colors[0])
    axes[2].plot(steps, target[:, 13], ":", label="Right command feedback", color=colors[1])
    axes[2].set_title("Gripper command / normalized — physical gripper position unavailable", loc="left")
    axes[2].legend(ncol=4, fontsize=8)
    error = np.abs(target[:, ARM_INDICES] - actual)
    axes[3].plot(steps, error.mean(axis=1), label="Mean |target − actual|", color="#12685d")
    axes[3].plot(steps, error.max(axis=1), label="Max |target − actual|", color="#be5835")
    axes[3].set_title("Arm tracking error / radians", loc="left")
    axes[3].legend(fontsize=8)
    for item, arrays in predictions:
        request = item.get("request_id")
        if request not in starts or "full_action" not in arrays:
            continue
        full = arrays["full_action"][0]
        if full.ndim != 2 or full.shape[1] != 14:
            continue
        xs = starts[request] + np.arange(1, len(full) + 1)
        for axis, indices in ((axes[0], range(6)), (axes[1], range(7, 13)), (axes[2], (6, 13))):
            for color_index, index in enumerate(indices):
                axis.plot(xs, full[:, index], color=colors[color_index], linewidth=0.65, alpha=0.22)
    for axis in axes:
        for boundary in sorted(set(starts.values())):
            axis.axvline(boundary + 1, color="#586270", linewidth=0.5, alpha=0.3)
        axis.grid(alpha=0.15)
        axis.set_xlabel("Simulator action count; faint lines show full predicted horizons")
    fig.suptitle(
        f"{info['task']} · attempt {info['attempt']} · episode {info['episode_id']} · seed {info['seed']}", fontsize=15
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def artifact_link(path: Path, output: Path, label: str) -> str:
    import os

    relative = os.path.relpath(path, output).replace("\\", "/")
    return f'<a href="{quote(relative, safe="/")}">{html.escape(label)}</a>'


def generate_report(run: Path, output: Path, task: str | None = None, episode: int | None = None) -> dict:
    run, output = run.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    summary = summarize_run(run, write=False)
    warnings = list(summary["warnings"])
    manifest = read_json(run / "run_manifest.json", {}, warnings)
    tasks = summary["tasks"] if manifest else legacy_tasks(run)
    if task:
        tasks = [item for item in tasks if item["task"] == task]
    episodes, anomalies, sections, timings = [], [], [], []
    first_tasks = set()
    directories = list((run / "eval_results").glob("*/episodes/episode_*"))
    directories.extend((run / "eval_results").glob("*/attempts/attempt_*/episodes/episode_*"))
    for directory in sorted(directories):
        info = read_json(directory / "episode.json", {}, warnings)
        if (
            not info
            or (task and info.get("task") != task)
            or (episode is not None and info.get("episode_id") != episode)
        ):
            continue
        inference = read_events(directory / "inference.jsonl", warnings)
        execution = read_events(directory / "execution.jsonl", warnings)
        found, predictions = episode_anomalies(run, info, inference, execution, warnings)
        anomalies.extend(found)
        episodes.append({**info, "anomalies": len(found), "log_directory": str(directory)})
        for item in inference:
            if item.get("event") == "request_end" and item.get("request_id") != "reset":
                timings.append({"rtt_ms": item.get("rtt_ms"), **(item.get("server_timing") or {})})
        figure_name = f"{info['task']}_attempt{info['attempt']}_episode{info['episode_id']}.png"
        render = bool(found) or info["task"] not in first_tasks or episode is not None
        figure = output / "figures" / figure_name
        has_plot = render and plot_episode(figure, info, execution, predictions)
        first_tasks.add(info["task"])
        links = [
            artifact_link(directory / "episode.json", output, "Episode"),
            artifact_link(directory / "inference.jsonl", output, "Inference"),
            artifact_link(directory / "execution.jsonl", output, "Execution"),
        ]
        if info.get("video"):
            links.append(artifact_link(run / info["video"], output, "Video"))
        if info.get("model_metadata"):
            links.append(artifact_link(run / info["model_metadata"], output, "Model/config"))
        plots = (
            f'<img loading="lazy" src="figures/{quote(figure_name)}" alt="Action and physical joint traces">'
            if has_plot
            else '<p class="muted">No curve generated; detailed data may be unavailable. Select this episode to request curves.</p>'
        )
        evidence = html.escape(json.dumps(found, ensure_ascii=False, indent=2))
        sections.append(f'''<details class="episode" data-task="{html.escape(str(info["task"]), quote=True)}"
            data-result="{html.escape(str(info.get("reason") or info.get("status")), quote=True)}"
            data-seed="{info["seed"]}" data-anomaly="{bool(found)}">
            <summary><strong>{html.escape(info["task"])}</strong> / attempt {info["attempt"]} / episode {info["episode_id"]}
            <span>seed {info["seed"]} · {html.escape(str(info.get("reason") or info.get("status")))} · {len(found)} observations</span></summary>
            <nav>{" · ".join(links)}</nav>{plots}<details><summary>Diagnostic evidence</summary><pre>{evidence}</pre></details></details>''')
    write_csv(
        output / "tasks.csv",
        tasks,
        ["task", "status", "selected_attempt", "successes", "episodes", "errors", "attempts", "legacy_rate"],
    )
    write_csv(
        output / "episodes.csv",
        episodes,
        [
            "task",
            "attempt",
            "episode_id",
            "seed",
            "reason",
            "status",
            "executed_actions",
            "duration_ms",
            "logging_ms",
            "anomalies",
            "video",
            "log_directory",
        ],
    )
    write_csv(
        output / "anomalies.csv",
        anomalies,
        ["task", "attempt", "episode_id", "seed", "kind", "request_id", "step", "evidence"],
    )
    rows = "".join(
        f"<tr><td>{html.escape(row['task'])}</td><td>{html.escape(row['status'])}</td><td>{row.get('selected_attempt')}</td><td>{row.get('successes')} / {row.get('episodes')}</td></tr>"
        for row in tasks
    )
    rates = [item for item in timings if item.get("infer_ms") is not None]
    latency = {
        key: {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}
        for key in ("infer_ms", "sample_ms", "preprocess_ms", "unapply_ms", "rtt_ms", "logging_ms")
        if (values := [float(item[key]) for item in rates if item.get(key) is not None])
    }
    diagnostics = {
        "counts": dict(Counter(row["kind"] for row in anomalies)),
        "timings": latency,
        "episode_metrics": [
            {key: row.get(key) for key in ("task", "attempt", "episode_id", "duration_ms", "logging_ms", "metrics")}
            for row in episodes
        ],
        "artifact_bytes": sum(path.stat().st_size for path in run.rglob("*") if path.is_file()),
        "thresholds_rad": {
            "action_jump": JUMP_RAD,
            "command_change": TARGET_CHANGE_RAD,
            "small_motion": SMALL_MOTION_RAD,
        },
        "warnings": warnings,
    }
    from deploy.eval_logging import atomic_json, record

    atomic_json(output / "diagnostics.json", record(run_id=run.name, **diagnostics))
    rate = f"{summary['success_rate']:.1%}" if summary["success_rate"] is not None and manifest else "Unavailable"
    task_options = "".join(f"<option>{html.escape(name)}</option>" for name in sorted({row["task"] for row in tasks}))
    content = TEMPLATE.replace("@@RUN@@", html.escape(run.name)).replace("@@RATE@@", rate)
    content = content.replace(
        "@@COVERAGE@@", f"{summary['completed_tasks']} / {summary['requested_tasks']}" if manifest else "Legacy"
    )
    content = content.replace("@@EPISODES@@", str(len(episodes))).replace("@@OBSERVATIONS@@", str(len(anomalies)))
    content = (
        content.replace("@@TASKS@@", rows)
        .replace("@@OPTIONS@@", task_options)
        .replace("@@SECTIONS@@", "".join(sections))
    )
    content = content.replace("@@MANIFEST@@", html.escape(json.dumps(manifest, ensure_ascii=False, indent=2)))
    content = content.replace("@@DIAGNOSTICS@@", html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2)))
    content = content.replace("@@WARNINGS@@", html.escape("\n".join(warnings) or "No parse warnings."))
    (output / "report.html").write_text(content, encoding="utf-8")
    return diagnostics


TEMPLATE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@RUN@@ · Evaluation evidence</title><style>
:root{--paper:#f1efe8;--ink:#172d31;--muted:#627174;--line:#c9ceca;--accent:#14685d;--rust:#a8492e}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 'Segoe UI',sans-serif}
main{max-width:1440px;margin:auto;padding:42px 5vw 80px}header{border-top:6px solid var(--ink);padding-top:25px}
.eyebrow{font:12px Consolas,monospace;letter-spacing:.18em;color:var(--accent)}h1{font:clamp(38px,5vw,64px)/1.1 Georgia,serif;margin:18px 0}
h2{font:28px Georgia,serif;margin:42px 0 14px}.run{font:13px Consolas,monospace;overflow-wrap:anywhere;color:var(--muted)}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);margin:32px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.metric{padding:22px 16px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric b{display:block;font:38px Georgia,serif}.metric span{font-size:12px;text-transform:uppercase;letter-spacing:.08em}
table{border-collapse:collapse;width:100%;text-align:left}th{font:12px Consolas,monospace;color:var(--muted);text-transform:uppercase}td,th{padding:12px;border-bottom:1px solid var(--line)}
.note{border-left:3px solid var(--rust);padding:10px 18px;background:#e9e5db;color:#4f5654}.filters{display:flex;gap:12px;flex-wrap:wrap;padding:16px 0;position:sticky;top:0;background:var(--paper);z-index:1;border-bottom:1px solid var(--line)}
input,select,button{font:inherit;padding:8px 10px;color:var(--ink);background:#faf9f4;border:1px solid var(--line);border-radius:0}input{min-width:170px}input[type=checkbox]{min-width:auto}button{cursor:pointer}
.episode{border-bottom:1px solid var(--line);padding:14px 0}.episode>summary{display:flex;gap:20px;justify-content:space-between;cursor:pointer;flex-wrap:wrap}.episode>summary span{font:13px Consolas,monospace;color:var(--muted)}
details[open]>summary{margin-bottom:14px}summary{cursor:pointer}nav{margin:12px 0}a{color:var(--accent);text-underline-offset:4px}a:focus-visible,summary:focus-visible,input:focus-visible{outline:2px solid var(--rust)}
img{width:100%;background:white;border:1px solid var(--line)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 Consolas,monospace;background:#e6e7df;padding:18px;max-height:520px;overflow:auto}.muted{color:var(--muted)}
footer{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);font:12px Consolas,monospace}.hidden{display:none!important}
@media(max-width:720px){main{padding:24px 18px}.metrics{grid-template-columns:1fr 1fr}.metric{border-bottom:1px solid var(--line)}table{font-size:12px}.episode>summary{display:block}}
</style><main><header><div class="eyebrow">ROBOTWIN / EVALUATION OBSERVATORY</div><h1>Follow the action.<br>Inspect the evidence.</h1><div class="run">@@RUN@@</div></header>
<section class="metrics"><div class="metric"><b>@@RATE@@</b><span>Official success rate</span></div><div class="metric"><b>@@COVERAGE@@</b><span>Complete task coverage</span></div><div class="metric"><b>@@EPISODES@@</b><span>Indexed episodes</span></div><div class="metric"><b>@@OBSERVATIONS@@</b><span>Diagnostic observations</span></div></section>
<p class="note">Official rates use complete attempts only. Partial attempts and legacy results remain visible below. Observations are evidence, not established failure causes. Gripper feedback represents commands; physical finger positions are unavailable.</p>
<h2>Tasks & attempts</h2><table><thead><tr><th>Task</th><th>Status</th><th>Selected attempt</th><th>Success / episodes</th></tr></thead><tbody>@@TASKS@@</tbody></table>
<h2>Episode evidence</h2><div class="filters"><select id="task" aria-label="Task"><option value="">All tasks</option>@@OPTIONS@@</select><select id="result" aria-label="Result"><option value="">All results</option><option>success</option><option>step_limit</option><option>invalid_action</option><option>exception</option><option>interrupted</option><option>running</option></select><input id="seed" placeholder="Filter by seed" aria-label="Seed"><label><input id="anomaly" type="checkbox"> With observations</label><button id="expand">Expand visible</button></div>
@@SECTIONS@@<p class="muted">Faint curves show complete predicted horizons, including unexecuted suffixes. Vertical lines mark inference boundaries. Select an episode with --episode to generate its curves.</p>
<h2>Timing & diagnostic thresholds</h2><p class="muted">infer_ms: policy execution; sample_ms: model sampling including CPU return; rtt_ms: client round trip; logging_ms: prediction write. Historical prev_total_ms includes waiting for client execution and is not model latency. Compare full/off runs using matching tasks and precision; no cross-run overhead estimate is inferred.</p><pre>@@DIAGNOSTICS@@</pre>
<h2>Run configuration</h2><details><summary>Resolved launch configuration and environment</summary><pre>@@MANIFEST@@</pre></details><details><summary>Recovery / missing-data warnings</summary><pre>@@WARNINGS@@</pre></details><footer>Offline report · schema v1 · <a href="tasks.csv">Tasks CSV</a> · <a href="episodes.csv">Episodes CSV</a> · <a href="anomalies.csv">Observations CSV</a></footer>
</main><script>
const episodes=[...document.querySelectorAll('.episode')];const controls=['task','result','seed','anomaly'].map(id=>document.getElementById(id));
function filter(){const[t,r,s,a]=controls;for(const e of episodes)e.classList.toggle('hidden',(t.value&&e.dataset.task!==t.value)||(r.value&&e.dataset.result!==r.value)||(s.value&&!e.dataset.seed.includes(s.value))||(a.checked&&e.dataset.anomaly!=='True'));}
for(const c of controls)c.addEventListener('input',filter);document.getElementById('expand').addEventListener('click',()=>{for(const e of episodes)if(!e.classList.contains('hidden'))e.open=true});
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--episode", type=int)
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error(f"Evaluation run directory does not exist: {args.run}")
    generate_report(args.run, args.output or args.run / "analysis", args.task, args.episode)
    print(f"Report: {(args.output or args.run / 'analysis') / 'report.html'}")


if __name__ == "__main__":
    main()
