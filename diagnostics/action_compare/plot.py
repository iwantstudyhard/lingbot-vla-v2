"""Plot measured action diagnostics; never infer actions from video pixels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import ARM, COLORS, JOINTS, boundary_summary, differences, json_write


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = root / "analysis"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    runs = {}
    for label in ("ours", "official"):
        folder = root / label
        if not (folder / "completion.json").is_file():
            raise ValueError(f"Incomplete evaluation: {folder}; inspect its log")
        runs[label] = {"records": read_lines(folder / "executed.jsonl"),
                       "requests": read_lines(folder / "requests.jsonl"),
                       "completion": json.loads((folder / "completion.json").read_text()),
                       "servers": json.loads((folder / "servers.json").read_text())}
        if len(runs[label]["records"]) < 2 or not runs[label]["requests"]:
            raise ValueError(f"Insufficient measured actions in {folder}")
    first_hashes = [runs[label]["requests"][0]["observation_hash"] for label in runs]
    same_initial = len(set(first_hashes)) == 1
    warnings = []
    if not same_initial:
        warnings.append("Initial closed-loop observations differ. Paired predictions within "
                        "each request still use identical inputs; do not treat rollouts as aligned.")
    for key in ("horizon", "denoising_steps", "image_size"):
        a, b = [runs["ours"]["servers"][label].get(key) for label in ("ours", "official")]
        if a != b:
            warnings.append(f"Model inference configuration differs: {key}: ours={a}, official={b}")

    rows, metrics = [], {}
    for primary, run in runs.items():
        records = run["records"]
        executed = np.asarray([item["action"] for item in records])
        actual = np.asarray([item["actual_after"] for item in records])
        chunks = np.asarray([item["chunk"] for item in records])
        run.update(executed=executed, actual=actual, chunks=chunks)
        metrics[primary] = {
            "completion": run["completion"],
            "executed_command_increments": differences(executed),
            "chunk_boundary_increments": boundary_summary(executed, chunks),
            "mean_post_action_tracking_error_rad": float(np.abs(executed[:, ARM] - actual[:, ARM]).mean()),
            "max_post_action_tracking_error_rad": float(np.abs(executed[:, ARM] - actual[:, ARM]).max()),
            "topp_failed_or_zero_length": sum(
                1 for item in records for entry in item["topp"].values()
                if entry["error"] is not None or entry["points"] == 0),
        }
        for request in run["requests"]:
            path = root / primary / "predictions" / f"{request['call']:04d}.npz"
            with np.load(path, allow_pickle=False) as data:
                names = data["normalized_joint_names"].tolist()
                arm_idx = [i for i, name in enumerate(names) if name.startswith("action.arm.position[")]
                if len(arm_idx) != 12:
                    raise ValueError("Expected 12 valid normalized arm joints; padding excluded")
                for model in ("ours", "official"):
                    normalized = data[f"{model}_normalized"][:, arm_idx]
                    row = {"rollout_primary": primary, "model": model, "call": request["call"],
                           "scene_seed": request["scene_seed"], "sampling_seed": request["sampling_seed"],
                           "observation_hash": request["observation_hash"],
                           "full_horizon": len(data[model]), "executed_prefix": request["use_length"],
                           **differences(data[model]),
                           "normalized_mean_abs_delta": float(np.abs(np.diff(normalized, axis=0)).mean()),
                           "normalized_abs_max": float(np.abs(normalized).max())}
                    rows.append(row)
    with (output / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_write(output / "summary.json", {"same_initial_observation": same_initial,
               "warnings": warnings, "metrics": metrics,
               "notes": ["Arm increments use 12 joints only; gripper units are excluded.",
                         "Increments are rad/action, not physical velocity or acceleration.",
                         "Closed-loop rollouts diverge after different actions; they are not ground truth.",
                         "Normalized coordinate systems may differ between models; physical qpos is the common space.",
                         "Full-50-action predictions share exact observations and per-request sampling seeds.",
                         "Diagnostic-capped failures are NOT benchmark success-rate measurements."]})

    def save(fig, filename):
        fig.savefig(output / filename, dpi=150, facecolor="white")
        plt.close(fig)

    def joint_grid(title, subtitle):
        fig, axes = plt.subplots(7, 2, figsize=(14, 17), sharex=True)
        fig.suptitle(title + "\n" + subtitle, fontsize=14, y=.995)
        for j, axis in enumerate(axes.flat):
            axis.set_title(JOINTS[j], loc="left", fontsize=10)
            axis.grid(alpha=.18)
            axis.set_ylabel("grip command [0..1]" if j in (6, 13) else "joint target (rad)")
        return fig, axes

    # Identical input within the first request on our trajectory: genuine paired comparison.
    first_file = root / "ours/predictions/0000.npz"
    with np.load(first_file, allow_pickle=False) as data:
        first = {label: data[label].copy() for label in ("official", "ours")}
        norm = {label: data[f"{label}_normalized"].copy() for label in ("official", "ours")}
        norm_names = data["normalized_joint_names"].tolist()
        initial_state = data["state"].copy()
    length = runs["ours"]["requests"][0]["use_length"]
    fig, axes = joint_grid("Paired first-chunk predictions", "Identical RGB/state/instruction and sampling seed | full horizon 50 | shaded tail is NOT executed")
    for j, axis in enumerate(axes.flat):
        for label in ("official", "ours"):
            axis.plot(first[label][:, j], color=COLORS[label],
                      linestyle="-" if label == "official" else "--", label=label)
        axis.axhline(initial_state[j], color="#555555", lw=.8, linestyle=":")
        axis.axvspan(length - .5, 49.5, color="#EEEEEE", zorder=0)
    axes[0, 0].legend(loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Predicted action index (NOT video seconds)")
    fig.tight_layout(rect=(0, 0, 1, .955))
    save(fig, "01_same_observation_first_chunk.png")

    fig, axes = joint_grid("Executed targets and measured joint positions",
                          "Two separate closed-loop trajectories | action-index alignment only | dotted vertical lines mark replans")
    for j, axis in enumerate(axes.flat):
        for label, run in runs.items():
            axis.plot(run["executed"][:, j], color=COLORS[label], lw=1.1)
            if j in ARM:
                axis.plot(run["actual"][:, j], color=COLORS[label], lw=1.1, linestyle="--", alpha=.75)
        # Normally the same 20-action boundaries; use observed rather than assumed indices.
        starts = [item["action_start"] for item in runs["ours"]["requests"]][1:]
        for start in starts:
            axis.axvline(start - .5, color="#BBBBBB", lw=.6, linestyle=":")
    handles = [Line2D([0], [0], color=COLORS[label], label=label) for label in ("official", "ours")]
    handles += [Line2D([0], [0], color="#555555", label="target"),
                Line2D([0], [0], color="#555555", linestyle="--", label="measured arm qpos")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(.5, .955))
    for axis in axes[-1]:
        axis.set_xlabel("Executed action index (not physical time)")
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, "02_executed_targets_and_actual.png")

    arm_idx = [i for i, name in enumerate(norm_names) if name.startswith("action.arm.position[")]
    maps = {label: np.abs(np.diff(values[:, arm_idx], axis=0)).T for label, values in norm.items()}
    vmax = max(float(values.max()) for values in maps.values()) or 1
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("First-chunk normalized arm-action increments\nSame observation/seed | 12 real arm joints; padding and grippers excluded", fontsize=13)
    for axis, label in zip(axes, ("official", "ours")):
        image = axis.imshow(maps[label], origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        axis.set_title(label, loc="left")
        axis.set_ylabel("Arm joint index")
        axis.axvline(length - 1.5, color="#555555", linestyle="--", lw=1)
    axes[-1].set_xlabel("Consecutive predicted-action difference index (full horizon 50)")
    fig.colorbar(image, ax=axes, label="Absolute increment in each model's normalized coordinates")
    save(fig, "03_normalized_increment_heatmap.png")

    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, label in zip((-.18, .18), ("official", "ours")):
        info = metrics[label]["chunk_boundary_increments"]
        for i, category in enumerate(("inside", "boundary")):
            value = info[category]
            if value is not None:
                axis.bar(i + offset, value, width=.34, color=COLORS[label],
                         label=label if i == 0 else None, edgecolor="#444444", linewidth=.4)
                axis.annotate(f"{value:.4f}\nn={info[category + '_count']}",
                              (i + offset, value), ha="center", va="bottom", fontsize=9,
                              xytext=(0, 4), textcoords="offset points")
    axis.set_xticks([0, 1], ["Inside executed chunks", "Across replan boundaries"])
    bar_max = max((metrics[label]["chunk_boundary_increments"][category] or 0)
                  for label in metrics for category in ("inside", "boundary"))
    axis.set_ylim(0, (bar_max or 1) * 1.35)
    axis.set_ylabel("Mean absolute target increment (rad/action; 12 arm joints)")
    axis.set_title("Executed command increments by location\nSeparate closed-loop trajectories, one diagnostic episode/model; not accuracy")
    axis.legend()
    axis.grid(axis="y", alpha=.18)
    fig.tight_layout()
    save(fig, "04_inside_vs_boundary.png")
    # Sensitivity to sampling noise, holding the initial image/state/instruction fixed.
    with np.load(root / "ours/noise_probes.npz", allow_pickle=False) as probes:
        fig, axis = plt.subplots(figsize=(13, 5))
        for offset, label in zip((-.18, .18), ("official", "ours")):
            first_actions = probes[label][:, 0, :][:, ARM]
            spread = first_actions.std(axis=0)
            axis.bar(np.arange(12) + offset, spread, width=.34, color=COLORS[label],
                     edgecolor="#444444", linewidth=.4, label=label)
            metrics[label]["first_action_sampling_std_mean_rad"] = float(spread.mean())
        axis.set_xticks(np.arange(12), [JOINTS[j] for j in ARM], rotation=30, ha="right")
        axis.set_ylabel("First predicted action standard deviation (rad)")
        axis.set_ylim(bottom=0)
        axis.set_title(f"Sampling-noise sensitivity at one identical observation\n{len(probes['seeds'])} paired seeds; population std for these probes only, not calibrated uncertainty")
        axis.legend()
        axis.grid(axis="y", alpha=.18)
        fig.tight_layout()
        save(fig, "05_sampling_noise_sensitivity.png")
    # Add noise sensitivity to the summary without discarding earlier evidence notes.
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["metrics"] = metrics
    json_write(output / "summary.json", summary)
    print(f"Measured diagnostic charts saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run.resolve())
