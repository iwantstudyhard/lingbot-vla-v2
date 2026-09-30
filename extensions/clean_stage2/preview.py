"""Offline diagnostic previews from ONLY the selected clean dataset.

Training checkpoint previews come from the real consumed training batches;
this separate command only helps inspect settings before spending GPU time.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from .augmentation import augment_views, load_settings, sample_plan
from lingbotvla.utils.episode_boundaries import bounded_timestamps
from .previews import render_comparison
from lingbotvla.utils.arguments import training_output_path, workspace_path

REPO_ROOT = workspace_path(".")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
MODEL_CAMERAS = ("camera_top", "camera_wrist_left", "camera_wrist_right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--settings", default=str(REPO_ROOT / "extensions/clean_stage2/augmentation.json"))
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 500, 1000])
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = workspace_path(args.dataset_root)
    settings = load_settings(workspace_path(args.settings))
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = {}
    columns = ["episode_index", "length"] + [f"videos/observation.images.{camera}/{field}"
        for camera in CAMERAS for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    for file in sorted((root / "meta/episodes").rglob("*.parquet")):
        for episode in pq.read_table(file, columns=columns).to_pylist():
            if episode["episode_index"] in args.episodes:
                episodes[episode["episode_index"]] = episode
    if set(args.episodes) != set(episodes):
        raise ValueError("Unknown episode requested")
    output = workspace_path(args.output_dir) if args.output_dir else training_output_path({}, [], "stage2_preview")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for episode_index in args.episodes:
        episode = episodes[episode_index]
        current, future = {}, {}
        frame = min(30, max(0, episode["length"] // 3))
        for camera, model_camera in zip(CAMERAS, MODEL_CAMERAS):
            prefix = f"videos/observation.images.{camera}"
            times, _ = bounded_timestamps([frame / info["fps"], (frame + 49) / info["fps"]],
                episode[f"{prefix}/from_timestamp"], episode[f"{prefix}/to_timestamp"], info["fps"])
            video = root / f"videos/observation.images.{camera}/chunk-{episode[f'{prefix}/chunk_index']:03d}/file-{episode[f'{prefix}/file_index']:03d}.mp4"
            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot open video: {video}")
                decoded = []
                for time in times:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, round(time * info["fps"]))
                    success, bgr = capture.read()
                    if not success:
                        raise RuntimeError(f"Cannot decode video: {video} at {time}")
                    rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB)
                    decoded.append(rgb.transpose(2, 0, 1).copy())
                current[model_camera], future[model_camera] = decoded
            finally:
                capture.release()
        for branch in settings["branch_probabilities"]:
            if settings["branch_probabilities"][branch] == 0:
                continue
            seed = next(seed for seed in range(10000) if sample_plan(settings, seed)["branch"] == branch)
            aug, aug_future, masks, plan = augment_views(current, future, settings, episode_index, seed)
            keys = list(current)
            raw = np.stack([np.stack([current[key], future[key]]) for key in keys])
            augmented = np.stack([np.stack([aug[key], aug_future[key]]) for key in keys])
            mask = np.stack([masks[key] for key in keys])
            filename = f"episode_{episode_index}_{branch}.jpg"
            render_comparison(raw, augmented, mask, plan).save(output / filename, quality=92)
            records.append({"episode": episode_index, "frame": frame, "image": filename, **plan})
    (output / "manifest.json").write_text(json.dumps({
        "source": str(root.resolve()), "decoder": "OpenCV diagnostic; actual training previews use training batches",
        "settings": settings, "examples": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
