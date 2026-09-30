"""Bounded main-process collection of actual training examples."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class PreviewRecorder:
    def __init__(self, progress, max_samples=4):
        self.progress = progress
        self.max_samples = max_samples
        self.samples = {}
        self.counts = Counter()
        self.fallback_count = 0
        self.sample_count = 0

    def collect(self, batch, step, rank):
        self.progress.fill_(step)
        keys = ("stage2_preview_raw", "stage2_preview_aug", "stage2_preview_mask", "stage2_preview_json")
        payload = [batch.pop(key, None) for key in keys]
        if rank != 0 or payload[0] is None:
            return
        raw, augmented, mask, metadata = payload
        for index, serialized in enumerate(metadata):
            record = json.loads(serialized)
            self.sample_count += 1
            self.counts[record["branch"]] += 1
            self.fallback_count += int(record["overlay_fallback"])
            # One recent sample per branch avoids previews containing only raw clean.
            self.samples[record["branch"]] = (
                raw[index].cpu().numpy().copy(), augmented[index].cpu().numpy().copy(),
                mask[index].cpu().numpy().copy(), {**record, "optimizer_step": step},
            )

    def snapshot(self, step, visualization_run_dir):
        directory = Path(visualization_run_dir) / "analysis" / "by_checkpoint" / f"global_step_{step}" / "augmentation"
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for number, (raw, augmented, mask, metadata) in enumerate(list(self.samples.values())[:self.max_samples]):
            image = render_comparison(raw, augmented, mask, metadata)
            filename = f"sample_{number}_{metadata['branch']}.jpg"
            image.save(directory / filename, quality=92)
            records.append({**metadata, "image": filename})
        manifest = {
            "checkpoint_step": step, "scope": "rank0 consumed samples since previous snapshot",
            "sample_count": self.sample_count, "branch_counts": dict(self.counts),
            "overlay_fallback_count": self.fallback_count, "examples": records,
            "teacher_input": "clean RGB when teacher_clean=true",
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.samples.clear()
        self.counts.clear()
        self.sample_count = self.fallback_count = 0


def render_comparison(raw, augmented, mask, metadata):
    """Columns: clean current, augmented current, clean future, augmented future, safe mask."""
    height, width = raw.shape[-2:]
    canvas = Image.new("RGB", (width * 5, (height + 26) * len(raw)), "white")
    draw = ImageDraw.Draw(canvas)
    for camera_index in range(len(raw)):
        panels = [raw[camera_index, 0], augmented[camera_index, 0],
                  raw[camera_index, 1], augmented[camera_index, 1]]
        y = camera_index * (height + 26)
        for column, value in enumerate(panels):
            canvas.paste(Image.fromarray(value.transpose(1, 2, 0).astype(np.uint8)), (column * width, y + 26))
        rgb_mask = np.repeat((mask[camera_index] * 255)[..., None], 3, axis=-1).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb_mask), (width * 4, y + 26))
        draw.text((4, y + 4), f"cam{camera_index} | clean / aug / future clean / future aug / editable mask", fill="black")
    return canvas
