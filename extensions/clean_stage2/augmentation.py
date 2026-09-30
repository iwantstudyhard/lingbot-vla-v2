"""Clean-pixel photometric transforms and guarded procedural backgrounds.

No random-data images, simulator calls, geometry warps, or learned generators.
One replay plan per sample; local spatial patterns are fixed across time.
"""

from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CAMERAS = ("camera_top", "camera_wrist_left", "camera_wrist_right")


def load_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    probabilities = settings["branch_probabilities"]
    if set(probabilities) != {"clean", "photometric", "texture", "clutter"}:
        raise ValueError("Expected four explicit stage2 augmentation branches")
    if any(p < 0 for p in probabilities.values()) or not np.isclose(sum(probabilities.values()), 1):
        raise ValueError("Branch probabilities must sum to one")
    for name in ("brightness", "contrast", "gamma", "saturation", "white_balance",
                 "noise_std", "blur_sigma", "jpeg_quality", "shadow_strength"):
        low, high = settings[name]
        if not np.isfinite([low, high]).all() or low > high or low < 0:
            raise ValueError(f"Invalid augmentation range: {name}")
    if settings["gamma"][0] <= 0 or settings["jpeg_quality"][0] < 1 or settings["jpeg_quality"][1] > 100:
        raise ValueError("Invalid gamma/JPEG range")
    if not 0 <= settings["texture_alpha"] <= 1 or not 0 <= settings["clutter_max_image_fraction"] <= 0.05:
        raise ValueError("Unsafe texture/clutter strength")
    if not isinstance(settings.get("safe_profiles", {}), dict):
        raise ValueError("safe_profiles must be a mapping")
    return settings


def sample_plan(settings, seed):
    rng = np.random.default_rng(seed)
    names = list(settings["branch_probabilities"])
    branch = str(rng.choice(names, p=list(settings["branch_probabilities"].values())))
    plan = {"seed": int(seed), "branch": branch}
    for name in ("brightness", "contrast", "gamma", "saturation", "shadow_strength"):
        plan[name] = float(rng.uniform(*settings[name]))
    plan["white_balance"] = rng.uniform(*settings["white_balance"], size=3).tolist()
    # Do not stack every imaging degradation on every sample.
    plan["degradation"] = str(rng.choice(["none", "noise", "blur", "jpeg"], p=[0.4, 0.25, 0.15, 0.2]))
    plan["noise_std"] = float(rng.uniform(*settings["noise_std"]))
    plan["blur_sigma"] = float(rng.uniform(*settings["blur_sigma"]))
    plan["jpeg_quality"] = int(rng.integers(settings["jpeg_quality"][0], settings["jpeg_quality"][1] + 1))
    plan["shadow"] = bool(rng.random() < 0.3)
    plan["texture_color"] = rng.uniform(0.25, 0.75, size=3).tolist()
    strength = float(settings.get("strength", 1.0))
    if not 0 <= strength <= 1:
        raise ValueError("Strength must be within [0, 1]")
    for name in ("brightness", "contrast", "gamma", "saturation"):
        plan[name] = 1 + (plan[name] - 1) * strength
    plan["white_balance"] = [1 + (v - 1) * strength for v in plan["white_balance"]]
    plan["noise_std"] *= strength
    plan["blur_sigma"] *= strength
    plan["shadow_strength"] *= strength
    plan["jpeg_quality"] = round(100 - (100 - plan["jpeg_quality"]) * strength)
    if strength == 0:
        plan["degradation"] = "none"
    plan["strength"] = strength
    return plan


def _rectangle_mask(size, rectangles):
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    for rectangle in rectangles:
        if len(rectangle) != 4 or not all(0 <= float(v) <= 1 for v in rectangle):
            raise ValueError("Safe-profile rectangles must be normalized xyxy")
        x0, y0, x1, y1 = rectangle
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Invalid safe-profile rectangle")
        mask[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)] = 255
    return mask


def safe_mask(settings, episode, camera, size):
    """Only manually audited episode+camera profiles can authorize overlays.

    Default is all protected. No keyword/task-index inference or white-pixel
    heuristic is treated as semantic segmentation. Wrist overlays are disabled.
    Profiles must protect the entire trajectory, not just the first frame.
    """
    empty = np.zeros((size[1], size[0]), dtype=bool)
    if camera != "camera_top":
        return empty
    profile = settings.get("safe_profiles", {}).get(str(episode), {}).get(camera)
    if not profile:
        return empty
    if profile.get("reviewed_entire_episode") is not True:
        raise ValueError("Background profile needs whole-episode review")
    editable = _rectangle_mask(size, profile.get("editable_rectangles", []))
    protected = _rectangle_mask(size, profile.get("protected_rectangles", []))
    editable[protected > 0] = 0
    margin = int(settings["safe_margin_pixels"])
    if margin < 1:
        raise ValueError("Safe mask needs a positive protection margin")
    # Shrink editable areas before all blur/compositing; final alpha also uses it.
    return np.asarray(Image.fromarray(editable).filter(ImageFilter.MinFilter(2 * margin + 1))) == 255


def _pattern(size, seed):
    width, height = size
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(6, 8), dtype=np.uint8)
    low_frequency = np.asarray(Image.fromarray(noise).resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    stripe = np.sin(x / max(8, width / 12) + y / max(8, height / 12))
    return np.clip(0.5 + 0.25 * (low_frequency - 0.5) + 0.08 * stripe, 0, 1)


def augment_rgb(image, plan, settings, mask, camera_index):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Stage2 expects RGB uint8 HWC before model normalization")
    if mask.shape != image.shape[:2]:
        raise ValueError("Safe mask/image size mismatch")
    if plan["branch"] == "clean":
        return image.copy(), 0.0
    height, width = image.shape[:2]
    size = (width, height)
    value = image.astype(np.float32) / 255
    gray = np.sum(value * [0.299, 0.587, 0.114], axis=-1, keepdims=True)
    value = gray + plan["saturation"] * (value - gray)
    value = (value - 0.5) * plan["contrast"] + 0.5
    value = np.clip(value * plan["brightness"] * np.array(plan["white_balance"]), 0, 1)
    value = np.power(value, plan["gamma"])
    # A fixed camera-local field is replayed on current and future frames.
    pattern = _pattern(size, plan["seed"] + camera_index * 1009)
    if plan["shadow"]:
        value *= 1 - plan["shadow_strength"] * pattern[..., None]
    # Apply imaging degradation BEFORE overlays so JPEG/blur cannot leak
    # synthetic shapes through the final protected-pixel boundary.
    if plan["degradation"] == "noise":
        rng = np.random.default_rng(plan["seed"] + camera_index * 1009 + 7)
        value += rng.normal(0, plan["noise_std"], size=value.shape)
    result = Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8))
    if plan["degradation"] == "blur":
        result = result.filter(ImageFilter.GaussianBlur(plan["blur_sigma"]))
    elif plan["degradation"] == "jpeg":
        stream = BytesIO()
        result.save(stream, format="JPEG", quality=plan["jpeg_quality"])
        stream.seek(0)
        with Image.open(stream) as decoded:
            result = decoded.convert("RGB").copy()
    value = np.array(result, dtype=np.float32) / 255
    coverage = 0.0
    if plan["branch"] in ("texture", "clutter") and mask.any():
        color = np.array(plan["texture_color"])
        texture = np.clip(color[None, None, :] + (pattern[..., None] - 0.5) * 0.4, 0, 1)
        if plan["branch"] == "texture":
            alpha = mask.astype(np.float32) * settings["texture_alpha"] * plan["strength"]
        else:
            rng = np.random.default_rng(plan["seed"] + camera_index * 1009)
            shape = Image.new("L", size)
            draw = ImageDraw.Draw(shape)
            for _ in range(3):
                x, y = rng.integers(0, width), rng.integers(0, height)
                w, h = max(2, int(width * 0.07)), max(2, int(height * 0.07))
                coordinates = (int(x), int(y), int(x + w), int(y + h))
                if rng.random() < 0.5:
                    draw.ellipse(coordinates, fill=255)
                else:
                    draw.rectangle(coordinates, fill=255)
            # Feather first, then enforce the safe region again: no alpha leakage.
            alpha = np.asarray(shape.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32) / 255
            alpha *= mask
            eligible = np.flatnonzero(alpha > 0)
            budget = int(width * height * settings["clutter_max_image_fraction"])
            if len(eligible) > budget:
                keep = rng.choice(eligible, size=budget, replace=False)
                limited = np.zeros_like(alpha)
                limited.flat[keep] = alpha.flat[keep]
                alpha = limited
            alpha *= 0.5 * plan["strength"]
        coverage = float(np.mean(alpha > 0))
        value = value * (1 - alpha[..., None]) + texture * alpha[..., None]
    return np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), coverage


def augment_views(current, future, settings, episode, seed):
    """Apply one scene plan to CHW numpy arrays, retaining original images."""
    plan = sample_plan(settings, seed)
    outputs, future_outputs, masks, coverage = {}, {}, {}, {}
    for index, (key, image) in enumerate(current.items()):
        camera = key.rsplit(".", 1)[-1]
        size = (image.shape[-1], image.shape[-2])
        mask = safe_mask(settings, episode, camera, size)
        rgb, amount = augment_rgb(image.transpose(1, 2, 0), plan, settings, mask, index)
        outputs[key] = rgb.transpose(2, 0, 1).copy()
        masks[key] = mask.astype(np.uint8)
        coverage[key] = amount
        if key in future:
            if future[key].shape != image.shape:
                raise ValueError("Current/future camera sizes must match")
            rgb, _ = augment_rgb(future[key].transpose(1, 2, 0), plan, settings, mask, index)
            future_outputs[key] = rgb.transpose(2, 0, 1).copy()
    plan["overlay_coverage"] = coverage
    plan["overlay_fallback"] = plan["branch"] in ("texture", "clutter") and not any(coverage.values())
    return outputs, future_outputs, masks, plan
