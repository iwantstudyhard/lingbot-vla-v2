"""CPU regression tests; no LeRobot/NCCL/teacher weights needed.

Definition loading executes the real preprocessing classes while avoiding
unavailable production-only imports. It is NOT a torchcodec integration test.
"""

import ast
from collections import OrderedDict, defaultdict
from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING
import unittest

import numpy as np
from pydantic import BaseModel
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import yaml

from extensions.clean_stage2.augmentation import augment_views, augment_rgb, load_settings, safe_mask, sample_plan
from importlib.util import module_from_spec, spec_from_file_location
_boundary_spec = spec_from_file_location("episode_boundaries", Path(__file__).resolve().parents[1] / "lingbotvla/utils/episode_boundaries.py")
_boundary_module = module_from_spec(_boundary_spec)
_boundary_spec.loader.exec_module(_boundary_module)
bounded_timestamps = _boundary_module.bounded_timestamps
from extensions.clean_stage2.previews import PreviewRecorder

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "extensions/clean_stage2/augmentation.json"


def definitions(path, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def preprocessing_namespace():
    namespace = dict(globals())
    namespace.update({"logger": logging.getLogger("stage2-test"), "IMAGE_KEYS": (),
                      "IGNORE_INDEX": -100, "_is_quaternion_relative_type": lambda _: False})
    definitions(ROOT / "lingbotvla/data/vla_data/transform.py", namespace)
    definitions(ROOT / "lingbotvla/data/vla_data/utils.py", namespace)
    namespace.update({"VLADataset": object, "MultiVLADataset": object, "LeRobotDataset": object,
                      "LEROBOT_DATASET_API": "v3"})
    definitions(ROOT / "extensions/clean_stage2/dataset.py", namespace)
    return namespace


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        value = torch.arange(kwargs["max_length"]).unsqueeze(0)
        return {"input_ids": value, "attention_mask": torch.ones_like(value)}


class FakeImageProcessor:
    merge_size = 2

    def __call__(self, image):
        return {"pixel_values": image.float() / 255}


def raw_sample():
    rng = np.random.default_rng(11)
    image = torch.tensor(rng.integers(0, 256, (2, 3, 48, 64), dtype=np.uint8))
    return {
        "observation.state": torch.linspace(0.0, 0.5, 14),
        "action": torch.linspace(0.0, 0.5, 14).repeat(50, 1),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
        "observation.images.cam_high": image.clone(),
        "observation.images.cam_left_wrist": image.clone(),
        "observation.images.cam_right_wrist": image.clone(),
        "episode_index": torch.tensor(0), "frame_index": torch.tensor(30),
        "timestamp": torch.tensor(2.0), "task_index": torch.tensor(0), "task": "adjust bottle",
    }


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(SETTINGS)
        self.images = {key: np.full((3, 48, 64), 140, dtype=np.uint8)
                       for key in ("camera_top", "camera_wrist_left", "camera_wrist_right")}

    def test_clean_branch_is_bitwise_identity(self):
        seed = next(i for i in range(50) if sample_plan(self.settings, i)["branch"] == "clean")
        out, future, masks, plan = augment_views(self.images, self.images, self.settings, 0, seed)
        for key in out:
            np.testing.assert_array_equal(out[key], self.images[key])
            np.testing.assert_array_equal(future[key], self.images[key])
        self.assertEqual(plan["branch"], "clean")

    def test_scene_and_temporal_replay(self):
        settings = deepcopy(self.settings)
        settings["branch_probabilities"] = {"clean": 0, "photometric": 1, "texture": 0, "clutter": 0}
        for seed in range(12):
            out, future, _, plan = augment_views(self.images, self.images, settings, 0, seed)
            for key in out:
                np.testing.assert_array_equal(out[key], future[key])
                self.assertEqual(out[key].dtype, np.uint8)
                self.assertEqual(out[key].shape, self.images[key].shape)
            if plan["degradation"] != "noise" and not plan["shadow"]:
                np.testing.assert_array_equal(out["camera_top"], out["camera_wrist_left"])

    def test_seed_reproducible_and_inputs_not_mutated(self):
        before = deepcopy(self.images)
        first = augment_views(self.images, self.images, self.settings, 0, 20)
        second = augment_views(self.images, self.images, self.settings, 0, 20)
        for key in self.images:
            np.testing.assert_array_equal(first[0][key], second[0][key])
            np.testing.assert_array_equal(before[key], self.images[key])
        self.assertEqual(first[3], second[3])

    def test_probabilities_approximately_thirty_percent_clean(self):
        clean = sum(sample_plan(self.settings, seed)["branch"] == "clean" for seed in range(2000))
        self.assertTrue(0.27 < clean / 2000 < 0.33)

    def test_no_profile_no_overlay_and_wrist_always_protected(self):
        for camera in self.images:
            self.assertFalse(safe_mask(self.settings, 0, camera, (64, 48)).any())
        self.settings["safe_profiles"] = {"0": {"camera_wrist_left": {
            "reviewed_entire_episode": True, "editable_rectangles": [[0, 0, 1, 1]]}}}
        self.assertFalse(safe_mask(self.settings, 0, "camera_wrist_left", (64, 48)).any())

    def test_texture_falls_back_without_authorization(self):
        seed = next(i for i in range(100) if sample_plan(self.settings, i)["branch"] == "texture")
        _, _, _, plan = augment_views(self.images, self.images, self.settings, 0, seed)
        self.assertTrue(plan["overlay_fallback"])
        self.assertFalse(any(plan["overlay_coverage"].values()))

    def test_whole_episode_review_required(self):
        self.settings["safe_profiles"] = {"0": {"camera_top": {"editable_rectangles": [[0, 0, 1, 1]]}}}
        with self.assertRaises(ValueError):
            safe_mask(self.settings, 0, "camera_top", (64, 48))

    def test_final_overlay_does_not_leak_into_protected_pixels_even_with_jpeg(self):
        rgb = self.images["camera_top"].transpose(1, 2, 0)
        mask = np.zeros((48, 64), dtype=bool)
        mask[:20, :30] = True
        plan = sample_plan(self.settings, 20)
        plan.update(branch="texture", degradation="jpeg")
        augmented, _ = augment_rgb(rgb, plan, self.settings, mask, 0)
        baseline, _ = augment_rgb(rgb, plan, self.settings, np.zeros_like(mask), 0)
        np.testing.assert_array_equal(augmented[~mask], baseline[~mask])
        self.assertTrue(np.any(augmented[mask] != baseline[mask]))

    def test_clutter_coverage_capped(self):
        rgb = np.full((240, 320, 3), 140, dtype=np.uint8)
        for seed in range(20):
            plan = sample_plan(self.settings, seed)
            plan["branch"] = "clutter"
            _, fraction = augment_rgb(rgb, plan, self.settings, np.ones((240, 320), dtype=bool), 0)
            self.assertLessEqual(fraction, self.settings["clutter_max_image_fraction"])

    def test_zero_strength_is_identity(self):
        self.settings["strength"] = 0
        out, _, _, _ = augment_views(self.images, self.images, self.settings, 0, 20)
        for key in out:
            np.testing.assert_array_equal(out[key], self.images[key])

    def test_invalid_probability_fails(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            self.settings["branch_probabilities"]["clean"] = 0.9
            path.write_text(json.dumps(self.settings), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)


class BoundaryTests(unittest.TestCase):
    def test_episode_zero_last_row_cannot_read_episode_one(self):
        times, flags = bounded_timestamps([137 / 15, 138 / 15], 0, 9.2, 15)
        self.assertEqual([round(t * 15) for t in times], [137, 137])
        self.assertEqual(flags, [False, True])

    def test_offset_episode_and_both_ends(self):
        times, flags = bounded_timestamps([-1, 0, 999], 9.2, 18.666666666666668, 15)
        self.assertEqual([round(t * 15) for t in times], [138, 138, 279])
        self.assertEqual(flags, [True, False, True])

    def test_empty_or_off_grid_or_nonfinite_segments_fail(self):
        for start, end, fps in [(0, 0, 15), (0, 1, 0), (0.03, 1, 15), (0, float("inf"), 15)]:
            with self.assertRaises(ValueError):
                bounded_timestamps([0], start, end, fps)

    def test_full_actual_metadata_if_available(self):
        data = Path("D:/Google下载/RoboTwin_lerobot_v30/RoboTwin_lerobot_v30")
        if not data.exists():
            self.skipTest("Local clean dataset not available")
        import pyarrow.parquet as pq
        count = 0
        cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        columns = ["length"] + [f"videos/observation.images.{c}/{field}"
            for c in cameras for field in ("from_timestamp", "to_timestamp")]
        for file in (data / "meta/episodes").rglob("*.parquet"):
            for ep in pq.read_table(file, columns=columns).to_pylist():
                for camera in cameras:
                    prefix = f"videos/observation.images.{camera}"
                    start, end = ep[f"{prefix}/from_timestamp"], ep[f"{prefix}/to_timestamp"]
                    times, _ = bounded_timestamps([0, (ep["length"] - 1) / 15, ep["length"] / 15 + 49 / 15], start, end, 15)
                    self.assertTrue(all(start - 1e-6 <= t < end for t in times))
                    count += 1
        self.assertEqual(count, 7500)


class PreprocessingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = preprocessing_namespace()

    def make_transform(self, stage2=False, teacher_clean=True):
        data = SimpleNamespace(
            cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
            joints=["{'arm.position': 14}", "{'end.position': 14}", "{'effector.position': 2}"],
            norm_type=["{'arm.position': 'bounds_99_woclip'}", "{'end.position': 'bounds_99_woclip'}", "{'effector.position': 'bounds_99_woclip'}"],
        )
        config = SimpleNamespace(max_state_dim=55, max_action_dim=55, tokenizer_max_length=72)
        processor = SimpleNamespace(image_processor=FakeImageProcessor(), tokenizer=FakeTokenizer())
        kwargs = dict(use_depth_align=True, use_future_image=True)
        kwargs["norm_stats_path"] = str(ROOT / "assets/norm_stats/robotwin_clean_verified.json")
        name = "FeatureTransform"
        if stage2:
            name = "Stage2FeatureTransform"
            settings = load_settings(SETTINGS)
            settings["branch_probabilities"] = {"clean": 0, "photometric": 1, "texture": 0, "clutter": 0}
            settings["teacher_clean"] = teacher_clean
            kwargs.update(settings=settings, progress=torch.tensor(500))
        return self.namespace[name](str(ROOT / "configs/robot_configs/robotwin.yaml"), data, config, processor, **kwargs)

    def test_verified_normalization_roundtrip_preserves_physical_actions(self):
        transform = self.make_transform()
        original = raw_sample()
        transformed = transform.apply(deepcopy(original))
        restored = transform.unapply(transformed)
        np.testing.assert_allclose(np.asarray(restored["action"]), original["action"].numpy(), atol=2e-5)
        self.assertEqual(Path(transform.norm_stats_path), ROOT / "assets/norm_stats/robotwin_clean_verified.json")

    def test_training_dataset_passes_the_configured_statistics_to_actual_transform(self):
        tree = ast.parse((ROOT / "lingbotvla/data/vla_data/base_dataset.py").read_text())
        dataset = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VLADataset")
        calls = [node for node in ast.walk(dataset) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "FeatureTransform"]
        self.assertEqual(len(calls), 1)
        keyword = next(item for item in calls[0].keywords if item.arg == "norm_stats_path")
        self.assertEqual(ast.unparse(keyword.value), "getattr(dataset_config, 'norm_stats_file', None)")

    def test_stage2_keeps_state_actions_masks_and_language_bitwise_equal(self):
        original = raw_sample()
        first = self.make_transform().apply(deepcopy(original))
        second = self.make_transform(stage2=True).apply(deepcopy(original))
        for key in ("state", "actions", "action_is_pad", "joint_mask", "action_joint_mask", "state_joint_mask", "lang_tokens", "lang_masks"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertNotIn("stage2_preview_raw", first)
        self.assertTrue(torch.equal(original["observation.state"], raw_sample()["observation.state"]))

    def test_clean_teacher_augmented_student(self):
        original = raw_sample()
        first = self.make_transform().apply(deepcopy(original))
        second = self.make_transform(stage2=True).apply(deepcopy(original))
        self.assertTrue(torch.equal(first["pil_images"], second["pil_images"]))
        self.assertTrue(torch.equal(first["future_pil_images"], second["future_pil_images"]))
        self.assertFalse(torch.equal(first["images"], second["images"]))
        self.assertTrue(torch.equal(second["stage2_preview_raw"][:, 0], second["pil_images"]))

    def test_teacher_ablation_uses_augmented_teacher(self):
        second = self.make_transform(stage2=True, teacher_clean=False).apply(raw_sample())
        self.assertTrue(torch.equal(second["stage2_preview_aug"][:, 0], second["pil_images"]))

    def test_eval_does_not_augment_or_emit_training_previews(self):
        first = self.make_transform().apply(raw_sample(), policy_eval=True)
        second = self.make_transform(stage2=True).apply(raw_sample(), policy_eval=True)
        self.assertTrue(torch.equal(first["images"], second["images"]))
        self.assertNotIn("stage2_preview_raw", second)

    def test_recorder_pops_payload_and_writes_actual_batch_examples(self):
        feature = self.make_transform(stage2=True).apply(raw_sample())
        keys = ("stage2_preview_raw", "stage2_preview_aug", "stage2_preview_mask")
        batch = {key: feature[key].unsqueeze(0) for key in keys}
        batch["stage2_preview_json"] = [feature["stage2_preview_json"]]
        batch["state"] = feature["state"].unsqueeze(0)
        recorder = PreviewRecorder(torch.tensor(0))
        recorder.collect(batch, 500, 0)
        self.assertEqual(list(batch), ["state"])
        with TemporaryDirectory() as temporary:
            recorder.snapshot(500, temporary)
            directory = Path(temporary) / "analysis/by_checkpoint/global_step_500/augmentation"
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["examples"][0]["episode"], 0)
            self.assertTrue(list(directory.glob("*.jpg")))
        self.assertEqual(recorder.sample_count, 0)

    def test_action_tail_mask_still_excludes_padding_and_gradients(self):
        ns = definitions(ROOT / "lingbotvla/models/vla/lingbot_vla/loss_utils.py", {"torch": torch})
        losses = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]], requires_grad=True)
        reduced, _, _ = ns["reduce_action_losses"](losses, joint_mask=torch.ones_like(losses, dtype=torch.bool),
            action_is_pad=torch.tensor([[False, True]]), action_dim=2)
        self.assertEqual(reduced.item(), 1.5)
        reduced.backward()
        self.assertEqual(torch.count_nonzero(losses.grad[:, 1:]).item(), 0)

    def test_stage1_configuration_and_default_entry_remain_separate(self):
        first = yaml.safe_load((ROOT / "configs/vla/robotwin/robotwin_clean_stage1.yaml").read_text())
        second = yaml.safe_load((ROOT / "extensions/clean_stage2/config.yaml").read_text())
        self.assertEqual(first["train"]["max_steps"], 18000)
        self.assertFalse(first["train"]["enable_resume"])
        self.assertNotIn("stage2_augmentation_config", first["data"])
        self.assertFalse(second["train"]["enable_resume"])
        self.assertNotEqual(first["train"]["output_dir"], second["train"]["output_dir"])
        for key in ("freeze_vit", "freeze_vision_encoder", "train_expert_only", "global_batch_size", "micro_batch_size", "gradient_accumulation_steps"):
            self.assertEqual(first["train"][key], second["train"][key])
        tree = ast.parse((ROOT / "tasks/vla/train_lingbotvla.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        self.assertTrue(all(isinstance(value, ast.Constant) and value.value is None for value in main.args.kw_defaults))
        self.assertNotIn("extensions.clean_stage2", (ROOT / "tasks/vla/train_lingbotvla.py").read_text())


if __name__ == "__main__":
    unittest.main()
