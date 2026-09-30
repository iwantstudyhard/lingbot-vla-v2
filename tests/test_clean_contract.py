"""Normalization, clean-source and launcher regression tests (CPU only)."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from lingbotvla.utils.normalization_contract import freeze_normalization, resolve_inference_normalization, semantic_hash
from lingbotvla.utils.checkpoint_retention import prune_old_checkpoints, update_best_checkpoint_record, best_checkpoint_path
from tools.verify_clean_norm import chunk_weights, weighted_stats
from tools.clean_training_common import REPO_ROOT, preflight, environment, validate_hf
from tools.launch_clean_stage1 import prepare_command as stage1_command
from extensions.clean_stage2.launch import prepare_command as stage2_command


class CleanContractTests(unittest.TestCase):
    def setUp(self):
        self.stats_file = REPO_ROOT / "assets/norm_stats/robotwin_clean_verified.json"
        self.stats = json.loads(self.stats_file.read_bytes())
        self.required = {"data": {"require_normalization_contract": True}}

    def test_snapshot_is_actual_training_and_inference_source(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            record = freeze_normalization(self.stats_file, run, 548893)
            hf = run / "checkpoints/global_step_500/hf_ckpt"
            self.assertEqual(resolve_inference_normalization(hf, self.required), record["path"])
            self.assertEqual(semantic_hash(json.loads(Path(record["path"]).read_bytes())), semantic_hash(self.stats))
            self.assertEqual(freeze_normalization(self.stats_file, run, 548893, True)["path"], record["path"])
            with self.assertRaises(ValueError):
                freeze_normalization(self.stats_file, run, 548893)

    def test_snapshot_accepts_current_launcher_log(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            log = run / "stage1_current.log"
            log.write_text("current launch\n", encoding="utf-8")
            env = {"LINGBOT_TRAIN_RUN_DIR": str(run), "LINGBOT_TRAIN_RUN_ID": "current",
                   "TRAIN_LOG_FILE": str(log)}
            with patch.dict("os.environ", env, clear=True):
                record = freeze_normalization(self.stats_file, run, 548893)
                self.assertEqual(Path(record["path"]), run / "normalization/norm_stats.json")
                self.assertEqual(record["source"], str(self.stats_file.resolve()))
                self.assertEqual(log.read_text(encoding="utf-8"), "current launch\n")
                with self.assertRaises(ValueError):
                    freeze_normalization(self.stats_file, run, 548893)

    def test_launcher_log_does_not_allow_old_training_artifacts(self):
        for artifact in ("checkpoints", "previous.log"):
            with self.subTest(artifact=artifact), TemporaryDirectory() as tmp:
                run = Path(tmp) / "run"
                run.mkdir()
                log = run / "stage1_current.log"
                log.touch()
                if artifact == "checkpoints":
                    (run / artifact).mkdir()
                else:
                    (run / artifact).touch()
                env = {"LINGBOT_TRAIN_RUN_DIR": str(run), "LINGBOT_TRAIN_RUN_ID": "current",
                       "TRAIN_LOG_FILE": str(log)}
                with patch.dict("os.environ", env, clear=True):
                    with self.assertRaisesRegex(ValueError, "new output directory"):
                        freeze_normalization(self.stats_file, run, 548893)
                self.assertFalse((run / "normalization").exists())

    def test_snapshot_rejects_unowned_log_directory(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            log = run / "stage1_current.log"
            log.touch()
            env = {"LINGBOT_TRAIN_RUN_DIR": str(run), "LINGBOT_TRAIN_RUN_ID": "current",
                   "TRAIN_LOG_FILE": str(log)}
            for missing in env:
                with self.subTest(missing=missing):
                    incomplete = {key: value for key, value in env.items() if key != missing}
                    with patch.dict("os.environ", incomplete, clear=True):
                        with self.assertRaisesRegex(ValueError, "new output directory"):
                            freeze_normalization(self.stats_file, run, 548893)
            with patch.dict("os.environ", {**env, "LINGBOT_TRAIN_RUN_DIR": str(run.parent)}, clear=True):
                with self.assertRaisesRegex(ValueError, "new output directory"):
                    freeze_normalization(self.stats_file, run, 548893)

    def test_inference_refuses_wrong_override_or_corrupted_snapshot(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            record = freeze_normalization(self.stats_file, run, 548893)
            hf = run / "checkpoints/global_step_500/hf_ckpt"
            with self.assertRaises(ValueError):
                resolve_inference_normalization(hf, self.required, REPO_ROOT / "assets/norm_stats/robotwin.json")
            changed = deepcopy(self.stats)
            changed["norm_stats"]["action.arm.position"]["q99"][0] += 0.1
            Path(record["path"]).write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_inference_normalization(hf, self.required)

    def test_snapshot_can_move_with_whole_run(self):
        with TemporaryDirectory() as tmp:
            old = Path(tmp) / "old"
            freeze_normalization(self.stats_file, old, 548893)
            new = Path(tmp) / "new"
            old.rename(new)
            path = resolve_inference_normalization(new / "checkpoints/global_step_500/hf_ckpt", self.required)
            self.assertEqual(Path(path), new / "normalization/norm_stats.json")

    def test_missing_contract_and_legacy_resume_fail(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "old"
            (run / "checkpoints/global_step_2000").mkdir(parents=True)
            with self.assertRaises(ValueError):
                freeze_normalization(self.stats_file, run, 548893, True)
            with self.assertRaises(ValueError):
                resolve_inference_normalization(run / "checkpoints/global_step_2000/hf_ckpt", self.required)

    def test_counts_nonfinite_and_schema_fail(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                freeze_normalization(self.stats_file, Path(tmp) / "run", 123)
            bad = deepcopy(self.stats)
            bad["norm_stats"]["action.arm.position"]["mean"][0] = float("nan")
            source = Path(tmp) / "bad.json"
            source.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                freeze_normalization(source, Path(tmp) / "run", 548893)

    def test_semantic_hash_independent_of_json_format(self):
        self.assertEqual(semantic_hash(self.stats), semantic_hash(json.loads(json.dumps(self.stats, indent=4))))

    def test_source_metadata_and_all_camera_segments(self):
        root = Path("D:/Google下载/RoboTwin_lerobot_v30/RoboTwin_lerobot_v30")
        if not root.exists():
            self.skipTest("Full clean dataset unavailable")
        config = yaml.safe_load((REPO_ROOT / "configs/vla/robotwin/robotwin_clean_stage1.yaml").read_text())
        with TemporaryDirectory() as tmp:
            marker = Path(tmp) / "dependency"
            marker.write_text("path-only fixture")
            manifest = Path(tmp) / "manifest.txt"
            manifest.write_text(f"robotwin {root.as_posix()}\n", encoding="utf-8")
            config["data"]["train_path"] = str(manifest)
            config["model"]["tokenizer_path"] = str(marker)
            depth, video = config["train"]["align_params"]["depth"], config["train"]["align_params"]["video"]
            depth["moge_path"] = depth["morgbd_path"] = str(marker)
            video["ckpt_path"] = video["config_path"] = str(marker)
            self.assertEqual(preflight(config), self.stats_file.resolve())

    def test_chunk_weights_match_explicit_terminal_padded_windows(self):
        for length in (1, 2, 7, 50, 139):
            explicit = np.zeros(length, dtype=np.int64)
            for anchor in range(length):
                for delta in range(50):
                    explicit[min(length - 1, anchor + delta)] += 1
            np.testing.assert_array_equal(chunk_weights(length, 50), explicit)
        result = weighted_stats([[0], [10]], [1, 3])
        self.assertEqual(result["mean"], [7.5])

    def test_retention_is_best_plus_latest_within_three(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (500, 1000, 1500, 2000):
                (root / f"global_step_{step}").mkdir()
            update_best_checkpoint_record(root, step=500, metric_value=0.1, window_start_step=1, window_end_step=500)
            prune_old_checkpoints(root, 3, preferred_paths=[best_checkpoint_path(root)])
            self.assertEqual(sorted(path.name for path in root.iterdir() if path.is_dir()),
                             ["global_step_1500", "global_step_2000", "global_step_500"])

    def test_environment_and_missing_shards(self):
        with self.assertRaises(ValueError):
            environment("0,0,1,2", 62500)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}")
            (root / "part.safetensors").touch()
            (root / "model.safetensors.index.json").write_text('{"weight_map":{"x":"missing.safetensors"}}')
            with self.assertRaises(ValueError):
                validate_hf(root)

    def test_stage1_fresh_command_has_no_augmentation_entry(self):
        with TemporaryDirectory() as tmp:
            weights = Path(tmp) / "weights"
            weights.mkdir()
            (weights / "config.json").write_text("{}")
            (weights / "model.safetensors").touch()
            args = SimpleNamespace(gpus="0,1,2,3", master_port=62500, resume_run=None,
                                   init_hf=str(weights), output_dir=str(Path(tmp) / "output"))
            with patch("tools.launch_clean_stage1.preflight"):
                command, env = stage1_command(args)
            self.assertIn("tasks/vla/train_lingbotvla.py", command)
            self.assertEqual(command[command.index("--train.enable_resume") + 1], "false")
            self.assertFalse(any("stage2" in value for value in command))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_stage2_refuses_legacy_checkpoint(self):
        with TemporaryDirectory() as tmp:
            weights = Path(tmp) / "old/checkpoints/global_step_2000/hf_ckpt"
            weights.mkdir(parents=True)
            (weights / "config.json").write_text("{}")
            (weights / "model.safetensors").touch()
            args = SimpleNamespace(init_hf=str(weights), steps=500, gpus="0,1,2,3",
                                   master_port=62510, output_dir=None)
            with patch("extensions.clean_stage2.launch.preflight", return_value=self.stats_file):
                with self.assertRaisesRegex(ValueError, "historical mixed-stat"):
                    stage2_command(args)

    def test_stage2_fresh_command_preserves_scales_and_separate_entry(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "stage1"
            freeze_normalization(self.stats_file, run, 548893)
            (run / "lingbotvla_cli.yaml").write_text("data: {}")
            weights = run / "checkpoints/global_step_500/hf_ckpt"
            weights.mkdir(parents=True)
            (weights / "config.json").write_text("{}")
            (weights / "model.safetensors").touch()
            args = SimpleNamespace(init_hf=str(weights), steps=500, gpus="0,1,2,3",
                                   master_port=62510, output_dir=str(Path(tmp) / "stage2"))
            with patch("extensions.clean_stage2.launch.preflight", return_value=self.stats_file):
                command, _ = stage2_command(args)
            self.assertIn("extensions/clean_stage2/train.py", command)
            self.assertNotIn("--train.load_checkpoint_path", command)
            self.assertEqual(command[command.index("--train.lr_warmup_ratio") + 1], "0.3")


if __name__ == "__main__":
    unittest.main()
