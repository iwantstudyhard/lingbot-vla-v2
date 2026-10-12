"""CPU regressions for split DCP layout and exact audited stage-one resume."""

import ast
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict
import unittest
from unittest.mock import Mock, patch

import yaml

from lingbotvla.utils.checkpoint_layout import resolve_resume_checkpoint, validate_training_checkpoint
from lingbotvla.utils.normalization_contract import freeze_normalization
from tools.clean_training_common import REPO_ROOT
from tools.launch_clean_stage1 import prepare_command


def fixture_checkpoint(path, world_size=4):
    for component in ("model", "optimizer"):
        store = path / component
        store.mkdir(parents=True)
        (store / ".metadata").write_bytes(b"fixture: structural checks only")
        (store / "__0_0.distcp").write_bytes(b"fixture tensor data")
    extra = path / "extra_state"
    extra.mkdir()
    for rank in range(world_size):
        (extra / f"extra_state_rank_{rank}.pt").write_bytes(b"fixture extra state")
    return path


class LayoutTests(unittest.TestCase):
    def test_real_layout_does_not_require_root_metadata(self):
        with TemporaryDirectory() as tmp:
            path = fixture_checkpoint(Path(tmp) / "global_step_19500")
            self.assertFalse((path / ".metadata").exists())
            self.assertEqual(validate_training_checkpoint(path, 4), path.resolve())

    def test_root_metadata_and_hf_alone_do_not_allow_resume(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "global_step_19500"
            (path / "hf_ckpt").mkdir(parents=True)
            (path / ".metadata").write_bytes(b"wrong location")
            (path / "hf_ckpt/config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, r"model/\.metadata.*optimizer/\.metadata"):
                validate_training_checkpoint(path, 4)

    def test_missing_or_empty_components_are_rejected(self):
        for relative in ("model/.metadata", "optimizer/.metadata", "model/__0_0.distcp",
                         "optimizer/__0_0.distcp", "extra_state/extra_state_rank_3.pt"):
            for empty in (False, True):
                with self.subTest(relative=relative, empty=empty), TemporaryDirectory() as tmp:
                    path = fixture_checkpoint(Path(tmp) / "global_step_19500")
                    if empty:
                        (path / relative).write_bytes(b"")
                    else:
                        (path / relative).unlink()
                    with self.assertRaisesRegex(ValueError, "Incomplete split DCP"):
                        validate_training_checkpoint(path, 4)

    def test_every_requested_rank_must_exist(self):
        with TemporaryDirectory() as tmp:
            path = fixture_checkpoint(Path(tmp) / "global_step_19500", 2)
            validate_training_checkpoint(path, 2)
            with self.assertRaisesRegex(ValueError, "extra_state_rank_2"):
                validate_training_checkpoint(path, 4)

    def test_latest_uses_numeric_steps_and_explicit_choice_is_pinned(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = fixture_checkpoint(root / "global_step_9500")
            newer = fixture_checkpoint(root / "global_step_19500")
            (root / "global_step_best").mkdir()
            self.assertEqual(resolve_resume_checkpoint(root, 4), newer.resolve())
            self.assertEqual(resolve_resume_checkpoint(root, 4, 9500), older.resolve())

    def test_incomplete_latest_does_not_silently_fall_back(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = fixture_checkpoint(root / "global_step_19500")
            (root / "global_step_20000/hf_ckpt").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "--resume-step"):
                resolve_resume_checkpoint(root, 4)
            self.assertEqual(resolve_resume_checkpoint(root, 4, 19500), older.resolve())

    def test_missing_checkpoint_is_not_created(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing"
            with self.assertRaises(ValueError):
                resolve_resume_checkpoint(root, 4, 19500)
            self.assertFalse(root.exists())

    def test_invalid_world_sizes_and_steps(self):
        with TemporaryDirectory() as tmp:
            for value in (0, -1, True, 1.5):
                with self.subTest(world_size=value), self.assertRaises(ValueError):
                    validate_training_checkpoint(tmp, value)
            for value in (-1, True, 1.5):
                with self.subTest(step=value), self.assertRaises(ValueError):
                    resolve_resume_checkpoint(tmp, 4, value)

    def test_actual_cpu_dcp_stores_have_nested_metadata(self):
        import torch
        import torch.distributed.checkpoint as dcp
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "global_step_19500"
            for component in ("model", "optimizer"):
                dcp.save({"probe": torch.tensor([1.0, 2.0])}, checkpoint_id=path / component)
            (path / "extra_state").mkdir()
            torch.save({"global_step": 19500}, path / "extra_state/extra_state_rank_0.pt")
            validate_training_checkpoint(path, 1)
            self.assertFalse((path / ".metadata").exists())
            state = {"probe": torch.zeros(2)}
            dcp.load(state, checkpoint_id=path / "model")
            torch.testing.assert_close(state["probe"], torch.tensor([1.0, 2.0]))


class LauncherTests(unittest.TestCase):
    def setup_run(self, root):
        run = root / "run"
        stats = REPO_ROOT / "assets/norm_stats/robotwin_clean_verified.json"
        norm = freeze_normalization(stats, run, 548893)
        weights = root / "initial_weights"
        weights.mkdir()
        (weights / "config.json").write_text("{}")
        (weights / "model.safetensors").write_bytes(b"fixture")
        cfg = {"model": {"model_path": str(weights)},
               "data": {"require_normalization_contract": True, "norm_stats_file": norm["path"]},
               "train": {"output_dir": str(run), "max_steps": 30000, "lr": 1e-5}}
        (run / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return run, cfg

    def args(self, run=None, step=None):
        return SimpleNamespace(gpus="0,1,2,3", master_port=62500, resume_run=str(run) if run else None,
                               resume_step=step, init_hf=None, output_dir=None, smoke=False)

    def test_resume_pins_nested_dcp_without_changing_loss_or_schedule(self):
        with TemporaryDirectory() as tmp:
            run, cfg = self.setup_run(Path(tmp))
            cp = fixture_checkpoint(run / "checkpoints/global_step_19500")
            before = (run / "lingbotvla_cli.yaml").read_bytes()
            with patch("tools.launch_clean_stage1.preflight"):
                command, env = prepare_command(self.args(run, 19500))
            self.assertEqual(command[command.index("--train.load_checkpoint_path") + 1], str(cp.resolve()))
            self.assertEqual(command[command.index("--train.enable_resume") + 1], "true")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
            self.assertEqual(before, (run / "lingbotvla_cli.yaml").read_bytes())
            self.assertNotIn("--train.lr", command)
            self.assertNotIn("--train.max_steps", command)

    def test_resume_step_requires_resume_run(self):
        with self.assertRaisesRegex(ValueError, "requires --resume-run"):
            prepare_command(self.args(step=19500))

    def test_old_configured_checkpoint_needs_explicit_override(self):
        with TemporaryDirectory() as tmp:
            run, cfg = self.setup_run(Path(tmp))
            cp = fixture_checkpoint(run / "checkpoints/global_step_19500")
            cfg["train"]["load_checkpoint_path"] = "old/global_step_18000"
            (run / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "choose --resume-step"):
                prepare_command(self.args(run))
            with patch("tools.launch_clean_stage1.preflight"):
                command, _ = prepare_command(self.args(run, 19500))
            self.assertEqual(command[command.index("--train.load_checkpoint_path") + 1], str(cp.resolve()))


class OptimizerRestoreTests(unittest.TestCase):
    def test_production_cpu_roundtrip_restores_model_optimizer_and_training_state(self):
        import torch
        from lingbotvla.checkpoint.checkpointer import DistributedCheckpointer

        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        inputs = torch.tensor([[0.3, 0.7]])
        model(inputs).square().sum().backward()
        optimizer.step()
        extra = {"global_step": 19500, "lr_scheduler": {"last_epoch": 19500},
                 "train_dataloader": {"position": 1234}, "torch_rng_state": torch.get_rng_state()}
        with TemporaryDirectory() as tmp, patch("lingbotvla.checkpoint.checkpointer.dist.get_rank", return_value=0):
            DistributedCheckpointer.save(tmp, {"model": model, "optimizer": optimizer, "extra_state": extra})
            validate_training_checkpoint(tmp, 1)
            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.9)
            loaded = DistributedCheckpointer.load(tmp, {"model": restored_model,
                "optimizer": restored_optimizer, "extra_state": {}})
            self.assertEqual(loaded["extra_state"]["global_step"], 19500)
            self.assertEqual(loaded["extra_state"]["lr_scheduler"], extra["lr_scheduler"])
            self.assertEqual(loaded["extra_state"]["train_dataloader"], extra["train_dataloader"])
            torch.testing.assert_close(loaded["extra_state"]["torch_rng_state"], extra["torch_rng_state"])
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], 0.01)
            for original, restored in zip(model.parameters(), restored_model.parameters()):
                torch.testing.assert_close(original, restored)
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[original][name], restored_optimizer.state[restored][name])
            # A further step must match, not merely the saved parameter values.
            for network, opt in ((model, optimizer), (restored_model, restored_optimizer)):
                opt.zero_grad(set_to_none=True)
                network(inputs).square().sum().backward()
                opt.step()
            for original, restored in zip(model.parameters(), restored_model.parameters()):
                torch.testing.assert_close(original, restored, rtol=0, atol=0)

    def loader_namespace(self, optimizer_failure=False):
        # Execute the production load method without importing GPU/FSDP modules.
        tree = ast.parse((REPO_ROOT / "lingbotvla/checkpoint/checkpointer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DistributedCheckpointer")
        cls = deepcopy(cls)
        cls.bases = [ast.Name(id="object", ctx=ast.Load())]
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load"]
        def load(**kwargs):
            if optimizer_failure and Path(kwargs["storage_reader"]).name == "optimizer":
                raise RuntimeError("fixture optimizer failed")
        import os
        namespace = dict(os=os, Dict=Dict, Any=Any,
                         ModelState=lambda model: model, OptimizerState=lambda **kwargs: kwargs,
                         FileSystemReader=lambda path: path,
                         DefaultLoadPlanner=lambda **kwargs: SimpleNamespace(**kwargs),
                         dcp=SimpleNamespace(load=Mock(side_effect=load)),
                         dist=SimpleNamespace(get_rank=lambda: 0),
                         torch=SimpleNamespace(load=Mock(return_value={"global_step": 19500,
                             "lr_scheduler": "saved scheduler", "train_dataloader": "saved dataloader"})),
                         logger=SimpleNamespace(info_rank0=Mock()), _MODEL_DIR="model", _EMA_DIR="ema",
                         _OPTIMIZER_DIR="optimizer", _EXTRA_STATE_DIR="extra_state",
                         _EXTRA_STATE_FORMAT="extra_state_rank_{}.pt")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "checkpointer.load", "exec"), namespace)
        return namespace

    def test_optimizer_failure_propagates_instead_of_silently_resetting(self):
        ns = self.loader_namespace(True)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
                ns["DistributedCheckpointer"].load(tmp, {"model": object(), "optimizer": object(), "extra_state": {}})
        ns["torch"].load.assert_not_called()

    def test_exact_resume_loads_optimizer_strictly_and_preserves_extra_state(self):
        ns = self.loader_namespace()
        with TemporaryDirectory() as tmp:
            result = ns["DistributedCheckpointer"].load(tmp, {"model": object(), "optimizer": object(), "extra_state": {}})
        calls = ns["dcp"].load.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(Path(calls[1].kwargs["storage_reader"]).name, "optimizer")
        self.assertFalse(calls[1].kwargs["planner"].allow_partial_load)
        self.assertEqual(result["extra_state"]["global_step"], 19500)
        self.assertEqual(result["extra_state"]["lr_scheduler"], "saved scheduler")
        self.assertEqual(result["extra_state"]["train_dataloader"], "saved dataloader")


if __name__ == "__main__":
    unittest.main()
