"""CPU-only regressions for the optional deadline continuation path."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from lingbotvla.utils.checkpoint_retention import prune_old_checkpoints, update_best_checkpoint_record
from lingbotvla.utils.continuation_schedule import build_continuation_scheduler, lr_multiplier, validate_continuation_plan
from tools.clean_training_common import REPO_ROOT
from tools.launch_clean_continue import prepare_command


class ScheduleTests(unittest.TestCase):
    def optimizer_and_schedule(self):
        params = [torch.nn.Parameter(torch.tensor([0.1])), torch.nn.Parameter(torch.tensor([0.2]))]
        optimizer = torch.optim.AdamW([{"params": [params[0]], "lr": 1e-5},
                                       {"params": [params[1]], "lr": 1e-5 * 2.8284271247461903}])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.359)
        for param in params:
            param.grad = torch.ones_like(param)
        optimizer.step()
        optimizer.zero_grad()
        return params, optimizer, scheduler

    def build(self, optimizer, scheduler, step=19500, **changes):
        options = dict(global_step=step, start_step=19500, end_step=48000,
                       warmup_steps=200, peak_lr=1e-5, min_lr=5e-6, original_base_lr=1e-5)
        options.update(changes)
        return build_continuation_scheduler(optimizer, scheduler, **options)

    def test_first_lr_is_exact_saved_value_and_buffers_rng_weights_do_not_change(self):
        params, optimizer, restored = self.optimizer_and_schedule()
        before = deepcopy(optimizer.state_dict())
        weights = [param.detach().clone() for param in params]
        rng = torch.get_rng_state().clone()
        scheduler = self.build(optimizer, restored)
        self.assertEqual(scheduler.last_epoch, 19500)
        for old, group in zip(before["param_groups"], optimizer.param_groups):
            self.assertAlmostEqual(old["lr"], group["lr"], places=16)
        for index, param in enumerate(params):
            torch.testing.assert_close(param, weights[index], rtol=0, atol=0)
            for key, value in before["state"][index].items():
                torch.testing.assert_close(optimizer.state[param][key], value, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_warmup_peak_floor_and_expert_multiplier(self):
        _, optimizer, restored = self.optimizer_and_schedule()
        scheduler = self.build(optimizer, restored)
        plan = scheduler.clean_continuation_plan
        self.assertAlmostEqual(lr_multiplier(19500, plan, 0), 0.359)
        self.assertAlmostEqual(lr_multiplier(19600, plan, 0), (0.359 + 1) / 2)
        self.assertEqual(lr_multiplier(19700, plan, 0), 1)
        self.assertEqual(lr_multiplier(48000, plan, 0), 0.5)
        self.assertEqual(lr_multiplier(80000, plan, 0), 0.5)
        self.assertAlmostEqual(plan["peak_lrs"][1] / plan["peak_lrs"][0], 2.8284271247461903)
        rates = [lr_multiplier(step, plan, 0) for step in range(19700, 48001, 100)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))

    def test_midplan_resume_has_same_next_update_and_does_not_restart_warmup(self):
        params, optimizer, restored = self.optimizer_and_schedule()
        scheduler = self.build(optimizer, restored)
        for _ in range(450):
            for param in params:
                param.grad = torch.ones_like(param)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        saved_optimizer = deepcopy(optimizer.state_dict())
        saved_schedule = deepcopy(scheduler.state_dict())
        copied = [torch.nn.Parameter(param.detach().clone()) for param in params]
        resumed_optimizer = torch.optim.AdamW([{"params": [copied[0]], "lr": 1e-5},
                                                {"params": [copied[1]], "lr": 2.8284271247461903e-5}])
        placeholder = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: 1)
        placeholder.load_state_dict(saved_schedule)
        resumed_optimizer.load_state_dict(saved_optimizer)
        resumed = self.build(resumed_optimizer, placeholder, step=19950)
        self.assertEqual(resumed.last_epoch, scheduler.last_epoch)
        self.assertEqual(resumed.clean_continuation_plan, scheduler.clean_continuation_plan)
        for expected, actual in zip(scheduler.get_last_lr(), resumed.get_last_lr()):
            self.assertAlmostEqual(expected, actual, places=16)
        self.assertGreater(resumed.get_last_lr()[0], 9.9e-6)
        for values, opt, sched in ((params, optimizer, scheduler), (copied, resumed_optimizer, resumed)):
            for param in values:
                param.grad = torch.ones_like(param)
            opt.step()
            sched.step()
        for expected, actual in zip(params, copied):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(scheduler.get_last_lr(), resumed.get_last_lr())

    def test_implicit_midcheckpoint_restart_is_rejected(self):
        _, optimizer, restored = self.optimizer_and_schedule()
        with self.assertRaisesRegex(ValueError, "First LR restart"):
            self.build(optimizer, restored, step=20000)

    def test_existing_plan_cannot_be_silently_changed(self):
        _, optimizer, restored = self.optimizer_and_schedule()
        scheduler = self.build(optimizer, restored)
        for changes in ({"end_step": 50000}, {"warmup_steps": 100}, {"min_lr": 3e-6}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Saved continuation plan differs"):
                self.build(optimizer, scheduler, **changes)

    def test_invalid_plans_rejected(self):
        for values in ((19500, 48000, 0, 1e-5, 5e-6), (19500, 19600, 200, 1e-5, 5e-6),
                       (19500, 48000, 200, 1e-5, 2e-5), (19500, 48000, 200, float("nan"), 5e-6),
                       (-1, 48000, 200, 1e-5, 5e-6)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_continuation_plan(*values)

    def test_plan_survives_real_distributed_checkpoint_roundtrip(self):
        from lingbotvla.checkpoint.checkpointer import DistributedCheckpointer
        params, optimizer, previous = self.optimizer_and_schedule()
        scheduler = self.build(optimizer, previous)
        model = torch.nn.ParameterList(params)
        extra = {"global_step": 19500, "lr_scheduler": scheduler.state_dict()}
        with TemporaryDirectory() as tmp, patch("lingbotvla.checkpoint.checkpointer.dist.get_rank", return_value=0):
            DistributedCheckpointer.save(tmp, {"model": model, "optimizer": optimizer, "extra_state": extra})
            target_params, target, placeholder = self.optimizer_and_schedule()
            loaded = DistributedCheckpointer.load(tmp, {"model": torch.nn.ParameterList(target_params),
                                                        "optimizer": target, "extra_state": {}})
            placeholder.load_state_dict(loaded["extra_state"]["lr_scheduler"])
            resumed = self.build(target, placeholder, step=loaded["extra_state"]["global_step"])
            self.assertEqual(resumed.clean_continuation_plan, scheduler.clean_continuation_plan)
            self.assertEqual(resumed.get_last_lr(), scheduler.get_last_lr())


class LauncherTests(unittest.TestCase):
    def args(self, run):
        return SimpleNamespace(resume_run=str(run), resume_step=19500, until=48000,
            peak_lr=1e-5, min_lr=5e-6, warmup_steps=200, seconds_per_step=34,
            gpus="0,1,2,3", master_port=62500, dry_run=True)

    def fixture(self, root):
        run = root / "run"
        run.mkdir()
        config = {"train": {"lr": 1e-5, "global_batch_size": 32,
                            "micro_batch_size": 1, "gradient_accumulation_steps": 8,
                            "max_steps": 30000, "output_dir": str(run)}, "data": {}}
        path = run / "lingbotvla_cli.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        cp = run / "checkpoints/global_step_19500"
        command = ["bash", "train.sh", "tasks/vla/train_lingbotvla.py", str(path),
                   "--train.load_checkpoint_path", str(cp)]
        return run, path, config, command

    def test_explicit_plan_preserves_source_configuration_and_no_batch_loss_or_precision_overrides(self):
        with TemporaryDirectory() as tmp:
            run, path, _, command = self.fixture(Path(tmp))
            original = path.read_bytes()
            with patch("tools.launch_clean_continue.prepare_stage1", return_value=(command, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})) as base, \
                 patch("tools.launch_clean_continue.validate_training_checkpoint") as validate:
                actual, _ = prepare_command(self.args(run))
            base.assert_called_once()
            self.assertEqual(path.read_bytes(), original)
            validate.assert_called_once_with(run / "checkpoints/global_step_19500", 4)
            self.assertEqual(actual[actual.index("--train.continuation_start_step") + 1], "19500")
            self.assertEqual(actual[actual.index("--train.max_steps") + 1], "48000")
            self.assertEqual(actual[actual.index("--train.max_checkpoints_to_keep") + 1], "3")
            for field in ("lr", "global_batch_size", "enable_fp32", "enable_mixed_precision", "vla_loss_type"):
                self.assertNotIn(f"--train.{field}", actual)

    def test_resume_existing_plan_keeps_its_origin(self):
        with TemporaryDirectory() as tmp:
            run, path, config, command = self.fixture(Path(tmp))
            config["train"].update(continuation_start_step=19500, continuation_warmup_steps=200,
                                   continuation_peak_lr=1e-5, continuation_min_lr=5e-6,
                                   max_steps=48000, checkpoint_pinned_step=19500)
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            command[-1] = str(run / "checkpoints/global_step_25000")
            args = self.args(run)
            args.resume_step = 25000
            args.until = args.peak_lr = args.min_lr = args.warmup_steps = None
            with patch("tools.launch_clean_continue.prepare_stage1", return_value=(command, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})), \
                 patch("tools.launch_clean_continue.validate_training_checkpoint"):
                actual, _ = prepare_command(args)
            self.assertEqual(actual[actual.index("--train.continuation_start_step") + 1], "19500")
            self.assertEqual(actual[actual.index("--train.load_checkpoint_path") + 1], str(run / "checkpoints/global_step_25000"))

    def test_batch_mismatch_fails_without_modifying_saved_config(self):
        with TemporaryDirectory() as tmp:
            run, path, _, command = self.fixture(Path(tmp))
            before = path.read_bytes()
            with patch("tools.launch_clean_continue.prepare_stage1", return_value=(command, {"CUDA_VISIBLE_DEVICES": "0,1"})), \
                 self.assertRaisesRegex(ValueError, "batch/accumulation"):
                prepare_command(self.args(run))
            self.assertEqual(path.read_bytes(), before)

    def test_entrypoint_is_opt_in_and_normal_resume_remains_default(self):
        text = (REPO_ROOT / "lingbotvla/utils/arguments.py").read_text(encoding="utf-8")
        self.assertIn("continuation_start_step: Optional[int] = field(\n        default=None", text)
        source = (REPO_ROOT / "tasks/vla/train_lingbotvla.py").read_text(encoding="utf-8")
        self.assertLess(source.index("Checkpointer.load(cp, state"), source.index("lr_scheduler = build_continuation_scheduler("))
        self.assertIn("if args.train.continuation_start_step is not None:", source)


class RetentionTests(unittest.TestCase):
    def test_baseline_best_and_latest_fit_three_slots_and_analysis_is_not_deleted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            root.mkdir()
            for step in (19500, 20000, 20500, 21000, 21500):
                (root / f"global_step_{step}").mkdir()
            analysis = Path(tmp) / "visualizations/global_step_20000"
            analysis.mkdir(parents=True)
            update_best_checkpoint_record(root, step=20500, metric_value=0.02,
                                          window_start_step=20001, window_end_step=20500)
            prune_old_checkpoints(root, 3, preferred_paths=[root / "global_step_19500", root / "global_step_20500"])
            kept = {item.name for item in root.iterdir() if item.is_dir()}
            self.assertEqual(kept, {"global_step_19500", "global_step_20500", "global_step_21500"})
            self.assertTrue(analysis.exists())


if __name__ == "__main__":
    unittest.main()
