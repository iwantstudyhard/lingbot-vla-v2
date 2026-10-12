"""Regressions for parsing complete saved YAML, not just launcher argv."""

import ast
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass, field
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Literal, Optional
import unittest
from unittest.mock import patch

import yaml

from lingbotvla.utils.arguments import DataArguments, EvalArguments, ModelArguments, TrainingArguments, parse_args, save_args


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Options:
    labels: List[str] = field(default_factory=lambda: ["original"])
    optional_labels: Optional[List[str]] = field(default_factory=list)
    optional_count: Optional[int] = 7
    optional_rate: Optional[float] = 0.1
    optional_text: Optional[str] = "original"
    optional_flag: Optional[bool] = True
    optional_dict: Optional[Dict[str, Any]] = None
    text: str = "original"


@dataclass
class Root:
    options: Options = field(default_factory=Options)


class SavedArgumentTests(unittest.TestCase):
    def parse(self, config, overrides=(), suffix="yaml"):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / f"saved.{suffix}"
            source.write_text(json.dumps(config) if suffix == "json" else yaml.safe_dump(config), encoding="utf-8")
            cwd = Path.cwd()
            try:
                with patch.dict(os.environ, {"WORKSPACE": str(root)}, clear=True), \
                     patch.object(sys, "argv", ["fixture.py", str(source), *overrides]), \
                     redirect_stderr(io.StringIO()):
                    return parse_args(Root)
            finally:
                os.chdir(cwd)

    def test_empty_saved_list_clears_nonempty_default(self):
        args = self.parse({"options": {"labels": [], "optional_labels": []}})
        self.assertEqual(args.options.labels, [])
        self.assertEqual(args.options.optional_labels, [])

    def test_null_optionals_are_none_not_defaults_or_literal_null_strings(self):
        values = {key: None for key in ("optional_labels", "optional_count", "optional_rate",
                                       "optional_text", "optional_flag", "optional_dict")}
        args = self.parse({"options": values})
        for key in values:
            with self.subTest(key=key):
                self.assertIsNone(getattr(args.options, key))

    def test_cli_overrides_empty_and_null_yaml(self):
        args = self.parse({"options": {"labels": [], "optional_count": None,
                                       "optional_flag": None, "optional_text": None}},
                          ("--options.labels", "arm", "gripper", "--options.optional_count", "42",
                           "--options.optional_flag", "false", "--options.optional_text=override"))
        self.assertEqual(args.options.labels, ["arm", "gripper"])
        self.assertEqual(args.options.optional_count, 42)
        self.assertFalse(args.options.optional_flag)
        self.assertEqual(args.options.optional_text, "override")

    def test_populated_list_and_dict_are_unchanged(self):
        args = self.parse({"options": {"labels": ["arm", "gripper"],
                                       "optional_dict": {"loss_weight": 0.004, "enabled": True}}})
        self.assertEqual(args.options.labels, ["arm", "gripper"])
        self.assertEqual(args.options.optional_dict, {"loss_weight": 0.004, "enabled": True})

    def test_json_empty_and_null_values(self):
        args = self.parse({"options": {"labels": [], "optional_count": None}}, suffix="json")
        self.assertEqual(args.options.labels, [])
        self.assertIsNone(args.options.optional_count)

    def test_nonnullable_null_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "options.text.*null"):
            self.parse({"options": {"text": None}})

    def test_unknown_null_argument_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            self.parse({"options": {"typo": None}})

    def test_explicit_cli_list_still_requires_items(self):
        with self.assertRaises(SystemExit):
            self.parse({}, ("--options.labels",))

    def test_actual_save_args_roundtrip_preserves_empty_and_null(self):
        expected = Root(Options(labels=[], optional_count=None, optional_rate=None, optional_text=None,
                                optional_flag=None, optional_dict=None))
        with TemporaryDirectory() as tmp:
            save_args(expected, tmp)
            saved = yaml.safe_load((Path(tmp) / "lingbotvla_cli.yaml").read_text(encoding="utf-8"))
        actual = self.parse(saved)
        self.assertEqual(asdict(actual), asdict(expected))


class RealTrainingSchemaTests(unittest.TestCase):
    def setUp(self):
        # Use the exact production dataclass definitions and real parse_args,
        # without importing vision models or initializing any CUDA processes.
        tree = ast.parse((REPO_ROOT / "tasks/vla/train_lingbotvla.py").read_text(encoding="utf-8"))
        names = {"MyTrainingArguments", "MyDataArguments", "Arguments"}
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        self.module = ModuleType("saved_training_schema_fixture")
        self.module.__dict__.update(dataclass=dataclass, field=field, Optional=Optional, List=List,
                                   Literal=Literal, Dict=Dict, Any=Any, TrainingArguments=TrainingArguments,
                                   DataArguments=DataArguments, ModelArguments=ModelArguments, EvalArguments=EvalArguments)
        self.registration = patch.dict(sys.modules, {self.module.__name__: self.module})
        self.registration.start()
        self.addCleanup(self.registration.stop)
        exec(compile(tree, "train_lingbotvla.dataclasses", "exec"), self.module.__dict__)
        self.module.trainer = SimpleNamespace(MyDataArguments=self.module.MyDataArguments,
                                             Arguments=self.module.Arguments)
        stage2 = ast.parse((REPO_ROOT / "extensions/clean_stage2/train.py").read_text(encoding="utf-8"))
        stage2.body = [node for node in stage2.body if isinstance(node, ast.ClassDef)
                       and node.name in {"Stage2DataArguments", "Stage2Arguments"}]
        exec(compile(stage2, "clean_stage2.dataclasses", "exec"), self.module.__dict__)

    def parse(self, config, overrides=(), save_and_reload=False, schema=None):
        schema = schema or self.module.Arguments
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "saved.yaml"
            config = json.loads(json.dumps(config))
            config["train"]["output_dir"] = str(root / "run")
            source.write_text(yaml.safe_dump(config), encoding="utf-8")
            train = config["train"]
            world_size = train["global_batch_size"] // (train["micro_batch_size"] * train["gradient_accumulation_steps"])
            env = {"WORKSPACE": str(root), "LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": str(world_size),
                   "LINGBOT_TRAIN_RUN_ID": "argument_roundtrip_fixture"}
            cwd = Path.cwd()
            try:
                with patch.dict(os.environ, env, clear=True), \
                     patch.object(sys, "argv", ["train_lingbotvla.py", str(source), *overrides]), \
                     redirect_stderr(io.StringIO()):
                    args = parse_args(schema)
                    if save_and_reload:
                        save_args(args, str(root / "run"))
                        sys.argv = ["train_lingbotvla.py", str(root / "run/lingbotvla_cli.yaml")]
                        restored = parse_args(schema)
                        self.assertEqual(asdict(restored), asdict(args))
                    return args
            finally:
                os.chdir(cwd)

    def stage1_config(self):
        return yaml.safe_load((REPO_ROOT / "configs/vla/robotwin/robotwin_clean_stage1.yaml").read_text(encoding="utf-8"))

    def test_real_fresh_stage1_full_saved_configuration_roundtrip(self):
        args = self.parse(self.stage1_config(), save_and_reload=True)
        self.assertEqual(args.model.basic_modules, [])
        self.assertIsNone(args.model.vlm_repo_id)
        self.assertIsNone(args.train.continuation_start_step)
        self.assertIsNone(args.train.load_checkpoint_path)
        self.assertFalse(args.train.freeze_vision_encoder)

    def test_real_19500_continuation_cli_overrides_on_complete_saved_config(self):
        config = self.stage1_config()
        config["model"].update(basic_modules=[], vlm_repo_id=None)
        config["train"].update(gradient_accumulation_steps=8, max_steps=30000,
                               muon_exclude_name_patterns=[], load_checkpoint_path=None,
                               continuation_start_step=None, checkpoint_pinned_step=None)
        overrides = ("--train.enable_resume", "true", "--train.load_checkpoint_path", "/fixture/global_step_19500",
                     "--train.max_steps", "48000", "--train.num_train_epochs", "999999999",
                     "--train.continuation_start_step", "19500", "--train.continuation_warmup_steps", "200",
                     "--train.continuation_peak_lr", "1e-5", "--train.continuation_min_lr", "5e-6",
                     "--train.checkpoint_pinned_step", "19500", "--train.max_checkpoints_to_keep", "3")
        args = self.parse(config, overrides, save_and_reload=True)
        self.assertEqual(args.train.continuation_start_step, 19500)
        self.assertEqual(args.train.max_steps, 48000)
        self.assertEqual(args.train.global_batch_size, 32)
        self.assertEqual(args.train.gradient_accumulation_steps, 8)
        self.assertEqual(args.train.continuation_peak_lr, 1e-5)
        self.assertEqual(args.train.continuation_min_lr, 5e-6)
        self.assertFalse(args.train.freeze_vision_encoder)
        self.assertFalse(args.train.enable_fp32)
        self.assertEqual(args.model.basic_modules, [])
        self.assertIsNone(args.model.vlm_repo_id)

    def test_real_stage2_full_saved_configuration_roundtrip_stays_separate(self):
        config = yaml.safe_load((REPO_ROOT / "extensions/clean_stage2/config.yaml").read_text(encoding="utf-8"))
        args = self.parse(config, save_and_reload=True, schema=self.module.Stage2Arguments)
        self.assertFalse(args.train.enable_resume)
        self.assertIsNone(args.train.continuation_start_step)
        self.assertIsNone(args.train.load_checkpoint_path)
        self.assertEqual(args.train.global_batch_size, 32)
        self.assertTrue(args.data.stage2_augmentation_config.endswith("augmentation.json"))


if __name__ == "__main__":
    unittest.main()
