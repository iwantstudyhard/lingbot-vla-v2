"""Saved training configurations must remain readable when resuming."""

from dataclasses import asdict, dataclass, field
from typing import List

import pytest
import yaml

from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args


@dataclass
class ModelRoot:
    model: ModelArguments = field(default_factory=ModelArguments)


@dataclass
class TrainingRoot:
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)


@dataclass
class ListOptions:
    basic_modules: List[str] = field(default_factory=lambda: ["default_module"])


@dataclass
class ListRoot:
    model: ListOptions = field(default_factory=ListOptions)


def test_saved_model_configuration_round_trips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    original = ModelRoot(model=ModelArguments(model_path=str(tmp_path / "weights")))
    save_args(original, str(tmp_path))
    monkeypatch.setattr("sys.argv", ["train.py", str(tmp_path / "lingbotvla_cli.yaml")])

    restored = parse_args(ModelRoot)

    assert asdict(restored) == asdict(original)


@pytest.mark.parametrize("modules", [[], ["decoder", "vision"]])
def test_saved_lists_preserve_explicit_values(tmp_path, monkeypatch, modules):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": {"basic_modules": modules}}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["train.py", str(config)])

    assert parse_args(ListRoot).model.basic_modules == modules


def test_cli_list_overrides_saved_empty_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  basic_modules: []\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["train.py", str(config), "--model.basic_modules", "decoder"])

    assert parse_args(ListRoot).model.basic_modules == ["decoder"]


def test_saved_training_configuration_resumes_with_cli_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LINGBOT_TRAIN_RUN_ID", "resume-test")
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("QWEN3VL_DIR", raising=False)
    monkeypatch.delenv("QWEN3VL_PATH", raising=False)
    for name, value in {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "4"}.items():
        monkeypatch.setenv(name, value)
    original = TrainingRoot(
        model=ModelArguments(model_path=str(tmp_path / "weights"), moe_implementation="fused"),
        data=DataArguments(train_path=str(tmp_path / "data.txt")),
        train=TrainingArguments(
            output_dir=str(tmp_path / "run"), max_steps=18000,
            micro_batch_size=1, gradient_accumulation_steps=8, global_batch_size=32,
            optimizer="muon", ckpt_manager="dcp",
        ),
    )
    save_args(original, original.train.output_dir)
    monkeypatch.setattr("sys.argv", [
        "train.py", str(tmp_path / "run/lingbotvla_cli.yaml"),
        "--train.enable_resume", "true", "--train.max_steps", "30000",
    ])

    restored = parse_args(TrainingRoot)

    assert asdict(restored.model) == asdict(original.model)
    assert restored.data.train_path == original.data.train_path
    assert restored.train.enable_resume is True
    assert restored.train.max_steps == 30000
    assert restored.train.load_checkpoint_path is None
    assert restored.train.wandb_name is None
    assert restored.train.muon_exclude_name_patterns == []
    assert restored.train.gradient_accumulation_steps == 8
    assert restored.train.global_batch_size == 32
