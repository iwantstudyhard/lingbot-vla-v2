"""Explicit second-stage entry: weight initialization, fresh optimizer/scheduler."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from tasks.vla import train_lingbotvla as trainer
from extensions.clean_stage2.augmentation import load_settings
from extensions.clean_stage2.dataset import build_stage2_dataset
from extensions.clean_stage2.previews import PreviewRecorder
from lingbotvla.utils.normalization_contract import resolve_inference_normalization, semantic_hash


@dataclass
class Stage2DataArguments(trainer.MyDataArguments):
    stage2_augmentation_config: str = field(default="extensions/clean_stage2/augmentation.json")


@dataclass
class Stage2Arguments(trainer.Arguments):
    data: Stage2DataArguments = field(default_factory=Stage2DataArguments)


def main():
    progress = torch.zeros((), dtype=torch.int64).share_memory_()
    recorder = PreviewRecorder(progress)
    runtime = {}

    def validate(args):
        if args.train.enable_resume or args.train.load_checkpoint_path:
            raise ValueError("New stage2 must initialize HF weights only; optimizer/scheduler resume is forbidden")
        if args.data.datasets_type != "vla" or args.data.dataloader_type != "native":
            raise ValueError("Stage2 requires the native VLA dataloader")
        output = Path(args.train.output_dir).resolve()
        # Validation runs before NCCL init; only rank0 owns directory creation.
        # Other ranks must not reject the marker concurrently written by rank0.
        if args.train.global_rank == 0 and output.exists():
            raise ValueError(f"Refusing to reuse an output directory: {output}")
        weights = Path(args.model.model_path).resolve()
        if not (weights / "config.json").is_file():
            raise ValueError("--model.model_path must be a complete stage-one hf_ckpt directory")
        if output == weights or weights in output.parents or output in weights.parents:
            raise ValueError("Stage2 output must not overlap the source checkpoint")
        if not args.train.enable_training_visualization:
            raise ValueError("Stage2 previews require enable_training_visualization=true")
        if args.data.image_augment:
            raise ValueError("Do not stack legacy image_augment with stage2 augmentation")
        source_manifest = weights.parent.parent.parent / "normalization/manifest.json"
        if not source_manifest.is_file():
            raise ValueError("Stage2 requires an audited stage1 run; old mixed-stat checkpoints are not compatible with this new pipeline")
        normalization_source = resolve_inference_normalization(
            weights, {"data": {"require_normalization_contract": True}})
        source_norm = json.loads(Path(normalization_source).read_bytes())
        configured_norm = json.loads(Path(args.data.norm_stats_file).read_bytes())
        if semantic_hash(source_norm) != semantic_hash(configured_norm):
            raise ValueError("Stage2 must preserve the exact stage1 normalization scales")
        args.data.norm_stats_file = normalization_source
        if not args.data.require_normalization_contract or not args.data.video_episode_boundary:
            raise ValueError("Stage2 requires audited normalization and episode boundaries")
        runtime["settings"] = load_settings(args.data.stage2_augmentation_config)
        if args.train.global_rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            record = {
                "stage": 2, "initial_hf_weights": str(weights), "optimizer_scheduler": "fresh",
                "dataset_manifest": args.data.train_path, "augmentation": runtime["settings"],
                "normalization_source": normalization_source,
                "normalization_count": source_norm.get("count"),
                "normalization_semantic_sha256": semantic_hash(source_norm),
                "first_stage_reader": "new audited clean CFR v3", "second_stage_reader": "episode-bounded CFR v3",
            }
            (output / "stage2_run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def build(**kwargs):
        return build_stage2_dataset(**kwargs, settings=runtime["settings"], progress=progress)

    trainer.main(arguments_class=Stage2Arguments, dataset_builder=build,
                 validate_args=validate, batch_callback=recorder.collect,
                 checkpoint_callback=recorder.snapshot)


if __name__ == "__main__":
    main()
