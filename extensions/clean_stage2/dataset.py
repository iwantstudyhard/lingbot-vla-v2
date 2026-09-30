"""Stage-two-only dataset classes. Stage one never imports this module."""

import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import Resize

from lingbotvla.data.vla_data.base_dataset import (
    LEROBOT_DATASET_API, LeRobotDataset, LeRobotDatasetMetadata, VLADataset,
    _filter_supported_kwargs, _resolve_lerobot_location,
)
from lingbotvla.data.vla_data.multi_vla_dataset import MultiVLADataset, get_all_tasks
from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.data.vla_data.video_utils import decode_video_frames

from .augmentation import augment_views, load_settings
from lingbotvla.utils.episode_boundaries import bounded_timestamps


class BoundedLeRobotDataset(LeRobotDataset):
    def _query_videos(self, query_timestamps, ep_idx):
        if LEROBOT_DATASET_API != "v3":
            raise ValueError("Stage2 boundary protection requires LeRobot v3")
        episode = self.meta.episodes[ep_idx]
        item = {}
        flags = []
        for key, relative in query_timestamps.items():
            times, clamped = bounded_timestamps(
                relative, float(episode[f"videos/{key}/from_timestamp"]),
                float(episode[f"videos/{key}/to_timestamp"]), float(self.meta.fps),
            )
            frames = decode_video_frames(
                self.root / self.meta.get_video_file_path(ep_idx, key),
                times, self.tolerance_s, self.video_backend,
            )
            item[key] = frames.squeeze(0)
            flags.append(clamped)
        # Diagnostic flags, not an action mask or a new auxiliary-loss mask.
        item["stage2_video_clamped"] = torch.tensor(flags, dtype=torch.bool)
        return item


class Stage2FeatureTransform(FeatureTransform):
    def __init__(self, *args, settings, progress, **kwargs):
        super().__init__(*args, image_augment=False, **kwargs)
        self.settings = settings
        self.progress = progress
        self._context = None
        self._preview = None

    def apply(self, item, policy_eval=False):
        self._context = None
        self._preview = None
        if not policy_eval and not self.disabled_image_features:
            self._context = {
                "episode": int(item["episode_index"]),
                "frame": int(item["frame_index"]),
                "task": str(item["task"]),
                "video_clamped": item.get("stage2_video_clamped", torch.zeros(3, 2, dtype=torch.bool)).tolist(),
            }
        result = super().apply(item, policy_eval=policy_eval)
        if self._preview is not None and not self.return_item_befor_padding:
            preview = self._preview
            if self.use_depth_align and self.settings.get("teacher_clean", True):
                result["pil_images"] = preview["raw"][:, 0]
                if self.use_future_image:
                    result["future_pil_images"] = preview["raw"][:, 1]
            result["stage2_preview_raw"] = preview["raw"]
            result["stage2_preview_aug"] = preview["aug"]
            result["stage2_preview_mask"] = preview["mask"]
            result["stage2_preview_json"] = json.dumps(preview["metadata"], ensure_ascii=False)
        self._context = None
        self._preview = None
        return result

    def pad_and_concat(self, item, w_action=True):
        batch = super().pad_and_concat(item, w_action=w_action)
        if self._context is None or not batch["image"]:
            return batch
        keys = [key for key in self.feature_config.images if key in batch["image"]]
        original = {key: batch["image"][key].cpu().numpy().copy() for key in keys}
        future = {key: batch["future_image"][key].cpu().numpy().copy() for key in keys
                  if key in batch["future_image"]}
        seed = int(torch.randint(0, 2**31 - 1, ()).item())
        settings = dict(self.settings)
        settings["strength"] = min(1.0, max(0.0, int(self.progress.item()) / 500))
        augmented, augmented_future, masks, plan = augment_views(
            original, future, settings, self._context["episode"], seed,
        )
        batch["image"] = {key: torch.from_numpy(value) for key, value in augmented.items()}
        batch["future_image"] = {key: torch.from_numpy(value) for key, value in augmented_future.items()}
        raw_pairs, augmented_pairs = [], []
        for key in keys:
            raw_pairs.append(np.stack([original[key], future.get(key, original[key])]))
            augmented_pairs.append(np.stack([augmented[key], augmented_future.get(key, augmented[key])]))
        self._preview = {
            "raw": torch.from_numpy(np.stack(raw_pairs)),
            "aug": torch.from_numpy(np.stack(augmented_pairs)),
            "mask": torch.from_numpy(np.stack([masks[key] for key in keys])),
            "metadata": {**self._context, **plan, "cameras": keys},
        }
        return batch


class Stage2VLADataset(VLADataset):
    """Reuse stage-one normalization/padding but explicitly choose new readers."""

    def __init__(self, repo_id, data_name, dataset_config, config, processor,
                 use_depth_align, settings, progress):
        self.processor, self.config = processor, config
        self.chunk_size = dataset_config.chunk_size
        self.data_name = data_name
        self.disabled_image_features = False
        self.use_depth_align = use_depth_align
        self.use_future_image = dataset_config.use_future_image
        self.return_item = False
        self.transform = None
        self.feature_transform = Stage2FeatureTransform(
            str(Path(dataset_config.robot_config_root) / f"{data_name}.yaml"),
            dataset_config, config, processor, chunk_size=self.chunk_size,
            use_depth_align=use_depth_align, use_future_image=self.use_future_image,
            norm_stats_path=dataset_config.norm_stats_file,
            settings=settings, progress=progress,
        )
        self.action_features = self.feature_transform.actions
        self.state_features = self.feature_transform.states
        self.image_features = self.feature_transform.images
        repo_name, root = _resolve_lerobot_location(repo_id)
        if root is None:
            raise ValueError("Stage2 only accepts an explicit local official-clean dataset")
        metadata_kwargs = _filter_supported_kwargs(
            LeRobotDatasetMetadata.__init__, {"repo_id": repo_name, "root": root},
        )
        self.dataset_meta = LeRobotDatasetMetadata(**metadata_kwargs)
        if self.dataset_meta.info["codebase_version"] != "v3.0":
            raise ValueError("Stage2 is validated for LeRobot v3.0 CFR datasets only")
        delta = {**self.get_delta_timestamps(), **self.get_video_delta_timestamps()}
        self.dataset = BoundedLeRobotDataset(
            repo_id=repo_name, root=root, image_transforms=Resize((dataset_config.img_size,) * 2),
            delta_timestamps=delta, load_image=True,
        )


class Stage2MultiDataset(MultiVLADataset):
    def __init__(self, dataset_config, config, processor, use_depth_align, settings, progress):
        self.config, self.processor, self.return_item = config, processor, False
        self.data_names, self.repo_ids = get_all_tasks(dataset_config.train_path)
        if len(self.repo_ids) != 1 or self.data_names != ["robotwin"]:
            raise ValueError("Stage2 requires one explicit robotwin clean-only manifest entry")
        self._datasets = [Stage2VLADataset(
            self.repo_ids[0], "robotwin", dataset_config, config, processor,
            use_depth_align, settings, progress,
        )]
        self.feature_transforms = {"robotwin": self._datasets[0].feature_transform}
        self.dataset_start_index = [0]

    def __getitem__(self, idx):
        # Do not hide boundary/augmentation errors with 200 random substitutions.
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.getdata(idx)


def build_stage2_dataset(*, dataset_config, model_config, config, processor,
                         use_depth_align, settings, progress):
    if dataset_config.data_name != "multi" or dataset_config.image_augment:
        raise ValueError("Stage2 uses its own augmentation; legacy image_augment must stay false")
    return Stage2MultiDataset(dataset_config, config, processor, use_depth_align, settings, progress)
