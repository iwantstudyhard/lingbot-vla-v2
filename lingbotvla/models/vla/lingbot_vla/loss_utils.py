# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn.functional as F


def _match_batch_dimension(mask: torch.Tensor, target_batch: int, name: str) -> torch.Tensor:
    """Repeat a per-sample mask when a loss variant repeats the batch."""
    if mask.shape[0] == target_batch:
        return mask
    if mask.shape[0] == 0 or target_batch % mask.shape[0] != 0:
        raise ValueError(
            f"{name} batch dimension {mask.shape[0]} cannot match loss batch dimension {target_batch}."
        )
    return mask.repeat(target_batch // mask.shape[0], *([1] * (mask.ndim - 1)))


def _build_action_loss_mask(
    losses: torch.Tensor,
    *,
    joint_mask: torch.Tensor | None,
    action_is_pad: torch.Tensor | None,
    action_dim: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build the valid ``[B, T, D]`` mask shared by action losses."""
    if losses.ndim != 3:
        raise ValueError(f"losses must have shape [B, T, D], got {tuple(losses.shape)}")

    if joint_mask is None:
        if action_dim <= 0 or action_dim > losses.shape[-1]:
            raise ValueError(
                f"action_dim must be in [1, {losses.shape[-1]}], got {action_dim}."
            )
        loss_mask = torch.ones_like(losses[:, :, :action_dim], dtype=torch.bool)
    else:
        if joint_mask.ndim != 3:
            raise ValueError(
                f"joint_mask must have shape [B, T, D], got {tuple(joint_mask.shape)}"
            )
        loss_mask = _match_batch_dimension(joint_mask.bool(), losses.shape[0], "joint_mask")
        if loss_mask.shape[1:] != losses.shape[1:]:
            raise ValueError(
                f"joint_mask shape {tuple(loss_mask.shape)} does not match losses {tuple(losses.shape)}."
            )

    metrics: dict[str, torch.Tensor] = {}
    if action_is_pad is not None:
        if action_is_pad.ndim != 2:
            raise ValueError(
                f"action_is_pad must have shape [B, T], got {tuple(action_is_pad.shape)}"
            )
        action_is_pad = _match_batch_dimension(
            action_is_pad.bool(), losses.shape[0], "action_is_pad"
        )
        if action_is_pad.shape[1] != losses.shape[1]:
            raise ValueError(
                f"action_is_pad horizon {action_is_pad.shape[1]} does not match "
                f"loss horizon {losses.shape[1]}."
            )
        valid_timestep_mask = ~action_is_pad
        loss_mask = loss_mask & valid_timestep_mask.unsqueeze(-1)
        metrics["action/padding_ratio"] = action_is_pad.float().mean().detach()
        metrics["action/valid_timestep_ratio"] = valid_timestep_mask.float().mean().detach()
    else:
        metrics["action/padding_ratio"] = losses.new_zeros(())
        metrics["action/valid_timestep_ratio"] = losses.new_ones(())
    return loss_mask, metrics


def compute_motion_frame_weights(
    actions: torch.Tensor,
    *,
    joint_mask: torch.Tensor | None,
    action_is_pad: torch.Tensor | None,
    alpha: float,
    max_weight: float,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Upweight action frames with visible expert motion.

    The score is the mean absolute action delta from the previous valid frame.
    Scores are normalized by each sample's valid-frame mean, then converted to
    weights ``1 + alpha * normalized_score`` and capped by ``max_weight``.
    Static frames remain in the objective with weight 1.
    """
    if alpha <= 0:
        return None, {
            "action/motion_weight_mean": actions.new_ones(()),
            "action/motion_weight_max": actions.new_ones(()),
            "action/motion_frame_ratio": actions.new_zeros(()),
        }
    if max_weight < 1:
        raise ValueError(f"motion frame max weight must be >= 1, got {max_weight}")
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, T, D], got {tuple(actions.shape)}")

    valid_mask, _ = _build_action_loss_mask(
        actions,
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
        action_dim=actions.shape[-1],
    )
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    deltas = (actions[:, 1:] - actions[:, :-1]).abs()
    pair_counts = pair_mask.sum(dim=-1).clamp(min=1)
    pair_scores = torch.where(pair_mask, deltas, torch.zeros_like(deltas)).sum(dim=-1) / pair_counts

    scores = actions.new_zeros(actions.shape[:2])
    scores[:, 1:] = pair_scores
    valid_time = valid_mask.any(dim=-1)
    score_sum = torch.where(valid_time, scores, torch.zeros_like(scores)).sum(dim=-1, keepdim=True)
    score_count = valid_time.sum(dim=-1, keepdim=True).clamp(min=1)
    score_scale = (score_sum / score_count).clamp(min=1e-6)
    normalized_scores = scores / score_scale
    weights = (1.0 + alpha * normalized_scores).clamp(max=max_weight)
    weights = torch.where(valid_time, weights, torch.ones_like(weights))

    valid_weights = weights[valid_time]
    motion_frames = (scores > 1e-6) & valid_time
    return weights, {
        "action/motion_weight_mean": valid_weights.mean().detach() if valid_weights.numel() else actions.new_ones(()),
        "action/motion_weight_max": valid_weights.max().detach() if valid_weights.numel() else actions.new_ones(()),
        "action/motion_frame_ratio": (
            motion_frames.sum().float() / valid_time.sum().clamp(min=1).float()
        ).detach(),
    }


def _masked_temporal_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean L1 over a temporal action mask, safely excluding invalid slots."""
    losses = F.l1_loss(prediction, target, reduction="none")
    masked_losses = torch.where(mask, losses, torch.zeros_like(losses))
    return masked_losses.sum() / mask.sum().clamp(min=1)


def compute_action_smoothness_losses(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    joint_mask: torch.Tensor | None,
    action_is_pad: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first- and second-difference L1 losses on action trajectories."""
    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            "predicted_actions and target_actions must have the same shape, got "
            f"{tuple(predicted_actions.shape)} and {tuple(target_actions.shape)}"
        )
    if predicted_actions.ndim != 3:
        raise ValueError(
            f"action trajectories must have shape [B, T, D], got {tuple(predicted_actions.shape)}"
        )

    base_mask, _ = _build_action_loss_mask(
        predicted_actions,
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
        action_dim=predicted_actions.shape[-1],
    )
    if predicted_actions.shape[1] < 2:
        zero = predicted_actions.new_zeros(())
        return zero, zero

    pair_mask = base_mask[:, 1:] & base_mask[:, :-1]
    predicted_delta = predicted_actions[:, 1:] - predicted_actions[:, :-1]
    target_delta = target_actions[:, 1:] - target_actions[:, :-1]
    velocity_loss = _masked_temporal_l1(predicted_delta, target_delta, pair_mask)

    if predicted_actions.shape[1] < 3:
        acceleration_loss = predicted_actions.new_zeros(())
    else:
        triple_mask = base_mask[:, 2:] & base_mask[:, 1:-1] & base_mask[:, :-2]
        predicted_acceleration = (
            predicted_actions[:, 2:] - 2 * predicted_actions[:, 1:-1] + predicted_actions[:, :-2]
        )
        target_acceleration = (
            target_actions[:, 2:] - 2 * target_actions[:, 1:-1] + target_actions[:, :-2]
        )
        acceleration_loss = _masked_temporal_l1(
            predicted_acceleration,
            target_acceleration,
            triple_mask,
        )
    return velocity_loss, acceleration_loss


def reduce_action_losses(
    losses: torch.Tensor,
    *,
    joint_mask: torch.Tensor | None,
    action_is_pad: torch.Tensor | None,
    action_dim: int,
    element_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce per-action losses while excluding padding and applying optional weights."""
    if joint_mask is None:
        if action_dim <= 0 or action_dim > losses.shape[-1]:
            raise ValueError(
                f"action_dim must be in [1, {losses.shape[-1]}], got {action_dim}."
            )
        losses = losses[:, :, :action_dim]

    loss_mask, metrics = _build_action_loss_mask(
        losses,
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
        action_dim=action_dim,
    )
    if element_weights is None:
        weights = torch.ones_like(losses)
    else:
        if element_weights.ndim == 2:
            weights = element_weights.unsqueeze(-1).expand_as(losses)
        elif element_weights.ndim == 3:
            weights = element_weights
        else:
            raise ValueError(
                "element_weights must have shape [B, T] or [B, T, D], "
                f"got {tuple(element_weights.shape)}"
            )
        if weights.shape != losses.shape:
            raise ValueError(
                f"element_weights shape {tuple(weights.shape)} does not match losses {tuple(losses.shape)}"
            )
        weights = weights.to(dtype=losses.dtype).clamp_min(0)

    # ``where`` prevents a non-finite value in a masked padding slot from
    # contaminating the reduction (NaN * 0 is still NaN).
    masked_losses = torch.where(loss_mask, losses, torch.zeros_like(losses))
    effective_weights = torch.where(loss_mask, weights, torch.zeros_like(weights))
    weighted_losses = masked_losses * effective_weights
    weighted_counts = effective_weights.sum(dim=(1, 2)).clamp(min=1)
    batch_mean_losses = weighted_losses.sum(dim=(1, 2)) / weighted_counts
    loss_vla = weighted_losses.sum() / effective_weights.sum().clamp(min=1)
    metrics["action/effective_weight_mean"] = effective_weights.sum().detach() / loss_mask.sum().clamp(min=1)
    return loss_vla, batch_mean_losses, metrics
