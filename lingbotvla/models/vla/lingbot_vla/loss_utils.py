# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import torch


def _match_batch_dimension(mask: torch.Tensor, target_batch: int, name: str) -> torch.Tensor:
    """Repeat a per-sample mask when a loss variant repeats the batch."""
    if mask.shape[0] == target_batch:
        return mask
    if mask.shape[0] == 0 or target_batch % mask.shape[0] != 0:
        raise ValueError(
            f"{name} batch dimension {mask.shape[0]} cannot match loss batch dimension {target_batch}."
        )
    return mask.repeat(target_batch // mask.shape[0], *([1] * (mask.ndim - 1)))


def reduce_action_losses(
    losses: torch.Tensor,
    *,
    joint_mask: torch.Tensor | None,
    action_is_pad: torch.Tensor | None,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce per-action losses while excluding joint and episode-tail padding.

    Args:
        losses: Per-element loss with shape ``[batch, horizon, action_dim]``.
        joint_mask: Valid joint dimensions with the same shape as ``losses``
            (or a batch dimension that evenly repeats to match it).
        action_is_pad: Episode-tail padding flags with shape ``[batch, horizon]``.
        action_dim: Number of action dimensions to use when ``joint_mask`` is absent.
    """
    if losses.ndim != 3:
        raise ValueError(f"losses must have shape [B, T, D], got {tuple(losses.shape)}")

    if joint_mask is None:
        if action_dim <= 0 or action_dim > losses.shape[-1]:
            raise ValueError(
                f"action_dim must be in [1, {losses.shape[-1]}], got {action_dim}."
            )
        losses = losses[:, :, :action_dim]
        loss_mask = torch.ones_like(losses, dtype=torch.bool)
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

    # ``where`` prevents a non-finite value in a masked padding slot from
    # contaminating the reduction (NaN * 0 is still NaN).
    masked_losses = torch.where(loss_mask, losses, torch.zeros_like(losses))
    valid_counts = loss_mask.sum(dim=(1, 2)).clamp(min=1)
    batch_mean_losses = masked_losses.sum(dim=(1, 2)) / valid_counts
    loss_vla = masked_losses.sum() / loss_mask.sum().clamp(min=1)
    return loss_vla, batch_mean_losses, metrics
