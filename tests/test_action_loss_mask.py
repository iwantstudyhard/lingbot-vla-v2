import pytest
import torch

from lingbotvla.models.vla.lingbot_vla.loss_utils import reduce_action_losses


def test_episode_tail_padding_is_excluded_from_loss_and_gradient():
    losses = torch.tensor(
        [[[1.0, 2.0], [100.0, 100.0], [100.0, 100.0]]],
        requires_grad=True,
    )
    joint_mask = torch.ones_like(losses, dtype=torch.bool)
    action_is_pad = torch.tensor([[False, True, True]])

    loss, batch_losses, metrics = reduce_action_losses(
        losses,
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
        action_dim=2,
    )

    assert loss.item() == pytest.approx(1.5)
    assert batch_losses.tolist() == pytest.approx([1.5])
    assert metrics["action/padding_ratio"].item() == pytest.approx(2 / 3)
    loss.backward()
    assert losses.grad[0, 0].tolist() == pytest.approx([0.5, 0.5])
    assert torch.count_nonzero(losses.grad[0, 1:]).item() == 0


def test_joint_and_temporal_masks_are_combined():
    losses = torch.tensor([[[1.0, 50.0, 3.0], [100.0, 100.0, 100.0]]])
    joint_mask = torch.tensor([[[True, False, True], [True, False, True]]])
    action_is_pad = torch.tensor([[False, True]])

    loss, batch_losses, _ = reduce_action_losses(
        losses,
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
        action_dim=3,
    )

    assert loss.item() == pytest.approx(2.0)
    assert batch_losses.tolist() == pytest.approx([2.0])


def test_masks_repeat_with_repeat_loss_batch_variants():
    losses = torch.tensor(
        [
            [[1.0], [100.0]],
            [[3.0], [100.0]],
            [[5.0], [100.0]],
            [[7.0], [100.0]],
        ]
    )
    joint_mask = torch.ones(2, 2, 1, dtype=torch.bool)
    action_is_pad = torch.tensor([[False, True], [False, True]])

    loss, batch_losses, _ = reduce_action_losses(
        losses,
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
        action_dim=1,
    )

    assert loss.item() == pytest.approx(4.0)
    assert batch_losses.tolist() == pytest.approx([1.0, 3.0, 5.0, 7.0])
