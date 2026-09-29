from pathlib import Path

import pytest

from lingbotvla.utils.checkpoint_retention import prune_old_checkpoints


def _make_checkpoints(root: Path, *steps: int) -> None:
    for step in steps:
        checkpoint = root / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "sentinel.txt").write_text(str(step), encoding="utf-8")


def test_retention_keeps_the_two_newest(tmp_path: Path):
    _make_checkpoints(tmp_path, 2000, 4000, 6000)

    removed = prune_old_checkpoints(tmp_path, 2)

    assert [path.name for path in removed] == ["global_step_2000"]
    assert not (tmp_path / "global_step_2000").exists()
    assert (tmp_path / "global_step_4000").is_dir()
    assert (tmp_path / "global_step_6000").is_dir()


def test_pending_hf_checkpoint_is_protected_until_later_pass(tmp_path: Path):
    _make_checkpoints(tmp_path, 2000, 4000, 6000)
    oldest = tmp_path / "global_step_2000"

    assert prune_old_checkpoints(tmp_path, 2, protected_paths=[oldest]) == []
    assert oldest.is_dir()
    assert prune_old_checkpoints(tmp_path, 2) == [oldest.resolve()]


def test_zero_disables_cleanup_and_negative_is_rejected(tmp_path: Path):
    _make_checkpoints(tmp_path, 2000, 4000, 6000)

    assert prune_old_checkpoints(tmp_path, 0) == []
    assert len(list(tmp_path.glob("global_step_*"))) == 3
    with pytest.raises(ValueError):
        prune_old_checkpoints(tmp_path, -1)
