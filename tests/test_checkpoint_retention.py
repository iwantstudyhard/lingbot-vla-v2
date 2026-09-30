from pathlib import Path

import pytest

from lingbotvla.utils.checkpoint_retention import (
    best_checkpoint_path,
    load_best_checkpoint_record,
    prune_old_checkpoints,
    update_best_checkpoint_record,
)


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


def test_retention_keeps_best_and_two_newest(tmp_path: Path):
    _make_checkpoints(tmp_path, 500, 1000, 1500, 2000)
    best = tmp_path / "global_step_500"

    removed = prune_old_checkpoints(tmp_path, 3, preferred_paths=[best])

    assert [path.name for path in removed] == ["global_step_1000"]
    assert best.is_dir()
    assert (tmp_path / "global_step_1500").is_dir()
    assert (tmp_path / "global_step_2000").is_dir()
    assert len(list(tmp_path.glob("global_step_*"))) == 3


def test_best_checkpoint_record_is_lower_is_better_and_atomic(tmp_path: Path):
    _make_checkpoints(tmp_path, 500, 1000)

    first, changed = update_best_checkpoint_record(
        tmp_path,
        step=500,
        metric_value=0.2,
        window_start_step=1,
        window_end_step=500,
    )
    assert changed
    assert first["step"] == 500
    assert best_checkpoint_path(tmp_path) == (tmp_path / "global_step_500").resolve()

    unchanged, changed = update_best_checkpoint_record(
        tmp_path,
        step=1000,
        metric_value=0.3,
        window_start_step=501,
        window_end_step=1000,
    )
    assert not changed
    assert unchanged["step"] == 500

    improved, changed = update_best_checkpoint_record(
        tmp_path,
        step=1000,
        metric_value=0.1,
        window_start_step=501,
        window_end_step=1000,
    )
    assert changed
    assert improved["step"] == 1000
    assert load_best_checkpoint_record(tmp_path)["metric_value"] == pytest.approx(0.1)


def test_zero_disables_cleanup_and_negative_is_rejected(tmp_path: Path):
    _make_checkpoints(tmp_path, 2000, 4000, 6000)

    assert prune_old_checkpoints(tmp_path, 0) == []
    assert len(list(tmp_path.glob("global_step_*"))) == 3
    with pytest.raises(ValueError):
        prune_old_checkpoints(tmp_path, -1)
