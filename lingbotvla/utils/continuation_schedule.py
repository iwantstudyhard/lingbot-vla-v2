"""Explicit LR continuation without discarding restored optimizer state.

The plan is stored in the scheduler state, so later resumes keep the original
warmup origin rather than starting another warmup. Disabled by default.
"""

from __future__ import annotations

import math


def validate_continuation_plan(start_step, end_step, warmup_steps, peak_lr, min_lr):
    for name, value in (("start_step", start_step), ("end_step", end_step),
                        ("warmup_steps", warmup_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Continuation {name} must be a non-negative integer")
    if warmup_steps == 0 or end_step <= start_step + warmup_steps:
        raise ValueError("Continuation end must be after the positive warmup interval")
    if not all(math.isfinite(value) and value > 0 for value in (peak_lr, min_lr)):
        raise ValueError("Continuation learning rates must be positive and finite")
    if min_lr > peak_lr:
        raise ValueError("Continuation min_lr cannot exceed peak_lr")


def lr_multiplier(step, plan, group_index):
    """Global optimizer step -> relative LR; endpoint is clamped, not cycled."""
    elapsed = max(0, step - plan["start_step"])
    warmup = plan["warmup_steps"]
    if elapsed < warmup:
        start = plan["start_factors"][group_index]
        return start + (1.0 - start) * elapsed / warmup
    progress = min(1.0, (elapsed - warmup) /
                   (plan["end_step"] - plan["start_step"] - warmup))
    floor = plan["min_lr"] / plan["peak_lr"]
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_continuation_scheduler(optimizer, restored_scheduler, *, global_step,
                                 start_step, end_step, warmup_steps,
                                 peak_lr, min_lr, original_base_lr):
    """Replace only the schedule AFTER the model/optimizer checkpoint loads.

    Constructor LR writes are intentional here. Adam/Muon buffers and group
    assignments are not reset. Group LR ratios follow the original base LRs.
    The first update uses the actual saved LR, not a rounded log value.
    """
    from torch.optim.lr_scheduler import LambdaLR

    validate_continuation_plan(start_step, end_step, warmup_steps, peak_lr, min_lr)
    if not start_step <= global_step < end_step:
        raise ValueError("Restored step is outside the continuation interval")
    requested = dict(version=1, start_step=start_step, end_step=end_step,
                     warmup_steps=warmup_steps, peak_lr=peak_lr, min_lr=min_lr)
    saved = getattr(restored_scheduler, "clean_continuation_plan", None)
    if saved is not None:
        if any(saved.get(key) != value for key, value in requested.items()):
            raise ValueError("Saved continuation plan differs; refusing a silent LR restart")
        plan = dict(saved)
    else:
        if global_step != start_step:
            raise ValueError("First LR restart must originate at the restored checkpoint step")
        if not math.isfinite(original_base_lr) or original_base_lr <= 0:
            raise ValueError("Original peak base LR must be positive and finite")
        bases = restored_scheduler.base_lrs
        if len(bases) != len(optimizer.param_groups):
            raise ValueError("Restored scheduler/optimizer group count differs")
        peaks = [float(base) * peak_lr / original_base_lr for base in bases]
        if not all(math.isfinite(value) and value > 0 for value in peaks):
            raise ValueError("Restored base learning rates must be positive and finite")
        starts = [float(group["lr"]) / peak
                  for group, peak in zip(optimizer.param_groups, peaks)]
        if not all(math.isfinite(value) and value >= 0 for value in starts):
            raise ValueError("Restored learning rates must be non-negative and finite")
        plan = dict(requested, peak_lrs=peaks, start_factors=starts)
    if len(plan["peak_lrs"]) != len(optimizer.param_groups) or len(plan["start_factors"]) != len(optimizer.param_groups):
        raise ValueError("Saved continuation group count differs from optimizer")
    for group, peak in zip(optimizer.param_groups, plan["peak_lrs"]):
        group["initial_lr"] = peak
    multipliers = [lambda step, index=index: lr_multiplier(step, plan, index)
                   for index in range(len(optimizer.param_groups))]
    scheduler = LambdaLR(optimizer, multipliers, last_epoch=global_step - 1)
    # LambdaLR includes additional attributes in state_dict; plain lambdas do
    # not serialize closures. Rebuild from this explicit plan on every resume.
    scheduler.clean_continuation_plan = plan
    return scheduler
