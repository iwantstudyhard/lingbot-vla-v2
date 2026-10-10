# AutoHorizon soft-pointer functions are adapted from hatchetProject/AutoHorizon,
# commit c7504f1756109103f2cfcc2e23f1b1a23841c885 (Apache License 2.0).
# https://github.com/hatchetProject/AutoHorizon/blob/c7504f1756109103f2cfcc2e23f1b1a23841c885/src/openpi/models_pytorch/pi0_pytorch.py
# Licensed under the Apache License, Version 2.0; see LICENSE in this repository.
# http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Execution horizons compatible with the published AutoHorizon implementation."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


UPSTREAM_COMMIT = "c7504f1756109103f2cfcc2e23f1b1a23841c885"


@dataclass(frozen=True)
class AutoHorizonConfig:
    mode: str = "fixed"
    attention_step: int = 3
    hold_thr: float = 0.3
    max_entropy_q: float = 0.9

    @property
    def enabled(self) -> bool:
        return self.mode != "fixed"

    def validate(self, *, chunk_ret: bool, use_length: int, predicted_horizon: int, num_steps: int) -> None:
        if self.mode not in ("fixed", "observe", "auto"):
            raise ValueError(f"Invalid horizon_mode: {self.mode}")
        if not self.enabled:
            return
        if not chunk_ret:
            raise ValueError("AutoHorizon observe/auto requires chunk_ret=True")
        if not isinstance(self.attention_step, int) or not 1 <= self.attention_step <= num_steps:
            raise ValueError(f"autohorizon_attention_step must be in 1..{num_steps}, got {self.attention_step}")
        if not math.isfinite(self.hold_thr) or self.hold_thr < 0:
            raise ValueError("autohorizon_hold_thr must be finite and nonnegative")
        if not math.isfinite(self.max_entropy_q) or not 0 <= self.max_entropy_q <= 1:
            raise ValueError("autohorizon_max_entropy_q must be finite and in 0..1")
        if not 1 <= use_length <= predicted_horizon:
            raise ValueError(f"AutoHorizon use_length must be in 1..{predicted_horizon}, got {use_length}")

    def metadata(self) -> dict:
        return {**asdict(self), "run_len": 1, "upstream_commit": UPSTREAM_COMMIT}


@torch.no_grad()
def estimate_horizon(attention: torch.Tensor, config: AutoHorizonConfig) -> tuple[int | None, dict]:
    """Validate a single action matrix; return an explicit reason on invalid estimates."""
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1] or attention.shape[0] < 1:
        return None, {"fallback_reason": "invalid_attention_shape"}
    if attention.shape[0] == 1:
        return 1, {"method": "single_action"}
    attention = attention.float()
    if not torch.isfinite(attention).all():
        return None, {"fallback_reason": "nonfinite_attention"}
    row_mass = attention.sum(-1)
    if (attention < 0).any() or not torch.isfinite(row_mass).all() or (row_mass <= 1e-12).any():
        return None, {"fallback_reason": "invalid_attention_mass"}
    horizon, diagnostics = bidir_soft_pointer(
        attention, hold_thr=config.hold_thr, max_entropy_q=config.max_entropy_q,
    )
    if not torch.isfinite(horizon) or not 1 <= horizon.item() <= attention.shape[0]:
        return None, {"fallback_reason": "invalid_horizon"}
    return int(horizon.item()), diagnostics


# Keep the upstream backward-coordinate and join_row behavior for reproduction.
@torch.no_grad()
def pick_horizon_softpointer(attn: torch.Tensor,
                             hold_thr: float = 0.3,
                             max_entropy_q: float = 0.9):
    """
    Estimate number of executed actions (N) from an attention map.
    Assumes a single forward soft pointer (no smoothing, run_len=1).

    Args:
        attn: [T, T] attention matrix (post-softmax preferred)
        hold_thr: threshold for detecting plateau in pointer (default=0.3)
        max_entropy_q: quantile cutoff for reliable low-entropy rows

    Returns:
        N: estimated executed actions (int tensor on same device)
        diags: diagnostics dictionary
    """
    assert attn.ndim == 2 and attn.size(0) == attn.size(1)
    T = attn.size(0)
    dev = attn.device
    eps = 1e-12

    # --- Row normalize ---
    A = attn / (attn.sum(-1, keepdim=True) + eps)

    # --- Soft pointer μ_i = E[j] per row ---
    idx = torch.arange(T, device=dev, dtype=A.dtype)
    mu = (A * idx).sum(-1)  # [T]

    # --- Enforce monotonic non-decreasing pointer ---
    mu = torch.maximum(mu, torch.cummax(mu, dim=0).values)

    # --- Entropy-based reliability mask ---
    ent = -(A.clamp_min(eps) * (A.clamp_min(eps)).log()).sum(-1) / math.log(T)
    thr_ent = torch.quantile(ent.float(), q=min(max_entropy_q, 0.999))
    reliable = ent <= thr_ent

    # --- Pointer increment Δμ ---
    prev = torch.cat([mu.new_tensor([0.0]), mu[:-1]])
    dmu = mu - prev
    dmu[0] = mu[0]

    # --- Detect first plateau (Δμ < hold_thr and reliable) ---
    is_hold = (dmu < hold_thr) & reliable
    stop_row = torch.nonzero(is_hold, as_tuple=False)
    stop_row = int(stop_row[0].item()) if len(stop_row) > 0 else T - 1

    # --- Executed actions N = floor(μ_stop) + 1 ---
    N = torch.clamp(torch.floor(mu[min(stop_row, T - 1)]) + 1, 1, T)

    # --- Diagnostics ---
    diags = dict(
        mu=mu,
        dmu=dmu,
        entropy=ent,
        entropy_thr=thr_ent,
        reliable=reliable,
        stop_row=stop_row,
        method="softpointer_simplified",
    )

    return N, diags

@torch.no_grad()
def _soft_pointer_prefix(A: torch.Tensor,
                         hold_thr: float = 0.3,
                         run_len: int = 1,
                         max_entropy_q: float = 0.9,):
    """
    One-way soft pointer that advances until a plateau (prefix horizon).
    Returns: mu [T], dmu [T], reliable mask [T], stop_row (int)
    """
    assert A.ndim == 2 and A.size(0) == A.size(1)
    dev, T = A.device, A.size(0)
    eps = 1e-12

    # Row-normalize just in case
    A = A / (A.sum(-1, keepdim=True) + eps)

    # Soft pointer mu_i = E[j]
    idx = torch.arange(T, device=dev, dtype=A.dtype)  # 0..T-1
    mu = (A * idx).sum(-1)  # [T]

    # Enforce monotone non-decreasing (no backward jumps)
    mu = torch.maximum(mu, torch.cummax(mu, dim=0).values)

    # Entropy mask (reliability)
    ent = -(A.clamp_min(eps) * (A.clamp_min(eps)).log()).sum(-1) / math.log(T)
    thr_ent = torch.quantile(ent.float(), q=min(max_entropy_q, 0.999))
    reliable = (ent <= thr_ent)

    # Progress Δμ, baseline at 0
    prev = torch.cat([mu.new_tensor([0.0]), mu[:-1]])
    dmu = mu - prev
    dmu[0] = mu[0] - 0.0

    # First run_len consecutive "holds" among reliable rows
    is_hold = (dmu < hold_thr) & reliable
    win = F.conv1d(is_hold.float()[None, None, :], torch.ones(1,1,run_len, device=dev)).squeeze()
    if win.numel() and (win >= run_len - 1e-6).any():
        stop_row = int(torch.nonzero(win >= run_len, as_tuple=False)[0].item())
    else:
        stop_row = T - 1

    return mu, dmu, reliable, stop_row

@torch.no_grad()
def bidir_soft_pointer(attn: torch.Tensor,
                       hold_thr: float = 0.3,
                       run_len: int = 1,
                       max_entropy_q: float = 0.9):
    """
    Forward + backward soft pointers. If they overlap, stitch them to cover all actions.
    Returns:
        j_hat: [T] estimated key index per row (float, 0-based)
        join_row: int where we stitch (or None if no overlap)
        diags: dict with forward/backward mu, horizons, etc.
    """
    assert attn.ndim == 2 and attn.size(0) == attn.size(1)
    T = attn.size(0)
    eps = 1e-12

    # Ensure row-normalized
    A = attn / (attn.sum(-1, keepdim=True) + eps)

    # ---- forward prefix pointer ----
    mu_f, dmu_f, rel_f, stop_f = _soft_pointer_prefix(
        A, hold_thr=hold_thr, run_len=run_len,
        max_entropy_q=max_entropy_q,
    )
    j_f = mu_f  # 0-based expected key index
    N_f = int(torch.clamp(torch.floor(mu_f[min(stop_f, T-1)]) + 1, 1, T).item())

    # ---- backward suffix pointer ----
    # Flip both axes, run the same logic, then map back.
    A_rev = torch.flip(A, dims=(0, 1))
    mu_b_rev, dmu_b_rev, rel_b_rev, stop_b_rev = _soft_pointer_prefix(
        A_rev, hold_thr=hold_thr, run_len=run_len,
        max_entropy_q=max_entropy_q,
    )
    # Map reversed μ back to original indexing:
    # row i in original corresponds to row T-1-i in reversed;
    # key index j in original corresponds to T-1 - j in reversed.
    mu_b = torch.flip(T - 1 - mu_b_rev, dims=(0,))  # [T]
    j_b = mu_b
    N_b = int(torch.clamp(torch.floor(mu_b[max(T-1-stop_b_rev, 0)]) + 1, 1, T).item())

    # ---- decide if forward & backward cover the whole range ----
    # We look for the earliest row where forward pointer "meets or passes" backward pointer.
    # Allow a 1-key tolerance to account for soft estimates.
    gap = j_b - j_f  # positive if backward is to the right of forward
    meet_mask = (gap <= 1.0)
    join_row = None
    if meet_mask.any():
        join_row = int(torch.nonzero(meet_mask, as_tuple=False)[0].item())

    diags = dict(
        mu_forward=mu_f, dmu_forward=dmu_f, reliable_forward=rel_f, stop_forward=stop_f, N_forward=N_f,
        mu_backward=mu_b, # already mapped back
        stop_backward=T-1-stop_b_rev, N_backward=N_b,
        join_row=join_row, gap=gap, method="bidir_soft_pointer"
    )
    N_f = diags["N_forward"]
    N_b = diags["N_backward"]

    if (N_f + N_b >= T) and (join_row is not None):
        N = T
    else:
        N = N_f
    return torch.tensor(N), diags
