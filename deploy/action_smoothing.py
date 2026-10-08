"""Optional inference-only filtering of RoboTwin absolute joint targets.

Layout: left arm[0:6], left gripper[6], right arm[7:13], right gripper[13].
Window smoothing uses only the already returned chunk; EMA state persists across
chunks and must be reset to the observed joints at the start of each episode.
This is not a velocity limit: simulator actions have variable physical duration.
"""
import math
import numpy as np


ARM_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


class RoboTwinActionSmoother:
    def __init__(self, mode="none", alpha=0.35, window=5, max_delta=0.05):
        if mode not in ("none", "ema"):
            raise ValueError("action_smoothing must be none or ema")
        if not math.isfinite(float(alpha)) or not 0 < float(alpha) <= 1:
            raise ValueError("smoothing_alpha must be finite and in (0, 1]")
        if isinstance(window, bool) or int(window) != window or not 1 <= window <= 51 or window % 2 != 1:
            raise ValueError("smoothing_window must be an odd integer in 1..51")
        if not math.isfinite(float(max_delta)) or float(max_delta) < 0:
            raise ValueError("smoothing_max_delta must be finite and nonnegative")
        self.mode, self.alpha = mode, float(alpha)
        self.window, self.max_delta = int(window), float(max_delta)
        self._previous = None

    @property
    def enabled(self):
        return self.mode != "none"

    def config(self):
        return dict(mode=self.mode, alpha=self.alpha, window=self.window, max_delta_rad_per_action=self.max_delta)

    @staticmethod
    def _checked(values, ndim):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != ndim or values.shape[-1] != 14 or not np.isfinite(values).all():
            raise ValueError("Smoothing requires finite RoboTwin physical qpos with 14 joints")
        if ndim == 2 and len(values) == 0:
            raise ValueError("Empty action chunk")
        return values.copy()

    def reset(self, observed_joints=None):
        self._previous = None
        if self.enabled and observed_joints is not None:
            self._previous = self._checked(observed_joints, 1)[ARM_INDICES]

    def prepare_chunk(self, actions):
        if not self.enabled:
            return np.asarray(actions).copy()
        actions = self._checked(actions, 2)
        if self.window > 1:
            radius = self.window // 2
            padded = np.pad(actions[:, ARM_INDICES], ((radius, radius), (0, 0)), mode="edge")
            # No interpolation, resampling, image changes, or gripper smoothing.
            actions[:, ARM_INDICES] = sum(padded[offset:offset + len(actions)] for offset in range(self.window)) / self.window
        return actions

    def filter_action(self, action):
        if not self.enabled:
            return np.asarray(action).copy()
        result = self._checked(action, 1)
        target = result[ARM_INDICES]
        if self._previous is None:
            raise RuntimeError("Reset smoother with observed joints before execution")
        delta = self.alpha * (target - self._previous)
        if self.max_delta > 0:
            delta = np.clip(delta, -self.max_delta, self.max_delta)
        self._previous = self._previous + delta
        result[ARM_INDICES] = self._previous
        return result
