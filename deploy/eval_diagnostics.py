"""Passive RoboTwin feedback and episode lifecycle (no simulator imports)."""

from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .eval_logging import EventWriter, atomic_json, episode_directory_name, exception_fields, record


class InvalidActionError(ValueError):
    """A prediction cannot safely be passed to the simulator."""


def robot_feedback(env) -> dict:
    """Read state without rendering or stepping; gripper values are commands."""
    robot = env.robot
    return {
        "control_target": robot.get_left_arm_jointState() + robot.get_right_arm_jointState(),
        "actual_arm_qpos": robot.get_left_arm_real_jointState()[:-1] + robot.get_right_arm_real_jointState()[:-1],
        "gripper_command": [robot.get_left_gripper_val(), robot.get_right_gripper_val()],
        "left_ee_pose": robot.get_left_ee_pose(),
        "right_ee_pose": robot.get_right_ee_pose(),
    }


def joint_metadata(env) -> dict:
    robot = env.robot
    joints = robot.left_arm_joints + robot.right_arm_joints
    return {
        "arm_joint_names": [joint.get_name() for joint in joints],
        "arm_joint_limits": [np.asarray(joint.get_limits()).reshape(-1, 2)[0] for joint in joints],
        "control_order": ["left_arm[0:6]", "left_gripper_command[6]", "right_arm[7:13]", "right_gripper_command[13]"],
        "actual_arm_order": ["left_arm[0:6]", "right_arm[6:12]"],
        "gripper_measurement": "unavailable; gripper getters report command values",
    }


class EpisodeTrace:
    def __init__(self, task_dir: Path, context: dict, enabled: bool) -> None:
        self.context = context
        self.enabled = enabled
        self.directory = task_dir / "episodes" / episode_directory_name(context)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.results = EventWriter(task_dir / "episode_results.jsonl", context)
        self.inference = EventWriter(self.directory / "inference.jsonl", context)
        self.execution = EventWriter(self.directory / "execution.jsonl", context)
        self.started = time.monotonic()
        self.info = record(context, status="running", reason=None, executed_actions=0)
        self.stage = "setup"
        self.request_id = None
        self.plans = []
        self.finished = False
        self.logging_ms = 0.0
        self.metrics = {
            "request_count": 0,
            "infer_ms_total": 0.0,
            "rtt_ms_total": 0.0,
            "execution_ms_total": 0.0,
            "prediction_write_ms_total": 0.0,
        }
        self.save()

    def save(self) -> None:
        atomic_json(self.directory / "episode.json", self.info)

    def update(self, **fields) -> None:
        self.info.update(record(self.context, **fields))
        self.save()

    def log(self, writer: EventWriter, event: str, **fields) -> None:
        if self.enabled:
            start = time.monotonic()
            writer.write(event, request_id=self.request_id, **fields)
            self.logging_ms += (time.monotonic() - start) * 1000

    def finish(self, reason: str, env, exc: BaseException | None = None) -> None:
        if self.finished:
            return
        fields = exception_fields(exc, self.stage) if exc else {}
        self.update(
            status="finished",
            reason=reason,
            duration_ms=(time.monotonic() - self.started) * 1000,
            take_action_cnt=getattr(env, "take_action_cnt", 0),
            eval_success=getattr(env, "eval_success", False),
            logging_ms=self.logging_ms,
            metrics=self.metrics,
            **fields,
        )
        self.results.write(
            "episode_end", **{k: v for k, v in self.info.items() if k not in ("event", "schema_version", "timestamp")}
        )
        self.finished = True

    @contextmanager
    def planning(self, env):
        """Observe TOPP results without changing simulator fallback behavior."""
        originals = []
        try:
            if self.enabled:
                for arm in ("left", "right"):
                    planner = getattr(env.robot, f"{arm}_mplib_planner", None)
                    if planner is None or not hasattr(planner, "TOPP"):
                        self.log(self.execution, "diagnostic_unavailable", diagnostic=f"{arm}_TOPP")
                        continue
                    original = planner.TOPP
                    originals.append((planner, original))
                    planner.TOPP = self._observe_planner(original, arm)
            yield
        finally:
            for planner, original in originals:
                planner.TOPP = original

    def _observe_planner(self, original, arm):
        def observed(*args, **kwargs):
            start = time.monotonic()
            try:
                result = original(*args, **kwargs)
                points = len(result[1])
                self.plans.append(
                    {
                        "arm": arm,
                        "status": "ok" if points else "empty",
                        "trajectory_points": points,
                        "duration_ms": (time.monotonic() - start) * 1000,
                    }
                )
                return result
            except Exception as exc:
                self.plans.append(
                    {
                        "arm": arm,
                        "status": "exception",
                        "duration_ms": (time.monotonic() - start) * 1000,
                        **exception_fields(exc, "TOPP"),
                    }
                )
                raise

        return observed


def execute_chunk(env, actions, trace: EpisodeTrace) -> int:
    """Validate and execute only the prefix actually accepted by take_action()."""
    trace.stage = "validate_action"
    actions = np.asarray(actions)
    expected_dim = len(env.robot.left_arm_joints) + len(env.robot.right_arm_joints) + 2
    if (
        actions.ndim not in (1, 2)
        or actions.shape[-1] != expected_dim
        or actions.size == 0
        or not (np.issubdtype(actions.dtype, np.floating) or np.issubdtype(actions.dtype, np.integer))
        or not np.isfinite(actions).all()
    ):
        raise InvalidActionError(
            f"Invalid action shape/values: {actions.shape}, expected (*, {expected_dim}) finite values"
        )
    chunk = actions[None] if actions.ndim == 1 else actions
    executed = 0
    for index, action in enumerate(chunk):
        if env.eval_success or env.take_action_cnt >= env.step_lim:
            break
        # Keep the original frame refresh cadence; feedback never calls get_obs().
        if index > 0 and env.eval_video_path is not None:
            env.get_obs()
        trace.stage = "execute"
        before_count = env.take_action_cnt
        before = robot_feedback(env) if trace.enabled else None
        trace.plans = []
        trace.log(
            trace.execution,
            "action_start",
            chunk_index=index,
            take_action_cnt=before_count,
            action=action,
            before=before,
        )
        start = time.monotonic()
        try:
            env.take_action(action)
        except BaseException as exc:
            count_delta = env.take_action_cnt - before_count
            elapsed = (time.monotonic() - start) * 1000
            trace.metrics["execution_ms_total"] += elapsed
            if count_delta > 0:
                trace.info["executed_actions"] += count_delta
            trace.log(
                trace.execution,
                "action_error",
                chunk_index=index,
                action=action,
                take_action_cnt=env.take_action_cnt,
                before=before,
                planning=trace.plans,
                execution_ms=elapsed,
                count_delta=count_delta,
                **exception_fields(exc, "execute"),
            )
            raise
        execution_ms = (time.monotonic() - start) * 1000
        trace.metrics["execution_ms_total"] += execution_ms
        after = robot_feedback(env) if trace.enabled else None
        count_delta = env.take_action_cnt - before_count
        if count_delta > 0:
            executed += count_delta
            trace.info["executed_actions"] += count_delta
        trace.log(
            trace.execution,
            "action_end",
            chunk_index=index,
            action=action,
            before=before,
            after=after,
            take_action_cnt=env.take_action_cnt,
            count_delta=count_delta,
            planning=trace.plans,
            execution_ms=execution_ms,
            eval_success=env.eval_success,
        )
        if count_delta <= 0:
            raise RuntimeError("take_action returned without advancing the simulator action counter")
        if env.eval_success and env.eval_video_path is not None:
            env.get_obs()
    trace.log(
        trace.inference,
        "chunk_end",
        returned_actions=len(chunk),
        executed_actions=executed,
        unexecuted_suffix=len(chunk) - executed,
        take_action_cnt=env.take_action_cnt,
    )
    return executed


@contextmanager
def interrupt_on_signal():
    """Allow normal Python finalizers to record TERM/INT as interrupted episodes."""

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Evaluation received signal {signum}")

    previous = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
