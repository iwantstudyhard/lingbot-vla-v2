"""Evaluation contracts tested without checkpoints, CUDA, or RoboTwin assets."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import random
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from deploy.eval_diagnostics import EpisodeTrace, InvalidActionError, execute_chunk, interrupt_on_signal
from deploy.eval_logging import (
    EventWriter,
    atomic_json,
    atomic_npz,
    format_progress,
    prediction_path,
    read_events,
    read_json,
    read_task_attempts,
    summarize_run,
)
from deploy.robotwin_evaluation import evaluate_task


ROOT = Path(__file__).resolve().parents[1]


class Joint:
    def get_name(self):
        return "test_joint"

    def get_limits(self):
        return np.array([[-1.0, 1.0]])


class Robot:
    def __init__(self):
        self.left_arm_joints, self.right_arm_joints = [Joint() for _ in range(6)], [Joint() for _ in range(6)]
        self.target, self.actual = np.zeros(14), np.zeros(12)
        self.left_mplib_planner = SimpleNamespace(TOPP=lambda *a, **k: ([], np.zeros((2, 6)), [], [], 0))
        self.right_mplib_planner = SimpleNamespace(TOPP=lambda *a, **k: ([], np.zeros((2, 6)), [], [], 0))

    def get_left_arm_jointState(self):
        return self.target[:7].tolist()

    def get_right_arm_jointState(self):
        return self.target[7:].tolist()

    def get_left_arm_real_jointState(self):
        return self.actual[:6].tolist() + [float(self.target[6])]

    def get_right_arm_real_jointState(self):
        return self.actual[6:].tolist() + [float(self.target[13])]

    def get_left_gripper_val(self):
        return float(self.target[6])

    def get_right_gripper_val(self):
        return float(self.target[13])

    def get_left_ee_pose(self):
        return [0, 0, 0, 1, 0, 0, 0]

    get_right_ee_pose = get_left_ee_pose


class Env:
    def __init__(self, success_at=None, step_limit=3):
        self.robot, self.eval_success, self.take_action_cnt = Robot(), False, 0
        self.eval_video_path, self.render_freq = None, 0
        self.step_lim, self.success_at = step_limit, success_at
        self.test_num, self.suc, self.plan_success = 0, 0, True
        self.observations = 0

    def setup_demo(self, seed, **kwargs):
        self.take_action_cnt, self.eval_success, self.seed = 0, False, seed
        self.robot.target[:] = 0

    def play_once(self):
        return {"info": {"seed": self.seed}}

    def close_env(self, **kwargs):
        pass

    def check_success(self):
        return self.seed != 100000

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        self.observations += 1
        return {
            "observation": {
                key: {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
                for key in ("head_camera", "left_camera", "right_camera")
            },
            "joint_action": {"vector": self.robot.target.copy()},
        }

    def take_action(self, action):
        if self.take_action_cnt >= self.step_lim or self.eval_success:
            return
        for arm in ("left", "right"):
            try:
                getattr(self.robot, arm + "_mplib_planner").TOPP([])
            except RuntimeError:
                pass  # Match the simulator's silent planning fallback.
        self.robot.target = np.array(action)
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at


class FakePolicy:
    def __init__(self, root, invalid=False):
        self.norm = root / "source_norm.json"
        self.norm.write_text('{"norm_stats": {}}\n')
        self.invalid, self.calls = invalid, 0

    def infer(self, obs):
        if obs.get("reset"):
            return {"action": None}
        self.calls += 1
        action = np.full((4, 14), 0.2)
        if self.invalid:
            action[0, 0] = np.nan
        return {"action": action}

    def infer_with_diagnostics(self, obs):
        result = self.infer(obs)
        full = np.full((1, 50, 14), 0.2)
        result["_diagnostics"] = {
            "available": True,
            "model_forward": True,
            "timing": {},
            "arrays": {
                "full_action": full,
                "returned_action": result["action"],
                "normalized_actions": np.zeros((1, 50, 55)),
                "action_joint_mask": np.ones((1, 55), dtype=bool),
                "state": np.zeros((1, 55)),
            },
        }
        return result

    def evaluation_metadata(self):
        return {"normalization_path": str(self.norm), "features": {}, "normalization_types": {}}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        atomic_json(self.run / "run_manifest.json", {"tasks": ["lift_pot"], "num_episodes": 2})

    def trace(self, enabled=True, episode=0, attempt=1):
        directory = self.run / "eval_results/lift_pot"
        return EpisodeTrace(
            directory,
            {
                "run_id": "run",
                "task": "lift_pot",
                "attempt": attempt,
                "episode_id": episode,
                "seed": 100001,
                "slot": 0,
            },
            enabled,
        )


class LoggingTests(Fixture):
    def test_progress_restores_rate_and_lifecycle_state(self):
        self.assertEqual(format_progress(self.run, "lift_pot", 1), "episodes 0/2, success 0, rate N/A (initializing)")
        trace = self.trace(enabled=False)
        self.assertIn("rate N/A (running)", format_progress(self.run, "lift_pot", 1))
        trace.finish("success", Env())
        self.assertEqual(format_progress(self.run, "lift_pot", 1), "episodes 1/2, success 1, rate 100.0% (running)")
        root = trace.directory.parent.parent
        EventWriter(root / "episode_results.jsonl", {"attempt": 1}).write(
            "episode_end", episode_id=1, reason="step_limit"
        )
        EventWriter(root / "attempt_results.jsonl", {"attempt": 1}).write("attempt_end", status="complete")
        self.assertEqual(format_progress(self.run, "lift_pot", 1), "episodes 2/2, success 1, rate 50.0% (complete)")
        self.assertIn("rate N/A (initializing)", format_progress(self.run, "lift_pot", 2))

    def test_progress_excludes_exception_from_rate_and_recovers_partial_line(self):
        root = self.run / "eval_results/lift_pot"
        EventWriter(root / "task_config.jsonl", {"attempt": 1}).write("task_config", requested_episodes=5)
        writer = EventWriter(root / "episode_results.jsonl", {"attempt": 1})
        writer.write("episode_end", episode_id=0, reason="step_limit")
        writer.write("episode_end", episode_id=1, reason="exception")
        with writer.path.open("a") as stream:
            stream.write('{"unfinished":')
        self.assertEqual(format_progress(self.run, "lift_pot", 1), "episodes 1/5, success 0, rate 0.0% (exception)")
        EventWriter(root / "attempt_results.jsonl", {"attempt": 1}).write("attempt_end", status="interrupted")
        self.assertIn("(interrupted)", format_progress(self.run, "lift_pot", 1))

    def test_flush_strict_json_and_truncated_line_recovery(self):
        path = self.run / "events.jsonl"
        EventWriter(path).write("test", numeric=np.array([1, np.inf, np.nan]))
        self.assertEqual(read_events(path)[0]["numeric"], [1.0, None, None])
        with path.open("a") as stream:
            stream.write('{"unfinished":')
        warnings = []
        self.assertEqual(len(read_events(path, warnings)), 1)
        self.assertEqual(len(warnings), 1)
        EventWriter(path).write("next_attempt")
        self.assertEqual([item["event"] for item in read_events(path)], ["test", "next_attempt"])

    def test_numeric_npz_no_pickle_no_overwrite_and_cleanup(self):
        path = self.run / "prediction.npz"
        atomic_npz(path, {"action": np.array([[np.nan, 1]])})
        with np.load(path, allow_pickle=False) as data:
            self.assertTrue(np.isnan(data["action"][0, 0]))
        with self.assertRaises(FileExistsError):
            atomic_npz(path, {"action": [3]})
        with self.assertRaises(ValueError):
            atomic_npz(self.run / "bad.npz", {"value": np.array([{}], dtype=object)})
        self.assertFalse(list(self.run.glob("*.tmp")))

    def test_context_path_validation(self):
        context = self.trace().context | {"request_id": "000001"}
        self.assertIn("episode_0_seed_100001", str(prediction_path(self.run, context)))
        for fields in ({"task": "../outside"}, {"request_id": "../outside"}, {"run_id": "other"}, {"attempt": 0}):
            with self.assertRaises(ValueError):
                prediction_path(self.run, context | fields)

    def test_attempt_isolation_and_complete_only_statistics(self):
        for attempt, reasons in ((1, ["success", "exception"]), (2, ["step_limit", "step_limit"]), (3, ["success"])):
            root = self.run / "eval_results/lift_pot"
            EventWriter(root / "task_config.jsonl", {"attempt": attempt}).write("task_config", requested_episodes=2)
            EventWriter(root / "attempt_results.jsonl", {"attempt": attempt}).write(
                "attempt_end", status="complete" if attempt == 2 else "error"
            )
            for episode, reason in enumerate(reasons):
                EventWriter(root / "episode_results.jsonl", {"attempt": attempt}).write(
                    "episode_end", episode_id=episode, reason=reason
                )
        result = summarize_run(self.run)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["successes"], 0)
        self.assertEqual(result["tasks"][0]["selected_attempt"], 2)
        before = (self.run / "summary.json").read_bytes()
        summarize_run(self.run, write=False)
        self.assertEqual(before, (self.run / "summary.json").read_bytes())
        EventWriter(self.run / "scheduler_events.jsonl").write("task_exit", task="lift_pot", attempt=2, exit_code=1)
        self.assertEqual(summarize_run(self.run)["episodes"], 0)

    def test_previously_saved_nested_logs_remain_readable(self):
        from tools.analyze_eval_logs import generate_report

        root = self.run / "eval_results/lift_pot/attempts/attempt_1"
        atomic_json(root / "task_config.json", {"requested_episodes": 2})
        atomic_json(root / "attempt_result.json", {"status": "complete"})
        for episode in range(2):
            EventWriter(root / "episode_results.jsonl").write("episode_end", episode_id=episode, reason="success")
            atomic_json(
                root / "episodes" / f"episode_{episode}_seed_{100001 + episode}" / "episode.json",
                {
                    "task": "lift_pot",
                    "attempt": 1,
                    "episode_id": episode,
                    "seed": 100001 + episode,
                    "status": "finished",
                    "reason": "success",
                    "eval_trace": "off",
                },
            )
        self.assertEqual(summarize_run(self.run)["success_rate"], 1)
        self.assertIn("rate 100.0% (complete)", format_progress(self.run, "lift_pot", 1))
        generate_report(self.run, self.run / "analysis")
        self.assertIn("episode_1_seed_100002", (self.run / "analysis/episodes.csv").read_text(encoding="utf-8-sig"))

    def test_flat_retry_after_truncated_line_can_complete(self):
        root = self.run / "eval_results/lift_pot"
        for attempt in (1, 2):
            EventWriter(root / "task_config.jsonl", {"attempt": attempt}).write("task_config", requested_episodes=2)
            writer = EventWriter(root / "episode_results.jsonl", {"attempt": attempt})
            writer.write("episode_end", episode_id=0, reason="success")
            if attempt == 1:
                with writer.path.open("a") as stream:
                    stream.write('{"attempt": 1, "unfinished":')
            else:
                writer.write("episode_end", episode_id=1, reason="step_limit")
                EventWriter(root / "attempt_results.jsonl", {"attempt": attempt}).write(
                    "attempt_end", status="complete"
                )
        result = summarize_run(self.run)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["tasks"][0]["selected_attempt"], 2)
        self.assertTrue(result["warnings"])


class ExecutionTests(Fixture):
    def test_success_and_step_limit_truncate_chunk(self):
        for episode, success_at in enumerate((2, None)):
            env, trace = Env(success_at=success_at), self.trace(episode=episode)
            self.assertEqual(execute_chunk(env, np.ones((5, 14)), trace), 2 if success_at else 3)
            ends = [item for item in read_events(trace.execution.path) if item["event"] == "action_end"]
            self.assertEqual(len(ends), env.take_action_cnt)
            self.assertEqual(ends[-1]["after"]["actual_arm_qpos"], [0.0] * 12)
            self.assertEqual(ends[-1]["after"]["control_target"], [1.0] * 14)
            self.assertEqual(env.observations, 0)

    def test_single_action_and_invalid_action(self):
        env, trace = Env(), self.trace()
        self.assertEqual(execute_chunk(env, np.zeros(14), trace), 1)
        for bad in (np.ones((2, 13)), np.full(14, np.nan), np.zeros((0, 14))):
            with self.assertRaises(InvalidActionError):
                execute_chunk(env, bad, trace)
        self.assertEqual(env.take_action_cnt, 1)

    def test_planner_error_captured_and_original_restored(self):
        env, trace = Env(), self.trace()

        def failure(*args):
            raise RuntimeError("TOPP failed")

        env.robot.left_mplib_planner.TOPP = failure
        original_right = env.robot.right_mplib_planner.TOPP
        with trace.planning(env):
            execute_chunk(env, np.zeros(14), trace)
        self.assertIs(env.robot.left_mplib_planner.TOPP, failure)
        self.assertIs(env.robot.right_mplib_planner.TOPP, original_right)
        plans = read_events(trace.execution.path)[-1]["planning"]
        self.assertEqual(plans[0]["status"], "exception")
        self.assertIn("TOPP failed", plans[0]["traceback"])

    def test_execution_exception_leaves_correlated_event(self):
        env, trace = Env(), self.trace()

        def failure(action):
            raise RuntimeError("physics failure")

        env.take_action = failure
        with self.assertRaisesRegex(RuntimeError, "physics failure"):
            execute_chunk(env, np.zeros(14), trace)
        self.assertEqual(read_events(trace.execution.path)[-1]["event"], "action_error")

    def test_signal_restores_handlers(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            with interrupt_on_signal():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_empty_planning_trajectory_is_observable(self):
        env, trace = Env(), self.trace()
        env.robot.left_mplib_planner.TOPP = lambda *a: ([], np.zeros((0, 6)), [], [], 0)
        with trace.planning(env):
            execute_chunk(env, np.zeros(14), trace)
        self.assertEqual(read_events(trace.execution.path)[-1]["planning"][0]["status"], "empty")


class PolicyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch required for actual inference method tests")
        cls.torch = torch
        tree = ast.parse((ROOT / "deploy/lingbot_vla_v2_policy.py").read_text())
        policy = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LingbotVLAv2Server"
        )
        # Execute the production methods with a lightweight feature transform/model;
        # importing the whole policy would require transformers and CUDA assets.
        methods = [
            "_pad_and_stack_tensors",
            "_infer_batch",
            "_unapply_batched_actions",
            "infer",
            "infer_with_diagnostics",
        ]
        policy.body = [node for node in policy.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        namespace = {"torch": torch, "np": np, "time": time}
        exec(compile(ast.Module(body=[policy], type_ignores=[]), "<production policy methods>", "exec"), namespace)
        cls.policy_class = namespace["LingbotVLAv2Server"]

    def policy(self, chunk_ret=True, use_length=3):
        torch = self.torch
        policy = self.policy_class.__new__(self.policy_class)
        policy.chunk_ret, policy.use_length = chunk_ret, use_length
        policy.use_bf16, policy.use_compile = False, False
        policy.horizon_config = SimpleNamespace(enabled=False)
        policy._last_horizon = None
        policy.global_step = 0
        policy.last_action_chunk = policy.last_normalized_action_chunk = policy._last_full_diagnostics = None
        policy.action_key = ["action"]
        calls = []

        def sample(*args, **kwargs):
            calls.append(1)
            return torch.randn(1, 50, 55, dtype=torch.bfloat16 if policy.use_bf16 else torch.float32).float()

        policy.sample_actions_fn = sample
        mapping = [*range(6), 50, *range(6, 12), 51]
        transform = SimpleNamespace(unapply=lambda item: {"action": item["actions"][:, mapping]})
        policy.vla = SimpleNamespace(sample_actions_batch=sample, feature_transform=transform)
        mask = torch.zeros(55, dtype=torch.bool)
        mask[mapping] = True
        policy._prepare_model_input = lambda obs: {
            "state": torch.ones(55),
            "state_joint_mask": mask,
            "action_joint_mask": mask,
            "lang_tokens": torch.ones(3, dtype=torch.int64),
            "lang_masks": torch.ones(3, dtype=torch.bool),
        }
        return policy, calls

    def test_full_capture_preserves_actions_rng_and_forward_count(self):
        torch = self.torch
        observations = {"observation.state": np.zeros(14)}
        plain, plain_calls = self.policy()
        traced, traced_calls = self.policy()
        torch.manual_seed(23)
        plain_result = plain.infer(observations)
        plain_rng = torch.get_rng_state().clone()
        np_state, random_state = np.random.get_state(), random.getstate()
        torch.manual_seed(23)
        traced_result = traced.infer_with_diagnostics(observations)
        np.testing.assert_array_equal(plain_result["action"], traced_result["action"])
        self.assertTrue(torch.equal(plain_rng, torch.get_rng_state()))
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertEqual(random_state, random.getstate())
        self.assertEqual((len(plain_calls), len(traced_calls)), (1, 1))
        self.assertIsNone(traced.last_normalized_action_chunk)
        arrays = traced_result["_diagnostics"]["arrays"]
        self.assertEqual(arrays["normalized_actions"].shape, (1, 50, 55))
        self.assertEqual(arrays["full_action"].shape, (1, 50, 14))
        np.testing.assert_array_equal(arrays["full_action"][0, :3], traced_result["action"])

    def test_cached_step_mode_does_not_resample(self):
        plain, calls = self.policy(chunk_ret=False)
        observed, traced_calls = self.policy(chunk_ret=False)
        self.torch.manual_seed(45)
        results = [plain.infer({"observation.state": np.zeros(14)})["action"] for _ in range(5)]
        self.torch.manual_seed(45)
        traces = [observed.infer_with_diagnostics({"observation.state": np.zeros(14)}) for _ in range(5)]
        for expected, actual in zip(results, traces):
            np.testing.assert_array_equal(expected, actual["action"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(traced_calls), 2)
        self.assertEqual([item["_diagnostics"]["model_forward"] for item in traces], [True, False, False, True, False])

    def test_bf16_capture_preserves_actions_and_rng(self):
        plain, plain_calls = self.policy()
        observed, observed_calls = self.policy()
        plain.use_bf16 = observed.use_bf16 = True
        observation = {"observation.state": np.zeros(14)}
        self.torch.manual_seed(18)
        expected = plain.infer(observation)
        state = self.torch.get_rng_state().clone()
        self.torch.manual_seed(18)
        actual = observed.infer_with_diagnostics(observation)
        np.testing.assert_array_equal(expected["action"], actual["action"])
        self.assertTrue(self.torch.equal(state, self.torch.get_rng_state()))
        self.assertEqual((len(plain_calls), len(observed_calls)), (1, 1))

    def test_later_normalized_request_keeps_original_forward_policy(self):
        plain, calls = self.policy(chunk_ret=False)
        traced, traced_calls = self.policy(chunk_ret=False)
        observation = {"observation.state": np.zeros(14)}
        plain.infer(observation)
        traced.infer_with_diagnostics(observation)
        plain.infer(observation, return_normalized=True)
        traced.infer(observation, return_normalized=True)
        self.assertEqual((len(calls), len(traced_calls)), (2, 2))


class IntegrationTests(Fixture):
    def test_autohorizon_full_and_off_execution(self):
        try:
            from test_autohorizon import policy_fixture
            import torch
        except ImportError:
            self.skipTest("torch and einops required for AutoHorizon integration")
        for mode in ("observe", "auto"):
            for enabled in (True, False):
                with self.subTest(mode=mode, full_trace=enabled):
                    server = self.server()
                    policy, calls = policy_fixture(mode)
                    policy.action_key = ["action"]
                    policy.vla.feature_transform.unapply = lambda item: {
                        "action": torch.cat([item["actions"], torch.zeros(6, 6)], dim=1),
                    }
                    # Keep the lightweight normalization metadata and reset for this fake environment.
                    policy.evaluation_metadata = server._policy.evaluation_metadata
                    policy.reset = lambda **kwargs: None
                    server._policy = policy
                    task = f"horizon_{mode}_{enabled}"
                    root = self.evaluate(server, env=Env(success_at=1), enabled=enabled, task=task)
                    episode = next((root / "episodes").iterdir())
                    info = read_json(episode / "episode.json")
                    expected_length = 3 if mode == "observe" else 6
                    metrics = info["metrics"]["autohorizon"]
                    self.assertEqual(metrics["selected_counts"], {str(expected_length): 1})
                    self.assertEqual(metrics["executed_counts"], {"1": 1})
                    self.assertEqual(metrics["estimated_counts"], {"6": 1})
                    self.assertEqual(len(calls), 2)  # one prediction in each of the two episodes
                    predictions = list((episode / "predictions").glob("*.npz"))
                    if enabled:
                        with np.load(predictions[0], allow_pickle=False) as arrays:
                            self.assertEqual(arrays["full_action"].shape, (1, 6, 14))
                            self.assertEqual(arrays["action_attention"].shape, (1, 6, 6))
                            self.assertEqual(arrays["returned_action"].shape, (expected_length, 14))
                        events = read_events(episode / "inference.jsonl")
                        executed = next(event for event in events if event["event"] == "horizon_execution")
                        self.assertEqual(executed["executed_actions"], 1)
                    else:
                        self.assertEqual(predictions, [])

    def server(self, invalid=False, slot=0):
        try:
            from deploy.websocket_policy_server import WebsocketPolicyServer
        except ModuleNotFoundError as exc:
            if exc.name == "websockets":
                self.skipTest("websockets required")
            raise
        return WebsocketPolicyServer(FakePolicy(self.run, invalid), eval_run_dir=str(self.run), eval_slot=slot)

    def evaluate(self, server, env=None, enabled=True, attempt=1, task="lift_pot"):
        root = self.run / "eval_results" / task
        root.mkdir(parents=True, exist_ok=True)

        class Client:
            def get_server_metadata(self):
                return server._metadata

            def infer(self, obs):
                return server._infer(dict(obs))

        args = {
            "task_name": task,
            "policy_name": "test",
            "task_config": "demo_clean",
            "ckpt_setting": "test",
            "render_freq": 0,
            "clear_cache_freq": 10,
        }
        usr = {
            "_task_dir": str(root),
            "run_dir": str(self.run),
            "robo_name": "robotwin",
            "num_episodes": 2,
            "seed": 0,
            "attempt": attempt,
            "slot": server._slot,
            "eval_trace": "full" if enabled else "off",
            "launcher_owned": True,
        }
        evaluate_task(env or Env(), args, Client(), usr, lambda info: "lift the pot", RuntimeError)
        return root

    def test_end_to_end_seed_filter_and_report_recovery(self):
        from tools.analyze_eval_logs import generate_report

        server = self.server()
        root = self.evaluate(server)
        seeds = read_events(root / "seed_checks.jsonl")
        self.assertFalse([item for item in seeds if item["event"] == "seed_check_end"][0]["accepted"])
        results = read_events(root / "episode_results.jsonl")
        self.assertEqual([item["seed"] for item in results], [100001, 100002])
        self.assertTrue(all(item["reason"] == "step_limit" for item in results))
        summarize_run(self.run)
        before = (self.run / "summary.json").read_bytes()
        predictions = list(root.glob("episodes/*/predictions/*.npz"))
        self.assertEqual(len(predictions), 2)
        predictions[0].write_bytes(b"broken NPZ")
        with next(root.glob("episodes/*/execution.jsonl")).open("a") as stream:
            stream.write('{"partial":')
        diagnostic = generate_report(self.run, self.run / "analysis")
        self.assertIn("target_changed_small_actual_motion", diagnostic["counts"])
        self.assertIn("prediction_unavailable", diagnostic["counts"])
        self.assertTrue(diagnostic["warnings"])
        self.assertEqual(before, (self.run / "summary.json").read_bytes())
        self.assertTrue((self.run / "analysis/report.html").is_file())
        self.assertTrue(list((self.run / "analysis/figures").glob("*.png")))
        metadata = read_json(next((self.run / "inference_logs").glob("*.model_*.json")))
        content = (self.run / metadata["normalization_snapshot"]).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["normalization_sha256"])

    def test_invalid_prediction_retained_attempt_fails(self):
        server = self.server(invalid=True)
        with self.assertRaises(InvalidActionError):
            self.evaluate(server)
        root = self.run / "eval_results/lift_pot"
        self.assertEqual(read_events(root / "episode_results.jsonl")[0]["reason"], "invalid_action")
        with np.load(next(root.glob("episodes/*/predictions/*.npz")), allow_pickle=False) as data:
            self.assertTrue(np.isnan(data["returned_action"]).any())
        self.assertEqual(summarize_run(self.run)["episodes"], 0)

    def test_flat_retries_preserve_predictions_and_do_not_mix_success_rates(self):
        from tools.analyze_eval_logs import generate_report

        server = self.server()
        root = self.evaluate(server, env=Env(success_at=1))
        originals = {path: path.read_bytes() for path in root.glob("episodes/*/predictions/*.npz")}
        self.evaluate(server, attempt=2)
        self.assertEqual(len(list(root.glob("episodes/*/predictions/*.npz"))), 4)
        self.assertTrue(all(path.read_bytes() == content for path, content in originals.items()))
        self.assertFalse((root / "attempts").exists())
        self.assertEqual(len(read_events(root / "task_config.jsonl")), 2)
        self.assertEqual(len(read_events(root / "attempt_results.jsonl")), 2)
        self.assertEqual(summarize_run(self.run)["success_rate"], 0)
        self.assertIn("success 2, rate 100.0% (complete)", format_progress(self.run, "lift_pot", 1))
        self.assertIn("success 0, rate 0.0% (complete)", format_progress(self.run, "lift_pot", 2))
        generate_report(self.run, self.run / "analysis")
        import csv

        with (self.run / "analysis/episodes.csv").open(encoding="utf-8-sig") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        with self.assertRaisesRegex(FileExistsError, "Refusing to reuse attempt"):
            self.evaluate(server, attempt=2)
        self.assertEqual(len(read_events(root / "task_config.jsonl")), 2)

    def test_write_failure_and_disconnect_and_interruption(self):
        for attempt, failure in (
            (1, OSError("disk full")),
            (2, ConnectionError("lost socket")),
            (3, KeyboardInterrupt("TERM")),
        ):
            server = self.server()
            target = "deploy.websocket_policy_server.atomic_npz" if attempt == 1 else ""
            if target:
                with patch(target, side_effect=failure), self.assertRaises(OSError):
                    self.evaluate(server, attempt=attempt)
            else:
                with patch.object(server, "_infer", side_effect=failure), self.assertRaises(type(failure)):
                    self.evaluate(server, attempt=attempt)
        self.assertEqual(summarize_run(self.run)["episodes"], 0)
        root = self.run / "eval_results/lift_pot"
        attempts = read_task_attempts(root)
        self.assertEqual([item["attempt"] for item in attempts], [1, 2, 3])
        self.assertEqual(attempts[-1]["episodes"][0]["reason"], "interrupted")
        self.assertEqual(len(list(root.glob("episodes/*/episode.json"))), 3)
        self.assertFalse((root / "attempts").exists())

    def test_off_mode_and_unsupported_handshake(self):
        server = self.server()
        self.evaluate(server, enabled=False)
        self.assertFalse(list((self.run / "eval_results").rglob("*.npz")))
        self.assertEqual(summarize_run(self.run)["completed_tasks"], 1)
        server._metadata["eval_trace_schema"] = None
        with self.assertRaisesRegex(RuntimeError, "Full tracing requires"):
            self.evaluate(server, attempt=2)

    def test_context_rejects_wrong_slot(self):
        server = self.server()
        context = self.trace().context | {"slot": 1, "request_id": "000000"}
        with self.assertRaisesRegex(ValueError, "different server slot"):
            server._infer({"_eval_context": context})
        self.assertEqual(server._policy.calls, 0)

    def test_unclosed_events_and_legacy_report(self):
        from tools.analyze_eval_logs import generate_report

        trace = self.trace()
        trace.inference.write("request_start", request_id="000000")
        trace.execution.write("action_start", request_id="000000", chunk_index=0, take_action_cnt=0)
        result = generate_report(self.run, self.run / "analysis")
        self.assertEqual(set(result["counts"]), {"unclosed_request", "unclosed_action", "unclosed_episode"})
        legacy = self.run / "legacy"
        root = legacy / "eval_results/lift_pot"
        root.mkdir(parents=True)
        (root / "_result.txt").write_text("Timestamp: today\n\n0.0")
        generate_report(legacy, legacy / "analysis")
        report = (legacy / "analysis/report.html").read_text()
        self.assertIn("trace unavailable", report)
        self.assertIn("Unavailable", report)

    def test_bounds_diagnostic_ignores_padded_dimensions(self):
        from tools.analyze_eval_logs import episode_anomalies

        trace = self.trace()
        metadata_path = self.run / "metadata.json"
        atomic_json(
            metadata_path,
            {
                "normalization_types": {"action.arm.position": "bounds_99_woclip"},
                "features": {
                    "joints": ["arm.position", "effector.position"],
                    "joints_max_dim": {"arm.position": 50, "effector.position": 5},
                },
            },
        )
        normalized = np.zeros((1, 50, 55))
        normalized[:, :, 40] = 20  # Padded output is excluded from diagnosis.
        normalized[:, :, 0] = 2
        mask = np.zeros((1, 55), dtype=bool)
        mask[:, :12] = True
        mask[:, 50:52] = True
        path = prediction_path(self.run, trace.context | {"request_id": "000000"})
        atomic_npz(path, {"normalized_actions": normalized, "action_joint_mask": mask})
        info = trace.info | {"model_metadata": "metadata.json", "status": "finished", "reason": "step_limit"}
        found, _ = episode_anomalies(
            self.run,
            info,
            [
                {
                    "event": "request_end",
                    "request_id": "000000",
                    "diagnostic": {"artifact": path.relative_to(self.run).as_posix()},
                }
            ],
            [],
            [],
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["evidence"]["max_abs"], 2)

    def test_two_slot_real_websocket_roundtrip(self):
        import websockets.asyncio.server
        import websockets.sync.client

        from deploy.msgpack_numpy import Packer, unpackb

        servers = [self.server(slot=slot) for slot in (0, 1)]
        ready = threading.Event()
        loop = asyncio.new_event_loop()
        ports, listeners = [], []

        async def start():
            for server in servers:
                listener = await websockets.asyncio.server.serve(server._handler, "127.0.0.1", 0, compression=None)
                listeners.append(listener)
                ports.append(listener.sockets[0].getsockname()[1])
            ready.set()

        def worker():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start())
            loop.run_forever()
            for listener in listeners:
                listener.close()
                loop.run_until_complete(listener.wait_closed())
            loop.close()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(ready.wait(10))
            for slot, port in enumerate(ports):
                context = self.trace(episode=slot).context | {"slot": slot, "request_id": "000000"}
                with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", compression=None) as socket:
                    self.assertEqual(unpackb(socket.recv())["eval_trace_schema"], 1)
                    socket.send(Packer().pack({"observation.state": np.zeros(14), "_eval_context": context}))
                    reply = unpackb(socket.recv())
                    self.assertEqual(reply["_eval_trace"]["slot"], slot)
                    self.assertTrue((self.run / reply["_eval_trace"]["artifact"]).exists())
            self.assertEqual(len(list((self.run / "eval_results").rglob("*.npz"))), 2)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
