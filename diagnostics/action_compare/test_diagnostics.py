"""CPU-only tests. Mock models/physics are fixtures, not evaluation evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ARM, boundary_summary, checked_actions, differences, observation_hash


class MetricTests(unittest.TestCase):
    def test_gripper_units_do_not_pollute_arm_metrics(self):
        actions = np.zeros((6, 14), np.float32)
        actions[:, [6, 13]] = np.arange(6)[:, None] * 100
        self.assertEqual(differences(actions)["mean_abs_delta_rad_per_action"], 0)

    def test_padding_or_wrong_joint_count_rejected(self):
        with self.assertRaises(ValueError):
            checked_actions(np.zeros((50, 55)))
        with self.assertRaises(ValueError):
            checked_actions(np.full((50, 14), np.nan))

    def test_boundaries_from_measured_chunk_ids(self):
        actions = np.zeros((6, 14), np.float32)
        actions[3:, ARM] = 2
        summary = boundary_summary(actions, [0, 0, 0, 1, 1, 1])
        self.assertEqual(summary["inside"], 0)
        self.assertEqual(summary["boundary"], 2)
        self.assertEqual(summary["boundary_count"], 1)
        self.assertIsNone(boundary_summary(actions, [0] * 6)["boundary"])

    def test_observation_hash_is_order_independent_and_pixel_sensitive(self):
        obs = {"image": np.zeros((2, 2, 3), np.uint8), "task": "lift pot"}
        self.assertEqual(observation_hash(obs), observation_hash(dict(reversed(list(obs.items())))))
        before = observation_hash(obs)
        obs["image"][0, 0, 0] = 1
        self.assertNotEqual(before, observation_hash(obs))


class EvaluatorTests(unittest.TestCase):
    def test_same_observation_shadow_and_real_execution_capture(self):
        spec = importlib.util.spec_from_file_location("diag_evaluate", HERE / "evaluate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sent = {"ours": [], "official": []}

        class FakeClient:
            def __init__(self, port):
                self.label = "ours" if port == 10000 else "official"
                self._ws = types.SimpleNamespace(close=lambda: None)

            def get_server_metadata(self):
                return {"diagnostic_token": "unit-test", "label": self.label, "horizon": 50,
                        "denoising_steps": 10}

            def infer(self, observation):
                if observation.get("reset"):
                    return {"action": None}
                sent[self.label].append(dict(observation))
                actions = np.tile(np.arange(50, dtype=np.float32)[:, None] * .01, (1, 14))
                if self.label == "ours":
                    actions[1::2, :] += .03
                names = [f"action.arm.position[{i}]" for i in range(12)] + [
                    f"action.effector.position[{i}]" for i in range(2)]
                return {"action": actions, "diagnostic_normalized_valid": actions,
                        "diagnostic_joint_names": names,
                        "diagnostic_seed": observation["_diagnostic_seed"], "server_timing": {}}

        class FakeEnv:
            def __init__(self):
                self.take_action_cnt = 0
                self.step_lim = 10
                self.eval_success = False
                self.q = np.zeros(14, dtype=float)
                self.robot = types.SimpleNamespace(
                    get_left_arm_real_jointState=lambda: self.q[:7].tolist(),
                    get_right_arm_real_jointState=lambda: self.q[7:].tolist(),
                    left_mplib_planner=types.SimpleNamespace(TOPP=self.topp),
                    right_mplib_planner=types.SimpleNamespace(TOPP=self.topp))

            def topp(self):
                return None, np.zeros((3, 6)), None, None, None

            def setup_demo(self, **kwargs):
                pass

            def play_once(self):
                pass

            def take_action(self, action):
                if self.take_action_cnt >= self.step_lim:
                    return
                self.robot.left_mplib_planner.TOPP()
                self.robot.right_mplib_planner.TOPP()
                self.q = np.asarray(action).copy()
                self.take_action_cnt += 1

        fake_evaluator = types.SimpleNamespace(class_decorator=lambda task: FakeEnv(),
                                               UnStableError=type("UnStableError", (Exception,), {}))

        def fake_main(config):
            env = fake_evaluator.class_decorator(config["task_name"])
            env.setup_demo(seed=100000)
            client = sys.modules["script.deploy.websocket_client_policy"].WebsocketClientPolicy()
            client.infer({"reset": True, "robo_name": "robotwin", "path_to_pi_model": None})
            while env.take_action_cnt < env.step_lim:
                obs = {"observation.state": env.q.copy(), "task": "lift pot",
                       "observation.images.cam_high": np.zeros((4, 4, 3), np.uint8)}
                response = client.infer(obs)
                for action in response["action"]:
                    env.take_action(action)

        fake_evaluator.main = fake_main
        fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda target: None))
        client_module = types.ModuleType("deploy.websocket_client_policy")
        client_module.WebsocketClientPolicy = FakeClient
        script_pkg = types.ModuleType("script")
        script_pkg.__path__ = []
        script_deploy_pkg = types.ModuleType("script.deploy")
        script_deploy_pkg.__path__ = []
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "ours"
            argv = ["evaluate.py", "--primary", "ours", "--ours-port", "10000",
                    "--official-port", "10001", "--token", "unit-test", "--output", str(output),
                    "--use-length", "3", "--max-actions", "4"]
            modules = {"deploy.websocket_client_policy": client_module,
                       "script": script_pkg, "script.deploy": script_deploy_pkg}
            import os
            previous_cwd = os.getcwd()
            try:
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                     patch.dict(os.environ, {"ROBOTWIN_DIR": tmp}), \
                     patch.object(module.importlib.util, "spec_from_file_location", return_value=fake_spec), \
                     patch.object(module.importlib.util, "module_from_spec", return_value=fake_evaluator):
                    module.main()
            finally:
                os.chdir(previous_cwd)
            requests = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
            actions = [json.loads(line) for line in (output / "executed.jsonl").read_text().splitlines()]
            self.assertEqual(len(actions), 4)
            self.assertEqual([row["chunk"] for row in actions], [0, 0, 0, 1])
            self.assertEqual(len(requests), 2)
            self.assertTrue(json.loads((output / "completion.json").read_text())["diagnostic_truncated"])
            self.assertEqual(actions[0]["actual_after"], actions[0]["action"])
            self.assertEqual(actions[0]["topp"]["left"]["points"], 3)
            for ours, official in zip(sent["ours"], sent["official"]):
                self.assertEqual(observation_hash(ours), observation_hash(official))
            with np.load(output / "predictions/0000.npz", allow_pickle=False) as data:
                self.assertEqual(data["ours"].shape, (50, 14))


class ServerWrapperTests(unittest.TestCase):
    def test_valid_dimensions_and_tensor_free_response(self):
        import torch
        spec = importlib.util.spec_from_file_location("diag_server", HERE / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "stats.json"
            stats_path.write_text('{"count": 548893}')
            init_overrides = []

            class FakePolicy:
                def __init__(self, *pos, **kw):
                    init_overrides.append(kw.get("robot_norm_path"))
                    self.robot_norm_path = str(stats_path)
                    self.config = types.SimpleNamespace(chunk_size=50, num_steps=10)
                    self.data_config = types.SimpleNamespace(img_size=256)
                    self.vla = types.SimpleNamespace(feature_transform=types.SimpleNamespace(
                        feature_config=types.SimpleNamespace(joints=["arm.position", "effector.position"]),
                        actions=["action.arm.position", "action.effector.position"],
                        normalizer=types.SimpleNamespace(norm_stats={
                            "action.arm.position": {"mean": torch.zeros(12)},
                            "action.effector.position": {"mean": torch.zeros(2)}})))

                def _unapply_batched_actions(self, applied, actions):
                    return {"action": actions[0, :, applied[0]["action_joint_mask"]].numpy()}

                def infer(self, obs, return_normalized=False):
                    if obs.get("reset"):
                        return {"action": None}
                    # Two padded dimensions deliberately have huge values.
                    values = torch.randn(1, 50, 16)
                    values[:, :, 12:14] = 100000
                    mask = torch.ones(16, dtype=torch.bool)
                    mask[12:14] = False
                    result = self._unapply_batched_actions([{"action_joint_mask": mask}], values)
                    if return_normalized:
                        result["_normalized_actions"] = values[0]
                    return result

            class FakeWebsocketServer:
                def __init__(self, policy, **kwargs):
                    self.policy = policy

                def serve_forever(self):
                    self.policy.infer({"reset": True})
                    a = self.policy.infer({"_diagnostic_seed": 42})
                    b = self.policy.infer({"_diagnostic_seed": 42})
                    np.testing.assert_array_equal(a["action"], b["action"])
                    self_test.assertEqual(a["diagnostic_normalized_valid"].shape, (50, 14))
                    self_test.assertLess(np.abs(a["diagnostic_normalized_valid"]).max(), 100000)
                    self_test.assertEqual(len(a["diagnostic_joint_names"]), 14)
                    self_test.assertFalse(any(isinstance(v, torch.Tensor) for v in a.values()))

            self_test = self
            policy_module = types.ModuleType("deploy.lingbot_vla_v2_policy")
            policy_module.LingbotVLAv2Server = FakePolicy
            policy_module.set_seed_everywhere = torch.manual_seed
            ws_module = types.ModuleType("deploy.websocket_policy_server")
            ws_module.WebsocketPolicyServer = FakeWebsocketServer
            norm_module = types.ModuleType("lingbotvla.utils.normalization_contract")
            norm_module.semantic_hash = lambda data: "unit-test-hash"
            argv = ["server.py", "--model", tmp, "--label", "ours", "--port", "10000",
                    "--token", "unit-test", "--manifest", str(Path(tmp) / "manifest.json"),
                    "--norm-stats", str(stats_path)]
            with patch.dict(sys.modules, {policy_module.__name__: policy_module,
                                          ws_module.__name__: ws_module, norm_module.__name__: norm_module}), \
                 patch.object(sys, "argv", argv):
                module.main()
            self.assertEqual(init_overrides, [str(stats_path.resolve())])


class NormalizationPreflightTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("diag_run", HERE / "run.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "run"
        self.hf = self.run / "checkpoints/global_step_18000/hf_ckpt"
        self.hf.mkdir(parents=True)
        self.stats = {"count": 6062592, "norm_stats": {}}
        for prefix in ("action", "observation.state"):
            for joint, dim in (("arm.position", 12), ("effector.position", 2)):
                self.stats["norm_stats"][f"{prefix}.{joint}"] = {
                    name: [value] * dim for name, value in (
                        ("mean", 0), ("std", 1), ("min", -1), ("max", 1),
                        ("q01", -.9), ("q99", .9), ("q02", -.8), ("q98", .8))}
        self.original = Path(self.tmp.name) / "official_original.json"
        self.original.write_text(json.dumps(self.stats))
        self.write_config(False)

    def write_config(self, required):
        (self.run / "lingbotvla_cli.yaml").write_text(json.dumps({
            "data": {"require_normalization_contract": required, "norm_stats_file": None}}))

    def snapshot(self):
        from lingbotvla.utils.normalization_contract import semantic_hash
        folder = self.run / "normalization"
        folder.mkdir()
        (folder / "norm_stats.json").write_text(json.dumps(self.stats))
        (folder / "manifest.json").write_text(json.dumps({
            "file": "norm_stats.json", "semantic_sha256": semantic_hash(self.stats)}))
        return folder / "norm_stats.json"

    def test_legacy_null_path_fails_before_gpu_startup(self):
        with self.assertRaisesRegex(ValueError, "Legacy normalization file missing"):
            self.module.normalization_preflight(self.hf)

    def test_legacy_explicit_stats_resolve_without_editing_config(self):
        config_before = (self.run / "lingbotvla_cli.yaml").read_bytes()
        path, metadata = self.module.normalization_preflight(self.hf, self.original)
        self.assertEqual(path, self.original.resolve())
        self.assertEqual(metadata["count"], 6062592)
        self.assertEqual((self.run / "lingbotvla_cli.yaml").read_bytes(), config_before)

    def test_new_contract_still_uses_snapshot_and_rejects_wrong_override(self):
        self.write_config(True)
        snapshot = self.snapshot()
        self.assertEqual(self.module.normalization_preflight(self.hf)[0], snapshot.resolve())
        other = json.loads(json.dumps(self.stats))
        other["norm_stats"]["action.arm.position"]["mean"][0] = .5
        self.original.write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError, "override disagrees"):
            self.module.normalization_preflight(self.hf, self.original)

    def test_required_snapshot_still_refuses_legacy_fallback(self):
        self.write_config(True)
        with self.assertRaisesRegex(ValueError, "no normalization snapshot"):
            self.module.normalization_preflight(self.hf, self.original)


if __name__ == "__main__":
    unittest.main()
