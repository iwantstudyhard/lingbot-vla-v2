"""AutoHorizon contracts using production methods without importing heavyweight model dependencies."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import einops
import numpy as np
import torch
import torch.nn as nn
import yaml

from deploy.eval_diagnostics import InvalidActionError
from deploy.robotwin_evaluation import validate_execution_horizon


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "lingbotvla/models/vla/lingbot_vla"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


auto = load_module("autohorizon_under_test", MODEL / "autohorizon.py")
upstream = load_module("autohorizon_reference", ROOT / "tests/fixtures/autohorizon_upstream.py")


def production_namespace():
    return {"torch": torch, "nn": nn, "F": torch.nn.functional, "einops": einops, "np": np, "math": math, "time": time,
            "yaml": yaml, "workspace_path": lambda value: ROOT / value,
            "AutoHorizonConfig": auto.AutoHorizonConfig, "estimate_horizon": auto.estimate_horizon}


def load_production(path, name, methods=None, namespace=None):
    namespace = namespace if namespace is not None else production_namespace()
    node = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body if getattr(node, "name", None) == name)
    if isinstance(node, ast.ClassDef):
        node.bases = [ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())]
        node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


namespace = production_namespace()
for function in ("our_eager_attention_forward", "make_att_2d_masks", "create_sinusoidal_pos_embedding"):
    load_production(MODEL / "utils.py", function, namespace=namespace)
Expert = load_production(MODEL / "modeling_lingbot_vla_v2.py", "QwenvlWithExpertV2Model",
                         ("forward", "handle_kv_cache", "_apply_deepstack"), namespace)
Flow = load_production(MODEL / "modeling_lingbot_vla_v2.py", "FlowMatchingV2",
                       ("sample_actions", "predict_velocity", "_build_full_position_ids"), namespace)
Flow.embed_suffix = load_production(MODEL / "modeling_lingbot_vla.py", "FlowMatching", ("embed_suffix",), namespace).embed_suffix
Policy = load_production(ROOT / "deploy/lingbot_vla_v2_policy.py", "LingbotVLAv2Server",
                         ("__init__", "reset", "_pad_and_stack_tensors", "_infer_batch", "_unapply_batched_actions",
                          "infer", "infer_with_diagnostics"), namespace)
Preprocess = load_production(ROOT / "deploy/lingbot_vla_v2_policy.py", "PolicyPreprocessMixin",
                             ("_to_device_image_grid_thw", "sample_actions_batch"), namespace)


class AlgorithmTests(unittest.TestCase):
    def test_matches_locked_upstream(self):
        generator = torch.Generator().manual_seed(72)
        for length in (2, 6, 10, 50):
            matrices = [torch.ones(length, length), torch.eye(length),
                        torch.rand(length, length, generator=generator),
                        torch.nn.functional.one_hot(torch.zeros(length, dtype=torch.long), length).float()]
            for matrix in matrices:
                for hold, entropy in ((0.3, 0.9), (0.1, 0.5), (0.5, 1.0)):
                    for name in ("pick_horizon_softpointer", "bidir_soft_pointer"):
                        with self.subTest(length=length, hold=hold, entropy=entropy, function=name):
                            expected, reference = getattr(upstream, name)(matrix, hold_thr=hold, max_entropy_q=entropy)
                            actual, diagnostics = getattr(auto, name)(matrix, hold_thr=hold, max_entropy_q=entropy)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            for key in reference:
                                if isinstance(reference[key], torch.Tensor):
                                    torch.testing.assert_close(diagnostics[key], reference[key], rtol=0, atol=0)
                                else:
                                    self.assertEqual(diagnostics[key], reference[key])

    def test_preserves_backward_coordinate_behavior(self):
        matrix = torch.zeros(6, 6)
        matrix[:3, 0], matrix[3:, 5] = 1, 1
        horizon, diagnostic = auto.bidir_soft_pointer(matrix)
        self.assertEqual((horizon.item(), diagnostic["N_forward"], diagnostic["N_backward"]), (6, 1, 6))
        self.assertIsNotNone(diagnostic["join_row"])

    def test_invalid_attention_and_single_action(self):
        config = auto.AutoHorizonConfig("auto")
        self.assertEqual(auto.estimate_horizon(torch.ones(1, 1), config)[0], 1)
        for matrix, reason in ((torch.zeros(6, 6), "invalid_attention_mass"),
                               (torch.full((6, 6), float("nan")), "nonfinite_attention"),
                               (-torch.ones(6, 6), "invalid_attention_mass"),
                               (torch.ones(6, 5), "invalid_attention_shape")):
            horizon, diagnostic = auto.estimate_horizon(matrix, config)
            self.assertIsNone(horizon)
            self.assertEqual(diagnostic["fallback_reason"], reason)

    def test_configuration_validation(self):
        defaults = dict(chunk_ret=True, use_length=3, predicted_horizon=6, num_steps=4)
        auto.AutoHorizonConfig("auto").validate(**defaults)
        for config in (auto.AutoHorizonConfig("bad"), auto.AutoHorizonConfig("auto", 0),
                       auto.AutoHorizonConfig("auto", 5), auto.AutoHorizonConfig("auto", hold_thr=float("nan")),
                       auto.AutoHorizonConfig("auto", max_entropy_q=1.1)):
            with self.assertRaises(ValueError):
                config.validate(**defaults)
        for values in ({"chunk_ret": False}, {"use_length": 0}, {"use_length": 7}):
            with self.assertRaises(ValueError):
                auto.AutoHorizonConfig("auto").validate(**(defaults | values))


class Decoder(nn.Module):
    """Small deterministic decoder around the real attention/cache/denoising code."""
    def __init__(self, expert, device):
        super().__init__()
        self.expert = expert
        self.query = nn.Linear(8, 8, bias=False).to(device)
        self.key = nn.Linear(8, 4, bias=False).to(device)
        self.value = nn.Linear(8, 4, bias=False).to(device)
        self.output = nn.Linear(8, 8, bias=False).to(device)

    def forward(self, hidden, attention=None, start=0, end=0, compute_kqv=False, **kwargs):
        if compute_kqv:
            hidden = hidden.to(self.query.weight.dtype)
            return (self.query(hidden).reshape(*hidden.shape[:2], 4, 2),
                    self.key(hidden).reshape(*hidden.shape[:2], 2, 2),
                    self.value(hidden).reshape(*hidden.shape[:2], 2, 2))
        output = hidden + self.output(attention[:, start:end].to(self.output.weight.dtype))
        return (output, torch.ones(1, device=hidden.device)) if self.expert else output


def small_flow(device="cpu", batch=1):
    torch.manual_seed(13)
    expert = Expert()
    expert.config = SimpleNamespace(attention_implementation="eager", final_norm_adanorm=False,
                                   qwen_expert_config=SimpleNamespace(num_hidden_layers=2))
    language = SimpleNamespace(layers=[Decoder(False, device) for _ in range(2)], norm=nn.Identity())
    action = SimpleNamespace(layers=[Decoder(True, device) for _ in range(2)], norm=nn.Identity())
    expert.qwenvl = SimpleNamespace(model=SimpleNamespace(language_model=language),
                                   config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2)))
    expert.qwen_expert = SimpleNamespace(model=action)
    expert.attention_interface = namespace["our_eager_attention_forward"]
    # A deterministic positional transform ensures capture occurs after position handling.
    expert.apply_mrope = lambda q, k, ids: (q * 0.9, k * 0.9)
    flow = Flow()
    flow.config = SimpleNamespace(n_action_steps=6, max_action_dim=8, num_steps=4, use_cache=True, proj_width=8)
    flow.qwenvl_with_expert = expert
    flow.state_proj = nn.Linear(8, 8).to(device)
    flow.action_in_proj = nn.Linear(8, 8).to(device)
    flow.action_time_mlp_in = nn.Linear(16, 8).to(device)
    flow.action_time_mlp_out = nn.Linear(8, 8).to(device)
    flow.action_out_proj = nn.Linear(8, 8).to(device)
    flow.block_future_depth_to_action = False
    flow._block_suffix_to_future_video_if_enabled_ = lambda mask, **kwargs: mask
    prefix = torch.randn(batch, 3, 8, device=device)
    masks = torch.ones(batch, 3, dtype=torch.bool, device=device)
    positions = torch.arange(3, device=device).expand(3, batch, 3)
    flow.embed_prefix = lambda *args, **kwargs: (prefix, masks, torch.zeros_like(masks), positions, None, None)
    return flow


class ModelCaptureTests(unittest.TestCase):
    def test_attention_preserves_output_and_excludes_state_with_gqa(self):
        torch.manual_seed(17)
        query, key, value = torch.randn(2, 7, 4, 2), torch.randn(2, 11, 2, 2), torch.randn(2, 11, 2, 2)
        mask = torch.ones(2, 7, 11, dtype=torch.bool)
        mask[:, :, 1] = False
        eager = namespace["our_eager_attention_forward"]
        output = eager(query, key, value, mask)
        actual, attention = eager(query, key, value, mask, action_attention_size=6)
        torch.testing.assert_close(actual, output, rtol=0, atol=0)
        expanded = key.repeat_interleave(2, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, expanded) * 2 ** -0.5
        scores = torch.where(mask[:, None], scores, -2.3819763e38)
        expected = scores.softmax(-1)[:, :, -6:, -6:].mean(1)
        torch.testing.assert_close(attention, expected, rtol=0, atol=0)
        self.assertEqual(attention.shape, (2, 6, 6))
        self.assertEqual(attention.dtype, torch.float32)
        self.assertIsNone(attention._base)

    def assert_sampling_parity(self, device, dtype=torch.float32):
        flow = small_flow(device, batch=2)
        flow.to(dtype=dtype)
        state = torch.randn(2, 8, device=device, dtype=dtype)
        noise = torch.randn(2, 6, 8, device=device, dtype=dtype)
        with torch.no_grad():
            plain = flow.sample_actions(None, None, None, None, state, noise=noise.clone())
            rng = torch.get_rng_state().clone()
            captured = []
            eager = flow.qwenvl_with_expert.attention_interface
            def recording(*args, **kwargs):
                result = eager(*args, **kwargs)
                if kwargs.get("action_attention_size"):
                    captured.append(result[1].clone())
                return result
            flow.qwenvl_with_expert.attention_interface = recording
            actual, summary = flow.sample_actions(None, None, None, None, state, noise=noise.clone(), attention_step=3)
        torch.testing.assert_close(actual, plain, rtol=0, atol=0)
        torch.testing.assert_close(summary, torch.stack(captured).mean(0) / 4, rtol=0, atol=0)
        self.assertEqual(len(captured), 2)  # two layers, one denoising step; no prefix capture
        self.assertEqual(summary.shape, (2, 6, 6))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        return flow

    def test_cpu_full_denoising_parity(self):
        self.assert_sampling_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_full_denoising_parity_fp32_and_bf16(self):
        self.assert_sampling_parity("cuda")
        self.assert_sampling_parity("cuda", torch.bfloat16)

    def test_capture_guards(self):
        flow = small_flow()
        for step in (0, 5):
            with self.assertRaises(ValueError):
                flow.sample_actions(None, None, None, None, torch.ones(1, 8), attention_step=step)
        flow._use_compile_predict_velocity = True
        with self.assertRaisesRegex(ValueError, "non-compiled"):
            flow.sample_actions(None, None, None, None, torch.ones(1, 8), attention_step=3)
        flow.qwenvl_with_expert.config.attention_implementation = "flex"
        with self.assertRaisesRegex(ValueError, "eager"):
            flow.qwenvl_with_expert.forward(action_attention_size=6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_preprocessing_wrapper_keeps_actions_and_cuda_rng(self):
        flow = small_flow("cuda")
        wrapper = Preprocess()
        observation = {"images": torch.zeros(1, 1, 3, 2, 2), "img_masks": torch.ones(1, 1, dtype=torch.bool),
                       "lang_tokens": torch.ones(1, 2, dtype=torch.long), "lang_masks": torch.ones(1, 2, dtype=torch.bool),
                       "state": torch.ones(1, 8)}
        torch.cuda.manual_seed(42)
        plain = wrapper.sample_actions_batch(dict(observation), sample_compile_fn=flow.sample_actions)
        rng = torch.cuda.get_rng_state().clone()
        torch.cuda.manual_seed(42)
        observed, attention = wrapper.sample_actions_batch(dict(observation), sample_compile_fn=flow.sample_actions,
                                                           attention_step=3)
        torch.testing.assert_close(observed, plain, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
        self.assertEqual(observed.device.type, "cpu")
        self.assertEqual(attention.device.type, "cuda")


def policy_fixture(mode="auto", attention=None):
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.horizon_config = auto.AutoHorizonConfig(mode)
    policy.chunk_ret, policy.use_length, policy.use_bf16, policy.use_compile = True, 3, False, False
    policy.global_step = 0
    policy.last_action_chunk = policy.last_normalized_action_chunk = policy._last_full_diagnostics = policy._last_horizon = None
    policy.action_key = ["action", "other_action"]
    mask = torch.ones(8, dtype=torch.bool)
    policy._prepare_model_input = lambda obs: {"state": torch.ones(8), "state_joint_mask": mask, "action_joint_mask": mask,
                                             "lang_tokens": torch.ones(2, dtype=torch.long), "lang_masks": torch.ones(2, dtype=torch.bool)}
    policy.vla = SimpleNamespace(feature_transform=SimpleNamespace(unapply=lambda item: {
        "action": item["actions"][:, :4] * 2, "other_action": item["actions"][:, 4:] * 3,
    }))
    calls = []
    def sample(*args, **kwargs):
        calls.append(kwargs)
        actions = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)
        matrix = torch.eye(6)[None] if attention is None else attention
        return (actions, matrix) if "attention_step" in kwargs else actions
    policy.vla.sample_actions_batch = policy.sample_actions_fn = sample
    return policy, calls


class PolicyTests(unittest.TestCase):
    def test_modes_and_full_diagnostics(self):
        observation = {"observation.state": np.zeros(8)}
        for mode, length in (("fixed", 3), ("observe", 3), ("auto", 6)):
            with self.subTest(mode=mode):
                policy, calls = policy_fixture(mode)
                result = policy.infer_with_diagnostics(observation)
                self.assertEqual(result["action"].shape, (length, 4))
                self.assertEqual(result["other_action"].shape, (length, 4))
                arrays = result["_diagnostics"]["arrays"]
                self.assertEqual(arrays["full_action"].shape, (1, 6, 4))
                self.assertEqual(arrays["normalized_actions"].shape, (1, 6, 8))
                self.assertEqual(len(calls), 1)
                if mode == "fixed":
                    self.assertNotIn("execution_horizon", result)
                    self.assertNotIn("attention_step", calls[0])
                else:
                    self.assertEqual(result["execution_horizon"], length)
                    self.assertEqual(result["estimated_execution_horizon"], 6)
                    self.assertEqual(arrays["action_attention"].shape, (1, 6, 6))

    def test_fallback_and_invalid_shape(self):
        for matrix in (torch.zeros(1, 6, 6), torch.full((1, 6, 6), float("nan")), torch.ones(1, 6, 5)):
            policy, _ = policy_fixture(attention=matrix)
            result = policy.infer({"observation.state": np.zeros(8)}, return_normalized=True)
            self.assertEqual(result["horizon_method"], "fixed_fallback")
            self.assertIsNone(result["estimated_execution_horizon"])
            self.assertEqual(result["action"].shape[0], 3)
            self.assertEqual(result["_normalized_actions"].shape[0], 3)

    def test_short_and_changing_horizons(self):
        attention = torch.zeros(1, 6, 6)
        attention[:, :, 0] = 1
        policy, calls = policy_fixture(attention=attention)
        first = policy.infer({})
        self.assertEqual((first["execution_horizon"], first["action"].shape[0]), (1, 1))
        sample = policy.vla.sample_actions_batch
        policy.vla.sample_actions_batch = lambda *args, **kwargs: (sample(*args, **kwargs)[0], torch.eye(6)[None])
        second = policy.infer({})
        self.assertEqual((second["execution_horizon"], second["action"].shape[0]), (6, 6))
        self.assertEqual(len(calls), 2)
        self.assertEqual(policy.global_step, 2)

    def test_batch_rejection_and_model_errors_propagate(self):
        policy, calls = policy_fixture()
        with self.assertRaisesRegex(ValueError, "one observation"):
            policy.infer({"batch": [{}, {}]})
        self.assertEqual(calls, [])
        policy.vla.sample_actions_batch = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed"))
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            policy.infer({})

    def test_reset_clears_prediction_state(self):
        policy, _ = policy_fixture()
        policy.infer({})
        self.assertIsNotNone(policy._last_horizon)
        policy.config = SimpleNamespace(chunk_size=6)
        policy.processor, policy.data_config, policy.robot_norm_path = None, None, None
        namespace["FeatureTransform"] = lambda *args, **kwargs: SimpleNamespace(org_features={"actions": ["action"]})
        self.assertEqual(policy.infer({"reset": True, "robo_name": "robotwin"}), {"action": None})
        self.assertEqual(policy.global_step, 0)
        self.assertIsNone(policy._last_horizon)
        self.assertIsNone(policy.last_action_chunk)
        self.assertIsNone(policy._last_full_diagnostics)

    def test_constructor_forces_noncompiled_path(self):
        namespace["apply_lingbot_qwen3_vl_patch"] = lambda: None
        vla = SimpleNamespace(model=SimpleNamespace(float=lambda: None))
        vla.cuda = vla.eval = lambda: vla
        with patch.object(Policy, "load_vla", create=True, return_value=vla):
            policy = Policy(use_bf16=False, use_compile=True, horizon_mode="auto")
        self.assertTrue(policy.requested_use_compile)
        self.assertFalse(policy.use_compile)

    def test_declared_length_validation(self):
        validate_execution_horizon({"action": np.zeros((3, 14)), "execution_horizon": 3, "predicted_horizon": 6})
        validate_execution_horizon({"action": np.zeros(14)})
        for horizon in (0, 2, 7, True, 3.0):
            with self.assertRaises(InvalidActionError):
                validate_execution_horizon({"action": np.zeros((3, 14)), "execution_horizon": horizon, "predicted_horizon": 6})


if __name__ == "__main__":
    unittest.main()
