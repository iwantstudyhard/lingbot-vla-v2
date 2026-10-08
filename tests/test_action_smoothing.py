"""CPU checks only: fixtures establish filtering behavior, not task success."""
import ast
import importlib.util
from pathlib import Path
import unittest
import sys
import os
import json
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("action_smoothing_tested", ROOT / "deploy/action_smoothing.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Smoother, ARM = module.RoboTwinActionSmoother, module.ARM_INDICES


class SmoothingTests(unittest.TestCase):
    def test_default_is_exact_identity_and_does_not_mutate(self):
        raw = np.arange(70, dtype=np.float32).reshape(5, 14)
        before = raw.copy()
        smoother = Smoother()
        prepared = smoother.prepare_chunk(raw)
        result = np.array([smoother.filter_action(row) for row in prepared])
        np.testing.assert_array_equal(result, raw)
        self.assertEqual(result.dtype, raw.dtype)
        np.testing.assert_array_equal(raw, before)

    def test_only_arm_joints_are_smoothed_grippers_exact(self):
        raw = np.tile(np.arange(7, dtype=np.float64)[:, None], (1, 14))
        raw[:, 6] = [0, 1, 0, 1, 0, 1, 0]
        raw[:, 13] = [1, 0, 1, 0, 1, 0, 1]
        before = raw.copy()
        smoother = Smoother('ema')
        smoother.reset(np.zeros(14))
        prepared = smoother.prepare_chunk(raw)
        result = np.array([smoother.filter_action(row) for row in prepared])
        np.testing.assert_array_equal(result[:, [6, 13]], raw[:, [6, 13]])
        np.testing.assert_array_equal(raw, before)
        self.assertLessEqual(np.abs(np.diff(np.vstack([np.zeros(14), result])[:, ARM], axis=0)).max(), .0500000001)

    def test_ema_persists_across_chunks_and_reset_removes_previous_episode(self):
        smoother = Smoother('ema', alpha=.5, window=1, max_delta=0)
        smoother.reset(np.zeros(14))
        first = smoother.filter_action(smoother.prepare_chunk(np.ones((1, 14)))[0])
        second = smoother.filter_action(smoother.prepare_chunk(np.ones((1, 14)))[0])
        np.testing.assert_allclose(first[ARM], .5)
        np.testing.assert_allclose(second[ARM], .75)
        smoother.reset(np.ones(14)*2)
        np.testing.assert_allclose(smoother.filter_action(np.ones(14))[ARM], 1.5)

    def test_constant_targets_unchanged_and_linear_interior_preserved(self):
        smoother = Smoother('ema', alpha=1, window=5, max_delta=0)
        smoother.reset(np.ones(14)*3)
        constant = np.ones((7, 14))*3
        np.testing.assert_allclose(smoother.prepare_chunk(constant), constant)
        np.testing.assert_allclose(smoother.filter_action(constant[0]), constant[0])
        ramp = np.tile(np.arange(9.)[:, None], (1, 14))
        np.testing.assert_allclose(smoother.prepare_chunk(ramp)[2:-2], ramp[2:-2])

    def test_high_frequency_noise_is_reduced(self):
        raw = np.zeros((40, 14)); raw[::2, ARM] = .1; raw[1::2, ARM] = -.1
        smoother = Smoother('ema', max_delta=0)
        smoother.reset(np.zeros(14))
        result = np.array([smoother.filter_action(row) for row in smoother.prepare_chunk(raw)])
        self.assertLess(np.abs(np.diff(result[:, ARM], axis=0)).mean(), .02)

    def test_invalid_inputs_fail_closed(self):
        for options in [dict(mode='bad'), dict(alpha=0), dict(alpha=float('nan')), dict(alpha=1.1),
                        dict(window=2), dict(window=0), dict(window=53), dict(window=1.5),
                        dict(max_delta=-1), dict(max_delta=float('inf'))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                Smoother(**options)
        smoother = Smoother('ema')
        for raw in [np.zeros((4, 55)), np.full((4, 14), np.nan), np.zeros((0, 14))]:
            with self.assertRaises(ValueError): smoother.prepare_chunk(raw)
        with self.assertRaises(ValueError): smoother.reset(np.zeros(55))
        with self.assertRaises(RuntimeError): smoother.filter_action(np.zeros(14))

    def test_evaluator_applies_filters_resets_and_writes_raw_and_command(self):
        # Compile just the eval function: do not import simulator dependencies.
        source = (ROOT / 'experiment/robotwin/eval_policy_client_lingbotvla.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'eval_policy')
        compile(ast.Module(body=[function], type_ignores=[]), '<evaluator>', 'exec')
        self.assertIn('smoother.reset(observation["joint_action"]["vector"])', source)
        self.assertIn('TASK_ENV.take_action(filtered)', source)
        self.assertIn('raw_action=', source)
        self.assertIn('executed_action=', source)
        self.assertNotIn('TASK_ENV.take_action(act)', source)

    def test_launcher_forwards_options_and_syncs_helper(self):
        source = (ROOT / 'experiment/robotwin/start_robotwin_infer_and_eval.sh').read_text(encoding='utf-8')
        for option in ['action_smoothing', 'smoothing_alpha', 'smoothing_window', 'smoothing_max_delta']:
            self.assertIn('--'+option+' ${'+option+'}', source)
        self.assertIn('msgpack_numpy.py action_smoothing.py', source)
        self.assertIn('action_smoothing=none', source)
        self.assertNotIn('gpu_id=$(( slot % num_gpus ))', source)

    def test_real_eval_function_enabled_and_disabled_without_video(self):
        source = (ROOT / 'experiment/robotwin/eval_policy_client_lingbotvla.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'eval_policy')
        scope = dict(np=np, Path=Path, os=os, sys=sys, json=json, WORKSPACE=ROOT,
                     UnStableError=type('UnStableError', (Exception,), {}),
                     generate_episode_descriptions=lambda *args: [{'test': ['hang a mug']}])
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<real_eval>', 'exec'), scope)

        class Env:
            eval_video_path = None
            render_freq = 0
            plan_success = True
            step_lim = 6

            def setup_demo(self, **kwargs):
                self.take_action_cnt = 0
                self.eval_success = False
                self.q = np.ones(14) * .2
                self.commands = []

            def play_once(self): return {'info': {}}
            def close_env(self, **kwargs): pass
            def check_success(self): return True
            def set_instruction(self, instruction): self.instruction = instruction
            def get_instruction(self): return self.instruction

            def get_obs(self):
                return {'joint_action': {'vector': self.q.copy()},
                        'observation': {key: {'rgb': np.zeros((2, 2, 3), np.uint8)}
                                        for key in ['head_camera', 'left_camera', 'right_camera']}}

            def take_action(self, command):
                self.commands.append(np.asarray(command).copy())
                self.q = np.asarray(command).copy()
                self.take_action_cnt += 1
                self.eval_success = self.take_action_cnt == 6

        raw = np.tile(np.array([.26, .04, .26])[:, None], (1, 14))
        raw[:, [6, 13]] = [[0, 1], [1, 0], [0, 1]]

        class Client:
            def infer(self, obs):
                if obs.get('reset'): return {}
                return {'action': raw.copy(), 'server_timing': {}}

        for mode in ['none', 'ema']:
            with self.subTest(mode=mode), TemporaryDirectory() as tmp, patch.object(sys, 'path', sys.path.copy()):
                env = Env()
                args = dict(task_name='hanging_mug', policy_name='ACT', render_freq=0,
                            clear_cache_freq=10, task_config='demo_clean', ckpt_setting='test')
                opts = dict(robo_name='robotwin', output_dir=tmp, action_smoothing=mode)
                _, successes = scope['eval_policy']('hanging_mug', env, args, Client(), 100000,
                                                    test_num=1, instruction_type='test', usr_args=opts)
                self.assertEqual(successes, 1)  # Fixture termination, not real task evidence.
                expected = Smoother(mode)
                expected.reset(np.ones(14) * .2)
                commands = [expected.filter_action(row) for _ in range(2) for row in expected.prepare_chunk(raw)]
                np.testing.assert_allclose(env.commands, commands)
                trace_path = Path(tmp) / 'hanging_mug/action_smoothing.jsonl'
                self.assertEqual(trace_path.exists(), mode == 'ema')
                if mode == 'ema':
                    entries = [json.loads(line) for line in trace_path.read_text().splitlines()]
                    self.assertEqual(entries[0]['event'], 'episode_start')
                    self.assertEqual([r['action_index'] for r in entries[1:]], list(range(6)))
                    np.testing.assert_allclose([r['executed_action'] for r in entries[1:]], commands)
                    np.testing.assert_array_equal([r['raw_action'] for r in entries[1:]], np.tile(raw, (2, 1)))


if __name__ == '__main__':
    unittest.main()
