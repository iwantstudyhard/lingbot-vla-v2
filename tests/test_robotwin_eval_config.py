"""CPU regression for launcher/evaluator config resolution; no simulator import."""
import argparse
import ast
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'experiment/robotwin/deploy_policy.yml'


class EvalConfigTests(unittest.TestCase):
    def parser(self, workspace):
        tree = ast.parse((ROOT / 'experiment/robotwin/eval_policy_client_lingbotvla.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args_and_config')
        scope = dict(argparse=argparse, os=os, Path=Path, yaml=yaml, WORKSPACE=workspace)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<real_config_parser>', 'exec'), scope)
        return scope['parse_args_and_config']

    def test_absolute_config_with_distinct_workspace_and_simulator_cwd(self):
        with TemporaryDirectory(prefix='eval config ') as tmp:
            workspace = Path(tmp) / 'workspace'
            simulator = Path(tmp) / 'RoboTwin'
            workspace.mkdir(); simulator.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(simulator)
                argv = ['eval.py', '--config', str(CONFIG), '--overrides',
                        '--task_name', 'lift_pot', '--num_episodes', '1',
                        '--action_smoothing', 'ema', '--smoothing_alpha', '0.35',
                        '--port', '19640', '--eval_video_log', 'True']
                with patch.object(sys, 'argv', argv):
                    config = self.parser(workspace)()
                self.assertEqual(config['task_name'], 'lift_pot')
                self.assertEqual(config['robo_name'], 'robotwin')
                self.assertEqual(config['policy_name'], 'ACT')
                self.assertEqual(config['num_episodes'], 1)
                self.assertEqual(config['port'], 19640)
                self.assertEqual(config['action_smoothing'], 'ema')
                self.assertEqual(config['smoothing_alpha'], .35)
                self.assertIs(config['eval_video_log'], True)
            finally:
                os.chdir(previous)

    def test_relative_config_remains_workspace_relative_for_manual_usage(self):
        argv = ['eval.py', '--config', 'experiment/robotwin/deploy_policy.yml']
        with patch.object(sys, 'argv', argv):
            config = self.parser(ROOT)()
        self.assertEqual(config['action_smoothing'], 'none')

    def test_launcher_checks_and_passes_maintained_absolute_config_before_gpu_start(self):
        source = (ROOT / 'experiment/robotwin/start_robotwin_infer_and_eval.sh').read_text(encoding='utf-8')
        self.assertIn('eval_policy_config="${inference_workdir%/}/experiment/robotwin/deploy_policy.yml"', source)
        self.assertIn("--config '${eval_policy_config}'", source)
        self.assertNotIn('--config policy/${policy_name}/deploy_policy.yml', source)
        self.assertLess(source.index('if [ ! -f "$eval_policy_config" ]'), source.index('# Phase 1: start inference-side'))


if __name__ == '__main__':
    unittest.main()
