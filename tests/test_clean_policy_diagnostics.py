"""CPU regression tests; passing these does NOT certify the server's CUDA kernels."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.clean_policy.common import ARMS, action_chunk, action_metrics, vector_metrics, dataset_path, verify_selected_sources
from diagnostics.clean_policy.paths import forward_mode, full_velocity
from diagnostics.clean_policy.run import discover_models
from diagnostics.clean_policy.training_audit import audit, scheduled_lr
from diagnostics.clean_policy.report import build_report


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(attention_implementation='eager')
        self.attention_interface = 'original_interface'

    def get_attention_interface(self):
        return self.config.attention_implementation


class FakeFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True, align_params={'keep_task_tokens': True},
                                      sequence_wise_loss_coeff=.001, router_z_loss_coeff=.0001, action_fp32=False)
        self.qwenvl_with_expert = FakeBackbone()
        self.action_out_proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.action_out_proj.weight.copy_(torch.eye(2))
        self.register_buffer('tokens_per_expert', torch.tensor([3., 4.]))

    def forward(self, actions, noise, time, **inputs):
        self.tokens_per_expert.add_(1)
        assert not torch.is_grad_enabled()
        assert self.config.align_params == {}
        return self.action_out_proj(noise-actions)


class CleanPolicyDiagnosticsTests(unittest.TestCase):
    def test_tail_padding_cannot_hide_or_inflate_error(self):
        target = np.zeros((5, 14))
        pred = target.copy()
        pred[:3, ARMS] = .1
        pred[3:] = 1000
        result = action_metrics(pred, target, [False, False, False, True, True])
        self.assertAlmostEqual(result['full_arm_mae_rad'], .1)
        self.assertAlmostEqual(result['pred_d1_rad_per_action1'], 0)
        self.assertEqual(result['valid_actions'], 3)

    def test_arm_and_gripper_units_are_separate(self):
        pred, target = np.zeros((4, 14)), np.zeros((4, 14))
        pred[:, [6, 13]] = 1
        result = action_metrics(pred, target, np.zeros(4, bool))
        self.assertEqual(result['full_arm_mae_rad'], 0)
        self.assertEqual(result['full_gripper_mae_native'], 1)

    def test_smooth_wrong_action_is_not_good_prediction(self):
        target = np.zeros((50, 14))
        target[:, ARMS] = np.arange(50)[:, None] * .02
        pred = np.zeros_like(target)
        result = action_metrics(pred, target, np.zeros(50, bool))
        self.assertEqual(result['pred_d1_rad_per_action1'], 0)
        self.assertGreater(result['full_arm_mae_rad'], .4)

    def test_nonfinite_and_allpadding_rejected(self):
        x = np.zeros((3, 14))
        with self.assertRaises(ValueError):
            action_metrics(x, x, np.ones(3, bool))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            action_metrics(x, x, np.zeros(3, bool))

    def test_vector_masks_out_invalid_joints(self):
        a, b = np.zeros((1, 50, 55)), np.zeros((1, 50, 55))
        a[..., 14:] = 100
        valid = np.arange(55)[None, None, :] < 14
        self.assertEqual(vector_metrics(a, b, valid)['mae'], 0)

    def fixture_table(self):
        return pa.Table.from_pylist([dict(index=i, episode_index=2, frame_index=i-10, timestamp=(i-10)/15,
                        task_index=0, action=[float(i)]*14, **{'observation.state': [float(i)]*14}) for i in range(10, 14)])

    def test_chunk_starts_at_current_frame_and_does_not_cross_episode(self):
        first, state, target, pad = action_chunk(self.fixture_table(), dict(index=12, end=14, episode=2, offset=2), 5)
        np.testing.assert_equal(target[:, 0], [12, 13, 13, 13, 13])
        np.testing.assert_equal(pad, [False, False, True, True, True])
        self.assertEqual(first['frame_index'], 2)

    def test_missing_action_row_is_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            action_chunk(self.fixture_table().slice(0, 2), dict(index=10, end=14, episode=2, offset=0), 5)

    def test_forward_flags_buffers_and_config_restored_after_error(self):
        flow = FakeFlow().eval()
        flow.action_out_proj.train()  # Mixed child modes must survive.
        old_config, old_flags = vars(flow.config).copy(), [m.training for m in flow.modules()]
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            with forward_mode(flow, 'flex_cached', training=True):
                flow.tokens_per_expert.add_(7)
                raise RuntimeError('probe failed')
        self.assertEqual(vars(flow.config), old_config)
        self.assertEqual([m.training for m in flow.modules()], old_flags)
        torch.testing.assert_close(flow.tokens_per_expert, torch.tensor([3., 4.]))
        self.assertEqual(flow.qwenvl_with_expert.attention_interface, 'original_interface')

    def test_full_path_capture_uses_actual_forward_and_no_weight_changes(self):
        flow = FakeFlow().eval()
        weights = flow.action_out_proj.weight.detach().clone()
        actions, noise = torch.ones(1, 3, 2), torch.full((1, 3, 2), 2.)
        result = full_velocity(flow, {}, actions, noise, torch.tensor([.5]), training=True)
        torch.testing.assert_close(result, noise-actions)
        torch.testing.assert_close(flow.action_out_proj.weight, weights)
        torch.testing.assert_close(flow.tokens_per_expert, torch.tensor([3., 4.]))
        self.assertFalse(flow.action_out_proj._forward_hooks)

    def test_action_fp32_not_falsely_reported_as_measured(self):
        flow = FakeFlow()
        flow.config.action_fp32 = True
        with self.assertRaisesRegex(ValueError, 'action_fp32'):
            full_velocity(flow, {}, None, None, None)

    def test_discovery_is_sorted_and_rejects_duplicate_label(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (18000, 500, 17500):
                (root / f'checkpoints/global_step_{step}/hf_ckpt').mkdir(parents=True)
            self.assertEqual([x[0] for x in discover_models(root, [], None)], ['step_000500', 'step_017500', 'step_018000'])
            with self.assertRaises(ValueError):
                discover_models(root, [f'step_018000={root}'], None)
            with self.assertRaises(ValueError):
                discover_models(None, [f'../unsafe={root}'], None)

    def test_manifest_paths_with_spaces_supported(self):
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'train.txt'
            manifest.write_text('robotwin datasets/with spaces\n', encoding='utf-8')
            result = dataset_path({'data': {'train_path': str(manifest)}})
            self.assertEqual(result, ROOT / 'datasets/with spaces')

    def test_selected_source_hash_mismatch_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root/'report.json'
            report.write_text(json.dumps(dict(comparison_passed=True, source_fingerprint='clean',
                              data_sources=[dict(path='meta/info.json', sha256='good')])) )
            norm = root/'norm.json'
            norm.write_text(json.dumps(dict(verification=dict(source_fingerprint='clean'))))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_selected_sources(dict(dataset=str(root), info_sha256='wrong',
                                        episode_metadata_sha256={}, selected_data_sha256={}), norm, report)

    def test_lr_mirror_matches_actual_repository_scheduler(self):
        # Compile the real scheduler function without importing GPU optimizer deps.
        source = ast.parse((ROOT / 'lingbotvla/optim/lr_scheduler.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name=='get_cosine_schedule_with_warmup')
        namespace = {'math': math, 'LambdaLR': torch.optim.lr_scheduler.LambdaLR}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<real repo cosine schedule>', 'exec'), namespace)
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-5)
        actual = namespace[node.name](optimizer, num_warmup_steps=540, num_training_steps=18000, init_lr=1e-5, min_lr=1e-6)
        train = dict(max_steps=18000, lr=1e-5, lr_min=1e-6, lr_warmup_ratio=.03, lr_decay_style='cosine')
        for step in (0, 1, 539, 540, 5000, 17500, 18000, 19000):
            self.assertAlmostEqual(scheduled_lr(train, step), 1e-5 * actual.lr_lambdas[0](step), places=15)

    def test_actual_muon_builder_ignores_vit_lr_and_scales_experts(self):
        source = ast.parse((ROOT / 'lingbotvla/optim/optimizer.py').read_text(encoding='utf-8'))
        nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ('_split_param_groups_by_scaled_lr', 'build_muon_optimizer')]
        params, names = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(2)], ['model.qwenvl.visual.weight', 'model.layers.0.mlp.experts.gate_proj']
        def fake_optimizer(groups, **kwargs):
            return SimpleNamespace(param_groups=groups)
        namespace = dict(torch=torch, Tensor=torch.Tensor, Sequence=Sequence, Tuple=Tuple, Dict=Dict, List=List, Any=Any, re=re,
                         split_muon_adamw_params=lambda *a, **kw: (params, [], names, []),
                         DistributedMuon=fake_optimizer, AdamW=fake_optimizer,
                         CombinedOptimizer=lambda opts: opts, is_torch_npu_available=lambda: False)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual Muon grouping>', 'exec'), namespace)
        args = SimpleNamespace(use_moe=True, use_moe_expert_lr=True, token_moe_layers=[0], token_num_experts=32, token_top_k=4, vit_lr=1e-6)
        groups = namespace['build_muon_optimizer'](None, args, lr=1e-5)[0].param_groups
        self.assertEqual(groups[0]['lr'], 1e-5)
        self.assertIs(groups[0]['params'][0], params[0])
        self.assertAlmostEqual(groups[1]['lr'], 1e-5*math.sqrt(8))

    def test_partial_report_does_not_call_missing_worker_complete(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            plan = dict(models=[dict(label='ours')], data=dict(samples=[]), precision='bf16', noise_seeds=[42,43], denoising_steps=[10], interpretation='TRAIN only')
            (run / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
            build_report(run)
            summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['incomplete_models'], ['ours'])
            self.assertIn('未完成', (run / 'REPORT.md').read_text(encoding='utf-8'))

    def test_report_draws_curves_and_measured_summary(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            folder = run / 'ours/ep0000_f0000'
            folder.mkdir(parents=True)
            plan = dict(models=[dict(label='ours')], data=dict(samples=[dict(id='ep0000_f0000')]),
                        precision='bf16', noise_seeds=[42,43], denoising_steps=[10], interpretation='SYNTHETIC CPU TEST ONLY')
            (run / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
            target = np.zeros((50,14), np.float32)
            prediction = target.copy()
            prediction[:, ARMS] = np.sin(np.arange(50))[:,None] * .1
            np.savez_compressed(folder / 'expert.npz', target=target, pad=np.zeros(50,bool))
            np.savez_compressed(folder / 'prediction_steps10_seed42.npz', prediction=prediction)
            (folder / 'seed_variability_steps10.json').write_text(json.dumps(dict(first_action_arm_std_rad=.03)))
            row = dict(steps=10, seed=42, sample='ep0000_f0000', metrics=action_metrics(prediction,target,np.zeros(50,bool)))
            (run/'ours/sampling.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            (run/'ours/paths.jsonl').write_text(json.dumps(dict(sample='ep0000_f0000',time=.5,
                  comparisons=dict(cache_only=dict(mae=.001, max_abs=.003, relative_l2=.01)),errors={}))+'\n',encoding='utf-8')
            (run/'ours/completion.json').write_text('{}')
            build_report(run)
            summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(summary['sampling'][0]['first_action_seed_std_rad'], .03)
            from PIL import Image
            for path in (run/'checkpoint_comparison.png',run/'actions_ep0000_f0000.png', run/'forward_path_comparison.png'):
                with Image.open(path) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertGreater(np.asarray(image).std(), 1)

    def test_cpu_prepare_only_uses_real_contract_no_cuda_or_video(self):
        # Fake tiny HF file for preflight ONLY; never claimed loadable weights.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / 'run/checkpoints/global_step_18000/hf_ckpt'
            hf.mkdir(parents=True)
            (hf / 'config.json').write_text('{}')
            (hf / 'model.safetensors').write_bytes(b'preflight-only-not-weights')
            qwen = root / 'qwen'
            qwen.mkdir()
            (qwen / 'config.json').write_text('{}')
            config = yaml.safe_load((ROOT / 'configs/vla/robotwin/robotwin_clean_stage1.yaml').read_text(encoding='utf-8'))
            (root / 'run/lingbotvla_cli.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            from lingbotvla.utils.normalization_contract import semantic_hash
            norm = json.loads((ROOT / 'assets/norm_stats/robotwin_clean_verified.json').read_bytes())
            folder = root / 'run/normalization'
            folder.mkdir()
            (folder / 'norm_stats.json').write_text(json.dumps(norm), encoding='utf-8')
            (folder / 'manifest.json').write_text(json.dumps(dict(file='norm_stats.json', semantic_sha256=semantic_hash(norm))), encoding='utf-8')
            official_hf = root / 'official/checkpoints/global_step_50000/hf_ckpt'
            official_hf.mkdir(parents=True)
            (official_hf/'config.json').write_text('{}')
            (official_hf/'model.safetensors').write_bytes(b'preflight-only-not-weights')
            official_config = deepcopy(config)
            official_config['data']['require_normalization_contract'] = False
            official_config['data']['norm_stats_file'] = None
            (root/'official/lingbotvla_cli.yaml').write_text(yaml.safe_dump(official_config), encoding='utf-8')
            with patch('diagnostics.clean_policy.run.select_samples', return_value=dict(dataset=str(root), samples=[])), \
                 patch('diagnostics.clean_policy.run.verify_selected_sources'):
                with patch.object(sys, 'argv', ['diagnostic', '--train-run', str(root / 'run'), '--qwen', str(qwen),
                                               '--official', str(official_hf), '--official-norm', str(ROOT/'assets/norm_stats/robotwin.json'),
                                               '--prepare-only', '--output', str(root/'out')]):
                    from diagnostics.clean_policy.run import main
                    main()
            plans = list((root / 'out').glob('*/plan.json'))
            self.assertEqual(len(plans), 1)
            saved = json.loads(plans[0].read_text(encoding='utf-8'))
            self.assertEqual([m['normalization']['count'] for m in saved['models']], [548893,6062592])
            self.assertFalse(list((root / 'out').rglob('*.log')))


if __name__ == '__main__':
    unittest.main()
