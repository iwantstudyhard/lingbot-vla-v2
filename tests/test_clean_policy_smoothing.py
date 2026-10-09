"""CPU checks for paired saved-prediction smoothing, not GPU/rollout tests."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.action_smoothing import RoboTwinActionSmoother
from diagnostics.clean_policy.common import ARMS, GRIPS, action_metrics, sha256, write_json
from diagnostics.clean_policy.smoothing import (
    build_measurements, extended_metrics, load_evidence, run_comparison, smooth_chunk,
)


def fixture(run):
    run.mkdir()
    sample = dict(id='ep0001_f0000', episode=1, task='Test clean demonstration')
    plan = dict(models=[dict(label=x) for x in ('step_000500', 'official_reference')],
                data=dict(samples=[sample]), noise_seeds=[42, 43], denoising_steps=[10], precision='bf16')
    write_json(run / 'plan.json', plan)
    write_json(run / 'completion.json', dict(status='complete', failures=[]))
    target = np.zeros((6, 14), np.float32)
    target[:, 0] = np.arange(6) * .02
    target[4:] = target[3]
    target[:, GRIPS] = np.arange(6)[:, None] * .1
    pad = np.array([False]*4 + [True]*2)
    for model in plan['models']:
        folder = run / model['label'] / sample['id']
        folder.mkdir(parents=True)
        write_json(folder / 'sample.json', sample)
        np.savez_compressed(folder / 'expert.npz', target=target, pad=pad, state=target[0])
        rows = []
        for seed in plan['noise_seeds']:
            pred = target.copy()
            pred[:, ARMS] += (.1 if seed == 42 else -.1)
            np.savez_compressed(folder / f'prediction_steps10_seed{seed}.npz', prediction=pred)
            rows.append(dict(label=model['label'], sample=sample['id'], episode=1,
                             steps=10, seed=seed, actual_denoising_calls=10,
                             denoising_times=np.linspace(1, .1, 10).tolist(), metrics=action_metrics(pred, target, pad)))
        (folder.parent / 'sampling.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
        write_json(folder.parent / 'completion.json', dict(status='complete', samples=1, sampling_rows=2))


class CleanPolicySmoothingTests(unittest.TestCase):
    def test_exact_deployment_filter_and_gripper_identity(self):
        rng = np.random.default_rng(123)
        pred, state = rng.normal(size=(50, 14)), rng.normal(size=14)
        smoother = RoboTwinActionSmoother('ema', .35, 5, .05)
        smoother.reset(state)
        expected = np.stack([smoother.filter_action(a) for a in smoother.prepare_chunk(pred[:20])])
        actual = smooth_chunk(pred, state, 20)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, GRIPS], pred[:20, GRIPS])
        d = np.diff(np.vstack([state[ARMS], actual[:, ARMS]]), axis=0)
        self.assertLessEqual(float(np.abs(d).max()), .05 + 1e-12)

    def test_truncation_before_window_no_future_leak(self):
        pred, state = np.zeros((50, 14)), np.zeros(14)
        pred[20:, ARMS] = 100
        prefix = smooth_chunk(pred, state, 20, max_delta=0)
        full = smooth_chunk(pred, state, 50, max_delta=0)
        self.assertEqual(float(np.abs(prefix[:, ARMS]).max()), 0)
        self.assertGreater(float(np.abs(full[:20, ARMS]).max()), 0)

    def test_independent_state_reset_no_seed_or_phase_leak(self):
        pred, state = np.ones((50, 14)), np.zeros(14)
        first = smooth_chunk(pred, state, 20)
        smooth_chunk(pred * -10, state + 20, 20)
        np.testing.assert_array_equal(first, smooth_chunk(pred, state, 20))
        self.assertNotEqual(first[0, 0], smooth_chunk(pred, state + .5, 20)[0, 0])

    def test_expert_control_detects_lag_not_a_smoothing_win(self):
        target = np.zeros((50, 14))
        target[:, ARMS] = np.arange(50)[:, None] * .02
        filtered = smooth_chunk(target, target[0], 20)
        metrics = extended_metrics(filtered, target[:20], np.zeros(20, bool))
        self.assertGreater(metrics['full_arm_mae_rad'], .02)
        self.assertGreater(metrics['last_valid_arm_mae_rad'], .03)
        self.assertEqual(action_metrics(target[:20], target[:20], np.zeros(20, bool))['full_arm_mae_rad'], 0)

    def test_padding_and_separate_arm_gripper_metrics(self):
        target = np.zeros((5, 14))
        pred = target.copy()
        pred[:3, ARMS] = .1
        pred[:, GRIPS] = .7
        pred[3:, ARMS] = 999
        result = extended_metrics(pred, target, np.array([False]*3 + [True]*2))
        self.assertAlmostEqual(result['full_arm_mae_rad'], .1)
        self.assertAlmostEqual(result['full_gripper_mae_native'], .7)
        self.assertAlmostEqual(result['last_valid_arm_mae_rad'], .1)
        self.assertEqual(result['moving_arm_mae_rad'], None)

    def test_fixture_complete_coverage_summary_and_source_immutability(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            before = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
            output = run_comparison(source, Path(td) / 'output', use_length=3, plots=False)
            after = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            done = json.loads((output / 'completion.json').read_text())
            self.assertEqual(done['input_predictions'], 4)
            self.assertEqual(done['metric_rows'], 16)
            self.assertTrue(done['source_unchanged'])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary['sampling']), 8)
            self.assertEqual(len(summary['expert_filter_control']), 2)
            for row in summary['sampling']:
                self.assertEqual(row['prediction_rows'], 2)
                self.assertEqual(row['observations'], 1)
                self.assertEqual(row['actual_denoising_calls_min'], 10)
                if row['profile'] == 'raw':
                    self.assertAlmostEqual(row['full_valid_seed_std_rad'], .1, places=6)

    def test_duplicate_or_partial_coverage_rejected(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            sampling = source / 'step_000500/sampling.jsonl'
            first = sampling.read_text().splitlines()[0]
            sampling.write_text(first + '\n' + first + '\n')
            with self.assertRaisesRegex(ValueError, 'coverage'):
                load_evidence(source)

    def test_incomplete_worker_rejected(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            write_json(source / 'official_reference/completion.json', dict(status='running'))
            with self.assertRaisesRegex(ValueError, 'Incomplete worker'):
                load_evidence(source)

    def test_saved_metric_disagrees_with_prediction_rejected(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            file = source / 'step_000500/ep0001_f0000/prediction_steps10_seed42.npz'
            np.savez_compressed(file, prediction=np.zeros((6, 14)))
            with self.assertRaisesRegex(ValueError, 'metrics disagree'):
                load_evidence(source)

    def test_model_expert_state_mismatch_rejected(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            file = source / 'official_reference/ep0001_f0000/expert.npz'
            with np.load(file) as z:
                target, pad, state = z['target'].copy(), z['pad'].copy(), z['state'].copy()
            state[0] += .5
            np.savez_compressed(file, target=target, pad=pad, state=state)
            with self.assertRaisesRegex(ValueError, 'Expert/state differs'):
                load_evidence(source)

    def test_output_cannot_overwrite_source(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            with self.assertRaisesRegex(ValueError, 'immutable source'):
                run_comparison(source, source / 'new', use_length=3, plots=False)
            self.assertFalse((source / 'new').exists())

    def test_invalid_smoothing_options_fail_before_creating_output(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            for kwargs in (dict(use_length=0), dict(use_length=100), dict(alpha=0), dict(window=4), dict(max_delta=-1)):
                with self.assertRaises(ValueError):
                    run_comparison(source, Path(td) / 'output', plots=False, **kwargs)
            self.assertFalse((Path(td) / 'output').exists())

    def test_single_seed_does_not_claim_zero_seed_variability(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            evidence = load_evidence(source)
            evidence['plan']['noise_seeds'] = [42]
            evidence['predictions'] = {k: v for k, v in evidence['predictions'].items() if k[-1] == 42}
            measured = build_measurements(evidence, use_length=3)
            for row in measured['summaries']:
                self.assertIsNone(row['full_valid_seed_std_rad'])
                self.assertIsNone(row['first_action_seed_std_rad'])

    def test_short_prefix_reports_unavailable_difference_metric(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            output = run_comparison(source, Path(td) / 'output', use_length=1, plots=False)
            self.assertIn('|NA|', (output / 'REPORT.md').read_text(encoding='utf-8'))

    def test_unsafe_path_identifier_rejected(self):
        with TemporaryDirectory() as td:
            source = Path(td) / 'source'
            fixture(source)
            plan_path = source / 'plan.json'
            plan = json.loads(plan_path.read_text())
            plan['models'][0]['label'] = '../elsewhere'
            write_json(plan_path, plan)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                load_evidence(source)


if __name__ == '__main__':
    unittest.main()
