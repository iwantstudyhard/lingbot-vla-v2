"""CPU-only, paired smoothing comparison of saved clean-policy predictions.

Never loads weights, changes sampling, starts a simulator, or edits the source
run. The executed-prefix comparison truncates BEFORE prepare_chunk, exactly as
deployment does. Each recorded observation is an independent one-chunk probe,
not a closed-loop trajectory; the hypothetical full chunk is labeled separately.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from uuid import uuid4

import numpy as np

from deploy.action_smoothing import RoboTwinActionSmoother
from .common import ARMS, GRIPS, ROOT, action_metrics, sha256, write_json


def safe_component(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Unsafe model/sample identifier: {value!r}')
    return value


def load_evidence(run, official_label='official_reference'):
    """Fail closed on incomplete, unequal or altered metric/array evidence."""
    run = Path(run).resolve()
    manifest = {}

    def record(path):
        if not path.is_file() or not path.resolve().is_relative_to(run):
            raise ValueError(f'Missing or out-of-run source: {path}')
        manifest[path.relative_to(run).as_posix()] = sha256(path)
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding='utf-8'))

    completion = read(run / 'completion.json')
    if completion.get('status') != 'complete' or completion.get('failures'):
        raise ValueError('Source run is incomplete; do not compare partial workers')
    plan = read(run / 'plan.json')
    labels = [safe_component(m['label']) for m in plan['models']]
    samples = [safe_component(s['id']) for s in plan['data']['samples']]
    seeds, steps = plan['noise_seeds'], plan['denoising_steps']
    for name, values in (('labels', labels), ('samples', samples), ('seeds', seeds), ('steps', steps)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f'Empty/duplicate {name}')
    if official_label not in labels or len(labels) < 2:
        raise ValueError('Need the official reference and at least one own checkpoint')
    if any(type(v) is not int or v < 0 for v in seeds + steps) or any(v < 1 for v in steps):
        raise ValueError('Seeds/steps must be nonnegative/positive integers')
    expected = {(s, t, n) for s in samples for t in steps for n in seeds}
    predictions, experts, metadata, records = {}, {}, {}, {}
    for label in labels:
        worker = read(run / label / 'completion.json')
        if (worker.get('status') != 'complete' or worker.get('samples') != len(samples)
                or worker.get('sampling_rows') != len(expected)):
            raise ValueError(f'Incomplete worker: {label}')
        sampling = record(run / label / 'sampling.jsonl')
        rows = [json.loads(line) for line in sampling.read_text(encoding='utf-8').splitlines() if line.strip()]
        keys = [(r['sample'], r['steps'], r['seed']) for r in rows]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f'Unequal/duplicate prediction coverage: {label}')
        for sample in samples:
            folder = run / label / sample
            sample_meta = read(folder / 'sample.json')
            if sample_meta['id'] != sample:
                raise ValueError(f'Sample identifier mismatch: {folder}')
            with np.load(record(folder / 'expert.npz'), allow_pickle=False) as z:
                target, pad, state = z['target'].copy(), z['pad'].copy(), z['state'].copy()
            if (target.ndim != 2 or target.shape[1] != 14 or state.shape != (14,)
                    or pad.shape != (len(target),) or pad.dtype != bool
                    or pad[0] or np.any(np.diff(pad.astype(int)) < 0)
                    or not np.isfinite(state).all()):
                raise ValueError(f'Invalid expert action/state/padding: {folder}')
            action_metrics(target, target, pad)  # Includes finite target validation.
            if sample in experts:
                if any(not np.array_equal(a, b) for a, b in zip(experts[sample], (target, pad, state))):
                    raise ValueError(f'Expert/state differs between models: {sample}')
                if metadata[sample].get('task') != sample_meta.get('task'):
                    raise ValueError(f'Task differs between models: {sample}')
            else:
                experts[sample] = target, pad, state
                metadata[sample] = sample_meta
        for row in rows:
            sample, t, n = row['sample'], row['steps'], row['seed']
            if row['label'] != label:
                raise ValueError(f'Row label mismatch: {label}')
            times = row.get('denoising_times', [])
            calls = row.get('actual_denoising_calls')
            if type(calls) is not int or calls < 1 or calls != len(times):
                raise ValueError(f'Missing/inconsistent actual sampler calls: {label}/{sample}')
            path = run / label / sample / f'prediction_steps{t}_seed{n}.npz'
            with np.load(record(path), allow_pickle=False) as z:
                pred = z['prediction'].copy()
            target, pad, _ = experts[sample]
            measured = action_metrics(pred, target, pad)
            for key, value in measured.items():
                if key not in row['metrics'] or not np.isclose(row['metrics'][key], value, rtol=1e-7, atol=1e-9):
                    raise ValueError(f'Saved metrics disagree with arrays: {label}/{sample}/{key}')
            key = label, sample, t, n
            predictions[key], records[key] = pred, row
    return dict(run=run, plan=plan, labels=labels, samples=samples, experts=experts,
                metadata=metadata, predictions=predictions, records=records, manifest=manifest)


def smooth_chunk(prediction, state, length, alpha=.35, window=5, max_delta=.05):
    """Independent prefix probe using the exact deployment filter, no resampling."""
    if type(length) is not int or not 1 <= length <= len(prediction):
        raise ValueError('Chunk length must be within the saved action horizon')
    smoother = RoboTwinActionSmoother('ema', alpha, window, max_delta)
    smoother.reset(state)
    prepared = smoother.prepare_chunk(prediction[:length])
    result = np.stack([smoother.filter_action(a) for a in prepared])
    if not np.array_equal(result[:, GRIPS], prediction[:length, GRIPS]):
        raise AssertionError('Smoothing changed grippers')
    return result


def extended_metrics(pred, target, pad, static_threshold=.005):
    result = action_metrics(pred, target, pad)
    valid = ~pad
    error = (pred - target)[valid][:, ARMS]
    static = np.ptp(target[valid][:, ARMS], axis=0) <= static_threshold
    result.update(arm_p95_abs_error_rad=float(np.quantile(np.abs(error), .95)),
                  last_valid_arm_mae_rad=float(np.abs(error[-1]).mean()),
                  arm_temporal_bias_rad=float(np.abs(error.mean(axis=0)).mean()),
                  static_joints=int(static.sum()), moving_joints=int((~static).sum()))
    for name, mask in (('static', static), ('moving', ~static)):
        result[name + '_arm_mae_rad'] = float(np.abs(error[:, mask]).mean()) if mask.any() else None
    return result


def build_measurements(evidence, use_length=20, alpha=.35, window=5, max_delta=.05):
    rows, outputs, control_rows = [], {}, []
    # Validate the filter even before any output directory is created.
    config = RoboTwinActionSmoother('ema', alpha, window, max_delta).config()
    horizons = {len(y) for y, _, _ in evidence['experts'].values()}
    if len(horizons) != 1 or type(use_length) is not int or not 1 <= use_length <= min(horizons):
        raise ValueError('Inconsistent horizon or invalid --use-length')
    scopes = [('executed_prefix', use_length), ('hypothetical_full_chunk', min(horizons))]
    for sample in evidence['samples']:
        target, pad, state = evidence['experts'][sample]
        for scope, length in scopes:
            filtered = smooth_chunk(target, state, length, alpha, window, max_delta)
            control_rows.append(dict(label='expert_filter_control', sample=sample, scope=scope,
                                     metrics=extended_metrics(filtered, target[:length], pad[:length])))
            outputs[('expert_filter_control', sample, None, None, scope, 'ema')] = filtered
    for key, pred in evidence['predictions'].items():
        label, sample, steps, seed = key
        target, pad, state = evidence['experts'][sample]
        source = evidence['records'][key]
        for scope, length in scopes:
            for profile in ('raw', 'ema'):
                values = pred[:length].copy() if profile == 'raw' else smooth_chunk(pred, state, length, alpha, window, max_delta)
                metrics = extended_metrics(values, target[:length], pad[:length])
                rows.append(dict(label=label, sample=sample, episode=source['episode'],
                                 denoising_steps=steps, actual_denoising_calls=source['actual_denoising_calls'],
                                 seed=seed, scope=scope, profile=profile, metrics=metrics))
                outputs[key + (scope, profile)] = values
    summaries = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row['label'], row['denoising_steps'], row['scope'], row['profile'])].append(row)
    for (label, steps, scope, profile), group in groups.items():
        summary = dict(label=label, denoising_steps=steps, scope=scope, profile=profile,
                       prediction_rows=len(group), observations=len(evidence['samples']),
                       actual_denoising_calls_min=min(r['actual_denoising_calls'] for r in group),
                       actual_denoising_calls_max=max(r['actual_denoising_calls'] for r in group))
        for metric in group[0]['metrics']:
            values = [r['metrics'][metric] for r in group if r['metrics'][metric] is not None]
            summary[metric] = float(np.mean(values)) if values else None
            if metric in ('static_arm_mae_rad', 'moving_arm_mae_rad'):
                summary[metric + '_prediction_rows'] = len(values)
        first_stds, full_stds = [], []
        for sample in evidence['samples']:
            stack = np.stack([outputs[(label, sample, steps, n, scope, profile)] for n in evidence['plan']['noise_seeds']])
            valid = ~evidence['experts'][sample][1][:stack.shape[1]]
            if len(stack) > 1:
                first_stds.append(float(stack[:, 0, ARMS].std(axis=0, ddof=0).mean()))
                full_stds.append(float(stack[:, valid][:, :, ARMS].std(axis=0, ddof=0).mean()))
        summary['first_action_seed_std_rad'] = float(np.mean(first_stds)) if first_stds else None
        summary['full_valid_seed_std_rad'] = float(np.mean(full_stds)) if full_stds else None
        summaries.append(summary)
    controls = []
    for scope, _ in scopes:
        subset = [r for r in control_rows if r['scope'] == scope]
        control = dict(scope=scope, observations=len(subset), label='expert_filter_control')
        for metric in subset[0]['metrics']:
            values = [r['metrics'][metric] for r in subset if r['metrics'][metric] is not None]
            control[metric] = float(np.mean(values)) if values else None
        controls.append(control)
    return dict(rows=rows, summaries=summaries, controls=controls, control_rows=control_rows,
                outputs=outputs, scopes=scopes, filter=config)


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def paired_changes(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row['label'], row['sample'], row['denoising_steps'], row['seed'], row['scope'])][row['profile']] = row['metrics']
    result = []
    for (label, sample, steps, seed, scope), pair in groups.items():
        row = dict(label=label, sample=sample, denoising_steps=steps, seed=seed, scope=scope)
        for key in ('full_arm_mae_rad', 'error_d2_rad_per_action2', 'last_valid_arm_mae_rad', 'full_gripper_mae_native'):
            if key in pair['raw'] and key in pair['ema']:
                row[key + '_ema_minus_raw'] = pair['ema'][key] - pair['raw'][key]
        row['arm_mae_improved'] = row['full_arm_mae_rad_ema_minus_raw'] < 0
        result.append(row)
    return result


def choose_plot_model(labels, official_label):
    own = [x for x in labels if x != official_label]
    return max(own, key=lambda x: (int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else -1, x))


def plot_results(output, evidence, measured, official_label, plot_model):
    """Chart contract: paired equal-input comparisons; lower errors are better.

    Bar charts for discrete checkpoints/profiles, zero baseline; physical joint
    lines only across action index (not seconds or playback frames). Explicit
    blue/orange/neutral palette, raw dashed/faded and EMA solid, same axis bounds
    within each joint, unfiltered expert black. Plot every input at every sampler
    setting for the selected own checkpoint; CSV includes ALL checkpoints/seeds.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    blue, orange, ink, grey = '#2563eb', '#d97706', '#222222', '#8b8b8b'
    for steps in evidence['plan']['denoising_steps']:
        print(f'[plot] requested denoise {steps}: overview + {len(evidence["samples"])} paired inputs', flush=True)
        subset = [r for r in measured['summaries'] if r['denoising_steps'] == steps and r['scope'] == 'executed_prefix']
        lookup = {(r['label'], r['profile']): r for r in subset}
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = [('full_arm_mae_rad', 'Arm action MAE', 'rad'),
                   ('error_d2_rad_per_action2', 'Second-difference error', 'rad/action²'),
                   ('full_valid_seed_std_rad', 'Across-seed action std', 'rad')]
        for ax, (metric, title, unit) in zip(axes, metrics):
            for i, label in enumerate(evidence['labels']):
                color = orange if label == official_label else blue
                for profile, shift in (('raw', -.18), ('ema', .18)):
                    value = lookup[(label, profile)].get(metric)
                    if value is not None:
                        ax.bar(i + shift, value, width=.33, color=color, alpha=.45 if profile == 'raw' else 1,
                               hatch='//' if profile == 'raw' else None, edgecolor=ink, linewidth=.5,
                               label=profile.upper() if i == 0 else None)
            control = next(r for r in measured['controls'] if r['scope'] == 'executed_prefix')
            if metric in control and control[metric] is not None:
                ax.axhline(control[metric], color=grey, linestyle=':', label='Expert + same filter')
            ax.set_xticks(range(len(evidence['labels'])), [x.replace('step_', '') for x in evidence['labels']], rotation=20)
            ax.set_title(title + ' (lower is better)')
            ax.set_ylabel(unit)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
        calls = sorted({r['actual_denoising_calls_min'] for r in subset} | {r['actual_denoising_calls_max'] for r in subset})
        fig.suptitle(f'First {measured["scopes"][0][1]} actions: truncate BEFORE smoothing | requested {steps}, actual calls {calls}')
        fig.legend(handles=[Patch(facecolor=blue, alpha=.45, hatch='//', edgecolor=ink, label='RAW'),
                            Patch(facecolor=blue, edgecolor=ink, label='EMA'),
                            Line2D([], [], color=grey, linestyle=':', label='Expert + same filter')],
                   loc='upper center', bbox_to_anchor=(.5, .925), ncol=3, frameon=False)
        fig.text(.5, .01, f'{len(evidence["samples"])} training inputs, {len(evidence["plan"]["noise_seeds"])} paired seeds; offline accuracy, NOT success rate', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .86))
        fig.savefig(output / f'overview_steps{steps}.png', dpi=150)
        plt.close(fig)
        for sample in evidence['samples']:
            for scope, length in measured['scopes']:
                target, pad, _ = evidence['experts'][sample]
                valid = ~pad[:length]
                x = np.flatnonzero(valid)
                fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
                seed = evidence['plan']['noise_seeds'][0]
                for joint, ax in zip(ARMS, axes.flat):
                    ax.plot(x, target[:length, joint][valid], color=ink, lw=2, label='Expert (unfiltered)')
                    for label, color, name in ((plot_model, blue, 'Ours'), (official_label, orange, 'Official')):
                        for profile in ('raw', 'ema'):
                            values = measured['outputs'][(label, sample, steps, seed, scope, profile)]
                            ax.plot(x, values[:, joint][valid], color=color, lw=1.7 if profile == 'ema' else 1,
                                    linestyle='-' if profile == 'ema' else '--', alpha=1 if profile == 'ema' else .5,
                                    label=f'{name} {profile.upper()}')
                    ax.set_title(f'{"L" if joint < 6 else "R"} joint {joint if joint < 6 else joint-7} (rad)', fontsize=10)
                    ax.grid(alpha=.18)
                fig.legend(handles=[Line2D([], [], color=ink, lw=2, label='Expert (unfiltered)'),
                                    Line2D([], [], color=blue, ls='--', alpha=.5, label='Ours raw'),
                                    Line2D([], [], color=blue, label='Ours EMA'),
                                    Line2D([], [], color=orange, ls='--', alpha=.5, label='Official raw'),
                                    Line2D([], [], color=orange, label='Official EMA')],
                           loc='upper center', bbox_to_anchor=(.5, .955), ncol=5, frameon=False)
                fig.suptitle(f'{sample} | {plot_model} vs official | seed {seed} | requested denoise {steps}\n'
                             + (f'Deployment prefix ({length} actions), independent observed-state reset' if scope == 'executed_prefix'
                                else f'HYPOTHETICAL single {length}-action chunk; NOT replan-every-{measured["scopes"][0][1]} execution'), fontsize=12)
                fig.supxlabel('Future action index — NOT video frame number or elapsed seconds')
                fig.text(.5, .045, evidence['metadata'][sample].get('task', '')[:140], ha='center', fontsize=9)
                fig.tight_layout(rect=(0, .07, 1, .91))
                fig.savefig(output / f'actions_{sample}_steps{steps}_{scope}.png', dpi=130)
                plt.close(fig)
            print(f'[plot] denoise {steps}: {sample} complete', flush=True)


def report_text(evidence, measured, changes, plot_model, official_label):
    lines = ['# 同输入平滑对照', '',
             '**这是已有 clean 训练集输入的离线比较，不是重新 rollout 或任务成功率。**', '',
             f'来源：`{evidence["run"]}`。原始数组/目录不修改，模型、噪声和采样器都没有重跑或改变。',
             f'精度沿用原评测 `{evidence["plan"]["precision"]}`；{len(evidence["samples"])} 输入；种子 {evidence["plan"]["noise_seeds"]}。', '',
             f'复用 `deploy/action_smoothing.py`，配置 `{measured["filter"]}`。只过滤12臂关节，夹爪逐元素保持原值。',
             '每个独立输入/种子都从该输入的实测关节 state 初始化；不能把不同示教阶段串成一条虚假的轨迹。', '',
             '## 主比较：实际发送的动作前缀', '',
             f'先截取前 {measured["scopes"][0][1]} 动作，再做窗口均值、EMA和单动作增量限幅，与部署顺序一致。',
             '真实闭环中EMA历史会跨chunk保留；这里仅测从实测state出发的一段，无法证明后续闭环边界或成功率。',
             '专家真值始终不平滑。expert_filter_control 是把完美专家动作也过滤后产生的误差，用于量化滤波滞后。', '',
             '|模型|请求/实际采样次数|原始臂MAE(rad)|EMA臂MAE(rad)|EMA后二阶差分误差|原始→EMA换种子std(rad)|改善的预测数|',
             '|---|---|---:|---:|---:|---|---|']
    lookup = {(r['label'], r['denoising_steps'], r['scope'], r['profile']): r for r in measured['summaries']}
    for steps in evidence['plan']['denoising_steps']:
        for label in evidence['labels']:
            raw, ema = [lookup[(label, steps, 'executed_prefix', p)] for p in ('raw', 'ema')]
            def format_metric(row, metric):
                return 'NA' if row.get(metric) is None else f'{row[metric]:.6f}'

            pair = [r for r in changes if r['label'] == label and r['denoising_steps'] == steps and r['scope'] == 'executed_prefix']
            lines.append(f'|{label}|{steps}/{ema["actual_denoising_calls_min"]}–{ema["actual_denoising_calls_max"]}|'
                         f'{raw["full_arm_mae_rad"]:.6f}|{ema["full_arm_mae_rad"]:.6f}|{format_metric(ema, "error_d2_rad_per_action2")}|'
                         f'{format_metric(raw, "full_valid_seed_std_rad")}→{format_metric(ema, "full_valid_seed_std_rad")}|'
                         f'{sum(r["arm_mae_improved"] for r in pair)}/{len(pair)}|')
    for r in measured['controls']:
        lines += ['', f'专家滤波控制 `{r["scope"]}`：臂MAE {r["full_arm_mae_rad"]:.6f} rad，'
                  f'最后有效动作MAE {r["last_valid_arm_mae_rad"]:.6f} rad（未滤波专家误差为0）。']
    primary_steps = evidence['plan']['denoising_steps'][0]
    raw, ema = [lookup[(plot_model, primary_steps, 'executed_prefix', p)] for p in ('raw', 'ema')]
    official_raw = lookup[(official_label, primary_steps, 'executed_prefix', 'raw')]
    lines += ['', '## 所选检查点的关键读数', '',
              f'`{plot_model}`，请求 {primary_steps} 采样次数，前 {measured["scopes"][0][1]} 动作：',
              f'- 臂MAE：原始 {raw["full_arm_mae_rad"]:.6f} → 平滑 {ema["full_arm_mae_rad"]:.6f} rad；'
              f'官方不平滑 {official_raw["full_arm_mae_rad"]:.6f} rad。']
    if raw.get('error_d2_rad_per_action2') is not None:
        lines += [f'- 二阶差分误差：{raw["error_d2_rad_per_action2"]:.6f} → '
                  f'{ema["error_d2_rad_per_action2"]:.6f} rad/action²；更小不等于动作位置更准确。']
    if raw['moving_arm_mae_rad'] is not None:
        lines += [f'- 运动关节MAE：{raw["moving_arm_mae_rad"]:.6f} → {ema["moving_arm_mae_rad"]:.6f} rad；'
                  '需特别检查因滞后导致的变差，不宜只观察静止关节。']
    lines += ['', '## 如何看图和数据', '',
              f'- `overview_steps*.png`：全部检查点原始/EMA并排；灰线显示专家滤波误差。',
              f'- `actions_*_executed_prefix.png`：{plot_model}和{official_label}的12关节、同种子曲线，黑线为真实值。',
              '- `actions_*_hypothetical_full_chunk.png`：假设一次执行完整50动作，只作辅助，不能冒充20动作重规划评测。',
              '- `summary.csv/json`：每模型、采样步数、前缀/全长、raw/ema；所有夹爪误差、静止/运动关节误差、末动作误差。',
              '- `paired_changes.csv`：同一输入和种子的逐项变化，ema_minus_raw < 0 表示误差下降。',
              '- `per_prediction.csv`：逐输入/种子指标，保留少数样本的变差情况，不只看平均值。',
              '- `filtered_actions/`：平滑数组和未平滑专家真值；`expert_filter_control/`：完美专家动作经过滤波的滞后控制。',
              '- `source_manifest.json`：所有读取文件的SHA256，输出完成前再次检查来源未变化。', '',
              '## 指标口径和局限', '',
              '先对每输入的有效动作×12臂关节取平均，再对同数量的输入/种子平均；不含55D padding或专家尾部padding。',
              '静止关节=该有效chunk专家范围≤0.005 rad；无该类关节的样本不参加该类平均，summary记录相应行数。',
              '差分单位为rad/action与rad/action²，不是rad/s或物理加速度。种子std是这些种子的总体std，不是任务轨迹std。',
              '如果原评测请求30次但实际只有29次，本报告原样保留，不把后处理伪装成修复采样器后的新预测。',
              '初始state锚定本身可降低第一动作误差/std，所以必须同时看整段误差、末动作误差和专家滤波控制。',
              '没有任何自动选最佳滤波参数、修改训练loss、归一化、精度或视觉冻结配置。官方是混合数据参考，不是同预算基线。']
    return '\n'.join(lines) + '\n'


def run_comparison(run, output_base=None, use_length=20, alpha=.35, window=5,
                   max_delta=.05, official_label='official_reference', plot_model=None, plots=True):
    evidence = load_evidence(run, official_label)
    measured = build_measurements(evidence, use_length, alpha, window, max_delta)
    plot_model = plot_model or choose_plot_model(evidence['labels'], official_label)
    if plot_model not in evidence['labels'] or plot_model == official_label:
        raise ValueError('--plot-model must name an own checkpoint in plan.json')
    base = Path(output_base or ROOT / 'outputs/eval_outputs/clean_policy_smoothing').resolve()
    if base.is_relative_to(evidence['run']):
        raise ValueError('Output must be outside the immutable source run')
    output = base / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'Output: {output}', flush=True)
    write_json(output / 'completion.json', dict(status='running'))
    write_json(output / 'source_manifest.json', dict(source_run=str(evidence['run']), files=evidence['manifest']))
    write_json(output / 'plan.json', dict(source_plan=evidence['plan'], filter=measured['filter'],
               use_length=use_length, official_label=official_label, plot_model=plot_model,
               code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
                            (Path(__file__).resolve(), ROOT / 'deploy/action_smoothing.py', ROOT / 'diagnostics/clean_policy/common.py')},
               interpretation='Postprocess SAME saved predictions; independent one-chunk TRAIN-set probes, NOT rollouts'))
    changes = paired_changes(measured['rows'])
    write_csv(output / 'summary.csv', measured['summaries'])
    write_json(output / 'summary.json', dict(sampling=measured['summaries'], expert_filter_control=measured['controls']))
    write_csv(output / 'paired_changes.csv', changes)
    write_csv(output / 'per_prediction.csv', [{**{k: v for k, v in r.items() if k != 'metrics'}, **r['metrics']} for r in measured['rows']])
    write_csv(output / 'expert_filter_control.csv', [{**{k: v for k, v in r.items() if k != 'metrics'}, **r['metrics']} for r in measured['control_rows']])
    for (label, sample, steps, seed, scope, profile), values in measured['outputs'].items():
        if profile != 'ema':
            continue
        folder = output / ('expert_filter_control' if steps is None else 'filtered_actions') / label / sample
        folder.mkdir(parents=True, exist_ok=True)
        target, pad, state = evidence['experts'][sample]
        raw = target if steps is None else evidence['predictions'][(label, sample, steps, seed)]
        name = scope if steps is None else f'steps{steps}_seed{seed}_{scope}'
        np.savez_compressed(folder / f'{name}.npz', prediction=values, raw_prediction=raw[:len(values)],
                            target=target[:len(values)], pad=pad[:len(values)], state=state)
    (output / 'REPORT.md').write_text(report_text(evidence, measured, changes, plot_model, official_label), encoding='utf-8')
    if plots:
        plot_results(output, evidence, measured, official_label, plot_model)
    for relative, digest in evidence['manifest'].items():
        if sha256(evidence['run'] / relative) != digest:
            raise RuntimeError(f'Source changed during comparison: {relative}; output NOT certified complete')
    write_json(output / 'completion.json', dict(status='complete', source_unchanged=True,
               input_predictions=len(evidence['predictions']), metric_rows=len(measured['rows']),
               expert_control_rows=len(measured['control_rows']), source_files=len(evidence['manifest'])))
    print(f'Complete; source unchanged. Read {output / "REPORT.md"}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Completed clean_policy_diagnostics run with saved NPZ predictions')
    parser.add_argument('--output', help='Parent directory; a unique time_UUID child is always created')
    parser.add_argument('--use-length', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=.35)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--max-delta', type=float, default=.05, help='rad/action, not rad/s; zero disables cap')
    parser.add_argument('--official-label', default='official_reference')
    parser.add_argument('--plot-model', help='Own label for joint plots (default: latest numeric checkpoint)')
    parser.add_argument('--no-plots', action='store_true', help='Save numerical results without PNGs')
    args = parser.parse_args()
    run_comparison(args.run, args.output, args.use_length, args.alpha, args.window,
                   args.max_delta, args.official_label, args.plot_model, not args.no_plots)


if __name__ == '__main__':
    main()
