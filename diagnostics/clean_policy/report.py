"""Aggregate only measured diagnostics; incomplete paths must never become a pass."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .common import ARMS, write_json


def read_jsonl(path):
    if not path.is_file():
        return []
    # A killed worker may leave a partial LAST line; retain complete earlier evidence.
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            break
    return result


def build_report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    plan = json.loads((run / 'plan.json').read_text(encoding='utf-8'))
    summaries, path_results, missing = [], [], []
    for model in plan['models']:
        label = model['label']
        completion = run / label / 'completion.json'
        if not completion.is_file():
            missing.append(label)
        groups = defaultdict(list)
        for row in read_jsonl(run / label / 'sampling.jsonl'):
            groups[row['steps']].append(row)
        for steps, rows in sorted(groups.items()):
            keys = sorted({k for r in rows for k in r['metrics']})
            summary = dict(label=label, denoising_steps=steps, prediction_rows=len(rows),
                           observations=len({r['sample'] for r in rows}), complete=completion.is_file())
            counts = [r['actual_denoising_calls'] for r in rows if 'actual_denoising_calls' in r]
            if counts:
                summary['actual_denoising_calls_min'] = min(counts)
                summary['actual_denoising_calls_max'] = max(counts)
            # Every observation has same number of seeds in a complete worker.
            # Partial rows are labeled partial; never silently compare unequal samples.
            summary.update({k: float(np.mean([r['metrics'][k] for r in rows if k in r['metrics']])) for k in keys})
            stds = []
            for sample in plan['data']['samples']:
                path = run / label / sample['id'] / f'seed_variability_steps{steps}.json'
                if path.is_file():
                    v = json.loads(path.read_text(encoding='utf-8'))['first_action_arm_std_rad']
                    if v is not None:
                        stds.append(v)
            summary['first_action_seed_std_rad'] = float(np.mean(stds)) if stds else None
            summaries.append(summary)
        for row in read_jsonl(run / label / 'paths.jsonl'):
            path_results.append(dict(label=label, **row))
    write_json(run / 'summary.json', dict(sampling=summaries, paths=path_results,
               incomplete_models=missing, interpretation=plan['interpretation']))
    if path_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, comparison in zip(axes.flat, ('cache_only', 'attention_only', 'train_eval_kernels', 'end_to_end_paths')):
            for model in plan['models']:
                points = defaultdict(list)
                for row in path_results:
                    if row['label'] == model['label'] and comparison in row['comparisons']:
                        points[row['time']].append(row['comparisons'][comparison]['mae'])
                if points:
                    times = sorted(points)
                    axis.plot(times, [np.mean(points[t]) for t in times], 'o-', label=model['label'])
            axis.set_title(comparison)
            axis.set_xlabel('Flow time t (1 = noise, 0 = actions)')
            axis.set_ylabel('Valid-joint velocity MAE between paths')
            axis.grid(alpha=.2)
            if axis.lines:
                axis.legend(fontsize=8)
        fig.suptitle('Same inputs / noise / weights — missing branches are NOT a pass')
        fig.tight_layout()
        fig.savefig(run / 'forward_path_comparison.png', dpi=150)
        plt.close(fig)
    if summaries:
        keys = sorted({k for r in summaries for k in r})
        with (run / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for axis, metric, title in zip(axes, ('full_arm_mae_rad', 'error_d2_rad_per_action2', 'first_action_seed_std_rad'),
                                      ('Endpoint arm MAE (rad)', 'Second-difference error (rad/action²)', 'First-action seed std (rad)')):
            for steps in plan['denoising_steps']:
                subset = [r for r in summaries if r['denoising_steps']==steps and r.get(metric) is not None]
                axis.plot([r['label'] for r in subset], [r[metric] for r in subset], 'o-', label=f'{steps} denoise steps')
            axis.set_title(title)
            axis.tick_params(axis='x', rotation=35)
            axis.grid(alpha=.2)
            axis.legend()
        fig.suptitle('TRAIN-set offline diagnostics — not rollout success; official is a mixed-data reference')
        fig.tight_layout()
        fig.savefig(run / 'checkpoint_comparison.png', dpi=150)
        plt.close(fig)
    for sample in plan['data']['samples'][:6]:
        fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
        found_target = False
        for model in plan['models']:
            folder = run / model['label'] / sample['id']
            pred_file = folder / f'prediction_steps{plan["denoising_steps"][0]}_seed{plan["noise_seeds"][0]}.npz'
            if not pred_file.is_file():
                continue
            with np.load(folder / 'expert.npz', allow_pickle=False) as expert, np.load(pred_file, allow_pickle=False) as prediction:
                valid = ~expert['pad']
                target, pred = expert['target'][valid], prediction['prediction'][valid]
                for index, axis in zip(ARMS, axes.flat):
                    if not found_target:
                        axis.plot(target[:, index], color='black', linewidth=2, label='expert clean action')
                    axis.plot(pred[:, index], alpha=.8, label=model['label'])
                    axis.set_title(f'Physical joint {index} (rad)')
                    axis.grid(alpha=.2)
                found_target = True
        if found_target:
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f'{sample["id"]} — same recorded input, seed {plan["noise_seeds"][0]}, NO smoothing')
            fig.supxlabel('Future action index (dataset convention, not video playback frames)')
            fig.tight_layout()
            fig.savefig(run / f'actions_{sample["id"]}.png', dpi=130)
        plt.close(fig)
    lines = ['# 三项诊断结果', '',
             '仅使用已有官方 clean 示教；没有模拟器 rollout、外部样本或新训练数据。', '',
             '**这是训练集上的离线误差，不是成功率；官方模型训练数据和训练量不同。**', '',
             f'精度 {plan["precision"]}；不编译；不平滑；固定 {len(plan["data"]["samples"])} 个输入；种子 {plan["noise_seeds"]}。', '',
             '## 1. 同输入前向路径', '',
             'paths.jsonl/summary.json 分别隔离 KV cache、attention 实现、train/eval kernel，再比较总差异。',
             'forward_path_comparison.png 展示不同噪声时刻的路径差异；空白分支表示没有可用测量，不表示误差为0。',
             '采样噪声和 x_t 完全相同，屏蔽无效关节和尾部 padding；辅助 loss 不算、不加载 teacher，任务 token 不移除。',
             '这是 no-grad 前向检查，不验证 FSDP/backward/optimizer。BF16 的微小误差不等于错误，不能套一个通用阈值。', '',
             f'已记录 {len(path_results)} 条路径探针；其中 {sum(bool(r["errors"]) for r in path_results)} 条存在失败分支。',
             '**失败/缺失分支是未完成诊断，不是“路径一致”。**', '',
             '## 2. 从纯噪声完整采样 vs 专家动作', '',
             'summary.csv：full/first20/first action 关节 MAE（rad）、夹爪 MAE（原单位）、',
             '一阶/二阶差分误差、换随机种子的动作标准差；sample.json 同时记录训练/部署预处理差异与归一化回环误差。',
             '专家 chunk 是 action[t:t+50]，不足部分重复尾帧但不计分；不把动作幅度小误当准确。', '',
             '## 3. 多检查点与采样步数', '',
             'checkpoint_comparison.png 和 actions_*.png 比较同一组输入，不同检查点、10/30 等采样步数。',
             '更晚检查点在同一数据上同时减少 endpoint MAE、差分误差、seed std，才支持继续学习；不是仅凭训练 loss。',
             '如果只剩一个检查点，则不能得出训练时间趋势；官方参考不能代替我们自己的早晚检查点。', '',
             '## 下一步判断', '',
             '- cache/attention/kernel 差异大：先定位该分支，不要先加训练时长或平滑 loss。',
             '- 路径接近，但 10→30 采样明显改善：先查采样离散误差；这不是自动证明模型已收敛。',
             '- 路径接近，完整采样差，晚检查点持续改善：考虑 training_audit/ 中的小规模受控续训。',
             '- 训练集完整采样都差且不改善：检查每关节、运动阶段、噪声 t 分层；不能只用平均 loss 决定加 LR。',
             '- 离线准确而闭环失败：转查控制执行、观测时序、chunk/replan、闭环分布偏移。', '',
             f'未完成的 worker：{missing or "无"}。逐模型日志保存在总目录 *.log。']
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    build_report(Path(args.run))


if __name__ == '__main__':
    main()
