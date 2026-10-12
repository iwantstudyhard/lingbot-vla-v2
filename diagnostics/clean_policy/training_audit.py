"""Audit the saved stage1 configuration, without touching training or importing CUDA."""
import argparse
import csv
import json
import math
from pathlib import Path
from .common import write_json


def scheduled_lr(train, step):
    """Mirror this repo's cosine LambdaLR (verified against its actual function)."""
    total, peak = train['max_steps'], float(train['lr'])
    warmup, decay = int(total * train.get('lr_warmup_ratio', 0)), int(total * train.get('lr_decay_ratio', 1))
    start, floor = float(train.get('lr_start', 0)), float(train['lr_min'])
    if train.get('lr_decay_style') != 'cosine' or not 0 < peak or not 0 <= floor <= peak or not 0 <= warmup < decay <= total:
        raise ValueError('Audit supports a valid cosine schedule only; refusing to guess other schedules')
    if step < warmup:
        return start + (peak-start)*step/max(1, warmup)
    if step > decay:
        return floor
    progress = (step-warmup)/max(1, decay-warmup)
    return floor + (peak-floor)*0.5*(1+math.cos(math.pi*progress))


def audit(config):
    train, data = config['train'], config['data']
    frames, batch, total = int(data.get('expected_num_frames', 548893)), train['global_batch_size'], train['max_steps']
    multiplier = math.sqrt(train['token_num_experts']/train['token_top_k']) if train.get('use_moe_expert_lr') and train.get('use_moe') else 1.0
    result = dict(optimizer=train['optimizer'], global_batch=batch, micro_batch=train['micro_batch_size'],
                  accumulation=train['gradient_accumulation_steps'], max_steps=total,
                  sample_exposures=total*batch, equivalent_frame_passes=total*batch/frames,
                  warmup_steps=int(total*train.get('lr_warmup_ratio', 0)),
                  base_lr_peak=float(train['lr']), base_lr_final=scheduled_lr(train, total),
                  routed_expert_group_multiplier=multiplier,
                  routed_expert_lr_final=scheduled_lr(train, total)*multiplier,
                  configured_vit_lr=train.get('vit_lr'),
                  vit_lr_used_by_current_muon_builder=False if train['optimizer']=='muon' else None,
                  vision_frozen=train.get('freeze_vision_encoder'), freeze_vit=train.get('freeze_vit'),
                  loss_type=train['loss_type'],
                  warnings=[
                      'Frame passes count sampled starting frames; overlapping action horizons are not independent new demonstrations.',
                      'Current Muon builder groups base vs routed experts; it does NOT use vit_lr for a separate vision group.',
                      'Logged LR is the next-step group LR after scheduler.step; Muon also applies shape-dependent update scaling.',
                      'No obvious extra gradient-accumulation division was found in current code; this is not a distributed backward equivalence test.',
                      'DCP resume restores optimizer/scheduler/global_step. Changing YAML lr alone does not reliably restart learning rate.',
                      'Lowest training loss checkpoint is not necessarily the best policy checkpoint.',
                      'A training-set demonstration test cannot establish clean rollout success or random-scene generalization.'
                  ])
    result['conditional_continuation'] = dict(
        condition='Only AFTER forward paths agree and endpoint/checkpoint diagnostics support undertraining',
        initialization='Selected audited clean HF weights only; NEW output, optimizer and scheduler; NOT --resume-run',
        extra_steps=5000, global_batch=batch, base_lr=1e-5, lr_min=5e-6,
        warmup_steps=100, lr_warmup_ratio=0.02, augmentation=False,
        save_steps=500, max_checkpoints_to_keep=3,
        note='Small controlled extension, not a promise of success. Do not add augmentation/optimizer/loss changes simultaneously.')
    return result


def save_audit(directory, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'audit.json', result)
    (directory / 'README.md').write_text(
        '# 学习率与训练配置审阅\n\n'
        f'- 当前配置：{result["max_steps"]} optimizer steps，global batch {result["global_batch"]}，'
        f'约 {result["equivalent_frame_passes"]:.3f} 次起始帧曝光。\n'
        f'- 基础 LR：峰值 {result["base_lr_peak"]:.3g}，结束 {result["base_lr_final"]:.3g}；'
        f'路由专家组倍率 {result["routed_expert_group_multiplier"]:.3f}。\n'
        '- Muon 当前没有单独使用 vit_lr；它不是“视觉部分完全没更新”，而是没有独立视觉 LR 分组。\n'
        '- 是否改动历史训练时的实现，需要历史 Git commit；仅凭当前代码不能完全追溯。\n'
        '- 先读总目录 REPORT.md。路径一致且完整采样仍在改善时，才考虑一个 5000 步 clean-only 对照续训：'
        '仅加载选中 HF 权重，新 optimizer/scheduler，基础 LR 1e-5 → 5e-6，warmup 100 步。\n'
        '- 这里没有修改训练配置，也没有自动发起训练。不要为了追低训练 loss 直接训练几十万步。\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load((Path(args.run) / 'lingbotvla_cli.yaml').read_text(encoding='utf-8'))
    result = audit(config)
    save_audit(args.output, result)
    train = config['train']
    with (Path(args.output) / 'lr_schedule.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['optimizer_step', 'base_group_lr', 'routed_expert_group_lr'])
        for step in range(train['max_steps']+1):
            lr = scheduled_lr(train, step)
            writer.writerow([step, lr, lr*result['routed_expert_group_multiplier']])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
