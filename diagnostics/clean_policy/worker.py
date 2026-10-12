"""One checkpoint per process/GPU. Only reads existing clean videos and weights."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from .common import CAMERAS, ARMS, action_chunk, action_metrics, vector_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    spec = next(x for x in plan['models'] if x['label'] == args.label)
    output = Path(args.plan).parent / args.label
    output.mkdir(exist_ok=True)
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from torchvision.transforms.v2 import Resize
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, set_seed_everywhere
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.data.vla_data.video_utils import decode_video_frames
    from lingbotvla.utils.episode_boundaries import bounded_timestamps
    from .paths import full_velocity, cached_velocity

    if not torch.cuda.is_available():
        raise RuntimeError('Worker requires CUDA; use --prepare-only for CPU preflight')
    dtype = torch.bfloat16 if plan['precision'] == 'bf16' else torch.float32
    policy = LingbotVLAv2Server(spec['hf'], robot_norm_path=spec['norm'], chunk_ret=True,
                               use_length=50, use_bf16=dtype == torch.bfloat16,
                               use_fp32=dtype == torch.float32, use_compile=False)
    policy.reset('robotwin')
    flow = policy.vla.model
    horizon = flow.config.n_action_steps
    if horizon != 50 or flow.config.max_action_dim != 55:
        raise ValueError('For fair fixed-noise comparison, this tool requires horizon=50 / max_action_dim=55')
    config = spec['config']
    if args.label != 'official_reference' and (config['data'].get('image_augment') or config['data'].get('stage2_augmentation_config')):
        raise ValueError('This probe reproduces unaugmented stage1 preprocessing only; do not claim stage2 train-path equivalence')
    future = bool(config['data'].get('use_future_image', False))
    train_transform = FeatureTransform(
        str(Path(__file__).resolve().parents[2] / 'configs/robot_configs/robotwin.yaml'),
        policy.data_config, policy.config, policy.processor,
        norm_stats_path=spec['norm'], chunk_size=horizon,
        use_depth_align=bool(config['train'].get('align_params')),
        use_future_image=future, image_augment=False)
    resize = Resize((getattr(policy.data_config, 'img_size', 256),) * 2)
    tasks_table = pq.read_table(Path(plan['data']['dataset']) / 'meta/tasks.parquet').to_pylist()
    tasks = {r['task_index']: r.get('__index_level_0__', r.get('task')) for r in tasks_table}
    if not tasks or any(not isinstance(v, str) for v in tasks.values()):
        raise ValueError('Unsupported tasks.parquet instruction column')
    write_json(output / 'environment.json', dict(
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, precision=plan['precision'], compile=False, smoothing=False,
        horizon=horizon, dimensions=55, training_attention=config['train'].get('attention_implementation'),
        norm=spec['normalization'], original_training_image_augment=config['data'].get('image_augment', False),
        probe_image_augment=False, note='No backward, optimizer, simulator or weight writes; reference original augmentation is not reproduced'))

    def cpu(tensor):
        return tensor.detach().float().cpu().numpy()

    def inputs_from(applied):
        result = {}
        for key in ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'state', 'image_grid_thw'):
            if key in applied:
                x = applied[key].unsqueeze(0).to('cuda')
                result[key] = x.to(dtype) if key in ('images', 'state') else x
        return result

    tables = {}
    rows = []
    path_rows = []
    with (output / 'sampling.jsonl').open('w', encoding='utf-8', buffering=1) as metrics_file, \
         (output / 'paths.jsonl').open('w', encoding='utf-8', buffering=1) as paths_file:
        for sample_no, sample in enumerate(plan['data']['samples']):
            start_time = time.monotonic()
            write_json(output / 'progress.json', dict(stage='decoding', sample=sample_no + 1, total=len(plan['data']['samples'])))
            # Keep at most one parquet in memory (files can be ~100MB).
            if sample['data_file'] not in tables:
                tables.clear()
                tables[sample['data_file']] = pq.read_table(sample['data_file'], columns=[
                    'index', 'episode_index', 'frame_index', 'timestamp', 'task_index', 'action', 'observation.state'])
            first, state, target, pad = action_chunk(tables[sample['data_file']], sample, horizon)
            task = tasks[first['task_index']]
            # The original action feature is named 'action', so its padding key
            # is 'action_is_pad', exactly as returned by LeRobotDataset.
            raw = {'observation.state': torch.from_numpy(state.copy()), 'action': torch.from_numpy(target.copy()),
                   'task': task, 'action_is_pad': torch.from_numpy(pad.copy())}
            observation = {'observation.state': state.copy(), 'task': task}
            for cam in CAMERAS:
                video = sample['videos'][cam]
                relative_times = [float(first['timestamp'])]
                if future:
                    relative_times.append(float(first['timestamp']) + (horizon-1)/sample['fps'])
                times, _ = bounded_timestamps(relative_times, video['start'], video['end'], sample['fps'])
                frames = decode_video_frames(video['path'], times, tolerance_s=1/sample['fps'] + 1e-4,
                                             backend=plan['video_backend'])
                if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
                    raise ValueError(f'Expected native uint8 TCHW video frames, got {frames.dtype} / {frames.shape}')
                observation[cam] = frames[0].permute(1, 2, 0).cpu().numpy()
                raw[cam] = resize(frames if future else frames[0])
            # Both transforms are real repo functions, not a second handwritten normalizer.
            trained = train_transform.apply(raw, policy_eval=False)
            deployed = policy._prepare_model_input(observation)
            training_inputs, deploy_inputs = inputs_from(trained), inputs_from(deployed)
            preprocessing = {}
            for key in training_inputs:
                a, b = training_inputs[key], deploy_inputs[key]
                if a.shape != b.shape:
                    raise ValueError(f'Training/deploy input shape mismatch: {key}: {a.shape} != {b.shape}')
                preprocessing[key] = vector_metrics(cpu(a), cpu(b), np.ones(tuple(a.shape), bool))
            actions = trained['actions'].unsqueeze(0).to(device='cuda', dtype=dtype)
            valid = (~pad)[None, :, None] & trained['action_joint_mask'].cpu().numpy()[None, None, :].astype(bool)
            # Confirm the ground-truth transform/inverse BEFORE measuring the model.
            restored = train_transform.unapply(dict(trained, actions=trained['actions'].clone()))['action'].numpy()
            inverse_error = float(np.abs(restored - target).max())
            if inverse_error > 1e-4:
                raise ValueError(f'Target normalization round-trip failed: {inverse_error}')
            sample_dir = output / sample['id']
            sample_dir.mkdir()
            np.savez_compressed(sample_dir / 'expert.npz', target=target, pad=pad, state=state,
                                normalized_target=cpu(actions), action_joint_mask=trained['action_joint_mask'].numpy())
            write_json(sample_dir / 'sample.json', dict(sample, task=task, preprocessing=preprocessing,
                                                       target_roundtrip_max_abs=inverse_error,
                                                       hold_current_state_metrics=action_metrics(np.repeat(state[None], horizon, axis=0), target, pad)))
            if sample_no < plan['path_samples']:
                set_seed_everywhere(plan['noise_seeds'][0])
                noise = torch.randn(actions.shape, device='cuda', dtype=dtype)
                for t in (0.1, 0.5, 0.9, 1.0):
                    time_tensor = torch.tensor([t], device='cuda', dtype=dtype)
                    xt = time_tensor[:, None, None]*noise + (1-time_tensor[:, None, None])*actions
                    velocities, errors = {}, {}
                    for mode, attention, training in (
                        ('full_eager_eval', 'eager', False),
                        ('full_training_attention_eval', config['train'].get('attention_implementation', 'flex_cached'), False),
                        ('full_training_attention_train', config['train'].get('attention_implementation', 'flex_cached'), True)):
                        try:
                            set_seed_everywhere(plan['noise_seeds'][0])
                            velocities[mode] = cpu(full_velocity(flow, training_inputs, actions, noise, time_tensor, attention, training))
                        except Exception:
                            errors[mode] = traceback.format_exc()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    try:
                        velocities['cached_eager_eval'] = cpu(cached_velocity(flow, training_inputs, xt, time_tensor))
                    except Exception:
                        errors['cached_eager_eval'] = traceback.format_exc()
                    comparisons = {}
                    for label, a, b in (
                        ('cache_only', 'cached_eager_eval', 'full_eager_eval'),
                        ('attention_only', 'full_training_attention_eval', 'full_eager_eval'),
                        ('train_eval_kernels', 'full_training_attention_train', 'full_training_attention_eval'),
                        ('end_to_end_paths', 'cached_eager_eval', 'full_training_attention_train')):
                        if a in velocities and b in velocities:
                            comparisons[label] = vector_metrics(velocities[a], velocities[b], valid)
                    for mode, velocity in velocities.items():
                        comparisons[mode + '_fm_l1'] = vector_metrics(velocity, cpu(noise-actions), valid)['mae']
                    record = dict(sample=sample['id'], time=t, comparisons=comparisons, errors=errors,
                                  status='incomplete' if errors else 'measured', preprocessing=preprocessing,
                                  scope='Same processed training input; no-grad forward only, NOT distributed/backward equivalence')
                    paths_file.write(json.dumps(record, allow_nan=False) + '\n')
                    path_rows.append(record)
                    np.savez_compressed(sample_dir / f'velocities_t{t:.1f}.npz', **velocities, noise=cpu(noise), x_t=cpu(xt))
                    print(f'[path] {args.label} {sample["id"]} t={t}: {record["status"]}', flush=True)
            for steps in plan['denoising_steps']:
                predictions = []
                flow.config.num_steps = steps
                for seed in plan['noise_seeds']:
                    write_json(output / 'progress.json', dict(stage='sampling', sample=sample_no + 1,
                               total=len(plan['data']['samples']), steps=steps, seed=seed))
                    set_seed_everywhere(seed)
                    noise = torch.randn(actions.shape, device='cuda', dtype=dtype)
                    # Record the ACTUAL production loop count too: requested
                    # steps and BF16 accumulated time need not be identical.
                    original_velocity = flow.predict_velocity
                    denoising_calls = []
                    def measured_velocity(*velocity_args, **velocity_kwargs):
                        denoising_calls.append(float(velocity_args[4][0].float().item()))
                        return original_velocity(*velocity_args, **velocity_kwargs)
                    flow.predict_velocity = measured_velocity
                    try:
                        with torch.no_grad():
                            normalized = flow.sample_actions(**deploy_inputs, noise=noise.clone())
                    finally:
                        flow.predict_velocity = original_velocity
                    pred = policy._unapply_batched_actions([deployed], normalized.float().cpu())['action'][0]
                    predictions.append(pred)
                    metrics = action_metrics(pred, target, pad)
                    metrics['normalized_endpoint_mae'] = vector_metrics(cpu(normalized), cpu(actions), valid)['mae']
                    record = dict(label=args.label, sample=sample['id'], episode=sample['episode'],
                                  steps=steps, actual_denoising_calls=len(denoising_calls),
                                  denoising_times=denoising_calls, seed=seed, metrics=metrics)
                    rows.append(record)
                    metrics_file.write(json.dumps(record, allow_nan=False) + '\n')
                    np.savez_compressed(sample_dir / f'prediction_steps{steps}_seed{seed}.npz',
                                        prediction=pred, normalized_prediction=cpu(normalized))
                seed_std = float(np.stack(predictions)[:, ~pad][:, :, ARMS].std(axis=0).mean()) if len(predictions) > 1 else None
                write_json(sample_dir / f'seed_variability_steps{steps}.json', dict(
                    first_action_arm_std_rad=float(np.stack(predictions)[:, 0, ARMS].std(axis=0).mean()) if len(predictions) > 1 else None,
                    full_valid_arm_std_rad=seed_std, seeds=plan['noise_seeds'],
                    note='Population std over these seeds; null when only one seed, not evidence of zero variability'))
            print(f'[sample {sample_no+1}/{len(plan["data"]["samples"])}] {args.label} {sample["id"]} {time.monotonic()-start_time:.1f}s', flush=True)
    write_json(output / 'completion.json', dict(status='complete', samples=len(plan['data']['samples']),
               sampling_rows=len(rows), path_rows=len(path_rows),
               path_errors=sum(bool(r['errors']) for r in path_rows)))


if __name__ == '__main__':
    main()
