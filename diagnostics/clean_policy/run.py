"""Three offline diagnostics on existing clean demonstrations, with owned GPU workers."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from diagnostics.clean_policy.common import dataset_path, select_samples, verify_selected_sources, write_json, sha256, local_path


def discover_models(run, explicit, official):
    models = []
    if run:
        for p in (local_path(run) / 'checkpoints').glob('global_step_*/hf_ckpt'):
            match = re.fullmatch(r'global_step_(\d+)', p.parent.name)
            if match:
                models.append((f'step_{int(match[1]):06d}', p))
        models.sort(key=lambda item: item[0])
        if not models:
            raise ValueError('No exported hf_ckpt in --train-run; DCP-only folders are not loadable here')
    for value in explicit:
        label, sep, path = value.partition('=')
        if not sep or label == 'official_reference' or not re.fullmatch(r'[A-Za-z0-9_-]+', label):
            raise ValueError('--model expects safe_label=/absolute/path/hf_ckpt')
        models.append((label, local_path(path)))
    if official:
        models.append(('official_reference', local_path(official)))
    if not models or len({label for label, _ in models}) != len(models):
        raise ValueError('Provide --train-run or --model, with unique labels')
    return models


def stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-run', help='Compare ALL retained exported checkpoints in this run')
    parser.add_argument('--model', action='append', default=[], help='Repeat label=/path/hf_ckpt')
    parser.add_argument('--official', help='Optional mixed-data reference; not an equal-data baseline')
    parser.add_argument('--official-norm', default=str(ROOT / 'assets/norm_stats/robotwin.json'))
    parser.add_argument('--dataset', help='Existing original clean v3 dataset; otherwise resolve training manifest')
    parser.add_argument('--qwen', required=True)
    parser.add_argument('--gpus', default='2,3', help='Physical GPU IDs; at most one model per GPU')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--episodes', type=int, default=4, help='Three existing observation frames per episode')
    parser.add_argument('--sample-seed', type=int, default=20261008)
    parser.add_argument('--noise-seeds', default='42,43')
    parser.add_argument('--denoising-steps', default='10,30')
    parser.add_argument('--path-samples', type=int, default=3, help='First N observations per model get 4-time path probes')
    parser.add_argument('--video-backend', choices=['torchcodec', 'pyav'], default='torchcodec')
    parser.add_argument('--timeout', type=int, default=7200, help='Hard per-model worker timeout seconds')
    parser.add_argument('--output', default=str(ROOT / 'outputs/eval_outputs/clean_policy_diagnostics'))
    parser.add_argument('--prepare-only', action='store_true', help='CPU-only metadata/normalization preflight; no model import')
    args = parser.parse_args()
    gpu_ids = args.gpus.split(',')
    seeds = [int(x) for x in args.noise_seeds.split(',')]
    steps = [int(x) for x in args.denoising_steps.split(',')]
    if (not all(x.isdigit() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)
            or len(set(seeds)) != len(seeds) or any(x < 0 or x >= 2**32 for x in seeds)
            or any(x < 1 or x > 100 for x in steps) or len(set(steps)) != len(steps)
            or args.path_samples < 1 or args.timeout < 1):
        parser.error('Invalid/duplicate GPUs, seeds, denoising steps, path-samples or timeout')
    qwen = local_path(args.qwen)
    if not (qwen / 'config.json').is_file():
        parser.error(f'Qwen config.json missing: {qwen}')
    from tools.clean_training_common import validate_hf
    from diagnostics.action_compare.run import normalization_preflight
    import yaml
    specs = []
    source_config = None
    for label, path in discover_models(args.train_run, args.model, args.official):
        hf = validate_hf(path)
        config_file = hf.parent.parent.parent / 'lingbotvla_cli.yaml'
        config = yaml.safe_load(config_file.read_text(encoding='utf-8'))
        norm, metadata = normalization_preflight(hf, local_path(args.official_norm) if label == 'official_reference' else None)
        if label != 'official_reference':
            if not config['data'].get('require_normalization_contract'):
                raise ValueError(f'{label}: requires audited clean stage1 normalization contract')
            if config['data'].get('image_augment') or config['data'].get('stage2_augmentation_config'):
                raise ValueError(f'{label}: this diagnostic reproduces unaugmented stage1 preprocessing only')
            source_config = source_config or config
        specs.append(dict(label=label, hf=str(hf), norm=str(norm), normalization=metadata,
                          config=config, config_sha256=sha256(config_file)))
        print(f'[preflight] {label}: normalization {metadata}', flush=True)
    if source_config is None:
        parser.error('At least one clean-trained checkpoint is required')
    norms = {s['normalization']['semantic_sha256'] for s in specs if s['label'] != 'official_reference'}
    if len(norms) != 1:
        raise ValueError('Clean checkpoints must use exactly the same normalization')
    data = select_samples(dataset_path(source_config, args.dataset), args.episodes, args.sample_seed)
    data['selected_data_sha256'] = {p: sha256(p) for p in sorted({s['data_file'] for s in data['samples']})}
    verify_selected_sources(data, next(s['norm'] for s in specs if s['label'] != 'official_reference'))
    run = local_path(args.output) / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    plan = dict(models=specs, data=data, precision=args.precision, noise_seeds=seeds,
                denoising_steps=steps, path_samples=min(args.path_samples, len(data['samples'])),
                video_backend=args.video_backend, qwen=str(qwen), gpus=gpu_ids,
                interpretation='Offline training-set action accuracy, NOT task success; official used more/different training data')
    write_json(run / 'plan.json', plan)
    from diagnostics.clean_policy.training_audit import audit, save_audit
    save_audit(run / 'training_audit', audit(source_config))
    print(f'Run directory: {run}\nDataset: {data["dataset"]}\nSamples: {len(data["samples"])}', flush=True)
    if args.prepare_only:
        print('PREPARE ONLY: no GPU/model/video decoding performed. Re-run without --prepare-only to measure.')
        return
    queued, active, failures = list(specs), {}, []
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    if os.name == 'posix':
        signal.signal(signal.SIGTERM, interrupt)
    try:
        while queued or active:
            for gpu in gpu_ids:
                if queued and gpu not in active:
                    spec = queued.pop(0)
                    env = dict(os.environ, WORKSPACE=str(ROOT), CUDA_VISIBLE_DEVICES=gpu,
                               QWEN3VL_DIR=str(qwen), QWEN3VL_PATH=str(qwen), PYTHONUNBUFFERED='1')
                    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
                    log = (run / f'{spec["label"]}.log').open('w', encoding='utf-8')
                    proc = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'diagnostics.clean_policy.worker',
                                             '--plan', str(run / 'plan.json'), '--label', spec['label']],
                                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=os.name == 'posix')
                    active[gpu] = (proc, spec['label'], log, time.monotonic())
                    print(f'[launch] {spec["label"]}: physical GPU {gpu}, log {log.name}', flush=True)
            for gpu, (proc, label, log, started) in list(active.items()):
                elapsed = time.monotonic() - started
                if elapsed > args.timeout and proc.poll() is None:
                    stop_owned(proc)
                    print(f'[timeout] {label}', flush=True)
                if proc.poll() is not None:
                    log.close()
                    if proc.returncode:
                        failures.append(dict(label=label, exit_code=proc.returncode))
                    print(f'[exit] {label}: {proc.returncode}', flush=True)
                    del active[gpu]
                else:
                    progress = run / label / 'progress.json'
                    try:
                        stage = json.loads(progress.read_text(encoding='utf-8'))
                    except (OSError, ValueError):
                        stage = {'stage': 'loading weights'}
                    print(f'[progress GPU {gpu}] {label}: {stage}, elapsed {int(elapsed)}s', flush=True)
            if active:
                time.sleep(10)
    finally:
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        previous_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN) if os.name == 'posix' else None
        try:
            for proc, label, log, _ in active.values():
                stop_owned(proc)
                log.close()
            print(f'Owned workers stopped. Evidence retained: {run}', flush=True)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
    from diagnostics.clean_policy.report import build_report
    build_report(run)
    path_errors = sum(json.loads(p.read_text(encoding='utf-8')).get('path_errors', 0)
                      for p in run.glob('*/completion.json'))
    write_json(run / 'completion.json', dict(
        status='partial_failure' if failures else ('sampling_complete_paths_incomplete' if path_errors else 'complete'),
        failures=failures, path_errors=path_errors))
    if failures:
        raise SystemExit('Some GPU workers failed. See per-model .log; report is explicitly partial.')
    if path_errors:
        print(f'WARNING: sampling completed, but {path_errors} path probes have unsupported/failed branches. NOT a forward-equivalence pass.', flush=True)
    print(f'Read {run / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
