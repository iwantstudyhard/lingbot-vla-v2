"""CPU-only indexing and metrics; no model, simulator or CUDA import."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = tuple('observation.images.' + x for x in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'))
ARMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPS = np.array([6, 13])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_path(value, root=ROOT):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def dataset_path(config, override=None):
    if override:
        return local_path(override)
    manifest = local_path(config['data']['train_path'])
    lines = [x.strip() for x in manifest.read_text(encoding='utf-8').splitlines()
             if x.strip() and not x.lstrip().startswith('#')]
    if len(lines) != 1 or lines[0].split(maxsplit=1)[0] != 'robotwin':
        raise ValueError('Specify --dataset: expected one robotwin entry in the clean training manifest')
    return local_path(lines[0].split(maxsplit=1)[1])


def select_samples(dataset, episodes=4, seed=20261008):
    import pyarrow.parquet as pq
    dataset = Path(dataset)
    info = json.loads((dataset / 'meta/info.json').read_text(encoding='utf-8'))
    if (info.get('codebase_version') != 'v3.0' or info.get('total_frames') != 548893
            or info.get('total_episodes') != 2500):
        raise ValueError('This diagnostic is restricted to the audited clean v3 dataset: 548893 frames / 2500 episodes')
    columns = ['episode_index', 'length', 'dataset_from_index', 'dataset_to_index',
               'data/chunk_index', 'data/file_index']
    columns += [f'videos/{cam}/{key}' for cam in CAMERAS
                for key in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
    files = sorted((dataset / 'meta/episodes').rglob('*.parquet'))
    rows = [r for f in files for r in pq.read_table(f, columns=columns).to_pylist()]
    rows.sort(key=lambda r: r['episode_index'])
    if len(rows) != 2500 or len({r['episode_index'] for r in rows}) != 2500:
        raise ValueError('Episode metadata is incomplete or duplicated')
    if not 1 <= episodes <= len(rows):
        raise ValueError('--episodes must be in [1, 2500]')
    # Stratify by episode index, then sample once in each bin. Not a held-out split.
    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(bin_)) for bin_ in np.array_split(np.arange(len(rows)), episodes)]
    samples = []
    for i in selected:
        ep = rows[i]
        if ep['length'] != ep['dataset_to_index'] - ep['dataset_from_index'] or ep['length'] < 3:
            raise ValueError(f'Invalid episode extent: {ep["episode_index"]}')
        data_file = dataset / info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index'])
        if not data_file.is_file():
            raise FileNotFoundError(data_file)
        for phase in (0.0, 0.4, 0.8):
            offset = min(ep['length'] - 1, int(phase * ep['length']))
            videos = {}
            for cam in CAMERAS:
                stem = f'videos/{cam}/'
                path = dataset / info['video_path'].format(video_key=cam, chunk_index=ep[stem + 'chunk_index'], file_index=ep[stem + 'file_index'])
                if not path.is_file():
                    raise FileNotFoundError(path)
                videos[cam] = dict(path=str(path), start=ep[stem + 'from_timestamp'], end=ep[stem + 'to_timestamp'])
            samples.append(dict(id=f'ep{ep["episode_index"]:04d}_f{offset:04d}',
                                episode=ep['episode_index'], offset=offset,
                                index=ep['dataset_from_index'] + offset,
                                end=ep['dataset_to_index'], length=ep['length'],
                                data_file=str(data_file), videos=videos, fps=info['fps']))
    return dict(dataset=str(dataset), info_sha256=sha256(dataset / 'meta/info.json'),
                episode_metadata_sha256={f.relative_to(dataset).as_posix(): sha256(f) for f in files},
                seed=seed, interpretation='Existing TRAINING demonstrations, NOT held-out validation or rollout success',
                samples=samples)


def verify_selected_sources(data, norm_file, report_file=ROOT / 'docs/clean_training/norm_verification.json'):
    """Bind selected action/episode files to the independently audited clean copy."""
    report = json.loads(Path(report_file).read_text(encoding='utf-8'))
    stats = json.loads(Path(norm_file).read_text(encoding='utf-8'))
    if not report.get('comparison_passed') or stats.get('verification', {}).get('source_fingerprint') != report['source_fingerprint']:
        raise ValueError('Clean source verification provenance differs from this checkpoint')
    expected = {r['path']: r['sha256'] for r in report['data_sources']}
    dataset = Path(data['dataset'])
    actual = {'meta/info.json': data['info_sha256'], **data['episode_metadata_sha256']}
    actual.update({Path(p).relative_to(dataset).as_posix(): digest for p, digest in data['selected_data_sha256'].items()})
    for path, digest in actual.items():
        if expected.get(path) != digest:
            raise ValueError(f'Selected clean source checksum mismatch: {path}')
    data['verified_selected_source_files'] = len(actual)
    data['tasks_sha256'] = sha256(dataset / 'meta/tasks.parquet')
    data['source_verification_note'] = 'Selected action parquet + all episode metadata + info match clean norm audit. Video pixels are decoded from the supplied existing dataset, not compared to a canonical full-video hash.'


def action_chunk(table, sample, horizon):
    """Same action[t:t+H] convention as VLADataset; tail repeats are masked."""
    import pyarrow.compute as pc
    start, end = int(sample['index']), min(int(sample['index']) + horizon, int(sample['end']))
    rows = table.filter(pc.and_(pc.greater_equal(table['index'], start), pc.less(table['index'], end))).to_pylist()
    rows.sort(key=lambda r: r['index'])
    if [r['index'] for r in rows] != list(range(start, end)):
        raise ValueError('Missing/non-contiguous action rows; a chunk must not cross parquet/episode boundaries')
    if not rows or any(r['episode_index'] != sample['episode'] for r in rows):
        raise ValueError('Action chunk crossed an episode boundary')
    first = rows[0]
    if first['frame_index'] != sample['offset']:
        raise ValueError('Frame metadata/index mismatch')
    actions = np.asarray([r['action'] for r in rows], dtype=np.float32)
    state = np.asarray(first['observation.state'], dtype=np.float32)
    if actions.shape != (len(rows), 14) or state.shape != (14,) or not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError('Expected finite physical 14D RoboTwin joints')
    pad = np.arange(horizon) >= len(actions)
    actions = np.concatenate([actions, np.repeat(actions[-1:], horizon - len(actions), axis=0)])
    return first, state, actions, pad


def action_metrics(pred, target, pad):
    pred, target, pad = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64), np.asarray(pad, dtype=bool)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 14 or pad.shape != (len(pred),):
        raise ValueError('Expected matching [H,14] actions and [H] padding')
    if not np.isfinite(pred).all() or not np.isfinite(target).all() or pad.all():
        raise ValueError('Non-finite or all-padded actions')
    result = {}
    for n, label in ((len(pred), 'full'), (min(20, len(pred)), 'first20')):
        p, y, valid = pred[:n], target[:n], ~pad[:n]
        result[label + '_arm_mae_rad'] = float(np.abs(p[valid][:, ARMS] - y[valid][:, ARMS]).mean())
        result[label + '_gripper_mae_native'] = float(np.abs(p[valid][:, GRIPS] - y[valid][:, GRIPS]).mean())
    valid = ~pad
    for order in (1, 2):
        window_valid = np.array([valid[i:i+order+1].all() for i in range(len(valid)-order)])
        if window_valid.any():
            dp, dy = np.diff(pred[:, ARMS], n=order, axis=0), np.diff(target[:, ARMS], n=order, axis=0)
            for key, value in (('pred', dp), ('expert', dy), ('error', dp-dy)):
                result[f'{key}_d{order}_rad_per_action{order}'] = float(np.abs(value[window_valid]).mean())
    result['first_arm_mae_rad'] = float(np.abs(pred[0, ARMS] - target[0, ARMS]).mean())
    result['valid_actions'] = int(valid.sum())
    return result


def vector_metrics(a, b, valid):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    valid = np.broadcast_to(valid, a.shape)
    if a.shape != b.shape or not valid.any() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid vector-field comparison')
    difference, ref = (a-b)[valid], b[valid]
    return dict(mae=float(np.abs(difference).mean()), max_abs=float(np.abs(difference).max()),
                relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(ref), 1e-8)))
