# RoboTwin Data and Evaluation

Paths follow [dir_standard.md](../../docs/dir_standard.md). RoboTwin is a pinned
submodule, with XPolicyLab pinned inside it. Initialize the recorded versions from
the lingbot-vla-v2 workspace:

```bash
git submodule update --init --recursive RoboTwin
export WORKSPACE="$PWD"
export ROBOTWIN_DIR="$WORKSPACE/RoboTwin"
```

Install the simulator dependencies using the existing RoboTwin installation guide.
Its scripts live in `RoboTwin/scripts/`, task configs in `RoboTwin/env_cfg/task_config/`,
and cuRobo installs into `RoboTwin/envs/curobo/`. Simulator assets stay under
`RoboTwin/assets/`.

## Download or Collect Trajectories

```bash
bash "$ROBOTWIN_DIR/scripts/download_xpolicylab_data.sh" adjust_bottle
# Optional custom collection (requires the simulator environment):
bash "$ROBOTWIN_DIR/collect_data.sh" adjust_bottle demo_clean 0
```

Both commands default to
`$WORKSPACE/datasets/RoboTwin/<task_config>/<task>/<embodiment>/`, retaining the
existing `data/`, `video/`, `instruction/`, seed and cache files. Use
`ROBOTWIN_DATA_ROOT` (legacy alias `XPOLICYLAB_DATA_ROOT`) to override this root.
Archive caching defaults to `<data root>/download_cache/` and supports
`HF_ARCHIVE_CACHE`. Relative local paths resolve against `WORKSPACE`.

## Convert to LeRobot

Use an environment with the matching LeRobot version. Existing converters preserve
the trajectory data and image decoding behavior; their input root matches download
and collection, and their output defaults to `$WORKSPACE/datasets/<repo_id>/`.
`HF_LEROBOT_HOME` remains an explicit override for the output root.

```bash
python "$ROBOTWIN_DIR/XPolicyLab/scripts/transform_lerobot_v21_format.py" \
  "demo_clean.*.aloha_agilex" --repo_id robotwin_demo_clean_aloha_agilex --max_episode 50
python "$ROBOTWIN_DIR/XPolicyLab/scripts/transform_lerobot_v30_format.py" \
  "demo_clean.*.aloha_agilex" --repo_id RoboTwin_lerobot_v30 --max_episode 50
```

The training manifests stay versioned in `assets/training_data/`. Point them at the
actual converted dataset directories under `datasets/`.

## Simulated Evaluation (Inference + RoboTwin Sim)

After training, evaluate your checkpoint on the 50 RoboTwin tasks (100 episodes each) with
[`start_robotwin_infer_and_eval.sh`](./start_robotwin_infer_and_eval.sh). It starts
`num_gpus * num_per_gpu` resident inference servers (one port each), then runs the sim tasks
in a **queue-scheduled** fashion against free slots — finishing a task frees its slot and
starts the next. All flags below also accept the equivalent environment variable
(`MODEL_DIR`, `ROBOTWIN_DIR`, `OUTPUT_DIR`, `CONDA_SH`, `QWEN3VL_DIR`, `INFERENCE_ENV`, `SIM_ENV`).
Paths follow [dir_standard.md](../../docs/dir_standard.md). Explicit flags take priority;
legacy `MODEL_PATH`, `EVAL_WORKDIR`, `OUTPUT_BASE`, and `QWEN3VL_PATH` remain supported.
`OUTPUT_DIR` is the artifact root; `--output_base`/`OUTPUT_BASE` are already the evaluation
category directory and do not receive another `eval_outputs` suffix.

### Prerequisites

1. **RoboTwin repo** cloned and its dependencies installed. The launcher
   needs the repo **root** path (the one containing `envs/`, `assets/`, `env_cfg/`, `scripts/`).
2. **Two conda environments**:
   - inference side — e.g. `lingbotvla` (PyTorch + this repo's model code).
   - sim side — `RoboTwin` (sapien / mplib / curobo / open3d …), built against numpy 1.26.x.
3. **Qwen3-VL backbone** checkpoint used by the VLA vision-language encoder (`QWEN3VL_DIR`).
4. **Your trained HF checkpoint** (`model_path`), e.g. `.../global_step_xxxxx/hf_ckpt`.

> [!IMPORTANT]
> Release validation of the published checkpoint uses **FP32 inference**. The lower-memory
> BF16 mode is useful for pipeline checks, but can produce materially different success
> rates.

> The launcher **auto-copies** the eval client (`eval_policy_client_lingbotvla.py` + the small
> `deploy/` helpers) from this repo into `<RoboTwin>/script/`, and **self-heals** the curobo
> embodiment `.yml` and editable-install `.pth` paths to point at the RoboTwin checkout you
> pass. The synchronized files are ignored by the RoboTwin repository; their source remains versioned in the main project.

### Run the full benchmark (50 tasks)

> Substitute every `/path/to/...` and the conda env / `conda.sh` path for your machine.

```bash
# Defaults are anchored to the workspace, independently of the caller's cwd.
QWEN3VL_DIR="$PWD/models/Qwen3-VL-4B-Instruct" \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path     /path/to/your/checkpoint/hf_ckpt \
    --conda_sh       /path/to/miniconda3/etc/profile.d/conda.sh \
    --inference_env  lingbotvla \
    --sim_env        RoboTwin \
    --task_config    demo_clean \
    --num_tasks 50 --num_gpus 4 --num_per_gpu 1
```

Use `--task_config demo_randomized` for the randomized benchmark.

**GPU / concurrency**
- `num_gpus` × `num_per_gpu` = number of concurrent sim slots (one inference server per slot).
- `--num_per_gpu 1` is the safe starting point. In our current software stack, one FP32
  policy server plus its simulator uses roughly 32 GB, so leave additional headroom.
- A 24 GB GPU generally requires BF16 (`--use_bf16 True --use_fp32 False`), which does not reproduce the published FP32 benchmark. Increasing `num_per_gpu` can OOM; the script retries each task up to 3 times, but persistent OOM skips the task.

### Smoke test (1 task, 1 GPU)

Verify the pipeline end-to-end without waiting for the full run:

```bash
QWEN3VL_DIR="$PWD/models/Qwen3-VL-4B-Instruct" \
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path   /path/to/your/checkpoint/hf_ckpt \
    --conda_sh     /path/to/miniconda3/etc/profile.d/conda.sh \
    --task_config  demo_clean \
    --num_tasks 1 --num_episodes 1 --num_gpus 1 --num_per_gpu 1
```

The run dir is printed at startup (`Run directory: ...`). You should see `Success rate: N/N =>
...` lines appear in the task log. The smoke command runs one episode. Full evaluation
defaults to **100 episodes per task** when `--num_episodes` is omitted.

### Output layout

```
outputs/eval_outputs/<exp>_<step>k_<task_config>_<timestamp>/
├── stats.txt                 # final per-task table + overall success rate
├── inference_pids.txt
├── eval_pids.txt
├── inference_logs/           # one log per inference server / port
├── eval_logs/                # one log per task (per-step progress, success rate)
└── eval_results/             # per-task videos: episodeN_success.mp4 ...
```

### Useful flags

| flag | meaning |
|------|---------|
| `--no_video` | disable per-episode video recording (faster, no videos saved) |
| `--keep_inference` | leave inference servers resident after sim finishes |
| `--start_port` | base port for inference servers (default 9330, slot *i* uses base + i) |
| `--num_episodes` | episodes evaluated per task (default 100; use 1 for a smoke test) |
| `--progress_interval` | seconds between task/episode progress reports in the launcher terminal (default 30) |
| `--use_length` | actions executed before observing/replanning (default 10; model predicts a 50-action horizon) |
| `--server_ready_timeout` | seconds to wait for every policy server's `/healthz` endpoint before starting simulation (default 1800) |
| `--robo_name` | robot config name (default `robotwin`) |
| `--task_config` | RoboTwin setting: `demo_clean` or `demo_randomized` |
| `--use_bf16` / `--use_fp32` | inference precision; release reproduction uses `False` / `True` |
| `--use_compile` | enable lazy `torch.compile` (default `True`; first request takes longer) |
| `--inference_script` | inference-side module (default `deploy/lingbot_vla_v2_policy.py`) |

The default executes 10 actions from each 50-action prediction and then replans
from a fresh observation. The evaluation client still renders one frame per
executed simulator action, so saved videos show continuous motion instead of one
frame repeated for an entire action chunk. Set `--use_length 5` or `1` for a more
reactive (but slower) closed loop; use `50` only when intentionally reproducing
the original open-loop chunk setting.

### Optional action smoothing (diagnostic mitigation, not a training fix)

Defaults remain unchanged: `--action_smoothing none` sends the raw predictions.
To test smoothing, append:

```bash
--action_smoothing ema --smoothing_alpha 0.35 --smoothing_window 5 --smoothing_max_delta 0.05
```

This filters **physical absolute targets of the 12 arm joints only**. Grippers at
indices 6 and 13 are untouched. The 5-action centered average uses the already
returned chunk (no extra inference or resampling). EMA persists across chunk
boundaries and resets to the observed joint targets at each episode start.
`max_delta` caps each joint's target change in rad/action, **not rad/s**; 0 disables
the cap. The filter introduces lag and can harm timing or contacts. It cannot
correct the wrong task/pose or guarantee a higher success rate.

Keep model, precision, `use_length`, scenes and episode count identical for the
baseline and filtered comparison. Use a new `--output_base` category for each.
`eval_results/<task>/action_smoothing.jsonl` records episode/scene seed, raw targets,
windowed targets and commands actually submitted to the simulator when enabled,
including runs with `--no_video`. `stats.txt` records all filter settings.

To select physical GPUs 2 and 3 reliably, use `--num_gpus 2 --gpu_ids 2,3`; setting
only `CUDA_VISIBLE_DEVICES` is insufficient because the launcher assigns devices
per process. The GPU count must match the list; duplicate IDs are rejected.

Run CPU checks: `python tests/test_action_smoothing.py`. These tests establish
filter mechanics, not real evaluation success.

### Monitor / stop

```bash
RUN=outputs/eval_outputs/<exp>_<step>k_<task_config>_<timestamp>

# overall progress (done/skip/fail summary)
sed 's/\x1b\[[0-9;]*m//g' $RUN/eval_logs/*.log | grep -E "Success rate" | tail

# per-task latest success rate
for f in $RUN/eval_logs/*.log; do
  printf "%-24s %s\n" "$(basename $f .log)" \
    "$(grep -aoE 'Success rate: [0-9]+/[0-9]+ => [0-9.]+%' "$f" | tail -1)"
done
```

Stop everything with `Ctrl-C` (the launcher traps it and kills all child processes), or kill
the launcher plus any lingering `deploy.lingbot_vla_v2_policy` / `eval_policy_client_lingbotvla.py`
processes.
