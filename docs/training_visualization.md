# Training visualization and checkpoint isolation

The RobotWin training job uses two deliberately separate roots on the server.

## Server layout

```text
/scratch/YF_Data/lingbot_workspace/
├── lingbot-vla-v2/                              # maintained Git repository
│   ├── training_logs/                           # unique stdout/stderr logs
│   └── train_outputs/
│       └── robotwin_clean_freeze_vision_bf16/
│           └── runs/
│               └── 20260929_143012_a1b2c3d4/   # unique directory per launch
│                   ├── lingbotvla_cli.yaml       # effective training config copy
│                   └── analysis/
│                       ├── data/
│                       │   ├── training_metrics_live.jsonl
│                       │   ├── training_metrics.csv
│                       │   ├── training_metrics_rolling_200.csv
│                       │   ├── window_summary.csv
│                       │   └── summary.json
│                       ├── figures/              # latest full-run figures
│                       └── by_checkpoint/
│                           ├── index.csv
│                           ├── global_step_20000/ # eight checkpoint-scoped snapshots
│                           └── global_step_40000/
└── train_outputs/
    └── robotwin_clean_freeze_vision_bf16/
        └── checkpoints/                          # model/optimizer checkpoints only
            ├── global_step_20000/
            └── global_step_40000/
```

The visualizer only lists checkpoint directory names under the external model
run. It never creates, edits, moves, or deletes anything below `checkpoints/`.

## Automatic behavior

Rank 0 appends one structured JSONL record per optimizer step using a background
writer. Records are flushed every 100 steps by default and always flushed before
a checkpoint visualization is rendered.

If training resumes from an earlier step, the renderer keeps the valid prefix
and discards the superseded metric tail when building plots; the append-only
JSONL source itself remains intact for audit and recovery.

After each successful checkpoint save (step interval, final `max_steps`, or epoch
interval), rank 0 generates:

1. eight latest full-run figures;
2. eight figures frozen at that checkpoint step;
3. raw and rolling CSV files;
4. 1000-step window statistics, outlier candidates, and JSON summaries;
5. a checkpoint index and effective-config copy;
6. `analysis/report.html` plus a self-contained gallery page for each checkpoint.

Rendering is best-effort and isolated in a subprocess. A plotting failure is
logged but does not invalidate or delete a successfully saved model checkpoint.

## Configuration

The formal configuration enables the feature explicitly:

```yaml
train:
  enable_training_visualization: true
  training_visualization_flush_steps: 100
  training_visualization_rolling_window: 200
```

The experiment name is not hard-coded. By default the code takes the final
directory name from `train.output_dir`, then creates a unique directory for each
launch:

```text
<repository>/train_outputs/<experiment-name>/runs/<timestamp>_<random-id>/analysis
```

Two launches using the same training configuration therefore never overwrite
each other's metrics or plots. `train.sh` exports the same unique run ID to all
torchrun workers and also includes it in the stdout log filename.
Directory creation is atomic (`exist_ok=False`); if a manually supplied run ID
already exists, training stops before the optimizer loop instead of overwriting
the previous visualization data.

`training_visualization_output_dir` remains available as an optional base
directory override. When supplied as a relative path, it is resolved against
the Git repository root; the unique `runs/<run-id>` portion is still appended.

## Training logs

`train.sh` now writes a new timestamped log under `training_logs/` for every run.
Set `TRAIN_LOG_FILE` to choose an explicit path when needed. The structured JSONL
metric log is independent of stdout/stderr, so losing a console log no longer
destroys the data needed for plots.
