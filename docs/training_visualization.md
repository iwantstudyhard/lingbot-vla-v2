# Training visualization and checkpoint isolation

The workspace layout is defined by [dir_standard.md](dir_standard.md). Training
artifacts share one run directory; the internal visualization layout is retained.

```text
outputs/train_outputs/<experiment>_<run_id>/
├── <config>_<run_id>.log
├── lingbotvla_cli.yaml
├── checkpoints/global_step_N/
├── model_assets/
├── normalization/                       # audited clean runs
├── runs/                                # TensorBoard
└── visualizations/runs/<run_id>/
    ├── lingbotvla_cli.yaml
    └── analysis/
        ├── data/training_metrics_live.jsonl
        ├── figures/
        ├── configs/
        ├── report.html
        └── by_checkpoint/global_step_N/
            └── augmentation/            # stage two only
```

`OUTPUT_DIR` overrides the `outputs/` root. An explicit `--train.output_dir`
selects the final run directory without adding another classification directory.
The visualizer inspects checkpoint names and never writes into checkpoints.

## Automatic behavior

Rank 0 appends one structured JSONL record per optimizer step using a background
writer. Records are flushed every 100 steps by default and always flushed before
a checkpoint visualization is rendered.

If training resumes from an earlier step, the renderer keeps the valid prefix
and discards the superseded metric tail when building plots; the append-only
JSONL source itself remains intact for audit and recovery.

After each successful checkpoint save (step interval, final `max_steps`, or epoch
interval), rank 0 generates:

1. nine latest full-run figures;
2. nine figures frozen at that checkpoint step;
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

The default experiment label comes from the configuration filename. `train.sh`
chooses one timestamp-and-UUID run ID and shares it across workers, console logs,
and visualizations. The default visualization location is:

```text
<train.output_dir>/visualizations/runs/<run_id>/analysis
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

`train.sh` now writes a new timestamped log inside the training run directory for every run.
Set `TRAIN_LOG_FILE` to choose an explicit path when needed. The structured JSONL
metric log is independent of stdout/stderr, so losing a console log no longer
destroys the data needed for plots.
