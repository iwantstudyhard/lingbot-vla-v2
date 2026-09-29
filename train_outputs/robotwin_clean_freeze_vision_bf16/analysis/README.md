# RobotWin BF16 4-GPU 训练分析

## 结论摘要

- 日志完整解析出 **14,991** 个逐步记录，覆盖 step **1–14991**。
- 前 500 步平均总损失为 **0.1963**，最后 500 步为 **0.0894**，相对变化 **-54.5%**。
- 日志最后更新时间为 **09/28/2026 19:24:44**；末尾没有正常训练结束或 checkpoint 保存记录，因此这是截断日志，不代表训练在该步失败。
- 配置 `save_steps=20000`；当前日志未覆盖 step 20000，故不能从这份日志恢复该检查点区间的指标。
- 模型 checkpoint 目录保持只读；所有图和表仅保存在本 `analysis/` 目录。

## 汇总报告

- `report.html`：可离线打开的交互式技术报告，包含关键指标、趋势图、窗口统计和配置差异。
- `artifact.json`：报告的可复现数据与版式定义。

## 图表索引

- `figures/01_loss_overview.png`：总损失与 VLA 损失，含 200 步移动平均。
- `figures/02_auxiliary_losses.png`：深度、未来深度、未来视频、MoE 辅助损失。
- `figures/03_weighted_loss_contributions.png`：按训练权重还原的损失贡献与总损失核对。
- `figures/04_learning_rates.png`：基础参数与 MoE 专家参数学习率。
- `figures/05_optimization_health.png`：梯度范数、单步耗时、教师前向耗时。
- `figures/06_moe_health.png`：MaxVio、路由 sigmoid、平衡损失与 z-loss。
- `figures/07_loss_distribution_by_phase.png`：五个训练阶段的损失分布。
- `figures/08_metric_correlation.png`：逐步指标相关性（仅描述相关，不代表因果）。

## 数据与配置

- `data/training_metrics.csv`：逐步原始指标与还原后的加权项。
- `data/training_metrics_rolling_200.csv`：移动平均指标。
- `data/window_summary.csv`：每 1000 步窗口统计。
- `data/outliers.csv`：基于 median + 6×MAD 的稳健异常候选。
- `data/summary.json`：机器可读汇总、完整性检查和告警计数。
- `configs/effective_logged_config.json`：日志启动时打印的最终有效参数。
- `configs/config_comparison.md`：官方、试运行、正式配置的语义差异。

## Checkpoint 可视化快照

快照仅写入 `analysis/by_checkpoint/`，绝不写入 `checkpoints/`。

| 名称 | 状态 | 日志覆盖到 |
|---|---|---:|
| `latest_partial_step_14991` | partial_log | 14991 |

## 重跑命令

```powershell
python tools/analyze_training_log.py `
  --log bf16_4gpu_formal.log `
  --config configs/vla/robotwin/robotwin_clean_freeze_vision_bf16_formal.yaml `
  --official-config configs/vla/robotwin/robotwin.yaml `
  --alternate-config configs/vla/robotwin/robotwin_clean_freeze_vision_bf16.yaml `
  --run-dir train_outputs/robotwin_clean_freeze_vision_bf16
```

把更完整的日志放回同一路径后重跑，脚本会自动识别 `checkpoints/global_step_*`，并在 `analysis/by_checkpoint/` 创建对应的只读分析快照。
