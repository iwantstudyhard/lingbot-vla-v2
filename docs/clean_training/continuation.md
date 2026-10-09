# 14 天期限下的 clean 续训

这是阶段一的**可选独立入口**，不是重新从头训练，也不是第二阶段增强。普通阶段一和第二阶段默认都不启用这个学习率机制。

## 本次安排

- 从完整的 `global_step_19500` 恢复模型、optimizer、数据位置及代码原有 RNG 状态；不清空 optimizer 动量。
- 只重建 scheduler：前 200 步从检查点实际 LR 平缓升到原来的峰值 `1e-5`，之后 cosine 缓慢降到 `5e-6`，到**总步数 48000**停止。expert 的相对 LR 倍率保留。
- 仍然使用 clean 数据、已验证归一化、BF16、4 卡、micro batch 1、accumulation 8、global batch 32；视觉继续解冻，动作 horizon 50、padding mask 与 loss 不改。不自动提高 batch 或改采样方式。
- 每 500 步保存与生成可视化。保留 19500 基线、窗口平均**训练 loss** 最佳、最新检查点，最多 3 个已完成目录；重复角色不占两份。异步 HF 导出期间仍可能暂时多出目录，需要预留空间。“最佳”不是闭环成功率最佳。
- LR 计划跟随 scheduler 写入分布式检查点；以后中断恢复沿用这次计划，不重复 warmup。运行根目录配置会记录这些有效参数。`--dry-run` 不改文件、不启训练。

## 时间账

当前约 17152 个 optimizer step 一个 epoch，19500 约 1.14 epoch。按 34 秒/步估计：

| 目标总步数 | 后续步数 | 仅训练时间 |
|---|---:|---:|
| 48000，约 2.80 epoch | 28500 | 约 11.22 天 |
| 137216，约 8 epoch | 117716 | 约 46.32 天 |

48000 为 14 天截止前留出约 2.8 天的毛余量，检查点导出、故障和最终评测也要占时间。这不是承诺 11.22 天必然完成；以实际日志的每步耗时更新估计。不能为了凑到 8 epoch 盲目减小 batch：epoch 数相同但每次更新统计、总更新数和速度都改变了。

## 现在启动

先将本地新代码手动提交/推送，再在服务器拉取该分支。确保没有另一个训练进程占用 0–3 卡/62500 端口：

```bash
cd /scratch/lingbot_ws/lingbot-vla-v2
git pull --ff-only origin codex/action-comparison-diagnostics
conda activate lingbotvla
export WORKSPACE="$PWD"
export TRAIN_RUN="$WORKSPACE/outputs/train_outputs/robotwin_clean_stage1_20261001_021839_b6a91337"
export QWEN3VL_DIR="$WORKSPACE/models/Qwen3-VL-4B-Instruct/snapshots/master"
export QWEN3VL_PATH="$QWEN3VL_DIR"
unset TRAIN_LOG_FILE TRAIN_LOG_DIR LINGBOT_TRAIN_RUN_ID LINGBOT_TRAIN_RUN_DIR

bash tools/train_clean_continue.sh \
  --resume-run "$TRAIN_RUN" --resume-step 19500 \
  --until 48000 --peak-lr 1e-5 --min-lr 5e-6 --warmup-steps 200 \
  --gpus 0,1,2,3 --master-port 62500
```

启动后应看到 `Restored optimizer state`、成功加载 19500 和 `[continuation_lr]`；第一步实际更新仍用已保存 LR，随后 200 步上升。tqdm 从 19500/48000 开始。不能只看到预检查成功就认为所有训练 tensor 已加载。

完整保存的配置包含 `basic_modules: []` 等空列表和 `null`。参数解析器现在直接保留这些类型，不再将空列表拼成无值的 `--model.basic_modules`，也不将 `null` 转成字符串路径。若看到 `expected at least one argument`，先拉取这个修复；不要往列表填虚假的模块名，也不用删除/改写运行配置、归一化或检查点。

如果希望先预览，加 `--dry-run`，看完去掉即可。不要使用 `--init-hf` 代替完整恢复，也不要删 optimizer 文件。不要另外从旧 19500 重复启动同一个目录。

## 以后再次中断

选择新的完整检查点，例如保存到了 25000：

```bash
bash tools/train_clean_continue.sh \
  --resume-run "$TRAIN_RUN" --resume-step 25000 \
  --gpus 0,1,2,3 --master-port 62500
```

不再指定新 LR 或终点，读取保存的同一计划，从 25000 接着走，原点仍是 19500。不能拿尚未启用本计划的旧 20000 等检查点冒充本计划中途恢复。计划不匹配会拒绝，避免悄悄改学习率。

## 不占用主训练 GPU 的验收安排

19500 是固定对照。到 25000、30000、36000、42000 时，用同样 clean 示教输入、随机种子、采样步数做完整动作误差诊断，再用相同任务/种子、相同平滑与 use_length 做少量闭环对照。闭环评测的 rollout **只用于评测，不写入训练集**。

优先在独立评测 GPU/朋友的服务器上做；主训练 4 卡全部在用，不能再在同一卡上启动评测模型。如果没有额外卡，在完整检查点保存后短暂停训评测，再完整 resume，不需要等待耗时全量评测。

本次 LR 调整是有记录的有限实验，不保证性能改善。不要仅按训练 loss 最小选最终模型。若 25000/30000 固定样本动作误差和闭环均明显不改善/恶化，应保留基线、优先检查目标/推理一致性，不能一直等到 8 epoch。第二阶段增强入口已经独立，先保证 clean 动作基本可用再做小规模增强，不同时更换 loss、归一化和视觉增强来掩盖主问题。
