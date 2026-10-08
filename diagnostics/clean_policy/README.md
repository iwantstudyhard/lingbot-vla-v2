# 低 loss、动作却抖：三个独立诊断

这里只读模型和**已有官方 clean 示教**，不启动 RoboTwin，不 rollout，不生成训练样本。
默认 BF16、无 compile、无平滑。第一阶段和第二阶段训练代码、配置都没有改。

## 直接运行（服务器）

确认 Git 拉取了本次新增文件，然后执行。`--train-run` 会自动找出该 run 内**所有仍保留的 hf_ckpt**，不用手填每一个检查点；DCP-only 不会当作 HF 权重。
官方模型仅作参考（它用了更多 clean+random 数据），各自沿用自己的归一化，绝不混用。

```bash
cd /scratch/lingbot_ws/lingbot-vla-v2
conda activate lingbotvla

python -B -m diagnostics.clean_policy.run \
  --train-run "$PWD/outputs/train_outputs/robotwin_clean_stage1_20261001_021839_b6a91337" \
  --official "$PWD/models/lingbot-vla-v2-6b-robotwin/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt" \
  --official-norm "$PWD/assets/norm_stats/robotwin.json" \
  --qwen "$PWD/models/Qwen3-VL-4B-Instruct/snapshots/master" \
  --gpus 2,3 \
  --episodes 4 \
  --noise-seeds 42,43 \
  --denoising-steps 10,30 \
  --path-samples 3
```

数据集默认从这个训练 run 记录的 `data.train_path` 清单读取。若服务器搬过路径，明确追加：

```bash
  --dataset /scratch/lingbot_ws/lingbot-vla-v2/datasets/RoboTwin_lerobot_v30
```

请使用真实目录，不要为了跑通随意复制一份不同数据。可先在上述命令末尾加 `--prepare-only`，只检查配置、归一化契约、选中 clean 数据/视频路径，不占 GPU；它不检查视频能否解码或权重能否完成前向。
环境是已有能训练、推理的 `lingbotvla`，使用相同视频解码依赖；不是 `RoboTwin` 环境。默认 torchcodec，可显式 `--video-backend pyav`，不要无依据换解码后端。

每张 GPU 同时只加载一个模型；多个检查点排队运行。终端每 10 秒显示加载/解码/采样进度；细节在 `step_*.log` / `official_reference.log`。
Ctrl+C 只终止本命令自己的 worker，不重试、不重启，不清理其他训练/评测进程。

快速试跑：改为 `--episodes 1 --path-samples 1`（3 个已有输入，仍保留两个种子及两个采样步数）。先确认该小规模能完成，再扩大。
只测指定检查点时，不用 `--train-run`，改为一个或多个 `--model ours18k=/完整路径/hf_ckpt`。

## 三项具体测什么

1. **前向路径对照**：同权重、同输入、同噪声、同 `x_t`，在 t=0.1/0.5/0.9/1.0 比较实际训练 full forward 和部署 cached velocity。
   分别隔离 cache、eager/flex attention、train/eval MoE kernel。`no_grad` 下使用 train 标志不等于训练更新；没有 backward/optimizer/FSDP 检查。
   关闭的只是辅助 loss 计算，不移除辅助任务 token 或改变动作 backbone。分支报错会保留异常并标明未完成，不能当作通过。
2. **完整动作采样**：从纯噪声采样50动作，和同一 clean 示教的 `action[t:t+50]` 比较。
   统计物理关节误差（rad）、夹爪误差（原单位）、动作差分误差、多个噪声种子的动作 std；同时对照 10/30 采样步。
   专家尾部 padding 不计分。归一化目标由现有 FeatureTransform 产生，先核对反归一化回环。
   另外保存真实训练/部署预处理的输入差异；比如 uint8 与 float resize 可能有微小差异，需要量化而不是猜。
3. **检查点比较**：同一组输入/种子，逐个评估保留的检查点。若只有一个本方检查点，就不能判断时间趋势。
   三个很接近的末期检查点也只能说明局部趋势，不能证明从头到尾的收敛。

这些是**训练集诊断**，不是 held-out validation，也不是任务成功率。12个输入只是排查起点，不代表全部任务。
默认按 episode 序号分层，选4个 episode，每个取起始/40%/80%阶段；样本、原始数据哈希、配置、归一化哈希写入 plan.json。
选中的动作 parquet、episode 元数据和 info.json 还会与已独立核验的 clean 来源哈希核对，不仅检查总帧数。
没有保存全部官方视频的规范哈希，因此不声称视频内容已做全量比特级核验。
比较模型加载的是各自 HF 导出，而非 DCP 内存权重；无法单凭此工具证明导出过程无误。

## 看哪里

终端打印唯一目录 `outputs/eval_outputs/clean_policy_diagnostics/<时间_UUID>/`：

- `REPORT.md`：先读这个，包括失败/未完成的分支。
- `forward_path_comparison.png`：cache、attention、train/eval kernel 等前向差异随噪声时刻的变化。
- `checkpoint_comparison.png`：动作 MAE、二阶差分误差、换种子的 std。
- `actions_*.png`：12个物理臂关节的专家/各模型曲线，不含平滑。
- `summary.csv` / `summary.json`：全部指标与路径误差；逐模型 `paths.jsonl` 记录每个 t 的 FM L1。
- `<模型>/<sample>/sample.json`：三个视角、时间、task、预处理差异、归一化回环误差、保持当前关节不动的误差基线。
- `<模型>/<sample>/*.npz`：原始预测、专家动作、向量场、噪声，不只保留图。
- `training_audit/audit.json`：实际 LR 计划、epoch 曝光次数和有条件续训建议。

需要重画已生成的结果，不启动 GPU：

```bash
python -B -m diagnostics.clean_policy.report --run /完整路径/时间_UUID
```

## 学习率审阅与后续（先诊断，暂不启动）

18k、global batch32，约 1.049 次起始帧曝光；cosine 的 540 步 warmup 后从1e-5降到1e-6。
路由专家组倍率 sqrt(32/4)=2.828。**当前 Muon builder 没有使用 `vit_lr` 做独立视觉组**；视觉未冻结，但跟基础组 LR。
Muon 的矩阵更新还有形状相关的 LR 调整，不能把不同优化器的数字直接等同。
梯度累计代码是每 microbatch除以累计次数后 backward，完成一次累计才 clip/step/scheduler.step；当前没看到额外除一次 global batch 的问题。
历史 run 没记录代码 commit 时，不能据当前文件保证当时逐字相同。

单独审阅保存配置（CPU，不训练）：

```bash
python -B -m diagnostics.clean_policy.training_audit \
  --run "$PWD/outputs/train_outputs/robotwin_clean_stage1_20261001_021839_b6a91337" \
  --output "$PWD/outputs/eval_outputs/training_lr_audit_18k"
```

只有当前向路径接近、完整采样/检查点趋势支持训练不足，再试**额外5000步 clean-only**：
选较好 HF 权重，开新目录，重新初始化 optimizer/scheduler，基础 LR 1e-5→5e-6，warmup100步，global batch32，4卡micro1/accum8。
保持视觉解冻、mask、归一化、辅助 loss 不变，暂不叠加第二阶段增强或新平滑 loss。
每500步保存，最多3个，原来“best”仍是最低平均训练loss，**不是成功率最佳**。

**不要直接给旧 run 加 `--resume-run` 并改 lr 就以为学习率重启了**：DCP 会恢复旧 optimizer/scheduler/global_step。
也不要在诊断结果出来前一次改 LR、loss、增强、采样、控制器；这样无法辨认效果来自哪一个改动。
本次没有修改任何正式训练超参数，也没有发起续训。
