# 竞赛 clean-only 两阶段训练入口

2026-09-30。完整改动交接见 [官方对比与重构交接](../REFACTOR_HANDOFF.md)；增强细节见 [第二阶段说明](../../extensions/clean_stage2/README.md)。

## 先区分历史模型与新流程

历史 YAML 的确声明了 `robotwin_clean_only.json`。但修改前 `VLADataset` 未向 `FeatureTransform` 传 `norm_stats_path`，实际回退至机器人配置的 `robotwin.json`。仅检查 YAML 或 count 日志不能证明实际归一化来源。该漏接已修复，训练检查实际 transform 实例；新模型推理使用所属运行的统计快照。

历史两个统计文件 count 分别为 6,062,592 和 548,893。前者配合官方混合数据范例，不代表你加载了 random 视频。历史服务器若曾替换统计/代码，则应以当时运行文件为准；不要根据当前 checkout 给所有历史模型下绝对结论。

严格的新 clean-only 基线应从原始官方预训练权重重新进行 post-training，**不是随机初始化，也不加载旧模型的 optimizer/scheduler**。不要把旧混合统计模型换成新统计直接 resume。旧 checkpoints、日志和分析保留，不能视为新流程的阶段一模型。

## 训练前只修改这些路径

从仓库根目录、训练环境启动。当前保持原 /scratch 服务器布局，未自动迁移云服务器。

1. `assets/training_data/robotwin_clean_only.txt`：唯一一行 `robotwin /绝对路径/官方clean的LeRobot数据集`。
2. `configs/vla/robotwin/robotwin_clean_stage1.yaml` 的官方预训练 `model.model_path`。
3. 两阶段配置各自的 `model.tokenizer_path`、`align_params.depth.moge_path/morgbd_path`、`align_params.video.ckpt_path/config_path`。
4. 第二阶段配置独立位于 `extensions/clean_stage2/config.yaml`，不动态继承阶段一文件。

新运行默认全部位于仓库 `train_outputs/robotwin_clean_stage1|robotwin_clean_stage2/<时间戳_UUID>`。不更改以前外部 `train_outputs/.../global_step_20000` 的位置。每次启动自动新名字，不覆盖前次。

保留官方训练环境 Python 3.12/PyTorch 2.8 路线及现有 LeRobot v3/解码依赖；本地 CPU 测试不能代替服务器安装版本验收。启动器只验证权重/依赖路径、clean 文件指纹、元数据和统计，不加载 GPU 模型。

## 第一阶段：无在线增强的新 clean 基线

```bash
cd /scratch/YF_Data/lingbot_workspace/lingbot-vla-v2
bash tools/train_clean_stage1.sh --gpus 0,1,2,3 --dry-run
# 所有检查通过后执行：
bash tools/train_clean_stage1.sh --gpus 0,1,2,3
```

配置：BF16，4 卡，micro batch=1，accumulation=8，global batch=32；基础 LR=1e-5、cosine 至1e-6、3% warmup；max_steps=18000。这是 optimizer step，不是18000个 epoch。548893/32 约17153步一个完整样本遍历，drop_last/分片会使实际略有差异；不能据此保证已经收敛。

视觉塔、语言/动作模型可训练：`freeze_vision_encoder=false, freeze_vit=false, train_expert_only=false`；depth/video teacher 不训练。动作 horizon 仍50，不因为评测执行前10个动作而缩短训练 horizon。

每500步保存，完成后保留最多3个，包括窗口平均训练总 loss 最低的检查点与其余最新检查点。它不是评测成功率最好的模型。异步HF导出保护正在读取的 checkpoint，写新模型/临时HF文件期间可能短暂超过3个目录，磁盘应预留一次写入的空间。图表不随旧权重淘汰。

短训练先用 `bash tools/train_clean_stage1.sh --gpus 0,1,2,3 --smoke`。它在新目录显式设置5步/第5步保存，确认统计日志、mask、实际解码、checkpoint和图表后，再运行不带smoke的正式命令。这样的小运行不能作为收敛模型，也不能作为正式训练目录继续恢复。

## 第一阶段中断恢复

仅限**本次新流程**已经建立 normalization contract 的运行：

```bash
bash tools/train_clean_stage1.sh \
  --resume-run /绝对路径/新阶段一运行目录 \
  --gpus 0,1,2,3 --dry-run
# 核对后去掉 --dry-run
```

读取该运行保存的 `lingbotvla_cli.yaml`，恢复模型、optimizer、scheduler、dataloader与代码原有 RNG 状态；自动尝试最新到更早 DCP checkpoint。没有有效 checkpoint 则报错，不悄悄从零开始。运行目录被移动需要显式迁移有效配置中的路径，不建议运行中重构代码。中断至最近保存步之间的训练不可恢复。

当前不新增跨机器/不同world size的严格确定性续训承诺，也没有完整重写所有worker/CUDA RNG恢复机制。

## 第二阶段：独立增强微调

选经评测表现良好的**新阶段一** HF checkpoint：

```bash
bash extensions/clean_stage2/train.sh \
  --init-hf /绝对路径/新阶段一运行/checkpoints/global_step_N/hf_ckpt \
  --gpus 0,1,2,3 --steps 500 --dry-run
# 核对后去掉 --dry-run
```

只初始化HF模型权重，重新创建optimizer/scheduler；基础LR=3e-6，500步pilot warmup150步，最大增强强度约前500步递增。因此500步是链路/早期趋势pilot，不是充分训练效果验证；延长实验时用新的完整训练计划，例如 `--steps 1500`，不要把不同scheduler的pilot拼成无记录的实验。

拒绝旧模型（没有统计快照）、不兼容统计、已有输出、resume参数和上一阶段二实验。第一阶段入口不会加载增强代码。第二阶段没有实现中断后完整状态恢复，不能拿第一阶段入口恢复第二阶段。

30% raw /70% augmented 是每样本随机分支期望比例，非每batch严格比例。默认 safe_profiles 空，20% texture与10% clutter分支回退为光度，所以**默认实际是30% raw+70% photometric**。人工审核后再授权头相机安全背景，腕相机不覆盖。

## 统计核算与规则边界

当前实际训练统计为 `assets/norm_stats/robotwin_clean_verified.json`。全部548893行、2500 episodes、14维原始state/action，无NaN/Inf且行号连续。动作统计沿用官方绝对动作、合并50步chunk、终端动作重复的统计语义。

这次独立核算使用float64矩和精确加权经验分位数；旧clean-only使用自适应5000-bin直方图，因此不是逐字节相同。最大mean/std绝对差约4.9e-6，min/max一致；实际使用q01/q99最大差约0.00474，其他分位数最大差约0.00824。核对采用范围缩放容差，详情见 `norm_verification.json`，不能描述成所有浮点值完全一致。

启动器核对29个原始Parquet/episode元数据/info文件的SHA256，并检查7500个视频片段边界和视频文件存在。它**没有散列全部视频内容，也不能独立证明下载来源或官方授权**；若服务器副本重打包导致字节不同，应重新审计，不能直接去掉检查。

运行保存 `normalization/norm_stats.json + manifest.json`。训练实际读取该快照；推理自动读取同一快照并校验语义哈希。部署时带整个运行的配置与normalization目录，不能只复制hf_ckpt。

不修改原始数据，不读random数据训练，不新增simulator rollout，不导入外部背景。合成图形只来自程序像素操作；是否符合赛事最终解释，应由主办方确认。评测结果不回灌训练。

## 输出与查看

```text
train_outputs/robotwin_clean_stage1|robotwin_clean_stage2/<时间戳_UUID>/
  lingbotvla_cli.yaml
  normalization/{norm_stats.json,manifest.json}
  stage2_run.json                         # 仅第二阶段
  checkpoints/
    best_checkpoint.json
    global_step_N/{DCP文件,hf_ckpt/}
  visualizations/runs/<本次启动id>/
    lingbotvla_cli.yaml
    analysis/
      data/training_metrics_live.jsonl
      figures/
      report.html
      by_checkpoint/global_step_N/
        report.html / PNG / summary.json
        augmentation/                   # 仅第二阶段：真实训练batch图与参数
training_logs/<配置名>_<本次启动id>.log
```

图表保存是 best-effort：渲染失败不毁掉训练，但日志会告警并留诊断。验收须真实检查图片存在，不能仅看save checkpoint成功。

## 评测

沿用修复后的 `experiment/robotwin/start_robotwin_infer_and_eval.sh`。比赛 random 设置通常用 `--task_config demo_randomized`，实际名称以安装的RoboTwin配置为准。模型、seed、场景、use_length、精度必须记录并保持对比一致；4卡是并行任务槽，不是把单条轨迹模型分片到4卡。

先1任务×1episode冒烟，再50任务×5episodes初筛。以官方发布复现默认 `--use_bf16 False --use_fp32 True` 评测；若显存不足改BF16需标注，不能拿两种精度的成功率混比。训练BF16不要求推理也必须BF16。

阶段二是否提升必须通过配对评测/消融确定；不存在百分之百提升保证。
