# 使用流程：准备 → 训练 → 评测

本指南以当前 **clean-only 两阶段训练**为主线，命令在 Linux/CUDA 的 Bash 环境执行。路径以 [目录规范](dir_standard.md) 为准；详细配方见 [clean 训练指南](clean_training/README.md)。示例中的 `<run_id>`、`<N>` 和 `/path/to/...` 需替换为实际值。

## 1. 设置工作区与环境

```bash
cd /path/to/lingbot-vla-v2
export WORKSPACE="$PWD"
export OUTPUT_DIR="$WORKSPACE/outputs"
export ROBOTWIN_DIR="$WORKSPACE/RoboTwin"
export QWEN3VL_DIR="$WORKSPACE/models/Qwen3-VL-4B-Instruct"

# 首次准备训练环境；已有环境可直接激活
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla
```

训练环境采用 Python 3.12 / PyTorch 2.8；安装脚本也安装深度依赖和 LeRobot。两阶段 clean 启动器当前要求 **4 张不同 GPU**，示例使用 `0,1,2,3`。

只有仿真评测需要 RoboTwin 环境和仿真资源，训练已有数据不需要安装模拟器。按 [RoboTwin 安装说明](../RoboTwin/README.md) 创建独立的 `RoboTwin` 环境后，在该环境中执行：

```bash
git submodule update --init --recursive RoboTwin
conda activate RoboTwin
cd "$ROBOTWIN_DIR"
bash scripts/_install.sh
# 安装脚本会更新 XPolicyLab 到上游最新版本；安装后恢复本项目固定版本
git submodule update --init --recursive XPolicyLab
python -m pip install -e ./XPolicyLab
bash scripts/_download_assets.sh
cd "$WORKSPACE"
conda activate lingbotvla
```

## 2. 准备权重与数据

| 输入 | 默认位置 |
| --- | --- |
| 阶段一 foundation 权重 | `models/lingbot-vla-v2-6b/` |
| Qwen3-VL 权重及 tokenizer | `models/Qwen3-VL-4B-Instruct/` |
| MoGe-2 权重 | `models/depth/moge2-vitb-normal.pt` |
| LingBot-Depth teacher | `models/lingbot-vla-v2-6b/depth/model.pt` |
| DINO-Video teacher 与配置 | `models/lingbot-vla-v2-6b/dino_video/{teacher_step_10000.pth,config.yaml}` |
| 已核验的 clean LeRobot v3 数据 | `datasets/RoboTwin_lerobot_v30/` |
| 训练数据清单 | `assets/training_data/robotwin_clean_only.txt` |
| 已核验归一化统计 | `assets/norm_stats/robotwin_clean_verified.json` |

模型下载脚本会在 `--local_dir` 下追加仓库名称，因此以下命令的落点正好对应表中前两项：

```bash
python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir models
python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir models
```

换主机时需另行准备模型文件，`models/` 被 Git 忽略，拉取相同代码不会同步权重。阶段一 `--dry-run` 也会检查本地输入；`--init-hf` 必须指向直接包含 `config.json` 和权重分片的目录。可先检查：

```bash
test -f "$WORKSPACE/models/lingbot-vla-v2-6b/config.json" && echo "foundation config OK"
# 若报 Missing HF config，查找模型是否下载到了旧目录或多嵌套了一层
find "$WORKSPACE/models" "$WORKSPACE/lingbot-vla" -type f -name config.json 2>/dev/null
```

旧 README 的下载示例曾落到 `lingbot-vla/lingbot-vla-v2-6b/`。下载脚本的 `--local_dir` 应传 `models`，若传 `models/lingbot-vla-v2-6b`，会再追加一次仓库名。

`--init-hf` / `MODEL_DIR` 只覆盖主模型，不改变 depth/video teacher 路径。若模型保存在 Hugging Face 缓存的 `snapshots/<版本>/`，可将完整 snapshot 链接到规范位置（目标路径必须尚不存在）：

```bash
FOUNDATION="/absolute/path/to/models--robbyant--lingbot-vla-v2-6b/snapshots/<版本>"
# 先确认文件完整；只看到 depth/dino_video 目录不能证明其内容已下载
ls "$FOUNDATION/config.json" "$FOUNDATION/depth/model.pt" \
  "$FOUNDATION/dino_video/teacher_step_10000.pth" "$FOUNDATION/dino_video/config.yaml"
# 确认上面的检查通过后执行；不覆盖已有模型目录
mkdir -p "$WORKSPACE/models"
ln -sT "$FOUNDATION" "$WORKSPACE/models/lingbot-vla-v2-6b"
```

也可分别在 `configs/vla/robotwin/robotwin_clean_stage1.yaml` 和 `extensions/clean_stage2/config.yaml` 中，将 `train.align_params.depth.morgbd_path`、`train.align_params.video.ckpt_path/config_path` 改为实际文件的绝对路径。阶段二的 `--init-hf` 指向阶段一 checkpoint，teacher 仍来自原 foundation。MoGe-2 和 Qwen 是独立输入，仍按下文准备。保留缓存 snapshot 所依赖的 `blobs/`，不要仅复制 `config.json` 或删除其链接目标。

MoGe-2 从 [项目 README 的权重来源](../README.md#model-download) 获取；按表中路径放置，或将两阶段配置中的 `moge_path` 改为实际文件。确认 foundation 下载中包含 depth/video teacher 文件。

clean 清单默认只有一行：`robotwin datasets/RoboTwin_lerobot_v30`。已有核验数据可直接使用；外部副本可将这一行改为其绝对路径。**日常训练不需要重算统计或修改核验报告。**

若需要从原始轨迹重新准备数据，使用已有脚本：

```bash
# 下载全部可用 clean 任务；原始轨迹写入 datasets/RoboTwin/demo_clean/...
bash "$ROBOTWIN_DIR/scripts/download_xpolicylab_data.sh"
# 在装有匹配 LeRobot v3 的环境中转换
python "$ROBOTWIN_DIR/XPolicyLab/scripts/transform_lerobot_v30_format.py" \
  "demo_clean.*.aloha_agilex" --repo_id RoboTwin_lerobot_v30 --max_episode 50
```

新转换的数据不保证与现有核验副本具有相同文件指纹。clean 启动器要求 2500 episodes、548893 frames，并核对版本化报告；校验不符时应重新审计后更新对应统计与报告，不能跳过检查。下载/转换细节见 [数据准备说明](../experiment/robotwin/README.md)。

## 3. 计算与核验归一化统计

这里生成的是供训练、推理读取的 **state/action 统计 JSON**，不将归一化后的数值写回数据集。当前已核验的 clean 副本直接使用 `assets/norm_stats/robotwin_clean_verified.json`，可跳到第 4 节；新数据、自定义任务集合或重新审计时才执行以下步骤。

### 自定义数据：运行通用统计脚本

先准备对应的 `configs/robot_configs/<机器人名>.yaml`，其中 state/action 映射、绝对或相对动作设置应与训练一致。以下两种输入方式二选一，在 `lingbotvla` 环境中执行：

```bash
# 独立统计运行；不要提前创建 NORM_RUN，每次计算使用新目录
NORM_ID="$(date +%Y%m%d_%H%M%S)_$(python -c 'import uuid; print(uuid.uuid4().hex[:8])')"
NORM_RUN="$OUTPUT_DIR/train_outputs/norm_robotwin_$NORM_ID"

# 方式 A：单个 LeRobot 数据目录
CUDA_VISIBLE_DEVICES=0 LINGBOT_TRAIN_RUN_ID="$NORM_ID" bash train.sh \
  scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
  --data.data_name robotwin \
  --data.train_path datasets/RoboTwin_lerobot_v30 \
  --data.robot_config_root configs/robot_configs \
  --data.data_ratio_for_norm_compute 1.0 --data.norm_merge_chunk_dim true \
  --train.chunk_size 50 --train.output_dir "$NORM_RUN" \
  --data.norm_path "$NORM_RUN/norm_stats.json"
```

若输入是 `assets/training_data/` 下的多数据集清单，用以下命令替代方式 A；`robotwin.txt` 是现有混合数据清单，按实验换成自己的清单：

```bash
# 方式 B：清单中只统计第一列等于 robotwin 的条目
CUDA_VISIBLE_DEVICES=0 LINGBOT_TRAIN_RUN_ID="$NORM_ID" bash train.sh \
  scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
  --data.data_name multi --data.robot_name robotwin \
  --data.train_path assets/training_data/robotwin.txt \
  --data.robot_config_root configs/robot_configs \
  --data.data_ratio_for_norm_compute 1.0 --data.norm_merge_chunk_dim true \
  --train.chunk_size 50 --train.output_dir "$NORM_RUN" \
  --data.norm_path "$NORM_RUN/norm_stats.json"
```

`data_name` / `robot_name` 对应机器人配置文件名；`1.0` 表示全量统计；`chunk_size=50` 与当前动作 horizon 一致；`norm_merge_chunk_dim=true` 合并动作 chunk 维度。若训练采用其他 horizon 或动作语义，统计配置也要同步。多 GPU 可改 `CUDA_VISIBLE_DEVICES`，脚本分片计算后合并结果。

输出包含 count 以及各特征的 mean/std、min/max 和分位数。日志与临时清单在 `NORM_RUN` 内，临时清单读取后自动删除。统计命令会实际读取 state/action 并计算，没有 `--dry-run`，不进行模型训练。

通用训练应使用相同的数据输入，并显式添加 `--data.norm_stats_file "$NORM_RUN/norm_stats.json"`，或在其配置的 `data.norm_stats_file` 中设置该路径。默认回退来源是机器人配置的 `norm_stats` 字段；不要只生成 JSON 而未让训练读取它。通用统计结果不能直接替换两阶段 clean 的核验统计。

### clean 数据：运行独立核验脚本

此入口仅用于完整 clean-50 数据（2500 episodes、548893 frames），在 CPU 上读取 Parquet，核对原始 state/action，并按合并 50 步绝对动作的语义重算统计：

```bash
VERIFY_RUN="$OUTPUT_DIR/train_outputs/clean_norm_verify_$(date +%Y%m%d_%H%M%S)_$(python -c 'import uuid; print(uuid.uuid4().hex[:8])')"
python tools/verify_clean_norm.py \
  --dataset-root datasets/RoboTwin_lerobot_v30 \
  --reference assets/norm_stats/robotwin_clean_only.json \
  --output "$VERIFY_RUN/norm_stats.json" \
  --report "$VERIFY_RUN/norm_verification.json"
```

检查报告的 `comparison_passed`、count、episodes 和 `source_fingerprint`。输出路径不能已存在；比较失败时只留下报告，不写新统计。该脚本使用精确加权经验分位数，通用脚本使用运行直方图，二者不要求逐字节一致。

核验产物先留在本次输出目录供审阅，脚本不会自动替换版本化参考文件。若数据确实变更，经审核后需同步更新 `assets/norm_stats/robotwin_clean_verified.json` 与 `docs/clean_training/norm_verification.json`，保持来源指纹与统计一致，再运行阶段一 `--dry-run`。仅给启动器换一个 JSON 路径不足以通过其核验。

## 4. 阶段一：建立 clean 基线

```bash
# 只核验输入并打印启动命令，不启动训练
bash tools/train_clean_stage1.sh \
  --init-hf models/lingbot-vla-v2-6b --gpus 0,1,2,3 --dry-run
# 实际运行 5 步，检查解码、checkpoint 和图表
bash tools/train_clean_stage1.sh \
  --init-hf models/lingbot-vla-v2-6b --gpus 0,1,2,3 --smoke
# 正式训练：每次启动生成新的运行目录
bash tools/train_clean_stage1.sh \
  --init-hf models/lingbot-vla-v2-6b --gpus 0,1,2,3
```

配置为 `configs/vla/robotwin/robotwin_clean_stage1.yaml`：默认 BF16、全局 batch 32、18000 个 optimizer steps，每 500 步保存，最终最多保留 3 个 checkpoint。它从 foundation 权重初始化；5 步 smoke 是独立小实验，不作为正式基线继续恢复。

显式 `--init-hf` 可避免已有 `MODEL_DIR`（可能指向评测权重）覆盖阶段一初始化。记下启动时打印的运行目录：

```bash
STAGE1_RUN="$OUTPUT_DIR/train_outputs/robotwin_clean_stage1_<run_id>"
```

中断恢复使用同一运行目录；先检查，再去掉 `--dry-run` 执行：

```bash
bash tools/train_clean_stage1.sh \
  --resume-run "$STAGE1_RUN" --gpus 0,1,2,3 --dry-run
# 检查通过后恢复
bash tools/train_clean_stage1.sh \
  --resume-run "$STAGE1_RUN" --gpus 0,1,2,3
```

恢复模型、optimizer、scheduler 等已保存状态；仅支持当前带归一化契约的新阶段一运行。

## 5. 阶段一评测与 checkpoint 选择

从已完成 HF 导出的 checkpoint 中选择一个；训练 loss 最低的 checkpoint 不等于成功率最高。

```bash
CHECKPOINT="$STAGE1_RUN/checkpoints/global_step_<N>/hf_ckpt"
export CONDA_SH="/path/to/miniconda3/etc/profile.d/conda.sh"
export INFERENCE_ENV=lingbotvla
export SIM_ENV=RoboTwin

# 先测试 1 个任务、1 个 episode，确认推理与仿真链路
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path "$CHECKPOINT" --task_config demo_clean \
  --num_tasks 1 --num_episodes 1 --num_gpus 1 --num_per_gpu 1 \
  --use_bf16 False --use_fp32 True

# 正式评测：50 个任务，每个 100 个 episodes
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path "$CHECKPOINT" --task_config demo_randomized \
  --num_tasks 50 --num_episodes 100 --num_gpus 4 --num_per_gpu 1 \
  --use_bf16 False --use_fp32 True
```

评测 clean 场景时将 `demo_randomized` 改为 `demo_clean`；初筛可用 `--num_episodes 5`。默认每次预测 50 个动作、执行前 10 个后重新观察，`--use_length` 可显式设置。并行 GPU 数控制任务槽数。

官方发布模型复现使用 FP32。显存不足可改 `--use_bf16 True --use_fp32 False`，但比较 checkpoint 时应固定场景、seed、精度、episode 数和 `use_length`。脚本当前 seed 固定为 0；`--no_video` 可关闭录像，`Ctrl-C` 停止本次运行。

评测目录启动时会打印，结果查看 `stats.txt`、`eval_logs/` 和 `eval_results/<任务>/_result.txt`。完整参数见 [评测说明](../experiment/robotwin/README.md#simulated-evaluation-inference--robotwin-sim)。

## 6. 阶段二：从选中的阶段一模型增强微调

`CHECKPOINT` 必须指向当前核验流程产生的阶段一 `hf_ckpt`。先预览增强，再核验并训练：

```bash
python -m extensions.clean_stage2.preview \
  --dataset-root datasets/RoboTwin_lerobot_v30 --episodes 0 500 1000

bash extensions/clean_stage2/train.sh \
  --init-hf "$CHECKPOINT" --gpus 0,1,2,3 --steps 500 --dry-run
# 核对后实际训练；500 步用于先导实验
bash extensions/clean_stage2/train.sh \
  --init-hf "$CHECKPOINT" --gpus 0,1,2,3 --steps 500
```

配置为 `extensions/clean_stage2/config.yaml`，增强设置为 `extensions/clean_stage2/augmentation.json`。只加载 HF 模型权重，重新建立 optimizer/scheduler；默认无背景授权区域时，实际为约 30% 原图、70% 光度增强。新实验可显式设 `--steps 1500`。

阶段二每次创建新目录，**目前不支持完整中断恢复**。完成后将 `CHECKPOINT` 换为阶段二运行下的 `checkpoints/global_step_<N>/hf_ckpt`，复用第 5 节命令，与阶段一做相同设置的评测。

## 7. 查看产物与离线评估

| 产物 | 默认位置 |
| --- | --- |
| 训练运行 | `$OUTPUT_DIR/train_outputs/robotwin_clean_stage1_<run_id>/`；阶段二标签为 `robotwin_clean_stage2` |
| 统计 / 核验输出 | 第 3 节的 `NORM_RUN/norm_stats.json`；`VERIFY_RUN/{norm_stats.json,norm_verification.json}` |
| 控制台日志 / TensorBoard | `<运行目录>/*.log` / `<运行目录>/runs/` |
| checkpoint / 归一化快照 | `<运行目录>/checkpoints/` / `<运行目录>/normalization/` |
| 训练可视化报告 | `<运行目录>/visualizations/runs/<启动id>/analysis/report.html` |
| 阶段二实际 batch 增强预览 | 上述 `analysis/by_checkpoint/global_step_<N>/augmentation/` |
| 阶段二离线预览 | `$OUTPUT_DIR/train_outputs/stage2_preview_<run_id>/` |
| RoboTwin 评测 | `$OUTPUT_DIR/eval_outputs/<实验名>_<步数>k_<task_config>_<时间戳>/` |
| open-loop 轨迹图 | `$OUTPUT_DIR/eval_outputs/open_loop_<run_id>/` |

open-loop 只在已有轨迹上对比预测与真实动作，不等同于仿真成功率评测：

```bash
python scripts/open_loop_eval.py \
  --model_path "$CHECKPOINT" --robo_name robotwin \
  --data_path "$WORKSPACE/datasets/RoboTwin_lerobot_v30" \
  --traj_ids 0 --use_length 50
```

clean 模型推理会读取所属运行的统计快照；迁移模型时应保留运行配置和 `normalization/`，不能只拷贝 `hf_ckpt/`。查看训练曲线可打开 HTML，或执行 `tensorboard --logdir "$STAGE1_RUN/runs"`。

## 8. 其他入口与辅助工具

| 用途 | 入口 / 配置 | 何时使用 |
| --- | --- | --- |
| 通用 LingBot-VLA 训练 | `train.sh` + `tasks/vla/train_lingbotvla.py` | 混合数据或自定义数据；示例配置 `configs/vla/robotwin/robotwin.yaml`，不等同于 clean-only 流程 |
| Distributed Muon 配置 | `configs/vla/robotwin/robotwin_dist_muon.yaml` | 通用训练的优化器变体 |
| PI0 训练 | `train.sh` + `tasks/vla/train_pi0.py` | 需要对应模型和专用配置，不能直接套用 clean LingBot 配方 |
| 自定义数据统计 | `train.sh` + `scripts/compute_norm_stats.py` + `configs/vla/norm_compute/post_data.yaml` | 可执行命令与接入方式见第 3 节 |
| clean 数据重新核验 | `tools/verify_clean_norm.py` | 数据变更后审计，命令见第 3 节 |
| 控制台日志分析 | `tools/analyze_training_log.py` | 指定 `--log`、`--config`、`--run-dir`，默认写入该运行的 `analysis/` |
| 原生 XPolicyLab 评测 | `RoboTwin/scripts/eval_policy.sh` | 其他 XPolicyLab 策略；本项目 LingBot 模型使用第 5 节入口 |

通用训练示例（使用其混合数据清单）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh \
  tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml \
  --data.train_path assets/training_data/robotwin.txt --data.data_name multi
```

若第 3 节为该清单生成了新的统计，在此命令末尾添加 `--data.norm_stats_file "$NORM_RUN/norm_stats.json"`。

已有控制台日志可独立重做分析，先将 `TRAIN_LOG` 设为该运行的实际日志文件：

```bash
TRAIN_LOG="$STAGE1_RUN/<实际日志文件名>.log"
python tools/analyze_training_log.py \
  --log "$TRAIN_LOG" --config "$STAGE1_RUN/lingbotvla_cli.yaml" \
  --run-dir "$STAGE1_RUN" --output "$STAGE1_RUN/analysis"
```

默认输入相对 `WORKSPACE` 解析，输出集中到 `OUTPUT_DIR`；显式路径优先于环境变量和配置默认值。训练的 `--output-dir` / `--train.output_dir` 是最终运行目录，评测的 `--output_base` 是评测分类根，均不额外重复套层。
