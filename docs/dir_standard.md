# 目录结构与路径规范

本文定义 `lingbot-vla-v2` 工作区的目录布局、版本管理边界，以及训练和评测产物的默认存放位置。

## 1. 工作区目录树

```text
lingbot-vla-v2/
├── .git/                              # Git 元数据
├── .gitignore
├── .gitmodules                        # RoboTwin 子模块路径与 URL
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-depth.txt
├── setup.py
├── train.sh
│
├── assets/                            # 项目资源，纳入版本管理
│   ├── norm_stats/
│   ├── training_data/                 # 数据清单，不存放完整数据集
│   ├── LingBot_VLA_2_0.pdf
│   └── *.png                          # 项目图表和示意图
│
├── configs/                           # 项目配置，纳入版本管理
│   ├── robot_configs/
│   └── vla/
│       ├── norm_compute/
│       ├── real_robot/
│       └── robotwin/
│
├── datasets/                          # 本地完整数据集，不纳入版本管理
├── extensions/                        # 扩展代码，纳入版本管理
│
├── deploy/                            # 部署代码，纳入版本管理
│   ├── image_tools.py
│   ├── lingbot_vla_v2_policy.py
│   ├── msgpack_numpy.py
│   ├── utils.py
│   ├── websocket_client_policy.py
│   └── websocket_policy_server.py
│
├── docker/
│   └── Dockerfile
│
├── docs/                              # 项目文档，纳入版本管理
│   ├── config/
│   ├── training_visualization.md
│   └── dir_standard.md                # 本目录与路径规范
│
├── experiment/                        # 本项目实验入口，纳入版本管理
│   └── robotwin/                      # 本项目的 RoboTwin 评测脚本和说明
│
├── lingbotvla/                        # 项目 Python 源码，纳入版本管理
│   ├── checkpoint/
│   ├── data/
│   │   ├── multimodal/
│   │   └── vla_data/
│   ├── distributed/
│   │   ├── fsdp/
│   │   ├── fsdp2/
│   │   ├── moe/
│   │   └── sequence_parallel/
│   ├── models/                        # Python 模型源码，不是权重存放目录
│   │   └── vla/
│   │       ├── lingbot_vla/
│   │       ├── pi0/
│   │       └── vision_models/
│   │           ├── align_heads/
│   │           ├── dino_video/
│   │           ├── lingbot-depth/
│   │           └── MoGe/
│   ├── ops/
│   ├── optim/
│   ├── schedulers/
│   └── utils/
│
├── models/                            # 本地下载的模型权重，不纳入版本管理
│
├── outputs/                           # 统一运行产物根目录，不纳入版本管理
│   ├── train_outputs/
│   │   └── <实验名>_<run_id>/          # 每次训练启动对应一个目录
│   │       ├── checkpoints/            # checkpoint；实际内容由训练配置决定
│   │       ├── model_assets/           # 训练使用的模型资源副本
│   │       ├── runs/                  # TensorBoard 等运行记录
│   │       ├── normalization/          # clean 训练统计快照与 manifest
│   │       ├── stage2_run.json         # 仅第二阶段
│   │       ├── images/                 # 启用视觉诊断时生成
│   │       ├── visualizations/
│   │       │   └── runs/<run_id>/
│   │       │       ├── lingbotvla_cli.yaml
│   │       │       └── analysis/
│   │       │           ├── data/       # JSONL、CSV、统计与配置比较
│   │       │           ├── figures/
│   │       │           ├── configs/
│   │       │           ├── report.html
│   │       │           └── by_checkpoint/global_step_N/
│   │       │               └── augmentation/  # 仅第二阶段
│   │       ├── analysis/               # 手动离线分析工具的默认输出
│   │       ├── trace/                  # 启用 profiling 时生成
│   │       ├── wandb/                  # 启用 WandB 时生成
│   │       ├── lingbotvla_cli.yaml     # 本次启动的有效训练配置
│   │       ├── <配置名>_<run_id>.log   # 训练控制台日志
│   │       └── 其他训练产物
│   │
│   └── eval_outputs/
│       └── <实验名>_<检查点步数>k_<task_config>_<时间戳>/
│           ├── stats.txt
│           ├── run_manifest.json
│           ├── scheduler_events.jsonl
│           ├── summary.json
│           ├── normalization/
│           ├── inference_pids.txt
│           ├── eval_pids.txt
│           ├── inference_logs/
│           │   └── <推理服务或端口>.log
│           ├── eval_logs/
│           │   └── <任务名>.log
│           └── eval_results/
│               └── <任务名>/
│                   ├── _result.txt
│                   ├── task_summary.json
│                   └── attempts/attempt_N/
│                       ├── task_config.json
│                       ├── attempt_result.json
│                       ├── seed_checks.jsonl
│                       ├── episode_results.jsonl
│                       └── episodes/episode_I_seed_S/
│                           ├── episode.json
│                           ├── inference.jsonl
│                           ├── execution.jsonl
│                           ├── predictions/*.npz
│                           └── episode*_success.mp4
│
├── RoboTwin/                          # 独立 Git 子模块，固定到明确提交
│
├── scripts/                           # 数据下载、统计和评测脚本
├── tasks/
│   └── vla/                           # 训练入口源码，不是运行产物目录
├── tests/                             # 测试代码
└── tools/                             # 分析、环境和可视化工具
```

## 2. 版本管理约定

### 纳入版本管理

- 项目源码、配置、脚本、测试和文档。
- `assets/` 中的项目资源，以及 `assets/training_data/` 中的数据清单；清单不代表完整数据集。
- `RoboTwin/` 子模块的 Git 引用。子模块应固定到明确提交，更新时提交子模块引用变更。

### 不纳入版本管理

- `datasets/`：完整数据集及其本地副本。
- 根目录 `models/`：下载的模型权重和本地模型文件。
- `outputs/`：训练、评测、分析及其他运行产物。
- 临时日志、缓存、备份文件和 Python 字节码等生成文件。

根目录的 `models/` 专用于模型权重；`lingbotvla/models/` 是项目源码，应继续纳入版本管理。`assets/training_data/` 存放数据清单，不替代 `datasets/`。

## 3. 目录环境变量

启动脚本可以通过环境变量接收工作区、数据、模型和输出根目录。以下示例适用于 Bash 等支持 `export` 的 Shell，应在项目根目录执行：

```bash
cd /path/to/lingbot-vla-v2

export WORKSPACE="$(pwd)"
export ROBOTWIN_DIR="$WORKSPACE/RoboTwin"
export MODEL_DIR="$WORKSPACE/models/lingbot-vla-v2-6b-robotwin"
export QWEN3VL_DIR="$WORKSPACE/models/Qwen3-VL-4B-Instruct"
export OUTPUT_DIR="$WORKSPACE/outputs"

```

变量约定如下：

| 变量 | 用途 | 默认位置 |
| --- | --- | --- |
| `WORKSPACE` | `lingbot-vla-v2` 仓库根目录 | 启动脚本解析得到的仓库根目录 |
| `ROBOTWIN_DIR` | RoboTwin 子模块位置 | `$WORKSPACE/RoboTwin` |
| `MODEL_DIR` | LingBot VLA 模型权重位置 | `$WORKSPACE/models/lingbot-vla-v2-6b-robotwin` |
| `QWEN3VL_DIR` | Qwen3-VL 权重位置 | `$WORKSPACE/models/Qwen3-VL-4B-Instruct` |
| `OUTPUT_DIR` | 统一运行产物根目录 | `$WORKSPACE/outputs` |

`OUTPUT_DIR` 指向统一产物根目录；训练和评测分别使用其下的 `train_outputs/` 和 `eval_outputs/`。显式命令行路径优先，其次是规范环境变量、旧变量别名，最后是配置默认值。`MODEL_PATH`、`QWEN3VL_PATH`、`EVAL_WORKDIR`、`OUTPUT_BASE` 继续兼容；`OUTPUT_BASE` 和 `--output_base` 直接表示评测分类目录，不再追加 `eval_outputs/`。

clean 阶段一默认从 `models/lingbot-vla-v2-6b/` 的 foundation 权重初始化；阶段二必须用 `--init-hf` 选择阶段一 checkpoint。`MODEL_DIR` 的表格默认用于评测，不能将评测权重默认替换 clean 阶段一的 foundation 权重。RoboTwin 已登记为子模块；其内部 XPolicyLab 继续由 RoboTwin 固定提交引用。cuRobo 按已有安装脚本安装到 `RoboTwin/envs/curobo/`，不登记为子模块。

RoboTwin 的源码、`env_cfg/`、`description/` 和仿真资源 `assets/` 保留其内部布局。下载及采集的完整轨迹默认位于 `$WORKSPACE/datasets/RoboTwin/<task_config>/<task>/<embodiment>/`，其中 `data/`、`video/`、`instruction/`、种子及缓存结构不变。`ROBOTWIN_DATA_ROOT`（旧别名 `XPOLICYLAB_DATA_ROOT`）可覆盖轨迹根，`HF_ARCHIVE_CACHE` 可覆盖下载缓存根，默认 `<轨迹根>/download_cache/`。LeRobot 转换默认输出到 `$WORKSPACE/datasets/<repo_id>/`，继续支持 `HF_LEROBOT_HOME` 覆盖；这些变量中的相对路径按工作区解析。

## 4. 路径解析约定

1. 仓库内的默认路径以 `WORKSPACE`（即 `lingbot-vla-v2/` 根目录）为基准，不依赖调用者当前所在目录。
2. 启动脚本应先确定仓库根目录，再解析相对路径；从其他目录调用时也应指向同一工作区。
3. 仓库内路径优先使用相对路径或由环境变量拼接出的路径，避免写入个人机器专属的绝对路径。
4. 数据集、模型权重和运行产物均通过各自目录变量或配置项定位；需要放在仓库外时，可以通过变量覆盖默认位置。
5. 脚本不得把大型运行产物写入源码目录、`assets/` 或仓库根目录。

## 5. 训练产物约定

每次训练启动生成一个唯一的训练运行目录：

```text
$OUTPUT_DIR/train_outputs/<实验名>_<run_id>/
```

训练配置、checkpoint、模型资源、TensorBoard 记录、训练日志、可视化和分析结果等本次启动产生的文件，默认统一放入该目录。原有工具生成的子目录结构保持不变；规范统一的是运行产物根路径。

`run_id` 用于区分每次启动，应在同一次启动的所有相关进程间保持一致，并确保目录名不会与已有运行目录冲突。日志和配置应归属于对应的训练运行目录，不再按默认设置散落在仓库根目录或其他独立输出根目录。显式 `train.output_dir` 直接指定最终运行目录。恢复训练继续使用原运行目录，并为这次启动生成独立的日志与可视化 ID。

离线阶段二预览沿用图片和 manifest，默认位于 `$OUTPUT_DIR/train_outputs/stage2_preview_<run_id>/`；统计计算的临时清单位于本次运行的 `tmp/`，读入后沿用原清理行为。目录树中的可选产物仅在原有功能启用时生成。


## 6. 评测产物约定

评测启动脚本沿用现有输出结构，并将其根目录设为：

```text
$OUTPUT_DIR/eval_outputs/<实验名>_<检查点步数>k_<task_config>_<时间戳>/
```

该目录名作为本次评测的运行 ID。一次脚本启动生成一个运行目录，包含本次启动涉及的全部推理服务日志、任务日志、PID 文件、统计文件和评测结果。任务结果写入 `eval_results/<任务名>/`，完整预测、逐动作反馈和视频按 `attempts/attempt_N/episodes/episode_I_seed_S/` 隔离保存，兼容汇总 `_result.txt` 保留在任务根目录。详细字段和离线报告见 [eval_logging.md](eval_logging.md)。open-loop 评测的轨迹图默认位于 `$OUTPUT_DIR/eval_outputs/open_loop_<run_id>/`，不生成 RoboTwin 专用的日志、PID 或视频目录。

RoboTwin 原生 XPolicyLab 调度器保留其微秒时间戳运行目录名，默认落在 `$OUTPUT_DIR/eval_outputs/<时间戳>/`。其实际产物为 `logs/`、`jobs/`、`summary.json`，任务结果和视频集中到同次运行的 `eval_results/<任务>/<策略>/<task_config>/<checkpoint>/<时间戳>/`。独立策略服务入口同样在 `eval_outputs/<时间戳>/` 保存现有服务日志，单任务客户端保留任务内部层级。显式 `--output-dir` 保持原生调度器的分类根含义，客户端 `--output_dir` 为任务结果根，不重复追加 `eval_outputs/`。客户端同步到 RoboTwin 的文件由主项目管理，并在子仓库中忽略。
