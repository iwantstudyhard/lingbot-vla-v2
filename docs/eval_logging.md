# RoboTwin 测评日志与离线排查

批量启动器默认启用 `--eval_trace full`。每次推理保存完整动作 horizon，
每次仿真动作保存下发值、控制目标、实际臂位置和规划反馈。没有新增图像采集；
视频仍由原来的 `--no_video` / `--video_fps` 控制。

## 启动与分析

沿用原来的模型、conda 和仿真参数，追加日志选项即可：

```bash
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path /path/to/checkpoints/global_step_18000/hf_ckpt \
  --conda_sh /path/to/miniconda3/etc/profile.d/conda.sh \
  --num_tasks 1 --num_episodes 2 --num_gpus 1 --num_per_gpu 1 \
  --eval_trace full

python tools/analyze_eval_logs.py --run outputs/eval_outputs/<run>
python tools/analyze_eval_logs.py --run outputs/eval_outputs/<run> \
  --task lift_pot --episode 0 --output outputs/eval_outputs/<run>/analysis_lift_pot_0
```

打开生成的 `analysis/report.html`，按任务、回合结果、seed 或诊断观察筛选。
报告链接对应回合的日志、实际模型配置和视频；曲线中的浅色线表示完整预测，
包括未执行的后缀，竖线表示重新推理的位置。

`--eval_trace off` 保留运行配置、种子筛选、回合终止结果、调度事件和回合级耗时，
关闭逐请求/逐动作追踪。用于对比记录开销或普通运行。
对于自定义 `--inference_script`，full 模式要求服务支持下面的协议与启动参数；
off 模式仍可使用不支持追踪的自定义服务。

终端进度保留 `episodes 2/5, success 0, rate 0.0% (running)`，括号显示
`initializing / running / complete / error / interrupted`，异常回合也可显示其终止原因。
首个回合尚未完成时成功率显示 `N/A`。这里的成功率仅统计当前 attempt 已完成的
策略回合；重试之间独立计数，正式成功率仍只采用完整 attempt。

单独启动服务时传入 `--eval_run_dir <run> --eval_slot <slot>`，客户端传入
`--run_dir <run> --attempt 1 --slot <slot> --eval_trace full`。
两端必须访问同一运行目录：完整预测由服务写盘，客户端校验路径及文件存在后才执行。
直接运行客户端默认 attempt 为 1；再次运行相同任务必须使用新的 attempt 编号。
默认批量入口已自动管理这些参数，并同步仿真侧轻量辅助模块。

## 目录与关联

```text
<run>/
├── run_manifest.json              # 启动参数、实际路径、代码版本、环境
├── scheduler_events.jsonl         # 服务就绪、任务派发/退出/重试、退出信号
├── summary.json                   # 完整任务覆盖率和正式成功率
├── stats.txt                      # 同源的文本汇总
├── normalization/                # 实际归一化源文件的原字节快照
├── inference_logs/
│   ├── slot_<id>.events.jsonl
│   └── slot_<id>.model_<hash>.json # 配置、精度、checkpoint、维度映射
├── eval_logs/<task>.log           # 原有控制台输出
├── eval_results/<task>/
│   ├── _result.txt                # 完整 attempt 的兼容成功率
│   ├── task_summary.json
│   ├── task_config.jsonl          # 每次启动/重试的实际配置，追加写入
│   ├── attempt_results.jsonl      # 每次启动/重试的结束状态，追加写入
│   ├── seed_checks.jsonl
│   ├── episode_results.jsonl
│   └── episodes/episode_<id>_seed_<seed>_attempt_<n>/
│       ├── episode.json
│       ├── inference.jsonl
│       ├── execution.jsonl
│       ├── predictions/request_<id>.npz
│       └── episode<id>_<reason>.mp4
└── analysis/
    ├── report.html
    ├── diagnostics.json          # 耗时、记录大小、阈值、读取警告
    ├── tasks.csv
    ├── episodes.csv
    ├── anomalies.csv
    └── figures/
```

事件和汇总使用 `schema_version=1`。关联键为
`run_id / task / attempt / episode_id / seed / slot / request_id`。
`request_id=reset` 表示回合初始化，动作请求从 `000000` 递增。
动作事件带 `chunk_index` 和 `take_action_cnt`。
时间戳为带 UTC 时区的 ISO 时间；耗时使用单调时钟，单位毫秒。

每个输出文件由一个进程负责写入，JSONL 每条 flush，NPZ 与 JSON 汇总原子替换。
重试共用任务目录，配置、种子筛选及结果按 `attempt` 字段追加到任务级 JSONL。
回合目录名带重试编号，已有预测和 episode 目录禁止覆盖；相同 attempt 编号禁止重复启动。
汇总及分析工具仍可读取之前生成的 `attempts/attempt_<n>/` 旧日志，新运行不生成这两级目录。
日志写入错误会使 attempt 失败，控制台和调度结果会明确反映错误。
flush 和原子替换保障进程崩溃时的可读性，不表示断电级持久性承诺。

## 推理产物与接口

原有 WebSocket `action`、`server_timing` 和普通 `infer()` 使用方式保持兼容。
新增可选请求 `_eval_context`，服务在特征变换之前移除该字段。
握手 `eval_trace_schema=1` 表示追踪支持；full 模式无法连接到旧服务时明确报错。
响应 `_eval_trace` 返回关联键、是否发生真实前向、缓存选中索引、预测相对路径和耗时。
reset 响应返回实际模型配置文件路径。

策略的 `infer_with_diagnostics()` 使用同一次前向，在截断 `use_length` 之前捕获：

| NPZ 字段 | 含义 |
| --- | --- |
| `input_state` | 原始输入，通常为 `(1, 14)` |
| `state` | 实际送入模型的归一化/补齐状态，保留 batch 维度 |
| `state_joint_mask` / `action_joint_mask` | 状态/动作的有效维度掩码 |
| `lang_tokens` / `lang_masks` | 实际语言输入与掩码 |
| `normalized_actions` | 完整归一化输出，通常为 `(1, 50, 55)` |
| `full_action` | 完整反归一化机器人动作，通常为 `(1, 50, 14)` |
| `returned_action` | 真正通过服务返回的动作前缀或单动作 |

数组不包含 pickle/object 数据。读取方式：

缓存单步返回时，`model_forward=false`，完整预测与 NPZ 输入对应最初产生该 chunk 的
前向；当前请求的输入状态另存于 `inference.jsonl`，`selected_step` 给出所用动作索引。

```python
import numpy as np

with np.load("request_000000.npz", allow_pickle=False) as trace:
    full = trace["full_action"][0]
    returned = trace["returned_action"]
    mask = trace["action_joint_mask"][0]
```

模型有效维度必须按掩码和 `features.joints_max_dim` 读取，不能直接将 padded 输出
前 14 维当作原机器人动作。当前机器人顺序为左臂 6 维、左夹爪命令、
右臂 6 维、右夹爪命令；训练特征顺序及反向映射另存于模型配置。
BF16 状态转为 FP32 数组保存，但保留 BF16 量化后的数值，实际精度记录在模型 metadata。

记录不会额外采样、重置随机种子或调整动作裁剪。服务 seed 为原来的 42，
只在模块初始化时设置；日志保存真实环境 seed 和实际指令。
没有保存图像和每次请求的完整 RNG 状态，因此这些日志用于动作与执行排查，
不提供逐请求原始视觉输入重放。

## 执行反馈和结果语义

`execution.jsonl` 中每个动作有 `action_start`，然后有 `action_end` 或 `action_error`。
下发动作的形状或有限性检查失败时，已保存的 NPZ 保留原始证据，并终止 attempt。
有限值超出统计范围或限位只作诊断记录，不新增控制裁剪。

- `control_target`：通过 `get_*_arm_jointState()` 读取的 drive target，包含夹爪命令。
- `actual_arm_qpos`：通过 `get_*_arm_real_jointState()` 读取，只取真实臂关节的 12 维。
- `gripper_command`：夹爪命令值；现有 getter 不能作为物理手指位置测量。
- `left_ee_pose / right_ee_pose`：执行前后的末端位姿。
- `planning`：左右臂 TOPP 调用的状态、轨迹点数、耗时和异常 traceback。
- `count_delta`：仿真动作计数变化；回合成功或耗尽步数时，chunk 剩余后缀不执行。

TOPP 记录包装器在回合结束时恢复，保持原始返回和异常行为；仿真内部仍按原来的方式处理规划失败。
反馈采集不额外调用 `get_obs()`、渲染或推进仿真。现有逐动作视频刷新保留。

回合结束原因：`success / step_limit / invalid_action / exception / interrupted`。
专家筛选的拒绝种子和异常写入 `seed_checks.jsonl`，不计入策略测评分母。
没有终止记录的回合或动作在报告中标记为未闭合，可关联进程退出码和信号进一步排查。

正式成功率只选择每个任务最后一个完整 attempt：要求请求的回合全部完成、
终止原因均为 success 或 step_limit，且已记录的进程退出码为 0。
多次重试的回合不混合计数。没有完整 attempt 的任务展示最后一次部分结果，
不进入正式成功率分母，并在任务覆盖率中体现缺失。
日志的最后一条文本成功率不再是统计来源。

## 报告诊断与耗时

默认生成每个任务首个回合和有诊断观察的回合曲线；`--episode` 可请求指定回合。
报告可以恢复截断 JSONL 和缺失/损坏 NPZ，记录警告并保留可用证据。
旧日志仅提取已有任务成功率，标记 trace 不可用，不生成不存在的动作。
分析工具只写分析输出目录，不改写原始日志和正式汇总。

当前诊断阈值明确显示在报告中：相邻臂命令跳变超过 0.5 rad；
命令变化超过 0.05 rad 且实际位移小于 0.001 rad。
归一化越界仅用于 bounds 系列的有效维度；关节越界使用实际关节限位。
这些是定位观察，不能单独解释任务失败。

| 耗时字段 | 含义 |
| --- | --- |
| `infer_ms` | 策略调用耗时，包含追踪数组捕获，不含 NPZ 写入 |
| `preprocess_ms` | 特征准备与组 batch |
| `sample_ms` | 模型采样及将结果返回 CPU 的耗时 |
| `unapply_ms` | 反归一化和机器人动作映射 |
| `rtt_ms` | 客户端请求往返，包含服务处理和记录 |
| `logging_ms` | server_timing 中为预测 NPZ 写入，episode.json 中为客户端详细事件写入 |
| `execution_ms` | 单次 take_action 调用 |
| `prev_total_ms` | 历史兼容字段，包含等待客户端动作执行，不能当作模型耗时 |

full/off 模式都在 `episode.json.metrics` 保存请求数、推理、往返、执行耗时总量。
报告提供请求延迟分位数和回合级指标，统计时排除 reset 请求。
实际开销对比应使用同任务、相同回合数、精度和 `use_length`，单独考虑首次 compile 的耗时。
使用 `diagnostics.json` 的回合指标和 `artifact_bytes` 比较，不能用合成接口测试估计 GPU 仿真开销。

## 验证

```bash
python -m unittest discover -s tests -p test_eval_logs.py -v
bash -n experiment/robotwin/start_robotwin_infer_and_eval.sh
```

测试覆盖生产推理方法的动作/RNG 一致性、完整 horizon 和 padded 维度、缓存行为、
成功/步数边界、状态语义、规划异常恢复、写入失败、断连、中断、重试与截断恢复，
并通过双槽位真实 WebSocket 进行关联验证。推理方法测试使用实际 torch CPU 张量及
轻量模型/特征替身，无需导入 CUDA 模型和加载 checkpoint。
部署验收还需在实际 Linux GPU/仿真环境完成一个任务两个回合，再完成双槽位运行，
核对视频、执行计数、状态和正式汇总，并测量 full/off 的实际开销。
