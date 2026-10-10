# AutoHorizon 首版接入与使用

实现分支：`ma/add_autohorizon`，基于 `ma/new_framework_main`。
上游锁定版本：`c7504f1756109103f2cfcc2e23f1b1a23841c885`。

## 已确定并实现的范围

首版覆盖 RoboTwin 单环境、`chunk_ret=True` 动作块推理。默认关闭。
`observe/auto` 使用非编译 eager 路径；不修改训练、checkpoint、归一化或去噪次数。
多样本 batch、单步缓存和编译优化不在首版范围；单元素 batch 仍按原协议返回批量维度。

AutoHorizon 选择的是执行长度 H，不是完整预测长度 T 或去噪次数。
模型完成原有采样后，执行前 H 步，再重新观测、推理。它不在块内利用新图像触发中断。
T 读取实际 checkpoint 的 `n_action_steps`，不写死为 50。

## 使用方式

沿用现有服务的模型路径、精度和机器人配置，增加以下参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--horizon_mode` | `fixed` | `fixed / observe / auto` |
| `--autohorizon_attention_step` | `3` | 从 1 开始计数的采样去噪步 |
| `--autohorizon_hold_thr` | `0.3` | soft pointer 平台阈值 |
| `--autohorizon_max_entropy_q` | `0.9` | entropy 筛选分位数 |
| `--use_length` | `10` | fixed/observe 的执行长度；auto 的失败回退长度 |

- `fixed`：沿用原有采样及固定前缀，不采集 attention，不增加 horizon 响应字段。
- `observe`：估计并记录 H，但仍执行 `use_length`，用于验证采集与对照算法。
- `auto`：执行估计的 H 步，估计失败时回退到 `use_length`。

`observe/auto` 即使传入 `--use_compile True` 也会强制关闭编译，在启动日志及模型元数据中记录请求值和实际值。
启动时校验 `chunk_ret=True`、采样步有效、阈值有限、分位数在 0..1、固定/回退长度在 1..T。
服务收到多样本 batch 时明确报错。去噪步数少于默认采样步 3 时，需显式选择一个有效采样步。

单独启动服务（模型路径替换为实际 checkpoint）：

```bash
python -m deploy.lingbot_vla_v2_policy \
  --model_path /path/to/checkpoint/hf_ckpt \
  --chunk_ret True --horizon_mode observe --use_length 10
```

批量 RoboTwin 启动器沿用原参数，只需追加 `--horizon_mode observe` 或 `--horizon_mode auto`，例如：

```bash
bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path /path/to/checkpoint/hf_ckpt \
  --conda_sh /path/to/miniconda3/etc/profile.d/conda.sh \
  --num_tasks 1 --num_episodes 2 --num_gpus 1 --num_per_gpu 1 \
  --horizon_mode auto --use_length 10 --eval_trace full
```

启动器只向内置 `deploy.lingbot_vla_v2_policy` 透传 horizon 参数；自定义推理脚本只能使用 fixed 模式。

## Attention 与上游复现

利用现有 eager attention 的 post-softmax 权重，在选定去噪步提取最后 T 行、最后 T 列。
本项目 suffix 是 `[state, actions...]`，因此该切片排除了 state、视觉、语言和任务 token。
提取发生在 RoPE、KV cache、GQA 展开和 mask 生效后。
每层立即平均 head，按层累加/平均，保留 `[B,T,T]`，不平均 batch。
仅保存摘要，不持有完整 attention 切片、不采集 prefix、不增加模型 forward。
摘要和 pointer 算法使用 FP32，通过显式可选返回值传递，不用共享 attention hook。

移植 `pick_horizon_softpointer`、`_soft_pointer_prefix` 和 `bidir_soft_pointer`。
主路径采用上游双向算法，`run_len=1`；许可证和来源记录在模块头部，仓库 LICENSE 为 Apache-2.0。
测试内保存锁定版本的独立原函数，用于逐项数值与诊断对照。

根据已确认的取舍，以下行为有意保留：

1. 上游只保存指定去噪步的矩阵，之后除以总去噪步数；这不是多步平均。
2. 后向 `N_b` 使用映射回正序后的坐标；不替换成论文中的后缀覆盖长度。
3. 完整 T 步执行条件仍保留 `join_row` 判定。

例如 T=6、前半段全关注第 0 步、后半段全关注第 5 步的矩阵，上游返回 H=6。
这一行为已有回归测试；论文修正版留待后续独立实验。
T=1 直接返回 H=1，避免原函数的 `log(1)` 和标量窗口边界。
矩阵形状错误、非有限值、负值或行内有效质量不足时明确回退并记录原因。
模型异常、归一化异常和非法动作不会被 horizon 回退吞掉。

## 响应与诊断

observe/auto 在原有 `action` 和 timing 基础上增加：

- `horizon_mode`、`estimated_execution_horizon`、`execution_horizon`、`predicted_horizon`。
- `horizon_method`：`autohorizon / fixed / fixed_fallback`。
- `fallback_reason`（成功时为 null）和 `attention_step`。

非批量 `action` 为物理单位 `[execution_horizon,D]`；所有 action 输出字段及可选归一化返回同步截断。
auto 估计成功时允许 H 大于 `use_length`，上限为 T。
客户端校验声明 H 和动作数组长度一致，并继续使用现有 `execute_chunk`。
回合成功或达到 step limit 时，实际执行数可小于 H。

full 日志保留完整 T 步归一化动作、全部反归一化动作、`action_attention` 摘要和 pointer tensor 诊断。
标量 H、停止位置、拼接信息和回退原因进入请求诊断；`horizon_ms` 单独记录估计耗时。
客户端记录所选长度和实际执行数；off 模式不保存逐请求数组，但回合 metrics 仍汇总估计/选定/实际执行长度分布和回退次数。
模型元数据记录全部算法参数、上游版本和实际编译设置。回合 reset 清空最近 horizon 与预测状态。

## 验证

- 上游对照：单向/双向算法、均匀/对角/随机/平台矩阵、阈值变化，以及已知后向坐标行为。
- 模型数值：生产 eager attention、expert forward、KV cache、真实 suffix embedding、predict_velocity 和去噪循环，使用小型确定性 decoder 测试；涵盖 CPU、CUDA FP32 与 BF16 输入。
- 数值比较：fixed 与采集路径完整动作逐值一致，CPU/CUDA 随机数状态不变；只采选定步，排除 state，GQA、batch 独立及无完整矩阵引用。
- 服务接口：三种模式、全部输出字段切片、完整预测保存、无效估计回退、模型异常传播、batch 拒绝、reset、参数错误和关闭编译。
- 客户端/日志：生产 WebSocket 服务与模拟环境验证 full/off、声明长度校验以及实际执行提前结束；既有固定长度和单步缓存回归测试继续运行。

相关测试：`tests/test_autohorizon.py`、`tests/test_eval_logs.py`。
执行（需 PyTorch、einops、NumPy、PyYAML、matplotlib、msgpack、websockets）：

```bash
python -m unittest discover -s tests -p test_autohorizon.py -v
python -m unittest discover -s tests -p test_eval_logs.py -v
bash -n experiment/robotwin/start_robotwin_infer_and_eval.sh
```

本地已验证 CPU/CUDA 小模型路径和模拟环境接口。尚未完成实际 LingBot checkpoint 的闭环成功率、推理 p50/p95 和峰值显存对照：当前工作区没有模型权重，且未配置可运行的完整 RoboTwin 资源。
后续在部署环境先运行 observe，再以同 checkpoint、任务、seed、精度及去噪步数比较 fixed H=5/10/20 与 auto。
统计成功率、H 分布、推理次数、完整回合耗时、动作块交界跳变和峰值显存，不预设 auto 必然提升成功率。

## 来源

- [上游锁定版本](https://github.com/hatchetProject/AutoHorizon/tree/c7504f1756109103f2cfcc2e23f1b1a23841c885)
- [核心函数](https://github.com/hatchetProject/AutoHorizon/blob/c7504f1756109103f2cfcc2e23f1b1a23841c885/src/openpi/models_pytorch/pi0_pytorch.py)
- [Apache-2.0 LICENSE](https://github.com/hatchetProject/AutoHorizon/blob/c7504f1756109103f2cfcc2e23f1b1a23841c885/LICENSE)
- [论文 v2](https://arxiv.org/html/2602.21445v2)
