# 官方模型 vs 我们模型：动作诊断

这是独立的诊断目录。不修改训练、归一化、正式推理脚本或 RoboTwin 源码。
需要把整个 `diagnostics/action_compare/` 放在最新 lingbot-vla-v2 仓库根目录下。

## 运行（服务器）

先确保 GPU 2、3 空闲，上一轮评测已经结束。两张卡各加载一份模型；不是两张卡合并显存。
在 lingbotvla 环境执行；仿真另外使用 RoboTwin 环境的 Python。

```bash
cd /scratch/lingbot_ws/lingbot-vla-v2
conda activate lingbotvla

python diagnostics/action_compare/run.py \
  --ours /scratch/lingbot_ws/lingbot-vla-v2/outputs/train_outputs/robotwin_clean_stage1_20261001_021839_b6a91337/checkpoints/global_step_18000/hf_ckpt \
  --official /scratch/lingbot_ws/lingbot-vla-v2/models/lingbot-vla-v2-6b-robotwin/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
  --official-norm /scratch/lingbot_ws/lingbot-vla-v2/assets/norm_stats/robotwin.json \
  --qwen /scratch/lingbot_ws/lingbot-vla-v2/models/Qwen3-VL-4B-Instruct/snapshots/master \
  --sim-python /home/kemove/miniforge3/envs/RoboTwin/bin/python \
  --gpus 2,3 --task hanging_mug --use-length 20 --max-actions 160
```

默认 FP32、关闭 compile、固定场景种子（从 100000 开始，通过原评测器的合法场景检查）。
每次推理两模型使用相同采样种子，便于对照；这是诊断专用设置，不等同于默认随机采样的正式评测。
执行前检查输入模型、Qwen、配置、端口；不会杀其他进程，不会改模型路径或归一化文件。
加载GPU权重前先用正式推理的同一个解析器核验两份归一化路径和内容，主终端输出各自路径、count和语义哈希。
官方旧配置可能声明 `norm_stats_file: null`，因此必须显式传入 `--official-norm`。
这个参数只给官方服务使用；我们的模型仍读取自己的 `normalization/norm_stats.json`，不会回退到官方混合统计。
原归一化契约、哈希校验和拒绝不一致override的逻辑不变。
官方旧版统计没有 `min/max`，预检查按保存配置中的归一化模式校验实际必需字段；
`bounds_99_woclip` 检查 `q01/q99` 及映射所需的 `mean`，不凭空补字段。
我们的 audited clean 快照仍执行完整严格校验，不能借旧格式兼容绕过契约。

2026-10-08 已将本地 `assets/norm_stats/robotwin.json` 与官方仓库同文件核对一致：
`count=6062592`，语义SHA256为 `0404d7cf69b5560a9adaedfb18b1f86715242ff3d7a0201f47face064eeefb5a`。
服务器文件也应核对该语义哈希，不只看名字或count。它仅用于发布的官方模型，不可替换我们clean模型的统计。
运行期间有逐动作进度，每10秒在主终端显示一次。遇到初始化异常打印完整堆栈并退出，不再无限换种子重试。
默认每个模型最多执行160个动作，便于尽快检查抖动。达到上限会标记 `diagnostic_truncated`，
此时失败不代表完整任务失败，不能拿结果计算正式成功率。要完整跑一回合使用 `--max-actions 0`。

## 对比方式

1. 我们的模型控制机器人；每次三路RGB、state、指令同时送给两套模型。
2. 官方模型控制机器人；再做同样的对照。两次都使用原评测器、同场景种子。
3. 每次两模型都预测完整50动作，只执行主模型前20动作，与正式 `use_length=20` 的截断机制一致。
4. 两模型各用原代码解析自己的归一化文件，绝不强行共用一个统计文件。
5. 记录原始输入和两模型的输出；另外读取真实关节qpos，而非把 observation.state 中的 drive target 当实际位置。
   两个夹爪值是原代码的归一化命令值，不声称它们是实际夹爪测量；跟踪误差只计算12个机械臂关节。
6. TOPP 的调用包装只记录结果/错误，然后原样返回或重新抛出，由原控制器继续处理，不改变插值或动作。
7. 首次观察额外对比4个采样种子，两模型种子成对相同；额外预测不执行机器人动作，用来检查残留噪声敏感性。

`observations/` 仅用于这次评测的故障诊断，**不是训练数据**。禁止加入训练manifest或二次训练。
没有生成额外训练样本、重绘观测或调用任何采集脚本。

## 去哪里看图

主终端会打印唯一目录，位于：

```text
outputs/eval_outputs/action_compare/<时间_随机ID>/
  ours_server.json / official_server.json    模型、归一化路径/哈希/帧数、去噪步数
  ours_server.log / official_server.log
  ours_eval.log / official_eval.log
  ours/                                   我们模型控制的轨迹
  official/                               官方模型控制的轨迹
    requests.jsonl                        观察哈希、场景/采样种子、指令
    executed.jsonl                        目标、实际qpos、chunk、TOPP异常
    observations/*.npz                    每次真正发送的图像和state
    predictions/*.npz                     两模型同输入的完整50动作及有效归一化输出
    eval_results/                         视频
    completion.json                       是否成功/是否诊断截断
    noise_probes.npz                       首次同一观察的4个采样种子输出
  analysis/
    01_same_observation_first_chunk.png    同观察、同采样种子的14维预测曲线
    02_executed_targets_and_actual.png     执行目标 vs 实际关节；标注chunk边界
    03_normalized_increment_heatmap.png    归一化空间、去除padding的关节增量
    04_inside_vs_boundary.png             chunk内部 vs 换chunk边界的目标跳变量
    05_sampling_noise_sensitivity.png      固定首次观察，不同采样种子的首动作波动
    paired_metrics.csv                    每次推理两模型的数值统计
    summary.json                          跟踪误差、TOPP异常、初始观察是否相同等
```

图1是严格相同输入的比较；图2是两个独立闭环轨迹，动作不同之后观察自然不同，
不能把两条轨迹的差当成模型误差或把官方轨迹当真值。归一化统计可能不同，
图3只定位模型各自归一化空间的高频输出；物理关节比较以图1的弧度为准。

图表方案：前两张为关节小多图；第三张为共享色标的增量热图；第四张为从零开始的柱状图。
第五张也是从零开始的柱状图，不把4次噪声探测当统计置信区间。
图1蓝色实线是官方、橙色虚线是我们；图2用颜色区分模型、实线表示目标、虚线表示实际qpos，夹爪面板只绘命令。
横轴是动作序号，不是视频秒数。增量为rad/action，不称作物理速度或加速度。
视频10FPS只是回放编码；现有控制器每个动作经过TOPP插值，不能用视频FPS乘chunk长度来确定实际控制周期。
绘图至少要求两条完整诊断记录、各至少两动作。首次不足则报错，不补造数据。

## 怎么判断

- 第一chunk内，我们的目标已高频来回而官方平滑：优先查模型采样/训练目标/动作序列，不只查换chunk。
- 归一化输出平滑，反归一化后的弧度异常放大：结合两份统计和关节映射进一步核对，不能直接替换统计。
- 目标平滑但真实qpos偏离，或TOPP有异常：优先查控制执行。
- 换chunk边界跳变显著更大，内部平滑：再做相同输入 `use_length=10/20/50` 对照。
- 平滑不等于任务正确；动作曲线只能定位故障，不能证明准确率或预测成功率。

训练loss是流匹配训练目标的平均误差，不是闭环成功率。即使均值低，
噪声采样、多步积分、关键关节误差和闭环分布偏移仍可能产生失败。

## 仅重新绘图 / CPU测试

```bash
python diagnostics/action_compare/plot.py /实际诊断目录
python -B -m unittest discover -s diagnostics/action_compare -p 'test_*.py' -v
```

CPU测试只用虚构模型/物理夹具检查采集和计算正确性；测试图不是模型结果。
真正GPU推理只能在有两套权重和RoboTwin运行环境的服务器执行，本地没有代跑。
