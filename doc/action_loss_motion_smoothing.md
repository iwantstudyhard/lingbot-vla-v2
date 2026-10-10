# 动作 Loss 修改说明：运动帧加权与轨迹平滑

更新时间：2026-10-10

## 当前状态

- 已停止服务器上原来的 continuation 训练进程（launcher PID 1892119 及其 torchrun worker）。
- 当前 Git 分支：`changeloss`。
- 本次修改保留了工作区原有的 `RoboTwin` 子模块改动、`experiment/robotwin/.infer_gpu23.z7upLr.sh` 和 `nohup.out`，没有覆盖或清理它们。

## 修改目标

原来的 `L1_fm` 只比较 Flow Matching 的逐元素速度场误差：

```text
x_t = t * noise + (1 - t) * action
u_t = noise - action
L_fm = mean(|v_t - u_t|)
```

它可以让平均 loss 较低，但没有直接约束动作序列的速度和加速度，也会让大量静止帧主导平均值。此次加入两个阶段：

1. **第二阶段：运动帧加权**：保留静止帧，但提高专家动作变化明显的时间步的权重。
2. **第三阶段：动作轨迹平滑**：从 Flow Matching 的当前预测恢复 clean-action 估计，再比较相邻动作差分和二阶差分。

## 第二阶段：运动帧加权

实现位置：

- `lingbotvla/models/vla/lingbot_vla/loss_utils.py`
- `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`
- `lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py`
- `tasks/vla/train_lingbotvla.py`
- `configs/vla/robotwin/robotwin_clean_stage1.yaml`

对每个有效动作帧计算它与前一个有效帧的平均绝对动作差：

```text
motion_score[t] = mean(abs(action[t] - action[t-1]))
```

每个样本内部用有效帧平均 motion score 做归一化，并设置：

```text
weight[t] = 1 + alpha * normalized_motion_score[t]
weight[t] = min(weight[t], max_weight)
```

静止帧的基础权重仍为 1，episode padding 和无效关节不会参与计算。动作 loss 的分母同步使用加权有效元素数，避免只放大分子导致 loss 尺度失真。

当前 clean RoboTwin 配置：

```yaml
motion_frame_weight_alpha: 0.5
motion_frame_weight_max: 3.0
```

这表示运动帧最多获得 3 倍权重，不会删除静止数据。

新增监控指标：

- `action/motion_weight_mean`
- `action/motion_weight_max`
- `action/motion_frame_ratio`
- `action/effective_weight_mean`

## 第三阶段：动作平滑 Loss

实现位置：

- `lingbotvla/models/vla/lingbot_vla/loss_utils.py`
- `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`

根据当前训练插值：

```text
x_t = t * noise + (1 - t) * action
v_t = noise - action
```

可以得到 clean-action 估计：

```text
action_hat = x_t - t * v_t
```

平滑项不直接约束随机噪声或随机速度目标，而是在 `action_hat` 上计算：

```text
L_velocity = mean(|delta(action_hat) - delta(action_gt)|)
L_acceleration = mean(|delta2(action_hat) - delta2(action_gt)|)
```

其中 `delta2(a[t]) = a[t] - 2*a[t-1] + a[t-2]`。所有差分都使用连续有效时间步和有效关节 mask，episode 尾部 padding 不会参与。

最终动作相关 loss 为：

```text
L_action = L_fm_weighted
         + smooth_velocity_loss_weight * L_velocity
         + smooth_acceleration_loss_weight * L_acceleration
```

当前 clean RoboTwin 配置：

```yaml
smooth_velocity_loss_weight: 0.05
smooth_acceleration_loss_weight: 0.01
```

这两个权重保持较小，先让 Flow Matching 主目标占主导，避免把必要的快速动作过度抹平。

新增训练日志和 TensorBoard 指标：

- `SmoothVel_Loss`
- `SmoothAcc_Loss`
- `MotionWeight`
- `training/action_smooth_velocity_loss`
- `training/action_smooth_acceleration_loss`
- `training/action_motion_weight_mean`

## 未改变的部分

- 基础 Flow Matching 的 `L1_fm` 仍然保留。
- 有效关节 mask 和 `action_is_pad` padding mask 仍然生效。
- 深度、未来深度、未来视频和 MoE 辅助 loss 的原有权重没有改变。
- 推理阶段的动作平滑配置没有在本次改动中强制打开；训练 loss 和推理后处理仍可单独做消融对比。

## 下一次训练建议

新的配置需要从一个已有 checkpoint 重新开始训练。建议至少对比：

1. 原始 checkpoint + 原始 loss；
2. 原始 checkpoint + 运动帧加权；
3. 原始 checkpoint + 运动帧加权 + 本文平滑 loss。

每组都使用相同任务、相同随机种子和相同推理设置，记录：

- 仿真成功率；
- 机械臂动作速度、加速度和 jerk；
- 平滑前后的动作曲线；
- `VLA_Loss`、`SmoothVel_Loss`、`SmoothAcc_Loss`；
- 运动帧和静止帧的分组 loss。

当前训练只按训练 loss 选择 checkpoint，后续仍应使用固定的 RoboTwin 验证任务参与 checkpoint 选择。

## 验证说明

本次没有重新启动训练，也没有运行完整仿真评测。修改后应先进行单卡、小 batch 的启动检查，再开始长时间训练。
