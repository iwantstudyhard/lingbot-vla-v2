# 独立第二阶段：合法 clean 像素增强

主命令、统计与续训约束见 [clean-only训练指南](../../docs/clean_training/README.md)；历史变化见 [重构交接](../../docs/REFACTOR_HANDOFF.md)。

本目录只通过独立训练入口启用，第一阶段不 import 本模块、不做在线增强、不接受增强配置。共同的归一化接线与episode边界修复用于两阶段新clean流程；不能因此宣称历史阶段一已使用修复。

## 配方

| 分支 | 抽样概率 | 行为 |
|---|---:|---|
| clean | 30% | 原始RGB不变 |
| photometric | 40% | 光度与概率性成像退化 |
| texture | 20% | 光度+授权背景内程序纹理；无授权则仅光度 |
| clutter | 10% | 光度+授权背景内软图形；无授权则仅光度 |

brightness0.75–1.25、contrast0.8–1.2、gamma0.8–1.25、saturation0.9–1.1、white balance各通道0.95–1.05。每样本至多选择一种noise/blur/JPEG退化：Gaussian sigma至3/255，blur sigma0.2–0.6，JPEG质量75–95；概率性soft shadow强度至0.2。

场景光度参数三路相机共享；同一路current/future复用噪声、阴影、纹理、图形。相机局部空间场不同，**不是物理三维一致的新场景**，也没有改变相机几何。第一约500 optimizer steps逐步增强；worker预取存在滞后，实际strength记录在预览JSON。

不做flip/旋转/透视/大crop，不改变state、action、语言、action mask。RGB uint8图像在模型image processor前增强；不改变动作归一化统计。默认depth/video teacher看同一clean样本的原始RGB，学生看增强图；`teacher_clean=false`是独立消融选项。

## 遮挡保护

所有区域默认保护；safe_profiles空时texture/clutter全部回退photometric，两路腕相机始终禁覆盖。数据中有贴边目标、静止目标、上方器具及颜色语义任务，不能把首帧角落/白色区域/低运动区域自动认定为背景。

审核某条episode**全部轨迹**，确认关键物体、机器人、交互区域不会进入可编辑区后，才可在augmentation.json配置：

```json
"safe_profiles": {
  "已审核的episode编号": {
    "camera_top": {
      "reviewed_entire_episode": true,
      "editable_rectangles": [[0.0, 0.0, 0.1, 0.1]],
      "protected_rectangles": [[0.0, 0.0, 1.0, 1.0]]
    }
  }
}
```

这是格式示意，不是可直接授权任何任务的区域；示意整图保护，不会覆盖。xyxy归一化坐标，可编辑区域向内收缩8像素。背景纹理alpha0.2；clutter使用软矩形/椭圆及程序低频/条纹色纹，像素覆盖≤整图3%、alpha≤0.5。JPEG/blur先执行，最后合成再限制mask，干扰不会因模糊泄漏到保护区。

当前**没有**默认全场景自动背景分割、外部纹理、目标粘贴、结构改小或动作/state噪声。没有实现全部形状库或一般random erasing；需任务保护才能扩展。程序纹理不是random场景替代品，也无法逼真复现真实3D遮挡。

## 命令

```bash
bash extensions/clean_stage2/train.sh \
  --init-hf /新阶段一运行/checkpoints/global_step_N/hf_ckpt \
  --gpus 0,1,2,3 --steps 500 --dry-run
```

移除dry-run启动。限定当前4卡BF16/GBS32配置，必须来自新audited stage1，同尺度；不加载optimizer/scheduler、不自动resume，不复用输出目录。新的阶段一启动命令单独是 `bash tools/train_clean_stage1.sh`。

离线预览不训练、不改数据：

```bash
python -m extensions.clean_stage2.preview \
  --dataset-root /scratch/YF_Data/lingbot_workspace/datasets/RoboTwin_lerobot_v30 \
  --episodes 0 500 1000
```

默认写repo下 `train_outputs/stage2_preview/<时间戳_UUID>`。这里只使用OpenCV作诊断，不替代生产解码器验证。

## 真实训练预览与边界

每次保存checkpoint时，预览位于其analysis快照的 `augmentation/`。从实际消费的rank0 batch收集，每分支至多一个近期样本；列为clean当前/aug当前/clean未来/aug未来/可编辑mask，行对应三相机。batch预览字段在forward前移除，避免进入模型。

manifest记录episode/frame、seed、参数、实际strength、分支计数、回退数和视频clamping；计数只代表rank0本次快照窗口，不代表全部GPU的全局分布。

每相机按v3 from/to的独占终点限制CFR读取，缺失尾帧重复最后已有帧，不读下一episode，不渲染/插值新observation。这不同于动作尾部padding loss mask；**当前未新增未来辅助loss的padding mask**。

该增强只能作为待评测假设：保留阶段一模型、固定对比评测精度/seed/use_length，先光度再安全clutter消融。不能承诺一定提高random任务成功率。
