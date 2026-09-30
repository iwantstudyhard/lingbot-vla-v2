# LingBot-VLA 2.0 / RoboTwin：官方对比与重构交接

更新：2026-09-30。接手同学请先读本文件，再读 [新clean训练指南](clean_training/README.md)。这不是“所有改动都已在服务器跑通”的保证书；这里明确区分已核验代码、本机CPU测试和仍需服务器验收的内容。

## 0. 对比基线与归属

- 本地保留的上游基线：`be969b8fd117fb70550c5d4bf4bc328211b5b1b6`，位于初次本地定制提交 `4b04853` 之前。可直接用 `git diff be969b8 -- <文件>` 对比已跟踪文件；未提交新文件还需单独查看。
- 本次通过网页核对 [官方README](https://github.com/Robbyant/lingbot-vla-v2)、[官方数据读取器](https://github.com/Robbyant/lingbot-vla-v2/blob/main/lingbotvla/data/vla_data/base_dataset.py)、[官方训练器](https://github.com/Robbyant/lingbot-vla-v2/blob/main/tasks/vla/train_lingbotvla.py)、[官方评测客户端](https://github.com/Robbyant/lingbot-vla-v2/blob/main/experiment/robotwin/eval_policy_client_lingbotvla.py)与[官方launcher](https://github.com/Robbyant/lingbot-vla-v2/blob/main/experiment/robotwin/start_robotwin_infer_and_eval.sh)。网页main是可变引用，提交列表缓存落后，本机git远程读取失败，**未把上述本地SHA冒称为当天远程HEAD**。
- `4b04853`：先前同学/项目已有的clean manifest、统计、自定义配置及日志等。
- `7564d5b`：训练分析与实时图表；`e22795a`至`3803443`等：推理修复；`fc152d9/960ed37/f3fe7e9`：新训练配置、mask、检查；`e32067d`：checkpoint保留策略。
- 本次待提交：归一化真实接线/快照闭环、完整clean独立核算、视频episode边界、隔离阶段二增强、两阶段启动器、清理与交接文档。没有代用户commit/push，也没有更改服务器上正在运行的进程。

官方现已有Muon/Distributed Muon、FSDP2、梯度累积、MoE、expert LR缩放、sequence/router辅助loss、未来depth/video teacher、TensorBoard、异步HF导出、多GPU队列评测等。**这些不是我们的原创新增**。动作空间、50步预测horizon和主体架构未重写，未实施“轻微改小架构”。

## 1. 功能对照总表

| 项目 | 官方/原先行为 | 本分支改动或新增 | 主要文件 | 注意事项 |
|---|---|---|---|---|
| 训练数据 | 官方范例clean+randomized | 竞赛仅本地官方clean，单条manifest、548893行/2500episodes、来源文件指纹启动检查 | assets/training_data/robotwin_clean_only.txt；tools/clean_training_common.py | 能核对副本字节，不能独立认证下载来源 |
| 归一化真实接线 | 默认transform从robot配置读stats；训练YAML的stats未传入 | 显式把data.norm_stats_file传入实际transform，并检查实例加载路径 | base_dataset.py；utils.py；train_lingbotvla.py | 修复这次发现的历史漏接，不再只比count |
| clean统计核算 | 官方自适应直方图统计；先前项目clean_only文件 | 从全部原始Parquet独立重算、核对旧clean_only，产出verified新文件及报告 | tools/verify_clean_norm.py；assets/norm_stats/robotwin_clean_verified.json；docs/clean_training/norm_verification.json | 精确经验分位数不同于直方图，非bitwise相同 |
| 训练/推理统计闭环 | 指向外部可变stats文件，可发生训练与推理不同源 | 每run保存统计+哈希manifest；推理自动读取该模型run快照，拒绝缺失/篡改/不兼容override | lingbotvla/utils/normalization_contract.py；deploy/lingbot_vla_v2_policy.py | 搬模型要带run配置及normalization，不能只拷HF目录 |
| 视觉是否训练 | 官方RoboTwin配置默认视觉可训练；同学旧冻结配置 | 新clean配置解冻visual；修正freeze_vit真实模型路径，启动检查实际可训练参数数目 | train_lingbotvla.py；robotwin_clean_stage1.yaml | 解冻不是架构创新；teacher仍冻结 |
| action尾部padding | 官方loss用joint mask，没有把episode action_is_pad并入reduction | 关节有效维×有效时间步mask；重复batch兼容；padding的loss与梯度排除 | models/vla/lingbot_vla/loss_utils.py；modeling_lingbot_vla_v2.py | 只改action监督，不是attention padding或所有future loss mask |
| 视频episode边界 | v3共享MP4只加from_timestamp，数据比视频长时尾部可读下一episode | 新clean配置启用每相机CFR有效区间clamp，终点exclusive，缺帧重复最后已有帧 | utils/episode_boundaries.py；base_dataset.py；extensions/clean_stage2/dataset.py | 当前不新增future auxiliary loss padding mask |
| 小显存训练配方 | 官方大规模RoboTwin训练GBS1024、max50k；用户4×48GB | BF16，micro1×4卡×accum8=GBS32，max18k，LR1e-5/cosine/3%warmup，关闭compile | robotwin_clean_stage1.yaml | 是硬件/实验配方，不是等价复现官方大batch指标或收敛承诺 |
| 两阶段解耦 | 无本竞赛独立视觉适配入口 | stage1无增强；stage2只加载选定stage1 HF权重、新optimizer/scheduler、独立config/output | tools/train_clean_stage1.sh；extensions/clean_stage2/train.sh/train.py/config.yaml | 旧混合统计模型被新stage2入口拒绝 |
| 合法在线增强 | 本分支此前无同步多视角的竞赛增强模块 | 30%raw、70%augmentation，scene共享参数、current/future replay、低强度光度退化 | extensions/clean_stage2/augmentation.py/.json；dataset.py | 第一阶段不import增强模块；不改动作/state/语言 |
| synthetic texture/clutter | 无背景保护的本竞赛模块 | 程序纹理、软矩形/椭圆等，头相机人工审核safe profile，保护边界、限制覆盖；无授权回退光度 | augmentation.py/.json | 默认safe_profiles空，所以默认无图形覆盖；腕相机禁止覆盖 |
| 增强预览 | 无本模块真实batch对比 | 保存真实消费rank0样本current/future、原图/增强图/mask及随机参数；另有离线OpenCV预览 | previews.py；preview.py；train.py | rank0窗口统计不是全GPU统计；离线预览不是生产codec验证 |
| 日志/运行隔离 | 官方通常由调用方安排stdout及output | train.sh时间戳UUID日志；启动器唯一新run；保留每run有效配置 | train.sh；training_metrics.py；两阶段launch.py | 历史外部checkpoint路径不移动；正式run默认repo内 |
| 训练图表 | 官方已有TensorBoard，不等于本地自动PNG报告 | 结构化JSONL，CSV/滚动均值/窗口统计，损失/梯度/LR/MoE/时间/吞吐等图与HTML；每次save快照 | training_metrics.py；tools/render_live_training_visuals.py；tools/analyze_training_log.py | 图表best-effort，错误有日志；不修改模型文件 |
| 保留3个checkpoint含best | 官方现有保存/异步HF，无本分支此rolling best策略 | 每500步保存，窗口平均total训练loss最低的best占1席，其余最新；atomic best记录；保护在途HF读取 | checkpoint_retention.py；arguments.py；train_lingbotvla.py；async_hf_checkpoint.py | best≠评测最佳；正在写/导出期间会临时超3，需要磁盘余量 |
| 连续推理视频 | 官方chunk执行中get_obs被注释，缺逐动作刷新 | 每后续动作前刷新录帧、成功时补终态帧；禁视频时不额外逐动作刷新 | experiment/robotwin/eval_policy_client_lingbotvla.py | 播放时长取决于动作数/fps，不等于训练demo长度或真实耗时 |
| use_length50→10 | 官方server/launcher默认50 | 默认执行预测前10动作再观测重规划；检查1..50合法 | deploy/lingbot_vla_v2_policy.py；start_robotwin_infer_and_eval.sh | 模型仍预测50；10是保守默认不是任务最优保证 |
| 评测兼容性 | 官方通用启动框架，用户安装env_cfg布局/Python环境不同 | task/camera/embodiment走CONFIGS_PATH；移除无用ACT.get_model；同步客户端/deploy helper；仿真用单独conda Python | eval_policy_client_lingbotvla.py；launcher | 会把仓库客户端同步到外部RoboTwin脚本，外部本地改动须先保护 |
| 服务就绪与连接 | 官方已有server/launcher | 增加healthz就绪门、超时/进程检查、localhost/NO_PROXY、握手异常重试 | launcher；deploy/websocket_client_policy.py | lazy compile的首请求仍可慢；有占用不等于成功评测 |
| 评测次数/进度/退出 | 官方已有多GPU队列；默认每任务100 | num_episodes、task_offset/skip_task、定期任务/episode进度、统计记录；SIGINT/SIGTERM清理后退出不再重试新任务 | launcher；eval_policy_client_lingbotvla.py | 4卡4槽各跑任务，非单条轨迹4卡联合推理 |
| 审计与目录卫生 | 先前有旧yaml、bak、tracked pyc/egg-info等 | 一个阶段一正式配置+独立阶段二目录；清理备份/缓存/暂停云helper；增加忽略规则、回归测试和文档 | .gitignore；tests/test_clean*.py；docs/ | 历史日志/图表与官方参照配置未删 |

“服务就绪等增加”的具体归属以本地固定基线的diff为准；官方未来若吸收类似改动，不能再把main全部差异照搬归功本分支。

## 2. 历史归一化事故说明：不能再漏查

历史两个自定义训练配置都写过 `norm_stats_file: assets/norm_stats/robotwin_clean_only.json`。这句话本身正确，但修改前调用链如下：

```text
训练YAML声明 clean_only
  -> VLADataset构造FeatureTransform（未传norm_stats_path）
  -> FeatureTransform默认None
  -> robot_config.norm_stats
  -> assets/norm_stats/robotwin.json
```

推理旧逻辑反而默认data.norm_stats_file，因此可以出现训练用混合stats、推理用cleanstats。不能只看clean loss下降、clean manifest、打印的count就判定闭环正确。这是此前排查遗漏，不要在重构总结中隐去。

用户服务器核对：robotwin.json count6062592、LF SHA256 `488ec6569c30dd69419358241efdb2e6c3488117768e7028a1e891be83e10db8`；clean_only count548893、LF SHA256 `8bd8a7fdd3b4a48d189fadfd2fb7ff7b12aef1a5763ccef52a085c7e202ae40e`。本地换行归一后吻合。这里只证明当时检查的文件，不是逐个历史进程取证。

新的stage1：clean_verified -> run/normalization快照 -> 训练实际transform -> 该run的推理transform/反归一化，全程同一语义哈希。

独立核算中，state均匀行统计，action按官方50步chunk合并权重统计，保留重复终端动作统计语义。action有效loss mask与统计权重是两个概念，未偷偷改成delta actions或逐步统计。mean/std差小于约4.9e-6，min/max相同；q01/q99最大差约0.00474，完整容差与所有源SHA见核算报告。精确分位数实现需作为独立实验版本记录。

## 3. 视频边界与padding的区别

实际clean v3数据全部2500episodes，三路CFR视频。例episode0数据139行、head视频仅138帧；原最后timestamp落入episode1。审核发现head619/left137/right100条episode存在“最后数据时刻越过本相机视频终点”的风险，三路有重叠，不能相加当独立样本数。这是元数据/解码例证，不是已经测得的成功率影响大小。

动作padding修复：停止把重复动作尾部算作真实监督。视频边界修复：防止RGB读到下一episode。两个修复不互相替代。未来图像末尾仍可能重复已有帧，当前未来depth/video loss并未新增全面padding排除。

第一次训练已有的padding修复未因此撤销；此次新的阶段一额外修复视频边界和归一化。历史训练目标和数据行为发生了差异，所以新输出、新初始化，不在旧2000/20k checkpoint上混用。

## 4. 增强实验边界

默认实际30%raw+70%photometric。合成背景纹理/clutter要全轨迹审核后授权，未自动从random数据学背景、不读取random图片作为训练背景。程序操作是像素增强，并非模拟器再生成observation。

三相机共享光度参数、各自局部空间场跨current/future一致；无3D映射，不应称为物理上严格一致的“新场景”。目标颜色有任务语义，未加入大hue轮换；腕相机目标占画面大，禁合成覆盖。静止目标也受保护，不能只用运动热图。

teacher默认clean RGB、student增强RGB是新训练机制选择；可进行teacher_aug消融。基础LR3e-6，不改MoE主体架构。当前未实施state噪声、task均衡采样、额外验证集、自动评测选best或新model head。

没有“增强百分之百提高性能”的保证。先完整验证新clean基线，再固定seed/任务/精度/use_length做配对评测，检查clean性能退化和random提升。每任务5条只能初筛。

## 5. 重构建议：保留什么、分离什么

建议同学重构时保留上游源码相对目录，避免破坏imports。先迁移入口与配置，不直接把核心包改名：

```text
lingbot-vla-v2/
  lingbotvla/                         # 上游核心+最小数据/loss/统计契约修复
  deploy/                            # 推理服务及连接器
  experiment/robotwin/                # 仿真评测入口
  configs/vla/robotwin/
    robotwin.yaml                    # 官方参照，非竞赛训练入口
    robotwin_dist_muon.yaml           # 官方参照（以实际文件名为准）
    robotwin_clean_stage1.yaml        # 竞赛唯一阶段一配置
  extensions/clean_stage2/            # 独立增强实验模块
  tools/                             # 启动/核算/分析工具
  assets/training_data/               # manifest，不放巨大数据
  assets/norm_stats/                  # 官方/历史参照与新verified统计
  docs/clean_training/                # 有效命令、核算证据
  tests/                             # CPU回归与服务器集成验收
  training_logs/                     # 忽略Git
  train_outputs/                     # 新运行忽略Git；保留旧离线分析
```

模型、datasets、RoboTwin仍可在外部workspace。必须存在的训练外部依赖：官方foundation VLA权重、Qwen3-VL-4B-Instruct processor/tokenizer/model config、MoGe权重、LingBot-Depth权重、DINO video teacher权重+config、完整clean LeRobot数据及解码运行库。RoboTwin/assets/仿真conda环境仅评测需要，不是clean训练数据生成依赖。

以后把路径集中管理时，确保两个阶段都显式解析同一设置；不要把多个相对路径解释成不同cwd。当前保持/scratch路径，暂停的云路径改写helper已清理。单卡A10080G配方未在这次实现/验证，不要把4卡GBS32启动器拿去单卡直接跑。

每个新run必须带有效config、normalization、初始权重来源、增强参数、日志、checkpoint、指标和评测配置。并列运行用不同master端口/推理端口/GPU，不根据输出名猜模型参数。

旧root日志和旧train_outputs分析是审计证据，不导入训练。新运行忽略Git；不要把DCP/HF大权重、数据、缓存推上GitHub。README历史官方命令只是参照，竞赛启动以新指南为准。

## 6. 清理清单与恢复

本次150个历史路径先逐文件备份并核对哈希：

- 4个旧clean自定义配置退出有效入口，unfreeze配置改为新的stage1配置；3个freeze/local变体删除。
- 7个代码/launcher .bak删除。
- 暂停的tools/configure_cloud_a100.sh删除。
- 133个tracked .pyc与5个egg-info文件删除；生成缓存未来由ignore保护。

Windows本地恢复备份：
`C:/Users/zrobot/.codex/visualizations/2026/09/29/01a0eadb-866e-7f62-a83b-a2d21788c5ce/cleanup_backup_20260930/`，有manifest.json和原相对路径。备份不在Git；原先tracked旧文件也可从清理前提交 `e32067d` 中用只读git show恢复到明确新位置，勿整个reset覆盖当前改动。

没有删除官方配置、官方robotwin.json、用于核对的旧clean_only.json、预训练资源、外部checkpoints、root历史日志或已离线生成图表。无法界定所有“无关文件”时不进行全仓库大删；上述清理是确定的过时入口与生成物。

## 7. 验证与服务器验收清单

本机：

- 全量clean Parquet独立核算通过，29个源文件指纹已记录。
- 7500相机片段算术边界检查通过。
- 37个clean模块CPU unittest通过：增强参数replay、保护区域、clean identity、padding梯度、真实预处理动作反归一化roundtrip、统计快照/篡改/搬迁、启动命令与legacy拒绝等。
- 实际clean三路OpenCV预览已经生成并查看。生产codec/LeRobot导入、CUDA/NCCL、teacher前向、4卡真实保存尚未在本机运行。

服务器必须：

1. 停止旧运行后更新代码，不在运行中偷偷换统计或配置。确认当前branch/commit，旧训练输出不删。
2. dry-run验证本地clean副本/权重/teacher路径；源字节不同先重新审计，不删除失败校验。
3. 新阶段一smoke5步，检查实际normalizer日志指向本run快照、视觉实际可训练数>0、loss/grad有限、没有跨episode读帧、checkpoint/HF/图片存在。
4. 正式阶段一；500步检查save/retention/plots，续训到同目标只能resume本次新run。读到所有checkpoint失败须报错。
5. 新模型推理冒烟：同一stats哈希、动作14维物理范围合理、连续视频、1条episode完成。规范比较优先FP32，BF16另标实验；官方README明确发布评测FP32，两者不能混比。
6. 阶段二独立新run，确认fresh optimizer/scheduler、实际增强预览和fallback。500步只是pilot，之后另定完整训练计划。
7. 同模型/seed/参数对比clean与random评测；成功率最佳模型另行记录，不把best_checkpoint.json的训练lossbest冒称评测冠军。

PyTorch/视觉开发技能影响了实现：增强放在CPU数据预处理、使用可重放参数、有限rank0预览、纯函数测试与统计契约；没有安装额外训练依赖，也没有宣称GPU已经验收。
