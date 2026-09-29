# 训练配置对照

正式配置：`C:/Users/zrobot/Desktop/lingbot-vla-v2/configs/vla/robotwin/robotwin_clean_freeze_vision_bf16_formal.yaml`

## 官方配置 → 正式训练配置

| 参数 | robotwin.yaml | robotwin_clean_freeze_vision_bf16_formal.yaml |
|---|---|---|
| `data.norm_stats_file` | `<missing>` | `assets/norm_stats/robotwin_clean_only.json` |
| `data.num_workers` | `8` | `4` |
| `data.train_path` | `assets/training_data/robotwin.txt` | `assets/training_data/robotwin_clean_only.txt` |
| `model.model_path` | `/path/to/pretain_ckpt/hf_ckpt` | `/scratch/YF_Data/lingbot_workspace/models/robbyant--lingbot-vla-v2-6b/snapshots/master` |
| `model.tokenizer_path` | `/path/to/Qwen3-VL-4B-Instruct` | `/scratch/YF_Data/lingbot_workspace/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master` |
| `train.align_params.depth.moge_path` | `/path/to/depth/moge2-vitb-normal.pt` | `/scratch/YF_Data/lingbot_workspace/models/depth/moge2-vitb-normal.pt` |
| `train.align_params.depth.morgbd_path` | `/path/to/depth/model.pt` | `/scratch/YF_Data/lingbot_workspace/models/robbyant--lingbot-vla-v2-6b/snapshots/master/depth/model.pt` |
| `train.align_params.video.ckpt_path` | `/path/to/dino_video/teacher_step_10000.pth` | `/scratch/YF_Data/lingbot_workspace/models/robbyant--lingbot-vla-v2-6b/snapshots/master/dino_video/teacher_step_10000.pth` |
| `train.align_params.video.config_path` | `/path/to/dino_video/config.yaml` | `/scratch/YF_Data/lingbot_workspace/models/robbyant--lingbot-vla-v2-6b/snapshots/master/dino_video/config.yaml` |
| `train.enable_fp32` | `True` | `False` |
| `train.enable_gradient_checkpointing` | `False` | `True` |
| `train.enable_training_visualization` | `<missing>` | `True` |
| `train.freeze_vision_encoder` | `False` | `True` |
| `train.freeze_vit` | `<missing>` | `True` |
| `train.global_batch_size` | `1024` | `4` |
| `train.micro_batch_size` | `32` | `1` |
| `train.output_dir` | `/path/to/save_ckpt` | `/scratch/YF_Data/lingbot_workspace/train_outputs/robotwin_clean_freeze_vision_bf16` |
| `train.save_steps` | `10000` | `20000` |
| `train.training_visualization_flush_steps` | `<missing>` | `100` |
| `train.training_visualization_output_dir` | `<missing>` | `train_outputs/robotwin_clean_freeze_vision_bf16` |
| `train.training_visualization_rolling_window` | `<missing>` | `200` |
| `train.use_compile` | `True` | `False` |

## 试运行配置 → 正式训练配置

| 参数 | robotwin_clean_freeze_vision_bf16.yaml | robotwin_clean_freeze_vision_bf16_formal.yaml |
|---|---|---|
| `train.enable_resume` | `False` | `True` |
| `train.output_dir` | `/scratch/YF_Data/lingbot_workspace/train_outputs/robotwin_clean_freeze_vision_bf16_trial` | `/scratch/YF_Data/lingbot_workspace/train_outputs/robotwin_clean_freeze_vision_bf16` |
| `train.save_steps` | `10000` | `20000` |
| `train.training_visualization_output_dir` | `train_outputs/robotwin_clean_freeze_vision_bf16_trial` | `train_outputs/robotwin_clean_freeze_vision_bf16` |

说明：列表和字典按解析后的 YAML 值比较，换行风格、注释和键顺序不会造成伪差异。
