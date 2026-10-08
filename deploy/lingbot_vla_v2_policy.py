import os
import sys
import time
import yaml
from types import SimpleNamespace

from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingbotvla.utils.arguments import workspace_path

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import (
    AutoConfig,
)

from typing import Dict, List, Optional, Type, Union, Tuple
import numpy as np
from torch import Tensor
from safetensors.torch import load_file
import torch
from torchvision.transforms.v2 import Resize
import torch.nn.functional as F
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.utils.normalization_contract import resolve_inference_normalization
from lingbotvla.models import build_processor
import time
import random

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed_everywhere(42)

BASE_MODEL_PATH = {
    'qwen3vl': os.environ.get(
        'QWEN3VL_DIR',
        os.environ.get('QWEN3VL_PATH') or str(workspace_path('models/Qwen3-VL-4B-Instruct')),
    ),
}

class PolicyPreprocessMixin:
    @staticmethod
    def _to_device_image_grid_thw(image_grid_thw, device):
        if image_grid_thw is None:
            return None
        return image_grid_thw.to(device=device, dtype=torch.long)

    @torch.no_grad
    def select_action(
        self, observation: dict[str, Tensor], use_bf16: bool = False
    ):
        self.eval()
        device = 'cuda'
        if use_bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        s1 = time.time()

        if len(observation['images'].shape) == 4:
            observation['images'] = observation['images'].unsqueeze(0)
            observation['img_masks'] = observation['img_masks'].unsqueeze(0)

        actions = self.model.sample_actions(
            observation['images'].to(dtype=dtype, device=device),
            observation['img_masks'].to(device=device),
            observation['lang_tokens'].unsqueeze(0).to(device=device),
            observation['lang_masks'].unsqueeze(0).to(device=device),
            observation['state'].unsqueeze(0).to(dtype=dtype, device=device),
            image_grid_thw=self._to_device_image_grid_thw(observation.get('image_grid_thw'), device),
        )
        delta_time = time.time() - s1
        print(f'sample_actions cost {delta_time} s')
        observation['actions'] = actions.squeeze(0).to(dtype=torch.float32, device='cpu')
        if use_bf16:
            observation['state'] = observation['state'].to(dtype=torch.float32)
        data = self.feature_transform.unapply(observation)
        return data

    @torch.no_grad
    def sample_actions_batch(
        self,
        observation: dict[str, Tensor],
        use_bf16: bool = False,
        use_compile: bool = False,
        capture_time: bool = False,
        sample_compile_fn: callable = None,
    ) -> Tensor:
        """Run one model forward for a batch of already-transformed observations.

        Single-sample inference in ``select_action`` builds a leading batch
        dimension just before calling ``sample_actions``. For Robocasa vector
        envs we already collate that dimension in the websocket server, so this
        method keeps the tensors batched and returns normalized action chunks
        with shape ``(B, chunk, joint_max_dim)``.
        """
        self.eval()
        device = "cuda"
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        s1 = time.time()

        images = observation["images"]
        img_masks = observation["img_masks"]
        lang_tokens = observation["lang_tokens"]
        lang_masks = observation["lang_masks"]
        state = observation["state"]
        image_grid_thw = observation.get("image_grid_thw", None)

        has_batch_dim = img_masks.ndim >= 2
        if not has_batch_dim:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)
        if lang_tokens.ndim == 1:
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if capture_time:
            with torch.inference_mode():
                for _ in range(3):
                    _ = sample_compile_fn(
                            images.to(dtype=dtype, device=device),
                            img_masks.to(device=device),
                            lang_tokens.to(device=device),
                            lang_masks.to(device=device),
                            state.to(dtype=dtype, device=device),
                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),
                    )
                torch.cuda.synchronize()

                iters = 5
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                for i in range(iters):
                    starts[i].record()
                    actions = sample_compile_fn(
                                    images.to(dtype=dtype, device=device),
                                    img_masks.to(device=device),
                                    lang_tokens.to(device=device),
                                    lang_masks.to(device=device),
                                    state.to(dtype=dtype, device=device),
                                    image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),
                    )
                    ends[i].record()
                torch.cuda.synchronize()
                gpu_times = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
                print(f"sample_actions avg time: {sum(gpu_times)/len(gpu_times):.4f} ms, min time: {min(gpu_times):.4f} ms, max time: {max(gpu_times):.4f} ms")
        else:
            actions = sample_compile_fn(
                            images.to(dtype=dtype, device=device),
                            img_masks.to(device=device),
                            lang_tokens.to(device=device),
                            lang_masks.to(device=device),
                            state.to(dtype=dtype, device=device),
                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),
            )

        delta_time = time.time() - s1
        print(f"sample_actions batch={actions.shape[0]} cost {delta_time} s")
        if use_bf16:
            observation["state"] = observation["state"].to(dtype=torch.float32)
        return actions.to(dtype=torch.float32, device="cpu")

class LingBotVlaV2InferencePolicy(PolicyPreprocessMixin, LingbotVlaV2Policy):
    pass # Only combine necessary functions


class LingbotVLAv2Server:
    '''
    policy wrapper to support action ensemble or chunk execution
    '''
    def __init__(
        self,
        path_to_pi_model="",
        robot_norm_path=None,
        adaptive_ensemble_alpha=0.1,
        action_ensemble_horizon=8,
        use_length=1,
        chunk_ret=False,
        use_bf16=True,
        use_fp32=False,
        use_compile=False,
    ) -> None:
        assert not (use_bf16 and use_fp32), 'Bfloat16 or Float32!!!'
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.use_length = use_length
        self.chunk_ret = chunk_ret
        self.robot_norm_path = robot_norm_path
        self._robot_norm_override = robot_norm_path

        self.task_description = None

        self.use_compile = use_compile
        apply_lingbot_qwen3_vl_patch()

        self.vla = self.load_vla(path_to_pi_model)
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16).cuda().eval()
        else:
            # fp32
            self.vla.model.float()
            self.vla = self.vla.cuda().eval()

        self.global_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        self._last_full_diagnostics = None
        self.checkpoint_path = str(workspace_path(path_to_pi_model))
        self.use_bf16 = use_bf16
        self.use_fp32 = use_fp32
        self.action_key: str= "action"

    def load_model_weights(self, path_to_pi_model, strict=True):
        all_safetensors = glob(os.path.join(path_to_pi_model, "*.safetensors"))
        merged_weights = {}

        for file_path in tqdm(all_safetensors):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_weights[key] = f.get_tensor(key)
        self.vla.load_state_dict(merged_weights, strict=strict)

    def merge_qwen_config(self, qwen_config):
        if hasattr(qwen_config, 'to_dict'):
            config_dict = qwen_config.to_dict()
        else:
            config_dict = qwen_config

        text_keys = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
            "max_position_embeddings",
            "hidden_act",
            "tie_word_embeddings",
            "tokenizer_path",
        }

        text_config = config_dict.get("text_config", {})
        for key in text_keys:
            if key in text_config:
                setattr(self.config, key, text_config[key])
                print(f"✅ Merged Qwen3-VL text: {key} = {text_config[key]}")
            elif key in config_dict:
                setattr(self.config, key, config_dict[key])
                print(f"✅ Merged LLM: {key} = {config_dict[key]}")

        if "vision_config" in config_dict:
            self.config.vision_config = qwen_config.vision_config
        else:
            print("⚠️ Warning: 'vision_config' not found in qwen_config!")

    def load_vla(self, path_to_pi_model) -> LingbotVlaV2Policy:
        path_to_pi_model = str(workspace_path(path_to_pi_model))
        self.checkpoint_path = path_to_pi_model
        print(f"loading model from: {path_to_pi_model}")
        
        # load training config
        training_config_path = Path(path_to_pi_model).parent.parent.parent/'lingbotvla_cli.yaml'
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        f.close()
        self.training_config = training_config

        # update model config according to training config
        training_model_config = training_config['model']
        training_model_config.update(training_config['train'])
        config = LingbotVLAV2Config(**training_model_config)
        for key, value in training_model_config.items():
            if not hasattr(config, key):
                setattr(config, key, value)

        # Set attention_implementation to 'eager' to speed up evaluation.
        config.attention_implementation = 'eager'
        
        # set base model according to training config
        training_base_model = training_config['model']['tokenizer_path']
        if 'qwen3' in training_base_model.lower() and 'vl' in training_base_model.lower():
            model_name = 'qwen3vl'
        else: 
            raise ValueError(f"Unsupported base model of {path_to_pi_model}")
        base_model_path = os.environ.get('QWEN3VL_DIR') or os.environ.get('QWEN3VL_PATH') or training_base_model or BASE_MODEL_PATH[model_name]
        if Path(base_model_path).is_absolute() or base_model_path.startswith(('models/', '.')) or workspace_path(base_model_path).exists():
            base_model_path = str(workspace_path(base_model_path))
        config.tokenizer_path = base_model_path
        self.model_name = model_name
        
        self.config = config
        qwen_config = AutoConfig.from_pretrained(base_model_path)
        self.merge_qwen_config(qwen_config)
        config = self.config

        if 'vocab_size' in training_config['model'] and training_config['model']['vocab_size'] != 0:
            config.vocab_size = training_config['model']['vocab_size']
        # config.num_steps = 4
        config.use_cache = True # is necessary in inference
        # load processors
        self.processor = build_processor(base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        data_config = SimpleNamespace(**training_config['data'])
        
        print('Initializing model ... ')

        self.vla = LingBotVlaV2InferencePolicy(config, eval=True)

        self.load_model_weights(path_to_pi_model, strict=True)
        
        self.vla.feature_transform = None
        self.data_config = data_config
        self.config = config
        self.vla.model._use_compile_predict_velocity = bool(self.use_compile)
        self.vla.model._compiled_predict_velocity = None
        self.sample_actions_fn = self.vla.model.sample_actions
        if self.use_compile:
            self.vla.model.qwenvl_with_expert = torch.compile(self.vla.model.qwenvl_with_expert)
            self.sample_actions_fn = torch.compile(self.vla.model.sample_actions)

        self.robot_norm_path = resolve_inference_normalization(
            path_to_pi_model, training_config, override=self._robot_norm_override,
        )
        print(f"Actual inference normalization: {self.robot_norm_path}")

        print('Model initialized ... ')

        return self.vla

    def reset(self, robo_name, path_to_pi_model = None) -> None:
        if path_to_pi_model is not None:
            self.vla = self.load_vla(path_to_pi_model)
            if self.use_bf16:
                self.vla = self.vla.to(torch.bfloat16).cuda().eval()
            else:
                #fp32
                self.vla.model.float()
                self.vla = self.vla.cuda().eval()

        self.global_step = 0
        self.last_action_chunk = None
        self.last_normalized_action_chunk = None
        self._last_full_diagnostics = None

        robot_config = str(workspace_path(f'configs/robot_configs/{robo_name}.yaml'))
        
        with open(robot_config, 'r') as f:
          self.robot_config = yaml.safe_load(f)

        feature_transform = FeatureTransform(robot_config, self.data_config, self.config, self.processor,\
                    chunk_size=self.config.chunk_size, norm_stats_path=self.robot_norm_path)
        # Load data processors
        self.vla.feature_transform = feature_transform
        self.action_key = feature_transform.org_features["actions"]
    def resize_image(self, observation):
        image_features  = self.vla.feature_transform.org_features['images']
        image_size = getattr(self.data_config, 'img_size', 256)
        resize = Resize((image_size, image_size))
        for image_feature in image_features:
            assert image_feature in observation
            assert len(observation[image_feature].shape)==3 and observation[image_feature].shape[-1] == 3
            image = torch.as_tensor(observation[image_feature]).permute(2, 0, 1).contiguous()
            image = image.to(dtype=torch.float32)
            observation[image_feature] = resize(image)

    def _unapply_batched_actions(self, transformed_observations, actions):
        action_chunk = {}
        for action in self.action_key:
            action_chunk[action] = []

        outputs = []
        for transformed, action in zip(transformed_observations, actions):
            single = dict(transformed)
            single['actions'] = action.to(dtype=torch.float32, device='cpu')
            if self.use_bf16 and 'state' in single:
                single['state'] = single['state'].to(dtype=torch.float32)
            data = self.vla.feature_transform.unapply(single)
            
            for action in self.action_key:
                # keep action keys after unapply
                value = data[action]
                if isinstance(value, torch.Tensor):
                    value = value.float().cpu().numpy()
                else:
                    value = np.asarray(value, dtype=np.float32)
                action_chunk[action].append(value)

        for action_key in action_chunk.keys():
            action_chunk[action_key] = np.stack(action_chunk[action_key], axis=0)

        return action_chunk

    def _prepare_model_input(self, observation):
        # not modify input observation
        observation = dict(observation)
        self.resize_image(observation)
        for k, v in list(observation.items()):
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)
        observation =  self.vla.feature_transform.apply(observation, policy_eval=True)
        if self.use_bf16:
            observation['state'] = observation['state'].to(torch.bfloat16)
        return observation

    @staticmethod
    def _pad_and_stack_tensors(values):
        shapes = [tuple(value.shape) for value in values]
        if len(set(shapes)) == 1:
            return torch.stack(values, dim=0)

        if all(value.ndim == 1 for value in values):
            max_len = max(value.shape[0] for value in values)
            fill_value = False if values[0].dtype == torch.bool else 0
            padded = []
            for value in values:
                out = torch.full(
                    (max_len,),
                    fill_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                out[: value.shape[0]] = value
                padded.append(out)
            return torch.stack(padded, dim=0)

        raise ValueError(f"Cannot batch tensors with different shapes: {shapes}")
    def _infer_batch(self, observations, return_normalized=False, return_diagnostics=False):
        if not isinstance(observations, (list, tuple)) or len(observations) == 0:
            raise ValueError("batch observation must be a non-empty list")
        started = time.monotonic()
        applied = [self._prepare_model_input(obs) for obs in observations] # bsize, dict{key }
        batch_observation = {}
        for key in applied[0].keys():
            values = [item[key] for item in applied]
            if isinstance(values[0], torch.Tensor):
                # Pad different 1d tokens to the max length: language, language mask
                batch_observation[key] = self._pad_and_stack_tensors(values)
            else:
                batch_observation[key] = values

        preprocess_ms = (time.monotonic() - started) * 1000
        diagnostic_arrays = {}
        if return_diagnostics:
            # Snapshot before sample_actions_batch changes the BF16 state dtype.
            for key in ('state', 'state_joint_mask', 'action_joint_mask', 'lang_tokens', 'lang_masks'):
                diagnostic_arrays[key] = batch_observation[key].detach().cpu().to(
                    dtype=torch.float32 if batch_observation[key].is_floating_point() else batch_observation[key].dtype
                ).numpy().copy()
            diagnostic_arrays['input_state'] = np.stack([obs['observation.state'] for obs in observations])
        sample_start = time.monotonic()
        actions = self.vla.sample_actions_batch(
            batch_observation,
            self.use_bf16,
            self.use_compile,
            capture_time=False,
            sample_compile_fn = self.sample_actions_fn,
        )
        
        sample_ms = (time.monotonic() - sample_start) * 1000
        unapply_start = time.monotonic()
        unnormalized_actions = self._unapply_batched_actions(applied, actions)
        unapply_ms = (time.monotonic() - unapply_start) * 1000
        if return_diagnostics:
            diagnostic_arrays['normalized_actions'] = actions.detach().cpu().numpy().copy()
            for key, value in unnormalized_actions.items():
                diagnostic_arrays[f'full_{key}'] = np.array(value, copy=True)
            return unnormalized_actions, actions, {
                'arrays': diagnostic_arrays, 'available': True,
                'timing': {'preprocess_ms': preprocess_ms, 'sample_ms': sample_ms, 'unapply_ms': unapply_ms},
            }
        if return_normalized:
            return unnormalized_actions, actions
        return unnormalized_actions
        
    def infer(self, observation, center_crop=True, return_normalized=False, return_diagnostics=False):
        """Generates an action with the VLA policy."""
        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if 'reset' in observation and observation['reset']:
            self.reset(robo_name=observation['robo_name'], path_to_pi_model=observation['path_to_pi_model'] if 'path_to_pi_model' in observation else None)
            return dict(action = None)

        is_batch = 'batch' in observation
        observations = observation['batch'] if is_batch else [observation]
        if not self.chunk_ret and self.use_length <= 0:
            raise ValueError(f"use_length must be > 0 when chunk_ret=False, got {self.use_length}")
        should_forward = (
            self.chunk_ret
            or self.last_action_chunk is None
            or (return_normalized and self.last_normalized_action_chunk is None)
            or self.global_step % self.use_length == 0
            or self.use_length == -1 
        )

        if should_forward:
            self._last_full_diagnostics = None
            if return_diagnostics:
                unnormalized_actions, normalized_actions, self._last_full_diagnostics = self._infer_batch(
                    observations, return_diagnostics=True,
                )
                if not return_normalized:
                    # Diagnostics must not populate a cache the ordinary call would leave empty.
                    normalized_actions = None
            elif return_normalized:
                unnormalized_actions, normalized_actions = self._infer_batch(
                    observations,
                    return_normalized=True,
                )
            else:
                unnormalized_actions = self._infer_batch(observations)
                normalized_actions = None
            if self.use_length > 0:
                for output_key in unnormalized_actions.keys():
                    assert self.use_length <= unnormalized_actions[output_key].shape[1]
                    unnormalized_actions[output_key] = unnormalized_actions[output_key][:, :self.use_length]
                if normalized_actions is not None:
                    assert self.use_length <= normalized_actions.shape[1]
                    normalized_actions = normalized_actions[:, :self.use_length]
            
            self.last_action_chunk = unnormalized_actions  # always (B, chunk, dim)
            self.last_normalized_action_chunk = normalized_actions

        if self.chunk_ret:
            action = self.last_action_chunk               # (B, chunk, dim)
            normalized_action = self.last_normalized_action_chunk
        else:
            step_idx = self.global_step % self.use_length
            action = {}
            for action_key in self.last_action_chunk.keys():
                action[action_key] = self.last_action_chunk[action_key][:, step_idx]
            normalized_action = (
                self.last_normalized_action_chunk[:, step_idx]
                if self.last_normalized_action_chunk is not None
                else None
            )

        if not is_batch:
            for action_key in action:
                action[action_key] = action[action_key][0]
            if normalized_action is not None:
                normalized_action = normalized_action[0]

        result = action
        if return_normalized:
            result = dict(result)
            result["_normalized_actions"] = normalized_action
        if return_diagnostics:
            result = dict(result)
            diagnostic = dict(self._last_full_diagnostics or {'arrays': {}, 'available': False, 'timing': {}})
            diagnostic['arrays'] = dict(diagnostic['arrays'])
            for key in self.action_key:
                diagnostic['arrays'][f'returned_{key}'] = np.array(result[key], copy=True)
            diagnostic.update(model_forward=should_forward, policy_step=self.global_step,
                              selected_step=None if self.chunk_ret else self.global_step % self.use_length)
            if not should_forward:
                diagnostic['timing'] = {}  # Cached prediction timings are not a new forward.
            result['_diagnostics'] = diagnostic

        self.global_step += 1
        
        return result

    def infer_with_diagnostics(self, observation):
        """Capture the existing forward; never run a second sampling pass."""
        return self.infer(observation, return_diagnostics=True)

    def evaluation_metadata(self):
        from .eval_logging import environment_info, json_value

        transform = self.vla.feature_transform
        return json_value({
            'checkpoint': self.checkpoint_path,
            'weight_files': [{'name': path.name, 'size': path.stat().st_size,
                              'mtime_ns': path.stat().st_mtime_ns}
                             for path in sorted(Path(self.checkpoint_path).glob('*.safetensors'))],
            'training_config': self.training_config,
            'model_config': self.config.to_dict(), 'robot_config': self.robot_config,
            'normalization_path': self.robot_norm_path,
            'normalization_types': transform.normalizer.norm_type if transform.normalizer else {},
            'features': {'states': transform.states, 'actions': transform.actions,
                         'joints': transform.feature_config.joints,
                         'joints_max_dim': transform.feature_config.joints_max_dim},
            'use_bf16': self.use_bf16, 'use_fp32': self.use_fp32,
            'actual_precision': 'bfloat16' if self.use_bf16 else 'float32',
            'gpu': {'name': torch.cuda.get_device_properties(0).name,
                    'total_memory': torch.cuda.get_device_properties(0).total_memory,
                    'cuda_version': torch.version.cuda},
            'use_compile': self.use_compile, 'use_length': self.use_length, 'chunk_ret': self.chunk_ret,
            'service_seed': 42, 'seed_policy': 'Seeded once at module import; reset does not reseed',
            'environment': environment_info(),
        })

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

import argparse
try:
    from .websocket_policy_server import WebsocketPolicyServer
except ImportError:
    from deploy.websocket_policy_server import WebsocketPolicyServer

def main():
    parser = argparse.ArgumentParser(description="Launch the Qwen3VL LingbotVlaV2 WebSocket policy server")

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("MODEL_DIR") or os.environ.get("MODEL_PATH") or str(workspace_path("models/lingbot-vla-v2-6b-robotwin")),
    )

    parser.add_argument(
        "--use_length",
        type=int,
        default=10,
        help="number of predicted actions to execute before replanning (model horizon is 50)"
    )

    parser.add_argument(
        "--chunk_ret",
        type=str2bool,
        default=True,
        help="return the selected action chunk in one response"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8006,
        help="WebSocket server port"
    )

    parser.add_argument(
        "--use_bf16",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--use_fp32",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--use_compile",
        type=str2bool,
        default=True,
    )

    parser.add_argument('--eval_run_dir', type=str, default=None)
    parser.add_argument('--eval_slot', type=int, default=0)
    args = parser.parse_args()

    model = LingbotVLAv2Server(
        args.model_path,
        use_length=args.use_length,
        chunk_ret=args.chunk_ret,
        use_bf16=args.use_bf16,
        use_fp32=args.use_fp32,
        use_compile=args.use_compile,
    )
    model_server = WebsocketPolicyServer(model, port=args.port,
                                        eval_run_dir=args.eval_run_dir, eval_slot=args.eval_slot)
    model_server.serve_forever()


if __name__ == "__main__":
    main()
