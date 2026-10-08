"""Opt-in server: original inference + full chunk and valid-dimension telemetry.

No edits to the deployed policy. The client executes a prefix of the returned
50-action chunk, just as the normal use_length option does. Re-seeding each
request is diagnostic-only, to couple sampling noise across the two models.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", choices=["official", "ours"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, set_seed_everywhere
    from deploy.websocket_policy_server import WebsocketPolicyServer
    from lingbotvla.utils.normalization_contract import semantic_hash

    class DiagnosticPolicy(LingbotVLAv2Server):
        def _unapply_batched_actions(self, applied, actions):
            result = super()._unapply_batched_actions(applied, actions)
            mask = applied[0]["action_joint_mask"].bool().cpu()
            self.diag_normalized = actions[0].cpu()[:, mask].float().numpy()
            self.diag_names = []
            transform = self.vla.feature_transform
            for joint in transform.feature_config.joints:
                key = f"action.{joint}"
                if key in transform.actions:
                    count = len(transform.normalizer.norm_stats[key]["mean"])
                    self.diag_names.extend(f"{key}[{i}]" for i in range(count))
            if len(self.diag_names) != self.diag_normalized.shape[1]:
                raise ValueError("Normalized valid-dimension telemetry mismatch")
            return result

        def infer(self, observation, **kwargs):
            obs = dict(observation)
            if not obs.get("reset"):
                seed = int(obs.pop("_diagnostic_seed"))
                set_seed_everywhere(seed)
            # Base policy is unchanged; only the isolated wrapper requests telemetry.
            result = super().infer(obs, return_normalized=True)
            result.pop("_normalized_actions", None)  # original tensor isn't msgpack-safe
            if not obs.get("reset"):
                result["diagnostic_normalized_valid"] = self.diag_normalized
                result["diagnostic_joint_names"] = self.diag_names
                result["diagnostic_seed"] = seed
            return result

    policy = DiagnosticPolicy(args.model, use_length=50, chunk_ret=True,
                              use_bf16=False, use_fp32=True, use_compile=False)
    stats_path = Path(policy.robot_norm_path).resolve()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    metadata = {
        "diagnostic_token": args.token, "label": args.label,
        "model": str(Path(args.model).resolve()),
        "normalization": str(stats_path), "norm_count": stats.get("count"),
        "norm_semantic_hash": semantic_hash(stats),
        "horizon": policy.config.chunk_size,
        "denoising_steps": getattr(policy.config, "num_steps", None),
        "image_size": getattr(policy.data_config, "img_size", 256),
        "precision": "FP32", "compile": False,
        "sampling": "same per-request seed in both models; diagnostic-only",
    }
    Path(args.manifest).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
