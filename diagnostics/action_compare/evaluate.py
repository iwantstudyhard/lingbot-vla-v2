"""Run the existing evaluator with isolated, opt-in observation/action logging.

Both policies see the identical observation and sampling seed on every request.
Only the selected primary policy controls the robot. Simulator observations are
diagnostic evidence only, NEVER training samples. No production files are edited.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import checked_actions, json_write, observation_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", choices=["official", "ours"], required=True)
    parser.add_argument("--ours-port", type=int, required=True)
    parser.add_argument("--official-port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="lift_pot")
    parser.add_argument("--use-length", type=int, default=20)
    parser.add_argument("--max-actions", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-probes", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "observations").mkdir()
    (output / "predictions").mkdir()
    robotwin = Path(os.environ["ROBOTWIN_DIR"]).resolve()
    os.chdir(robotwin)
    sys.path.insert(0, str(robotwin))
    sys.path.insert(0, str(robotwin / "script"))
    # Reuse the repository's helper directly, not a stale RoboTwin-side copy.
    from deploy.websocket_client_policy import WebsocketClientPolicy as Client
    import types
    helper = types.ModuleType("script.deploy.websocket_client_policy")
    helper.WebsocketClientPolicy = Client
    # Populate package parents before aliasing; these are existing RoboTwin dirs.
    import script
    import script.deploy
    sys.modules["script.deploy.websocket_client_policy"] = helper

    clients = {"ours": Client(port=args.ours_port),
               "official": Client(port=args.official_port)}
    metadata = {key: client.get_server_metadata() for key, client in clients.items()}
    for label, meta in metadata.items():
        if meta.get("diagnostic_token") != args.token or meta.get("label") != label:
            raise RuntimeError(f"Wrong/stale server on {label} port; refusing evaluation")
        if meta.get("horizon") != 50:
            raise ValueError("This comparison requires both models' action horizon=50")
    json_write(output / "servers.json", metadata)

    spec = importlib.util.spec_from_file_location(
        "diagnosed_robotwin_eval", ROOT / "experiment/robotwin/eval_policy_client_lingbotvla.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    env_ref = {}
    state = {"call": 0, "chunk": -1, "executing": False}
    action_log = (output / "executed.jsonl").open("w", encoding="utf-8", buffering=1)
    request_log = (output / "requests.jsonl").open("w", encoding="utf-8", buffering=1)
    original_class_decorator = evaluator.class_decorator

    def real_qpos(env):
        # The observation.state uses drive TARGETS, not actual qpos. Log both.
        return np.asarray(env.robot.get_left_arm_real_jointState() +
                          env.robot.get_right_arm_real_jointState(), dtype=float)

    def attach(task_name):
        env = original_class_decorator(task_name)
        env_ref["env"] = env
        original_setup = env.setup_demo
        original_play = env.play_once
        original_action = env.take_action

        def fatal_guard(function, *pos, **kw):
            try:
                return function(*pos, **kw)
            except evaluator.UnStableError:
                raise  # standard invalid-scene retry remains intact
            except Exception:
                traceback.print_exc()
                raise SystemExit("Diagnostic abort: simulator failed; no infinite retries")

        def setup(*pos, **kw):
            state["executing"] = False
            state["scene_seed"] = int(kw["seed"])
            return fatal_guard(original_setup, *pos, **kw)

        env.setup_demo = setup
        env.play_once = lambda *pos, **kw: fatal_guard(original_play, *pos, **kw)
        wrapped = set()
        current_topp = {}

        def instrument_topp():
            for side in ("left", "right"):
                planner = getattr(env.robot, f"{side}_mplib_planner")
                if id(planner) in wrapped:
                    continue
                original_topp = planner.TOPP

                def topp(*pos, _fn=original_topp, _side=side, **kw):
                    try:
                        result = _fn(*pos, **kw)
                        current_topp[_side] = {"points": int(len(result[1])), "error": None}
                        return result
                    except Exception as exc:
                        current_topp[_side] = {"points": 0, "error": repr(exc)}
                        raise  # preserve the original evaluator's fallback behavior

                planner.TOPP = topp
                wrapped.add(id(planner))

        def take_action(action, *pos, **kw):
            if not state["executing"]:
                return original_action(action, *pos, **kw)
            before_count = env.take_action_cnt
            if before_count >= env.step_lim or env.eval_success:
                return original_action(action, *pos, **kw)
            instrument_topp()
            current_topp.clear()
            before = real_qpos(env)
            start = time.monotonic()
            result = original_action(action, *pos, **kw)
            record = {
                "step": before_count, "chunk": state["chunk"],
                "action": np.asarray(action, dtype=float).tolist(),
                "actual_before": before.tolist(), "actual_after": real_qpos(env).tolist(),
                "elapsed_wall_s": time.monotonic() - start,
                "topp": dict(current_topp), "success": bool(env.eval_success),
            }
            action_log.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"[action-diagnostic] {args.primary} step {env.take_action_cnt}/{env.step_lim} "
                  f"chunk {state['chunk']}", flush=True)
            return result

        env.take_action = take_action
        return env

    evaluator.class_decorator = attach

    class PairedClient:
        def __init__(self, *pos, **kw):
            pass

        def infer(self, obs):
            if obs.get("reset"):
                # Never load the primary model's checkpoint into the shadow server.
                if obs.get("path_to_pi_model"):
                    raise ValueError("Checkpoint switching is disabled in paired diagnosis")
                responses = {label: client.infer(dict(obs)) for label, client in clients.items()}
                env = env_ref["env"]
                state["original_limit"] = env.step_lim
                if args.max_actions:
                    env.step_lim = min(env.step_lim, args.max_actions)
                state["executing"] = True
                return responses[args.primary]
            call = state["call"]
            state["chunk"] = call
            obs_hash = observation_hash(obs)
            # Same noise realization for both models for this exact observation.
            seed = 42 + call
            request = dict(obs, _diagnostic_seed=seed)
            responses = {label: client.infer(dict(request)) for label, client in clients.items()}
            raw = {label: checked_actions(reply["action"]) for label, reply in responses.items()}
            names = responses["ours"]["diagnostic_joint_names"]
            if names != responses["official"]["diagnostic_joint_names"]:
                raise ValueError("Models have different valid normalized joint definitions")
            normalized = {label: np.asarray(reply["diagnostic_normalized_valid"], dtype=np.float32)
                          for label, reply in responses.items()}
            for label, reply in responses.items():
                if int(reply["diagnostic_seed"]) != seed or not np.isfinite(normalized[label]).all():
                    raise ValueError("Invalid paired inference telemetry")
            np.savez_compressed(output / "observations" / f"{call:04d}.npz", **obs)
            np.savez_compressed(output / "predictions" / f"{call:04d}.npz",
                                ours=raw["ours"], official=raw["official"],
                                ours_normalized=normalized["ours"],
                                official_normalized=normalized["official"],
                                normalized_joint_names=np.asarray(names),
                                state=np.asarray(obs["observation.state"]))
            if call == 0:
                # Additional forward passes on the SAME saved observation, no robot actions.
                probes = {label: [raw[label]] for label in clients}
                for probe in range(1, args.noise_probes):
                    for label, client in clients.items():
                        response = client.infer(dict(obs, _diagnostic_seed=42 + probe))
                        probes[label].append(checked_actions(response["action"]))
                np.savez_compressed(output / "noise_probes.npz",
                                    **{label: np.stack(values) for label, values in probes.items()},
                                    seeds=np.arange(42, 42 + args.noise_probes))
            request_log.write(json.dumps({
                "call": call, "action_start": env_ref["env"].take_action_cnt,
                "scene_seed": state["scene_seed"], "sampling_seed": seed,
                "observation_hash": obs_hash, "instruction": obs["task"],
                "primary": args.primary, "use_length": args.use_length,
            }) + "\n")
            state["call"] += 1
            result = dict(responses[args.primary])
            result["action"] = raw[args.primary][:args.use_length].copy()
            return result

    helper.WebsocketClientPolicy = PairedClient
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = {
        "task_name": args.task, "task_config": "demo_clean", "num_episodes": 1,
        "train_config_name": "diagnostic", "seed": args.seed, "policy_name": "ACT",
        "port": args.ours_port, "robo_name": "robotwin", "video_fps": 10,
        "eval_video_log": True, "output_dir": str(output / "eval_results"),
    }
    started = time.monotonic()
    try:
        evaluator.main(config)
        env = env_ref["env"]
        json_write(output / "completion.json", {
            "primary": args.primary, "task": args.task, "scene_seed": state["scene_seed"],
            "success": bool(env.eval_success), "actions": env.take_action_cnt,
            "diagnostic_truncated": bool(not env.eval_success and args.max_actions and
                                         args.max_actions < state["original_limit"] and
                                         env.take_action_cnt >= args.max_actions),
            "original_action_limit": state["original_limit"],
            "wall_s": time.monotonic() - started,
        })
    finally:
        action_log.close()
        request_log.close()
        env = env_ref.get("env")
        if env is not None:
            state["executing"] = False
        for client in clients.values():
            client._ws.close()


if __name__ == "__main__":
    main()
