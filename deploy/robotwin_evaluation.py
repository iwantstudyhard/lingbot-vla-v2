"""RoboTwin evaluation lifecycle with passive tracing and isolated retries."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .eval_diagnostics import EpisodeTrace, InvalidActionError, execute_chunk, joint_metadata
from .eval_logging import (
    EventWriter,
    environment_info,
    exception_fields,
    prediction_path,
    read_task_attempts,
    summarize_run,
)


def start_video(env, directory: Path, video_size: str, video_fps: str):
    if env.eval_video_path is None:
        return None
    env.eval_video_path = str(directory)
    path = directory / f"episode{env.test_num}.mp4"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            video_fps,
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    env._set_eval_video_ffmpeg(process)
    return path


def infer_request(model, observation: dict, trace: EpisodeTrace, run: Path, request_id: str) -> dict:
    trace.request_id = request_id
    trace.info["request_id"] = request_id
    trace.stage = "reset" if observation.get("reset") else "infer"
    context = {**trace.context, "request_id": request_id}
    if trace.enabled or (observation.get("reset") and model.get_server_metadata().get("eval_trace_schema") == 1):
        observation["_eval_context"] = context
    trace.log(
        trace.inference,
        "request_start",
        input_state=observation.get("observation.state"),
        task_instruction=observation.get("task"),
        reset=bool(observation.get("reset")),
        executed_actions=trace.info["executed_actions"],
    )
    started = time.monotonic()
    result = model.infer(observation)
    rtt_ms = (time.monotonic() - started) * 1000
    diagnostic = result.get("_eval_trace", {})
    if not observation.get("reset"):
        trace.metrics["request_count"] += 1
        trace.metrics["rtt_ms_total"] += rtt_ms
        trace.metrics["infer_ms_total"] += result.get("server_timing", {}).get("infer_ms", 0)
        trace.metrics["prediction_write_ms_total"] += result.get("server_timing", {}).get("logging_ms", 0)
    if trace.enabled:
        if any(diagnostic.get(key) != value for key, value in context.items()):
            raise RuntimeError(f"Inference trace context mismatch: {context}")
        if not observation.get("reset"):
            expected = prediction_path(run, context)
            if diagnostic.get("artifact") != expected.relative_to(run).as_posix() or not expected.is_file():
                raise RuntimeError(f"Missing or mismatched shared prediction artifact: {expected}")
            if not diagnostic.get("available"):
                raise RuntimeError("Full diagnostics unavailable for this prediction")
    trace.log(
        trace.inference,
        "request_end",
        rtt_ms=rtt_ms,
        server_timing=result.get("server_timing"),
        diagnostic=diagnostic,
        returned_action=result.get("action"),
    )
    if diagnostic.get("model_metadata"):
        trace.update(model_metadata=diagnostic["model_metadata"])
    return result


def run_episode(
    env,
    args: dict,
    model,
    usr_args: dict,
    trace: EpisodeTrace,
    instruction_factory,
    episode_info: dict,
    video_size,
    video_fps,
) -> str:
    run = Path(usr_args["run_dir"]).resolve()
    video = None
    video_started = False
    reason = "exception"
    pending_error = None
    try:
        args["eval_video_save_dir"] = trace.directory
        env.setup_demo(now_ep_num=trace.context["episode_id"], seed=trace.context["seed"], is_test=True, **args)
        trace.stage = "instruction"
        instruction = instruction_factory(episode_info)
        env.set_instruction(instruction=instruction)
        trace.update(
            instruction=instruction,
            initialization=getattr(env, "info", episode_info),
            expert_initialization=episode_info,
            step_limit=env.step_lim,
            robot=joint_metadata(env),
            eval_trace="full" if trace.enabled else "off",
        )
        trace.stage = "video_start"
        video = start_video(env, trace.directory, video_size, video_fps)
        video_started = video is not None
        infer_request(
            model,
            {"reset": True, "robo_name": usr_args["robo_name"], "path_to_pi_model": usr_args.get("new_ckpt_path")},
            trace,
            run,
            "reset",
        )
        request = 0
        with trace.planning(env):
            while env.take_action_cnt < env.step_lim and not env.eval_success:
                trace.stage = "observation"
                observation = env.get_obs()
                formatted = {
                    "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
                    "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                    "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                    "observation.state": observation["joint_action"]["vector"],
                    "task": env.get_instruction(),
                }
                result = infer_request(model, formatted, trace, run, f"{request:06d}")
                execute_chunk(env, result["action"], trace)
                print(f"infer time {result.get('server_timing')}")
                request += 1
        reason = "success" if env.eval_success else "step_limit"
    except BaseException as exc:
        pending_error = exc
        reason = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "invalid_action"
            if isinstance(exc, InvalidActionError)
            else "exception"
        )
    finally:
        try:
            if video_started:
                env._del_eval_video_ffmpeg()
                if video.exists():
                    destination = video.with_name(f"episode{env.test_num}_{reason}.mp4")
                    video.rename(destination)
                    trace.info["video"] = destination.relative_to(run).as_posix()
                    print(f"Video saved: {destination}")
        except BaseException as exc:
            trace.log(trace.execution, "cleanup_error", **exception_fields(exc, "video_close"))
            if pending_error is None:
                pending_error, reason = exc, "exception"
                trace.stage = "video_close"
        try:
            env.close_env(clear_cache=((env.test_num + 2) % args["clear_cache_freq"] == 0))
            if getattr(env, "render_freq", 0):
                env.viewer.close()
        except BaseException as exc:
            trace.log(trace.execution, "cleanup_error", **exception_fields(exc, "env_close"))
            if pending_error is None:
                pending_error, reason = exc, "exception"
                trace.stage = "env_close"
        trace.finish(reason, env, pending_error)
    if pending_error is not None:
        raise pending_error
    return reason


def evaluate_task(
    env, args: dict, model, usr_args: dict, instruction_factory, unstable_error, video_size=None, video_fps="10"
):
    task_dir = Path(usr_args["_task_dir"])
    run = Path(usr_args["run_dir"]).resolve()
    context = {
        "run_id": run.name,
        "task": args["task_name"],
        "attempt": int(usr_args.get("attempt", 1)),
        "slot": int(usr_args.get("slot", 0)),
    }
    enabled = usr_args.get("eval_trace", "full") == "full"
    if any(item["attempt"] == context["attempt"] for item in read_task_attempts(task_dir)):
        raise FileExistsError(f"Refusing to reuse attempt {context['attempt']}: {task_dir}")
    EventWriter(task_dir / "task_config.jsonl", context).write(
        "task_config",
        requested_episodes=int(usr_args.get("num_episodes", 100)),
        user_args=usr_args,
        resolved_config=args,
        environment=environment_info(),
        server_metadata=model.get_server_metadata(),
    )
    requested = int(usr_args.get("num_episodes", 100))
    if requested < 1:
        raise ValueError("num_episodes must be positive")
    seeds = EventWriter(task_dir / "seed_checks.jsonl", context)
    env.suc, env.test_num = 0, 0
    seed = 100000 * (1 + int(usr_args["seed"]))
    args["eval_mode"] = True
    status = "error"
    error = {}
    try:
        if enabled and model.get_server_metadata().get("eval_trace_schema") != 1:
            raise RuntimeError("Full tracing requires a policy server launched with --eval_run_dir for this run")
        while env.test_num < requested:
            render_freq = args["render_freq"]
            args["render_freq"] = 0
            candidate = {**context, "episode_id": env.test_num, "seed": seed}
            seeds.write("seed_check_start", **candidate)
            expert_exception = None
            try:
                env.setup_demo(now_ep_num=env.test_num, seed=seed, is_test=True, **args)
                episode_info = env.play_once()
                env.close_env()
                accepted = bool(env.plan_success and env.check_success())
            except (unstable_error, Exception) as exc:
                expert_exception = exc
                print(f"Expert seed {seed} rejected: {type(exc).__name__}: {exc}")
                env.close_env()
                accepted = False
            finally:
                args["render_freq"] = render_freq
            if expert_exception is not None:
                seeds.write(
                    "seed_check_error",
                    **candidate,
                    accepted=False,
                    **exception_fields(expert_exception, "expert_check"),
                )
            else:
                seeds.write(
                    "seed_check_end",
                    **candidate,
                    accepted=accepted,
                    plan_success=env.plan_success,
                    initialization=episode_info,
                )
            if not accepted:
                seed += 1
                continue
            trace = EpisodeTrace(task_dir, candidate, enabled)
            reason = run_episode(
                env, args, model, usr_args, trace, instruction_factory, episode_info, video_size, video_fps
            )
            env.suc += reason == "success"
            env.test_num += 1
            print("Success!" if reason == "success" else "Fail!")
            print(
                f"{args['task_name']} | {args['policy_name']} | {args['task_config']} | {args['ckpt_setting']}\n"
                f"Success rate: {env.suc}/{env.test_num} => {round(env.suc / env.test_num * 100, 1)}%, current seed: {seed}\n"
            )
            seed += 1
        status = "complete"
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error"
        error = exception_fields(exc, "task")
        raise
    finally:
        EventWriter(task_dir / "attempt_results.jsonl", context).write("attempt_end", status=status, **error)
        # The launcher owns the run summary, avoiding concurrent task-level rewrites.
        if not usr_args.get("launcher_owned", False):
            summarize_run(run, status)
    return seed, env.suc
