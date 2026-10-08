# Simulator imports need the workspace bootstrap below.
# ruff: noqa: E402
import ast
import os
import sys
from pathlib import Path


WORKSPACE = Path(
    os.path.expandvars(os.environ.get("WORKSPACE") or str(Path(__file__).resolve().parents[2]))
).expanduser()
if not WORKSPACE.is_absolute():
    WORKSPACE = Path(__file__).resolve().parents[2] / WORKSPACE
WORKSPACE = WORKSPACE.resolve()
robotwin_dir = Path(
    os.path.expandvars(os.environ.get("ROBOTWIN_DIR") or os.environ.get("EVAL_WORKDIR") or "RoboTwin")
).expanduser()
if not robotwin_dir.is_absolute():
    robotwin_dir = WORKSPACE / robotwin_dir
os.chdir(robotwin_dir)
sys.path.append(str(robotwin_dir / "script"))

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
import argparse
import importlib
from datetime import datetime

import numpy as np
import yaml
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
from script.deploy.eval_diagnostics import interrupt_on_signal
from script.deploy.robotwin_evaluation import evaluate_task


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise SystemExit(f"Task class missing: {task_name}") from exc
    return env_class()


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = Path(CONFIGS_PATH) / "_camera_config.yml"

    assert camera_config_path.is_file(), f"task config file is missing: {camera_config_path}"

    with camera_config_path.open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args.get("ckpt_setting") or usr_args.get("train_config_name", "default")
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args.get("instruction_type")
    save_dir = None
    video_save_dir = None
    video_size = None
    video_fps = str(usr_args.get("video_fps", 10))

    task_config_path = Path(CONFIGS_PATH) / f"{task_config}.yml"
    with task_config_path.open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    instruction_type = instruction_type or args["eval_instruction"]
    args["eval_instruction"] = instruction_type

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"

    with embodiment_config_path.open("r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with (Path(CONFIGS_PATH) / "_camera_config.yml").open("r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if usr_args.get("output_dir"):
        output_path = Path(os.path.expandvars(usr_args["output_dir"])).expanduser()
        save_dir = (output_path if output_path.is_absolute() else WORKSPACE / output_path) / task_name
    else:
        output_root = Path(os.path.expandvars(os.environ.get("OUTPUT_DIR") or "outputs")).expanduser()
        if not output_root.is_absolute():
            output_root = WORKSPACE / output_root
        save_dir = (
            output_root
            / "eval_outputs"
            / f"{policy_name}_{ckpt_setting}_{task_config}_{current_time}"
            / "eval_results"
            / task_name
        )
        if save_dir.exists():
            raise ValueError(f"Refusing to reuse eval results: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = save_dir / "attempts" / f"attempt_{int(usr_args.get('attempt', 1))}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    usr_args["_attempt_dir"] = str(attempt_dir)
    usr_args.setdefault("run_dir", str(save_dir.parent.parent))
    if usr_args.get("eval_trace", "full") not in ("full", "off"):
        raise ValueError("eval_trace must be full or off")

    # 命令行 --eval_video_log 优先于 YAML 配置
    if "eval_video_log" in usr_args:
        args["eval_video_log"] = usr_args["eval_video_log"]

    if args["eval_video_log"]:
        video_save_dir = attempt_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + ", "
        + str(args["camera"]["collect_head_camera"])
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + ", "
        + str(args["camera"]["collect_wrist_camera"])
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    from script.deploy.websocket_client_policy import WebsocketClientPolicy

    model = WebsocketClientPolicy(port=usr_args["port"])

    def instruction_factory(episode_info):
        results = generate_episode_descriptions(
            args["task_name"], [episode_info["info"]], int(usr_args.get("num_episodes", 100))
        )
        return np.random.choice(results[0][instruction_type])

    evaluate_task(
        TASK_ENV, args, model, usr_args, instruction_factory, UnStableError, video_size=video_size, video_fps=video_fps
    )


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = Path(os.path.expandvars(args.config)).expanduser()
    if not config_path.is_absolute():
        config_path = WORKSPACE / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        if len(pairs) % 2:
            raise ValueError("Overrides must be key/value pairs")
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    usr_args = parse_args_and_config()

    with interrupt_on_signal():
        main(usr_args)
