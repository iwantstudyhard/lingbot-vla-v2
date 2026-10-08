"""Linux server launcher: paired official/ours diagnostic on GPUs 2 and 3.

Run from the inference conda environment. Simulator runs in a separate interpreter.
Only this launcher's own process groups are stopped. Training is untouched.
"""
import argparse
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import json_write


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--robotwin", type=Path, default=ROOT / "RoboTwin")
    parser.add_argument("--qwen", type=Path, default=None)
    parser.add_argument("--gpus", default="2,3", help="Physical CUDA device IDs: ours,official")
    parser.add_argument("--port", type=int, default=19540)
    parser.add_argument("--task", default="lift_pot")
    parser.add_argument("--use-length", type=int, default=20)
    parser.add_argument("--max-actions", type=int, default=160, help="Diagnostic cap, 0=full episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-probes", type=int, default=4)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--eval-timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eval_outputs/action_compare")
    return parser.parse_args()


def main():
    args = arguments()
    if os.name != "posix":
        raise SystemExit("Run this launcher on the Linux GPU server; plotting/tests work on CPU.")
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2 or not all(x.isdigit() for x in gpus):
        raise SystemExit("--gpus must contain two distinct physical GPU indices")
    if not 1 <= args.use_length <= 50 or args.max_actions < 0 or args.seed < 0:
        raise SystemExit("Require use-length in 1..50, max-actions>=0, seed>=0")
    if not 2 <= args.noise_probes <= 10:
        raise SystemExit("--noise-probes must be in 2..10")
    if args.eval_timeout < 1 or args.ready_timeout < 1 or not 1024 <= args.port < 65535:
        raise SystemExit("Invalid timeout or port range")
    for label in ("ours", "official"):
        model = getattr(args, label).resolve()
        if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            raise SystemExit(f"Missing HF model files: {model}")
        if not (model.parent.parent.parent / "lingbotvla_cli.yaml").is_file():
            raise SystemExit(f"Missing saved training config for {model}")
        setattr(args, label, model)
    if not args.sim_python.is_file():
        raise SystemExit(f"Missing simulator Python: {args.sim_python}")
    if not (args.robotwin / "env_cfg/task_config/demo_clean.yml").is_file():
        raise SystemExit("Missing RoboTwin clean task config")
    if not (args.robotwin / "script/deploy").is_dir():
        raise SystemExit("Missing RoboTwin script/deploy; run the normal evaluator setup first")
    for name in ("numpy", "matplotlib"):
        if importlib.util.find_spec(name) is None:
            raise SystemExit(f"Missing {name} in inference environment")
    qwen = args.qwen or Path(os.environ.get("QWEN3VL_DIR") or os.environ.get("QWEN3VL_PATH") or
                            str(ROOT / "models/Qwen3-VL-4B-Instruct/snapshots/master"))
    if not (qwen / "config.json").is_file():
        raise SystemExit(f"Missing Qwen config.json: {qwen}; pass --qwen")
    for port in (args.port, args.port + 1):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                raise SystemExit(f"Port {port} occupied; choose another --port, no process was killed")
    token = uuid.uuid4().hex
    run = args.output.resolve() / (datetime.now().strftime("%Y%m%d_%H%M%S_") + token[:8])
    run.mkdir(parents=True, exist_ok=False)
    json_write(run / "launch.json", {**{k: str(v) if isinstance(v, Path) else v
                                       for k, v in vars(args).items()}, "diagnostic_token": token})
    env = os.environ.copy()
    env.update(WORKSPACE=str(ROOT), ROBOTWIN_DIR=str(args.robotwin.resolve()),
               QWEN3VL_DIR=str(qwen.resolve()), QWEN3VL_PATH=str(qwen.resolve()),
               PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    processes, streams = [], []
    servers = {}

    def spawn(command, logfile, gpu):
        stream = logfile.open("w", encoding="utf-8")
        streams.append(stream)
        child_env = dict(env, CUDA_VISIBLE_DEVICES=gpu)
        child = subprocess.Popen(command, env=child_env, cwd=ROOT, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(child)
        return child

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    print(f"Run directory: {run}", flush=True)
    try:
        for i, label in enumerate(("ours", "official")):
            servers[label] = spawn([
                sys.executable, "-u", str(HERE / "server.py"), "--model", str(getattr(args, label)),
                "--label", label, "--port", str(args.port + i), "--token", token,
                "--manifest", str(run / f"{label}_server.json")], run / f"{label}_server.log", gpus[i])
            print(f"[server] {label} physical GPU {gpus[i]}, port {args.port + i}", flush=True)
        deadline = time.monotonic() + args.ready_timeout
        pending = set(servers)
        last_notice = 0
        while pending:
            for label, child in servers.items():
                if child.poll() is not None:
                    raise RuntimeError(f"{label} server exited; see {run / (label + '_server.log')}")
            for label in list(pending):
                port = args.port + (label == "official")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                        if response.status == 200:
                            pending.remove(label)
                            print(f"[ready] {label}", flush=True)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError("Model startup timed out; inspect server logs")
            if time.monotonic() - last_notice > 15:
                print(f"[loading] pending: {sorted(pending)}", flush=True)
                last_notice = time.monotonic()
            if pending:
                time.sleep(1)
        for index, primary in enumerate(("ours", "official"), 1):
            gpu = gpus[0 if primary == "ours" else 1]
            log = run / f"{primary}_eval.log"
            child = spawn([
                str(args.sim_python.resolve()), "-u", str(HERE / "evaluate.py"),
                "--primary", primary, "--ours-port", str(args.port),
                "--official-port", str(args.port + 1), "--token", token,
                "--output", str(run / primary), "--task", args.task,
                "--use-length", str(args.use_length), "--max-actions", str(args.max_actions),
                "--seed", str(args.seed), "--noise-probes", str(args.noise_probes)], log, gpu)
            print(f"[eval {index}/2] {primary} controls robot on physical GPU {gpu}; log: {log}", flush=True)
            start, last_notice = time.monotonic(), 0
            while child.poll() is None:
                for label, server in servers.items():
                    if server.poll() is not None:
                        raise RuntimeError(f"{label} server exited during evaluation")
                if time.monotonic() - start > args.eval_timeout:
                    raise TimeoutError(f"{primary} evaluation timed out; inspect {log}")
                if time.monotonic() - last_notice > 10:
                    if log.is_file():
                        with log.open("rb") as stream:
                            stream.seek(max(0, log.stat().st_size - 12000))
                            lines = stream.read().decode("utf-8", errors="replace").splitlines()
                        progress = next((line for line in reversed(lines)
                                         if "[action-diagnostic]" in line), None)
                    else:
                        progress = None
                    print(progress or f"[eval {index}/2] simulator setup, elapsed {time.monotonic()-start:.0f}s; see {log}", flush=True)
                    last_notice = time.monotonic()
                time.sleep(1)
            if child.returncode != 0:
                raise RuntimeError(f"{primary} evaluation failed (exit {child.returncode}); inspect {log}")
            print(f"[done {index}/2] {primary}", flush=True)
        from plot import render
        render(run)
        print(f"DONE: {run / 'analysis'}\nThis is diagnosis, NOT a benchmark success rate.", flush=True)
    finally:
        # Only process groups created above, never pkill or global CUDA cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for child in reversed(processes):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in processes:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        for stream in streams:
            stream.close()
        print(f"Owned processes cleaned up. Evidence retained: {run}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
