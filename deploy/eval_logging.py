"""Versioned, incremental evaluation artifacts; usable without torch or the simulator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    """Convert numerical/configuration values to strict JSON (nonfinite -> null)."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return json_value(value.detach().float().cpu().numpy())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return repr(value)


def record(context: dict | None = None, **fields: Any) -> dict:
    return {
        **json_value(context or {}),
        **json_value(fields),
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp(),
    }


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(json_value(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_npz(path: Path, arrays: dict) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prediction: {path}")
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        converted = {key: np.asarray(value) for key, value in arrays.items()}
        if any(value.dtype.hasobject for value in converted.values()):
            raise ValueError("Prediction artifacts must not contain object/pickle arrays")
        with os.fdopen(handle, "wb") as stream:
            handle = None
            np.savez_compressed(stream, **converted)
        os.replace(temporary, path)
    finally:
        if handle is not None:
            os.close(handle)
        Path(temporary).unlink(missing_ok=True)


class EventWriter:
    """Each file has one process owner; flush every event to survive abrupt exits."""

    def __init__(self, path: Path, context: dict | None = None) -> None:
        self.path = Path(path)
        self.context = context or {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> dict:
        item = record(self.context, event=event, **fields)
        with self.path.open("ab+") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    # A crashed attempt may leave a partial line in the shared task log.
                    stream.write(b"\n")
            stream.write((json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
        return item


def read_events(path: Path, warnings: list | None = None) -> list[dict]:
    if not path.exists():
        return []
    items = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Event must be an object")
            items.append(value)
        except (ValueError, TypeError) as exc:
            if warnings is not None:
                warnings.append(f"{path}:{line_number}: {exc}")
    return items


def read_json(path: Path, default: Any = None, warnings: list | None = None) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(default, dict) and not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except ValueError as exc:
        if warnings is None:
            raise
        warnings.append(f"{path}: {exc}")
        return default


def exception_fields(exc: BaseException, stage: str) -> dict:
    return {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def environment_info() -> dict:
    versions = {}
    for name in ("numpy", "torch", "transformers", "websockets", "sapien", "mplib", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "versions": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def git_info(directory: Path) -> dict:
    def run(*args: str) -> str | None:
        process = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": status,
        "dirty": bool(status) if status is not None else None,
        "submodules": run("submodule", "status"),
    }


def episode_directory_name(context: dict) -> str:
    """Keep retries distinct within a single task-level episodes directory."""
    return f"episode_{int(context['episode_id'])}_seed_{int(context['seed'])}_attempt_{int(context['attempt'])}"


def prediction_path(run: Path, context: dict) -> Path:
    """Construct paths from validated identifiers, never a client-supplied filesystem path."""
    for key in ("task", "request_id"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(context[key])):
            raise ValueError(f"Invalid evaluation identifier: {key}")
    if context["run_id"] != run.name:
        raise ValueError("Evaluation context belongs to a different run")
    numbers = {key: int(context[key]) for key in ("attempt", "episode_id", "seed")}
    if numbers["attempt"] < 1 or min(numbers["episode_id"], numbers["seed"]) < 0:
        raise ValueError("Invalid evaluation attempt/episode/seed")
    return (
        run
        / "eval_results"
        / context["task"]
        / "episodes"
        / episode_directory_name(numbers)
        / "predictions"
        / f"request_{context['request_id']}.npz"
    )


def read_task_attempts(root: Path, warnings: list | None = None) -> list[dict]:
    """Group flat task records by attempt; also read previously saved nested logs."""
    warnings = warnings if warnings is not None else []
    attempts = {}
    for directory in sorted((root / "attempts").glob("attempt_*")):
        attempt = int(directory.name.split("_")[-1])
        before = len(warnings)
        config = read_json(directory / "task_config.json", {}, warnings)
        terminal = read_json(directory / "attempt_result.json", {}, warnings)
        episodes = read_events(directory / "episode_results.jsonl", warnings)
        attempts[attempt] = {
            "attempt": attempt,
            "config": config,
            "terminal": terminal,
            "episodes": episodes,
            "valid": len(warnings) == before,
        }
    flat_attempts = {}
    for filename, field in (
        ("task_config.jsonl", "config"),
        ("attempt_results.jsonl", "terminal"),
        ("episode_results.jsonl", "episodes"),
    ):
        for item in read_events(root / filename, warnings):
            attempt = item.get("attempt")
            if not isinstance(attempt, int) or attempt < 1:
                warnings.append(f"{root / filename}: missing or invalid attempt")
                continue
            grouped = flat_attempts.setdefault(
                attempt,
                {
                    "attempt": attempt,
                    "config": {},
                    "terminal": {},
                    "episodes": [],
                },
            )
            if field == "episodes":
                grouped[field].append(item)
            else:
                grouped[field] = item
    for attempt, grouped in flat_attempts.items():
        # A damaged line from a prior retry must not invalidate later complete retries.
        # Completeness still requires a config, terminal record and every unique episode.
        attempts[attempt] = {**grouped, "valid": bool(grouped["config"])}
    return [attempts[attempt] for attempt in sorted(attempts)]


def summarize_run(run: Path, state: str | None = None, write: bool = True) -> dict:
    """Count only complete attempts; retain latest partial results separately."""
    run = Path(run)
    warnings = []
    manifest = read_json(run / "run_manifest.json", {}, warnings)
    scheduler = read_events(run / "scheduler_events.jsonl", warnings)
    names = manifest.get("tasks") or sorted(path.name for path in (run / "eval_results").glob("*"))
    task_rows = []
    for name in names:
        root = run / "eval_results" / name
        attempts = []
        for details in read_task_attempts(root, warnings):
            attempt = details["attempt"]
            config, episodes = details["config"], details["episodes"]
            expected = config.get("requested_episodes", manifest.get("num_episodes"))
            exits = [
                item
                for item in scheduler
                if item.get("task") == name and item.get("attempt") == attempt and item.get("event") == "task_exit"
            ]
            terminal = details["terminal"]
            complete = (
                terminal.get("status") == "complete"
                and len(episodes) == expected
                and len({item.get("episode_id") for item in episodes}) == expected
                and details["valid"]
                and all(item.get("reason") in ("success", "step_limit") for item in episodes)
                and (not exits or exits[-1].get("exit_code") == 0)
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "complete": complete,
                    "episodes": len(episodes),
                    "successes": sum(e.get("reason") == "success" for e in episodes),
                    "errors": sum(e.get("reason") not in ("success", "step_limit") for e in episodes),
                    "exit_code": exits[-1].get("exit_code") if exits else None,
                }
            )
        complete_attempts = [item for item in attempts if item["complete"]]
        selected = (complete_attempts or attempts or [{}])[-1]
        row = record(
            run_id=run.name,
            task=name,
            status="complete" if complete_attempts else "incomplete",
            selected_attempt=selected.get("attempt"),
            episodes=selected.get("episodes", 0),
            successes=selected.get("successes", 0),
            errors=selected.get("errors", 0),
            attempts=attempts,
        )
        if write:
            atomic_json(root / "task_summary.json", row)
        if write and complete_attempts:
            (root / "_result.txt").write_text(
                f"Timestamp: {timestamp()}\n\nSelected attempt: {selected['attempt']}\n\n"
                f"{selected['successes'] / selected['episodes']}\n",
                encoding="utf-8",
            )
        elif write and attempts:
            (root / "_result.txt").write_text("No complete attempt; official rate unavailable.\n", encoding="utf-8")
        task_rows.append(row)
    complete_rows = [row for row in task_rows if row["status"] == "complete"]
    successes = sum(row["successes"] for row in complete_rows)
    episodes = sum(row["episodes"] for row in complete_rows)
    old = read_json(run / "summary.json", {}, warnings)
    result = record(
        run_id=run.name,
        state=state or old.get("state", "running"),
        tasks=task_rows,
        completed_tasks=len(complete_rows),
        requested_tasks=len(names),
        successes=successes,
        episodes=episodes,
        success_rate=successes / episodes if episodes else None,
        coverage=len(complete_rows) / len(names) if names else 0,
        warnings=warnings,
    )
    if write:
        atomic_json(run / "summary.json", result)
    lines = [
        f"Run: {run.name}",
        f"State: {result['state']}",
        f"Complete task coverage: {len(complete_rows)}/{len(names)}",
        "Task | Status | Attempt | Success/Episodes",
    ]
    lines += [
        f"{r['task']} | {r['status']} | {r['selected_attempt']} | {r['successes']}/{r['episodes']}" for r in task_rows
    ]
    lines += [
        f"Official successes/episodes (complete attempts only): {successes}/{episodes}",
        f"Official success rate: {result['success_rate']}",
        "Incomplete tasks are excluded from the official denominator; see coverage.",
    ]
    if write:
        (run / "stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def format_progress(run: Path, task: str, attempt: int) -> str:
    """Show the current attempt's observed rate and lifecycle state."""
    root = run / "eval_results" / task
    details = next((item for item in read_task_attempts(root) if item["attempt"] == attempt), {})
    episodes = details.get("episodes", [])
    finished = [item for item in episodes if item.get("reason") in ("success", "step_limit")]
    successes = sum(item.get("reason") == "success" for item in finished)
    expected = details.get("config", {}).get(
        "requested_episodes", read_json(run / "run_manifest.json", {}).get("num_episodes", "?")
    )
    rate = f"{successes / len(finished) * 100:.1f}%" if finished else "N/A"
    terminal = details.get("terminal", {})
    state = terminal.get("status")
    if not state:
        if episodes and episodes[-1].get("reason") not in ("success", "step_limit"):
            state = episodes[-1].get("reason", "error")
        else:
            active = any((root / "episodes").glob(f"*_attempt_{attempt}/episode.json"))
            active = active or any((root / "attempts" / f"attempt_{attempt}" / "episodes").glob("*/episode.json"))
            state = "running" if episodes or active else "initializing"
    return f"episodes {len(finished)}/{expected}, success {successes}, rate {rate} ({state})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "event", "summary", "progress"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state")
    parser.add_argument("--event")
    parser.add_argument("--fields", nargs="*", default=[])
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--raw-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    fields = {}
    for pair in args.fields:
        key, value = pair.split("=", 1)
        try:
            value = json.loads(value.lower() if value in ("True", "False") else value)
        except ValueError:
            pass
        fields[key] = value
    if args.command == "init":
        atomic_json(
            args.run / "run_manifest.json",
            record(
                run_id=args.run.name,
                raw_args=args.raw_args,
                tasks=args.tasks,
                environment=environment_info(),
                repository=git_info(Path.cwd()),
                robotwin_repository=git_info(Path(fields.get("robotwin_dir", "RoboTwin"))),
                **fields,
            ),
        )
    elif args.command == "event":
        EventWriter(args.run / "scheduler_events.jsonl", {"run_id": args.run.name}).write(args.event, **fields)
    elif args.command == "progress":
        print(format_progress(args.run, fields["task"], fields["attempt"]))
    else:
        summarize_run(args.run, args.state)


if __name__ == "__main__":
    main()
