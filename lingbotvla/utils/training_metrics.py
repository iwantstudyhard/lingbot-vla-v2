"""Low-overhead structured metric logging for long-running training jobs."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def build_visualization_run_dir(
    repository_root: str | os.PathLike[str],
    checkpoint_output_dir: str | os.PathLike[str],
    *,
    base_dir_override: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
) -> Path:
    """Return a unique repository-side directory for one training launch."""

    repository_root = Path(repository_root)
    if base_dir_override:
        base_dir = Path(base_dir_override).expanduser()
        if not base_dir.is_absolute():
            base_dir = repository_root / base_dir
    else:
        base_dir = repository_root / "train_outputs" / Path(checkpoint_output_dir).name

    run_id = run_id or f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("Run ID may only contain letters, digits, '.', '_' and '-'.")
    return base_dir / "runs" / run_id


class AsyncTrainingMetricsWriter:
    """Append JSONL metrics from a background thread.

    The training thread only serializes a small dictionary and enqueues it.  The
    worker batches writes and flushes periodically, which avoids a synchronous
    NFS write on every optimizer step.  ``flush`` is used before rendering a
    checkpoint snapshot so the visualizer always sees every completed step.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        flush_every: int = 100,
        flush_interval_seconds: float = 10.0,
        queue_size: int = 10_000,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_seconds = max(0.1, float(flush_interval_seconds))
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max(1, queue_size))
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="training-metrics-writer", daemon=True)
        self._thread.start()

    def write(self, record: dict[str, Any]) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot write to a closed metrics writer.")
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=True)
        self._queue.put(("record", line))

    def flush(self, timeout: float = 120.0) -> None:
        self._raise_if_failed()
        if self._closed:
            return
        done = threading.Event()
        self._queue.put(("flush", done))
        self._wait_for_event(done, timeout, f"Timed out flushing training metrics to {self.path}")
        self._raise_if_failed()

    def close(self, timeout: float = 120.0) -> None:
        if self._closed:
            return
        self.flush(timeout=timeout)
        done = threading.Event()
        self._queue.put(("close", done))
        self._wait_for_event(done, timeout, f"Timed out closing training metrics writer for {self.path}")
        self._thread.join(timeout=max(0.1, timeout))
        self._closed = True
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Training metrics writer failed for {self.path}") from self._error

    def _wait_for_event(self, event: threading.Event, timeout: float, message: str) -> None:
        deadline = time.monotonic() + max(0.1, timeout)
        while not event.wait(timeout=min(0.1, max(0.0, deadline - time.monotonic()))):
            self._raise_if_failed()
            if not self._thread.is_alive():
                raise RuntimeError(f"Training metrics writer stopped unexpectedly for {self.path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(message)

    def _run(self) -> None:
        pending = 0
        last_flush = time.monotonic()
        try:
            with self.path.open("a", encoding="utf-8", buffering=1024 * 1024) as handle:
                while True:
                    timeout = max(0.1, self.flush_interval_seconds - (time.monotonic() - last_flush))
                    try:
                        command, payload = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        if pending:
                            handle.flush()
                            pending = 0
                        last_flush = time.monotonic()
                        continue

                    if command == "record":
                        handle.write(payload)
                        handle.write("\n")
                        pending += 1
                        if pending >= self.flush_every:
                            handle.flush()
                            pending = 0
                            last_flush = time.monotonic()
                    elif command == "flush":
                        handle.flush()
                        pending = 0
                        last_flush = time.monotonic()
                        payload.set()
                    elif command == "close":
                        handle.flush()
                        payload.set()
                        return
        except BaseException as exc:  # surface background failures to the training thread
            self._error = exc
            while True:
                try:
                    command, payload = self._queue.get_nowait()
                except queue.Empty:
                    break
                if command in {"flush", "close"}:
                    payload.set()
