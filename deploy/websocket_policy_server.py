import asyncio
import hashlib
import http
import json
import logging
import os
import time
import traceback
from pathlib import Path

import websockets.asyncio.server as _server
import websockets.frames

from .eval_logging import EventWriter, atomic_json, atomic_npz, exception_fields, prediction_path, record
from .msgpack_numpy import Packer, unpackb


logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        eval_run_dir: str | None = None,
        eval_slot: int = 0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._run = Path(eval_run_dir).resolve() if eval_run_dir else None
        self._slot = eval_slot
        self._metadata = dict(metadata or {})
        self._metadata["eval_trace_schema"] = 1 if self._run and hasattr(policy, "infer_with_diagnostics") else None
        self._events = (
            EventWriter(
                self._run / "inference_logs" / f"slot_{eval_slot}.events.jsonl",
                {"run_id": self._run.name, "slot": eval_slot},
            )
            if self._run
            else None
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=_optional_float_env("WEBSOCKET_PING_INTERVAL"),
            ping_timeout=_optional_float_env("WEBSOCKET_PING_TIMEOUT"),
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())
                action = self._infer(obs)

                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                # Historical metric includes recv() waiting for client execution.
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _infer(self, observation):
        context = observation.pop("_eval_context", None)
        started = time.monotonic()
        try:
            if context:
                if not self._metadata["eval_trace_schema"]:
                    raise ValueError("This policy server does not support full evaluation tracing")
                if int(context["slot"]) != self._slot:
                    raise ValueError("Evaluation context belongs to a different server slot")
                destination = prediction_path(self._run, context)
                self._events.write("request_start", **context, reset=bool(observation.get("reset")))
                started = time.monotonic()
                if observation.get("reset"):
                    result = self._policy.infer(observation)
                    infer_ms = (time.monotonic() - started) * 1000
                    metadata_path = self._save_model_metadata()
                    diagnostic = {"timing": {}, "available": True}
                    artifact = None
                else:
                    result = self._policy.infer_with_diagnostics(observation)
                    infer_ms = (time.monotonic() - started) * 1000
                    diagnostic = result.pop("_diagnostics")
                    write_started = time.monotonic()
                    atomic_npz(destination, diagnostic.pop("arrays"))
                    diagnostic["logging_ms"] = (time.monotonic() - write_started) * 1000
                    artifact = destination.relative_to(self._run).as_posix()
                    metadata_path = None
                result["_eval_trace"] = {
                    **context,
                    **diagnostic,
                    "artifact": artifact,
                    "model_metadata": metadata_path,
                }
                self._events.write(
                    "request_end",
                    **context,
                    artifact=artifact,
                    infer_ms=infer_ms,
                    diagnostic={k: v for k, v in diagnostic.items() if k != "arrays"},
                )
            else:
                result = self._policy.infer(observation)
                infer_ms = (time.monotonic() - started) * 1000
            result = dict(result)
            result["server_timing"] = {"infer_ms": infer_ms}
            if context:
                result["server_timing"].update(diagnostic.get("timing", {}))
                result["server_timing"]["logging_ms"] = diagnostic.get("logging_ms", 0)
            return result
        except BaseException as exc:
            if self._events:
                self._events.write("request_error", **(context or {}), **exception_fields(exc, "infer"))
            raise

    def _save_model_metadata(self):
        metadata = self._policy.evaluation_metadata()
        source = Path(metadata["normalization_path"])
        content = source.read_bytes()
        fingerprint = hashlib.sha256(content).hexdigest()
        relative = f"normalization/slot_{self._slot}_{fingerprint}.json"
        snapshot = self._run / relative
        if not snapshot.exists():
            # Preserve the source bytes so the published SHA256 verifies the snapshot.
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            temporary = snapshot.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(snapshot)
        metadata["normalization_sha256"] = fingerprint
        metadata["normalization_snapshot"] = relative
        # Hash the actual configuration, so checkpoint reloads keep prior metadata.
        from .eval_logging import json_value

        metadata_hash = hashlib.sha256(json.dumps(json_value(metadata), sort_keys=True).encode()).hexdigest()
        path = self._run / "inference_logs" / f"slot_{self._slot}.model_{metadata_hash}.json"
        if not path.exists():
            atomic_json(path, record(run_id=self._run.name, slot=self._slot, **metadata))
        return path.relative_to(self._run).as_posix()


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.lower() == "none":
        return None
    return float(value)
