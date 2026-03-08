"""Lightweight websocket transport shared by the PolaRiS OpenVLA client/server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np


_ARRAY_MARKER = "__ndarray__"


def encode_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            _ARRAY_MARKER: True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tobytes(),
        }
    if isinstance(value, (list, tuple)):
        return [encode_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode_payload(item) for key, item in value.items()}
    return value


def decode_payload(value: Any) -> Any:
    if isinstance(value, dict) and value.get(_ARRAY_MARKER):
        data = value.get("data", b"")
        dtype = np.dtype(str(value.get("dtype", "float32")))
        shape = tuple(int(v) for v in value.get("shape", []))
        array = np.frombuffer(data, dtype=dtype)
        return array.reshape(shape)
    if isinstance(value, list):
        return [decode_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: decode_payload(item) for key, item in value.items()}
    return value


@dataclass
class WebsocketPolicyClient:
    host: str
    port: int
    timeout_s: float = 60.0

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._infer_async(payload))

    async def _infer_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        import msgpack
        import websockets

        uri = f"ws://{self.host}:{int(self.port)}"
        async with websockets.connect(uri, max_size=None) as websocket:
            await websocket.send(msgpack.packb(encode_payload(payload), use_bin_type=True))
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.timeout_s)
        return decode_payload(msgpack.unpackb(raw, raw=False))
