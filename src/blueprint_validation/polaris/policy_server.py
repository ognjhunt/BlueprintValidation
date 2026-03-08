"""Websocket policy server for OpenVLA-backed PolaRiS evaluation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ..config import PolicyAdapterConfig
from ..policy_adapters.openvla_oft_adapter import OpenVLAOFTPolicyAdapter
from .openvla_client import normalize_openvla_action
from .websocket_policy import decode_payload, encode_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--observation-mode", default="external_only")
    return parser


async def serve(args: argparse.Namespace) -> None:
    import msgpack
    import websockets

    adapter = OpenVLAOFTPolicyAdapter(PolicyAdapterConfig(name="openvla_oft"))
    checkpoint = Path(args.checkpoint_path).resolve() if str(args.checkpoint_path).strip() else None
    handle = adapter.load_policy(
        model_name=str(args.model_name),
        checkpoint_path=checkpoint,
        device=str(args.device),
    )

    async def handler(websocket) -> None:
        async for raw_message in websocket:
            request = decode_payload(msgpack.unpackb(raw_message, raw=False))
            frame = request.get("image")
            task_prompt = str(request.get("instruction", "") or "").strip()
            action = adapter.predict_action(
                handle=handle,
                frame=frame,
                task_prompt=task_prompt,
                unnorm_key=None,
                device=str(args.device),
            )
            payload = {"action": normalize_openvla_action(action)}
            await websocket.send(msgpack.packb(encode_payload(payload), use_bin_type=True))

    async with websockets.serve(handler, str(args.host), int(args.port), max_size=None):
        print("POLARIS_OPENVLA_SERVER_READY", flush=True)
        await asyncio.Future()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(serve(args))


if __name__ == "__main__":  # pragma: no cover
    main()
