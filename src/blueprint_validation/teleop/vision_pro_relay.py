"""Vision Pro control relay for CloudXR/Isaac teleoperation."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


class VisionProRelayError(RuntimeError):
    """Raised when the Vision Pro relay cannot normalize or forward packets."""


@dataclass
class VisionProRelayConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 49111
    target_host: str = "127.0.0.1"
    target_port: int = 49110
    action_dim: int = 7
    connect_timeout_s: float = 120.0
    idle_timeout_s: float = 10.0
    target_retry_seconds: float = 0.5
    packet_log_path: Path | None = None
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    gripper_open_value: float = 1.0
    gripper_close_value: float = -1.0


def normalize_vision_pro_packet(
    packet: Mapping[str, Any],
    *,
    action_dim: int,
    translation_scale: float,
    rotation_scale: float,
    gripper_open_value: float,
    gripper_close_value: float,
) -> Dict[str, Any]:
    """Normalize Vision Pro control payloads into the recorder bridge contract."""
    if packet.get("done") is True:
        return {"done": True}

    if "action" in packet:
        action = np.asarray(packet["action"], dtype=np.float32).reshape(-1)
        if action.size != int(action_dim):
            raise VisionProRelayError(
                f"Direct action packet has size {action.size}, expected {action_dim}."
            )
        if not np.isfinite(action).all():
            raise VisionProRelayError("Direct action packet contains non-finite values.")
        return {"action": [float(v) for v in action.tolist()], "done": False}

    if "ee_delta_pose" in packet:
        pose = np.asarray(packet["ee_delta_pose"], dtype=np.float32).reshape(-1)
        if pose.size != 6:
            raise VisionProRelayError("ee_delta_pose packet must contain 6 values.")
        gripper_delta = float(packet.get("gripper_delta", 0.0) or 0.0)
        action = np.zeros((int(action_dim),), dtype=np.float32)
        action[:6] = pose
        action[6] = gripper_delta
        if not np.isfinite(action).all():
            raise VisionProRelayError("ee_delta_pose packet contains non-finite values.")
        return {"action": [float(v) for v in action.tolist()], "done": False}

    translation, rotation = _extract_pose_like_fields(packet)
    pinch_state = _extract_pinch(packet)
    action = np.zeros((int(action_dim),), dtype=np.float32)
    action[:3] = np.asarray(translation, dtype=np.float32) * float(translation_scale)
    action[3:6] = np.asarray(rotation, dtype=np.float32) * float(rotation_scale)
    if pinch_state is True:
        action[6] = float(gripper_close_value)
    elif pinch_state is False:
        action[6] = float(gripper_open_value)
    if not np.isfinite(action).all():
        raise VisionProRelayError("Normalized Vision Pro packet contains non-finite values.")
    return {"action": [float(v) for v in action.tolist()], "done": False}


def run_vision_pro_relay(config: VisionProRelayConfig) -> None:
    """Run a TCP JSON-lines relay from the Vision Pro client side into record-teleop."""
    packet_log = (
        config.packet_log_path.open("w", encoding="utf-8")
        if config.packet_log_path is not None
        else None
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((str(config.listen_host), int(config.listen_port)))
    server.listen(1)
    server.settimeout(float(config.connect_timeout_s))
    try:
        try:
            source_conn, source_addr = server.accept()
        except socket.timeout as exc:
            raise VisionProRelayError(
                "Vision Pro relay timed out waiting for a client connection on "
                f"{config.listen_host}:{config.listen_port}."
            ) from exc
        with source_conn:
            source_conn.settimeout(float(config.idle_timeout_s))
            target_conn = _connect_target(config)
            with target_conn:
                target_conn.settimeout(float(config.idle_timeout_s))
                buffer = b""
                while True:
                    try:
                        chunk = source_conn.recv(65536)
                    except socket.timeout as exc:
                        raise VisionProRelayError(
                            "Vision Pro relay source connection timed out waiting for data."
                        ) from exc
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            packet = json.loads(line.decode("utf-8"))
                        except Exception as exc:
                            raise VisionProRelayError(
                                f"Vision Pro relay received invalid JSON packet: {line[:120]!r}"
                            ) from exc
                        normalized = normalize_vision_pro_packet(
                            packet,
                            action_dim=int(config.action_dim),
                            translation_scale=float(config.translation_scale),
                            rotation_scale=float(config.rotation_scale),
                            gripper_open_value=float(config.gripper_open_value),
                            gripper_close_value=float(config.gripper_close_value),
                        )
                        if packet_log is not None:
                            packet_log.write(json.dumps({"source": packet, "normalized": normalized}) + "\n")
                            packet_log.flush()
                        target_conn.sendall((json.dumps(normalized) + "\n").encode("utf-8"))
                        if normalized.get("done") is True:
                            return
    finally:
        if packet_log is not None:
            packet_log.close()
        server.close()


def _connect_target(config: VisionProRelayConfig) -> socket.socket:
    deadline = time.monotonic() + float(config.connect_timeout_s)
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            conn.connect((str(config.target_host), int(config.target_port)))
            return conn
        except Exception as exc:  # pragma: no cover - timing dependent
            last_exc = exc
            conn.close()
            time.sleep(float(config.target_retry_seconds))
    raise VisionProRelayError(
        "Could not connect relay to teleop recorder bridge at "
        f"{config.target_host}:{config.target_port}: {last_exc}"
    )


def _extract_pose_like_fields(packet: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    if "translation" in packet or "rotation_rpy" in packet:
        return _vector3(packet.get("translation", [0.0, 0.0, 0.0]), "translation"), _vector3(
            packet.get("rotation_rpy", [0.0, 0.0, 0.0]), "rotation_rpy"
        )

    for hand_key in ("right_hand", "hand_pose", "pose"):
        hand_payload = packet.get(hand_key)
        if isinstance(hand_payload, Mapping):
            translation = hand_payload.get("translation", [0.0, 0.0, 0.0])
            rotation = hand_payload.get("rotation_rpy", hand_payload.get("rotation", [0.0, 0.0, 0.0]))
            return _vector3(translation, f"{hand_key}.translation"), _vector3(
                rotation, f"{hand_key}.rotation"
            )

    raise VisionProRelayError(
        "Vision Pro packet must contain one of: action, ee_delta_pose, translation/rotation_rpy, "
        "or a hand pose object with translation/rotation."
    )


def _extract_pinch(packet: Mapping[str, Any]) -> bool | None:
    if "pinch" in packet:
        return bool(packet["pinch"])
    gestures = packet.get("gestures")
    if isinstance(gestures, Mapping) and "pinch" in gestures:
        return bool(gestures["pinch"])
    return None


def _vector3(value: Any, label: str) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise VisionProRelayError(f"{label} must contain 3 values.")
    if not np.isfinite(arr).all():
        raise VisionProRelayError(f"{label} contains non-finite values.")
    return [float(v) for v in arr.tolist()]
