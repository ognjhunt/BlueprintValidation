"""Local teleop runtime and recorder for Isaac Lab-backed sessions."""

from __future__ import annotations

import json
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from ..common import write_json
from .contracts import load_and_validate_scene_package, write_teleop_manifests


class IsaacTeleopRuntimeError(RuntimeError):
    """Raised when the local Isaac teleop runtime cannot execute."""


@dataclass
class TeleopRecorderConfig:
    """Configuration for one local teleop recording session."""

    scene_root: Path
    output_dir: Path
    task_id: str
    task_text: str
    demo_index: int = 0
    robot_type: str = "franka"
    robot_asset_ref: str = "robot/franka/franka.usd"
    teleop_device: str = "keyboard"
    sim_backend: str = "isaac_sim"
    action_space: str = "ee_delta_pose_gripper"
    action_dim: int = 7
    max_steps: int = 200
    headless: bool = False
    success: Optional[bool] = None
    task_package: Optional[str] = None
    env_cfg_class: Optional[str] = None
    camera_keys: List[str] = field(default_factory=list)
    state_keys: List[str] = field(default_factory=list)
    scripted_commands: List[str] = field(default_factory=list)
    translation_step_m: float = 0.02
    rotation_step_rad: float = 0.12
    gripper_step: float = 1.0
    spacemouse_deadzone: float = 0.08
    spacemouse_translation_scale: float = 0.03
    spacemouse_rotation_scale: float = 0.18
    bridge_host: str = "0.0.0.0"
    bridge_port: int = 49110
    bridge_connect_timeout_s: float = 120.0
    bridge_idle_timeout_s: float = 10.0
    bridge_packet_log_enabled: bool = True
    confirm_success: bool = True
    max_attempts: int = 1
    attempt_pause_seconds: float = 0.5


@dataclass
class RecordedSession:
    """Materialized teleop session outputs prior to manifest packaging."""

    session_dir: Path
    video_paths: Dict[str, Path]
    calibration_refs: Dict[str, Path]
    action_sequence_path: Path
    state_sequence_path: Path
    lerobot_root: Path
    episode_ref: str
    state_keys: List[str]
    joint_names: List[str]
    num_steps: int


class TeleopBackend(Protocol):
    """Backend interface for recording teleop sessions."""

    def record(self, config: TeleopRecorderConfig) -> RecordedSession:
        ...


def record_teleop_session(
    config: TeleopRecorderConfig,
    *,
    backend: TeleopBackend | None = None,
    confirm_fn: Callable[[TeleopRecorderConfig, int, RecordedSession], bool] | None = None,
) -> Dict[str, Path]:
    """Record a teleop session and emit manifest artifacts."""
    if config.robot_type.strip().lower() != "franka":
        raise IsaacTeleopRuntimeError("record-teleop v1 currently supports robot_type=franka only.")
    if config.action_space.strip().lower() != "ee_delta_pose_gripper":
        raise IsaacTeleopRuntimeError(
            "record-teleop v1 currently supports action_space=ee_delta_pose_gripper only."
        )

    load_and_validate_scene_package(config.scene_root)
    backend = backend or IsaacLabLocalBackend()
    confirm = confirm_fn or _default_success_confirm
    last_recorded: RecordedSession | None = None
    final_success = False
    for attempt_idx in range(max(1, int(config.max_attempts))):
        attempt_output_dir = (
            config.output_dir / f"attempt_{attempt_idx:02d}"
            if int(config.max_attempts) > 1
            else config.output_dir
        )
        attempt_cfg = replace(config, output_dir=attempt_output_dir)
        recorded = backend.record(attempt_cfg)
        last_recorded = recorded
        is_success = (
            bool(config.success)
            if config.success is not None
            else (
                confirm(attempt_cfg, attempt_idx, recorded)
                if bool(config.confirm_success)
                else True
            )
        )
        if is_success:
            final_success = True
            break
        if attempt_idx >= max(1, int(config.max_attempts)) - 1:
            final_success = False
            break
        if attempt_output_dir.exists():
            shutil.rmtree(attempt_output_dir, ignore_errors=True)
        time.sleep(max(0.0, float(config.attempt_pause_seconds)))
    if last_recorded is None:
        raise IsaacTeleopRuntimeError("No teleop attempts were recorded.")
    recorded = last_recorded
    session = {
        "session_id": f"{config.scene_root.name}::{config.task_id}::{config.demo_index:03d}",
        "scene_id": config.scene_root.name,
        "task_id": config.task_id,
        "task_text": config.task_text,
        "demo_index": int(config.demo_index),
        "success": bool(final_success),
        "sim_backend": config.sim_backend,
        "teleop_device": config.teleop_device,
        "robot_type": config.robot_type,
        "robot_asset_ref": config.robot_asset_ref,
        "action_space": config.action_space,
        "action_dim": int(config.action_dim),
        "joint_names": list(recorded.joint_names),
        "state_keys": list(recorded.state_keys),
        "camera_ids": sorted(recorded.video_paths.keys()),
        "video_paths": {key: str(path) for key, path in recorded.video_paths.items()},
        "calibration_refs": {key: str(path) for key, path in recorded.calibration_refs.items()},
        "lerobot_root": str(recorded.lerobot_root),
        "episode_ref": recorded.episode_ref,
        "start_state_hash": _hash_start_state(recorded.state_sequence_path),
        "action_sequence_path": str(recorded.action_sequence_path),
        "state_sequence_path": str(recorded.state_sequence_path),
    }
    return write_teleop_manifests(
        output_dir=config.output_dir,
        source_name="teleop",
        sessions=[session],
    )


class IsaacLabLocalBackend:
    """Local teleop backend using an Isaac Lab ManagerBasedEnv."""

    def record(self, config: TeleopRecorderConfig) -> RecordedSession:
        modules = _load_isaac_lab_modules()
        device = str(config.teleop_device or "keyboard").strip().lower()
        if device not in {"keyboard", "spacemouse", "vision_pro"}:
            raise IsaacTeleopRuntimeError(
                "record-teleop currently supports teleop_device in "
                "{'keyboard', 'spacemouse', 'vision_pro'} only."
            )

        scene_root = config.scene_root.resolve()
        scene_info = load_and_validate_scene_package(scene_root)
        if not scene_info.get("has_isaac_lab", False):
            raise IsaacTeleopRuntimeError(
                f"Scene package does not include an isaac_lab directory: {scene_root}"
            )

        task_package = config.task_package or _default_task_package_name(scene_root)
        env_cfg_class = config.env_cfg_class or "TeleopEnvCfg"

        isaac_lab_root = scene_root / "isaac_lab"
        if str(isaac_lab_root) not in sys.path:
            sys.path.insert(0, str(isaac_lab_root))

        app_launcher = modules["AppLauncher"](
            {
                "headless": bool(config.headless),
            }
        )
        app_launcher.start()

        try:
            env_cfg = modules["parse_env_cfg"](
                _load_env_cfg(task_package, env_cfg_class, num_envs=1)
            )
            env = modules["ManagerBasedEnv"](env_cfg)
            try:
                return _record_with_env(env, config)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        finally:
            stop = getattr(app_launcher, "stop", None)
            if callable(stop):
                stop()


def _record_with_env(env: Any, config: TeleopRecorderConfig) -> RecordedSession:
    torch = _import_torch()
    session_dir = config.output_dir / f"{config.task_id}_{config.demo_index:03d}"
    session_dir.mkdir(parents=True, exist_ok=True)
    video_dir = session_dir / "videos"
    calib_dir = session_dir / "calibrations"
    lerobot_root = session_dir / "lerobot"
    video_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)
    lerobot_root.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    action_dim = _resolve_env_action_dim(env)
    if int(action_dim) != int(config.action_dim):
        raise IsaacTeleopRuntimeError(
            f"Isaac env action_dim={action_dim} does not match requested action_dim={config.action_dim}"
        )

    frames_by_camera: Dict[str, List[np.ndarray]] = {}
    state_rows: List[Dict[str, Any]] = []
    action_rows: List[List[float]] = []
    camera_keys = list(config.camera_keys)
    state_keys = list(config.state_keys)

    packet_log_path = session_dir / "vision_pro_bridge_packets.jsonl"
    for step_idx, action in enumerate(_action_stream(config, packet_log_path=packet_log_path)):
        if step_idx >= int(config.max_steps):
            break
        if action is None:
            break
        action_tensor = torch.tensor(np.asarray(action)[None, :], device=env.device, dtype=torch.float32)
        step_result = env.step(action_tensor)
        if not isinstance(step_result, tuple) or len(step_result) < 1:
            raise IsaacTeleopRuntimeError("Isaac env.step did not return observations.")
        obs = step_result[0]
        rewards = step_result[1] if len(step_result) > 1 else None
        dones = step_result[2] if len(step_result) > 2 else None
        info = step_result[3] if len(step_result) > 3 else {}
        del rewards, info

        frame_payload = extract_camera_frames(obs, requested_keys=camera_keys)
        if not frame_payload:
            raise IsaacTeleopRuntimeError(
                "No image-like observations found. Pass --camera-key to map Isaac camera observations."
            )
        if not camera_keys:
            camera_keys = _sorted_camera_keys(frame_payload)
        for camera_id, frame in frame_payload.items():
            frames_by_camera.setdefault(camera_id, []).append(frame)

        state_row = extract_state_row(obs, requested_keys=state_keys)
        if not state_keys:
            state_keys = sorted(state_row.keys())
        state_rows.append(state_row)
        action_rows.append([float(v) for v in action.tolist()])

        if _any_done(dones):
            break

    if not action_rows:
        raise IsaacTeleopRuntimeError("No teleop steps were recorded.")

    action_path = session_dir / "actions.json"
    state_path = session_dir / "states.json"
    write_json(action_rows, action_path)
    write_json(state_rows, state_path)

    video_paths: Dict[str, Path] = {}
    calibration_refs: Dict[str, Path] = {}
    for camera_id, frames in frames_by_camera.items():
        video_path = video_dir / f"{camera_id}.mp4"
        _write_mp4(video_path, frames)
        video_paths[camera_id] = video_path
        calib_path = calib_dir / f"{camera_id}_calibration.json"
        write_json(build_camera_calibration(camera_id, frames[0]), calib_path)
        calibration_refs[camera_id] = calib_path

    episode_ref = f"episode_{int(config.demo_index):06d}"
    _write_minimal_lerobot_root(
        root=lerobot_root,
        episode_ref=episode_ref,
        task_text=config.task_text,
        num_steps=len(action_rows),
    )

    return RecordedSession(
        session_dir=session_dir,
        video_paths=video_paths,
        calibration_refs=calibration_refs,
        action_sequence_path=action_path,
        state_sequence_path=state_path,
        lerobot_root=lerobot_root,
        episode_ref=episode_ref,
        state_keys=state_keys,
        joint_names=[f"joint_{idx}" for idx in range(max(7, int(config.action_dim)))],
        num_steps=len(action_rows),
    )


def extract_camera_frames(obs: Any, *, requested_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """Extract RGB frames from Isaac observations."""
    frames: Dict[str, np.ndarray] = {}
    candidates = _flatten_obs(obs)
    for key, value in candidates.items():
        canonical = str(key)
        if requested_keys and not _matches_requested_key(canonical, requested_keys):
            continue
        frame = _to_rgb_frame(value)
        if frame is not None and _looks_like_camera_key(canonical):
            frames[_canonical_camera_name(canonical)] = frame
    if not requested_keys and not frames:
        for key, value in candidates.items():
            frame = _to_rgb_frame(value)
            if frame is not None:
                frames[_canonical_camera_name(str(key))] = frame
    if requested_keys:
        missing = [key for key in requested_keys if not any(_matches_requested_key(k, [key]) for k in frames)]
        if missing:
            raise IsaacTeleopRuntimeError(
                f"Requested camera keys missing from observations: {', '.join(missing)}"
            )
    return frames


def extract_state_row(obs: Any, *, requested_keys: Sequence[str]) -> Dict[str, Any]:
    """Extract a JSON-serializable state row from observations."""
    flat = _flatten_obs(obs)
    state: Dict[str, Any] = {}
    for key, value in flat.items():
        if requested_keys and not _matches_requested_key(key, requested_keys):
            continue
        arr = _to_numeric_vector(value)
        if arr is None:
            continue
        state[key] = [float(v) for v in arr.tolist()]
    if requested_keys:
        missing = [key for key in requested_keys if key not in state]
        if missing:
            raise IsaacTeleopRuntimeError(
                f"Requested state keys missing from observations: {', '.join(missing)}"
            )
    if not state:
        raise IsaacTeleopRuntimeError(
            "No numeric state observations found. Pass --state-key to map Isaac state observations."
        )
    return state


def keyboard_command_to_action(
    command: str,
    *,
    action_dim: int,
    translation_step_m: float,
    rotation_step_rad: float,
    gripper_step: float,
) -> np.ndarray:
    """Map one text command into the fixed Franka teleop action vector."""
    if int(action_dim) != 7:
        raise IsaacTeleopRuntimeError(f"Keyboard teleop expects action_dim=7, got {action_dim}.")

    action = np.zeros((7,), dtype=np.float32)
    tokens = list((command or "").strip().lower())
    keymap = {
        "w": (0, translation_step_m),
        "s": (0, -translation_step_m),
        "d": (1, translation_step_m),
        "a": (1, -translation_step_m),
        "r": (2, translation_step_m),
        "f": (2, -translation_step_m),
        "i": (3, rotation_step_rad),
        "k": (3, -rotation_step_rad),
        "j": (4, rotation_step_rad),
        "l": (4, -rotation_step_rad),
        "u": (5, rotation_step_rad),
        "o": (5, -rotation_step_rad),
        "g": (6, gripper_step),
        "h": (6, -gripper_step),
    }
    for token in tokens:
        if token in {" ", "\t"}:
            continue
        if token not in keymap:
            raise IsaacTeleopRuntimeError(
                f"Unsupported keyboard token '{token}'. Use w/s/a/d/r/f/i/k/j/l/u/o/g/h."
            )
        idx, delta = keymap[token]
        action[idx] += float(delta)
    return action


def spacemouse_state_to_action(
    state: Any,
    *,
    action_dim: int,
    deadzone: float,
    translation_scale: float,
    rotation_scale: float,
    gripper_step: float,
) -> np.ndarray:
    """Map one SpaceMouse state packet into the fixed Franka teleop action vector."""
    if int(action_dim) != 7:
        raise IsaacTeleopRuntimeError(f"SpaceMouse teleop expects action_dim=7, got {action_dim}.")
    required = ("x", "y", "z", "roll", "pitch", "yaw")
    values = {name: float(getattr(state, name, 0.0)) for name in required}
    action = np.zeros((7,), dtype=np.float32)
    axes = [
        ("x", 0, translation_scale),
        ("y", 1, translation_scale),
        ("z", 2, translation_scale),
        ("roll", 3, rotation_scale),
        ("pitch", 4, rotation_scale),
        ("yaw", 5, rotation_scale),
    ]
    for attr, idx, scale in axes:
        value = values[attr]
        if abs(value) < float(deadzone):
            value = 0.0
        action[idx] = float(value * scale)
    buttons = getattr(state, "buttons", None)
    if isinstance(buttons, (list, tuple)) and buttons:
        if len(buttons) > 0 and bool(buttons[0]):
            action[6] += float(gripper_step)
        if len(buttons) > 1 and bool(buttons[1]):
            action[6] -= float(gripper_step)
    return action


def build_camera_calibration(camera_id: str, frame: np.ndarray) -> Dict[str, Any]:
    """Construct a minimal camera calibration payload from one frame."""
    height, width = frame.shape[:2]
    fx = width / 2.0
    fy = height / 2.0
    return {
        "camera_id": camera_id,
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "ppx": float(width / 2.0),
        "ppy": float(height / 2.0),
        "source": "record_teleop_v1_placeholder",
    }


def _write_minimal_lerobot_root(*, root: Path, episode_ref: str, task_text: str, num_steps: int) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_json({"dataset_name": "teleop_v1", "version": "0.1"}, meta_dir / "info.json")
    write_json({"num_episodes": 1, "num_steps": int(num_steps)}, meta_dir / "stats.json")
    (meta_dir / "tasks.jsonl").write_text(json.dumps({"task": task_text}) + "\n", encoding="utf-8")
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps({"episode_ref": episode_ref, "num_steps": int(num_steps)}) + "\n",
        encoding="utf-8",
    )


def _write_mp4(path: Path, frames: Sequence[np.ndarray], *, fps: int = 10) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise IsaacTeleopRuntimeError("OpenCV is required to write teleop MP4 videos.") from exc

    if not frames:
        raise IsaacTeleopRuntimeError(f"No frames available for video export: {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise IsaacTeleopRuntimeError("All teleop frames must share the same resolution.")
            writer.write(frame[:, :, ::-1].copy())
    finally:
        writer.release()


def _hash_start_state(path: Path) -> str:
    payload = path.read_bytes()
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _command_stream(config: TeleopRecorderConfig) -> Iterable[str]:
    if config.scripted_commands:
        for command in list(config.scripted_commands):
            yield str(command)
        return

    _print_keyboard_help()
    while True:
        command = input("teleop> ").strip()
        yield command
        if command in {"q", "quit", "done", "exit"}:
            break


def _action_stream(
    config: TeleopRecorderConfig,
    *,
    packet_log_path: Path,
) -> Iterable[np.ndarray | None]:
    device = str(config.teleop_device or "keyboard").strip().lower()
    if device == "keyboard":
        for command in _command_stream(config):
            if command in {"q", "quit", "done", "exit"}:
                yield None
                break
            yield keyboard_command_to_action(
                command,
                action_dim=int(config.action_dim),
                translation_step_m=float(config.translation_step_m),
                rotation_step_rad=float(config.rotation_step_rad),
                gripper_step=float(config.gripper_step),
            )
        return

    if device == "spacemouse":
        reader = _load_spacemouse_reader()
        _print_spacemouse_help()
        try:
            while True:
                state = reader.read()
                if state is None:
                    time.sleep(0.01)
                    continue
                buttons = getattr(state, "buttons", None)
                if isinstance(buttons, (list, tuple)) and len(buttons) >= 2 and all(bool(v) for v in buttons[:2]):
                    yield None
                    break
                yield spacemouse_state_to_action(
                    state,
                    action_dim=int(config.action_dim),
                    deadzone=float(config.spacemouse_deadzone),
                    translation_scale=float(config.spacemouse_translation_scale),
                    rotation_scale=float(config.spacemouse_rotation_scale),
                    gripper_step=float(config.gripper_step),
                )
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        return

    if device == "vision_pro":
        _print_vision_pro_help(config)
        for packet in _vision_pro_packet_stream(config, packet_log_path=packet_log_path):
            if packet.get("done") is True:
                yield None
                break
            yield vision_pro_packet_to_action(packet, action_dim=int(config.action_dim))
        return

    raise IsaacTeleopRuntimeError(f"Unsupported teleop device: {device}")


def _print_keyboard_help() -> None:
    print(
        "Keyboard teleop controls: "
        "w/s=x, a/d=y, r/f=z, i/k=roll, j/l=pitch, u/o=yaw, g=open, h=close, q=finish",
        flush=True,
    )


def _print_spacemouse_help() -> None:
    print(
        "SpaceMouse teleop: translation/rotation map directly to EE deltas; "
        "button 1=open, button 2=close, both buttons=finish.",
        flush=True,
    )


def _print_vision_pro_help(config: TeleopRecorderConfig) -> None:
    print(
        "Vision Pro teleop bridge: waiting for newline-delimited JSON control packets on "
        f"{config.bridge_host}:{config.bridge_port}.",
        flush=True,
    )
    print(
        "Accepted packet formats: "
        '{"action":[7 floats],"done":false} or {"ee_delta_pose":[6 floats],"gripper_delta":float,"done":false}',
        flush=True,
    )


def _resolve_env_action_dim(env: Any) -> int:
    action_manager = getattr(env, "action_manager", None)
    action_spec = getattr(action_manager, "action_spec", None)
    shape = getattr(action_spec, "shape", None)
    if shape is None:
        raise IsaacTeleopRuntimeError("Could not resolve env action dimension from Isaac env.")
    if isinstance(shape, tuple):
        return int(shape[0])
    return int(shape)


def _default_task_package_name(scene_root: Path) -> str:
    candidates = sorted(p for p in (scene_root / "isaac_lab").glob("*/__init__.py"))
    if not candidates:
        raise IsaacTeleopRuntimeError(
            "No Isaac Lab task package found under scene_root/isaac_lab. "
            "Pass --task-package explicitly."
        )
    return candidates[0].parent.name


def _default_success_confirm(
    config: TeleopRecorderConfig,
    attempt_idx: int,
    recorded: RecordedSession,
) -> bool:
    if config.success is not None:
        return bool(config.success)
    response = input(
        f"Did attempt {attempt_idx + 1} for task '{config.task_id}' succeed "
        f"({recorded.num_steps} steps recorded)? [y/N]: "
    ).strip().lower()
    return response in {"y", "yes"}


def vision_pro_packet_to_action(packet: Mapping[str, Any], *, action_dim: int) -> np.ndarray:
    """Convert a Vision Pro bridge packet into the fixed Franka action vector."""
    if int(action_dim) != 7:
        raise IsaacTeleopRuntimeError(f"Vision Pro teleop expects action_dim=7, got {action_dim}.")
    if "action" in packet:
        action = np.asarray(packet["action"], dtype=np.float32).reshape(-1)
        if action.size != int(action_dim):
            raise IsaacTeleopRuntimeError(
                f"Vision Pro packet action size {action.size} does not match action_dim={action_dim}."
            )
        if not np.isfinite(action).all():
            raise IsaacTeleopRuntimeError("Vision Pro packet contains non-finite action values.")
        return action

    ee_delta_pose = np.asarray(packet.get("ee_delta_pose", []), dtype=np.float32).reshape(-1)
    if ee_delta_pose.size != 6:
        raise IsaacTeleopRuntimeError(
            "Vision Pro packet must include either 'action' or 'ee_delta_pose' with 6 values."
        )
    gripper_delta = float(packet.get("gripper_delta", 0.0) or 0.0)
    action = np.zeros((7,), dtype=np.float32)
    action[:6] = ee_delta_pose
    action[6] = gripper_delta
    if not np.isfinite(action).all():
        raise IsaacTeleopRuntimeError("Vision Pro packet contains non-finite values.")
    return action


def _load_env_cfg(task_package: str, env_cfg_class: str, *, num_envs: int) -> Any:
    import importlib

    module = importlib.import_module(task_package)
    cfg_cls = getattr(module, env_cfg_class, None)
    if cfg_cls is None:
        raise IsaacTeleopRuntimeError(
            f"EnvCfg class '{env_cfg_class}' not found in task package '{task_package}'."
        )
    cfg = cfg_cls()
    if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
        cfg.scene.num_envs = int(num_envs)
    return cfg


def _load_isaac_lab_modules() -> Dict[str, Any]:
    try:
        from isaaclab.app import AppLauncher
        from isaaclab.envs import ManagerBasedEnv
        from isaaclab_tasks.utils import parse_env_cfg
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise IsaacTeleopRuntimeError(
            "Isaac Lab runtime not available. Install/run inside an Isaac Lab environment to use "
            "record-teleop."
        ) from exc
    return {
        "AppLauncher": AppLauncher,
        "ManagerBasedEnv": ManagerBasedEnv,
        "parse_env_cfg": parse_env_cfg,
    }


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise IsaacTeleopRuntimeError("PyTorch is required inside the Isaac teleop runtime.") from exc
    return torch


def _flatten_obs(obs: Any, prefix: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    if isinstance(obs, dict):
        for key, value in obs.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten_obs(value, next_prefix))
        return items
    items[prefix or "obs"] = obs
    return items


def _matches_requested_key(candidate: str, requested_keys: Sequence[str]) -> bool:
    token = candidate.strip().lower()
    for requested in requested_keys:
        req = str(requested).strip().lower()
        if token == req or token.endswith(f".{req}") or token.endswith(f"/{req}"):
            return True
    return False


def _looks_like_camera_key(key: str) -> bool:
    token = key.strip().lower()
    hints = ("rgb", "image", "camera", "wrist", "head", "left", "right", "front", "overhead")
    return any(hint in token for hint in hints)


def _canonical_camera_name(key: str) -> str:
    token = key.strip().lower()
    if "left" in token:
        return "left"
    if "right" in token:
        return "right"
    if "wrist" in token or "hand" in token:
        return "wrist"
    if "head" in token or "front" in token:
        return "head"
    if "overhead" in token:
        return "overhead"
    parts = [part for part in token.replace("/", ".").split(".") if part]
    return parts[-1] if parts else "camera"


def _sorted_camera_keys(frames: Mapping[str, np.ndarray]) -> List[str]:
    preferred = ["wrist", "head", "left", "right", "overhead"]
    present = list(frames.keys())
    ordered = [key for key in preferred if key in present]
    ordered.extend(sorted(key for key in present if key not in ordered))
    return ordered


def _to_rgb_frame(value: Any) -> np.ndarray | None:
    arr = _to_numpy(value)
    if arr is None:
        return None
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        return None
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0 if np.max(arr) <= 1.0 else np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return arr


def _to_numeric_vector(value: Any) -> np.ndarray | None:
    arr = _to_numpy(value)
    if arr is None:
        return None
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > 2:
        return None
    if arr.ndim == 2:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    try:
        arr = np.asarray(arr, dtype=np.float32)
    except Exception:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr.reshape(-1)


def _to_numpy(value: Any) -> np.ndarray | None:
    try:
        import torch
    except Exception:  # pragma: no cover - no torch at import time
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


def _any_done(dones: Any) -> bool:
    if dones is None:
        return False
    arr = _to_numpy(dones)
    if arr is None:
        return False
    return bool(np.any(arr))


def _load_spacemouse_reader():
    errors: List[Exception] = []
    try:
        import pyspacemouse  # type: ignore

        if not pyspacemouse.open():
            raise IsaacTeleopRuntimeError("pyspacemouse could not open a SpaceMouse device.")

        class _Reader:
            def read(self):
                return pyspacemouse.read()

            def close(self):
                close = getattr(pyspacemouse, "close", None)
                if callable(close):
                    close()

        return _Reader()
    except Exception as exc:  # pragma: no cover - optional dependency
        errors.append(exc)
    try:
        import spacemouse  # type: ignore

        if hasattr(spacemouse, "open") and not spacemouse.open():
            raise IsaacTeleopRuntimeError("spacemouse backend could not open a SpaceMouse device.")

        class _Reader:
            def read(self):
                return spacemouse.read()

            def close(self):
                close = getattr(spacemouse, "close", None)
                if callable(close):
                    close()

        return _Reader()
    except Exception as exc:  # pragma: no cover - optional dependency
        errors.append(exc)
    raise IsaacTeleopRuntimeError(
        "SpaceMouse support requires a compatible Python backend such as pyspacemouse. "
        f"Import/open attempts failed: {', '.join(type(err).__name__ for err in errors)}"
    )


def _vision_pro_packet_stream(
    config: TeleopRecorderConfig,
    *,
    packet_log_path: Path,
) -> Iterable[Dict[str, Any]]:
    """Yield control packets from a local TCP JSON-lines bridge."""
    packet_log_handle = None
    if bool(config.bridge_packet_log_enabled):
        packet_log_handle = packet_log_path.open("w", encoding="utf-8")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((str(config.bridge_host), int(config.bridge_port)))
    server.listen(1)
    server.settimeout(float(config.bridge_connect_timeout_s))
    try:
        try:
            conn, addr = server.accept()
        except socket.timeout as exc:
            raise IsaacTeleopRuntimeError(
                "Vision Pro bridge connect timeout expired before any client connected. "
                f"Expected a client on {config.bridge_host}:{config.bridge_port}."
            ) from exc
        print(f"Vision Pro bridge connected from {addr[0]}:{addr[1]}", flush=True)
        with conn:
            conn.settimeout(float(config.bridge_idle_timeout_s))
            buffer = b""
            while True:
                try:
                    chunk = conn.recv(65536)
                except socket.timeout as exc:
                    raise IsaacTeleopRuntimeError(
                        "Vision Pro bridge idle timeout expired while waiting for control packets."
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
                        raise IsaacTeleopRuntimeError(
                            f"Invalid Vision Pro bridge packet: {line[:120]!r}"
                        ) from exc
                    if packet_log_handle is not None:
                        packet_log_handle.write(json.dumps(packet) + "\n")
                        packet_log_handle.flush()
                    if not isinstance(packet, dict):
                        raise IsaacTeleopRuntimeError("Vision Pro bridge packet must be a JSON object.")
                    yield packet
    finally:
        if packet_log_handle is not None:
            packet_log_handle.close()
        server.close()
