"""Blueprint-owned PolaRiS evaluation runner and result normalization."""

from __future__ import annotations

import hashlib
import importlib
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

from ..common import write_json
from ..config import FacilityConfig, ValidationConfig
from ..teleop.runtime import (
    IsaacTeleopRuntimeError,
    _flatten_obs,
    _looks_like_camera_key,
    _resolve_env_action_dim,
    _to_rgb_frame,
    load_scene_env,
)
from ..video_io import ensure_h264_video, open_mp4_writer
from .openvla_client import normalize_openvla_action
from .runtime import PolarisSceneSpec, resolve_polaris_runtime, resolve_polaris_scene_spec
from .websocket_policy import WebsocketPolicyClient


@dataclass(frozen=True)
class CandidateEvalResult:
    label: str
    success_rate: float
    mean_progress: float
    num_rollouts: int
    report_path: Path
    csv_path: Path
    video_paths: List[str]


def run_polaris_comparison(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    frozen_model_name: str,
    frozen_checkpoint: Optional[Path],
    adapted_checkpoint: Path,
) -> dict[str, Any]:
    runtime = resolve_polaris_runtime(config)
    scene_spec = resolve_polaris_scene_spec(config, facility)
    output_dir = work_dir / "polaris_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scene_spec.primary_eligible and bool(config.eval_polaris.default_as_primary_gate):
        raise RuntimeError(
            "PolaRiS is configured as the primary gate, but the resolved scene handoff is not "
            f"primary-eligible ({scene_spec.detail})."
        )

    fake_backend = (
        os.environ.get("BLUEPRINT_POLARIS_FAKE_BACKEND", "0") == "1"
        or str(config.eval_polaris.policy_client or "").strip().lower() == "fake"
    )
    if fake_backend:
        frozen = _run_fake_candidate_eval(
            label="frozen_openvla",
            checkpoint_ref=str(frozen_checkpoint or frozen_model_name),
            output_dir=output_dir / "frozen_openvla",
            num_rollouts=int(config.eval_polaris.num_rollouts),
        )
        adapted = _run_fake_candidate_eval(
            label="adapted_openvla",
            checkpoint_ref=str(adapted_checkpoint),
            output_dir=output_dir / "adapted_openvla",
            num_rollouts=int(config.eval_polaris.num_rollouts),
        )
    else:
        if not runtime.runnable:
            raise RuntimeError("; ".join(runtime.issues))
        frozen = _run_live_candidate_eval(
            config=config,
            scene_spec=scene_spec,
            candidate_label="frozen_openvla",
            model_name=frozen_model_name,
            checkpoint_path=frozen_checkpoint,
            output_dir=output_dir / "frozen_openvla",
        )
        adapted = _run_live_candidate_eval(
            config=config,
            scene_spec=scene_spec,
            candidate_label="adapted_openvla",
            model_name=frozen_model_name,
            checkpoint_path=adapted_checkpoint,
            output_dir=output_dir / "adapted_openvla",
        )

    winner = _pick_winner(frozen, adapted)
    delta_vs_frozen = round(adapted.success_rate - frozen.success_rate, 6)
    normalized = {
        "gate_name": "polaris",
        "scene_mode": scene_spec.mode,
        "scene_detail": scene_spec.detail,
        "primary_eligible": bool(scene_spec.primary_eligible),
        "winner": winner,
        "frozen_openvla": {
            "success_rate": frozen.success_rate,
            "mean_progress": frozen.mean_progress,
            "num_rollouts": frozen.num_rollouts,
            "report_path": str(frozen.report_path),
            "csv_path": str(frozen.csv_path),
            "video_paths": frozen.video_paths,
        },
        "adapted_openvla": {
            "success_rate": adapted.success_rate,
            "mean_progress": adapted.mean_progress,
            "num_rollouts": adapted.num_rollouts,
            "report_path": str(adapted.report_path),
            "csv_path": str(adapted.csv_path),
            "video_paths": adapted.video_paths,
        },
        "delta_vs_frozen": delta_vs_frozen,
    }
    normalized_path = output_dir / "polaris_eval_summary.json"
    write_json(normalized, normalized_path)
    normalized["summary_path"] = str(normalized_path)
    return normalized


def _run_fake_candidate_eval(
    *,
    label: str,
    checkpoint_ref: str,
    output_dir: Path,
    num_rollouts: int,
) -> CandidateEvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.sha1(f"{label}:{checkpoint_ref}".encode("utf-8")).hexdigest()[:8], 16)
    success_rows: list[dict[str, Any]] = []
    successes = 0.0
    progresses = 0.0
    for episode in range(int(num_rollouts)):
        local = (seed + (episode * 2654435761)) & 0xFFFFFFFF
        success = 0.35 + ((local % 37) / 100.0)
        if "adapted" in label:
            success += 0.12
        success = min(max(success, 0.0), 0.98)
        progress = min(1.0, success + 0.08)
        successes += success
        progresses += progress
        success_rows.append(
            {
                "episode": episode,
                "success": round(success >= 0.5, 6),
                "success_score": round(success, 6),
                "progress": round(progress, 6),
            }
        )
    csv_path = output_dir / "eval_results.csv"
    csv_path.write_text(_rows_to_csv(success_rows))
    report_path = output_dir / "candidate_summary.json"
    result = CandidateEvalResult(
        label=label,
        success_rate=round(successes / num_rollouts, 6),
        mean_progress=round(progresses / num_rollouts, 6),
        num_rollouts=int(num_rollouts),
        report_path=report_path,
        csv_path=csv_path,
        video_paths=[],
    )
    write_json(
        {
            "label": label,
            "success_rate": result.success_rate,
            "mean_progress": result.mean_progress,
            "num_rollouts": result.num_rollouts,
            "backend": "fake",
        },
        report_path,
    )
    return result


def _run_live_candidate_eval(
    *,
    config: ValidationConfig,
    scene_spec: PolarisSceneSpec,
    candidate_label: str,
    model_name: str,
    checkpoint_path: Optional[Path],
    output_dir: Path,
) -> CandidateEvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    port = _reserve_free_port()
    server = _launch_policy_server(
        port=port,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=config.eval_polaris.device,
        observation_mode=config.eval_polaris.observation_mode,
    )
    try:
        client = WebsocketPolicyClient(host="127.0.0.1", port=port)
        rows, video_paths = _evaluate_scene_candidate(
            config=config,
            scene_spec=scene_spec,
            candidate_label=candidate_label,
            output_dir=output_dir,
            client=client,
        )
    finally:
        _terminate_process(server)
    if not rows:
        raise RuntimeError(f"No PolaRiS evaluation rows produced for {candidate_label}.")
    success_rate = float(np.mean([float(row.get("success_score", 0.0)) for row in rows]))
    mean_progress = float(np.mean([float(row.get("progress", 0.0)) for row in rows]))
    csv_path = output_dir / "eval_results.csv"
    csv_path.write_text(_rows_to_csv(rows))
    report_path = output_dir / "candidate_summary.json"
    result = CandidateEvalResult(
        label=candidate_label,
        success_rate=round(success_rate, 6),
        mean_progress=round(mean_progress, 6),
        num_rollouts=len(rows),
        report_path=report_path,
        csv_path=csv_path,
        video_paths=video_paths,
    )
    write_json(
        {
            "label": candidate_label,
            "success_rate": result.success_rate,
            "mean_progress": result.mean_progress,
            "num_rollouts": result.num_rollouts,
            "backend": scene_spec.mode,
            "scene_mode": scene_spec.mode,
        },
        report_path,
    )
    return result


def _evaluate_scene_candidate(
    *,
    config: ValidationConfig,
    scene_spec: PolarisSceneSpec,
    candidate_label: str,
    output_dir: Path,
    client: WebsocketPolicyClient,
) -> tuple[list[dict[str, Any]], list[str]]:
    if scene_spec.mode == "scene_package_bridge":
        return _evaluate_scene_package_bridge(
            config=config,
            scene_spec=scene_spec,
            candidate_label=candidate_label,
            output_dir=output_dir,
            client=client,
        )
    raise RuntimeError(
        f"Live PolaRiS evaluation for environment_mode={scene_spec.mode} is not available in "
        "the current runtime. Use a native bundle or set BLUEPRINT_POLARIS_FAKE_BACKEND=1."
    )


def _evaluate_scene_package_bridge(
    *,
    config: ValidationConfig,
    scene_spec: PolarisSceneSpec,
    candidate_label: str,
    output_dir: Path,
    client: WebsocketPolicyClient,
) -> tuple[list[dict[str, Any]], list[str]]:
    if scene_spec.scene_root is None:
        raise RuntimeError("scene_package_bridge requires scene_root")
    if os.environ.get("BLUEPRINT_UNSAFE_ALLOW_SCENE_PACKAGE_IMPORT", "0") != "1":
        raise RuntimeError(
            "scene_package_bridge is disabled by default because it imports executable Python "
            "from the scene package. Use native_bundle mode, or set "
            "BLUEPRINT_UNSAFE_ALLOW_SCENE_PACKAGE_IMPORT=1 only for trusted scene packages."
        )
    scene_root = scene_spec.scene_root
    loaded = load_scene_env(scene_root=scene_root, headless=True)
    env = None
    rows: list[dict[str, Any]] = []
    video_paths: list[str] = []
    try:
        env = loaded.env
        action_dim = int(_resolve_env_action_dim(env))
        for rollout_idx in range(int(config.eval_polaris.num_rollouts)):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            frames: list[np.ndarray] = []
            success_score = 0.0
            progress = 0.0
            for _step in range(int(config.eval_policy.max_steps_per_rollout)):
                frame = _extract_eval_frame(obs)
                if frame is not None:
                    frames.append(frame)
                request = {
                    "instruction": scene_spec.instruction,
                    "image": frame,
                }
                action = normalize_openvla_action(client.infer(request).get("action"))
                action = _bridge_action(action, target_dim=action_dim, action_mode=config.eval_polaris.action_mode)
                step_result = env.step(action.reshape(1, -1))
                if not isinstance(step_result, tuple) or len(step_result) < 4:
                    raise RuntimeError("Unexpected Isaac Lab step() return contract for scene_package_bridge")
                obs = step_result[0]
                info = step_result[-1] if len(step_result) >= 5 else {}
                success_score, progress = _extract_success_progress(env, info)
                if success_score >= 1.0:
                    break
            video_path = _write_rollout_video(output_dir / "videos", candidate_label, rollout_idx, frames)
            if video_path is not None:
                video_paths.append(str(video_path))
            rows.append(
                {
                    "episode": rollout_idx,
                    "success": round(success_score >= 1.0, 6),
                    "success_score": round(success_score, 6),
                    "progress": round(progress, 6),
                }
            )
    except IsaacTeleopRuntimeError as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        if env is not None:
            loaded.close()
    return rows, video_paths


def _load_scene_env_cfg(scene_root: Path) -> Any:
    module = importlib.import_module("isaac_lab")
    candidates = [
        getattr(module, name)
        for name in sorted(dir(module))
        if name.endswith("EnvCfg") and isinstance(getattr(module, name), type)
    ]
    if not candidates:
        raise RuntimeError(f"No *EnvCfg class found in {scene_root / 'isaac_lab'}")
    return candidates[0]()


def _extract_eval_frame(obs: Any) -> Optional[np.ndarray]:
    flat = _flatten_obs(obs)
    camera_items: list[np.ndarray] = []
    for key, value in flat.items():
        if not _looks_like_camera_key(key):
            continue
        frame = _to_rgb_frame(value)
        if frame is not None:
            camera_items.append(frame)
    if not camera_items:
        return None
    return camera_items[0]


def _bridge_action(action: np.ndarray, *, target_dim: int, action_mode: str) -> np.ndarray:
    if int(target_dim) == action.size:
        return action.astype(np.float32)
    mode = str(action_mode or "auto").strip().lower()
    if mode not in {"auto", "joint_position_bridge"}:
        raise RuntimeError(
            f"PolaRiS action mismatch: policy produced {action.size} dims, env expects {target_dim}."
        )
    if int(target_dim) == action.size + 1:
        bridged = np.zeros((int(target_dim),), dtype=np.float32)
        bridged[: action.size] = action
        bridged[-1] = 1.0 if float(action[-1]) > 0.0 else 0.0
        return bridged
    raise RuntimeError(
        f"PolaRiS action bridge cannot map policy action_dim={action.size} to env action_dim={target_dim}."
    )


def _extract_success_progress(env: Any, info: Any) -> tuple[float, float]:
    if isinstance(info, dict):
        rubric = info.get("rubric")
        if isinstance(rubric, dict):
            success = 1.0 if bool(rubric.get("success", False)) else 0.0
            progress = float(rubric.get("progress", success) or success)
            return success, progress
    task_success = getattr(env, "task_success", None)
    if task_success is not None:
        try:
            value = task_success[0]
        except Exception:
            value = task_success
        success = 1.0 if bool(value) else 0.0
        return success, success
    return 0.0, 0.0


def _write_rollout_video(
    video_root: Path,
    candidate_label: str,
    rollout_idx: int,
    frames: Iterable[np.ndarray],
) -> Optional[Path]:
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames if frame is not None]
    if not frames:
        return None
    video_root.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    height, width = first.shape[:2]
    output_path = video_root / f"{candidate_label}_{rollout_idx:03d}.mp4"
    writer = open_mp4_writer(output_path=output_path, fps=10.0, frame_size=(width, height), is_color=True)
    import cv2

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    checked = ensure_h264_video(output_path, min_decoded_frames=max(1, len(frames)), replace_source=True)
    return checked.path


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_policy_server(
    *,
    port: int,
    model_name: str,
    checkpoint_path: Optional[Path],
    device: str,
    observation_mode: str,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "blueprint_validation.polaris.policy_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-name",
        str(model_name),
        "--device",
        str(device),
        "--observation-mode",
        str(observation_mode),
    ]
    if checkpoint_path is not None:
        cmd.extend(["--checkpoint-path", str(checkpoint_path)])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 120.0
    lines: list[str] = []
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout is not None else ""
        if line:
            lines.append(line.rstrip())
            if "POLARIS_OPENVLA_SERVER_READY" in line:
                return proc
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    _terminate_process(proc)
    raise RuntimeError("Timed out starting PolaRiS OpenVLA policy server. " + " | ".join(lines[-10:]))


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def _pick_winner(frozen: CandidateEvalResult, adapted: CandidateEvalResult) -> str:
    if adapted.success_rate > frozen.success_rate:
        return "adapted_openvla"
    if adapted.success_rate < frozen.success_rate:
        return "frozen_openvla"
    if adapted.mean_progress > frozen.mean_progress:
        return "adapted_openvla"
    return "frozen_openvla"


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        parts = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                parts.append(f"{value:.6f}")
            else:
                parts.append(str(value))
        lines.append(",".join(parts))
    return "\n".join(lines) + "\n"
