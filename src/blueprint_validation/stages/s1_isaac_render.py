"""Stage 1: Isaac-backed scripted clip generation from a scene package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from ..common import StageResult, write_json
from ..config import FacilityConfig, ValidationConfig
from ..teleop.contracts import load_and_validate_scene_package
from ..teleop.runtime import (
    IsaacTeleopRuntimeError,
    load_scene_env,
    _resolve_env_action_dim,
    _sorted_camera_keys,
    extract_camera_frames,
)
from ..video_io import ensure_h264_video, open_mp4_writer
from .base import PipelineStage
from .render_backend import (
    active_render_backend,
    resolved_scene_package_path,
    unsafe_scene_package_imports_enabled,
)


class IsaacRenderStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1_isaac_render"

    @property
    def description(self) -> str:
        return "Render scripted Stage-1 clips from an Isaac scene package"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        if active_render_backend(config, facility, previous_results) != "isaac_scene":
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0.0,
                detail="Render backend resolved to gsplat; skipping Isaac Stage-1 render.",
            )

        scene_root = resolved_scene_package_path(facility, previous_results)
        if scene_root is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0.0,
                detail=(
                    "render.backend resolved to isaac_scene but no validated scene package is "
                    "available. Run s0a_scene_package first or configure facility.scene_package_path."
                ),
            )

        payload = load_and_validate_scene_package(scene_root)
        if not bool(payload.get("has_isaac_lab", False)):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0.0,
                detail=f"Scene package does not include an isaac_lab package: {scene_root}",
            )

        render_dir = work_dir / "isaac_renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = render_dir / "render_manifest.json"

        try:
            manifest_entries = _render_scripted_isaac_clips(
                config=config,
                scene_root=scene_root,
                render_dir=render_dir,
            )
        except (RuntimeError, IsaacTeleopRuntimeError) as exc:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0.0,
                detail=str(exc),
            )

        if not manifest_entries:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0.0,
                detail=(
                    "Isaac Stage-1 render produced no clips. Ensure the scene package env exposes "
                    "at least one RGB camera observation."
                ),
            )

        task_payload = _load_task_payload(scene_root)
        manifest = {
            "facility": facility.name,
            "scene_package_path": str(scene_root),
            "render_backend": "isaac_scene",
            "num_clips": len(manifest_entries),
            "scene_id": str(task_payload.get("scene_id", "") or scene_root.name),
            "clips": manifest_entries,
        }
        write_json(manifest, manifest_path)
        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0.0,
            outputs={
                "render_dir": str(render_dir),
                "manifest_path": str(manifest_path),
                "scene_package_path": str(scene_root),
                "num_clips": len(manifest_entries),
            },
            metrics={
                "num_clips": len(manifest_entries),
                "total_frames": sum(int(entry.get("num_frames", 0) or 0) for entry in manifest_entries),
                "render_backend": "isaac_scene",
                "scene_package_path": str(scene_root),
            },
        )


def _render_scripted_isaac_clips(
    *,
    config: ValidationConfig,
    scene_root: Path,
    render_dir: Path,
) -> List[dict]:
    if not unsafe_scene_package_imports_enabled():
        raise RuntimeError(
            "Isaac Stage-1 scene package rendering is disabled by default because it imports "
            "executable Python from the scene package. Set "
            "BLUEPRINT_UNSAFE_ALLOW_SCENE_PACKAGE_IMPORT=1 only for trusted scene packages."
        )
    loaded = load_scene_env(scene_root=scene_root, headless=True)
    try:
        env = loaded.env
        action_dim = int(_resolve_env_action_dim(env))
        num_frames = max(1, int(config.render.num_frames))
        num_rollouts = max(1, int(config.render.num_clips_per_path))
        fps = max(1, int(config.render.fps))
        manifest_entries: List[dict] = []

        for rollout_idx in range(num_rollouts):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            camera_buffers = _collect_clip_frames(
                env=env,
                initial_obs=obs,
                num_frames=num_frames,
                action_dim=action_dim,
                rollout_idx=rollout_idx,
            )
            if not camera_buffers:
                raise RuntimeError(
                    "No image-like observations found in Isaac env. Ensure the scene package "
                    "task exposes RGB camera observations."
                )
            for camera_id in _sorted_camera_keys(camera_buffers):
                frames = camera_buffers[camera_id]
                if not frames:
                    continue
                video_path = render_dir / f"{camera_id}_rollout_{rollout_idx:03d}.mp4"
                _write_clip_video(video_path, frames=frames, fps=fps)
                first = frames[0]
                manifest_entries.append(
                    {
                        "clip_name": f"{camera_id}_rollout_{rollout_idx:03d}",
                        "video_path": str(video_path),
                        "depth_video_path": "",
                        "fps": fps,
                        "num_frames": len(frames),
                        "resolution": [int(first.shape[0]), int(first.shape[1])],
                        "camera_id": camera_id,
                        "sim_backend": "isaac_lab",
                        "scene_package_path": str(scene_root),
                        "render_backend": "isaac_scene",
                    }
                )
        return manifest_entries
    finally:
        loaded.close()


def _collect_clip_frames(
    *,
    env: Any,
    initial_obs: Any,
    num_frames: int,
    action_dim: int,
    rollout_idx: int,
) -> Dict[str, List[np.ndarray]]:
    torch = _import_torch()
    obs = initial_obs
    camera_buffers: Dict[str, List[np.ndarray]] = {}
    for frame_idx in range(num_frames):
        frames = extract_camera_frames(obs, requested_keys=[])
        for camera_id, frame in frames.items():
            camera_buffers.setdefault(camera_id, []).append(np.asarray(frame, dtype=np.uint8))
        if frame_idx == num_frames - 1:
            break
        action = _scripted_action(frame_idx=frame_idx, rollout_idx=rollout_idx, action_dim=action_dim)
        step_result = env.step(torch.tensor(action[None, :], device=env.device, dtype=torch.float32))
        if not isinstance(step_result, tuple) or not step_result:
            raise RuntimeError("Unexpected Isaac env.step() contract during Stage-1 render.")
        obs = step_result[0]
    return camera_buffers


def _scripted_action(*, frame_idx: int, rollout_idx: int, action_dim: int) -> np.ndarray:
    action = np.zeros((int(action_dim),), dtype=np.float32)
    if int(action_dim) < 7:
        return action
    phase = (frame_idx + 1) * 0.2
    direction = -1.0 if rollout_idx % 2 else 1.0
    action[0] = 0.005 * direction
    action[1] = 0.004 * np.sin(phase)
    action[2] = 0.003 * np.cos(phase)
    action[5] = 0.02 * direction
    action[6] = 1.0 if ((frame_idx // 4) + rollout_idx) % 2 == 0 else -1.0
    return action


def _write_clip_video(path: Path, *, frames: List[np.ndarray], fps: int) -> None:
    first = np.asarray(frames[0], dtype=np.uint8)
    writer = open_mp4_writer(
        output_path=path,
        fps=float(fps),
        frame_size=(int(first.shape[1]), int(first.shape[0])),
        is_color=True,
    )
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency dependent
        writer.release()
        raise RuntimeError("OpenCV is required for Isaac Stage-1 video export.") from exc
    try:
        for frame in frames:
            arr = np.asarray(frame, dtype=np.uint8)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    ensure_h264_video(input_path=path, min_decoded_frames=max(1, len(frames)), replace_source=True)


def _load_task_payload(scene_root: Path) -> Mapping[str, object]:
    candidates = [
        scene_root / "geniesim" / "task_config.json",
        scene_root / "assets" / "scene_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                import json

                payload = json.loads(candidate.read_text())
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("PyTorch is required inside the Isaac render runtime.") from exc
    return torch
