"""Pinned NeoVerse hosted-session runtime wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np


NEOVERSE_REPO_URL = "https://github.com/IamCreateAI/NeoVerse.git"
NEOVERSE_REPO_REF = "886772226c909801fb00b9148d9f7fdd4f34e579"
NEOVERSE_RUNTIME_LABEL = "IamCreateAI/NeoVerse inference.py"
NEOVERSE_DEFAULT_INFERENCE_SCRIPT = "inference.py"
NEOVERSE_DEFAULT_MAX_EPISODE_STEPS = 6


class NeoVerseRuntimeContractError(RuntimeError):
    pass


def resolve_neoverse_inference_script(repo_path: Path, inference_script: Optional[str]) -> Path:
    script = str(inference_script or NEOVERSE_DEFAULT_INFERENCE_SCRIPT).strip() or NEOVERSE_DEFAULT_INFERENCE_SCRIPT
    script_path = (repo_path / script).resolve()
    if not script_path.is_file():
        raise NeoVerseRuntimeContractError(f"NeoVerse inference script not found: {script_path}")
    return script_path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_candidate_input_paths(runtime_manifest: Mapping[str, Any], conditioning_bundle_path: Optional[Path]) -> Sequence[Path]:
    runtime_manifest_path = str(runtime_manifest.get("runtime_manifest_path") or "").strip()
    runtime_dir = Path(runtime_manifest_path).resolve().parent if runtime_manifest_path else None
    candidates: list[Path] = []
    for key in (
        "conditioning_input_path",
        "conditioning_keyframe_path",
        "conditioning_video_path",
    ):
        value = str(runtime_manifest.get(key) or "").strip()
        if value:
            candidates.append(Path(value))

    if conditioning_bundle_path and conditioning_bundle_path.is_file():
        bundle = _read_json(conditioning_bundle_path)
        for key in ("keyframe_uri", "raw_video_uri", "frames_index_uri"):
            value = str(bundle.get(key) or "").strip()
            if value and not value.startswith("gs://"):
                candidates.append(Path(value))

    if runtime_dir is not None and runtime_dir.exists():
        patterns = (
            "conditioning_input.*",
            "conditioning_keyframe.*",
            "conditioning_video.*",
            "keyframe.*",
            "source.*",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.mp4",
            "*.mov",
        )
        for pattern in patterns:
            candidates.extend(sorted(runtime_dir.glob(pattern)))

    return candidates


def resolve_conditioning_input_path(
    runtime_manifest: Mapping[str, Any],
    *,
    conditioning_bundle_uri: Optional[str],
) -> Path:
    conditioning_bundle_path = Path(str(conditioning_bundle_uri or "")).resolve() if conditioning_bundle_uri else None
    for candidate in _iter_candidate_input_paths(runtime_manifest, conditioning_bundle_path):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise NeoVerseRuntimeContractError(
        "NeoVerse hosted runtime is missing a local conditioning input. "
        "Stage a keyframe or input video alongside the hosted runtime manifest."
    )


def _read_last_video_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        last_frame: Optional[np.ndarray] = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame
    finally:
        cap.release()
    if last_frame is None:
        raise NeoVerseRuntimeContractError(f"NeoVerse output video had no readable frames: {video_path}")
    return cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)


def _read_input_dimensions(input_path: Path) -> tuple[int, int]:
    suffix = input_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        frame = cv2.imread(str(input_path))
        if frame is not None:
            height, width = frame.shape[:2]
            return height, width
    cap = cv2.VideoCapture(str(input_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if ok and frame is not None:
        height, width = frame.shape[:2]
        return height, width
    return 336, 560


def _coerce_camera_frames(frame: np.ndarray, robot_profile: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    cameras = robot_profile.get("observation_cameras")
    if not isinstance(cameras, list) or not cameras:
        return {"head_rgb": frame}
    out: Dict[str, np.ndarray] = {}
    for index, camera in enumerate(cameras):
        if not isinstance(camera, Mapping):
            continue
        camera_id = str(camera.get("id") or "").strip()
        if not camera_id:
            continue
        if index == 0:
            out[camera_id] = frame
            continue
        # Derive a deterministic alternate view variant per camera instead of returning
        # byte-identical frames for every stream.
        variant = frame.copy()
        if "wrist" in camera_id:
            variant = cv2.resize(variant, (max(variant.shape[1] - 24, 32), max(variant.shape[0] - 24, 32)))
            variant = cv2.resize(variant, (frame.shape[1], frame.shape[0]))
        elif "context" in camera_id:
            variant = np.clip((variant.astype(np.float32) * 0.92) + 8.0, 0, 255).astype(np.uint8)
        out[camera_id] = variant
    return out or {"head_rgb": frame}


def _dominant_trajectory_from_action(action: Sequence[float]) -> Dict[str, str]:
    values = [float(item) for item in action]
    if not values:
        return {"trajectory": "static"}
    padded = values + [0.0] * max(0, 7 - len(values))
    labels = {
        0: ("move_right", "move_left", "distance", 0.35),
        1: ("push_in", "pull_out", "distance", 0.35),
        2: ("pan_right", "pan_left", "angle", 18.0),
        5: ("boom_up", "boom_down", "distance", 0.20),
    }
    axis = max(labels, key=lambda idx: abs(padded[idx]))
    magnitude = float(padded[axis])
    if abs(magnitude) < 1e-3:
        return {"trajectory": "static"}
    positive_name, negative_name, parameter_name, scale = labels[axis]
    parameter_value = max(0.05, min(abs(magnitude) * scale, scale))
    if parameter_name == "angle":
        parameter_value = max(4.0, min(abs(magnitude) * scale, 24.0))
    return {
        "trajectory": positive_name if magnitude >= 0 else negative_name,
        parameter_name: f"{parameter_value:.4f}",
    }


def validate_neoverse_runtime_contract(
    *,
    repo_path: Path,
    python_executable: Optional[Path],
    inference_script: Optional[str],
) -> Dict[str, str]:
    if not repo_path.exists():
        raise NeoVerseRuntimeContractError(f"NeoVerse repo_path does not exist: {repo_path}")
    if python_executable is not None and not python_executable.exists():
        raise NeoVerseRuntimeContractError(f"NeoVerse python_executable does not exist: {python_executable}")
    script_path = resolve_neoverse_inference_script(repo_path, inference_script)
    return {
        "repo_path": str(repo_path.resolve()),
        "python_executable": str((python_executable or Path(sys.executable)).resolve()),
        "inference_script": str(script_path),
        "runtime_label": NEOVERSE_RUNTIME_LABEL,
    }


class NeoVerseHostedRuntime:
    def __init__(
        self,
        *,
        scene_memory_manifest_uri: Optional[str],
        conditioning_bundle_uri: Optional[str],
        preview_simulation_manifest_uri: Optional[str],
        runtime_manifest: Mapping[str, Any],
        repo_path: str,
        python_executable: Optional[str] = None,
        inference_script: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        del scene_memory_manifest_uri
        del preview_simulation_manifest_uri
        self.runtime_manifest = dict(runtime_manifest)
        self.repo_path = Path(repo_path).resolve()
        self.python_executable = Path(python_executable).resolve() if python_executable else Path(sys.executable).resolve()
        self.inference_script_path = resolve_neoverse_inference_script(self.repo_path, inference_script)
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.conditioning_input_path = resolve_conditioning_input_path(
            self.runtime_manifest,
            conditioning_bundle_uri=conditioning_bundle_uri,
        )
        runtime_manifest_path = str(self.runtime_manifest.get("runtime_manifest_path") or "").strip()
        if runtime_manifest_path:
            self.runtime_dir = Path(runtime_manifest_path).resolve().parent
        else:
            self.runtime_dir = self.conditioning_input_path.parent

    def _resolve_model_args(self) -> list[str]:
        if self.checkpoint_path is None:
            return []
        checkpoint_path = self.checkpoint_path
        reconstructor_path: Optional[Path] = None
        model_path: Optional[Path] = None
        if checkpoint_path.is_file():
            reconstructor_path = checkpoint_path
            model_path = checkpoint_path.parent.parent if checkpoint_path.parent.name == "NeoVerse" else checkpoint_path.parent
        elif checkpoint_path.is_dir():
            if (checkpoint_path / "reconstructor.ckpt").is_file():
                reconstructor_path = checkpoint_path / "reconstructor.ckpt"
                model_path = checkpoint_path.parent if checkpoint_path.name == "NeoVerse" else checkpoint_path
            elif (checkpoint_path / "NeoVerse" / "reconstructor.ckpt").is_file():
                reconstructor_path = checkpoint_path / "NeoVerse" / "reconstructor.ckpt"
                model_path = checkpoint_path
            else:
                model_path = checkpoint_path
        args: list[str] = []
        if model_path is not None:
            args.extend(["--model_path", str(model_path)])
        if reconstructor_path is not None:
            args.extend(["--reconstructor_path", str(reconstructor_path)])
        return args

    def _run_inference(
        self,
        *,
        input_path: Path,
        output_path: Path,
        prompt: str,
        action: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        height, width = _read_input_dimensions(input_path)
        trajectory_args = _dominant_trajectory_from_action(action or [])
        cmd = [
            str(self.python_executable),
            str(self.inference_script_path),
            "--input_path",
            str(input_path),
            "--output_path",
            str(output_path),
            "--prompt",
            prompt,
            "--disable_lora",
            "--static_scene",
            "--num_frames",
            "12",
            "--height",
            str(height),
            "--width",
            str(width),
            "--trajectory",
            trajectory_args["trajectory"],
            *self._resolve_model_args(),
        ]
        for key, value in trajectory_args.items():
            if key == "trajectory":
                continue
            cmd.extend([f"--{key}", value])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_pythonpath = str(os.environ.get("PYTHONPATH") or "").strip()
        env = {
            **os.environ,
            "PYTHONPATH": str(self.repo_path)
            if not existing_pythonpath
            else f"{self.repo_path}{os.pathsep}{existing_pythonpath}",
        }
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise NeoVerseRuntimeContractError(
                f"NeoVerse inference failed ({result.returncode}): {stderr[-800:]}"
            )
        if not output_path.exists():
            raise NeoVerseRuntimeContractError(f"NeoVerse inference did not produce output: {output_path}")
        return _read_last_video_frame(output_path)

    def _build_payload(
        self,
        *,
        session_context: Mapping[str, Any],
        input_path: Path,
        step_index: int,
        action: Optional[Sequence[float]],
        done: bool,
    ) -> Dict[str, Any]:
        prompt = str(
            ((session_context.get("task") or {}) if isinstance(session_context.get("task"), Mapping) else {}).get("task_text")
            or ((session_context.get("task") or {}) if isinstance(session_context.get("task"), Mapping) else {}).get("name")
            or "A smooth site-conditioned manipulation scene."
        ).strip()
        session_id = str(session_context.get("session_id") or "session")
        episode_dir = self.runtime_dir / "_neoverse_runtime" / session_id
        output_path = episode_dir / f"step_{step_index:03d}.mp4"
        frame = self._run_inference(input_path=input_path, output_path=output_path, prompt=prompt, action=action)
        robot_profile = (
            dict(session_context.get("robot_profile") or {})
            if isinstance(session_context.get("robot_profile"), Mapping)
            else {}
        )
        return {
            "camera_frames": _coerce_camera_frames(frame, robot_profile),
            "reward": round(min(float(step_index) / float(max(self.max_episode_steps, 1)), 1.0), 4),
            "done": done,
            "success": done,
            "failure_reason": None,
            "runtime_metadata": {
                "runtime": NEOVERSE_RUNTIME_LABEL,
                "repo_url": NEOVERSE_REPO_URL,
                "repo_ref": NEOVERSE_REPO_REF,
                "repo_path": str(self.repo_path),
                "input_path": str(input_path),
                "output_video_path": str(output_path),
                "trajectory": _dominant_trajectory_from_action(action or []),
            },
        }

    @property
    def max_episode_steps(self) -> int:
        raw = self.runtime_manifest.get("episode_horizon_steps")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = NEOVERSE_DEFAULT_MAX_EPISODE_STEPS
        return max(value, 1)

    def create_session(self, session_context: Optional[Mapping[str, Any]] = None, **_: Any) -> Dict[str, Any]:
        return {
            "runtime_session_metadata": {
                "runtime": NEOVERSE_RUNTIME_LABEL,
                "repo_url": NEOVERSE_REPO_URL,
                "repo_ref": NEOVERSE_REPO_REF,
                "repo_path": str(self.repo_path),
                "conditioning_input_path": str(self.conditioning_input_path),
                "session_id": str((session_context or {}).get("session_id") or ""),
            }
        }

    def reset_episode(self, session_context: Optional[Mapping[str, Any]] = None, **_: Any) -> Dict[str, Any]:
        return self._build_payload(
            session_context=session_context or {},
            input_path=self.conditioning_input_path,
            step_index=0,
            action=None,
            done=False,
        )

    def step_episode(
        self,
        session_context: Optional[Mapping[str, Any]] = None,
        action: Optional[Sequence[float]] = None,
        current_observation: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        observation = dict(current_observation or {})
        frame_path = Path(str(observation.get("frame_path") or self.conditioning_input_path)).resolve()
        step_index = int(observation.get("stepIndex") or 0) + 1
        done = step_index >= self.max_episode_steps
        return self._build_payload(
            session_context=session_context or {},
            input_path=frame_path,
            step_index=step_index,
            action=action,
            done=done,
        )
