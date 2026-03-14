"""Wrapper that translates Blueprint runtime requests into NeoVerse inference calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .optional_dependencies import require_optional_dependency


_PREDEFINED_TRAJECTORIES = {
    "head_rgb": "static",
    "head": "static",
    "wrist": "move_right",
    "context": "orbit_left",
    "overhead": "tilt_down",
}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _require_cv2():
    return require_optional_dependency("cv2", extra="vision", purpose="NeoVerse runner image/video IO")


def _neoverse_repo_root() -> Path:
    value = str(os.environ.get("NEOVERSE_REPO_ROOT") or "").strip()
    if not value:
        raise RuntimeError("NEOVERSE_REPO_ROOT is not configured.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"NeoVerse repo root does not exist: {path}")
    inference_path = path / "inference.py"
    if not inference_path.is_file():
        raise RuntimeError(f"NeoVerse inference.py not found under {path}")
    return path


def _neoverse_python_bin(repo_root: Path) -> str:
    explicit = str(os.environ.get("NEOVERSE_PYTHON_BIN") or "").strip()
    if explicit:
        return explicit
    candidate = repo_root / ".venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def _model_root() -> Path:
    value = str(os.environ.get("NEOVERSE_MODEL_ROOT") or "").strip()
    if not value:
        raise RuntimeError("NEOVERSE_MODEL_ROOT is not configured.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"NeoVerse model root does not exist: {path}")
    return path


def _checkpoint_path(default_model_root: Path) -> Path:
    value = str(os.environ.get("NEOVERSE_CHECKPOINT_PATH") or "").strip()
    if value:
        path = Path(value).expanduser().resolve()
    else:
        path = default_model_root / "NeoVerse" / "reconstructor.ckpt"
    if not path.is_file():
        raise RuntimeError(f"NeoVerse checkpoint does not exist: {path}")
    return path


def _workspace_manifest(request_payload: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(request_payload.get("workspace_manifest_path") or "")).expanduser().resolve()
    if not path.is_file():
        return {}
    return _read_json(path)


def _snapshot(request_payload: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(request_payload.get("snapshot_path") or "")).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Snapshot path does not exist: {path}")
    return _read_json(path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _existing_file_candidates(values: Iterable[Any]) -> list[Path]:
    rows: list[Path] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        candidate = Path(text).expanduser().resolve()
        if candidate.is_file() and candidate not in rows:
            rows.append(candidate)
    return rows


def _conditioning_input_path(request_payload: Mapping[str, Any], workspace_manifest: Mapping[str, Any]) -> Path:
    registration = dict(workspace_manifest.get("registration") or {})
    spec = dict(workspace_manifest.get("spec") or {})
    conditioning = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
    local_paths = dict(conditioning.get("local_paths") or {}) if isinstance(conditioning.get("local_paths"), Mapping) else {}
    candidates = _existing_file_candidates(
        [
            registration.get("conditioning_source_path"),
            local_paths.get("raw_video_path"),
            local_paths.get("keyframe_path"),
        ]
    )
    if candidates:
        return candidates[0]
    base_frame_path = Path(str(request_payload.get("base_frame_path") or "")).expanduser().resolve()
    if not base_frame_path.is_file():
        raise RuntimeError(f"Base frame path does not exist: {base_frame_path}")
    return base_frame_path


def _prompt(snapshot: Mapping[str, Any]) -> str:
    presentation_config = dict(snapshot.get("presentation_config") or {})
    return str(presentation_config.get("prompt") or "A site-specific NeoVerse validation render.")


def _trajectory_for_camera(snapshot: Mapping[str, Any], camera: Mapping[str, Any]) -> str:
    presentation_config = dict(snapshot.get("presentation_config") or {})
    trajectory = presentation_config.get("trajectory")
    if isinstance(trajectory, Mapping):
        named = str(trajectory.get("trajectory") or "").strip()
        if named:
            return named
    camera_id = str(camera.get("id") or camera.get("cameraId") or "").strip().lower()
    role = str(camera.get("role") or "").strip().lower()
    for key, value in _PREDEFINED_TRAJECTORIES.items():
        if key in camera_id or key == role:
            return value
    return "static"


def _output_video_path(output_dir: Path, camera_id: str) -> Path:
    return output_dir / f"{camera_id}.mp4"


def _output_frame_path(output_dir: Path, camera_id: str) -> Path:
    return output_dir / f"{camera_id}-frame0.png"


def _attempt_log_paths(output_dir: Path, camera_id: str, reconstructor_device: str) -> tuple[Path, Path]:
    suffix = reconstructor_device.replace(":", "_")
    return (
        output_dir / f"{camera_id}-{suffix}.stdout.log",
        output_dir / f"{camera_id}-{suffix}.stderr.log",
    )


def _extract_first_frame(video_path: Path, frame_path: Path) -> None:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"NeoVerse output video is unreadable: {video_path}")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), frame)


def _normalized_device() -> str:
    raw = str(os.environ.get("NEOVERSE_DEVICE") or "cuda:0").strip() or "cuda:0"
    if raw == "cuda":
        return "cuda:0"
    return raw


def _normalized_thread_limit(raw_value: str | None) -> str:
    try:
        limit = int(str(raw_value or "").strip() or "1")
    except ValueError:
        limit = 1
    return str(max(1, limit))


def _thread_limited_env(base_env: Mapping[str, str]) -> Dict[str, str]:
    env = dict(base_env)
    thread_limit = _normalized_thread_limit(env.get("NEOVERSE_BLAS_NUM_THREADS"))
    for key in _THREAD_LIMIT_ENV_VARS:
        env.setdefault(key, thread_limit)
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    return env


def _reconstructor_devices() -> list[str]:
    preferred = _normalized_device()
    return [preferred, "cpu"] if preferred != "cpu" else ["cpu"]


def run_request(request_path: Path, response_path: Path) -> Dict[str, Any]:
    request_payload = _read_json(request_path)
    repo_root = _neoverse_repo_root()
    python_bin = _neoverse_python_bin(repo_root)
    python_bin_dir = str(Path(python_bin).expanduser().resolve().parent)
    model_root = _model_root()
    checkpoint_path = _checkpoint_path(model_root)
    workspace_manifest = _workspace_manifest(request_payload)
    snapshot = _snapshot(request_payload)
    input_path = _conditioning_input_path(request_payload, workspace_manifest)
    output_dir = Path(str(request_payload.get("output_dir") or "")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = [dict(item) for item in request_payload.get("cameras", []) if isinstance(item, Mapping)]
    if not cameras:
        cameras = [{"id": "head_rgb", "role": "head"}]
    low_vram = str(os.environ.get("NEOVERSE_LOW_VRAM") or "").strip().lower() in {"1", "true", "yes", "on"}
    project_root = _project_root()

    frame_rows: list[Dict[str, Any]] = []
    for camera in cameras:
        camera_id = str(camera.get("id") or camera.get("cameraId") or "head_rgb").strip() or "head_rgb"
        output_video = _output_video_path(output_dir, camera_id)
        output_frame = _output_frame_path(output_dir, camera_id)
        trajectory = _trajectory_for_camera(snapshot, camera)
        command = [
            python_bin,
            "-m",
            "blueprint_validation.neoverse_mixed_precision_inference",
            "--input_path",
            str(input_path),
            "--trajectory",
            trajectory,
            "--prompt",
            _prompt(snapshot),
            "--output_path",
            str(output_video),
            "--model_path",
            str(model_root),
            "--reconstructor_path",
            str(checkpoint_path),
            "--device",
            _normalized_device(),
        ]
        if input_path.suffix.lower() in _IMAGE_SUFFIXES:
            command.append("--static_scene")
        if low_vram:
            command.append("--low_vram")
        env = _thread_limited_env(os.environ)
        path_entries = [python_bin_dir]
        existing_path = str(env.get("PATH") or "").strip()
        if existing_path:
            path_entries.append(existing_path)
        env["PATH"] = os.pathsep.join(path_entries)
        py_entries = [str(project_root), str(repo_root)]
        existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
        if existing_pythonpath:
            py_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(py_entries)
        attempt_errors: list[str] = []
        completed = None
        for reconstructor_device in _reconstructor_devices():
            attempt_env = dict(env)
            attempt_env["NEOVERSE_RUNTIME_RECONSTRUCTOR_DEVICE"] = reconstructor_device
            completed = subprocess.run(command, capture_output=True, text=True, check=False, env=attempt_env)
            stdout_log_path, stderr_log_path = _attempt_log_paths(output_dir, camera_id, reconstructor_device)
            stdout_log_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_log_path.write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode == 0:
                break
            stderr = (completed.stderr or completed.stdout or "").strip()
            attempt_errors.append(
                f"{reconstructor_device}: {stderr[:500]} [logs: {stdout_log_path.name}, {stderr_log_path.name}]"
            )
        if completed is None or completed.returncode != 0:
            raise RuntimeError(
                f"NeoVerse inference failed for {camera_id}: {' | '.join(attempt_errors)}"
            )
        if not output_video.is_file():
            raise RuntimeError(f"NeoVerse output video missing for {camera_id}: {output_video}")
        _extract_first_frame(output_video, output_frame)
        frame_rows.append(
            {
                "cameraId": camera_id,
                "path": str(output_frame),
                "video_path": str(output_video),
                "trajectory": trajectory,
            }
        )

    response = {
        "camera_frames": frame_rows,
        "quality_flags": {"presentation_quality": "neoverse_generated"},
        "protected_region_violations": [],
        "debug_artifacts": {
            "request_path": str(request_path),
            "neoverse_repo_root": str(repo_root),
            "input_path": str(input_path),
            "python_bin": python_bin,
        },
    }
    _write_json(response_path, response)
    return response


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 2:
        print("Usage: python -m blueprint_validation.neoverse_runner_wrapper <request_json> <response_json>", file=sys.stderr)
        return 2
    request_path = Path(args[0]).expanduser().resolve()
    response_path = Path(args[1]).expanduser().resolve()
    run_request(request_path, response_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
