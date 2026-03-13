"""Launch the upstream NeoVerse Gradio UI with optional site-world preloading."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _infer_scene_type(path: Path) -> str:
    return "General scene" if path.suffix.lower() in VIDEO_SUFFIXES else "Static scene"


def _discover_capture_media(registration_path: Path) -> tuple[Optional[Path], Optional[str]]:
    capture_root = registration_path.expanduser().resolve().parents[2]
    raw_dir = capture_root / "raw"
    if not raw_dir.is_dir():
        return None, None
    preferred_names = (
        "walkthrough.mov",
        "walkthrough.mp4",
        "raw_video.mov",
        "raw_video.mp4",
        "keyframe.png",
        "keyframe.jpg",
        "keyframe.jpeg",
    )
    for name in preferred_names:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate, _infer_scene_type(candidate)
    for child in sorted(raw_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES.union(IMAGE_SUFFIXES):
            return child, _infer_scene_type(child)
    return None, None


def _resolve_preload_input(
    *,
    registration_path: Optional[Path],
    explicit_input_path: Optional[Path],
) -> tuple[Optional[Path], Optional[str]]:
    if explicit_input_path is not None:
        resolved = explicit_input_path.expanduser().resolve()
        if not resolved.is_file():
            raise RuntimeError(f"Preload input path does not exist: {resolved}")
        return resolved, _infer_scene_type(resolved)
    if registration_path is None:
        return None, None
    resolved_registration = registration_path.expanduser().resolve()
    if not resolved_registration.is_file():
        raise RuntimeError(f"Site-world registration path does not exist: {resolved_registration}")
    registration = json.loads(resolved_registration.read_text(encoding="utf-8"))
    if not isinstance(registration, dict):
        raise RuntimeError(f"Site-world registration is not a JSON object: {resolved_registration}")
    for key in ("conditioning_source_path", "base_frame_path"):
        candidate = Path(str(registration.get(key) or "")).expanduser().resolve()
        if candidate.is_file():
            return candidate, _infer_scene_type(candidate)
    return _discover_capture_media(resolved_registration)


def _import_neoverse_app(repo_root: Path, *, low_vram: bool, reconstructor_path: Optional[str]) -> Any:
    app_path = repo_root / "app.py"
    if not app_path.is_file():
        raise RuntimeError(f"NeoVerse app.py not found under {repo_root}")
    old_cwd = Path.cwd()
    old_argv = list(sys.argv)
    old_sys_path = list(sys.path)
    argv = [str(app_path)]
    if low_vram:
        argv.append("--low_vram")
    if reconstructor_path:
        argv.extend(["--reconstructor_path", reconstructor_path])
    try:
        os.chdir(repo_root)
        sys.argv = argv
        sys.path.insert(0, str(repo_root))
        spec = importlib.util.spec_from_file_location("blueprint_validation._neoverse_app", app_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import NeoVerse app from {app_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path[:] = old_sys_path


def _attach_site_world_preload(module: Any, *, preload_path: Path, scene_type: str) -> None:
    def _preload():
        state, pil_images, reconstruct_update = module.handle_upload([str(preload_path)], scene_type)
        state, glb_path, preview_update = module.reconstruct(state)
        return state, pil_images, reconstruct_update, scene_type, glb_path, preview_update

    module.demo.load(
        fn=_preload,
        inputs=[],
        outputs=[
            module.app_state,
            module.image_gallery,
            module.reconstruct_btn,
            module.scene_type,
            module.model3d,
            module.preview_btn,
        ],
    )


def _write_site_world_example(repo_root: Path, *, preload_path: Path, scene_type: str) -> None:
    examples_dir = repo_root / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = examples_dir / "gallery.json"
    example = {
        "name": "Blueprint Site-World Demo",
        "file": str(preload_path),
        "scene_type": scene_type,
        "camera_motion": "static",
        "angle": 0,
        "distance": 0,
        "orbit_radius": 0,
        "mode": "relative",
        "zoom_ratio": 1.0,
        "alpha_threshold": 1.0,
        "use_first_frame": True,
        "traj_file": None,
    }
    payload = [example]
    if gallery_path.is_file():
        try:
            existing = json.loads(gallery_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
        if isinstance(existing, list):
            payload.extend(
                item
                for item in existing
                if isinstance(item, dict) and str(item.get("name") or "") != example["name"]
            )
    gallery_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch NeoVerse Gradio with optional site-world preload.")
    parser.add_argument("--repo-root", default=str(Path("external/NeoVerse").resolve()))
    parser.add_argument("--site-world-registration", default="")
    parser.add_argument("--preload-input-path", default="")
    parser.add_argument("--scene-type", default="")
    parser.add_argument("--host", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--reconstructor-path", default=os.getenv("NEOVERSE_GRADIO_RECONSTRUCTOR_PATH", ""))
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(str(args.repo_root)).expanduser().resolve()
    registration_path = Path(args.site_world_registration).expanduser().resolve() if str(args.site_world_registration).strip() else None
    explicit_input_path = Path(args.preload_input_path).expanduser().resolve() if str(args.preload_input_path).strip() else None

    preload_path, inferred_scene_type = _resolve_preload_input(
        registration_path=registration_path,
        explicit_input_path=explicit_input_path,
    )
    scene_type = str(args.scene_type).strip() or inferred_scene_type or "General scene"
    if preload_path is not None:
        _write_site_world_example(repo_root, preload_path=preload_path, scene_type=scene_type)
    module = _import_neoverse_app(
        repo_root,
        low_vram=bool(args.low_vram),
        reconstructor_path=str(args.reconstructor_path).strip() or None,
    )
    os.chdir(repo_root)
    module.demo.queue(max_size=5).launch(
        server_name=str(args.host),
        server_port=int(args.port),
        show_error=True,
        share=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
