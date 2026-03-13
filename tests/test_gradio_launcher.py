from __future__ import annotations

import json
from pathlib import Path

from blueprint_validation.neoverse_gradio_launcher import (
    _discover_capture_media,
    _infer_scene_type,
    _resolve_preload_input,
    _write_site_world_example,
)


def test_infer_scene_type_uses_video_suffixes() -> None:
    assert _infer_scene_type(Path("clip.mp4")) == "General scene"
    assert _infer_scene_type(Path("frame.png")) == "Static scene"


def test_resolve_preload_input_prefers_explicit_input(tmp_path: Path) -> None:
    explicit = tmp_path / "input.mp4"
    explicit.write_bytes(b"video")

    preload_path, scene_type = _resolve_preload_input(
        registration_path=None,
        explicit_input_path=explicit,
    )

    assert preload_path == explicit.resolve()
    assert scene_type == "General scene"


def test_resolve_preload_input_uses_registration_conditioning_source(tmp_path: Path) -> None:
    conditioning = tmp_path / "conditioning.mp4"
    conditioning.write_bytes(b"video")
    spec_path = tmp_path / "site_world_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_world_id": "siteworld-1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "canonical_package_version": "pkg-v1",
            }
        ),
        encoding="utf-8",
    )
    registration_path = tmp_path / "site_world_registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "site_world_id": "siteworld-1",
                "scene_id": "scene-1",
                "capture_id": "capture-1",
                "spec_path": str(spec_path),
                "conditioning_source_path": str(conditioning),
            }
        ),
        encoding="utf-8",
    )

    preload_path, scene_type = _resolve_preload_input(
        registration_path=registration_path,
        explicit_input_path=None,
    )

    assert preload_path == conditioning.resolve()
    assert scene_type == "General scene"


def test_write_site_world_example_puts_preloaded_demo_first(tmp_path: Path) -> None:
    repo_root = tmp_path / "NeoVerse"
    preload = tmp_path / "conditioning.mp4"
    preload.write_bytes(b"video")
    gallery_dir = repo_root / "examples"
    gallery_dir.mkdir(parents=True)
    (gallery_dir / "gallery.json").write_text(
        json.dumps([{"name": "Other example", "file": "other.mp4"}]),
        encoding="utf-8",
    )

    _write_site_world_example(repo_root, preload_path=preload, scene_type="General scene")

    gallery = json.loads((gallery_dir / "gallery.json").read_text(encoding="utf-8"))
    assert gallery[0]["name"] == "Blueprint Site-World Demo"
    assert gallery[0]["file"] == str(preload)
    assert gallery[0]["scene_type"] == "General scene"
    assert gallery[1]["name"] == "Other example"


def test_discover_capture_media_falls_back_to_staged_walkthrough(tmp_path: Path) -> None:
    registration_path = tmp_path / "captures" / "capture-1" / "pipeline" / "evaluation_prep" / "site_world_registration.json"
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text("{}", encoding="utf-8")
    walkthrough = registration_path.parents[2] / "raw" / "walkthrough.mov"
    walkthrough.parent.mkdir(parents=True)
    walkthrough.write_bytes(b"video")

    preload_path, scene_type = _discover_capture_media(registration_path)

    assert preload_path == walkthrough
    assert scene_type == "General scene"
