from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_scene_package(root: Path) -> Path:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "usd").mkdir(parents=True, exist_ok=True)
    (root / "isaac_lab").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "scene_manifest.json").write_text('{"scene_id":"demo_scene"}')
    (root / "usd" / "scene.usda").write_text("#usda 1.0\n")
    return root


def test_s0a_scene_package_prefers_existing_path(sample_config, tmp_path):
    from blueprint_validation.stages.s0a_scene_package import ScenePackageStage

    fac = sample_config.facilities["test_facility"]
    scene_root = _write_scene_package(tmp_path / "scene")
    fac.scene_package_path = scene_root

    result = ScenePackageStage().run(sample_config, fac, tmp_path, {})

    assert result.status == "success"
    assert result.outputs["source"] == "facility.scene_package_path"
    assert result.outputs["scene_package_path"] == str(scene_root.resolve())
    assert fac.scene_package_path == scene_root.resolve()


def test_s0a_scene_package_builds_when_enabled(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.stages.s0a_scene_package import ScenePackageStage

    fac = sample_config.facilities["test_facility"]
    fac.scene_package_path = None
    sample_config.scene_builder.enabled = True
    sample_config.scene_builder.output_scene_root = tmp_path / "built_scene"

    built_root = _write_scene_package(tmp_path / "built_scene")

    def _fake_build(_config):
        return SimpleNamespace(
            scene_root=built_root,
            scene_manifest_path=built_root / "assets" / "scene_manifest.json",
            usd_scene_path=built_root / "usd" / "scene.usda",
            task_config_path=Path(""),
            isaac_lab_package_root=built_root / "isaac_lab",
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s0a_scene_package.build_scene_package",
        _fake_build,
    )

    result = ScenePackageStage().run(sample_config, fac, tmp_path, {})

    assert result.status == "success"
    assert result.outputs["source"] == "scene_builder"
    assert result.metrics["built_scene_package"] is True
    assert fac.scene_package_path == built_root


def test_isaac_render_stage_runs_from_scene_package(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s1_isaac_render import IsaacRenderStage

    fac = sample_config.facilities["test_facility"]
    scene_root = _write_scene_package(tmp_path / "scene")
    fac.scene_package_path = scene_root

    sample_config.render.backend = "auto"
    monkeypatch.setenv("BLUEPRINT_UNSAFE_ALLOW_SCENE_PACKAGE_IMPORT", "1")

    clip_video = tmp_path / "isaac_renders" / "head_rollout_000.mp4"
    clip_video.parent.mkdir(parents=True, exist_ok=True)
    clip_video.write_bytes(b"video")

    monkeypatch.setattr(
        "blueprint_validation.stages.s1_isaac_render._render_scripted_isaac_clips",
        lambda **kwargs: [
            {
                "clip_name": "head_rollout_000",
                "video_path": str(clip_video),
                "depth_video_path": "",
                "fps": 5,
                "num_frames": 4,
                "resolution": [32, 48],
                "camera_id": "head",
            }
        ],
    )

    previous_results = {
        "s0a_scene_package": StageResult(
            stage_name="s0a_scene_package",
            status="success",
            elapsed_seconds=0.0,
            outputs={"scene_package_path": str(scene_root)},
        )
    }
    result = IsaacRenderStage().run(sample_config, fac, tmp_path, previous_results)

    assert result.status == "success"
    assert result.metrics["render_backend"] == "isaac_scene"
    assert result.outputs["manifest_path"].endswith("isaac_renders/render_manifest.json")


def test_legacy_stage1_stages_skip_for_isaac_backend(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s1_render import RenderStage
    from blueprint_validation.stages.s1b_robot_composite import RobotCompositeStage
    from blueprint_validation.stages.s1c_gemini_polish import GeminiPolishStage

    fac = sample_config.facilities["test_facility"]
    fac.scene_package_path = _write_scene_package(tmp_path / "scene")
    sample_config.render.backend = "auto"
    sample_config.gemini_polish.enabled = True
    monkeypatch.setenv("BLUEPRINT_UNSAFE_ALLOW_SCENE_PACKAGE_IMPORT", "1")
    previous_results = {
        "s0a_scene_package": StageResult(
            stage_name="s0a_scene_package",
            status="success",
            elapsed_seconds=0.0,
            outputs={"scene_package_path": str(fac.scene_package_path)},
        )
    }

    render_result = RenderStage().run(sample_config, fac, tmp_path, previous_results)
    composite_result = RobotCompositeStage().run(sample_config, fac, tmp_path, previous_results)
    polish_result = GeminiPolishStage().run(sample_config, fac, tmp_path, previous_results)

    assert render_result.status == "skipped"
    assert composite_result.status == "skipped"
    assert polish_result.status == "skipped"
    assert "isaac_scene" in render_result.detail
    assert "simulator-native robot asset" in composite_result.detail
    assert "simulator-native robot imagery" in polish_result.detail


def test_render_backend_auto_falls_back_to_gsplat(sample_config):
    from blueprint_validation.stages.render_backend import active_render_backend

    fac = sample_config.facilities["test_facility"]
    fac.scene_package_path = None
    sample_config.scene_builder.enabled = False
    sample_config.render.backend = "auto"

    assert active_render_backend(sample_config, fac, {}) == "gsplat"


def test_render_backend_auto_stays_gsplat_without_unsafe_opt_in(sample_config, tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.render_backend import active_render_backend

    fac = sample_config.facilities["test_facility"]
    fac.scene_package_path = _write_scene_package(tmp_path / "scene")
    sample_config.scene_builder.enabled = True
    sample_config.render.backend = "auto"

    previous_results = {
        "s0a_scene_package": StageResult(
            stage_name="s0a_scene_package",
            status="success",
            elapsed_seconds=0.0,
            outputs={"scene_package_path": str(fac.scene_package_path)},
        )
    }

    assert active_render_backend(sample_config, fac, previous_results) == "gsplat"
