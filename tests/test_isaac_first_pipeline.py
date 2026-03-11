from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _write_scene_package(root: Path) -> Path:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "usd").mkdir(parents=True, exist_ok=True)
    pkg = root / "isaac_lab" / "scene_task"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "assets" / "scene_manifest.json").write_text('{"scene_id":"demo_scene"}')
    (root / "usd" / "scene.usda").write_text("#usda 1.0\n")
    (root / "isaac_lab" / "__init__.py").write_text("from .scene_task import TeleopEnvCfg, create_env\n")
    pkg.joinpath("__init__.py").write_text(
        "from .blueprint_env import TeleopEnvCfg, create_env\n"
        "__all__ = ['TeleopEnvCfg', 'create_env']\n"
    )
    pkg.joinpath("blueprint_env.py").write_text(
        "import numpy as np\n"
        "class _Spec:\n"
        "    shape = (7,)\n"
        "class _Mgr:\n"
        "    action_spec = _Spec()\n"
        "class TeleopEnvCfg:\n"
        "    pass\n"
        "class _Env:\n"
        "    device = 'cpu'\n"
        "    action_manager = _Mgr()\n"
        "    def __init__(self):\n"
        "        self.task_success = np.asarray([False], dtype=bool)\n"
        "    def reset(self):\n"
        "        return {'wrist_rgb': np.zeros((8, 8, 3), dtype=np.uint8), 'policy': np.zeros((3,), dtype=np.float32)}\n"
        "    def step(self, action):\n"
        "        return self.reset(), np.asarray([0.0], dtype=np.float32), np.asarray([False], dtype=bool), {'rubric': {'success': False, 'progress': 0.0}}\n"
        "    def close(self):\n"
        "        return None\n"
        "def create_env(*, headless=False):\n"
        "    return _Env()\n"
    )
    pkg.joinpath("blueprint_runtime.json").write_text(
        '{"schema_version":"v1","runtime_kind":"blueprint_scene_env","task_package":"scene_task","env_factory":"scene_task.create_env","env_cfg_class":"TeleopEnvCfg","action_dim":7,"camera_keys":["wrist_rgb"],"state_keys":["policy"]}'
    )
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
    fac.scene_memory_bundle_path = tmp_path / "scene_memory"
    runtime_dir = tmp_path / "scene_memory_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / "runtime_selection.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "selected_backend": "neoverse",
                "secondary_backend": "gen3c",
                "fallback_backend": "cosmos_transfer",
                "available_backends": ["neoverse", "gen3c", "cosmos_transfer"],
            }
        )
    )

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
        ),
        "s0b_scene_memory_runtime": StageResult(
            stage_name="s0b_scene_memory_runtime",
            status="success",
            elapsed_seconds=0.0,
            outputs={"runtime_selection_path": str(runtime_path)},
        ),
    }
    result = IsaacRenderStage().run(sample_config, fac, tmp_path, previous_results)

    assert result.status == "success"
    assert result.metrics["render_backend"] == "isaac_scene"
    assert result.outputs["scene_memory_runtime_backend"] == "neoverse"
    manifest = json.loads(Path(result.outputs["manifest_path"]).read_text())
    assert manifest["scene_memory_runtime"]["selected_backend"] == "neoverse"
    assert manifest["intake_lineage"]["preferred_intake_kind"] == "scene_memory_bundle"
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


def test_generated_scene_package_env_can_reset_and_step(tmp_path):
    from blueprint_validation.teleop.contracts import load_and_validate_scene_package
    from blueprint_validation.teleop.runtime import _resolve_env_action_dim, load_scene_env

    scene_root = _write_scene_package(tmp_path / "scene")
    payload = load_and_validate_scene_package(scene_root)
    assert payload["has_runnable_env"] is True

    loaded = load_scene_env(scene_root=scene_root, headless=True)
    try:
        obs = loaded.env.reset()
        assert "wrist_rgb" in obs
        assert _resolve_env_action_dim(loaded.env) == 7
        step_obs, reward, done, info = loaded.env.step([[0, 0, 0, 0, 0, 0, 0]])
        assert "wrist_rgb" in step_obs
        assert reward.shape == (1,)
        assert done.shape == (1,)
        assert "rubric" in info
    finally:
        loaded.close()
