"""Tests for Stage 1e minimal SplatSim interaction stage."""

from __future__ import annotations

from pathlib import Path


def test_splatsim_backend_sanitizes_generated_output_filename(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import read_json
    from blueprint_validation.config import ManipulationZoneConfig
    from blueprint_validation.synthetic.splatsim_pybullet_backend import run_splatsim_pybullet_backend

    class _FakeCapture:
        def __init__(self, *_args, **_kwargs):
            self._frame_idx = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 5:  # CAP_PROP_FPS
                return 5
            if prop == 3:  # CAP_PROP_FRAME_WIDTH
                return 64
            if prop == 4:  # CAP_PROP_FRAME_HEIGHT
                return 48
            return 0

        def read(self):
            import numpy as np

            if self._frame_idx > 2:
                return False, None
            self._frame_idx += 1
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self):
            return None

    class _FakeWriter:
        def __init__(self, path, *_args, **_kwargs):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"")

        def write(self, _frame):
            return None

        def release(self):
            return None

    class _FakeCV2:
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        FONT_HERSHEY_SIMPLEX = 0

        @staticmethod
        def VideoCapture(path):
            return _FakeCapture(path)

        @staticmethod
        def VideoWriter(path, *args, **kwargs):
            return _FakeWriter(path, *args, **kwargs)

        @staticmethod
        def VideoWriter_fourcc(*_args):
            return 0

        @staticmethod
        def line(*_args, **_kwargs):
            return None

        @staticmethod
        def circle(*_args, **_kwargs):
            return None

        @staticmethod
        def putText(*_args, **_kwargs):
            return None

    class _FakePB:
        DIRECT = 0
        GEOM_BOX = 1
        GEOM_SPHERE = 2
        POSITION_CONTROL = 3

        @staticmethod
        def connect(_mode):
            return 1

        @staticmethod
        def setAdditionalSearchPath(*_args, **_kwargs):
            return None

        @staticmethod
        def setGravity(*_args, **_kwargs):
            return None

        @staticmethod
        def loadURDF(*_args, **_kwargs):
            return 1

        @staticmethod
        def createCollisionShape(*_args, **_kwargs):
            return 1

        @staticmethod
        def createVisualShape(*_args, **_kwargs):
            return 1

        @staticmethod
        def createMultiBody(*_args, **_kwargs):
            return 1

        @staticmethod
        def resetBasePositionAndOrientation(*_args, **_kwargs):
            return None

        @staticmethod
        def setJointMotorControl2(*_args, **_kwargs):
            return None

        @staticmethod
        def stepSimulation(*_args, **_kwargs):
            return None

        @staticmethod
        def getBasePositionAndOrientation(*_args, **_kwargs):
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        @staticmethod
        def disconnect(*_args, **_kwargs):
            return None

    class _FakePBData:
        @staticmethod
        def getDataPath():
            return str(tmp_path)

    import sys

    monkeypatch.setitem(sys.modules, "cv2", _FakeCV2)
    monkeypatch.setitem(sys.modules, "pybullet", _FakePB)
    monkeypatch.setitem(sys.modules, "pybullet_data", _FakePBData)

    source_video = tmp_path / "in.mp4"
    source_video.write_bytes(b"x")

    source_manifest = {
        "clips": [
            {
                "clip_name": "../../.ssh/authorized_keys",
                "video_path": str(source_video),
                "depth_video_path": "",
                "fps": 5,
                "num_frames": 3,
                "resolution": [48, 64],
            }
        ]
    }
    sample_config.splatsim.horizon_steps = 4
    sample_config.splatsim.per_zone_rollouts = 1
    sample_config.splatsim.min_successful_rollouts_per_zone = 0
    facility = list(sample_config.facilities.values())[0]
    facility.manipulation_zones = [
        ManipulationZoneConfig(
            name="zone/../../evil",
            approach_point=[0.0, 0.0, 0.2],
            target_point=[0.1, 0.1, 0.2],
        )
    ]

    stage_dir = tmp_path / "stage"
    result = run_splatsim_pybullet_backend(
        config=sample_config,
        facility=facility,
        stage_dir=stage_dir,
        source_manifest=source_manifest,
        source_manifest_path=tmp_path / "source_manifest.json",
    )

    assert result["status"] == "success"
    manifest = read_json(stage_dir / "interaction_manifest.json")
    generated = manifest["clips"][1]
    output_path = Path(generated["video_path"]).resolve()

    assert ".." not in generated["clip_name"]
    assert output_path.parent == stage_dir.resolve()


def _write_source_manifest(path: Path) -> None:
    from blueprint_validation.common import write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    video_path = path.parent / "clip_000.mp4"
    video_path.write_bytes(b"not_a_real_video")
    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000",
                    "path_type": "orbit",
                    "clip_index": 0,
                    "num_frames": 4,
                    "resolution": [48, 64],
                    "fps": 5,
                    "video_path": str(video_path),
                    "depth_video_path": "",
                }
            ],
        },
        path,
    )


def test_stage_s1e_skips_when_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = False
    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"


def test_stage_s1e_success_path(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = True
    sample_config.splatsim.mode = "hybrid"

    source_manifest = tmp_path / "renders" / "render_manifest.json"
    _write_source_manifest(source_manifest)

    def _fake_backend(**kwargs):
        stage_dir = kwargs["stage_dir"]
        manifest_path = stage_dir / "interaction_manifest.json"
        write_json(
            {
                "facility": "Test Facility",
                "clips": [],
                "backend_used": "pybullet",
                "fallback_used": False,
            },
            manifest_path,
        )
        return {
            "status": "success",
            "reason": "ok",
            "manifest_path": str(manifest_path),
            "num_source_clips": 1,
            "num_generated_clips": 1,
            "num_successful_rollouts": 1,
            "fallback_used": False,
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s1e_splatsim_interaction.run_splatsim_pybullet_backend",
        _fake_backend,
    )

    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["fallback_used"] is False
    assert result.metrics["num_generated_clips"] == 1


def test_stage_s1e_hybrid_fallback(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import read_json
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = True
    sample_config.splatsim.mode = "hybrid"
    sample_config.splatsim.fallback_to_prior_manifest = True

    source_manifest = tmp_path / "renders" / "render_manifest.json"
    _write_source_manifest(source_manifest)

    def _fake_backend(**kwargs):
        del kwargs
        return {
            "status": "failed",
            "reason": "pybullet_unavailable",
            "manifest_path": "",
            "num_source_clips": 1,
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s1e_splatsim_interaction.run_splatsim_pybullet_backend",
        _fake_backend,
    )

    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["fallback_used"] is True

    manifest = read_json(tmp_path / "splatsim" / "interaction_manifest.json")
    assert manifest["fallback_used"] is True
    assert len(manifest["clips"]) == 1


def test_stage_s1e_prefers_previous_results_manifest(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = True
    sample_config.splatsim.mode = "hybrid"

    render_manifest = tmp_path / "renders" / "render_manifest.json"
    _write_source_manifest(render_manifest)

    # Stale higher-priority manifest on disk should be ignored in pipeline execution.
    stale_aug = tmp_path / "gaussian_augment" / "augmented_manifest.json"
    stale_aug.parent.mkdir(parents=True, exist_ok=True)
    write_json({"clips": []}, stale_aug)

    captured: dict = {}

    def _fake_backend(**kwargs):
        captured["source_manifest_path"] = kwargs["source_manifest_path"]
        stage_dir = kwargs["stage_dir"]
        manifest_path = stage_dir / "interaction_manifest.json"
        write_json({"clips": []}, manifest_path)
        return {
            "status": "success",
            "reason": "ok",
            "manifest_path": str(manifest_path),
            "num_source_clips": 1,
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s1e_splatsim_interaction.run_splatsim_pybullet_backend",
        _fake_backend,
    )

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_manifest)},
        )
    }
    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, previous_results)
    assert result.status == "success"
    assert captured["source_manifest_path"] == render_manifest
    assert result.metrics["source_stage"] == "s1_render"
    assert result.metrics["source_mode"] == "previous_results"


def test_stage_s1e_fails_on_invalid_source_manifest_before_backend_call(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = True
    sample_config.splatsim.mode = "hybrid"
    source_manifest = tmp_path / "renders" / "render_manifest.json"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    # invalid: missing clip_name required by strict stage1_source schema
                    "video_path": str(tmp_path / "renders" / "missing.mp4"),
                }
            ],
        },
        source_manifest,
    )

    called = {"value": False}

    def _fake_backend(**kwargs):
        del kwargs
        called["value"] = True
        return {"status": "success", "manifest_path": ""}

    monkeypatch.setattr(
        "blueprint_validation.stages.s1e_splatsim_interaction.run_splatsim_pybullet_backend",
        _fake_backend,
    )

    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "failed"
    assert "Invalid source manifest for SplatSim" in result.detail
    assert called["value"] is False
