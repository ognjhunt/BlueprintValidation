from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_validation.neoverse_production_runtime import NeoVerseProductionRuntimeStore
from blueprint_validation.neoverse_runtime_core import SmokeContractRuntimeStore


class _StubNeoVerseRunner:
    def readiness(self):
        return {
            "ready": True,
            "model_ready": True,
            "checkpoint_ready": True,
            "runner_command_ready": True,
        }

    def model_identity(self):
        return {"model_family": "neoverse", "model_id": "stub-model"}

    def checkpoint_identity(self):
        return {"checkpoint_id": "stub-checkpoint", "checkpoint_ready": True}

    def prepare_site_world(self, *, site_world_id, workspace_dir, spec, registration, health):
        manifest_path = workspace_dir / "stub_workspace_manifest.json"
        manifest_path.write_text(json.dumps({"site_world_id": site_world_id}), encoding="utf-8")
        return {"workspace_manifest_path": str(manifest_path)}

    def render_snapshot(
        self,
        *,
        site_world_id,
        session_id,
        workspace_dir,
        snapshot_path,
        output_dir,
        cameras,
        base_frame_path,
    ):
        del site_world_id, session_id, workspace_dir, snapshot_path, base_frame_path
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for camera in cameras:
            camera_id = str(camera.get("id") or "head_rgb")
            path = output_dir / f"{camera_id}-raw.png"
            path.write_bytes(b"raw-frame")
            frames.append({"cameraId": camera_id, "path": str(path)})
        return {
            "camera_frames": frames,
            "quality_flags": {"presentation_quality": "high"},
            "protected_region_violations": [],
            "debug_artifacts": {},
        }


def test_runtime_backends_report_distinct_kinds(tmp_path: Path) -> None:
    smoke = SmokeContractRuntimeStore(root_dir=tmp_path / "smoke", base_url="http://smoke.local")
    production = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "prod",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )

    assert smoke.runtime_info(service_version="1.0.0")["runtime_kind"] == "smoke_contract"
    assert production.runtime_info(service_version="1.0.0")["runtime_kind"] == "neoverse_production"


def test_production_runtime_round_trip_and_restart_recovery(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._load_frame",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._extract_first_frame",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._save_frame",
        lambda path, _frame: path.parent.mkdir(parents=True, exist_ok=True) or path.write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime.composite_runtime_layer",
        lambda **_kwargs: {
            "frame": np.zeros((16, 16, 3), dtype=np.uint8),
            "quality_flags": {"presentation_quality": "high"},
            "protected_region_violations": [],
            "debug_artifacts": {},
        },
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime.verify_canonical_package_version",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(store, "validate_spec", lambda *args, **kwargs: (True, [], []))

    registration = json.loads(sample_site_world_bundle["registration_path"].read_text(encoding="utf-8"))
    health = json.loads(sample_site_world_bundle["health_path"].read_text(encoding="utf-8"))
    spec = json.loads(sample_site_world_bundle["spec_path"].read_text(encoding="utf-8"))
    store.register_site_world_package(spec=spec, registration=registration, health=health)

    create_payload = store.create_session(
        registration["site_world_id"],
        session_id="session-1",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )
    assert create_payload["runtime_kind"] == "neoverse_production"

    reset_payload = store.reset_session("session-1")
    assert reset_payload["episode"]["runtimeKind"] == "neoverse_production"

    step_payload = store.step_session("session-1", action=[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert step_payload["episode"]["stepIndex"] == 1

    restarted_store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._load_frame",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._save_frame",
        lambda path, _frame: path.parent.mkdir(parents=True, exist_ok=True) or path.write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime.composite_runtime_layer",
        lambda **_kwargs: {
            "frame": np.zeros((16, 16, 3), dtype=np.uint8),
            "quality_flags": {"presentation_quality": "high"},
            "protected_region_violations": [],
            "debug_artifacts": {},
        },
    )
    state_payload = restarted_store.session_state("session-1")
    assert state_payload["runtime_kind"] == "neoverse_production"
    assert restarted_store.render_bytes("session-1", "head_rgb")
