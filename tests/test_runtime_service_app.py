from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from blueprint_validation.neoverse_production_runtime import NeoVerseProductionRuntimeStore
from blueprint_validation.runtime_service_app import create_runtime_app


class _FlakyNeoVerseRunner:
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        raise RuntimeError("runner exploded")


def test_failed_reset_still_allows_state_snapshot_read(
    tmp_path,
    sample_site_world_bundle,
    monkeypatch,
):
    runner = _FlakyNeoVerseRunner()
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=runner,
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
        "blueprint_validation.neoverse_production_runtime._coerce_camera_frame",
        lambda frame, _camera_id: frame,
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
    app = create_runtime_app(backend=store, title="test-runtime")
    client = TestClient(app)

    register_response = client.post(
        "/v1/site-worlds",
        json={
            "spec": spec,
            "registration": registration,
            "health": health,
        },
    )
    assert register_response.status_code == 200

    create_response = client.post(
        f"/v1/site-worlds/{registration['site_world_id']}/sessions",
        json={
            "session_id": "session-api-fail",
            "robot_profile_id": "mobile_manipulator_rgb_v1",
            "task_id": "task-1",
            "scenario_id": "scenario-default",
            "start_state_id": "start-default",
        },
    )
    assert create_response.status_code == 200

    reset_response = client.post("/v1/sessions/session-api-fail/reset", json={})
    assert reset_response.status_code == 400
    assert "runner exploded" in reset_response.json()["detail"]

    state_response = client.get("/v1/sessions/session-api-fail/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["observation"]["worldSnapshot"]["snapshotId"]
    assert state_payload["observation"]["runtimeMetadata"]["latest_render_error_code"] == "render_snapshot_failed"
    assert runner.calls == 1
