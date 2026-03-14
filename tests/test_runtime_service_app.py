from __future__ import annotations

import json
import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from blueprint_validation.neoverse_production_runtime import NeoVerseProductionRuntimeStore
from blueprint_validation.runtime_service_app import create_runtime_app

try:
    from blueprint_validation.gen3c_runtime import Gen3CAsyncCachedRuntimeStore
    from blueprint_validation.multi_backend_runtime import MultiBackendRuntimeStore
except ImportError:
    Gen3CAsyncCachedRuntimeStore = None
    MultiBackendRuntimeStore = None


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


def test_runtime_service_exposes_synthetic_presentation_manifests(
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
        "blueprint_validation.neoverse_production_runtime.verify_canonical_package_version",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(store, "validate_spec", lambda *args, **kwargs: (True, [], []))

    registration = json.loads(sample_site_world_bundle["registration_path"].read_text(encoding="utf-8"))
    health = json.loads(sample_site_world_bundle["health_path"].read_text(encoding="utf-8"))
    spec = json.loads(sample_site_world_bundle["spec_path"].read_text(encoding="utf-8"))
    spec.pop("presentation", None)

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
    payload = register_response.json()
    assert payload["presentation_world_manifest_url"].endswith(
        f"/v1/site-worlds/{registration['site_world_id']}/presentation-world-manifest"
    )
    assert payload["runtime_demo_manifest_url"].endswith(
        f"/v1/site-worlds/{registration['site_world_id']}/runtime-demo-manifest"
    )
    assert payload["presentation_derivation_mode"] == "canonical_pretty"
    assert payload["presentation_ui_optional"] is True

    presentation_response = client.get(
        f"/v1/site-worlds/{registration['site_world_id']}/presentation-world-manifest"
    )
    assert presentation_response.status_code == 200
    assert presentation_response.json()["derivation_mode"] == "canonical_pretty"

    runtime_demo_response = client.get(
        f"/v1/site-worlds/{registration['site_world_id']}/runtime-demo-manifest"
    )
    assert runtime_demo_response.status_code == 200
    assert runtime_demo_response.json()["ui_base_url"] is None
    assert runtime_demo_response.json()["ui_optional"] is True


def test_runtime_service_bootstrap_site_worlds_from_env(
    tmp_path,
    sample_site_world_bundle,
    monkeypatch,
):
    runtime_service = importlib.import_module("blueprint_validation.neoverse_runtime_service")

    class _ReadyRunner:
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
            raise RuntimeError("unused")

    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_ReadyRunner(),
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
        "blueprint_validation.neoverse_production_runtime.verify_canonical_package_version",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(store, "validate_spec", lambda *args, **kwargs: (True, [], []))
    monkeypatch.setenv(
        "NEOVERSE_RUNTIME_BOOTSTRAP_REGISTRATION_PATH",
        str(sample_site_world_bundle["registration_path"]),
    )

    runtime_service._bootstrap_site_worlds(store)

    payload = store.load_site_world("site-world-1")
    assert payload["site_world_id"] == "site-world-1"


def test_runtime_service_routes_requested_gen3c_backend(
    tmp_path,
    sample_site_world_bundle,
    monkeypatch,
):
    if MultiBackendRuntimeStore is None or Gen3CAsyncCachedRuntimeStore is None:
        pytest.skip("multi-backend runtime modules are not available in this checkout")
    runner = _FlakyNeoVerseRunner()
    neoverse_store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime-neoverse",
        base_url="http://prod.local",
        runner=runner,
    )
    store = MultiBackendRuntimeStore(
        root_dir=tmp_path / "runtime-router",
        neoverse_backend=neoverse_store,
        gen3c_backend=Gen3CAsyncCachedRuntimeStore(
            root_dir=tmp_path / "runtime-gen3c",
            base_url="http://prod.local",
        ),
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
    monkeypatch.setattr(neoverse_store, "validate_spec", lambda *args, **kwargs: (True, [], []))

    registration = json.loads(sample_site_world_bundle["registration_path"].read_text(encoding="utf-8"))
    health = json.loads(sample_site_world_bundle["health_path"].read_text(encoding="utf-8"))
    spec = json.loads(sample_site_world_bundle["spec_path"].read_text(encoding="utf-8"))
    spec["runtime_eligibility"] = {
        "launchable": True,
        "readiness_state": "launchable",
        "blockers": [],
        "warnings": [],
        "grounding_status": "grounded",
        "default_backend": "neoverse",
        "launchable_backends": ["neoverse", "gen3c"],
        "backend_variants": {},
    }
    spec["backend_variants"] = {
        "neoverse": {
            "backend_id": "neoverse",
            "site_world_id": registration["site_world_id"],
            "bundle_manifest_uri": "gs://bucket/neoverse/bundle_manifest.json",
            "adapter_manifest_uri": "gs://bucket/neoverse/adapter.json",
            "launchable": True,
            "readiness_state": "launchable",
            "blockers": [],
            "warnings": [],
            "runtime_mode": "interactive",
            "grounding_status": "grounded",
            "provenance": {
                "grounding_level": "observed",
                "confidence": 0.95,
                "evidence_sources": ["gs://bucket/scene_memory_manifest.json"],
                "canonical_truth": True,
                "presentation_only": False,
            },
            "conversion": {"deterministic": True, "source_artifacts": ["conditioning_bundle.json"]},
            "canonical_write_allowed": True,
        },
        "gen3c": {
            "backend_id": "gen3c",
            "site_world_id": registration["site_world_id"],
            "bundle_manifest_uri": "gs://bucket/gen3c/bundle_manifest.json",
            "adapter_manifest_uri": "gs://bucket/gen3c/adapter.json",
            "launchable": True,
            "readiness_state": "launchable",
            "blockers": [],
            "warnings": ["missing_optional_confidence"],
            "runtime_mode": "async_cached",
            "grounding_status": "grounded",
            "provenance": {
                "grounding_level": "generated",
                "confidence": 0.7,
                "evidence_sources": ["gs://bucket/conditioning_bundle.json"],
                "canonical_truth": False,
                "presentation_only": True,
            },
            "conversion": {"deterministic": True, "source_artifacts": ["arkit_poses.json", "arkit_intrinsics.json", "depth/"]},
            "canonical_write_allowed": False,
            "quality_flags": {"async_cached": True, "generative_backend": True},
        },
    }
    spec["runtime_eligibility"]["backend_variants"] = spec["backend_variants"]
    health["backend_variants"] = spec["backend_variants"]
    health["launchable_backends"] = ["neoverse", "gen3c"]
    health["default_backend"] = "neoverse"
    registration["backend_variants"] = spec["backend_variants"]
    registration["launchable_backends"] = ["neoverse", "gen3c"]
    registration["default_backend"] = "neoverse"

    app = create_runtime_app(backend=store, title="test-runtime")
    client = TestClient(app)
    register_response = client.post(
        "/v1/site-worlds",
        json={"spec": spec, "registration": registration, "health": health},
    )
    assert register_response.status_code == 200

    create_response = client.post(
        f"/v1/site-worlds/{registration['site_world_id']}/sessions",
        json={
            "session_id": "session-gen3c",
            "requested_backend": "gen3c",
            "robot_profile_id": "mobile_manipulator_rgb_v1",
            "task_id": "task-1",
            "scenario_id": "scenario-default",
            "start_state_id": "start-default",
        },
    )
    assert create_response.status_code == 200
    create_payload = create_response.json()
    assert create_payload["runtime_backend_selected"] == "gen3c"
    assert create_payload["runtime_execution_mode"] == "async_cached"

    reset_response = client.post("/v1/sessions/session-gen3c/reset", json={})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["episode"]["qualityFlags"]["async_cached"] is True

    state_response = client.get("/v1/sessions/session-gen3c/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["runtime_backend_selected"] == "gen3c"
    assert state_payload["runtime_execution_mode"] == "async_cached"
