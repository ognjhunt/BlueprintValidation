from __future__ import annotations

import json

from fastapi.testclient import TestClient

import blueprint_validation.neoverse_runtime_service as runtime_service
from blueprint_validation.neoverse_runtime_core import NeoVerseRuntimeStore
from tests.test_neoverse_runtime_core import _build_runtime_ready_spec
from tests.test_neoverse_runtime_core import _register_site_world


def test_runtime_service_reports_package_registration_as_supported(tmp_path, monkeypatch) -> None:
    store = NeoVerseRuntimeStore(tmp_path / "runtime", base_url="http://testserver")
    monkeypatch.setattr(runtime_service, "STORE", store)

    client = TestClient(runtime_service.app)
    response = client.get("/v1/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["site_world_package_registration"] is True
    assert payload["capabilities"]["site_world_registration"] is True
    assert payload["capabilities"]["site_world_build"] is False
    assert payload["capabilities"]["legacy_site_world_build"] is True


def test_runtime_service_marks_registration_modes(tmp_path, monkeypatch) -> None:
    spec = _build_runtime_ready_spec(tmp_path)
    store = NeoVerseRuntimeStore(tmp_path / "runtime", base_url="http://testserver")
    monkeypatch.setattr(runtime_service, "STORE", store)

    client = TestClient(runtime_service.app)
    package_response = client.post(
        "/v1/site-worlds",
        json={
            "spec": spec,
            "registration": {
                "schema_version": "v1",
                "site_world_id": "siteworld-1",
                "scene_id": spec["scene_id"],
                "capture_id": spec["capture_id"],
                "status": "ready",
                "task_catalog": list(spec.get("task_catalog") or []),
                "scenario_catalog": list(spec.get("scenario_catalog") or []),
                "start_state_catalog": list(spec.get("start_state_catalog") or []),
                "robot_profiles": list(spec.get("robot_profiles") or []),
                "canonical_package_uri": spec.get("canonical_package_uri"),
                "canonical_package_version": spec.get("canonical_package_version"),
            },
            "health": {
                "schema_version": "v1",
                "site_world_id": "siteworld-1",
                "scene_id": spec["scene_id"],
                "capture_id": spec["capture_id"],
                "healthy": True,
                "launchable": True,
                "status": "healthy",
                "canonical_package_version": spec.get("canonical_package_version"),
            },
        },
    )
    assert package_response.status_code == 200
    assert package_response.json()["registration_mode"] == "package_registration"

    legacy_response = client.post("/v1/site-worlds", json=spec)
    assert legacy_response.status_code == 200
    legacy_payload = legacy_response.json()
    assert legacy_payload["registration_mode"] == "legacy_spec_build"
    assert "Deprecated compatibility path" in legacy_payload["compatibility_notice"]


def test_runtime_service_coerces_legacy_trajectory_strings(tmp_path, monkeypatch) -> None:
    spec = _build_runtime_ready_spec(tmp_path)
    store = NeoVerseRuntimeStore(tmp_path / "runtime", base_url="http://testserver")
    registration = _register_site_world(store, spec)
    monkeypatch.setattr(runtime_service, "STORE", store)

    client = TestClient(runtime_service.app)
    response = client.post(
        f"/v1/site-worlds/{registration['site_world_id']}/sessions",
        json={
            "session_id": "session-service-1",
            "robot_profile_id": "mobile_manipulator_rgb_v1",
            "task_id": "task-1",
            "scenario_id": "scenario-default",
            "start_state_id": "start-default",
            "trajectory": "static",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["presentation_config"]["trajectory"] == {"trajectory": "static"}

    session_state_path = (
        tmp_path
        / "runtime"
        / "sessions"
        / "session-service-1"
        / "session_state.json"
    )
    session_state = json.loads(session_state_path.read_text(encoding="utf-8"))
    assert session_state["presentation_config"]["trajectory"] == {"trajectory": "static"}
