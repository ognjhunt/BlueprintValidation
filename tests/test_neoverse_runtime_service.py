from __future__ import annotations

import json

from fastapi.testclient import TestClient

import blueprint_validation.neoverse_runtime_service as runtime_service
from blueprint_validation.neoverse_runtime_core import NeoVerseRuntimeStore
from tests.test_neoverse_runtime_core import _build_runtime_ready_spec


def test_runtime_service_coerces_legacy_trajectory_strings(tmp_path, monkeypatch) -> None:
    spec = _build_runtime_ready_spec(tmp_path)
    store = NeoVerseRuntimeStore(tmp_path / "runtime", base_url="http://testserver")
    registration = store.build_site_world(spec)
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
