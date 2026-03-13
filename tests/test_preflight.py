from __future__ import annotations

from blueprint_validation.config import ValidationConfig
from blueprint_validation.preflight import run_preflight


class _HealthyRuntimeClient:
    def healthcheck(self):
        return {"status": "ok", "service": "fake-runtime", "runtime_kind": "neoverse_production"}

    def runtime_info(self):
        return {
            "runtime_kind": "neoverse_production",
            "production_grade": True,
            "engine_identity": {"engine": "neoverse"},
            "model_identity": {"model_id": "test-model"},
            "checkpoint_identity": {"checkpoint_id": "test-ckpt"},
            "readiness": {"model_ready": True, "checkpoint_ready": True},
            "capabilities": {
                "site_world_registration": True,
                "session_reset": True,
                "session_step": True,
                "session_render": True,
                "session_state": True,
            }
        }


def test_preflight_checks_runtime_and_bundle(sample_site_world_bundle, monkeypatch) -> None:
    config = ValidationConfig()
    monkeypatch.setenv("NEOVERSE_RUNTIME_SERVICE_URL", "http://runtime.local")
    monkeypatch.setattr("blueprint_validation.preflight._runtime_client", lambda *_args, **_kwargs: _HealthyRuntimeClient())
    checks = run_preflight(config, site_world_registration=sample_site_world_bundle["registration_path"])
    assert all(check.passed for check in checks)


class _SmokeRuntimeClient(_HealthyRuntimeClient):
    def runtime_info(self):
        payload = dict(super().runtime_info())
        payload["runtime_kind"] = "smoke_contract"
        payload["production_grade"] = False
        return payload


def test_preflight_fails_when_smoke_runtime_is_used_for_production(sample_site_world_bundle, monkeypatch) -> None:
    config = ValidationConfig()
    monkeypatch.setenv("NEOVERSE_RUNTIME_SERVICE_URL", "http://runtime.local")
    monkeypatch.setattr("blueprint_validation.preflight._runtime_client", lambda *_args, **_kwargs: _SmokeRuntimeClient())
    checks = run_preflight(config, site_world_registration=sample_site_world_bundle["registration_path"])
    by_name = {check.name: check for check in checks}
    assert by_name["runtime:kind"].passed is False
