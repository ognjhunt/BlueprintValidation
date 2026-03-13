from __future__ import annotations

from blueprint_validation.config import ValidationConfig
from blueprint_validation.preflight import run_preflight


class _HealthyRuntimeClient:
    def healthcheck(self):
        return {"status": "ok", "service": "fake-runtime"}

    def runtime_info(self):
        return {
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
