from __future__ import annotations

import json

from blueprint_validation.neoverse_runtime_client import (
    NeoVerseRuntimeClient,
    NeoVerseRuntimeClientConfig,
)


class _FakeResponse:
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_runtime_client_covers_json_and_bytes(monkeypatch):
    requests: list[tuple[str, str]] = []
    payloads: list[dict[str, object] | None] = []

    def _urlopen(request, timeout):
        requests.append((request.get_method(), request.full_url))
        payloads.append(json.loads(request.data.decode("utf-8")) if getattr(request, "data", None) else None)
        url = request.full_url
        if url.endswith("/healthz"):
            return _FakeResponse(json.dumps({"status": "ok"}).encode("utf-8"))
        if url.endswith("/v1/runtime"):
            return _FakeResponse(
                json.dumps(
                    {
                        "api_version": "v1",
                        "websocket_base_url": "ws://runtime.local",
                        "capabilities": {
                            "site_world_build": True,
                            "session_reset": True,
                            "session_step": True,
                            "session_render": True,
                            "session_state": True,
                            "session_stream": True,
                            "protected_region_locking": True,
                            "runtime_layer_compositing": True,
                            "debug_render_outputs": True,
                        },
                    }
                ).encode("utf-8")
            )
        if url.endswith("/v1/site-worlds"):
            return _FakeResponse(json.dumps({"site_world_id": "siteworld-1"}).encode("utf-8"))
        if url.endswith("/v1/site-worlds/siteworld-1"):
            return _FakeResponse(json.dumps({"site_world_id": "siteworld-1", "status": "ready"}).encode("utf-8"))
        if url.endswith("/v1/site-worlds/siteworld-1/health"):
            return _FakeResponse(json.dumps({"site_world_id": "siteworld-1", "launchable": True}).encode("utf-8"))
        if url.endswith("/v1/site-worlds/siteworld-1/sessions"):
            return _FakeResponse(
                json.dumps(
                    {
                        "session_id": "session-1",
                        "canonical_package_version": "pkg-v1",
                        "presentation_config": {"prompt": "prompt"},
                        "quality_flags": {},
                        "protected_region_violations": {"count": 0},
                    }
                ).encode("utf-8")
            )
        if url.endswith("/v1/sessions/session-1/reset"):
            return _FakeResponse(
                json.dumps(
                    {
                        "episode": {"stepIndex": 0, "done": False},
                        "canonical_package_version": "pkg-v1",
                        "presentation_config": {"prompt": "prompt"},
                    }
                ).encode("utf-8")
            )
        if url.endswith("/v1/sessions/session-1/step"):
            return _FakeResponse(
                json.dumps(
                    {
                        "episode": {"stepIndex": 1, "done": False},
                        "protected_region_violations": {"count": 0},
                    }
                ).encode("utf-8")
            )
        if url.endswith("/v1/sessions/session-1/state"):
            return _FakeResponse(
                json.dumps(
                    {
                        "status": "running",
                        "step_index": 1,
                        "canonical_package_version": "pkg-v1",
                        "quality_flags": {},
                    }
                ).encode("utf-8")
            )
        if "/v1/sessions/session-1/render" in url:
            return _FakeResponse(b"\x89PNG\r\n\x1a\nbytes", headers={"Content-Type": "image/png"})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("blueprint_validation.neoverse_runtime_client.urllib_request.urlopen", _urlopen)

    client = NeoVerseRuntimeClient(
        NeoVerseRuntimeClientConfig(
            service_url="http://runtime.local",
            api_key="secret",
            timeout_seconds=30,
        )
    )

    assert client.healthcheck()["status"] == "ok"
    assert client.runtime_info()["api_version"] == "v1"
    assert client.runtime_info()["capabilities"]["protected_region_locking"] is True
    assert client.build_site_world({"scene_id": "scene-1"})["site_world_id"] == "siteworld-1"
    assert client.get_site_world("siteworld-1")["status"] == "ready"
    assert client.get_site_world_health("siteworld-1")["launchable"] is True
    assert client.create_session(
        "siteworld-1",
        session_id="session-1",
        robot_profile_id="robot",
        task_id="task",
        scenario_id="scenario",
        start_state_id="start",
        canonical_package_version="pkg-v1",
        prompt="prompt",
        trajectory="static",
        presentation_model="model-a",
        debug_mode=True,
        unsafe_allow_blocked_site_world=True,
    )["session_id"] == "session-1"
    assert client.reset_session("session-1")["episode"]["stepIndex"] == 0
    assert client.step_session("session-1", action=[0, 0, 0])["episode"]["stepIndex"] == 1
    assert client.session_state("session-1")["status"] == "running"
    assert client.render_bytes("session-1").startswith(b"\x89PNG")
    assert ("GET", "http://runtime.local/v1/runtime") in requests
    session_request = next(payload for method, url, payload in zip([r[0] for r in requests], [r[1] for r in requests], payloads) if url.endswith("/v1/site-worlds/siteworld-1/sessions"))
    assert session_request is not None
    assert session_request["canonical_package_version"] == "pkg-v1"
    assert session_request["presentation_model"] == "model-a"
    assert session_request["trajectory"] == {"trajectory": "static"}
    assert session_request["debug_mode"] is True
    assert session_request["unsafe_allow_blocked_site_world"] is True


def test_runtime_client_stream_session_once(monkeypatch):
    class _FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def recv(self):
            return json.dumps({"status": "running", "step_index": 1})

    def _connect(url, additional_headers, open_timeout):
        assert url == "ws://runtime.local/v1/sessions/session-1/stream"
        return _FakeSocket()

    monkeypatch.setattr("blueprint_validation.neoverse_runtime_client.websocket_connect", _connect)
    monkeypatch.setattr(
        NeoVerseRuntimeClient,
        "runtime_info",
        lambda self: {"websocket_base_url": "ws://runtime.local", "api_version": "v1"},
    )

    client = NeoVerseRuntimeClient(
        NeoVerseRuntimeClientConfig(
            service_url="http://runtime.local",
            api_key="",
            timeout_seconds=30,
        )
    )
    payload = client.stream_session_once("session-1")
    assert payload["status"] == "running"
