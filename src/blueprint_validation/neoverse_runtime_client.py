"""HTTP client for the downstream NeoVerse site-world runtime service.

The supported intake is built site-world package registration. Spec-only build requests
remain available only for deprecated local compatibility workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from blueprint_contracts.site_world_contract import normalize_trajectory_payload

try:
    from websockets.sync.client import connect as websocket_connect
except Exception:  # pragma: no cover - optional dependency
    websocket_connect = None


def _env(name: str) -> str:
    import os

    return (os.getenv(name) or "").strip()


@dataclass(frozen=True)
class NeoVerseRuntimeClientConfig:
    service_url: str
    api_key: str
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "NeoVerseRuntimeClientConfig":
        timeout_raw = _env("NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS") or "120"
        return cls(
            service_url=_env("NEOVERSE_RUNTIME_SERVICE_URL").rstrip("/"),
            api_key=_env("NEOVERSE_RUNTIME_SERVICE_API_KEY"),
            timeout_seconds=max(1, int(timeout_raw)),
        )


class NeoVerseRuntimeClient:
    def __init__(self, config: NeoVerseRuntimeClientConfig) -> None:
        self.config = config

    def _request(
        self,
        *,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
        accept: str = "application/json",
    ) -> tuple[bytes, Mapping[str, str]]:
        if not self.config.service_url:
            raise RuntimeError("NeoVerse runtime service URL is not configured")
        body = None
        headers = {"Accept": accept}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib_request.Request(
            f"{self.config.service_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib_request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
                response_headers = dict(response.headers.items())
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail[:400]}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        return raw, response_headers

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        raw, _headers = self._request(method=method, path=path, payload=payload)
        decoded = raw.decode("utf-8")
        parsed = json.loads(decoded) if decoded.strip() else {}
        if not isinstance(parsed, Mapping):
            raise RuntimeError(f"{method} {path} returned non-object JSON")
        return parsed

    def _request_bytes(
        self,
        *,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
        accept: str,
    ) -> bytes:
        raw, _headers = self._request(method=method, path=path, payload=payload, accept=accept)
        return raw

    def healthcheck(self) -> Mapping[str, Any]:
        return self._request_json(method="GET", path="/healthz")

    def runtime_info(self) -> Mapping[str, Any]:
        return self._request_json(method="GET", path="/v1/runtime")

    def probe_runtime(self) -> Mapping[str, Any]:
        return {
            "healthz": dict(self.healthcheck()),
            "runtime": dict(self.runtime_info()),
        }

    def register_site_world_package(
        self,
        *,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Register a built site-world package with the runtime service."""
        return self._request_json(
            method="POST",
            path="/v1/site-worlds",
            payload={
                "spec": dict(spec),
                "registration": dict(registration),
                "health": dict(health),
            },
        )

    def get_site_world(self, site_world_id: str) -> Mapping[str, Any]:
        return self._request_json(
            method="GET",
            path=f"/v1/site-worlds/{urllib_parse.quote(site_world_id)}",
        )

    def get_site_world_health(self, site_world_id: str) -> Mapping[str, Any]:
        return self._request_json(
            method="GET",
            path=f"/v1/site-worlds/{urllib_parse.quote(site_world_id)}/health",
        )

    def create_session(
        self,
        site_world_id: str,
        *,
        session_id: Optional[str] = None,
        robot_profile_id: str,
        task_id: str,
        scenario_id: str,
        start_state_id: str,
        notes: str = "",
        canonical_package_uri: str | None = None,
        canonical_package_version: str | None = None,
        prompt: str | None = None,
        trajectory: Mapping[str, Any] | str | None = None,
        presentation_model: str | None = None,
        debug_mode: bool = False,
        unsafe_allow_blocked_site_world: bool = False,
    ) -> Mapping[str, Any]:
        return self._request_json(
            method="POST",
            path=f"/v1/site-worlds/{urllib_parse.quote(site_world_id)}/sessions",
            payload={
                "session_id": session_id,
                "robot_profile_id": robot_profile_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "start_state_id": start_state_id,
                "notes": notes,
                "canonical_package_uri": canonical_package_uri,
                "canonical_package_version": canonical_package_version,
                "prompt": prompt,
                "trajectory": normalize_trajectory_payload(trajectory),
                "presentation_model": presentation_model,
                "debug_mode": debug_mode,
                "unsafe_allow_blocked_site_world": bool(unsafe_allow_blocked_site_world),
            },
        )

    def reset_session(
        self,
        session_id: str,
        *,
        task_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        start_state_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        return self._request_json(
            method="POST",
            path=f"/v1/sessions/{urllib_parse.quote(session_id)}/reset",
            payload={
                "task_id": task_id,
                "scenario_id": scenario_id,
                "start_state_id": start_state_id,
            },
        )

    def step_session(self, session_id: str, *, action: Sequence[float]) -> Mapping[str, Any]:
        return self._request_json(
            method="POST",
            path=f"/v1/sessions/{urllib_parse.quote(session_id)}/step",
            payload={"action": list(action)},
        )

    def session_state(self, session_id: str) -> Mapping[str, Any]:
        return self._request_json(
            method="GET",
            path=f"/v1/sessions/{urllib_parse.quote(session_id)}/state",
        )

    def render_bytes(self, session_id: str, *, camera_id: str = "head_rgb") -> bytes:
        return self._request_bytes(
            method="GET",
            path=(
                f"/v1/sessions/{urllib_parse.quote(session_id)}/render"
                f"?camera_id={urllib_parse.quote(camera_id)}"
            ),
            accept="image/png",
        )

    def stream_session_once(self, session_id: str) -> Mapping[str, Any]:
        if websocket_connect is None:
            raise RuntimeError("websockets sync client is not installed")
        runtime = self.runtime_info()
        ws_base_url = str(runtime.get("websocket_base_url") or "").rstrip("/")
        if not ws_base_url:
            ws_base_url = self.config.service_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_base_url}/v1/sessions/{urllib_parse.quote(session_id)}/stream"
        headers: list[tuple[str, str]] = []
        if self.config.api_key:
            headers.append(("Authorization", f"Bearer {self.config.api_key}"))
        with websocket_connect(ws_url, additional_headers=headers, open_timeout=self.config.timeout_seconds) as websocket:
            raw = websocket.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(str(raw))
        if not isinstance(parsed, Mapping):
            raise RuntimeError("Session stream returned non-object JSON")
        return parsed

    def run_episode(
        self,
        *,
        session_id: str,
        actions: Sequence[Sequence[float]],
    ) -> Mapping[str, Any]:
        reset_payload = self.reset_session(session_id)
        steps = [dict(reset_payload)]
        episode = dict(reset_payload.get("episode", {}) or {})
        for action in actions:
            if bool(episode.get("done")):
                break
            payload = self.step_session(session_id, action=action)
            steps.append(dict(payload))
            episode = dict(payload.get("episode", {}) or {})
        return {
            "session_id": session_id,
            "steps": steps,
            "final_episode": episode,
        }

    def run_batch(
        self,
        *,
        site_world_id: str,
        robot_profile_id: str,
        task_id: str,
        scenario_id: str,
        start_state_id: str,
        num_episodes: int,
        action_provider: Callable[[int, Mapping[str, Any]], Sequence[float]],
    ) -> Mapping[str, Any]:
        session = self.create_session(
            site_world_id,
            session_id=None,
            robot_profile_id=robot_profile_id,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
        )
        session_id = str(session["session_id"])
        episodes = []
        for episode_index in range(max(0, int(num_episodes))):
            reset_payload = self.reset_session(
                session_id,
                task_id=task_id,
                scenario_id=scenario_id,
                start_state_id=start_state_id,
            )
            steps = [dict(reset_payload)]
            episode = dict(reset_payload.get("episode", {}) or {})
            while not bool(episode.get("done")):
                action = list(action_provider(episode_index, episode))
                payload = self.step_session(session_id, action=action)
                steps.append(dict(payload))
                episode = dict(payload.get("episode", {}) or {})
            episodes.append(
                {
                    "episode_index": episode_index,
                    "steps": steps,
                    "success": bool(episode.get("success", False)),
                }
            )
        return {
            "session_id": session_id,
            "episodes": episodes,
        }
