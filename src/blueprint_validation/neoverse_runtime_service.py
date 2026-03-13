"""FastAPI service exposing the downstream NeoVerse site-world runtime contract.

The supported intake is a built site-world package (`spec` + `registration` + `health`).
Spec-only payloads remain available only as a deprecated local compatibility path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Response, WebSocket
from pydantic import BaseModel, Field
import uvicorn
import asyncio

from blueprint_contracts.site_world_contract import normalize_trajectory_payload

from .neoverse_runtime_core import NeoVerseRuntimeStore


def _runtime_store() -> NeoVerseRuntimeStore:
    host = os.getenv("NEOVERSE_RUNTIME_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("NEOVERSE_RUNTIME_SERVICE_PORT", "8787"))
    base_url = os.getenv("NEOVERSE_RUNTIME_PUBLIC_BASE_URL", f"http://{host}:{port}")
    ws_base_url = os.getenv("NEOVERSE_RUNTIME_PUBLIC_WS_BASE_URL")
    root_dir = Path(os.getenv("NEOVERSE_RUNTIME_ROOT", "./data/neoverse-runtime"))
    return NeoVerseRuntimeStore(root_dir=root_dir, base_url=base_url, ws_base_url=ws_base_url)


STORE = _runtime_store()
app = FastAPI(title="NeoVerse Site World Runtime", version="1.0.0")


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    robot_profile_id: str
    task_id: str
    scenario_id: str
    start_state_id: str
    notes: str = ""
    canonical_package_uri: str | None = None
    canonical_package_version: str | None = None
    prompt: str | None = None
    trajectory: Dict[str, Any] | str | None = None
    presentation_model: str | None = None
    debug_mode: bool = False
    unsafe_allow_blocked_site_world: bool = False


class SessionResetRequest(BaseModel):
    task_id: str | None = None
    scenario_id: str | None = None
    start_state_id: str | None = None


class SessionStepRequest(BaseModel):
    action: list[float] = Field(default_factory=list)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "neoverse-site-world-runtime",
        "version": app.version,
    }


@app.get("/v1/runtime")
def runtime_info() -> Dict[str, Any]:
    return {
        "service": "neoverse-site-world-runtime",
        "version": app.version,
        "api_version": "v1",
        "runtime_base_url": STORE.base_url,
        "websocket_base_url": STORE.ws_base_url,
        "capabilities": {
            "site_world_package_registration": True,
            "site_world_registration": True,
            "site_world_build": False,
            "legacy_site_world_build": True,
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


@app.post("/v1/site-worlds")
def register_site_world(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if all(
            isinstance(payload.get(key), dict)
            for key in ("spec", "registration", "health")
        ):
            registration = dict(
                STORE.register_site_world_package(
                    spec=dict(payload["spec"]),
                    registration=dict(payload["registration"]),
                    health=dict(payload["health"]),
                )
            )
        else:
            registration = dict(STORE.build_site_world(payload))
        health = dict(STORE.load_site_world_health(str(registration.get("site_world_id") or "")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **registration,
        "health": health,
    }


@app.get("/v1/site-worlds/{site_world_id}")
def get_site_world(site_world_id: str) -> Dict[str, Any]:
    try:
        return dict(STORE.load_site_world(site_world_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"site world not found: {site_world_id}") from exc


@app.get("/v1/site-worlds/{site_world_id}/health")
def get_site_world_health(site_world_id: str) -> Dict[str, Any]:
    try:
        return dict(STORE.load_site_world_health(site_world_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"site world not found: {site_world_id}") from exc


@app.post("/v1/site-worlds/{site_world_id}/sessions")
def create_session(site_world_id: str, request: SessionCreateRequest) -> Dict[str, Any]:
    try:
        trajectory = normalize_trajectory_payload(request.trajectory)
        return dict(
            STORE.create_session(
                site_world_id,
                session_id=request.session_id,
                robot_profile_id=request.robot_profile_id,
                task_id=request.task_id,
                scenario_id=request.scenario_id,
                start_state_id=request.start_state_id,
                notes=request.notes,
                canonical_package_uri=request.canonical_package_uri,
                canonical_package_version=request.canonical_package_version,
                prompt=request.prompt,
                trajectory=trajectory,
                presentation_model=request.presentation_model,
                debug_mode=request.debug_mode,
                unsafe_allow_blocked_site_world=request.unsafe_allow_blocked_site_world,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"site world not found: {site_world_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/sessions/{session_id}/reset")
def reset_session(session_id: str, request: SessionResetRequest) -> Dict[str, Any]:
    try:
        return dict(
            STORE.reset_session(
                session_id,
                task_id=request.task_id,
                scenario_id=request.scenario_id,
                start_state_id=request.start_state_id,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/sessions/{session_id}/step")
def step_session(session_id: str, request: SessionStepRequest) -> Dict[str, Any]:
    try:
        return dict(STORE.step_session(session_id, action=request.action))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/sessions/{session_id}/state")
def session_state(session_id: str) -> Dict[str, Any]:
    try:
        return dict(STORE.session_state(session_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/sessions/{session_id}/render")
def render_session(session_id: str, camera_id: str = "head_rgb") -> Response:
    try:
        payload = STORE.render_bytes(session_id, camera_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=payload, media_type="image/png")


@app.websocket("/v1/sessions/{session_id}/stream")
async def stream_session(session_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        for _ in range(10_000):
            payload = dict(STORE.session_state(session_id))
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)
    except FileNotFoundError:
        await websocket.send_json({"error": f"session not found: {session_id}"})
    except Exception:
        # Client disconnects are expected in local smoke flows.
        pass
    finally:
        await websocket.close()


def main() -> int:
    host = os.getenv("NEOVERSE_RUNTIME_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("NEOVERSE_RUNTIME_SERVICE_PORT", "8787"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
