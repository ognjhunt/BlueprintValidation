from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_validation.config import ValidationConfig
from blueprint_validation.hosted_session import create_session, export_session, reset_session, run_batch, step_session


class _FakeRuntimeClient:
    def __init__(self) -> None:
        self._step_index = 0
        self.config = type("Config", (), {"service_url": "http://runtime.local"})()

    def register_site_world_package(self, *, spec, registration, health):
        return {
            "site_world_id": registration["site_world_id"],
            "runtime_kind": "native_world_model",
            "production_grade": True,
            "runtime_model_identity": {"model_id": "test-model"},
            "runtime_checkpoint_identity": {"checkpoint_id": "test-ckpt"},
        }

    def create_session(self, *_args, **_kwargs):
        return {"session_id": "remote-session-1", "observation_cameras": [{"id": "head_rgb", "role": "head"}]}

    def probe_runtime(self):
        return {
            "healthz": {"status": "ok", "runtime_kind": "neoverse_production"},
            "runtime": {
                "service": "fake",
                "runtime_kind": "native_world_model",
                "production_grade": True,
                "engine_identity": {"engine": "cosmos"},
                "model_identity": {"model_id": "test-model"},
                "checkpoint_identity": {"checkpoint_id": "test-ckpt"},
                "readiness": {"model_ready": True, "checkpoint_ready": True},
                "capabilities": {
                    "site_world_registration": True,
                    "session_reset": True,
                    "session_step": True,
                    "session_render": True,
                    "session_state": True,
                },
            },
        }

    def reset_session(self, *_args, **_kwargs):
        self._step_index = 0
        return {"episode": {"stepIndex": 0, "done": False, "reward": 0.0, "status": "ready", "observation": {"primaryCameraId": "head_rgb", "cameraFrames": [{"cameraId": "head_rgb", "framePath": "remote.png", "role": "head"}]}}}

    def step_session(self, *_args, **_kwargs):
        self._step_index += 1
        done = self._step_index >= 2
        return {"episode": {"stepIndex": self._step_index, "done": done, "reward": float(self._step_index), "status": "completed" if done else "running", "success": done, "observation": {"primaryCameraId": "head_rgb", "cameraFrames": [{"cameraId": "head_rgb", "framePath": f"remote_{self._step_index}.png", "role": "head"}]}}}

    def render_bytes(self, *_args, **_kwargs):
        return b"png-bytes"


def test_hosted_session_round_trip(tmp_path: Path, sample_site_world_bundle: dict[str, Path], monkeypatch) -> None:
    config = ValidationConfig()
    monkeypatch.setattr("blueprint_validation.hosted_session._resolve_runtime_client", lambda *_args, **_kwargs: _FakeRuntimeClient())
    monkeypatch.setattr("blueprint_validation.hosted_session._decode_png", lambda *_args, **_kwargs: np.zeros((16, 16, 3), dtype=np.uint8))
    monkeypatch.setattr("blueprint_validation.hosted_session._save_frame", lambda path, _frame: path.parent.mkdir(parents=True, exist_ok=True) or path.write_bytes(b"frame"))
    monkeypatch.setattr("blueprint_validation.hosted_session._write_video", lambda _frames, output_path: output_path.write_bytes(b"video"))

    session_dir = tmp_path / "session"
    create_payload = create_session(
        config=config,
        session_id="session-1",
        session_work_dir=session_dir,
        registration_path=sample_site_world_bundle["registration_path"],
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )
    assert create_payload["status"] == "ready"
    assert create_payload["runtime_kind"] == "native_world_model"

    reset_payload = reset_session(config=config, session_id="session-1", session_work_dir=session_dir)
    episode_id = reset_payload["episode"]["episodeId"]
    step_payload = step_session(
        config=config,
        session_work_dir=session_dir,
        episode_id=episode_id,
        action=[0, 0, 0, 0, 0, 0, 0],
    )
    assert step_payload["episode"]["stepIndex"] == 1

    batch_payload = run_batch(config=config, session_work_dir=session_dir, num_episodes=1, max_steps=2)
    assert batch_payload["summary"]["numEpisodes"] == 1

    export_payload = export_session(session_work_dir=session_dir)
    export_manifest = Path(export_payload["artifact_uris"]["export_manifest"])
    assert export_manifest.exists()
    exported = json.loads(export_manifest.read_text(encoding="utf-8"))
    assert exported["raw_bundle"]["rollout_count"] >= 1
    assert exported["runtime_kind"] == "native_world_model"
    assert (session_dir / "runtime_probe.json").exists()
