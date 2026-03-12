from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
CliRunner = pytest.importorskip("click.testing").CliRunner


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _png_bytes(color: int) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((48, 64, 3), color, dtype=np.uint8))
    assert ok is True
    return bytes(encoded.tobytes())


class _FakeRuntimeClient:
    def __init__(self) -> None:
        self.session_id = "session-1"
        self.step_index = 0
        self.done_after = 2
        self.config = type("Cfg", (), {"service_url": "http://runtime.local"})()

    def probe_runtime(self):
        return {
            "healthz": {"status": "ok"},
            "runtime": {
                "api_version": "v1",
                "websocket_base_url": "ws://runtime.local",
                "capabilities": {
                    "site_world_build": True,
                    "session_reset": True,
                    "session_step": True,
                    "session_render": True,
                    "session_state": True,
                    "session_stream": True,
                },
            },
        }

    def create_session(self, site_world_id: str, **kwargs):
        assert site_world_id == "siteworld-1"
        self.session_id = str(kwargs.get("session_id") or self.session_id)
        return {
            "session_id": self.session_id,
            "runtime_capabilities": {"supports_stream": True},
            "canonical_package_version": "pkg-v1",
            "presentation_config": {
                "prompt": kwargs.get("prompt") or "default prompt",
                "presentation_model": kwargs.get("presentation_model") or "default-model",
                "debug_mode": bool(kwargs.get("debug_mode", False)),
            },
            "quality_flags": {},
            "protected_region_violations": {"count": 0},
            "debug_artifacts": {},
            "observation_cameras": [
                {"id": "head_rgb", "role": "head", "required": True},
                {"id": "wrist_rgb", "role": "wrist", "required": False},
            ],
        }

    def reset_session(self, session_id: str, **kwargs):
        assert session_id == self.session_id
        self.step_index = 0
        return self._episode_payload()

    def step_session(self, session_id: str, *, action):
        assert session_id == self.session_id
        assert len(list(action)) == 7
        self.step_index += 1
        return self._episode_payload()

    def render_bytes(self, session_id: str, *, camera_id: str = "head_rgb") -> bytes:
        assert session_id == self.session_id
        base = 80 if camera_id == "head_rgb" else 120
        return _png_bytes(base + (self.step_index * 20))

    def _episode_payload(self):
        done = self.step_index >= self.done_after
        status = "completed" if done else ("running" if self.step_index > 0 else "ready")
        return {
            "session_id": self.session_id,
            "episode": {
                "episodeId": f"remote-episode-{self.step_index}",
                "taskId": "task-1",
                "task": "Walk to shelf staging and pick the blue tote",
                "scenarioId": "scenario-normal",
                "scenario": "Normal lighting",
                "startStateId": "start-dock",
                "startState": "Dock-side tote stack",
                "status": status,
                "stepIndex": self.step_index,
                "done": done,
                "reward": round(self.step_index / 2.0, 4),
                "success": True if done else None,
                "failureReason": None,
                "observation": {
                    "primaryCameraId": "head_rgb",
                    "cameraFrames": [
                        {
                            "cameraId": "head_rgb",
                            "role": "head",
                            "required": True,
                            "available": True,
                            "framePath": f"http://runtime.local/render/{self.step_index}/head_rgb",
                        },
                        {
                            "cameraId": "wrist_rgb",
                            "role": "wrist",
                            "required": False,
                            "available": True,
                            "framePath": f"http://runtime.local/render/{self.step_index}/wrist_rgb",
                        },
                    ],
                    "runtimeMetadata": {
                        "step_index": self.step_index,
                        "canonical_package_version": "pkg-v1",
                        "presentation_config": {"prompt": "policy prompt", "presentation_model": "demo-model"},
                        "quality_flags": {"presentation_quality": "nominal"},
                        "protected_region_violations": {"count": 0},
                        "debug_artifacts": {},
                    },
                    "worldSnapshot": {"status": status},
                },
                "observationCameras": [
                    {"cameraId": "head_rgb", "available": True},
                    {"cameraId": "wrist_rgb", "available": True},
                ],
                "actionTrace": [],
                "artifactUris": {},
            },
        }


def test_hosted_session_cli_flow(monkeypatch, tmp_path, sample_config):
    import blueprint_validation.cli as cli_module
    import blueprint_validation.hosted_session as hosted_session_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "load_config", lambda _path: sample_config)
    monkeypatch.setenv("BLUEPRINT_ALLOW_MOCK_POLICY_ADAPTER", "1")
    sample_config.scene_memory_runtime.neoverse.repo_path = None

    runtime_root = tmp_path / "site-world"
    registration_path = runtime_root / "site_world_registration.json"
    health_path = runtime_root / "site_world_health.json"
    spec_path = runtime_root / "site_world_spec.json"
    raw_dir = tmp_path / "capture" / "raw" / "arkit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "poses.jsonl").write_text("{}\n", encoding="utf-8")
    (raw_dir / "intrinsics.json").write_text("{}", encoding="utf-8")
    keyframe_path = tmp_path / "capture" / "raw" / "keyframe.png"
    keyframe_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(keyframe_path), np.full((48, 64, 3), 96, dtype=np.uint8))

    _write_json(
        registration_path,
        {
            "schema_version": "v1",
            "site_world_id": "siteworld-1",
            "build_id": "build-1",
            "scene_id": "scene-1",
            "capture_id": "cap-1",
            "site_submission_id": "site-sub-1",
            "runtime_base_url": "http://runtime.local",
            "task_catalog": [
                {
                    "id": "task-1",
                    "task_id": "task-1",
                    "task_text": "Walk to shelf staging and pick the blue tote",
                }
            ],
            "scenario_catalog": [{"id": "scenario-normal", "name": "Normal lighting"}],
            "start_state_catalog": [{"id": "start-dock", "name": "Dock-side tote stack"}],
            "robot_profiles": [
                {
                    "id": "mobile_manipulator_rgb_v1",
                    "display_name": "Mobile manipulator",
                    "embodiment_type": "mobile_manipulator",
                    "action_space": {"name": "ee_delta_pose_gripper", "dim": 7, "labels": []},
                    "observation_cameras": [
                        {"id": "head_rgb", "role": "head", "required": True, "default_enabled": True},
                        {"id": "wrist_rgb", "role": "wrist", "required": False, "default_enabled": True},
                    ],
                    "allowed_policy_adapters": ["mock"],
                    "default_policy_adapter": "mock",
                }
            ],
        },
    )
    _write_json(
        health_path,
        {
            "schema_version": "v1",
            "site_world_id": "siteworld-1",
            "build_id": "build-1",
            "launchable": True,
            "healthy": True,
            "blockers": [],
            "warnings": [],
        },
    )
    _write_json(
        spec_path,
        {
            "schema_version": "v1",
            "qualification_state": "ready",
            "downstream_evaluation_eligibility": True,
            "conditioning": {
                "local_paths": {
                    "keyframe_path": str(keyframe_path),
                    "arkit_poses_path": str(raw_dir / "poses.jsonl"),
                    "arkit_intrinsics_path": str(raw_dir / "intrinsics.json"),
                }
            },
        },
    )

    fake_client = _FakeRuntimeClient()
    monkeypatch.setattr(hosted_session_module, "_resolve_runtime_client", lambda config, registration: fake_client)

    work_dir = tmp_path / "session-work"
    runner = CliRunner()
    create = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "create",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--site-world-registration",
            str(registration_path),
            "--robot-profile-id",
            "mobile_manipulator_rgb_v1",
            "--task-id",
            "task-1",
            "--scenario-id",
            "scenario-normal",
            "--start-state-id",
            "start-dock",
            "--policy-json",
            json.dumps(
                {
                    "adapter_name": "mock",
                    "model_name": "mock-policy",
                    "canonical_package_version": "pkg-v1",
                    "prompt": "policy prompt",
                    "presentation_model": "demo-model",
                    "trajectory": {"trajectory": "static"},
                    "debug_mode": True,
                }
            ),
            "--export-mode",
            "raw_bundle",
            "--export-mode",
            "rlds_dataset",
        ],
    )
    assert create.exit_code == 0
    create_payload = json.loads(create.output)
    assert create_payload["runtime_backend_selected"] == "neoverse_service"
    assert create_payload["siteWorldId"] == "siteworld-1"
    assert create_payload["canonical_package_version"] == "pkg-v1"
    assert create_payload["presentation_config"]["prompt"] == "policy prompt"

    reset = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "reset",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--task-id",
            "task-1",
            "--scenario-id",
            "scenario-normal",
            "--start-state-id",
            "start-dock",
        ],
    )
    assert reset.exit_code == 0
    reset_payload = json.loads(reset.output)
    assert reset_payload["episode"]["episodeId"].startswith("episode-")
    assert reset_payload["episode"]["startStateId"] == "start-dock"
    assert reset_payload["episode"]["canonicalPackageVersion"] == "pkg-v1"
    assert reset_payload["episode"]["presentationConfig"]["presentation_model"] == "demo-model"
    assert (work_dir / "runtime_smoke.json").exists()

    step = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "step",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--episode-id",
            reset_payload["episode"]["episodeId"],
        ],
    )
    assert step.exit_code == 0
    step_payload = json.loads(step.output)
    assert step_payload["episode"]["stepIndex"] == 1
    assert step_payload["episode"]["observation"]["cameraFrames"][0]["available"] is True

    batch = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "run-batch",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--num-episodes",
            "3",
            "--task-id",
            "task-1",
            "--scenario-id",
            "scenario-normal",
            "--start-state-id",
            "start-dock",
        ],
    )
    assert batch.exit_code == 0
    batch_payload = json.loads(batch.output)
    assert batch_payload["summary"]["numEpisodes"] == 3
    assert batch_payload["artifact_uris"]["runtime_batch_manifest"].endswith("runtime_batch_manifest.json")

    export = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "export",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
        ],
    )
    assert export.exit_code == 0
    export_payload = json.loads(export.output)
    assert export_payload["artifact_uris"]["export_manifest"].endswith("export_manifest.json")
    assert export_payload["artifact_uris"]["rlds_dataset"].endswith("rlds_manifest.json")
