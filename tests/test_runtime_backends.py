from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_validation.neoverse_production_runtime import NeoVerseProductionRuntimeStore, NeoVerseRunnerTimeoutError
from blueprint_validation.neoverse_runtime_core import SmokeContractRuntimeStore


class _StubNeoVerseRunner:
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
        del site_world_id, session_id, workspace_dir, snapshot_path, base_frame_path
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for camera in cameras:
            camera_id = str(camera.get("id") or "head_rgb")
            path = output_dir / f"{camera_id}-raw.png"
            path.write_bytes(b"raw-frame")
            frames.append({"cameraId": camera_id, "path": str(path)})
        return {
            "camera_frames": frames,
            "quality_flags": {"presentation_quality": "high"},
            "protected_region_violations": [],
            "debug_artifacts": {},
        }


class _StubNeoVerseVideoRunner(_StubNeoVerseRunner):
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
        del site_world_id, session_id, workspace_dir, snapshot_path, base_frame_path
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for camera in cameras:
            camera_id = str(camera.get("id") or "head_rgb")
            path = output_dir / f"{camera_id}.mp4"
            path.write_bytes(b"video")
            frames.append({"cameraId": camera_id, "path": str(path)})
        return {
            "camera_frames": frames,
            "quality_flags": {"presentation_quality": "high"},
            "protected_region_violations": [],
            "debug_artifacts": {},
        }


class _FlakyNeoVerseRunner(_StubNeoVerseRunner):
    def __init__(self, failures: list[bool]) -> None:
        self.failures = list(failures)

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
        should_fail = self.failures.pop(0) if self.failures else False
        if should_fail:
            raise RuntimeError("runner exploded")
        return super().render_snapshot(
            site_world_id=site_world_id,
            session_id=session_id,
            workspace_dir=workspace_dir,
            snapshot_path=snapshot_path,
            output_dir=output_dir,
            cameras=cameras,
            base_frame_path=base_frame_path,
        )


class _TimeoutNeoVerseRunner(_StubNeoVerseRunner):
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
        del site_world_id, session_id, workspace_dir, snapshot_path, output_dir, cameras, base_frame_path
        raise NeoVerseRunnerTimeoutError("NeoVerse runner timed out after 45.0s")


def _write_partial_render_artifacts(
    store: NeoVerseProductionRuntimeStore,
    session_state: dict[str, object],
) -> Path:
    session_id = str(session_state["session_id"])
    snapshot_id = str(session_state["current_world_snapshot_id"])
    render_dir = store._session_dir(session_id) / "renders" / snapshot_id
    render_dir.mkdir(parents=True, exist_ok=True)
    output_path = render_dir / "head_rgb.png"
    output_path.write_bytes(b"frame")
    manifest_path = render_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "camera_frame_paths": {"head_rgb": str(output_path)},
                "primary_camera_id": "head_rgb",
                "quality_flags": {"presentation_quality": "degraded"},
                "protected_region_violations": [],
                "debug_artifacts": {"request_path": str(render_dir / "neoverse_render_request.json")},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_runtime_backends_report_distinct_kinds(tmp_path: Path) -> None:
    smoke = SmokeContractRuntimeStore(root_dir=tmp_path / "smoke", base_url="http://smoke.local")
    production = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "prod",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )

    assert smoke.runtime_info(service_version="1.0.0")["runtime_kind"] == "smoke_contract"
    assert production.runtime_info(service_version="1.0.0")["runtime_kind"] == "neoverse_production"


def test_production_runtime_round_trip_and_restart_recovery(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._load_frame",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._extract_first_frame",
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
    store.register_site_world_package(spec=spec, registration=registration, health=health)

    create_payload = store.create_session(
        registration["site_world_id"],
        session_id="session-1",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )
    assert create_payload["runtime_kind"] == "neoverse_production"

    reset_payload = store.reset_session("session-1")
    assert reset_payload["episode"]["runtimeKind"] == "neoverse_production"

    step_payload = store.step_session("session-1", action=[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert step_payload["episode"]["stepIndex"] == 1
    assert step_payload["episode"]["actionTrace"] == [
        {
            "index": 0,
            "action_0": 0.2,
            "action_1": 0.1,
            "action_2": 0.0,
            "action_3": 0.0,
            "action_4": 0.0,
            "action_5": 0.0,
            "action_6": 0.0,
        }
    ]

    restarted_store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
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
    state_payload = restarted_store.session_state("session-1")
    assert state_payload["runtime_kind"] == "neoverse_production"
    assert state_payload["action_trace"] == [
        {
            "index": 0,
            "action_0": 0.2,
            "action_1": 0.1,
            "action_2": 0.0,
            "action_3": 0.0,
            "action_4": 0.0,
            "action_5": 0.0,
            "action_6": 0.0,
        }
    ]
    assert restarted_store.render_bytes("session-1", "head_rgb")


def test_production_runtime_accepts_video_runner_outputs(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseVideoRunner(),
    )
    def _guard_load_frame(path: Path):
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            raise AssertionError("video output should not use _load_frame")
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._load_frame",
        _guard_load_frame,
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._extract_first_frame",
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
    store.register_site_world_package(spec=spec, registration=registration, health=health)
    store.create_session(
        registration["site_world_id"],
        session_id="session-video",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )

    reset_payload = store.reset_session("session-video")
    assert reset_payload["episode"]["observation"]["primaryCameraId"] == "head_rgb"
    assert store.render_bytes("session-video", "head_rgb")


def test_production_runtime_persists_snapshot_when_reset_render_fails(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_FlakyNeoVerseRunner([True, False]),
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
    store.register_site_world_package(spec=spec, registration=registration, health=health)
    store.create_session(
        registration["site_world_id"],
        session_id="session-reset-fail",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )

    with pytest.raises(RuntimeError, match="runner exploded"):
        store.reset_session("session-reset-fail")

    persisted = store.load_session("session-reset-fail")
    assert persisted["current_world_snapshot_id"]
    assert Path(str(persisted["current_world_snapshot_path"])).is_file()
    assert persisted["latest_render_error_code"] == "render_snapshot_failed"
    state_payload = store.session_state("session-reset-fail")
    assert state_payload["observation"]["worldSnapshot"]["snapshotId"] == persisted["current_world_snapshot_id"]
    assert state_payload["observation"]["cameraFrames"][0]["available"] is False
    assert state_payload["observation"]["runtimeMetadata"]["latest_render_error_code"] == "render_snapshot_failed"
    assert store.render_bytes("session-reset-fail", "head_rgb")


def test_production_runtime_persists_step_snapshot_when_render_fails(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_FlakyNeoVerseRunner([False, True]),
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
    store.register_site_world_package(spec=spec, registration=registration, health=health)
    store.create_session(
        registration["site_world_id"],
        session_id="session-step-fail",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )
    store.reset_session("session-step-fail")

    with pytest.raises(RuntimeError, match="runner exploded"):
        store.step_session("session-step-fail", action=[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    persisted = store.load_session("session-step-fail")
    assert persisted["step_index"] == 1
    assert persisted["current_world_snapshot_id"]
    assert Path(str(persisted["current_world_snapshot_path"])).is_file()
    assert persisted["latest_render_error_code"] == "render_snapshot_failed"
    state_payload = store.session_state("session-step-fail")
    assert state_payload["step_index"] == 1
    assert state_payload["observation"]["worldSnapshot"]["snapshotId"] == persisted["current_world_snapshot_id"]


def test_production_runtime_degrades_reset_when_runner_times_out(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_TimeoutNeoVerseRunner(),
    )
    monkeypatch.setattr(store, "validate_spec", lambda *args, **kwargs: (True, [], []))
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
            "quality_flags": {"presentation_quality": "degraded"},
            "protected_region_violations": [],
            "debug_artifacts": {},
        },
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime.verify_canonical_package_version",
        lambda **_kwargs: None,
    )

    registration = json.loads(sample_site_world_bundle["registration_path"].read_text(encoding="utf-8"))
    health = json.loads(sample_site_world_bundle["health_path"].read_text(encoding="utf-8"))
    spec = json.loads(sample_site_world_bundle["spec_path"].read_text(encoding="utf-8"))
    store.register_site_world_package(spec=spec, registration=registration, health=health)
    store.create_session(
        registration["site_world_id"],
        session_id="session-reset-timeout",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )

    reset_payload = store.reset_session("session-reset-timeout")

    assert reset_payload["episode"]["observation"]["primaryCameraId"] == "head_rgb"
    persisted = store.load_session("session-reset-timeout")
    assert persisted["latest_render_error_code"] is None
    assert Path(str(persisted["latest_render_paths"]["head_rgb"])).is_file()
    assert persisted["debug_artifacts"]["runner_fallback"] == "base_frame"
    assert "timed out" in persisted["debug_artifacts"]["runner_error"]


def test_production_runtime_recovers_reset_from_partial_render_artifacts(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )
    monkeypatch.setattr(store, "validate_spec", lambda *args, **kwargs: (True, [], []))
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime._load_frame",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime.verify_canonical_package_version",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        store,
        "_render_snapshot",
        lambda session_state, _snapshot: (
            _write_partial_render_artifacts(store, session_state),
            (_ for _ in ()).throw(RuntimeError("late persistence failure")),
        )[1],
    )

    registration = json.loads(sample_site_world_bundle["registration_path"].read_text(encoding="utf-8"))
    health = json.loads(sample_site_world_bundle["health_path"].read_text(encoding="utf-8"))
    spec = json.loads(sample_site_world_bundle["spec_path"].read_text(encoding="utf-8"))
    store.register_site_world_package(spec=spec, registration=registration, health=health)
    store.create_session(
        registration["site_world_id"],
        session_id="session-reset-recover",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )

    reset_payload = store.reset_session("session-reset-recover")

    assert reset_payload["episode"]["observation"]["primaryCameraId"] == "head_rgb"
    persisted = store.load_session("session-reset-recover")
    assert persisted["latest_render_paths"]["head_rgb"].endswith("head_rgb.png")
    assert persisted["render_manifest_path"].endswith("render_manifest.json")
    assert persisted["latest_render_error_code"] is None


def test_production_runtime_recovers_render_bytes_from_manifest_when_state_is_stale(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )
    monkeypatch.setattr(store, "validate_spec", lambda *args, **kwargs: (True, [], []))
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

    registration = json.loads(sample_site_world_bundle["registration_path"].read_text(encoding="utf-8"))
    health = json.loads(sample_site_world_bundle["health_path"].read_text(encoding="utf-8"))
    spec = json.loads(sample_site_world_bundle["spec_path"].read_text(encoding="utf-8"))
    store.register_site_world_package(spec=spec, registration=registration, health=health)
    store.create_session(
        registration["site_world_id"],
        session_id="session-render-recover",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )
    store.reset_session("session-render-recover")

    session_state = store.load_session("session-render-recover")
    render_path = Path(str(session_state["latest_render_paths"]["head_rgb"]))
    manifest_path = Path(str(session_state["render_manifest_path"]))
    session_state["latest_render_paths"] = {}
    session_state["render_manifest_path"] = None
    session_state["latest_render_error_code"] = "render_bytes_failed"
    session_state["latest_render_error_message"] = "stale state"
    store._persist_session_state(session_state)

    assert render_path.is_file()
    assert manifest_path.is_file()

    payload = store.render_bytes("session-render-recover", "head_rgb")

    assert payload == b"frame"
    persisted = store.load_session("session-render-recover")
    assert persisted["latest_render_paths"]["head_rgb"] == str(render_path)
    assert persisted["render_manifest_path"] == str(manifest_path)
    assert persisted["latest_render_error_code"] is None


def test_production_runtime_degrades_render_bytes_when_runner_times_out(
    tmp_path: Path,
    sample_site_world_bundle: dict[str, Path],
    monkeypatch,
) -> None:
    store = NeoVerseProductionRuntimeStore(
        root_dir=tmp_path / "runtime",
        base_url="http://prod.local",
        runner=_StubNeoVerseRunner(),
    )
    monkeypatch.setattr(store, "validate_spec", lambda *args, **kwargs: (True, [], []))
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
            "quality_flags": {"presentation_quality": "degraded"},
            "protected_region_violations": [],
            "debug_artifacts": {},
        },
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_production_runtime.verify_canonical_package_version",
        lambda **_kwargs: None,
    )

    registration = json.loads(sample_site_world_bundle["registration_path"].read_text(encoding="utf-8"))
    health = json.loads(sample_site_world_bundle["health_path"].read_text(encoding="utf-8"))
    spec = json.loads(sample_site_world_bundle["spec_path"].read_text(encoding="utf-8"))
    store.register_site_world_package(spec=spec, registration=registration, health=health)
    store.create_session(
        registration["site_world_id"],
        session_id="session-render-timeout",
        robot_profile_id="mobile_manipulator_rgb_v1",
        task_id="task-1",
        scenario_id="scenario-default",
        start_state_id="start-default",
    )
    store.reset_session("session-render-timeout")
    store.runner = _TimeoutNeoVerseRunner()

    session_state = store.load_session("session-render-timeout")
    Path(str(session_state["latest_render_paths"]["head_rgb"])).unlink()
    session_state["latest_render_paths"] = {}
    session_state["render_manifest_path"] = None
    store._persist_session_state(session_state)

    payload = store.render_bytes("session-render-timeout", "head_rgb")

    assert payload == b"frame"
    persisted = store.load_session("session-render-timeout")
    assert Path(str(persisted["latest_render_paths"]["head_rgb"])).is_file()
    assert persisted["debug_artifacts"]["runner_fallback"] == "base_frame"
