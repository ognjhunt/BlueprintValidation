from __future__ import annotations

from pathlib import Path

from blueprint_validation import neoverse_mixed_precision_inference as mixed_precision
from blueprint_validation import neoverse_runner_wrapper as wrapper


def test_conditioning_input_prefers_registered_video(tmp_path: Path) -> None:
    video_path = tmp_path / "walkthrough.mov"
    video_path.write_bytes(b"video")
    base_frame = tmp_path / "base.png"
    base_frame.write_bytes(b"frame")
    request_payload = {"base_frame_path": str(base_frame)}
    workspace_manifest = {
        "registration": {"conditioning_source_path": str(video_path)},
        "spec": {"conditioning": {"local_paths": {"raw_video_path": str(tmp_path / "other.mov")}}},
    }

    selected = wrapper._conditioning_input_path(request_payload, workspace_manifest)

    assert selected == video_path.resolve()


def test_conditioning_input_falls_back_to_spec_video(tmp_path: Path) -> None:
    raw_video_path = tmp_path / "walkthrough.mov"
    raw_video_path.write_bytes(b"video")
    base_frame = tmp_path / "base.png"
    base_frame.write_bytes(b"frame")
    request_payload = {"base_frame_path": str(base_frame)}
    workspace_manifest = {
        "registration": {},
        "spec": {"conditioning": {"local_paths": {"raw_video_path": str(raw_video_path)}}},
    }

    selected = wrapper._conditioning_input_path(request_payload, workspace_manifest)

    assert selected == raw_video_path.resolve()


def test_conditioning_input_falls_back_to_base_frame(tmp_path: Path) -> None:
    base_frame = tmp_path / "base.png"
    base_frame.write_bytes(b"frame")

    selected = wrapper._conditioning_input_path({"base_frame_path": str(base_frame)}, {})

    assert selected == base_frame.resolve()


def test_normalized_device_defaults_to_indexed_cuda(monkeypatch) -> None:
    monkeypatch.delenv("NEOVERSE_DEVICE", raising=False)
    assert wrapper._normalized_device() == "cuda:0"

    monkeypatch.setenv("NEOVERSE_DEVICE", "cuda")
    assert wrapper._normalized_device() == "cuda:0"

    monkeypatch.setenv("NEOVERSE_DEVICE", "cuda:3")
    assert wrapper._normalized_device() == "cuda:3"


def test_runtime_frame_budgets_prefers_higher_video_budget_then_degrades() -> None:
    assert mixed_precision._runtime_frame_budgets(static_scene=False, requested=8) == [8, 4, 2]
    assert mixed_precision._runtime_frame_budgets(static_scene=False, requested=4) == [4, 2]
    assert mixed_precision._runtime_frame_budgets(static_scene=True, requested=8) == [2]
