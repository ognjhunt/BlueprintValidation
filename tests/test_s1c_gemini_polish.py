"""Tests for Stage 1c Gemini polish lineage behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_dummy_video(path: Path, frames: int = 4) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (48, 32))
    for i in range(frames):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[..., 1] = 30 + i * 5
        writer.write(frame)
    writer.release()


def test_s1c_prefers_previous_results_lineage(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult, read_json, write_json
    from blueprint_validation.stages.s1c_gemini_polish import GeminiPolishStage

    sample_config.gemini_polish.enabled = True

    render_dir = tmp_path / "renders"
    render_video = render_dir / "clip_render.mp4"
    _write_dummy_video(render_video)
    render_manifest = render_dir / "render_manifest.json"
    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_render",
                    "video_path": str(render_video),
                    "depth_video_path": "",
                }
            ],
        },
        render_manifest,
    )

    # Stale robot-composite manifest exists on disk but should be ignored.
    stale_dir = tmp_path / "robot_composite"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_video = stale_dir / "clip_stale_robot.mp4"
    _write_dummy_video(stale_video)
    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_stale",
                    "video_path": str(stale_video),
                    "depth_video_path": "",
                }
            ],
        },
        stale_dir / "composited_manifest.json",
    )

    def _fake_polish(**kwargs):
        output_video = kwargs["output_video"]
        output_video.write_bytes(b"fake-polished")
        return {"num_sampled_frames": 1}

    monkeypatch.setattr(
        "blueprint_validation.stages.s1c_gemini_polish.polish_clip_with_gemini",
        _fake_polish,
    )

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_manifest)},
        )
    }
    stage = GeminiPolishStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, previous_results)

    assert result.status == "success"
    assert result.metrics["source_stage"] == "s1_render"
    assert result.metrics["source_mode"] == "previous_results"
    assert result.outputs["source_manifest_path"] == str(render_manifest)

    manifest = read_json(tmp_path / "gemini_polish" / "polished_manifest.json")
    assert manifest["num_clips"] == 1
    assert manifest["clips"][0]["clip_name"] == "clip_render"

