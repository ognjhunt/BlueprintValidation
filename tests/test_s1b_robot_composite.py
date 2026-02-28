"""Tests for Stage 1b robot compositing behavior."""

from __future__ import annotations

from pathlib import Path

from blueprint_validation.common import write_json
from blueprint_validation.stages.s1b_robot_composite import RobotCompositeStage
from blueprint_validation.synthetic.robot_compositor import CompositeMetrics


def test_robot_composite_falls_back_to_unfiltered_manifest(sample_config, tmp_path, monkeypatch):
    config = sample_config
    config.robot_composite.enabled = True
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='test'/>", encoding="utf-8")
    config.robot_composite.urdf_path = urdf_path

    work_dir = tmp_path / "facility_work"
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for idx in range(2):
        video_path = work_dir / f"clip_{idx}.mp4"
        cam_path = work_dir / f"clip_{idx}.camera.json"
        video_path.write_bytes(b"")
        cam_path.write_text('{"camera_path":[]}', encoding="utf-8")
        clips.append(
            {
                "clip_name": f"clip_{idx}",
                "video_path": str(video_path),
                "camera_path": str(cam_path),
            }
        )

    write_json(
        {
            "facility": "test",
            "num_clips": len(clips),
            "clips": clips,
        },
        render_dir / "render_manifest.json",
    )

    def fake_composite_robot_arm_into_clip(*, input_video: Path, output_video: Path, **kwargs):
        del input_video, kwargs
        output_video.write_bytes(b"")
        return CompositeMetrics(
            clip_name=output_video.stem,
            mean_visible_joint_ratio=0.0,
            mean_segment_length_px=0.0,
            geometry_consistency_score=0.0,
            passed=False,
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s1b_robot_composite.composite_robot_arm_into_clip",
        fake_composite_robot_arm_into_clip,
    )

    stage = RobotCompositeStage()
    facility = next(iter(config.facilities.values()))
    result = stage.run(config, facility, work_dir, previous_results={})

    assert result.status == "success"
    assert "fallback" in (result.detail or "").lower()
    assert result.metrics["num_passed_geometry_checks"] == 0
    assert result.metrics["num_output_clips"] == 2

    manifest = (work_dir / "robot_composite" / "composited_manifest.json").read_text(
        encoding="utf-8"
    )
    assert '"robot_composite_fallback": "all_composited_clips"' in manifest
    assert '"num_clips": 2' in manifest

