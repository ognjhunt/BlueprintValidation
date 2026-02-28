"""Tests for Cosmos Transfer runner contracts."""

from __future__ import annotations

from pathlib import Path


def test_build_controlnet_spec_depth(tmp_path):
    from blueprint_validation.enrichment.cosmos_runner import build_controlnet_spec

    video = tmp_path / "input.mp4"
    depth = tmp_path / "input_depth.mp4"
    out = tmp_path / "out.mp4"
    video.write_bytes(b"x")
    depth.write_bytes(b"x")

    spec = build_controlnet_spec(
        video_path=video,
        depth_path=depth,
        prompt="warehouse scene",
        output_path=out,
        controlnet_inputs=["rgb", "depth"],
    )

    assert spec["name"] == "out"
    assert spec["video_path"] == str(video)
    assert spec["depth"]["control_path"] == str(depth)
    assert "edge" not in spec


def test_build_cosmos_inference_command():
    from blueprint_validation.enrichment.cosmos_runner import build_cosmos_inference_command

    cmd = build_cosmos_inference_command(
        spec_path=Path("/tmp/spec.json"),
        output_dir=Path("/tmp/out"),
    )
    assert cmd == ["python", "examples/inference.py", "-i", "/tmp/spec.json", "-o", "/tmp/out"]


def test_resolve_cosmos_repo(tmp_path):
    from blueprint_validation.enrichment.cosmos_runner import resolve_cosmos_repo

    repo = tmp_path / "cosmos"
    script = repo / "examples" / "inference.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n")

    resolved = resolve_cosmos_repo(repo)
    assert resolved == repo
