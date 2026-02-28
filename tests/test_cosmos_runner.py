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


def test_run_cosmos_inference_sets_repo_pythonpath(tmp_path, monkeypatch):
    from blueprint_validation.enrichment.cosmos_runner import run_cosmos_inference

    repo = tmp_path / "cosmos"
    script = repo / "examples" / "inference.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    expected_out = tmp_path / "out" / "clip_variant.mp4"
    expected_out.parent.mkdir(parents=True, exist_ok=True)
    expected_out.write_bytes(b"x")

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        seen["env"] = kwargs.get("env", {})

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr(
        "blueprint_validation.enrichment.cosmos_runner.subprocess.run",
        fake_run,
    )

    # Avoid filesystem-dependent probe logic in this unit test.
    monkeypatch.setattr(
        "blueprint_validation.enrichment.cosmos_runner._resolve_generated_video",
        lambda path: expected_out,
    )

    generated = run_cosmos_inference(
        spec={"name": "clip_variant", "prompt": "prompt", "video_path": "/tmp/in.mp4"},
        expected_output_path=expected_out,
        cosmos_checkpoint=tmp_path / "ckpt",
        cosmos_repo=repo,
    )

    assert generated == expected_out
    assert seen["cwd"] == str(repo)
    assert seen["cmd"][:2] == ["python", "examples/inference.py"]
    pythonpath = seen["env"].get("PYTHONPATH", "")
    parts = pythonpath.split(":")
    assert parts[0] == str(repo)
    assert parts[1] == str(repo / "packages" / "cosmos-cuda")
    assert parts[2] == str(repo / "packages" / "cosmos-oss")
