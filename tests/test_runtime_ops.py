from __future__ import annotations

import json
from pathlib import Path

from blueprint_validation.neoverse_runner_wrapper import run_request
from blueprint_validation.neoverse_runtime_ops import bootstrap_neoverse_runtime


def test_neoverse_runner_wrapper_writes_response_manifest(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "NeoVerse"
    repo_root.mkdir()
    (repo_root / "inference.py").write_text("# fake inference\n", encoding="utf-8")
    model_root = repo_root / "models"
    checkpoint = model_root / "NeoVerse" / "reconstructor.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("ckpt", encoding="utf-8")
    base_frame = tmp_path / "frame.png"
    base_frame.write_bytes(b"frame")
    workspace_manifest = tmp_path / "workspace.json"
    workspace_manifest.write_text(json.dumps({"registration": {}}), encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps({"presentation_config": {"prompt": "test prompt", "trajectory": {"trajectory": "orbit_left"}}}),
        encoding="utf-8",
    )
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    output_dir = tmp_path / "output"
    request.write_text(
        json.dumps(
            {
                "workspace_manifest_path": str(workspace_manifest),
                "snapshot_path": str(snapshot),
                "base_frame_path": str(base_frame),
                "output_dir": str(output_dir),
                "cameras": [{"id": "head_rgb", "role": "head"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("NEOVERSE_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("NEOVERSE_MODEL_ROOT", str(model_root))
    monkeypatch.setenv("NEOVERSE_CHECKPOINT_PATH", str(checkpoint))

    def fake_run(cmd, capture_output, text, check):
        output_path = Path(cmd[cmd.index("--output_path") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr("blueprint_validation.neoverse_runner_wrapper.subprocess.run", fake_run)
    monkeypatch.setattr(
        "blueprint_validation.neoverse_runner_wrapper._build_static_scene_clip",
        lambda image_path, output_dir: image_path,
    )
    monkeypatch.setattr(
        "blueprint_validation.neoverse_runner_wrapper._extract_first_frame",
        lambda video_path, frame_path: frame_path.write_bytes(b"png"),
    )

    payload = run_request(request, response)
    assert response.exists()
    assert payload["camera_frames"][0]["cameraId"] == "head_rgb"
    assert Path(payload["camera_frames"][0]["path"]).exists()


def test_bootstrap_neoverse_runtime_writes_env_file(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "NeoVerse"
    env_file = tmp_path / "runtime_env.local"
    venv_python = repo_root / ".venv" / "bin" / "python"
    checkpoint = repo_root / "models" / "NeoVerse" / "reconstructor.ckpt"

    def fake_ensure_repo(repo_root: Path, git_url: str) -> None:
        del git_url
        repo_root.mkdir(parents=True, exist_ok=True)

    def fake_ensure_venv(repo_root: Path, python_bin: str) -> Path:
        del repo_root, python_bin
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
        return venv_python

    def fake_download_models(model_root: Path, hf_repo: str) -> None:
        del model_root, hf_repo
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("ckpt", encoding="utf-8")

    monkeypatch.setattr("blueprint_validation.neoverse_runtime_ops._ensure_repo", fake_ensure_repo)
    monkeypatch.setattr(
        "blueprint_validation.neoverse_runtime_ops._ensure_venv",
        fake_ensure_venv,
    )
    monkeypatch.setattr("blueprint_validation.neoverse_runtime_ops._install_neoverse_requirements", lambda *args, **kwargs: None)
    monkeypatch.setattr("blueprint_validation.neoverse_runtime_ops._download_models", fake_download_models)
    monkeypatch.setattr("blueprint_validation.neoverse_runtime_ops._detect_cuda_variant", lambda: "cu128")

    result = bootstrap_neoverse_runtime(repo_root=repo_root, env_file=env_file)
    text = env_file.read_text(encoding="utf-8")

    assert result.cuda_variant == "cu128"
    assert "NEOVERSE_REPO_ROOT" in text
    assert "NEOVERSE_RUNNER_COMMAND" in text
