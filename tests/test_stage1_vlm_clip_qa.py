from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_stage1_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "stage1_vlm_clip_qa.py"
    spec = importlib.util.spec_from_file_location("stage1_vlm_clip_qa", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_run_scoped_video_path_rejects_path_escape(tmp_path: Path):
    module = _load_stage1_module()
    with pytest.raises(ValueError, match="escapes run_dir"):
        module._resolve_run_scoped_video_path("../secret.txt", run_dir=tmp_path)


def test_is_video_path_uses_video_mime_guess(tmp_path: Path):
    module = _load_stage1_module()
    assert module._is_video_path(tmp_path / "clip.mp4") is True
    assert module._is_video_path(tmp_path / "secret.txt") is False
