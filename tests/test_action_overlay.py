"""Tests for WM-only scripted manipulation overlay rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_tiny_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for _ in range(6):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()


def test_overlay_scripted_trace_on_video_writes_output(tmp_path):
    from blueprint_validation.evaluation.action_overlay import overlay_scripted_trace_on_video

    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "overlay.mp4"
    _write_tiny_video(input_video)
    actions = [[0.01, -0.01, 0.01, 0, 0, 0, -1], [0.02, 0.0, -0.01, 0, 0, 0, 1]]

    out = overlay_scripted_trace_on_video(
        input_video_path=input_video,
        output_video_path=output_video,
        action_sequence=actions,
        target_label="trash can",
    )
    assert out == output_video
    assert out.exists()
    assert out.stat().st_size > 0
