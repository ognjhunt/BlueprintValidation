"""Tests for lightweight Stage-2 scene-index scaffolding."""

from __future__ import annotations

import numpy as np


def _camera_frame(c2w: np.ndarray) -> dict:
    return {"camera_to_world": [float(v) for v in c2w.reshape(-1).tolist()], "fov": 60.0}


def test_build_and_query_scene_index(tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.enrichment.scene_index import (
        build_scene_index,
        query_nearest_context_candidates,
    )

    camera_path_a = tmp_path / "clip_a_camera_path.json"
    camera_path_b = tmp_path / "clip_b_camera_path.json"

    c2w_a0 = np.eye(4, dtype=np.float64)
    c2w_a1 = np.eye(4, dtype=np.float64)
    c2w_a1[0, 3] = 1.0
    c2w_b0 = np.eye(4, dtype=np.float64)
    c2w_b0[0, 3] = 1.1

    write_json({"camera_path": [_camera_frame(c2w_a0), _camera_frame(c2w_a1)]}, camera_path_a)
    write_json({"camera_path": [_camera_frame(c2w_b0)]}, camera_path_b)

    render_manifest = {
        "clips": [
            {
                "clip_name": "clip_a",
                "video_path": str(tmp_path / "clip_a.mp4"),
                "camera_path": str(camera_path_a),
            },
            {
                "clip_name": "clip_b",
                "video_path": str(tmp_path / "clip_b.mp4"),
                "camera_path": str(camera_path_b),
            },
        ]
    }
    index_path = tmp_path / "scene_index.json"
    scene_index = build_scene_index(
        render_manifest=render_manifest,
        output_path=index_path,
        sample_every_n_frames=1,
    )

    assert index_path.exists()
    assert scene_index["num_entries"] == 3

    nearest = query_nearest_context_candidates(
        scene_index=scene_index,
        anchor_clip_name="clip_a",
        anchor_frame_index=1,
        k=1,
    )
    assert len(nearest) == 1
    assert nearest[0]["clip_name"] == "clip_b"
    assert nearest[0]["frame_index"] == 0
