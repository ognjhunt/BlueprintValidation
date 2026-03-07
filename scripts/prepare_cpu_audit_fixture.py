#!/usr/bin/env python3
"""Prepare a deterministic CPU-only fixture for CI audit commands."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import yaml


def _write_splat_ply(path: Path, *, seed: int = 7, num_points: int = 48) -> None:
    rng = np.random.default_rng(seed)
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for _ in range(num_points):
            x, y, z = rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(0.1, 1.8)
            s0, s1, s2 = rng.uniform(-3.0, -1.0, size=3)
            q = rng.normal(size=4).astype(np.float32)
            q /= max(1e-8, float(np.linalg.norm(q)))
            opacity = float(rng.uniform(-2.0, 2.0))
            dc0, dc1, dc2 = rng.uniform(-1.0, 1.0, size=3)
            f.write(
                struct.pack(
                    "<14f",
                    float(x),
                    float(y),
                    float(z),
                    float(s0),
                    float(s1),
                    float(s2),
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                    float(opacity),
                    float(dc0),
                    float(dc1),
                    float(dc2),
                )
            )


def _write_video(path: Path, *, seed: int = 17, num_frames: int = 8, fps: float = 6.0) -> None:
    rng = np.random.default_rng(seed)
    h, w = 64, 64
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed opening writer for fixture video: {path}")
    for idx in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.uint8((idx * 24) % 255)
        frame[:, :, 1] = np.uint8((80 + idx * 13) % 255)
        frame[:, :, 2] = rng.integers(0, 32, size=(h, w), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_fixture_config(path: Path, *, facility_id: str, scene_ply: Path) -> None:
    payload = {
        "schema_version": "v1",
        "project_name": "ci-cpu-audit",
        "facilities": {
            facility_id: {
                "name": "CI CPU Audit Facility",
                "ply_path": str(scene_ply),
                "description": "Synthetic fixture for CPU-only audit commands",
            }
        },
        "render": {
            "scene_aware": False,
            "camera_paths": [{"type": "orbit", "radius_m": 1.8}],
            "num_clips_per_path": 1,
            "num_frames": 8,
            "fps": 6,
            "resolution": [64, 64],
            "stage1_active_perception_enabled": False,
        },
        "eval_policy": {
            "mode": "research",
            "headline_scope": "wm_only",
        },
        "policy_finetune": {"enabled": False},
        "action_boost": {"enabled": False},
        "rollout_dataset": {"enabled": False},
        "robot_composite": {"enabled": False},
        "gemini_polish": {"enabled": False},
        "robosplat": {"enabled": False},
        "robosplat_scan": {"enabled": False},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_render_manifest(path: Path, *, video_path: Path) -> None:
    payload = {
        "facility": "CI CPU Audit Facility",
        "clips": [
            {
                "clip_name": "clip_000",
                "path_type": "orbit",
                "clip_index": 0,
                "num_frames": 8,
                "resolution": [64, 64],
                "fps": 6,
                "video_path": str(video_path),
                "depth_video_path": "",
                "camera_path": str(video_path.parent / "missing_camera_path.json"),
                "path_context": {},
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root for fixture files.",
    )
    parser.add_argument(
        "--facility-id",
        type=str,
        default="ci_facility",
        help="Facility identifier used by CLI audit commands.",
    )
    args = parser.parse_args()

    out = args.output_dir.resolve()
    facility_id = args.facility_id.strip() or "ci_facility"
    scene_ply = out / "scene.ply"
    config_path = out / "config.yaml"
    work_dir = out / "work" / facility_id
    render_dir = work_dir / "renders"
    render_manifest_path = render_dir / "render_manifest.json"
    clip_path = render_dir / "clip_000.mp4"

    _write_splat_ply(scene_ply)
    _write_fixture_config(config_path, facility_id=facility_id, scene_ply=scene_ply)
    _write_video(clip_path)
    _write_render_manifest(render_manifest_path, video_path=clip_path)

    print(f"fixture_root={out}")
    print(f"config_path={config_path}")
    print(f"facility_id={facility_id}")
    print(f"work_dir={work_dir}")
    print(f"render_manifest={render_manifest_path}")


if __name__ == "__main__":
    main()
