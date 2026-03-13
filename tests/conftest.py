"""Shared test fixtures and configuration."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", default=False, help="Run GPU tests")
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="need --run-integration option")
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option")
    else:
        skip_gpu = None
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option")
    else:
        skip_slow = None
    run_integration = config.getoption("--run-integration")

    for item in items:
        path = Path(str(getattr(item, "path", "")))
        if "integration" in path.parts:
            item.add_marker(pytest.mark.integration)
            if not run_integration:
                item.add_marker(skip_integration)
        if skip_gpu is not None and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if skip_slow is not None and "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def sample_ply(tmp_path) -> Path:
    """Create a minimal synthetic Gaussian splat PLY file for testing."""
    ply_path = tmp_path / "test.ply"
    n = 100  # 100 Gaussian points

    # Generate random data
    rng = np.random.default_rng(42)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
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

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            # xyz: random positions in a 10x10x3 box
            x, y, z = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 3)
            # scales (log-space)
            s0, s1, s2 = rng.uniform(-3, -1, size=3)
            # rotation quaternion (wxyz, normalized)
            q = rng.normal(size=4).astype(np.float32)
            q = q / np.linalg.norm(q)
            # opacity (pre-sigmoid)
            opacity = rng.uniform(-2, 2)
            # SH DC coefficients
            dc0, dc1, dc2 = rng.uniform(-1, 1, size=3)

            f.write(
                struct.pack(
                    "<14f",
                    x,
                    y,
                    z,
                    s0,
                    s1,
                    s2,
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                    opacity,
                    dc0,
                    dc1,
                    dc2,
                )
            )

    return ply_path


@pytest.fixture
def sample_ply_with_rgb(tmp_path) -> Path:
    """Create a minimal synthetic PLY with direct RGB vertex colors."""
    ply_path = tmp_path / "test_rgb.ply"
    n = 32
    rng = np.random.default_rng(7)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            x, y, z = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, 1.5)
            r = i % 256
            g = (2 * i) % 256
            b = (3 * i) % 256
            f.write(struct.pack("<3f3B", x, y, z, r, g, b))

    return ply_path


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal config for testing."""
    from blueprint_validation.config import (
        CameraPathSpec,
        FacilityConfig,
        RenderConfig,
        RolloutDatasetConfig,
        ValidationConfig,
    )

    ply_path = tmp_path / "test.ply"
    ply_path.touch()
    claim_benchmark_path = tmp_path / "claim_benchmark.json"
    claim_benchmark_path.write_text('{"version": 1, "task_specs": [], "assignments": []}')
    neoverse_repo = tmp_path / "vendor" / "neoverse"
    neoverse_repo.mkdir(parents=True, exist_ok=True)
    (neoverse_repo / "inference.py").write_text(
        """
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def _read_frame(path: str) -> np.ndarray:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        frame = cv2.imread(path)
        if frame is not None:
            return frame
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        return frame
    return np.full((48, 64, 3), 96, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--trajectory", default="static")
    parser.add_argument("--distance", default="0.0")
    parser.add_argument("--angle", default="0.0")
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--reconstructor_path", default="")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--static_scene", action="store_true")
    args = parser.parse_args()

    frame = _read_frame(args.input_path)
    frame = cv2.resize(frame, (args.width, args.height))
    boost = 12 if args.trajectory != "static" else 0
    frame = np.clip(frame.astype(np.int16) + boost, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (args.width, args.height),
    )
    for index in range(max(args.num_frames, 4)):
        shifted = np.roll(frame, shift=min(index, 6), axis=1)
        writer.write(shifted)
    writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )

    cfg = ValidationConfig(
        project_name="Test Project",
        facilities={
            "test_facility": FacilityConfig(
                name="Test Facility",
                ply_path=ply_path,
                claim_benchmark_path=claim_benchmark_path,
                description="A test facility",
                landmarks=["door", "table"],
            )
        },
        render=RenderConfig(
            resolution=(120, 160),
            fps=5,
            num_frames=4,
            camera_paths=[CameraPathSpec(type="orbit", radius_m=2.0)],
            num_clips_per_path=1,
        ),
    )
    cfg.rollout_dataset = RolloutDatasetConfig(
        export_dir=tmp_path / "policy_datasets",
    )
    # Most tests exercise OpenVLA stages directly; keep fixture on dual scope.
    cfg.eval_policy.headline_scope = "dual"
    # Keep legacy pipeline behavior in fixture-level tests unless a test opts in.
    cfg.action_boost.enabled = False
    # Default fixture behavior stays permissive; strict gates are opted into per-test.
    cfg.enrich.min_source_clips = 1
    cfg.enrich.min_valid_outputs = 1
    cfg.enrich.max_blur_reject_rate = 1.0
    cfg.enrich.green_frame_ratio_max = 1.0
    cfg.enrich.enable_visual_collapse_gate = False
    cfg.enrich.vlm_quality_gate_enabled = False
    cfg.enrich.vlm_quality_fail_closed = False
    cfg.enrich.source_clip_selection_fail_closed = False
    cfg.render.stage1_coverage_gate_enabled = False
    cfg.render.stage1_quality_planner_enabled = False
    cfg.render.stage1_quality_autoretry_enabled = False
    cfg.render.stage1_active_perception_enabled = False
    cfg.eval_policy.reliability.min_rollout_steps = 1
    cfg.policy_rl_loop.world_model_refresh_require_stage2_vlm_pass = False
    cfg.wm_refresh_loop.max_hard_negative_fraction = 1.0
    cfg.wm_refresh_loop.require_valid_video_decode = False
    cfg.wm_refresh_loop.enforce_vlm_quality_floor = False
    cfg.wm_refresh_loop.backfill_from_stage2_vlm_passed = False
    cfg.scene_memory_runtime.neoverse.allow_runtime_execution = True
    cfg.scene_memory_runtime.neoverse.repo_path = neoverse_repo
    cfg.scene_memory_runtime.neoverse.python_executable = Path(sys.executable)
    cfg.scene_memory_runtime.neoverse.inference_script = "inference.py"
    return cfg


@pytest.fixture
def sample_config_yaml(tmp_path) -> Path:
    """Create a sample YAML config file."""
    config_path = tmp_path / "validation.yaml"
    config_data = {
        "schema_version": "v1",
        "project_name": "Test",
        "facilities": {
            "test_a": {
                "name": "Test A",
                "ply_path": str(tmp_path / "a.ply"),
                "description": "Facility A",
                "landmarks": ["door"],
            }
        },
        "render": {
            "resolution": [120, 160],
            "fps": 5,
            "num_frames": 4,
            "camera_height_m": 1.0,
            "camera_paths": [{"type": "orbit", "radius_m": 2.0}],
        },
    }
    config_path.write_text(json.dumps(config_data))
    return config_path
