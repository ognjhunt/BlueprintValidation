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


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option")
        for item in items:
            if "slow" in item.keywords:
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

    cfg = ValidationConfig(
        project_name="Test Project",
        facilities={
            "test_facility": FacilityConfig(
                name="Test Facility",
                ply_path=ply_path,
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
