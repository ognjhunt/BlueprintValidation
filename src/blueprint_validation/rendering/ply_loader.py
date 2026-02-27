"""Load Gaussian splat PLY files into tensors for gsplat rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from ..common import get_logger

logger = get_logger("rendering.ply_loader")


@dataclass
class GaussianSplatData:
    """Gaussian splat data loaded from a PLY file."""

    means: torch.Tensor  # (N, 3) — xyz positions
    scales: torch.Tensor  # (N, 3) — log-space scales
    quats: torch.Tensor  # (N, 4) — rotations as quaternions (wxyz)
    opacities: torch.Tensor  # (N,) — sigmoid-space opacities
    sh_coeffs: torch.Tensor  # (N, K, 3) — spherical harmonics
    num_points: int

    def to(self, device: str | torch.device) -> GaussianSplatData:
        return GaussianSplatData(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            opacities=self.opacities.to(device),
            sh_coeffs=self.sh_coeffs.to(device),
            num_points=self.num_points,
        )

    @property
    def bounds_min(self) -> torch.Tensor:
        return self.means.min(dim=0).values

    @property
    def bounds_max(self) -> torch.Tensor:
        return self.means.max(dim=0).values

    @property
    def center(self) -> torch.Tensor:
        return self.means.mean(dim=0)


def load_splat(ply_path: Path, device: str = "cpu") -> GaussianSplatData:
    """Load a Gaussian splat PLY file into GaussianSplatData.

    Supports the standard 3DGS PLY format with properties:
    x, y, z, scale_0..2, rot_0..3, opacity, f_dc_0..2, f_rest_0..N
    """
    logger.info("Loading PLY: %s", ply_path)
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n = len(vertex.data)
    logger.info("Loaded %d Gaussian points", n)

    # Positions
    x = np.array(vertex["x"], dtype=np.float32)
    y = np.array(vertex["y"], dtype=np.float32)
    z = np.array(vertex["z"], dtype=np.float32)
    means = torch.from_numpy(np.stack([x, y, z], axis=-1))  # (N, 3)

    # Scales (log-space)
    scales = torch.from_numpy(
        np.stack(
            [
                np.array(vertex["scale_0"], dtype=np.float32),
                np.array(vertex["scale_1"], dtype=np.float32),
                np.array(vertex["scale_2"], dtype=np.float32),
            ],
            axis=-1,
        )
    )  # (N, 3)

    # Rotations (quaternions: wxyz)
    quats = torch.from_numpy(
        np.stack(
            [
                np.array(vertex["rot_0"], dtype=np.float32),
                np.array(vertex["rot_1"], dtype=np.float32),
                np.array(vertex["rot_2"], dtype=np.float32),
                np.array(vertex["rot_3"], dtype=np.float32),
            ],
            axis=-1,
        )
    )  # (N, 4)

    # Opacities (pre-sigmoid)
    opacities = torch.from_numpy(np.array(vertex["opacity"], dtype=np.float32))  # (N,)

    # Spherical harmonics — DC component
    sh_dc = torch.from_numpy(
        np.stack(
            [
                np.array(vertex["f_dc_0"], dtype=np.float32),
                np.array(vertex["f_dc_1"], dtype=np.float32),
                np.array(vertex["f_dc_2"], dtype=np.float32),
            ],
            axis=-1,
        )
    ).unsqueeze(1)  # (N, 1, 3)

    # Higher-order SH coefficients
    sh_rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
    )
    if sh_rest_names:
        sh_rest_raw = np.stack(
            [np.array(vertex[name], dtype=np.float32) for name in sh_rest_names],
            axis=-1,
        )  # (N, K*3)
        num_rest_coeffs = len(sh_rest_names) // 3
        sh_rest = torch.from_numpy(sh_rest_raw).reshape(n, num_rest_coeffs, 3)  # (N, K, 3)
        sh_coeffs = torch.cat([sh_dc, sh_rest], dim=1)  # (N, 1+K, 3)
    else:
        sh_coeffs = sh_dc  # (N, 1, 3)

    data = GaussianSplatData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_coeffs=sh_coeffs,
        num_points=n,
    )

    if device != "cpu":
        data = data.to(device)

    logger.info(
        "Splat bounds: min=%s, max=%s, center=%s",
        data.bounds_min.tolist(),
        data.bounds_max.tolist(),
        data.center.tolist(),
    )
    return data
