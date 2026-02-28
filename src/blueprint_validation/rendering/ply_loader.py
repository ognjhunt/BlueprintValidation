"""Load Gaussian splat PLY files into tensors for gsplat rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common import get_logger

logger = get_logger("rendering.ply_loader")
_SUPERSPLAT_CHUNK_SIZE = 256
_SH_C0 = np.float32(0.28209479177387814)


@dataclass
class GaussianSplatData:
    """Gaussian splat data loaded from a PLY file."""

    means: Any  # (N, 3) — xyz positions
    scales: Any  # (N, 3) — log-space scales
    quats: Any  # (N, 4) — rotations as quaternions (wxyz)
    opacities: Any  # (N,) — sigmoid-space opacities
    sh_coeffs: Any  # (N, K, 3) — spherical harmonics
    num_points: int

    def to(self, device: str | Any) -> GaussianSplatData:
        return GaussianSplatData(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            opacities=self.opacities.to(device),
            sh_coeffs=self.sh_coeffs.to(device),
            num_points=self.num_points,
        )

    @property
    def bounds_min(self):
        return self.means.min(dim=0).values

    @property
    def bounds_max(self):
        return self.means.max(dim=0).values

    @property
    def center(self):
        return self.means.mean(dim=0)


def _sorted_sh_names(properties, prefix: str = "f_rest_") -> list[str]:
    names = [p.name for p in properties if p.name.startswith(prefix)]
    return sorted(
        names,
        key=lambda name: int(name[len(prefix) :]) if name[len(prefix) :].isdigit() else name,
    )


def _is_supersplat_compressed_ply(plydata) -> bool:
    if "chunk" not in plydata or "vertex" not in plydata:
        return False
    vertex_names = {p.name for p in plydata["vertex"].properties}
    required = {"packed_position", "packed_rotation", "packed_scale", "packed_color"}
    return required.issubset(vertex_names)


def _decode_supersplat_compressed(plydata) -> tuple[np.ndarray, ...]:
    """Decode supersplat compressed PLY format into standard Gaussian attributes.

    Format reference: PlayCanvas supersplat `splat-transform`.
    """
    vertex = plydata["vertex"]
    chunk = plydata["chunk"]
    n = len(vertex.data)
    if n == 0:
        raise ValueError("Compressed PLY has zero vertices")

    chunk_indices = np.arange(n, dtype=np.int64) // _SUPERSPLAT_CHUNK_SIZE
    if len(chunk.data) <= int(chunk_indices[-1]):
        raise ValueError(
            f"Compressed PLY chunk table too small: {len(chunk.data)} for {n} vertices"
        )

    def chunk_field(name: str) -> np.ndarray:
        return np.asarray(chunk[name], dtype=np.float32)[chunk_indices]

    packed_position = np.asarray(vertex["packed_position"], dtype=np.uint32)
    packed_rotation = np.asarray(vertex["packed_rotation"], dtype=np.uint32)
    packed_scale = np.asarray(vertex["packed_scale"], dtype=np.uint32)
    packed_color = np.asarray(vertex["packed_color"], dtype=np.uint32)

    min_x = chunk_field("min_x")
    max_x = chunk_field("max_x")
    min_y = chunk_field("min_y")
    max_y = chunk_field("max_y")
    min_z = chunk_field("min_z")
    max_z = chunk_field("max_z")

    pos_x = ((packed_position >> 21) & 0x7FF).astype(np.float32) / 2047.0
    pos_y = ((packed_position >> 11) & 0x3FF).astype(np.float32) / 1023.0
    pos_z = (packed_position & 0x7FF).astype(np.float32) / 2047.0
    means = np.stack(
        [
            min_x + pos_x * (max_x - min_x),
            min_y + pos_y * (max_y - min_y),
            min_z + pos_z * (max_z - min_z),
        ],
        axis=-1,
    ).astype(np.float32)

    min_sx = chunk_field("min_scale_x")
    max_sx = chunk_field("max_scale_x")
    min_sy = chunk_field("min_scale_y")
    max_sy = chunk_field("max_scale_y")
    min_sz = chunk_field("min_scale_z")
    max_sz = chunk_field("max_scale_z")

    scale_x = ((packed_scale >> 21) & 0x7FF).astype(np.float32) / 2047.0
    scale_y = ((packed_scale >> 11) & 0x3FF).astype(np.float32) / 1023.0
    scale_z = (packed_scale & 0x7FF).astype(np.float32) / 2047.0
    scales = np.stack(
        [
            min_sx + scale_x * (max_sx - min_sx),
            min_sy + scale_y * (max_sy - min_sy),
            min_sz + scale_z * (max_sz - min_sz),
        ],
        axis=-1,
    ).astype(np.float32)

    largest = (packed_rotation >> 30).astype(np.int8)
    c0 = ((packed_rotation >> 20) & 0x3FF).astype(np.float32)
    c1 = ((packed_rotation >> 10) & 0x3FF).astype(np.float32)
    c2 = (packed_rotation & 0x3FF).astype(np.float32)
    decoded = (np.stack([c0, c1, c2], axis=-1) / 1023.0 - 0.5) / (np.sqrt(2.0) * 0.5)

    quats = np.zeros((n, 4), dtype=np.float32)
    component_orders = {
        0: (1, 2, 3),
        1: (0, 2, 3),
        2: (0, 1, 3),
        3: (0, 1, 2),
    }
    for largest_idx, (a, b, c) in component_orders.items():
        mask = largest == largest_idx
        if not np.any(mask):
            continue
        quats[mask, a] = decoded[mask, 0]
        quats[mask, b] = decoded[mask, 1]
        quats[mask, c] = decoded[mask, 2]
        sq = np.sum(quats[mask] * quats[mask], axis=1)
        quats[mask, largest_idx] = np.sqrt(np.clip(1.0 - sq, 0.0, 1.0))
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(norms, 1e-8)

    min_r = chunk_field("min_r")
    max_r = chunk_field("max_r")
    min_g = chunk_field("min_g")
    max_g = chunk_field("max_g")
    min_b = chunk_field("min_b")
    max_b = chunk_field("max_b")

    color_r = ((packed_color >> 24) & 0xFF).astype(np.float32) / 255.0
    color_g = ((packed_color >> 16) & 0xFF).astype(np.float32) / 255.0
    color_b = ((packed_color >> 8) & 0xFF).astype(np.float32) / 255.0
    alpha = (packed_color & 0xFF).astype(np.float32) / 255.0

    dc_r = min_r + color_r * (max_r - min_r)
    dc_g = min_g + color_g * (max_g - min_g)
    dc_b = min_b + color_b * (max_b - min_b)
    sh_dc = np.stack(
        [
            (dc_r - 0.5) / _SH_C0,
            (dc_g - 0.5) / _SH_C0,
            (dc_b - 0.5) / _SH_C0,
        ],
        axis=-1,
    ).astype(np.float32)

    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    opacities = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    sh_coeffs = sh_dc[:, None, :]
    if "sh" in plydata:
        sh = plydata["sh"]
        if len(sh.data) != n:
            raise ValueError(
                f"Compressed PLY has mismatched sh rows: expected {n}, found {len(sh.data)}"
            )
        sh_rest_names = _sorted_sh_names(sh.properties)
        if sh_rest_names:
            sh_rest_raw = np.stack(
                [np.asarray(sh[name], dtype=np.uint8) for name in sh_rest_names],
                axis=-1,
            ).astype(np.float32)
            sh_rest_raw = ((sh_rest_raw + 0.5) / 256.0 - 0.5) * 8.0
            usable = (sh_rest_raw.shape[1] // 3) * 3
            if usable > 0:
                sh_rest = sh_rest_raw[:, :usable].reshape(n, usable // 3, 3)
                sh_coeffs = np.concatenate([sh_dc[:, None, :], sh_rest], axis=1).astype(
                    np.float32
                )

    return means, scales, quats, opacities, sh_coeffs


def load_splat(ply_path: Path, device: str = "cpu") -> GaussianSplatData:
    """Load a Gaussian splat PLY file into GaussianSplatData.

    Supports:
    - Standard 3DGS PLY fields (x/y/z, scale_*, rot_*, opacity, f_dc_*, f_rest_*)
    - Supersplat compressed PLY fields (chunk + packed_position/rotation/scale/color + sh)
    """
    import torch
    from plyfile import PlyData

    logger.info("Loading PLY: %s", ply_path)
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n = len(vertex.data)
    logger.info("Loaded %d Gaussian points", n)

    vertex_names = {p.name for p in vertex.properties}
    if {"x", "y", "z"}.issubset(vertex_names):
        means_np = np.stack(
            [
                np.array(vertex["x"], dtype=np.float32),
                np.array(vertex["y"], dtype=np.float32),
                np.array(vertex["z"], dtype=np.float32),
            ],
            axis=-1,
        )  # (N, 3)
        scales_np = np.stack(
            [
                np.array(vertex["scale_0"], dtype=np.float32),
                np.array(vertex["scale_1"], dtype=np.float32),
                np.array(vertex["scale_2"], dtype=np.float32),
            ],
            axis=-1,
        )  # (N, 3)
        quats_np = np.stack(
            [
                np.array(vertex["rot_0"], dtype=np.float32),
                np.array(vertex["rot_1"], dtype=np.float32),
                np.array(vertex["rot_2"], dtype=np.float32),
                np.array(vertex["rot_3"], dtype=np.float32),
            ],
            axis=-1,
        )  # (N, 4)
        opacities_np = np.array(vertex["opacity"], dtype=np.float32)  # (N,)
        sh_dc = np.stack(
            [
                np.array(vertex["f_dc_0"], dtype=np.float32),
                np.array(vertex["f_dc_1"], dtype=np.float32),
                np.array(vertex["f_dc_2"], dtype=np.float32),
            ],
            axis=-1,
        )[:, None, :]  # (N, 1, 3)

        sh_rest_names = _sorted_sh_names(vertex.properties)
        if sh_rest_names:
            sh_rest_raw = np.stack(
                [np.array(vertex[name], dtype=np.float32) for name in sh_rest_names],
                axis=-1,
            )  # (N, K*3)
            num_rest_coeffs = len(sh_rest_names) // 3
            sh_rest = sh_rest_raw.reshape(n, num_rest_coeffs, 3)  # (N, K, 3)
            sh_coeffs_np = np.concatenate([sh_dc, sh_rest], axis=1).astype(np.float32)
        else:
            sh_coeffs_np = sh_dc.astype(np.float32)  # (N, 1, 3)
    elif _is_supersplat_compressed_ply(plydata):
        logger.info("Detected supersplat compressed PLY format; decoding packed attributes")
        means_np, scales_np, quats_np, opacities_np, sh_coeffs_np = _decode_supersplat_compressed(
            plydata
        )
    else:
        raise ValueError(
            "Unsupported PLY schema. Expected standard 3DGS fields "
            "or supersplat packed fields, got: "
            f"{', '.join(sorted(vertex_names))}"
        )

    means = torch.from_numpy(means_np)
    scales = torch.from_numpy(scales_np)
    quats = torch.from_numpy(quats_np)
    opacities = torch.from_numpy(opacities_np)
    sh_coeffs = torch.from_numpy(sh_coeffs_np)

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
