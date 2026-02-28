"""Tests for Gaussian splat PLY loader format support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueprint_validation.rendering.ply_loader import load_splat

plyfile = pytest.importorskip("plyfile")
PlyData = plyfile.PlyData
PlyElement = plyfile.PlyElement

_SH_C0 = 0.28209479177387814


def _pack_unorm(value: float, bits: int) -> int:
    max_int = (1 << bits) - 1
    return int(np.clip(np.floor(value * max_int + 0.5), 0, max_int))


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if value <= minimum:
        return 0.0
    if value >= maximum:
        return 1.0
    if maximum - minimum < 1e-5:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _pack_111011(x: float, y: float, z: float) -> np.uint32:
    return np.uint32(
        (_pack_unorm(x, 11) << 21)
        | (_pack_unorm(y, 10) << 11)
        | _pack_unorm(z, 11)
    )


def _pack_8888(x: float, y: float, z: float, w: float) -> np.uint32:
    return np.uint32(
        (_pack_unorm(x, 8) << 24)
        | (_pack_unorm(y, 8) << 16)
        | (_pack_unorm(z, 8) << 8)
        | _pack_unorm(w, 8)
    )


def _pack_rotation(quat: np.ndarray) -> np.uint32:
    q = np.asarray(quat, dtype=np.float32)
    q = q / np.linalg.norm(q)
    largest = int(np.argmax(np.abs(q)))
    if q[largest] < 0:
        q = -q

    norm = np.sqrt(2.0) * 0.5
    packed = largest
    for i in range(4):
        if i != largest:
            packed = (packed << 10) | _pack_unorm(float(q[i] * norm + 0.5), 10)
    return np.uint32(packed)


def _write_supersplat_compressed_fixture(path: Path) -> dict[str, np.ndarray]:
    n = 4
    means = np.array(
        [
            [-0.8, 0.1, 1.0],
            [0.2, -0.6, 0.4],
            [0.9, 0.5, -0.2],
            [0.1, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    scales = np.array(
        [
            [-2.3, -2.1, -1.9],
            [-1.8, -2.0, -2.4],
            [-2.7, -1.7, -2.2],
            [-2.0, -2.5, -2.1],
        ],
        dtype=np.float32,
    )
    quats = np.array(
        [
            [0.9239, 0.3827, 0.0, 0.0],
            [0.9659, 0.0, 0.2588, 0.0],
            [0.8660, 0.0, 0.0, 0.5],
            [0.7071, 0.3536, 0.3536, 0.5],
        ],
        dtype=np.float32,
    )
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    opacities = np.array([-1.1, 0.2, 1.3, -0.4], dtype=np.float32)
    sh_dc = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.4, 0.0, -0.1],
            [-0.2, 0.5, 0.2],
            [0.3, -0.4, 0.1],
        ],
        dtype=np.float32,
    )
    # One extra SH coefficient per channel (3 total fields).
    sh_rest = np.array(
        [
            [0.2, -0.1, 0.4],
            [0.0, 0.1, -0.3],
            [-0.2, 0.3, 0.0],
            [0.1, -0.4, 0.2],
        ],
        dtype=np.float32,
    )

    chunk_dtype = np.dtype(
        [
            ("min_x", "<f4"),
            ("min_y", "<f4"),
            ("min_z", "<f4"),
            ("max_x", "<f4"),
            ("max_y", "<f4"),
            ("max_z", "<f4"),
            ("min_scale_x", "<f4"),
            ("min_scale_y", "<f4"),
            ("min_scale_z", "<f4"),
            ("max_scale_x", "<f4"),
            ("max_scale_y", "<f4"),
            ("max_scale_z", "<f4"),
            ("min_r", "<f4"),
            ("min_g", "<f4"),
            ("min_b", "<f4"),
            ("max_r", "<f4"),
            ("max_g", "<f4"),
            ("max_b", "<f4"),
        ]
    )
    chunk = np.zeros(1, dtype=chunk_dtype)
    chunk["min_x"] = np.min(means[:, 0])
    chunk["min_y"] = np.min(means[:, 1])
    chunk["min_z"] = np.min(means[:, 2])
    chunk["max_x"] = np.max(means[:, 0])
    chunk["max_y"] = np.max(means[:, 1])
    chunk["max_z"] = np.max(means[:, 2])
    chunk["min_scale_x"] = np.min(scales[:, 0])
    chunk["min_scale_y"] = np.min(scales[:, 1])
    chunk["min_scale_z"] = np.min(scales[:, 2])
    chunk["max_scale_x"] = np.max(scales[:, 0])
    chunk["max_scale_y"] = np.max(scales[:, 1])
    chunk["max_scale_z"] = np.max(scales[:, 2])

    dc_for_pack = sh_dc * _SH_C0 + 0.5
    chunk["min_r"] = np.min(dc_for_pack[:, 0])
    chunk["min_g"] = np.min(dc_for_pack[:, 1])
    chunk["min_b"] = np.min(dc_for_pack[:, 2])
    chunk["max_r"] = np.max(dc_for_pack[:, 0])
    chunk["max_g"] = np.max(dc_for_pack[:, 1])
    chunk["max_b"] = np.max(dc_for_pack[:, 2])

    vertex_dtype = np.dtype(
        [
            ("packed_position", "<u4"),
            ("packed_rotation", "<u4"),
            ("packed_scale", "<u4"),
            ("packed_color", "<u4"),
        ]
    )
    vertex = np.zeros(n, dtype=vertex_dtype)

    for i in range(n):
        vertex["packed_position"][i] = _pack_111011(
            _normalize(means[i, 0], float(chunk["min_x"][0]), float(chunk["max_x"][0])),
            _normalize(means[i, 1], float(chunk["min_y"][0]), float(chunk["max_y"][0])),
            _normalize(means[i, 2], float(chunk["min_z"][0]), float(chunk["max_z"][0])),
        )
        vertex["packed_scale"][i] = _pack_111011(
            _normalize(scales[i, 0], float(chunk["min_scale_x"][0]), float(chunk["max_scale_x"][0])),
            _normalize(scales[i, 1], float(chunk["min_scale_y"][0]), float(chunk["max_scale_y"][0])),
            _normalize(scales[i, 2], float(chunk["min_scale_z"][0]), float(chunk["max_scale_z"][0])),
        )
        vertex["packed_rotation"][i] = _pack_rotation(quats[i])
        vertex["packed_color"][i] = _pack_8888(
            _normalize(dc_for_pack[i, 0], float(chunk["min_r"][0]), float(chunk["max_r"][0])),
            _normalize(dc_for_pack[i, 1], float(chunk["min_g"][0]), float(chunk["max_g"][0])),
            _normalize(dc_for_pack[i, 2], float(chunk["min_b"][0]), float(chunk["max_b"][0])),
            1.0 / (1.0 + float(np.exp(-opacities[i]))),
        )

    sh_dtype = np.dtype([("f_rest_0", "u1"), ("f_rest_1", "u1"), ("f_rest_2", "u1")])
    sh = np.zeros(n, dtype=sh_dtype)
    for i in range(n):
        for j, name in enumerate(("f_rest_0", "f_rest_1", "f_rest_2")):
            nvalue = float(sh_rest[i, j] / 8.0 + 0.5)
            sh[name][i] = np.uint8(np.clip(np.trunc(nvalue * 256.0), 0, 255))

    ply = PlyData(
        [
            PlyElement.describe(chunk, "chunk"),
            PlyElement.describe(vertex, "vertex"),
            PlyElement.describe(sh, "sh"),
        ],
        text=False,
    )
    ply.write(str(path))

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh_dc": sh_dc,
        "sh_rest": sh_rest,
    }


def test_load_splat_supports_supersplat_compressed(tmp_path):
    ply_path = tmp_path / "compressed.ply"
    expected = _write_supersplat_compressed_fixture(ply_path)

    loaded = load_splat(ply_path, device="cpu")

    assert loaded.num_points == 4
    means = loaded.means.numpy()
    scales = loaded.scales.numpy()
    quats = loaded.quats.numpy()
    opacities = loaded.opacities.numpy()
    sh_coeffs = loaded.sh_coeffs.numpy()

    assert means.shape == (4, 3)
    assert scales.shape == (4, 3)
    assert quats.shape == (4, 4)
    assert opacities.shape == (4,)
    assert sh_coeffs.shape == (4, 2, 3)

    assert np.allclose(means, expected["means"], atol=2e-3)
    assert np.allclose(scales, expected["scales"], atol=2e-3)

    # Quaternion sign is equivalent for q and -q; compare by absolute dot product.
    quat_alignment = np.abs(np.sum(quats * expected["quats"], axis=1))
    assert np.all(quat_alignment > 0.999)

    loaded_alpha = 1.0 / (1.0 + np.exp(-opacities))
    expected_alpha = 1.0 / (1.0 + np.exp(-expected["opacities"]))
    assert np.allclose(loaded_alpha, expected_alpha, atol=3e-3)

    assert np.allclose(sh_coeffs[:, 0, :], expected["sh_dc"], atol=3e-2)
    assert np.allclose(sh_coeffs[:, 1, :], expected["sh_rest"], atol=4e-2)
