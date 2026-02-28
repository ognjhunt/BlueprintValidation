"""Tests for gsplat renderer wrappers."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from blueprint_validation.rendering.gsplat_renderer import render_frame
from blueprint_validation.rendering.ply_loader import GaussianSplatData

torch = pytest.importorskip("torch")


class _DummyPose:
    width = 8
    height = 6

    def viewmat(self):
        return torch.eye(4, dtype=torch.float32)

    def K(self):
        return torch.eye(3, dtype=torch.float32)


def test_render_frame_expands_background_for_rgb_ed(monkeypatch):
    captured = {}

    def fake_rasterization(**kwargs):
        captured["background_shape"] = tuple(kwargs["backgrounds"].shape)
        h = kwargs["height"]
        w = kwargs["width"]
        renders = torch.zeros((1, h, w, 4), dtype=torch.float32)
        alphas = torch.zeros((1, h, w, 1), dtype=torch.float32)
        info = {}
        return renders, alphas, info

    monkeypatch.setitem(sys.modules, "gsplat", types.SimpleNamespace(rasterization=fake_rasterization))

    splat = GaussianSplatData(
        means=torch.zeros((1, 3), dtype=torch.float32),
        scales=torch.zeros((1, 3), dtype=torch.float32),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        opacities=torch.zeros((1,), dtype=torch.float32),
        sh_coeffs=torch.zeros((1, 1, 3), dtype=torch.float32),
        num_points=1,
    )

    rgb, depth = render_frame(splat, _DummyPose(), background=np.array([1.0, 1.0, 1.0]))

    assert captured["background_shape"] == (1, 4)
    assert rgb.shape == (6, 8, 3)
    assert depth.shape == (6, 8)
