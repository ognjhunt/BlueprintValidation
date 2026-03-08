"""Tests for PolaRiS OpenVLA client helpers."""

from __future__ import annotations

import numpy as np
import pytest


def test_stitch_external_wrist_images_concatenates_width():
    from blueprint_validation.polaris.openvla_client import stitch_external_wrist_images

    external = np.zeros((8, 10, 3), dtype=np.uint8)
    wrist = np.ones((8, 6, 3), dtype=np.uint8)
    stitched = stitch_external_wrist_images(external, wrist)
    assert stitched.shape == (8, 16, 3)
    assert stitched[:, :10].sum() == 0
    assert stitched[:, 10:].sum() == wrist.sum()


def test_extract_policy_observation_supports_external_wrist_stitched():
    from blueprint_validation.polaris.openvla_client import extract_policy_observation

    obs = {
        "splat": {
            "external_cam": np.zeros((4, 5, 3), dtype=np.uint8),
            "wrist_cam": np.ones((4, 2, 3), dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": np.arange(7, dtype=np.float32),
            "gripper_pos": np.array([0.5], dtype=np.float32),
        },
    }
    request = extract_policy_observation(obs, "external_wrist_stitched")
    assert request["image"].shape == (4, 7, 3)
    assert request["joint_position"].shape == (7,)
    assert request["gripper_position"].shape == (1,)


def test_normalize_openvla_action_rejects_non_finite():
    from blueprint_validation.polaris.openvla_client import normalize_openvla_action

    with pytest.raises(ValueError, match="non-finite"):
        normalize_openvla_action([0, 1, np.nan])


def test_normalize_openvla_action_rejects_wrong_dim():
    from blueprint_validation.polaris.openvla_client import normalize_openvla_action

    with pytest.raises(ValueError, match="Expected action_dim=7"):
        normalize_openvla_action(np.zeros((6,), dtype=np.float32), expected_dim=7)
