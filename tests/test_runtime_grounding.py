from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from blueprint_validation.runtime_grounding import (
    GroundingBundle,
    render_grounded_frame,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def _bundle(tmp_path: Path, *, regions: list[dict], render_policy: dict | None = None, variance_policy: dict | None = None) -> GroundingBundle:
    frame_path = tmp_path / "canonical.png"
    cv2.imwrite(str(frame_path), np.full((48, 64, 3), 96, dtype=np.uint8))
    protected_path = tmp_path / "protected_regions_manifest.json"
    render_policy_path = tmp_path / "canonical_render_policy.json"
    variance_path = tmp_path / "presentation_variance_policy.json"
    protected = {"schema_version": "v1", "regions": regions}
    render_policy = {"schema_version": "v1", "retry_budget": 1, "degraded_quality_threshold": 0.40, **(render_policy or {})}
    variance_policy = {"schema_version": "v1", **(variance_policy or {})}
    _write_json(protected_path, protected)
    _write_json(render_policy_path, render_policy)
    _write_json(variance_path, variance_policy)
    return GroundingBundle(
        canonical_package_version="pkg-v1",
        canonical_package_uri="canonical://pkg-v1",
        canonical_frame_path=frame_path,
        protected_regions_manifest_path=protected_path,
        canonical_render_policy_path=render_policy_path,
        presentation_variance_policy_path=variance_path,
        protected_regions_manifest=protected,
        canonical_render_policy=render_policy,
        presentation_variance_policy=variance_policy,
    )


def test_missing_provenance_defaults_to_editable(tmp_path: Path) -> None:
    left_mask = np.zeros((48, 64), dtype=bool)
    left_mask[:, :32] = True
    left_path = tmp_path / "left.png"
    _write_mask(left_path, left_mask)
    bundle = _bundle(
        tmp_path,
        regions=[
            {
                "region_id": "missing-provenance",
                "mask_path": str(left_path),
                "confidence": 0.95,
            }
        ],
    )

    rendered = render_grounded_frame(
        bundle,
        camera_id="head_rgb",
        prompt="prompt-a",
        trajectory="static",
        presentation_model="model-a",
        debug_mode=False,
        step_index=0,
        output_dir=tmp_path,
    )

    assert bool(np.any(rendered.locked_mask)) is False
    assert bool(np.any(rendered.editable_mask[:, :32])) is True


def test_task_critical_region_is_dilated(tmp_path: Path) -> None:
    tiny_mask = np.zeros((48, 64), dtype=bool)
    tiny_mask[24, 32] = True
    tiny_path = tmp_path / "tiny.png"
    _write_mask(tiny_path, tiny_mask)
    bundle = _bundle(
        tmp_path,
        regions=[
            {
                "region_id": "task-critical",
                "mask_path": str(tiny_path),
                "provenance": "observed",
                "observed_coverage": 0.95,
                "confidence": 0.9,
                "task_critical": True,
            }
        ],
    )

    rendered = render_grounded_frame(
        bundle,
        camera_id="head_rgb",
        prompt="prompt-a",
        trajectory="static",
        presentation_model="model-a",
        debug_mode=False,
        step_index=0,
        output_dir=tmp_path,
    )

    assert int(np.count_nonzero(rendered.locked_mask)) > 1


def test_low_confidence_reprojection_falls_back_to_editable(tmp_path: Path) -> None:
    locked_mask = np.zeros((48, 64), dtype=bool)
    locked_mask[:, :20] = True
    locked_path = tmp_path / "locked.png"
    _write_mask(locked_path, locked_mask)
    bundle = _bundle(
        tmp_path,
        regions=[
            {
                "region_id": "reprojection-weak",
                "mask_path": str(locked_path),
                "provenance": "observed",
                "observed_coverage": 0.95,
                "confidence": 0.9,
                "reprojection_confidence": 0.2,
            }
        ],
    )

    rendered = render_grounded_frame(
        bundle,
        camera_id="head_rgb",
        prompt="prompt-a",
        trajectory="move_right",
        presentation_model="model-a",
        debug_mode=False,
        step_index=0,
        output_dir=tmp_path,
    )

    assert bool(np.any(rendered.locked_mask)) is False
    assert bool(np.any(rendered.editable_mask)) is True


def test_degraded_quality_and_forced_violation_fallback(tmp_path: Path) -> None:
    locked_mask = np.zeros((48, 64), dtype=bool)
    locked_mask[:, :8] = True
    locked_path = tmp_path / "locked.png"
    _write_mask(locked_path, locked_mask)
    bundle = _bundle(
        tmp_path,
        regions=[
            {
                "region_id": "locked",
                "mask_path": str(locked_path),
                "provenance": "observed",
                "observed_coverage": 0.95,
                "confidence": 0.9,
            }
        ],
        variance_policy={"test_force_locked_violation_always": True},
    )

    rendered = render_grounded_frame(
        bundle,
        camera_id="head_rgb",
        prompt="prompt-a",
        trajectory="static",
        presentation_model="model-a",
        debug_mode=True,
        step_index=0,
        output_dir=tmp_path,
    )

    assert rendered.quality_flags["presentation_quality"] == "degraded"
    assert rendered.quality_flags["fallback_mode"] == "canonical_only"
    assert rendered.protected_region_violations["count"] > 0
    assert Path(rendered.debug_artifacts["final_composite"]).is_file()
