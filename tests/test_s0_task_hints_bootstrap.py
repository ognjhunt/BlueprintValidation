"""Tests for Stage 0 synthetic task-hints bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_validation.common import read_json
from blueprint_validation.config import CameraPathSpec, FacilityConfig, ValidationConfig
from blueprint_validation.rendering.vlm_scene_detector import (
    DetectedRegion,
    SceneDetectionResult,
)
from blueprint_validation.stages.s0_task_hints_bootstrap import (
    TaskHintsBootstrapStage,
    _derive_tasks_from_detections,
)


def test_bootstrap_skips_when_hints_already_exist(tmp_path):
    hints = tmp_path / "existing_task_targets.json"
    hints.write_text(json.dumps({"tasks": [], "manipulation_candidates": []}))

    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=Path("/tmp/a.ply"), task_hints_path=hints)
    stage = TaskHintsBootstrapStage()

    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "skipped"
    assert result.outputs["task_hints_path"] == str(hints)


def test_bootstrap_writes_synthetic_hints_from_vlm(sample_ply, tmp_path, monkeypatch):
    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply)
    stage = TaskHintsBootstrapStage()

    vlm_result = SceneDetectionResult(
        specs=[
            CameraPathSpec(type="manipulation", approach_point=[1.0, 2.0, 0.5], arc_radius_m=0.6),
            CameraPathSpec(type="manipulation", approach_point=[2.0, 3.0, 0.8], arc_radius_m=0.7),
            CameraPathSpec(type="manipulation", approach_point=[4.0, 1.5, 0.2], arc_radius_m=0.9),
        ],
        detections=[
            DetectedRegion(
                label="coffee_mug",
                center_3d=np.array([1.0, 2.0, 0.5], dtype=np.float32),
                extents_3d=np.array([0.12, 0.08, 0.15], dtype=np.float32),
                category="manipulation",
            ),
            DetectedRegion(
                label="cabinet_door",
                center_3d=np.array([2.0, 3.0, 0.8], dtype=np.float32),
                extents_3d=np.array([0.60, 0.04, 0.80], dtype=np.float32),
                category="articulation",
            ),
            DetectedRegion(
                label="loading_dock",
                center_3d=np.array([4.0, 1.5, 0.2], dtype=np.float32),
                extents_3d=np.array([1.50, 1.20, 0.20], dtype=np.float32),
                category="navigation",
            ),
        ],
        scene_type="kitchen",
        suggested_tasks=[
            {"suggested_task": "Pick up the coffee mug"},
            {"suggested_task": "Open the cabinet door"},
        ],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        lambda **kwargs: vlm_result,
    )

    work_dir = tmp_path / "outputs" / "a"
    result = stage.execute(config, fac, work_dir, {})
    assert result.status == "success"
    assert result.metrics["source"] == "vlm"
    assert result.metrics["num_candidates"] == 1
    assert result.metrics["num_articulation"] == 1
    assert result.metrics["num_navigation"] == 1
    assert result.metrics["scene_type"] == "kitchen"

    hints_path = Path(result.outputs["task_hints_path"])
    assert hints_path.exists()
    assert fac.task_hints_path == hints_path

    payload = read_json(hints_path)
    assert payload["bootstrap_generated"] is True
    assert payload["scene_type"] == "kitchen"

    manip = payload["manipulation_candidates"][0]
    assert manip["label"] == "coffee_mug"
    np.testing.assert_allclose(manip["boundingBox"]["extents"], [0.12, 0.08, 0.15], atol=1e-6)
    np.testing.assert_allclose(manip["boundingBox"]["center"], [1.0, 2.0, 0.5], atol=1e-10)

    artic = payload["articulation_hints"][0]
    assert artic["label"] == "cabinet_door"
    np.testing.assert_allclose(artic["boundingBox"]["extents"], [0.60, 0.04, 0.80], atol=1e-6)
    nav = payload["navigation_hints"][0]
    assert nav["label"] == "loading_dock"
    np.testing.assert_allclose(nav["boundingBox"]["extents"], [1.50, 1.20, 0.20], atol=1e-6)

    task_ids = {t["task_id"] for t in payload["tasks"]}
    assert "Pick up the coffee mug" in task_ids
    assert "Open the cabinet door" in task_ids

    # Later runs should deterministically reuse/skip existing hints.
    result2 = stage.execute(config, fac, work_dir, {})
    assert result2.status == "skipped"


def test_bootstrap_falls_back_to_clusters_when_vlm_empty(sample_ply, tmp_path, monkeypatch):
    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply)
    stage = TaskHintsBootstrapStage()

    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        lambda **kwargs: SceneDetectionResult(specs=[], detections=[]),
    )

    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "success"
    assert result.metrics["source"] == "cluster"
    assert result.metrics["num_candidates"] > 0

    payload = read_json(Path(result.outputs["task_hints_path"]))
    extents = [c["boundingBox"]["extents"] for c in payload["manipulation_candidates"]]
    # Cluster extents should not be the fixed legacy 0.35 cube for every object.
    assert any(not np.allclose(e, [0.35, 0.35, 0.35]) for e in extents)


def test_bootstrap_writes_centers_in_original_frame_for_y_up(sample_ply, tmp_path, monkeypatch):
    config = ValidationConfig()
    fac = FacilityConfig(name="A", ply_path=sample_ply, up_axis="y")
    stage = TaskHintsBootstrapStage()

    vlm_result = SceneDetectionResult(
        specs=[CameraPathSpec(type="manipulation", approach_point=[0.0, 0.0, 5.0], arc_radius_m=0.8)],
        detections=[
            DetectedRegion(
                label="target",
                center_3d=np.array([0.0, 0.0, 5.0], dtype=np.float32),
                extents_3d=np.array([0.2, 0.4, 0.6], dtype=np.float32),
                category="manipulation",
            )
        ],
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s0_task_hints_bootstrap.detect_and_generate_specs",
        lambda **kwargs: vlm_result,
    )

    result = stage.execute(config, fac, tmp_path / "outputs" / "a", {})
    assert result.status == "success"
    payload = read_json(Path(result.outputs["task_hints_path"]))
    center = payload["manipulation_candidates"][0]["boundingBox"]["center"]
    extents = payload["manipulation_candidates"][0]["boundingBox"]["extents"]
    np.testing.assert_allclose(center, [0.0, 5.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(extents, [0.2, 0.6, 0.4], atol=1e-8)


def test_derive_tasks_from_detections_prefers_suggestions_then_categories():
    detections = [
        DetectedRegion(label="mug", center_3d=np.zeros(3), category="manipulation"),
        DetectedRegion(label="door", center_3d=np.ones(3), category="articulation"),
    ]
    suggested = [{"suggested_task": "Pick up the mug"}]

    tasks = _derive_tasks_from_detections(detections, suggested, scene_type="office")
    task_ids = {t["task_id"] for t in tasks}

    assert "Pick up the mug" in task_ids
    assert "open_close_access_points" in task_ids
