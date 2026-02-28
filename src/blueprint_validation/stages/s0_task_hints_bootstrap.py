"""Stage 0: Bootstrap synthetic task hints for facilities without source metadata."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import CameraPathSpec, FacilityConfig, ValidationConfig
from ..rendering.scene_geometry import (
    OrientedBoundingBox,
    cluster_scene_points_with_extents,
    compute_camera_height,
    compute_scene_transform,
    compute_standoff_distance,
    detect_up_axis,
    is_identity_transform,
    transform_means,
)
from ..rendering.vlm_scene_detector import (
    DetectedRegion,
    SceneDetectionResult,
    detect_and_generate_specs,
)
from ..warmup import load_ply_means_and_colors_numpy
from .base import PipelineStage

logger = get_logger("stages.s0_task_hints_bootstrap")

_DEFAULT_HINTS_FILE = "task_targets.synthetic.json"


def _default_output_path(work_dir: Path) -> Path:
    return work_dir / "bootstrap" / _DEFAULT_HINTS_FILE


# ---------------------------------------------------------------------------
# Task derivation (scene-agnostic)
# ---------------------------------------------------------------------------

_CATEGORY_TO_TASK = {
    "manipulation": "pick_place_manipulation",
    "articulation": "open_close_access_points",
    # "navigation" has no downstream prompt mapping — intentionally omitted so
    # navigation detections don't emit unrecognized task IDs.
}


def _derive_tasks_from_detections(
    detections: List[DetectedRegion],
    suggested_tasks: List[dict],
    scene_type: str,
) -> List[dict]:
    """Derive task list from VLM detections and suggestions.

    Priority:
      1. VLM-suggested tasks (deduped by task description)
      2. Tasks inferred from detection categories
      3. Fallback ``pick_place_manipulation``
    """
    tasks: List[dict] = []
    seen: set = set()

    # 1. VLM-suggested tasks
    for st in suggested_tasks:
        desc = st.get("suggested_task") or st.get("task")
        if desc and desc not in seen:
            tasks.append({"task_id": desc, "source": "vlm_suggestion", "scene_type": scene_type})
            seen.add(desc)

    # 2. Category-inferred tasks
    categories_present = {d.category for d in detections}
    for cat in sorted(categories_present):
        task_id = _CATEGORY_TO_TASK.get(cat, f"{cat}_generic")
        if task_id not in seen:
            tasks.append({"task_id": task_id, "source": "category_inference", "scene_type": scene_type})
            seen.add(task_id)

    # 3. Fallback
    if not tasks:
        tasks.append({"task_id": "pick_place_manipulation", "source": "fallback", "scene_type": scene_type})

    return tasks


# ---------------------------------------------------------------------------
# Camera specs from geometric clusters
# ---------------------------------------------------------------------------


def _camera_specs_from_clusters(
    means: np.ndarray,
    max_specs: int = 6,
) -> Tuple[List[CameraPathSpec], np.ndarray]:
    """Cluster points and return camera specs + cluster extents."""
    cr = cluster_scene_points_with_extents(
        means=means.astype(np.float32),
        num_clusters=max_specs,
        max_points=25_000,
        seed=17,
    )
    specs: List[CameraPathSpec] = []
    selected_extents: List[np.ndarray] = []
    # Prefer dense clusters over outliers/noise.
    order = np.argsort(-cr.point_counts)
    for idx in order[:max_specs]:
        center = cr.centers[idx]
        ext = cr.extents[idx]
        obb = OrientedBoundingBox(
            instance_id="cluster",
            label="cluster",
            center=center.astype(np.float64),
            extents=ext.astype(np.float64),
            axes=np.eye(3),
        )
        standoff = compute_standoff_distance(obb)
        cam_height = compute_camera_height(obb)
        specs.append(
            CameraPathSpec(
                type="manipulation",
                approach_point=center.astype(np.float64).tolist(),
                arc_radius_m=standoff,
                height_override_m=cam_height,
                look_down_override_deg=45.0,
            )
        )
        selected_extents.append(ext.astype(np.float64))
    if selected_extents:
        extents = np.asarray(selected_extents, dtype=np.float64)
    else:
        extents = np.zeros((0, 3), dtype=np.float64)
    return specs, extents


# ---------------------------------------------------------------------------
# Inverse-frame helper
# ---------------------------------------------------------------------------


def _to_original_frame(points_corrected: np.ndarray, scene_T: np.ndarray) -> np.ndarray:
    """Map points from corrected (Z-up) frame back to native PLY frame."""
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = scene_T[:3, :3].T
    return transform_means(points_corrected, T_inv)


def _extents_to_original_frame(extents_corrected: np.ndarray, scene_T: np.ndarray) -> np.ndarray:
    """Map axis-aligned extents from corrected frame back to native PLY frame."""
    if extents_corrected.size == 0:
        return extents_corrected
    # Convert lengths under rotation by projecting through |R|.
    r_inv = scene_T[:3, :3].T
    return (np.abs(r_inv) @ extents_corrected.T).T


# ---------------------------------------------------------------------------
# Synthetic task hints builder
# ---------------------------------------------------------------------------


def _build_synthetic_task_hints(
    centers_original: np.ndarray,
    source: str,
    facility: FacilityConfig,
    scene_T: np.ndarray,
    tasks: List[dict],
    scene_type: str = "unknown",
    labels: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    extents: Optional[np.ndarray] = None,
) -> Dict:
    """Build the task_targets.synthetic.json payload.

    Accepts per-object labels, categories, and extents from VLM detections
    or cluster analysis. Falls back to generic values when not provided.
    """
    n = len(centers_original)
    if labels is None:
        labels = [f"target_{i + 1:02d}" for i in range(n)]
    if categories is None:
        categories = ["manipulation"] * n
    if extents is None:
        extents = np.full((n, 3), 0.35, dtype=np.float64)

    manipulation_candidates = []
    articulation_hints = []
    navigation_hints = []

    for idx, (center, label, cat, ext) in enumerate(
        zip(centers_original, labels, categories, extents)
    ):
        entry = {
            "instance_id": f"bootstrap_{label}",
            "label": label,
            "category": cat,
            "confidence": 0.6,
            "source": source,
            "boundingBox": {
                "center": np.asarray(center, dtype=float).tolist(),
                "extents": np.asarray(ext, dtype=float).tolist(),
                "axes": np.eye(3, dtype=np.float64).tolist(),
            },
        }

        if cat == "articulation":
            articulation_hints.append(entry)
        elif cat == "navigation":
            navigation_hints.append(entry)
        else:
            manipulation_candidates.append(entry)

    return {
        "bootstrap_generated": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "facility_name": facility.name,
        "source": source,
        "scene_type": scene_type,
        "tasks": tasks,
        "manipulation_candidates": manipulation_candidates,
        "articulation_hints": articulation_hints,
        "navigation_hints": navigation_hints,
        "scene_transform": scene_T.tolist(),
    }


# ---------------------------------------------------------------------------
# Stage implementation
# ---------------------------------------------------------------------------


class TaskHintsBootstrapStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s0_task_hints_bootstrap"

    @property
    def description(self) -> str:
        return "Bootstrap synthetic task_targets.json from first-pass VLM detections"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del previous_results

        if facility.task_hints_path is not None and facility.task_hints_path.exists():
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                outputs={"task_hints_path": str(facility.task_hints_path)},
                detail="facility.task_hints_path already exists",
            )

        output_path = facility.task_hints_path or _default_output_path(work_dir)
        if output_path.exists():
            facility.task_hints_path = output_path
            existing = read_json(output_path)
            return StageResult(
                stage_name=self.name,
                status="success",
                elapsed_seconds=0,
                outputs={"task_hints_path": str(output_path)},
                metrics={
                    "bootstrap_reused": True,
                    "num_candidates": len(existing.get("manipulation_candidates", [])),
                },
            )

        if not config.render.vlm_fallback:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="render.vlm_fallback=false and no task hints present",
            )

        if not facility.ply_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"PLY file not found: {facility.ply_path}",
            )

        # --- Load PLY with colors ---
        means_raw, colors = load_ply_means_and_colors_numpy(facility.ply_path)
        if len(means_raw) == 0:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"No points loaded from PLY: {facility.ply_path}",
            )

        # --- Resolve up-axis and compute scene transform ---
        resolved_facility = facility
        if facility.up_axis.lower().strip() == "auto":
            detected = detect_up_axis(means_raw)
            resolved_facility = replace(facility, up_axis=detected)
        else:
            detected = facility.up_axis

        scene_T = compute_scene_transform(resolved_facility)
        has_transform = not is_identity_transform(scene_T)
        if not has_transform:
            means_corrected = means_raw
        else:
            means_corrected = transform_means(means_raw, scene_T)
        scene_center = means_corrected.mean(axis=0)

        # --- VLM detection (with colors) ---
        detections: List[DetectedRegion] = []
        scene_type = "unknown"
        suggested_tasks: List[dict] = []
        vlm_result: Optional[SceneDetectionResult] = None
        source = "vlm"

        try:
            vlm_result = detect_and_generate_specs(
                splat_means_np=means_corrected,
                scene_center=scene_center,
                num_views=config.render.vlm_fallback_num_views,
                model=config.render.vlm_fallback_model,
                resolution=config.render.resolution,
                splat_colors=colors,
            )
            detections = vlm_result.detections
            scene_type = vlm_result.scene_type
            suggested_tasks = vlm_result.suggested_tasks
        except Exception:
            logger.warning("VLM bootstrap detection failed; falling back to geometric clusters", exc_info=True)
            vlm_result = None

        # --- Fallback to geometric clusters if VLM produced nothing ---
        cluster_extents: Optional[np.ndarray] = None
        if vlm_result is None or not vlm_result.specs:
            specs, cluster_extents = _camera_specs_from_clusters(means_corrected)
            source = "cluster"
        else:
            specs = vlm_result.specs

        approach_points_corr: List[List[float]] = [
            list(spec.approach_point)
            for spec in specs
            if spec.approach_point is not None
        ]
        if not approach_points_corr:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="No candidate regions generated for synthetic task hints",
            )

        # --- Build per-object metadata ---
        n = len(approach_points_corr)
        if source == "vlm" and detections:
            labels = [d.label for d in detections[:n]]
            categories = [d.category for d in detections[:n]]
            obj_extents = np.array([d.extents_3d for d in detections[:n]])
            # Pad if fewer detections than specs (shouldn't happen, but defensive)
            while len(labels) < n:
                labels.append(f"target_{len(labels) + 1:02d}")
                categories.append("manipulation")
            if len(obj_extents) < n:
                pad = np.full((n - len(obj_extents), 3), 0.35)
                obj_extents = np.vstack([obj_extents, pad])
        elif cluster_extents is not None:
            labels = [f"target_{i + 1:02d}" for i in range(n)]
            categories = ["manipulation"] * n
            obj_extents = cluster_extents[:n].astype(np.float64)
            if len(obj_extents) < n:
                pad = np.full((n - len(obj_extents), 3), 0.35)
                obj_extents = np.vstack([obj_extents, pad])
        else:
            labels = None
            categories = None
            obj_extents = None

        # --- Derive tasks ---
        tasks = _derive_tasks_from_detections(detections, suggested_tasks, scene_type)

        # --- Convert centers back to original PLY frame ---
        centers_corrected = np.asarray(approach_points_corr, dtype=np.float64)
        if not has_transform:
            centers_original = centers_corrected
        else:
            centers_original = _to_original_frame(centers_corrected, scene_T)

        if obj_extents is not None and has_transform:
            obj_extents = _extents_to_original_frame(np.asarray(obj_extents, dtype=np.float64), scene_T)

        payload = _build_synthetic_task_hints(
            centers_original=centers_original,
            source=source,
            facility=facility,
            scene_T=scene_T,
            tasks=tasks,
            scene_type=scene_type,
            labels=labels,
            categories=categories,
            extents=obj_extents,
        )
        payload["resolved_up_axis"] = resolved_facility.up_axis
        if facility.up_axis.lower().strip() == "auto":
            payload["detected_up_axis"] = detected
        write_json(payload, output_path)
        facility.task_hints_path = output_path

        num_manip = len(payload["manipulation_candidates"])
        num_artic = len(payload["articulation_hints"])
        num_nav = len(payload.get("navigation_hints", []))

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "task_hints_path": str(output_path),
            },
            metrics={
                "bootstrap_reused": False,
                "num_candidates": num_manip,
                "num_articulation": num_artic,
                "num_navigation": num_nav,
                "source": source,
                "scene_type": scene_type,
                "resolved_up_axis": resolved_facility.up_axis,
            },
        )
