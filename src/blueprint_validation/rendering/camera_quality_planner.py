"""Stage-1 camera path candidate generation and geometric quality planning."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from ..config import CameraPathSpec
from ..evaluation.camera_quality import analyze_target_visibility, project_target_to_poses
from .camera_paths import generate_path_from_spec

_BUDGET_TO_VARIANTS = {
    "low": [
        (0.00, 0.00, 0.0),
        (-0.08, 0.00, 0.0),
        (0.08, 0.00, 0.0),
    ],
    "medium": [
        (0.00, 0.00, 0.0),
        (-0.10, 0.00, 0.0),
        (0.10, 0.00, 0.0),
        (0.00, 0.06, 3.0),
        (0.00, -0.06, -3.0),
    ],
    "high": [
        (0.00, 0.00, 0.0),
        (-0.15, 0.00, 0.0),
        (0.15, 0.00, 0.0),
        (-0.08, 0.04, 2.0),
        (0.08, 0.04, 2.0),
        (-0.08, -0.04, -2.0),
        (0.08, -0.04, -2.0),
        (0.00, 0.08, 4.0),
        (0.00, -0.08, -4.0),
    ],
}


def plan_best_camera_spec(
    *,
    base_spec: CameraPathSpec,
    scene_center: np.ndarray,
    num_frames: int,
    camera_height: float,
    look_down_deg: float,
    resolution: tuple[int, int],
    start_offset: np.ndarray | None,
    manipulation_target_z_bias_m: float,
    budget: str,
    min_visible_frame_ratio: float,
    min_center_band_ratio: float,
    min_approach_angle_bins: int,
    angle_bin_deg: float,
    center_band_x: object,
    center_band_y: object,
) -> tuple[CameraPathSpec, int, Dict[str, float]]:
    """Return the best candidate spec for this path under deterministic bounded search."""
    candidates = _generate_candidate_specs(
        base_spec=base_spec,
        camera_height=float(camera_height),
        look_down_deg=float(look_down_deg),
        budget=str(budget or "medium").strip().lower(),
    )
    if len(candidates) <= 1:
        return base_spec, 1, {"planner_best_score": 0.0}

    best_spec = candidates[0]
    best_score = -1e9
    best_metrics: Dict[str, float] = {}
    best_idx = 0
    target_xyz = _resolve_target_xyz(base_spec)

    for idx, candidate in enumerate(candidates):
        poses = generate_path_from_spec(
            spec=candidate,
            scene_center=scene_center,
            num_frames=int(num_frames),
            camera_height=float(camera_height),
            look_down_deg=float(look_down_deg),
            resolution=resolution,
            start_offset=start_offset,
            manipulation_target_z_bias_m=float(manipulation_target_z_bias_m),
        )
        score, metrics = _score_candidate_geometric(
            poses=poses,
            target_xyz=target_xyz,
            min_visible_frame_ratio=float(min_visible_frame_ratio),
            min_center_band_ratio=float(min_center_band_ratio),
            min_approach_angle_bins=int(min_approach_angle_bins),
            angle_bin_deg=float(angle_bin_deg),
            center_band_x=center_band_x,
            center_band_y=center_band_y,
        )
        # Tie-break by earlier index to keep deterministic behavior stable.
        if score > best_score + 1e-9:
            best_score = score
            best_spec = candidate
            best_metrics = metrics
            best_idx = idx
        elif abs(score - best_score) <= 1e-9 and idx < best_idx:
            best_spec = candidate
            best_metrics = metrics
            best_idx = idx

    planner_metrics = {
        "planner_best_score": round(float(best_score), 6),
        **best_metrics,
    }
    return best_spec, len(candidates), planner_metrics


def _generate_candidate_specs(
    *,
    base_spec: CameraPathSpec,
    camera_height: float,
    look_down_deg: float,
    budget: str,
) -> List[CameraPathSpec]:
    offsets = _BUDGET_TO_VARIANTS.get(budget, _BUDGET_TO_VARIANTS["medium"])
    dedupe: Dict[tuple, CameraPathSpec] = {}
    for scale_delta, height_delta, look_delta in offsets:
        if base_spec.type == "manipulation":
            arc = max(0.15, float(base_spec.arc_radius_m) * (1.0 + float(scale_delta)))
            h0 = (
                float(base_spec.height_override_m)
                if base_spec.height_override_m is not None
                else float(camera_height)
            )
            l0 = (
                float(base_spec.look_down_override_deg)
                if base_spec.look_down_override_deg is not None
                else float(look_down_deg)
            )
            cand = replace(
                base_spec,
                arc_radius_m=float(arc),
                height_override_m=float(max(0.25, h0 + float(height_delta))),
                look_down_override_deg=float(max(5.0, min(80.0, l0 + float(look_delta)))),
            )
        elif base_spec.type == "orbit":
            radius = max(0.30, float(base_spec.radius_m) * (1.0 + float(scale_delta)))
            h0 = (
                float(base_spec.height_override_m)
                if base_spec.height_override_m is not None
                else float(camera_height)
            )
            l0 = (
                float(base_spec.look_down_override_deg)
                if base_spec.look_down_override_deg is not None
                else float(look_down_deg)
            )
            cand = replace(
                base_spec,
                radius_m=float(radius),
                height_override_m=float(max(0.25, h0 + float(height_delta))),
                look_down_override_deg=float(max(2.0, min(80.0, l0 + float(look_delta)))),
            )
        elif base_spec.type == "sweep":
            length = max(0.30, float(base_spec.length_m) * (1.0 + float(scale_delta)))
            h0 = (
                float(base_spec.height_override_m)
                if base_spec.height_override_m is not None
                else float(camera_height)
            )
            l0 = (
                float(base_spec.look_down_override_deg)
                if base_spec.look_down_override_deg is not None
                else float(look_down_deg)
            )
            cand = replace(
                base_spec,
                length_m=float(length),
                height_override_m=float(max(0.25, h0 + float(height_delta))),
                look_down_override_deg=float(max(2.0, min(80.0, l0 + float(look_delta)))),
            )
        else:
            cand = base_spec

        key = (
            str(cand.type),
            round(float(cand.arc_radius_m), 6),
            round(float(cand.radius_m), 6),
            round(float(cand.length_m), 6),
            (
                None
                if cand.height_override_m is None
                else round(float(cand.height_override_m), 6)
            ),
            (
                None
                if cand.look_down_override_deg is None
                else round(float(cand.look_down_override_deg), 6)
            ),
        )
        dedupe[key] = cand
    return list(dedupe.values())


def _resolve_target_xyz(spec: CameraPathSpec) -> List[float] | None:
    point = getattr(spec, "approach_point", None)
    if isinstance(point, list) and len(point) >= 3:
        try:
            return [float(point[0]), float(point[1]), float(point[2])]
        except Exception:
            return None
    return None


def _score_candidate_geometric(
    *,
    poses: List[object],
    target_xyz: List[float] | None,
    min_visible_frame_ratio: float,
    min_center_band_ratio: float,
    min_approach_angle_bins: int,
    angle_bin_deg: float,
    center_band_x: object,
    center_band_y: object,
) -> tuple[float, Dict[str, float]]:
    if not poses:
        return -1e6, {
            "planner_visible_ratio": 0.0,
            "planner_center_band_ratio": 0.0,
            "planner_approach_angle_bins": 0.0,
        }
    if target_xyz is None:
        return 0.50, {
            "planner_visible_ratio": 0.0,
            "planner_center_band_ratio": 0.0,
            "planner_approach_angle_bins": 0.0,
        }

    total_frames, visible_samples = project_target_to_poses(poses, target_xyz)
    visible, total, center, angle_bins = analyze_target_visibility(
        total_frames=total_frames,
        visible_samples=visible_samples,
        angle_bin_deg=float(angle_bin_deg),
        center_band_x=center_band_x,
        center_band_y=center_band_y,
    )
    if total <= 0:
        return -1e5, {
            "planner_visible_ratio": 0.0,
            "planner_center_band_ratio": 0.0,
            "planner_approach_angle_bins": 0.0,
        }

    visible_ratio = float(visible) / float(total)
    center_ratio = float(center) / float(total)
    angle_count = float(len(angle_bins))
    visible_component = min(1.0, max(0.0, visible_ratio / max(float(min_visible_frame_ratio), 1e-6)))
    center_component = min(1.0, max(0.0, center_ratio / max(float(min_center_band_ratio), 1e-6)))
    angle_component = min(
        1.0,
        max(0.0, angle_count / max(float(min_approach_angle_bins), 1.0)),
    )
    score = 0.45 * visible_component + 0.35 * center_component + 0.20 * angle_component
    return score, {
        "planner_visible_ratio": round(visible_ratio, 6),
        "planner_center_band_ratio": round(center_ratio, 6),
        "planner_approach_angle_bins": float(angle_count),
    }
