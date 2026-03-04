"""Deterministic Stage-1 active-perception helpers.

This module keeps Stage-1 VLM-critic behavior reproducible:
- budget policy resolution
- clip scope filtering
- candidate score fusion
- issue-tag to camera-parameter corrections
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List

from ..config import CameraPathSpec


ALLOWED_STAGE1_ISSUE_TAGS = {
    "target_missing",
    "target_off_center",
    "target_occluded",
    "camera_too_far",
    "camera_too_close",
    "camera_motion_too_fast",
    "blur_or_soft_focus",
    "unstable_view",
}


@dataclass(frozen=True)
class ProbeBudget:
    top_k: int
    probe_frames: int
    max_loops: int
    probe_resolution_scale: float


_BUDGET_PRESETS: dict[str, ProbeBudget] = {
    "low": ProbeBudget(top_k=1, probe_frames=9, max_loops=1, probe_resolution_scale=0.50),
    "medium": ProbeBudget(top_k=2, probe_frames=13, max_loops=2, probe_resolution_scale=0.67),
    "high": ProbeBudget(top_k=3, probe_frames=17, max_loops=2, probe_resolution_scale=0.75),
}


def resolve_probe_budget(
    *,
    candidate_budget: str,
    max_loops_cap: int,
    probe_frames_override: int,
    probe_resolution_scale_override: float,
) -> ProbeBudget:
    key = str(candidate_budget or "medium").strip().lower()
    base = _BUDGET_PRESETS.get(key, _BUDGET_PRESETS["medium"])
    loops = min(max(0, int(max_loops_cap)), int(base.max_loops))
    frames = int(base.probe_frames)
    if int(probe_frames_override) > 0:
        frames = int(probe_frames_override)
    scale = float(base.probe_resolution_scale)
    if float(probe_resolution_scale_override) > 0.0:
        scale = float(probe_resolution_scale_override)
    return ProbeBudget(
        top_k=max(1, int(base.top_k)),
        probe_frames=max(1, int(frames)),
        max_loops=max(0, int(loops)),
        probe_resolution_scale=max(0.1, min(1.0, float(scale))),
    )


def should_probe_clip(
    *,
    scope: str,
    path_type: str,
    path_context: dict | None,
) -> bool:
    normalized_scope = str(scope or "all").strip().lower()
    ptype = str(path_type or "").strip().lower()
    if normalized_scope == "all":
        return True
    if normalized_scope == "manipulation":
        return ptype == "manipulation"

    # targeted
    if ptype == "manipulation":
        return True
    if not isinstance(path_context, dict):
        return False
    for key in ("target_instance_id", "target_label", "target_category"):
        value = str(path_context.get(key, "")).strip()
        if value:
            return True
    point = path_context.get("approach_point")
    return isinstance(point, list) and len(point) >= 3


def compute_probe_resolution(
    *,
    base_resolution: tuple[int, int],
    scale: float,
) -> tuple[int, int]:
    h = max(1, int(base_resolution[0]))
    w = max(1, int(base_resolution[1]))
    sc = max(0.1, min(1.0, float(scale)))
    out_h = max(64, int(round(h * sc)))
    out_w = max(64, int(round(w * sc)))
    # Keep even dimensions for codec compatibility.
    if out_h % 2 == 1:
        out_h += 1
    if out_w % 2 == 1:
        out_w += 1
    return out_h, out_w


def combined_probe_score(
    *,
    geometric_score: float,
    task_score: float,
    visual_score: float,
    spatial_score: float,
) -> float:
    probe_norm = (
        max(0.0, min(10.0, float(task_score)))
        + max(0.0, min(10.0, float(visual_score)))
        + max(0.0, min(10.0, float(spatial_score)))
    ) / 30.0
    geom_norm = max(0.0, min(1.0, float(geometric_score)))
    return float(0.85 * probe_norm + 0.15 * geom_norm)


def probe_passes_thresholds(
    *,
    task_score: float,
    visual_score: float,
    spatial_score: float,
    min_task: float,
    min_visual: float,
    min_spatial: float,
) -> bool:
    return (
        float(task_score) >= float(min_task)
        and float(visual_score) >= float(min_visual)
        and float(spatial_score) >= float(min_spatial)
    )


def normalize_issue_tags(tags: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in tags:
        key = str(tag).strip().lower()
        if not key or key in seen:
            continue
        if key not in ALLOWED_STAGE1_ISSUE_TAGS:
            continue
        seen.add(key)
        out.append(key)
    return out


def apply_issue_tag_corrections(
    *,
    spec: CameraPathSpec,
    issue_tags: Iterable[str],
    default_camera_height: float,
    default_look_down_deg: float,
) -> CameraPathSpec:
    """Deterministic tag->camera correction matrix for bounded retries."""
    tags = normalize_issue_tags(issue_tags)
    updated = spec

    if "target_missing" in tags and str(updated.type).strip().lower() != "manipulation":
        updated = _convert_to_manipulation(
            updated,
            default_camera_height=float(default_camera_height),
            default_look_down_deg=float(default_look_down_deg),
        )

    if "camera_too_far" in tags:
        updated = _scale_standoff(updated, orbit_scale=0.88, sweep_scale=0.90, manip_scale=0.88)
    if "camera_too_close" in tags:
        updated = _scale_standoff(updated, orbit_scale=1.12, sweep_scale=1.10, manip_scale=1.12)

    if "target_off_center" in tags:
        updated = _add_look_down(
            updated,
            delta_deg=3.0,
            default_look_down_deg=float(default_look_down_deg),
        )
        updated = _scale_standoff(updated, orbit_scale=0.94, sweep_scale=1.0, manip_scale=0.94)

    if "target_occluded" in tags:
        updated = _add_height(
            updated,
            delta_m=0.06,
            default_camera_height=float(default_camera_height),
        )
        updated = _scale_standoff(updated, orbit_scale=1.08, sweep_scale=1.0, manip_scale=1.08)

    apply_motion_reduction = (
        "camera_motion_too_fast" in tags
        or "unstable_view" in tags
        or "blur_or_soft_focus" in tags
    )
    if apply_motion_reduction:
        updated = _apply_motion_reduction(updated)

    if "blur_or_soft_focus" in tags:
        updated = _add_look_down(
            updated,
            delta_deg=2.0,
            default_look_down_deg=float(default_look_down_deg),
        )

    return updated


def _convert_to_manipulation(
    spec: CameraPathSpec,
    *,
    default_camera_height: float,
    default_look_down_deg: float,
) -> CameraPathSpec:
    point = getattr(spec, "approach_point", None)
    if not (isinstance(point, list) and len(point) >= 3):
        # No approach point to aim at — zoom in aggressively and lower camera
        # to give the corrected render its best chance of capturing a target.
        return _scale_standoff(spec, orbit_scale=0.60, sweep_scale=0.60, manip_scale=0.60)

    if str(spec.type).strip().lower() == "orbit":
        base_radius = max(0.2, min(1.8, float(spec.radius_m) * 0.8))
    elif str(spec.type).strip().lower() == "sweep":
        base_radius = max(0.2, min(1.8, float(spec.length_m) * 0.20))
    else:
        base_radius = 0.45

    h0 = (
        float(spec.height_override_m)
        if spec.height_override_m is not None
        else float(default_camera_height)
    )
    l0 = (
        float(spec.look_down_override_deg)
        if spec.look_down_override_deg is not None
        else float(default_look_down_deg)
    )

    return CameraPathSpec(
        type="manipulation",
        approach_point=[float(point[0]), float(point[1]), float(point[2])],
        arc_radius_m=float(max(0.15, base_radius)),
        arc_span_deg=120.0,
        arc_phase_offset_deg=float(getattr(spec, "arc_phase_offset_deg", 0.0)),
        height_override_m=float(max(0.25, h0)),
        look_down_override_deg=float(max(5.0, min(80.0, l0))),
        source_tag=spec.source_tag,
        target_instance_id=spec.target_instance_id,
        target_label=spec.target_label,
        target_category=spec.target_category,
        target_role=spec.target_role,
    )


def _scale_standoff(
    spec: CameraPathSpec,
    *,
    orbit_scale: float,
    sweep_scale: float,
    manip_scale: float,
) -> CameraPathSpec:
    stype = str(spec.type).strip().lower()
    if stype == "orbit":
        return replace(spec, radius_m=float(max(0.30, float(spec.radius_m) * float(orbit_scale))))
    if stype == "sweep":
        return replace(spec, length_m=float(max(0.30, float(spec.length_m) * float(sweep_scale))))
    if stype == "manipulation":
        return replace(
            spec,
            arc_radius_m=float(max(0.15, float(spec.arc_radius_m) * float(manip_scale))),
        )
    return spec


def _add_height(
    spec: CameraPathSpec,
    *,
    delta_m: float,
    default_camera_height: float,
) -> CameraPathSpec:
    h0 = (
        float(spec.height_override_m)
        if spec.height_override_m is not None
        else float(default_camera_height)
    )
    return replace(spec, height_override_m=float(max(0.25, h0 + float(delta_m))))


def _add_look_down(
    spec: CameraPathSpec,
    *,
    delta_deg: float,
    default_look_down_deg: float,
) -> CameraPathSpec:
    l0 = (
        float(spec.look_down_override_deg)
        if spec.look_down_override_deg is not None
        else float(default_look_down_deg)
    )
    stype = str(spec.type).strip().lower()
    lo_min = 5.0 if stype == "manipulation" else 2.0
    updated = max(lo_min, min(80.0, l0 + float(delta_deg)))
    return replace(spec, look_down_override_deg=float(updated))


def _apply_motion_reduction(spec: CameraPathSpec) -> CameraPathSpec:
    stype = str(spec.type).strip().lower()
    if stype == "orbit":
        cur_orbits = int(spec.num_orbits)
        new_orbits = max(1, cur_orbits - 1)
        # When already at minimum orbits, reduce radius more aggressively (slows angular velocity).
        radius_scale = 0.78 if cur_orbits <= 1 else 0.88
        return replace(
            spec,
            num_orbits=new_orbits,
            radius_m=float(max(0.30, float(spec.radius_m) * radius_scale)),
        )
    if stype == "sweep":
        return replace(spec, length_m=float(max(0.30, float(spec.length_m) * 0.78)))
    if stype == "manipulation":
        return replace(
            spec,
            arc_radius_m=float(max(0.15, float(spec.arc_radius_m) * 0.82)),
            arc_span_deg=float(max(40.0, min(300.0, float(spec.arc_span_deg) * 0.72))),
        )
    return spec
