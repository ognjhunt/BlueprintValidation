"""Lightweight downstream review for already-qualified opportunities."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def build_lightweight_downstream_review(handoff: Mapping[str, Any]) -> Dict[str, Any]:
    qualification_state = str(handoff.get("qualification_state") or "not_ready_yet").strip().lower()
    eligible = bool(handoff.get("downstream_evaluation_eligibility"))
    scene_memory_package = (
        handoff.get("scene_memory_package")
        if isinstance(handoff.get("scene_memory_package"), Mapping)
        else {}
    )
    geometry_package = handoff.get("geometry_package") if isinstance(handoff.get("geometry_package"), Mapping) else {}
    scene_package = handoff.get("scene_package") if isinstance(handoff.get("scene_package"), Mapping) else {}
    target_robot_team = handoff.get("target_robot_team") if isinstance(handoff.get("target_robot_team"), Mapping) else {}

    review_mode = "lightweight_advisory"
    next_step = "Provide a named robot team/platform before full downstream evaluation."
    if target_robot_team:
        review_mode = "robot_specific_review"
        next_step = (
            "Run bounded downstream evaluation against the scoped task using the preferred "
            "scene-memory bundle when available, with geometry or strict simulator adapters "
            "only when justified."
        )

    return {
        "schema_version": "v1",
        "review_mode": review_mode,
        "qualification_state": qualification_state,
        "downstream_evaluation_eligibility": eligible,
        "has_scene_memory_bundle": bool(scene_memory_package),
        "has_geometry_bundle": bool(geometry_package),
        "has_scene_package": bool(scene_package),
        "has_target_robot_team": bool(target_robot_team),
        "can_run_full_pipeline": bool(eligible and target_robot_team),
        "next_step": next_step,
    }
