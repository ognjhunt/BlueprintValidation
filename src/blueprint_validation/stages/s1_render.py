"""Stage 1: Render Gaussian splat PLY to video clips via gsplat."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, List, Optional

import numpy as np

from ..common import StageResult, get_logger, sanitize_filename_component, write_json
from ..config import CameraPathSpec, FacilityConfig, VLMJudgeConfig, ValidationConfig
from ..evaluation.camera_quality import (
    analyze_target_visibility,
    evaluate_clip_quality,
    project_target_to_poses,
)
from ..evaluation.expected_focus import build_expected_focus_text as _build_expected_focus_text
from ..evaluation.task_hints import tasks_from_task_hints
from ..evaluation.vlm_judge import Stage1ProbeScore, score_stage1_probe
from ..enrichment.stage2_quality import evaluate_stage1_coverage_gate
from ..rendering.camera_paths import CameraPose, _look_at, generate_path_from_spec, save_path_to_json
from ..rendering.camera_quality_planner import (
    plan_best_camera_spec,  # noqa: F401
    rank_camera_spec_candidates,
)
from ..rendering.stage1_active_perception import (
    apply_issue_tag_corrections,
    combined_probe_score,
    compute_probe_resolution,
    probe_passes_thresholds,
    resolve_probe_budget,
    should_probe_clip,
)
from ..rendering.gsplat_renderer import render_video
from ..rendering.ply_loader import load_splat
from ..rendering.scene_geometry import (
    OrientedBoundingBox,
    OccupancyGrid,
    auto_populate_manipulation_zones,
    build_occupancy_grid,
    correct_upside_down_camera_poses,
    compute_scene_transform,
    filter_and_fix_poses,
    generate_scene_aware_specs,
    is_identity_transform,
    load_obbs_from_task_targets,
    resolve_facility_orientation,
    transform_camera_path_specs,
    transform_c2w,
    transform_means,
    transform_obbs,
)
from ..video_io import ensure_h264_video
from ..warmup import load_cached_clips, load_warmup_cache
from ..validation import ManifestValidationError, load_and_validate_manifest
from .base import PipelineStage

logger = get_logger("stages.s1_render")

_TARGET_PRESENCE_STRICT_MIN_VISIBLE_RATIO = 0.80
_TARGET_PRESENCE_STRICT_MIN_CENTER_RATIO = 0.70
# LOS is evaluated against a camera-facing target surface endpoint (not center).
# Keep this lower than visible/center ratios to avoid over-rejecting partially
# occluded but still task-usable views in dense kitchen scenes.
_TARGET_PRESENCE_STRICT_MIN_LOS_RATIO = 0.30
_TARGET_PRESENCE_STRICT_MIN_SIZE_RATIO = 0.07
_SCENE_LOCKED_SOURCE_TAG_PREFIX = "scene_locked:"
_KITCHEN_0787_LOCKED_SOURCE_TAG = "kitchen_0787_locked"
_FACILITY_A_LOCKED_SOURCE_TAG = f"{_SCENE_LOCKED_SOURCE_TAG_PREFIX}facility_a"
_KITCHEN_0787_LOCKED_MAX_TARGETS = 12
_FACILITY_A_LOCKED_MAX_TARGETS = 8
_SCENE_LOCKED_DEFAULTS: Dict[str, Dict[str, object]] = {
    "kitchen_0787": {
        "source_tag": _KITCHEN_0787_LOCKED_SOURCE_TAG,
        "max_targets": _KITCHEN_0787_LOCKED_MAX_TARGETS,
        "eye_offset_m": (0.78, -0.40, 0.34),
        "look_at_offset_m": (0.0, 0.0, 0.05),
        "probe_motion_radius_m": 0.006,
    },
    "facility_a": {
        "source_tag": _FACILITY_A_LOCKED_SOURCE_TAG,
        "max_targets": _FACILITY_A_LOCKED_MAX_TARGETS,
        "eye_offset_m": (0.95, -0.55, 0.38),
        "look_at_offset_m": (0.0, 0.0, 0.05),
        "probe_motion_radius_m": 0.008,
    },
}
# Backward-compatible names for the existing kitchen-specific path.
_KITCHEN_0787_LOCKED_DEFAULT_EYE_OFFSET_M = _SCENE_LOCKED_DEFAULTS["kitchen_0787"]["eye_offset_m"]
_KITCHEN_0787_LOCKED_DEFAULT_LOOK_AT_OFFSET_M = _SCENE_LOCKED_DEFAULTS["kitchen_0787"]["look_at_offset_m"]
# Keep probes near-static to reduce blur and preserve deterministic framing.
_KITCHEN_0787_LOCKED_DEFAULT_PROBE_MOTION_RADIUS_M = _SCENE_LOCKED_DEFAULTS["kitchen_0787"]["probe_motion_radius_m"]
# Absolute world-space camera table calibrated once for kitchen_0787 targets.
# Values are in the facility raw world frame (before any scene_transform).
_KITCHEN_0787_LOCKED_TARGET_POSE_TABLE: Dict[str, Dict[str, object]] = {
    "190": {  # bowl
        "eye_world_m": (-2.650050, 0.381133, 0.993174),
        "look_at_world_m": (-2.321157, 0.500840, 0.923174),
        "probe_motion_radius_m": 0.006,
    },
    "100": {  # dining table
        "eye_world_m": (-5.323978, -2.360751, 0.953628),
        "look_at_world_m": (-5.017560, -2.103635, 0.386780),
        "probe_motion_radius_m": 0.008,
    },
    "88": {  # pot
        "eye_world_m": (-2.293672, -1.109695, 1.064196),
        "look_at_world_m": (-2.643672, -1.109695, 0.884196),
        "probe_motion_radius_m": 0.006,
    },
    "102": {  # rice cooker
        "eye_world_m": (-4.768049, -0.312433, 1.124793),
        "look_at_world_m": (-5.168049, -0.312433, 1.024793),
        "probe_motion_radius_m": 0.008,
    },
    "157": {  # trash can (dining side)
        "eye_world_m": (2.108512, -2.685071, 0.310867),
        "look_at_world_m": (1.762102, -2.885071, 0.210867),
        "probe_motion_radius_m": 0.008,
    },
    "161": {  # trash can (kitchen side)
        "eye_world_m": (0.965335, 0.649215, 0.280509),
        "look_at_world_m": (1.341212, 0.786023, 0.180509),
        "probe_motion_radius_m": 0.008,
    },
    "105": {  # cupboard
        "eye_world_m": (-3.596826, 0.357602, 0.992629),
        "look_at_world_m": (-3.996826, 0.357602, 0.640393),
        "probe_motion_radius_m": 0.008,
    },
    "61": {  # door
        "eye_world_m": (-0.246415, 1.024736, 1.724800),
        "look_at_world_m": (-0.646415, 1.024736, 1.098000),
        "probe_motion_radius_m": 0.008,
    },
    "62": {  # door
        "eye_world_m": (-2.587842, -3.128122, 1.724800),
        "look_at_world_m": (-2.724650, -3.503999, 1.098000),
        "probe_motion_radius_m": 0.008,
    },
    "67": {  # window
        "eye_world_m": (-0.782650, 4.851871, 1.860662),
        "look_at_world_m": (-0.782650, 4.451871, 1.520414),
        "probe_motion_radius_m": 0.008,
    },
    "68": {  # window
        "eye_world_m": (-5.219785, 2.887736, 2.020662),
        "look_at_world_m": (-5.619785, 2.887736, 1.620414),
        "probe_motion_radius_m": 0.008,
    },
    "186": {  # floor lamp
        "eye_world_m": (-0.468377, -1.593787, 1.094312),
        "look_at_world_m": (-0.068377, -1.593787, 0.703945),
        "probe_motion_radius_m": 0.008,
    },
}
# Placeholder scene-locked calibration table for facility_a.
# Fill these exact entries from the first tiny GPU calibration render.
_FACILITY_A_LOCKED_TARGET_POSE_TABLE: Dict[str, Dict[str, object]] = {
    "101": {  # bowl
        "eye_world_m": None,
        "look_at_world_m": None,
        "probe_motion_radius_m": 0.008,
    },
    "102": {  # bottle
        "eye_world_m": None,
        "look_at_world_m": None,
        "probe_motion_radius_m": 0.008,
    },
    "103": {  # mug
        "eye_world_m": None,
        "look_at_world_m": None,
        "probe_motion_radius_m": 0.008,
    },
    "region::prep_counter": {
        "eye_world_m": None,
        "look_at_world_m": None,
        "probe_motion_radius_m": 0.010,
    },
    "region::sink": {
        "eye_world_m": None,
        "look_at_world_m": None,
        "probe_motion_radius_m": 0.010,
    },
    "region::pantry_shelf": {
        "eye_world_m": None,
        "look_at_world_m": None,
        "probe_motion_radius_m": 0.010,
    },
}


class RenderStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1_render"

    @property
    def description(self) -> str:
        return "Render PLY Gaussian splat to video clips at robot-height perspectives"

    def _build_scene_aware_specs(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        splat_means_np: np.ndarray,
        scene_center: np.ndarray,
        scene_T: Optional[np.ndarray] = None,
    ) -> tuple[List[CameraPathSpec], Optional[OccupancyGrid]]:
        """Build scene-aware camera specs and occupancy grid when enabled."""
        occupancy: Optional[OccupancyGrid] = None
        extra_specs: List[CameraPathSpec] = []

        if not config.render.scene_aware:
            return extra_specs, occupancy

        # Build occupancy grid for collision avoidance
        if config.render.collision_check:
            occupancy = build_occupancy_grid(
                splat_means_np,
                voxel_size=config.render.voxel_size_m,
                density_threshold=config.render.density_threshold,
            )

        # Primary path: load OBBs from task_targets.json
        if facility.task_hints_path and facility.task_hints_path.exists():
            obbs = load_obbs_from_task_targets(facility.task_hints_path)
            if obbs:
                if scene_T is not None:
                    obbs = transform_obbs(obbs, scene_T)
                selected_obbs = obbs
                if config.render.task_scoped_scene_aware:
                    task_pool = _build_task_prompt_pool(
                        config=config,
                        facility=facility,
                        profile=config.render.task_scoped_profile,
                    )
                    selected_obbs, scoped_stats, role_by_instance = _select_task_scoped_obbs(
                        obbs=obbs,
                        tasks=task_pool,
                        max_specs=max(1, int(config.render.task_scoped_max_specs)),
                        context_per_target=max(
                            0, int(config.render.task_scoped_context_per_target)
                        ),
                        overview_specs=max(0, int(config.render.task_scoped_overview_specs)),
                        fallback_specs=max(1, int(config.render.task_scoped_fallback_specs)),
                        center_dedupe_dist_m=float(
                            max(0.0, float(config.render.stage1_probe_dedupe_center_dist_m))
                        ),
                    )
                    logger.info(
                        "Task-scoped scene-aware selection: %d/%d OBBs "
                        "(targets=%d context=%d overview=%d fallback=%d)",
                        len(selected_obbs),
                        len(obbs),
                        scoped_stats.get("targets", 0),
                        scoped_stats.get("context", 0),
                        scoped_stats.get("overview", 0),
                        scoped_stats.get("fallback", 0),
                    )
                else:
                    role_by_instance = {}
                extra_specs = generate_scene_aware_specs(
                    selected_obbs,
                    occupancy,
                    target_roles_by_instance=role_by_instance,
                )
                if config.render.task_scoped_scene_aware:
                    extra_specs = [replace(spec, source_tag="task_scoped") for spec in extra_specs]
                facility.manipulation_zones = auto_populate_manipulation_zones(
                    facility.manipulation_zones, selected_obbs
                )
            return extra_specs, occupancy

        # Fallback: VLM detection on rendered overview frames
        if config.render.vlm_fallback:
            extra_specs = self._vlm_fallback(config, splat_means_np, scene_center)

        return extra_specs, occupancy

    def _vlm_fallback(
        self,
        config: ValidationConfig,
        splat_means_np: np.ndarray,
        scene_center: np.ndarray,
    ) -> List[CameraPathSpec]:
        """Attempt VLM-based object detection as a fallback."""
        try:
            from ..rendering.vlm_scene_detector import detect_and_generate_specs

            result = detect_and_generate_specs(
                splat_means_np=splat_means_np,
                scene_center=scene_center,
                num_views=config.render.vlm_fallback_num_views,
                model=config.render.vlm_fallback_model,
                resolution=config.render.resolution,
            )
            if isinstance(result, list):
                # Backward compatibility with older detector return shape.
                return result
            return list(getattr(result, "specs", []))
        except Exception:
            logger.warning(
                "VLM fallback detection failed; continuing with naive paths", exc_info=True
            )
            return []

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        if bool(config.render.stage1_strict_require_task_hints):
            if facility.task_hints_path is None or not Path(facility.task_hints_path).exists():
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "Stage 1 strict task-hints mode requires facility.task_hints_path to exist. "
                        "Provide task_targets.synthetic.json/task_targets.json and rerun."
                    ),
                    outputs={
                        "error_code": "s1_task_hints_required_missing",
                        "task_hints_path": (
                            str(facility.task_hints_path) if facility.task_hints_path is not None else None
                        )
                    },
                    metrics={"error_code": "s1_task_hints_required_missing"},
                )

        if (
            bool(config.render.stage1_active_perception_enabled)
            and bool(config.render.stage1_active_perception_fail_closed)
        ):
            missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
            if missing:
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "Stage 1 strict active-perception requires ffmpeg/ffprobe at runtime. "
                        f"Missing tools: {', '.join(missing)}."
                    ),
                    outputs={
                        "missing_tools": missing,
                        "error_code": "s1_missing_runtime_tools",
                    },
                    metrics={
                        "error_code": "s1_missing_runtime_tools",
                        "missing_tools_count": len(missing),
                    },
                )

        render_dir = work_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        probe_scores_path = (
            work_dir / "s1_probe_scores.jsonl"
            if bool(config.render.stage1_active_perception_enabled)
            else None
        )

        # Load the Gaussian splat
        device = "cuda" if _has_cuda() else "cpu"
        logger.info("Loading splat from %s (device=%s)", facility.ply_path, device)
        splat = load_splat(facility.ply_path, device=device)

        resolution = config.render.resolution
        num_frames = config.render.num_frames
        camera_height = config.render.camera_height_m
        look_down_deg = config.render.camera_look_down_deg
        fps = config.render.fps

        # Check for warmup cache — skip CPU-heavy prep if available
        cached_clips = load_cached_clips(work_dir)
        warmup_cache = load_warmup_cache(work_dir)
        if bool(config.render.stage1_active_perception_enabled):
            if cached_clips is not None:
                logger.info(
                    "Ignoring warmup cache because Stage-1 active perception is enabled."
                )
            cached_clips = None
            warmup_cache = None
        extra_specs_count = 0
        num_poses_roll_corrected = 0
        num_clips_with_roll_correction = 0
        quality_summary: Dict[str, object] = _empty_quality_summary()

        splat_means_raw = splat.means.cpu().numpy()
        obbs_for_orientation: Optional[List[OrientedBoundingBox]] = None
        if facility.task_hints_path and facility.task_hints_path.exists():
            try:
                obbs_for_orientation = load_obbs_from_task_targets(facility.task_hints_path)
            except Exception:
                logger.warning(
                    "Failed loading OBBs for orientation scoring from %s",
                    facility.task_hints_path,
                    exc_info=True,
                )

        facility, orientation_meta = resolve_facility_orientation(
            facility=facility,
            means_raw=splat_means_raw,
            obbs_raw=obbs_for_orientation,
            camera_look_down_deg=config.render.camera_look_down_deg,
            orientation_autocorrect_enabled=config.render.orientation_autocorrect_enabled,
            orientation_autocorrect_mode=config.render.orientation_autocorrect_mode,
        )

        if cached_clips is not None and warmup_cache is not None:
            cached_axis = str(warmup_cache.get("resolved_up_axis", "")).strip().lower()
            current_axis = str(facility.up_axis).strip().lower()
            if cached_axis and cached_axis != current_axis:
                logger.info(
                    "Ignoring warmup cache due to orientation mismatch "
                    "(cache=%s, current=%s)",
                    cached_axis,
                    current_axis,
                )
                cached_clips = None
                warmup_cache = None
            else:
                expected_quality_cache_key = _quality_cache_key(
                    config=config,
                    task_hints_path=facility.task_hints_path,
                )
                cached_quality_cache_key = str(warmup_cache.get("quality_cache_key", "")).strip()
                if cached_quality_cache_key != expected_quality_cache_key:
                    logger.info(
                        "Ignoring warmup cache due to quality planner mismatch "
                        "(cache_key=%s current_key=%s)",
                        cached_quality_cache_key,
                        expected_quality_cache_key,
                    )
                    cached_clips = None
                    warmup_cache = None

        # Compute scene orientation transform (e.g. Y-up → Z-up)
        scene_T = compute_scene_transform(facility)
        has_transform = not is_identity_transform(scene_T)
        if has_transform:
            logger.info("Scene transform active (up_axis=%s)", facility.up_axis)

        if cached_clips is not None:
            logger.info(
                "Using warmup cache: %d pre-computed clips (skipping occupancy grid + path gen)",
                len(cached_clips),
            )
            extra_specs_count = (warmup_cache or {}).get("scene_aware_specs", 0)
            (
                manifest_entries,
                num_poses_roll_corrected,
                num_clips_with_roll_correction,
                quality_summary,
            ) = self._render_from_cache(
                cached_clips,
                splat,
                render_dir,
                resolution,
                fps,
                config,
                scene_T if has_transform else None,
            )
        else:
            splat_means_np = splat_means_raw

            # Transform positions for scene geometry (camera paths, occupancy)
            if has_transform:
                splat_means_np = transform_means(splat_means_np, scene_T)
            scene_center = splat_means_np.mean(axis=0)
            scene_locked_profile = _resolve_scene_locked_profile(config, facility)
            if scene_locked_profile and (
                not bool(config.render.stage1_active_perception_enabled)
                or not bool(config.render.stage1_active_perception_fail_closed)
            ):
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        f"{scene_locked_profile} scene-locked mode requires "
                        "render.stage1_active_perception_enabled=true and "
                        "render.stage1_active_perception_fail_closed=true."
                    ),
                    outputs={
                        "render_dir": str(render_dir),
                        "facility_name": facility.name,
                        "scene_locked_profile": scene_locked_profile,
                        "error_code": "s1_scene_locked_requires_active_perception",
                    },
                    metrics={"error_code": "s1_scene_locked_requires_active_perception"},
                )

            # Scene-aware camera placement
            extra_specs, occupancy = self._build_scene_aware_specs(
                config, facility, splat_means_np, scene_center, scene_T if has_transform else None
            )
            extra_specs_count = len(extra_specs)

            base_specs = list(config.render.camera_paths)
            if has_transform:
                base_specs = transform_camera_path_specs(base_specs, scene_T)
            if scene_locked_profile:
                all_path_specs = _build_scene_locked_specs(
                    config=config,
                    facility=facility,
                    scene_transform=(scene_T if has_transform else None),
                    profile=scene_locked_profile,
                )
                if not all_path_specs:
                    return StageResult(
                        stage_name=self.name,
                        status="failed",
                        elapsed_seconds=0,
                        detail=(
                            f"{scene_locked_profile} scene-locked mode could not build any target-grounded "
                            "camera specs from task hints."
                        ),
                        outputs={
                            "render_dir": str(render_dir),
                            "task_hints_path": (
                                str(facility.task_hints_path)
                                if facility.task_hints_path is not None
                                else None
                            ),
                            "scene_locked_profile": scene_locked_profile,
                            "error_code": "s1_scene_locked_no_specs",
                        },
                        metrics={"error_code": "s1_scene_locked_no_specs"},
                    )
                extra_specs_count = len(all_path_specs)
                logger.info(
                    "%s scene-locked mode active: using %d deterministic target-grounded specs",
                    scene_locked_profile,
                    len(all_path_specs),
                )
            else:
                all_path_specs = base_specs + extra_specs
                if (
                    extra_specs
                    and bool(config.render.task_scoped_scene_aware)
                    and str(config.render.stage1_active_perception_scope).strip().lower()
                    == "targeted"
                ):
                    # Fast-validation mode: evaluate targeted, object-grounded specs before
                    # generic seed paths so early canary probes reflect task-focused capture quality.
                    all_path_specs = extra_specs + base_specs
                    logger.info(
                        "Using targeted-first path ordering for Stage-1 active-perception scope=targeted "
                        "(targeted_specs=%d seed_specs=%d)",
                        len(extra_specs),
                        len(base_specs),
                    )
                if extra_specs:
                    logger.info(
                        "Added %d scene-aware camera paths (total: %d)",
                        len(extra_specs),
                        len(all_path_specs),
                    )

            (
                manifest_entries,
                num_poses_roll_corrected,
                num_clips_with_roll_correction,
                quality_summary,
            ) = self._generate_and_render(
                config,
                splat,
                all_path_specs,
                scene_center,
                occupancy,
                render_dir,
                num_frames,
                camera_height,
                look_down_deg,
                resolution,
                fps,
                scene_T if has_transform else None,
                facility.description,
                probe_orientation_fix=str(getattr(facility, "video_orientation_fix", "none")),
                probe_scores_path=probe_scores_path,
            )

        quality_gate_passed = bool(
            int(quality_summary.get("num_quality_failures", 0)) == 0
            and (
                not bool(config.render.stage1_active_perception_enabled)
                or not bool(config.render.stage1_active_perception_fail_closed)
                or int(quality_summary.get("num_vlm_probe_failures", 0)) == 0
            )
        )
        quality_summary["quality_gate_passed"] = quality_gate_passed
        stage1_run_metadata = _build_stage1_run_metadata(config=config)
        enforce_stage1_fail = (
            bool(config.render.stage1_quality_planner_enabled)
            or (
                bool(config.render.stage1_active_perception_enabled)
                and bool(config.render.stage1_active_perception_fail_closed)
            )
        )
        if enforce_stage1_fail and not quality_gate_passed:
            detail_parts = [
                "Stage 1 quality gate failed after bounded regeneration."
            ]
            if int(quality_summary.get("num_quality_failures", 0)) > 0:
                detail_parts.append(
                    "cv_quality_failed="
                    f"{int(quality_summary.get('num_quality_failures', 0))}"
                )
                detail_parts.append(
                    "cv_retries="
                    f"{int(quality_summary.get('num_quality_retries', 0))}"
                )
            if (
                bool(config.render.stage1_active_perception_enabled)
                and bool(config.render.stage1_active_perception_fail_closed)
                and int(quality_summary.get("num_vlm_probe_failures", 0)) > 0
            ):
                detail_parts.append(
                    "vlm_probe_failed="
                    f"{int(quality_summary.get('num_vlm_probe_failures', 0))}"
                )
                detail_parts.append(
                    "vlm_probe_retries="
                    f"{int(quality_summary.get('num_vlm_probe_retries', 0))}"
                )
            detail = " ".join(detail_parts)
            manifest_path = render_dir / "render_manifest.json"
            manifest = {
                "facility": facility.name,
                "ply_path": str(facility.ply_path),
                "num_clips": len(manifest_entries),
                "scene_aware": config.render.scene_aware,
                "scene_aware_specs": extra_specs_count,
                "git_commit": stage1_run_metadata["git_commit"],
                "config_hash": stage1_run_metadata["config_hash"],
                "stage1_code_hash": stage1_run_metadata["stage1_code_hash"],
                "active_model_used": stage1_run_metadata["active_model_used"],
                "clips": manifest_entries,
            }
            write_json(manifest, manifest_path)
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                outputs={
                    "render_dir": str(render_dir),
                    "manifest_path": str(manifest_path),
                    "num_clips": len(manifest_entries),
                    "resolved_up_axis": facility.up_axis,
                    "error_code": "s1_quality_gate_failed",
                },
                metrics={
                    "num_clips": len(manifest_entries),
                    "total_frames": sum(e.get("num_frames", 0) for e in manifest_entries),
                    "resolution": list(resolution),
                    "scene_aware_specs": extra_specs_count,
                    "resolved_up_axis": facility.up_axis,
                    "orientation_candidates": orientation_meta.get("orientation_candidates"),
                    "orientation_score_selected": orientation_meta.get("orientation_score_selected"),
                    "orientation_score_runner_up": orientation_meta.get("orientation_score_runner_up"),
                    "num_poses_roll_corrected": num_poses_roll_corrected,
                    "num_clips_with_roll_correction": num_clips_with_roll_correction,
                    "git_commit": stage1_run_metadata["git_commit"],
                    "config_hash": stage1_run_metadata["config_hash"],
                    "stage1_code_hash": stage1_run_metadata["stage1_code_hash"],
                    "active_model_used": stage1_run_metadata["active_model_used"],
                    "error_code": "s1_quality_gate_failed",
                    **quality_summary,
                },
                detail=detail,
            )

        # Write manifest
        manifest_path = render_dir / "render_manifest.json"
        manifest = {
            "facility": facility.name,
            "ply_path": str(facility.ply_path),
            "num_clips": len(manifest_entries),
            "scene_aware": config.render.scene_aware,
            "scene_aware_specs": extra_specs_count,
            "git_commit": stage1_run_metadata["git_commit"],
            "config_hash": stage1_run_metadata["config_hash"],
            "stage1_code_hash": stage1_run_metadata["stage1_code_hash"],
            "active_model_used": stage1_run_metadata["active_model_used"],
            "clips": manifest_entries,
        }
        write_json(manifest, manifest_path)

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,  # filled by execute()
            outputs={
                "render_dir": str(render_dir),
                "manifest_path": str(manifest_path),
                "num_clips": len(manifest_entries),
                "resolved_up_axis": facility.up_axis,
            },
            metrics={
                "num_clips": len(manifest_entries),
                "total_frames": sum(e["num_frames"] for e in manifest_entries),
                "resolution": list(resolution),
                "scene_aware_specs": extra_specs_count,
                "resolved_up_axis": facility.up_axis,
                "orientation_candidates": orientation_meta.get("orientation_candidates"),
                "orientation_score_selected": orientation_meta.get("orientation_score_selected"),
                "orientation_score_runner_up": orientation_meta.get("orientation_score_runner_up"),
                "num_poses_roll_corrected": num_poses_roll_corrected,
                "num_clips_with_roll_correction": num_clips_with_roll_correction,
                "git_commit": stage1_run_metadata["git_commit"],
                "config_hash": stage1_run_metadata["config_hash"],
                "stage1_code_hash": stage1_run_metadata["stage1_code_hash"],
                "active_model_used": stage1_run_metadata["active_model_used"],
                **quality_summary,
            },
        )

    def run_geometry_canary(
        self,
        *,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        max_specs: int = 12,
        probe_frames_override: int = 0,
        targeted_only: bool = True,
    ) -> Dict[str, object]:
        """Run Stage-1 geometry-only target-presence checks without rendering.

        This evaluates the same pose-generation and geometric target gates used by
        active-perception probes (visibility, centering, projected size, LOS), but
        skips GPU rendering and VLM scoring entirely.
        """
        render_dir = work_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        rows_path = render_dir / "s1_geometry_canary_rows.jsonl"
        summary_path = render_dir / "s1_geometry_canary_summary.json"
        rows_path.unlink(missing_ok=True)

        logger.info(
            "Running geometry-only canary for facility=%s (max_specs=%d targeted_only=%s)",
            facility.name,
            int(max_specs),
            bool(targeted_only),
        )

        splat = load_splat(facility.ply_path, device="cpu")
        splat_means_raw = splat.means.cpu().numpy()

        obbs_for_orientation: Optional[List[OrientedBoundingBox]] = None
        if facility.task_hints_path and facility.task_hints_path.exists():
            try:
                obbs_for_orientation = load_obbs_from_task_targets(facility.task_hints_path)
            except Exception:
                logger.warning(
                    "Failed loading OBBs for geometry canary orientation scoring from %s",
                    facility.task_hints_path,
                    exc_info=True,
                )

        facility, orientation_meta = resolve_facility_orientation(
            facility=facility,
            means_raw=splat_means_raw,
            obbs_raw=obbs_for_orientation,
            camera_look_down_deg=config.render.camera_look_down_deg,
            orientation_autocorrect_enabled=config.render.orientation_autocorrect_enabled,
            orientation_autocorrect_mode=config.render.orientation_autocorrect_mode,
        )

        scene_T = compute_scene_transform(facility)
        has_transform = not is_identity_transform(scene_T)
        splat_means_np = splat_means_raw
        if has_transform:
            splat_means_np = transform_means(splat_means_np, scene_T)
        scene_center = splat_means_np.mean(axis=0)

        extra_specs, occupancy = self._build_scene_aware_specs(
            config,
            facility,
            splat_means_np,
            scene_center,
            scene_T if has_transform else None,
        )
        if occupancy is None and bool(config.render.collision_check):
            occupancy = build_occupancy_grid(
                splat_means_np,
                voxel_size=config.render.voxel_size_m,
                density_threshold=config.render.density_threshold,
            )

        base_specs = list(config.render.camera_paths)
        if has_transform:
            base_specs = transform_camera_path_specs(base_specs, scene_T)

        scene_locked_profile = _resolve_scene_locked_profile(config, facility)
        if scene_locked_profile:
            all_path_specs = _build_scene_locked_specs(
                config=config,
                facility=facility,
                scene_transform=(scene_T if has_transform else None),
                profile=scene_locked_profile,
            )
        else:
            all_path_specs = base_specs + extra_specs
            if (
                extra_specs
                and bool(config.render.task_scoped_scene_aware)
                and str(config.render.stage1_active_perception_scope).strip().lower() == "targeted"
            ):
                all_path_specs = extra_specs + base_specs

        budget = resolve_probe_budget(
            candidate_budget=config.render.stage1_quality_candidate_budget,
            max_loops_cap=int(config.render.stage1_active_perception_max_loops),
            probe_frames_override=int(config.render.stage1_probe_frames_override),
            probe_resolution_scale_override=float(config.render.stage1_probe_resolution_scale),
        )
        base_resolution = (int(config.render.resolution[0]), int(config.render.resolution[1]))
        probe_resolution = compute_probe_resolution(
            base_resolution=base_resolution,
            scale=float(budget.probe_resolution_scale),
        )
        probe_frames = (
            int(probe_frames_override)
            if int(probe_frames_override) > 0
            else int(max(1, int(budget.probe_frames)))
        )

        rows: List[Dict[str, object]] = []
        clip_index = 0
        max_specs = max(1, int(max_specs))
        for path_spec in all_path_specs:
            path_context = _path_context_from_spec(path_spec)
            is_scene_locked = _is_scene_locked_spec(path_spec)
            target_grounded = bool(
                should_probe_clip(
                    scope="targeted",
                    path_type=str(path_spec.type),
                    path_context=path_context,
                )
                and _is_target_grounded_path_context(path_context)
            )
            if is_scene_locked:
                target_grounded = True
            if bool(targeted_only) and not target_grounded:
                continue
            if len(rows) >= max_specs:
                break

            rng = np.random.default_rng(seed=clip_index * 42)
            offset = _sample_start_offset(
                config,
                path_spec,
                rng,
                clip_repeat_index=0,
                is_task_scoped=(
                    str(getattr(path_spec, "source_tag", "")).strip().lower() == "task_scoped"
                ),
                target_grounded=bool(target_grounded),
            )
            if is_scene_locked:
                offset = np.zeros(3, dtype=np.float64)

            (
                poses,
                pre_count,
                post_filter_count,
                post_resample_count,
                corrected_count,
            ) = self._build_render_poses(
                config=config,
                path_spec=path_spec,
                scene_center=scene_center,
                occupancy=occupancy,
                num_frames=int(probe_frames),
                camera_height=float(config.render.camera_height_m),
                look_down_deg=float(config.render.camera_look_down_deg),
                resolution=probe_resolution,
                start_offset=offset,
            )

            min_viable_ratio = float(config.render.stage1_probe_min_viable_pose_ratio)
            viable_ratio = float(post_filter_count) / max(1.0, float(probe_frames))
            min_unique_positions = int(config.render.stage1_probe_min_unique_positions)
            if is_scene_locked:
                min_unique_positions = min(min_unique_positions, 1)
            unique_positions = _count_unique_camera_positions(poses)
            target_xyz = _path_context_target_xyz(path_context)
            target_extents = _path_context_target_extents(path_context)

            visible_ratio: Optional[float] = None
            center_ratio: Optional[float] = None
            size_ratio: Optional[float] = None
            los_ratio: Optional[float] = None
            min_visible_ratio: Optional[float] = None
            min_center_ratio: Optional[float] = None
            min_los_ratio: Optional[float] = None

            status = "ok"
            reason = ""
            issue_tags: List[str] = []
            if not poses:
                status = "pose_generation_failed"
                reason = "probe_pose_generation_failed"
                if target_grounded:
                    issue_tags = ["target_missing"]
            elif viable_ratio < min_viable_ratio:
                status = "viability_reject"
                reason = (
                    f"probe_viability_reject ratio={viable_ratio:.3f} "
                    f"threshold={min_viable_ratio:.3f}"
                )
                issue_tags = ["camera_motion_too_fast"]
                if target_grounded:
                    issue_tags.append("target_missing")
            elif unique_positions < int(min_unique_positions):
                status = "degenerate_positions"
                reason = (
                    f"probe_degenerate_positions unique={unique_positions} "
                    f"min_required={min_unique_positions}"
                )
                issue_tags = ["unstable_view"]
                if target_grounded:
                    issue_tags.append("target_missing")
            elif target_grounded:
                if target_xyz is None:
                    status = "target_presence_reject"
                    reason = "target_presence_reject missing_target_annotation"
                    issue_tags = ["target_missing"]
                else:
                    total_frames, visible_samples = project_target_to_poses(
                        poses,
                        target_xyz,
                    )
                    visible_frames, total_frames, center_frames, _angle_bins = analyze_target_visibility(
                        total_frames=total_frames,
                        visible_samples=visible_samples,
                        angle_bin_deg=float(config.render.stage1_coverage_angle_bin_deg),
                        center_band_x=config.render.stage1_coverage_center_band_x,
                        center_band_y=config.render.stage1_coverage_center_band_y,
                    )
                    visible_ratio = (
                        float(visible_frames) / float(total_frames) if total_frames > 0 else 0.0
                    )
                    center_ratio = (
                        float(center_frames) / float(total_frames) if total_frames > 0 else 0.0
                    )
                    min_visible_ratio = max(
                        float(config.render.stage1_coverage_min_visible_frame_ratio),
                        float(_TARGET_PRESENCE_STRICT_MIN_VISIBLE_RATIO),
                    )
                    min_center_ratio = max(
                        float(config.render.stage1_coverage_min_center_band_ratio),
                        float(_TARGET_PRESENCE_STRICT_MIN_CENTER_RATIO),
                    )
                    if (
                        float(visible_ratio) < float(min_visible_ratio)
                        or float(center_ratio) < float(min_center_ratio)
                    ):
                        status = "target_presence_reject"
                        reason = (
                            "target_presence_reject "
                            f"visible={float(visible_ratio):.3f}/{float(min_visible_ratio):.3f} "
                            f"center={float(center_ratio):.3f}/{float(min_center_ratio):.3f}"
                        )
                        issue_tags = ["target_missing"]
                        if float(center_ratio) < float(min_center_ratio):
                            issue_tags.append("target_off_center")
                    elif target_extents is not None:
                        size_ratio = _estimate_target_projected_size_ratio(
                            poses=poses,
                            target_xyz=target_xyz,
                            target_extents_m=target_extents,
                        )
                        if float(size_ratio) < float(_TARGET_PRESENCE_STRICT_MIN_SIZE_RATIO):
                            status = "target_presence_reject"
                            reason = (
                                "target_presence_reject "
                                f"size={float(size_ratio):.3f}/"
                                f"{float(_TARGET_PRESENCE_STRICT_MIN_SIZE_RATIO):.3f}"
                            )
                            issue_tags = ["target_missing", "target_off_center"]

                    if status == "ok":
                        los_ratio = _compute_target_line_of_sight_ratio(
                            poses=poses,
                            target_xyz=target_xyz,
                            occupancy=occupancy,
                            min_clearance_m=float(config.render.min_clearance_m),
                        )
                        min_los_ratio = (
                            0.0
                            if bool(is_scene_locked)
                            else float(_TARGET_PRESENCE_STRICT_MIN_LOS_RATIO)
                        )
                        if float(los_ratio) < float(min_los_ratio):
                            status = "target_presence_reject"
                            reason = (
                                "target_presence_reject "
                                f"line_of_sight={float(los_ratio):.3f}/{float(min_los_ratio):.3f}"
                            )
                            issue_tags = ["target_missing", "target_occluded"]

            if target_grounded and not issue_tags and status != "ok":
                issue_tags = ["target_missing"]
            passed = bool(status == "ok")
            row = {
                "clip_name": f"geometry_{clip_index:03d}_{path_spec.type}",
                "clip_index": int(clip_index),
                "path_type": str(path_spec.type),
                "source_tag": str(getattr(path_spec, "source_tag", "") or "default"),
                "status": status,
                "passed": bool(passed),
                "reason": str(reason),
                "issue_tags": issue_tags,
                "target_grounded": bool(target_grounded),
                "target_instance_id": path_context.get("target_instance_id"),
                "target_label": path_context.get("target_label"),
                "target_role": path_context.get("target_role"),
                "target_xyz": target_xyz,
                "target_extents_m": target_extents,
                "num_frames_requested": int(probe_frames),
                "pre_filter_count": int(pre_count),
                "post_filter_count": int(post_filter_count),
                "post_resample_count": int(post_resample_count),
                "corrected_roll_count": int(corrected_count),
                "viable_pose_ratio": round(float(viable_ratio), 6),
                "min_viable_pose_ratio": round(float(min_viable_ratio), 6),
                "unique_positions": int(unique_positions),
                "min_unique_positions": int(min_unique_positions),
                "target_visible_ratio": (
                    None if visible_ratio is None else round(float(visible_ratio), 6)
                ),
                "target_center_ratio": (
                    None if center_ratio is None else round(float(center_ratio), 6)
                ),
                "target_size_ratio": None if size_ratio is None else round(float(size_ratio), 6),
                "target_los_ratio": None if los_ratio is None else round(float(los_ratio), 6),
                "min_target_visible_ratio": (
                    None
                    if min_visible_ratio is None
                    else round(float(min_visible_ratio), 6)
                ),
                "min_target_center_ratio": (
                    None if min_center_ratio is None else round(float(min_center_ratio), 6)
                ),
                "min_target_los_ratio": (
                    None if min_los_ratio is None else round(float(min_los_ratio), 6)
                ),
                "path_context": path_context,
            }
            _append_probe_score_row(rows_path, row)
            rows.append(row)
            clip_index += 1

        target_rows = [row for row in rows if bool(row.get("target_grounded", False))]
        first6 = target_rows[:6]

        def _avg(rows_subset: List[Dict[str, object]], key: str) -> Optional[float]:
            values: List[float] = []
            for item in rows_subset:
                value = item.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if not values:
                return None
            return round(float(np.mean(np.asarray(values, dtype=np.float64))), 6)

        summary: Dict[str, object] = {
            "facility_name": facility.name,
            "facility_description": facility.description,
            "ply_path": str(facility.ply_path),
            "task_hints_path": (
                str(facility.task_hints_path) if facility.task_hints_path is not None else None
            ),
            "kitchen_0787_locked_mode": bool(scene_locked_profile == "kitchen_0787"),
            "scene_locked_profile": scene_locked_profile,
            "targeted_only": bool(targeted_only),
            "probe_frames": int(probe_frames),
            "probe_resolution": [int(probe_resolution[0]), int(probe_resolution[1])],
            "max_specs_requested": int(max_specs),
            "num_specs_available": int(len(all_path_specs)),
            "num_rows": int(len(rows)),
            "num_target_grounded_rows": int(len(target_rows)),
            "num_target_grounded_passed": int(sum(1 for r in target_rows if bool(r.get("passed")))),
            "num_target_grounded_failed": int(
                sum(1 for r in target_rows if not bool(r.get("passed")))
            ),
            "target_missing_count": int(
                sum(
                    1
                    for r in target_rows
                    if "target_missing" in list(r.get("issue_tags", []) or [])
                )
            ),
            "first6_target_grounded_rows": int(len(first6)),
            "first6_target_grounded_passed": int(sum(1 for r in first6 if bool(r.get("passed")))),
            "first6_target_missing_count": int(
                sum(
                    1
                    for r in first6
                    if "target_missing" in list(r.get("issue_tags", []) or [])
                )
            ),
            "first6_avg_target_visible_ratio": _avg(first6, "target_visible_ratio"),
            "first6_avg_target_center_ratio": _avg(first6, "target_center_ratio"),
            "first6_avg_target_size_ratio": _avg(first6, "target_size_ratio"),
            "first6_avg_target_los_ratio": _avg(first6, "target_los_ratio"),
            "orientation_candidates": orientation_meta.get("orientation_candidates"),
            "orientation_score_selected": orientation_meta.get("orientation_score_selected"),
            "orientation_score_runner_up": orientation_meta.get("orientation_score_runner_up"),
            "rows_path": str(rows_path),
            "summary_path": str(summary_path),
        }
        write_json(summary, summary_path)
        return summary

    def run_post_s1_audit(
        self,
        *,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        geometry_max_specs: int = 12,
        geometry_probe_frames_override: int = 0,
        vlm_rescore_first: int = 6,
    ) -> Dict[str, object]:
        """Run CPU-only post-Stage-1 reliability checks in one pass.

        Checks:
        - geometry canary (no render),
        - manifest-level Stage-1 coverage gate,
        - video integrity/codec/decode sanity,
        - clip-level quality metrics from camera path + target metadata,
        - optional VLM rescoring on first N validated clips.
        """
        render_dir = work_dir / "renders"
        audit_dir = render_dir / "post_s1_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        clip_rows_path = audit_dir / "post_s1_audit_clip_rows.jsonl"
        vlm_rows_path = audit_dir / "post_s1_audit_vlm_rows.jsonl"
        summary_path = audit_dir / "post_s1_audit_summary.json"
        clip_rows_path.unlink(missing_ok=True)
        vlm_rows_path.unlink(missing_ok=True)

        manifest_path = render_dir / "render_manifest.json"
        if not manifest_path.exists():
            summary = {
                "status": "failed",
                "detail": (
                    "Render manifest missing. Run Stage 1 render first "
                    "(`... render --facility <id>`)."
                ),
                "facility_name": facility.name,
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
            }
            write_json(summary, summary_path)
            return summary

        try:
            render_manifest = load_and_validate_manifest(
                manifest_path,
                manifest_type="stage1_source",
                require_existing_paths=False,
            )
        except ManifestValidationError as exc:
            summary = {
                "status": "failed",
                "detail": f"Invalid render manifest for post-S1 audit: {exc}",
                "facility_name": facility.name,
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
            }
            write_json(summary, summary_path)
            return summary
        clips_raw = render_manifest.get("clips", [])
        clips: List[Dict[str, object]] = list(clips_raw) if isinstance(clips_raw, list) else []
        clip_by_name: Dict[str, Dict[str, object]] = {}
        for clip in clips:
            clip_name = str(clip.get("clip_name", "")).strip()
            if clip_name and clip_name not in clip_by_name:
                clip_by_name[clip_name] = clip

        logger.info(
            "Running post-S1 audit for facility=%s clips=%d",
            facility.name,
            len(clips),
        )

        geometry_summary: Dict[str, object] = {}
        geometry_error: Optional[str] = None
        try:
            geometry_summary = self.run_geometry_canary(
                config=config,
                facility=facility,
                work_dir=work_dir,
                max_specs=max(1, int(geometry_max_specs)),
                probe_frames_override=max(0, int(geometry_probe_frames_override)),
                targeted_only=True,
            )
        except Exception as exc:
            geometry_error = str(exc)
            logger.warning(
                "post_s1_audit geometry canary failed for facility=%s: %s",
                facility.name,
                exc,
                exc_info=True,
            )

        clip_rows: List[Dict[str, object]] = []
        quality_reason_counter: Counter[str] = Counter()
        validated_video_by_clip: Dict[str, Path] = {}
        for clip_idx, clip in enumerate(clips):
            clip_name = str(clip.get("clip_name", "")).strip() or "unknown"
            safe_clip_name = sanitize_filename_component(clip_name, fallback=f"clip_{clip_idx:04d}")
            path_type = str(clip.get("path_type", "")).strip().lower()
            path_context = clip.get("path_context") or {}
            if not isinstance(path_context, dict):
                path_context = {}
            target_xyz = _path_context_target_xyz(path_context)
            target_grounded = bool(_is_target_grounded_path_context(path_context))
            target_extents = _path_context_target_extents(path_context)

            video_path = Path(str(clip.get("video_path", "")))
            depth_video_path = Path(str(clip.get("depth_video_path", "")))
            camera_path = Path(str(clip.get("camera_path", "")))
            video_exists = bool(video_path.exists())
            depth_exists = bool(depth_video_path.exists())
            camera_path_exists = bool(camera_path.exists())

            validate_status = "ok"
            validate_error = ""
            validated_video_path: Optional[Path] = None
            validated_codec = ""
            decoded_frames = 0
            validated_resolution: Optional[List[int]] = None
            monochrome_warning = False
            content_max_std_dev: Optional[float] = None
            transcoded = False
            if video_exists:
                try:
                    min_frames = max(1, int(clip.get("num_frames", 1) or 1))
                    validated = ensure_h264_video(
                        input_path=video_path,
                        min_decoded_frames=min_frames,
                        output_path=audit_dir / f"{safe_clip_name}_{clip_idx:04d}_audit_h264.mp4",
                        replace_source=False,
                        crf=18,
                        preset="medium",
                    )
                    validated_video_path = Path(validated.path)
                    validated_codec = str(validated.codec_name)
                    decoded_frames = int(validated.decoded_frames)
                    validated_resolution = [
                        int(validated.width or 0),
                        int(validated.height or 0),
                    ]
                    monochrome_warning = bool(validated.content_monochrome_warning)
                    content_max_std_dev = (
                        None
                        if validated.content_max_std_dev is None
                        else float(validated.content_max_std_dev)
                    )
                    transcoded = bool(validated.transcoded)
                except Exception as exc:
                    validate_status = "video_invalid"
                    validate_error = str(exc)
            else:
                validate_status = "video_missing"
                validate_error = f"video missing: {video_path}"

            quality_metrics: Dict[str, object] = {}
            if video_exists and camera_path_exists:
                clip_for_quality = dict(clip)
                if validated_video_path is not None:
                    clip_for_quality["video_path"] = str(validated_video_path)
                quality_metrics = evaluate_clip_quality(
                    clip_entry=clip_for_quality,
                    target_xyz=target_xyz,
                    blur_laplacian_min=float(config.render.stage1_coverage_blur_laplacian_min),
                    min_visible_frame_ratio=float(
                        config.render.stage1_coverage_min_visible_frame_ratio
                    ),
                    min_center_band_ratio=float(config.render.stage1_coverage_min_center_band_ratio),
                    min_approach_angle_bins=int(config.render.stage1_coverage_min_approach_angle_bins),
                    angle_bin_deg=float(config.render.stage1_coverage_angle_bin_deg),
                    center_band_x=config.render.stage1_coverage_center_band_x,
                    center_band_y=config.render.stage1_coverage_center_band_y,
                    blur_sample_every_n_frames=int(
                        config.render.stage1_coverage_blur_sample_every_n_frames
                    ),
                    blur_max_samples=int(config.render.stage1_coverage_blur_max_samples_per_clip),
                    min_clip_score=float(config.render.stage1_quality_min_clip_score),
                    require_target=bool(path_type == "manipulation"),
                )
                for reason in list(quality_metrics.get("quality_reject_reasons", []) or []):
                    quality_reason_counter.update([str(reason)])

            row: Dict[str, object] = {
                "clip_name": clip_name,
                "clip_index": int(clip.get("clip_index", -1) or -1),
                "path_type": path_type,
                "source_tag": str(path_context.get("source_tag", "") or ""),
                "target_grounded": bool(target_grounded),
                "target_instance_id": path_context.get("target_instance_id"),
                "target_label": path_context.get("target_label"),
                "target_role": path_context.get("target_role"),
                "target_xyz": target_xyz,
                "target_extents_m": target_extents,
                "video_path": str(video_path),
                "depth_video_path": str(depth_video_path),
                "camera_path": str(camera_path),
                "video_exists": bool(video_exists),
                "depth_video_exists": bool(depth_exists),
                "camera_path_exists": bool(camera_path_exists),
                "video_validation_status": validate_status,
                "video_validation_error": validate_error,
                "validated_video_path": (None if validated_video_path is None else str(validated_video_path)),
                "validated_codec": validated_codec,
                "validated_decoded_frames": int(decoded_frames),
                "validated_resolution": validated_resolution,
                "validated_transcoded": bool(transcoded),
                "content_monochrome_warning": bool(monochrome_warning),
                "content_max_std_dev": content_max_std_dev,
                **quality_metrics,
            }
            _append_probe_score_row(clip_rows_path, row)
            clip_rows.append(row)
            if validated_video_path is not None:
                validated_video_by_clip[clip_name] = validated_video_path

        # Coverage gate audit from Stage-1 render manifest (forced ON for audit).
        coverage_summary: Dict[str, object] = {}
        coverage_error: Optional[str] = None
        try:
            original_gate_enabled = bool(config.render.stage1_coverage_gate_enabled)
            config.render.stage1_coverage_gate_enabled = True
            try:
                coverage_result = evaluate_stage1_coverage_gate(render_manifest, config)
            finally:
                config.render.stage1_coverage_gate_enabled = original_gate_enabled

            if coverage_result is None:
                coverage_summary = {
                    "coverage_gate_evaluated": False,
                    "coverage_gate_passed": None,
                    "coverage_gate_detail": "coverage gate returned no result",
                    "coverage_metrics": {},
                }
            else:
                coverage_summary = {
                    "coverage_gate_evaluated": True,
                    "coverage_gate_passed": bool(coverage_result.passed),
                    "coverage_gate_detail": str(coverage_result.detail),
                    "coverage_metrics": dict(coverage_result.metrics),
                }
        except Exception as exc:
            coverage_error = str(exc)
            logger.warning(
                "post_s1_audit coverage gate evaluation failed for facility=%s: %s",
                facility.name,
                exc,
                exc_info=True,
            )

        # Optional VLM rescoring on first N target-grounded, validated clips.
        vlm_rows: List[Dict[str, object]] = []
        vlm_requested = max(0, int(vlm_rescore_first))
        if vlm_requested > 0:
            candidate_rows = [
                row
                for row in clip_rows
                if bool(row.get("target_grounded", False))
                and str(row.get("video_validation_status", "")) == "ok"
                and isinstance(row.get("validated_video_path"), str)
            ]
            if len(candidate_rows) < vlm_requested:
                fallback_rows = [
                    row
                    for row in clip_rows
                    if str(row.get("video_validation_status", "")) == "ok"
                    and isinstance(row.get("validated_video_path"), str)
                    and row not in candidate_rows
                ]
                candidate_rows.extend(fallback_rows)
            selected_rows = candidate_rows[:vlm_requested]
            for idx, row in enumerate(selected_rows):
                clip_name = str(row.get("clip_name", "")).strip()
                clip = clip_by_name.get(clip_name, {})
                path_context = clip.get("path_context") or {}
                if not isinstance(path_context, dict):
                    path_context = {}
                expected_focus_text = str(
                    clip.get("expected_focus_text")
                    or _build_expected_focus_text(
                        path_type=str(row.get("path_type", "")),
                        path_context=path_context,
                    )
                )
                validated_path = Path(str(row.get("validated_video_path")))
                consensus_status = "ok"
                try:
                    consensus = _score_stage1_probe_consensus(
                        video_path=validated_path,
                        expected_focus_text=expected_focus_text,
                        config=config.eval_policy.vlm_judge,
                        facility_description=str(facility.description or ""),
                        votes=max(1, int(config.render.stage1_probe_consensus_votes)),
                        primary_model_only=bool(config.render.stage1_probe_primary_model_only),
                        tiebreak_extra_votes=max(
                            0, int(config.render.stage1_probe_tiebreak_extra_votes)
                        ),
                        tiebreak_spread_threshold=float(
                            config.render.stage1_probe_tiebreak_spread_threshold
                        ),
                    )
                except Exception as exc:
                    consensus = {
                        "score": None,
                        "error": str(exc),
                        "num_api_failures": 1,
                        "num_parse_failures": 0,
                        "votes_effective": 0,
                        "score_spread": None,
                        "active_model_used": str(config.eval_policy.vlm_judge.model),
                        "vote_rows": [],
                    }
                    consensus_status = "scoring_failed"

                if consensus.get("score") is None:
                    out = {
                        "row_index": int(idx),
                        "clip_name": clip_name,
                        "status": str(consensus_status),
                        "error": str(consensus.get("error", "probe_consensus_failed")),
                        "score_spread": consensus.get("score_spread"),
                        "votes_effective": int(consensus.get("votes_effective", 0)),
                        "active_model_used": str(
                            consensus.get("active_model_used")
                            or config.eval_policy.vlm_judge.model
                        ),
                        "target_grounded": bool(row.get("target_grounded", False)),
                    }
                else:
                    probe_score = consensus["score"]
                    out = {
                        "row_index": int(idx),
                        "clip_name": clip_name,
                        "status": "scored",
                        "task_score": float(probe_score.task_score),
                        "visual_score": float(probe_score.visual_score),
                        "spatial_score": float(probe_score.spatial_score),
                        "issue_tags": list(probe_score.issue_tags),
                        "reasoning": str(probe_score.reasoning or ""),
                        "score_spread": consensus.get("score_spread"),
                        "votes_effective": int(consensus.get("votes_effective", 0)),
                        "active_model_used": str(
                            consensus.get("active_model_used")
                            or config.eval_policy.vlm_judge.model
                        ),
                        "target_grounded": bool(row.get("target_grounded", False)),
                    }
                _append_probe_score_row(vlm_rows_path, out)
                vlm_rows.append(out)

        vlm_scored_rows = [row for row in vlm_rows if str(row.get("status")) == "scored"]
        vlm_target_missing = sum(
            1 for row in vlm_scored_rows if "target_missing" in list(row.get("issue_tags", []) or [])
        )
        vlm_task_avg = (
            None
            if not vlm_scored_rows
            else round(
                float(
                    np.mean(
                        np.asarray(
                            [float(row.get("task_score", 0.0) or 0.0) for row in vlm_scored_rows],
                            dtype=np.float64,
                        )
                    )
                ),
                6,
            )
        )
        vlm_visual_avg = (
            None
            if not vlm_scored_rows
            else round(
                float(
                    np.mean(
                        np.asarray(
                            [float(row.get("visual_score", 0.0) or 0.0) for row in vlm_scored_rows],
                            dtype=np.float64,
                        )
                    )
                ),
                6,
            )
        )
        vlm_spatial_avg = (
            None
            if not vlm_scored_rows
            else round(
                float(
                    np.mean(
                        np.asarray(
                            [float(row.get("spatial_score", 0.0) or 0.0) for row in vlm_scored_rows],
                            dtype=np.float64,
                        )
                    )
                ),
                6,
            )
        )

        summary: Dict[str, object] = {
            "status": "success",
            "facility_name": facility.name,
            "facility_description": facility.description,
            "manifest_path": str(manifest_path),
            "clip_rows_path": str(clip_rows_path),
            "vlm_rows_path": str(vlm_rows_path),
            "summary_path": str(summary_path),
            "num_clips_in_manifest": int(len(clips)),
            "num_target_grounded_clips": int(
                sum(1 for row in clip_rows if bool(row.get("target_grounded", False)))
            ),
            "num_videos_missing": int(
                sum(1 for row in clip_rows if str(row.get("video_validation_status")) == "video_missing")
            ),
            "num_videos_invalid": int(
                sum(1 for row in clip_rows if str(row.get("video_validation_status")) == "video_invalid")
            ),
            "num_monochrome_warnings": int(
                sum(1 for row in clip_rows if bool(row.get("content_monochrome_warning", False)))
            ),
            "num_quality_gate_passed": int(
                sum(1 for row in clip_rows if bool(row.get("quality_gate_passed", False)))
            ),
            "num_quality_gate_failed": int(
                sum(
                    1
                    for row in clip_rows
                    if row.get("quality_gate_passed") is False
                )
            ),
            "quality_reject_reason_counts": dict(sorted(quality_reason_counter.items())),
            "coverage_summary": coverage_summary,
            "coverage_error": coverage_error,
            "geometry_summary": geometry_summary,
            "geometry_error": geometry_error,
            "vlm_rescore_requested": int(vlm_requested),
            "vlm_rows_total": int(len(vlm_rows)),
            "vlm_rows_scored": int(len(vlm_scored_rows)),
            "vlm_rows_failed": int(len(vlm_rows) - len(vlm_scored_rows)),
            "vlm_avg_task_score": vlm_task_avg,
            "vlm_avg_visual_score": vlm_visual_avg,
            "vlm_avg_spatial_score": vlm_spatial_avg,
            "vlm_target_missing_count": int(vlm_target_missing),
        }
        write_json(summary, summary_path)
        return summary

    def _render_from_cache(
        self,
        cached_clips: List[Dict],
        splat,
        render_dir: Path,
        resolution: tuple,
        fps: int,
        config: ValidationConfig,
        scene_T: Optional[np.ndarray] = None,
    ) -> tuple[List[Dict], int, int, Dict[str, object]]:
        """Render clips using pre-computed camera paths from warmup cache."""
        from ..warmup import _deserialize_camera_poses

        manifest_entries: List[Dict] = []
        corrected_poses_total = 0
        corrected_clip_count = 0
        quality_summary = _empty_quality_summary()
        for clip_data in cached_clips:
            clip_name = clip_data["clip_name"]
            poses = _deserialize_camera_poses(clip_data["poses"])
            requested_count = int(clip_data.get("num_frames", len(poses)))
            pre_filter_count = len(poses)
            poses, corrected_count = correct_upside_down_camera_poses(poses)
            corrected_poses_total += corrected_count
            if corrected_count > 0:
                corrected_clip_count += 1
            post_filter_count = len(poses)
            post_resample_count = len(poses)

            # Convert cameras from corrected frame back to original PLY frame
            if scene_T is not None:
                from ..rendering.camera_paths import CameraPose

                poses = [
                    CameraPose(
                        c2w=transform_c2w(p.c2w, scene_T),
                        fx=p.fx,
                        fy=p.fy,
                        cx=p.cx,
                        cy=p.cy,
                        width=p.width,
                        height=p.height,
                    )
                    for p in poses
                ]

            # Save camera path
            save_path_to_json(poses, render_dir / f"{clip_name}_camera_path.json")

            # Render (GPU)
            output = render_video(
                splat=splat,
                poses=poses,
                output_dir=render_dir,
                clip_name=clip_name,
                fps=fps,
            )
            initial_camera = _camera_pose_metadata(poses[0]) if poses else None
            path_context = _cache_path_context(clip_data)
            manifest_entries.append(
                {
                    "clip_name": clip_name,
                    "path_type": clip_data["path_type"],
                    "clip_index": clip_data["clip_index"],
                    "num_frames": len(poses),
                    "requested_num_frames": requested_count,
                    "pre_filter_num_frames": pre_filter_count,
                    "post_filter_num_frames": post_filter_count,
                    "post_resample_num_frames": post_resample_count,
                    "resolution": list(resolution),
                    "fps": fps,
                    "video_path": str(output.video_path),
                    "depth_video_path": str(output.depth_video_path),
                    "camera_path": str(render_dir / f"{clip_name}_camera_path.json"),
                    "initial_camera": initial_camera,
                    "path_context": path_context,
                    "expected_focus_text": _build_expected_focus_text(
                        path_type=str(clip_data.get("path_type", "")),
                        path_context=path_context,
                    ),
                    **_manifest_vlm_probe_fields(
                        _default_vlm_probe_fields(
                            selected_fps=float(config.eval_policy.vlm_judge.video_metadata_fps),
                            candidate_count=1,
                        )
                    ),
                }
                )
            self._annotate_entry_quality(
                config=config,
                clip_entry=manifest_entries[-1],
                quality_retries_used=0,
                candidate_count_evaluated=1,
            )
            _update_quality_summary_from_entry(
                quality_summary=quality_summary,
                clip_entry=manifest_entries[-1],
                enforce_fail=bool(config.render.stage1_quality_planner_enabled),
            )
        quality_summary["quality_gate_passed"] = bool(
            int(quality_summary["num_quality_failures"]) == 0
            and int(quality_summary["num_vlm_probe_failures"]) == 0
        )
        return manifest_entries, corrected_poses_total, corrected_clip_count, quality_summary

    def _generate_and_render(
        self,
        config: ValidationConfig,
        splat,
        all_path_specs: List,
        scene_center: np.ndarray,
        occupancy: Optional[OccupancyGrid],
        render_dir: Path,
        num_frames: int,
        camera_height: float,
        look_down_deg: float,
        resolution: tuple,
        fps: int,
        scene_T: Optional[np.ndarray] = None,
        facility_description: str = "",
        probe_orientation_fix: str = "none",
        probe_scores_path: Optional[Path] = None,
    ) -> tuple[List[Dict], int, int, Dict[str, object]]:
        """Original path: generate camera paths from scratch, then render."""
        manifest_entries: List[Dict] = []
        clip_index = 0
        corrected_poses_total = 0
        corrected_clip_count = 0
        quality_summary = _empty_quality_summary()
        seen_render_fingerprints: List[Dict[str, object]] = []

        for path_spec in all_path_specs:
            is_scene_locked = _is_scene_locked_spec(path_spec)
            clip_repeats = int(config.render.num_clips_per_path)
            # Task-scoped paths are already object-specific; render once to control cost.
            if str(getattr(path_spec, "source_tag", "")).strip().lower() == "task_scoped":
                clip_repeats = int(config.render.task_scoped_num_clips_per_path)
            if is_scene_locked:
                # Deterministic scene-locked mode: one clip per target-grounded spec.
                clip_repeats = 1
            clip_repeats = max(1, clip_repeats)
            for clip_num in range(clip_repeats):
                path_context_for_scope = _path_context_from_spec(path_spec)
                target_grounded = bool(
                    should_probe_clip(
                        scope="targeted",
                        path_type=str(path_spec.type),
                        path_context=path_context_for_scope,
                    )
                    and _is_target_grounded_path_context(path_context_for_scope)
                )
                if is_scene_locked:
                    target_grounded = True
                # Add random offset for variety between clips
                rng = np.random.default_rng(seed=clip_index * 42)
                offset = _sample_start_offset(
                    config,
                    path_spec,
                    rng,
                    clip_repeat_index=clip_num,
                    is_task_scoped=(
                        str(getattr(path_spec, "source_tag", "")).strip().lower() == "task_scoped"
                    ),
                    target_grounded=False,
                )
                if is_scene_locked:
                    offset = np.zeros(3, dtype=np.float64)
                probe_offset = (
                    _sample_start_offset(
                        config,
                        path_spec,
                        rng,
                        clip_repeat_index=clip_num,
                        is_task_scoped=(
                            str(getattr(path_spec, "source_tag", "")).strip().lower()
                            == "task_scoped"
                        ),
                        target_grounded=True,
                    )
                    if target_grounded
                    else np.asarray(offset, dtype=np.float64).copy()
                )
                if is_scene_locked:
                    probe_offset = np.zeros(3, dtype=np.float64)
                requested_count = int(num_frames)
                if (
                    str(getattr(path_spec, "source_tag", "")).strip().lower() == "task_scoped"
                    and int(config.render.task_scoped_num_frames_override) > 0
                ):
                    requested_count = int(config.render.task_scoped_num_frames_override)
                clip_name = f"clip_{clip_index:03d}_{path_spec.type}"
                planned_spec = path_spec
                candidate_count_evaluated = 1
                ranked_candidates = []
                if bool(config.render.stage1_quality_planner_enabled) and not is_scene_locked:
                    ranked_candidates = rank_camera_spec_candidates(
                        base_spec=path_spec,
                        scene_center=scene_center,
                        num_frames=requested_count,
                        camera_height=camera_height,
                        look_down_deg=look_down_deg,
                        resolution=resolution,
                        start_offset=offset,
                        manipulation_target_z_bias_m=config.render.manipulation_target_z_bias_m,
                        budget=config.render.stage1_quality_candidate_budget,
                        min_visible_frame_ratio=config.render.stage1_coverage_min_visible_frame_ratio,
                        min_center_band_ratio=config.render.stage1_coverage_min_center_band_ratio,
                        min_approach_angle_bins=config.render.stage1_coverage_min_approach_angle_bins,
                        angle_bin_deg=config.render.stage1_coverage_angle_bin_deg,
                        center_band_x=config.render.stage1_coverage_center_band_x,
                        center_band_y=config.render.stage1_coverage_center_band_y,
                    )
                    if ranked_candidates:
                        best_candidate = ranked_candidates[0]
                        planned_spec = best_candidate.spec
                        candidate_count_evaluated = len(ranked_candidates)
                        planner_metrics = {
                            "planner_best_score": round(float(best_candidate.score), 6),
                            **dict(best_candidate.metrics),
                        }
                    else:
                        planner_metrics = {}
                else:
                    planner_metrics = {}
                probe_meta = _default_vlm_probe_fields(
                    selected_fps=float(config.eval_policy.vlm_judge.video_metadata_fps),
                    candidate_count=max(1, int(candidate_count_evaluated)),
                )
                should_run_probe = bool(config.render.stage1_active_perception_enabled) and bool(
                    is_scene_locked
                    or should_probe_clip(
                        scope=config.render.stage1_active_perception_scope,
                        path_type=str(path_spec.type),
                        path_context=path_context_for_scope,
                    )
                )
                if (
                    should_run_probe
                ):
                    planned_spec, probe_meta = self._run_active_perception_probe(
                        config=config,
                        splat=splat,
                        clip_name=clip_name,
                        initial_spec=planned_spec,
                        ranked_candidates=ranked_candidates,
                        scene_center=scene_center,
                        occupancy=occupancy,
                        render_dir=render_dir,
                        camera_height=camera_height,
                        look_down_deg=look_down_deg,
                        resolution=resolution,
                        start_offset=probe_offset,
                        fps=fps,
                        scene_T=scene_T,
                        facility_description=facility_description,
                        target_presence_enforced=target_grounded,
                        probe_orientation_fix=probe_orientation_fix,
                        probe_scores_path=probe_scores_path,
                        locked_mode=is_scene_locked,
                    )
                    _update_vlm_probe_summary(
                        quality_summary=quality_summary,
                        probe_meta=probe_meta,
                    )
                    if (
                        bool(config.render.stage1_active_perception_fail_closed)
                        and bool(probe_meta.get("vlm_probe_evaluated", False))
                        and not bool(probe_meta.get("vlm_probe_passed", False))
                    ):
                        logger.warning(
                            "Active-perception probe failed for %s: %s",
                            clip_name,
                            probe_meta.get("vlm_probe_fail_reason", "unknown"),
                        )
                        clip_index += 1
                        continue

                attempt = 0
                dedupe_regen_attempts = 0
                entry: Dict[str, object] | None = None
                while True:
                    (
                        poses,
                        pre_filter_count,
                        post_filter_count,
                        post_resample_count,
                        corrected_count,
                    ) = self._build_render_poses(
                        config=config,
                        path_spec=planned_spec,
                        scene_center=scene_center,
                        occupancy=occupancy,
                        num_frames=requested_count,
                        camera_height=camera_height,
                        look_down_deg=look_down_deg,
                        resolution=resolution,
                        start_offset=offset,
                    )
                    if not poses:
                        logger.warning(
                            "All poses rejected for %s clip %d — skipping",
                            planned_spec.type,
                            clip_num,
                        )
                        entry = None
                        break
                    corrected_poses_total += corrected_count
                    if corrected_count > 0:
                        corrected_clip_count += 1

                    render_poses = poses
                    if scene_T is not None:
                        from ..rendering.camera_paths import CameraPose

                        render_poses = [
                            CameraPose(
                                c2w=transform_c2w(p.c2w, scene_T),
                                fx=p.fx,
                                fy=p.fy,
                                cx=p.cx,
                                cy=p.cy,
                                width=p.width,
                                height=p.height,
                            )
                            for p in poses
                        ]

                    save_path_to_json(
                        render_poses,
                        render_dir / f"{clip_name}_camera_path.json",
                    )
                    output = render_video(
                        splat=splat,
                        poses=render_poses,
                        output_dir=render_dir,
                        clip_name=clip_name,
                        fps=fps,
                    )
                    dedupe_info = _detect_duplicate_clip(
                        video_path=Path(str(output.video_path)),
                        seen_fingerprints=seen_render_fingerprints,
                        similarity_threshold=float(
                            config.render.stage1_repeat_similarity_ssim_threshold
                        ),
                    )
                    if bool(config.render.stage1_repeat_dedupe_enabled) and bool(
                        dedupe_info.get("is_duplicate", False)
                    ):
                        quality_summary["num_repeat_duplicate_detected"] = int(
                            quality_summary["num_repeat_duplicate_detected"]
                        ) + 1
                        if dedupe_regen_attempts < int(
                            config.render.stage1_repeat_dedupe_max_regen_attempts
                        ):
                            dedupe_regen_attempts += 1
                            quality_summary["num_repeat_duplicate_regenerated"] = int(
                                quality_summary["num_repeat_duplicate_regenerated"]
                            ) + 1
                            offset = _offset_with_dedupe_jitter(
                                offset=offset,
                                attempt=dedupe_regen_attempts,
                                min_jitter=float(config.render.stage1_repeat_min_xy_jitter_m),
                            )
                            if str(planned_spec.type).strip().lower() == "manipulation":
                                planned_spec = replace(
                                    planned_spec,
                                    arc_phase_offset_deg=float(planned_spec.arc_phase_offset_deg)
                                    + float(18.0 * dedupe_regen_attempts),
                                )
                            continue
                        quality_summary["num_repeat_duplicate_unresolved"] = int(
                            quality_summary["num_repeat_duplicate_unresolved"]
                        ) + 1

                    initial_camera = _camera_pose_metadata(render_poses[0]) if render_poses else None
                    path_context = _path_context_from_spec(planned_spec)
                    entry = {
                        "clip_name": clip_name,
                        "path_type": planned_spec.type,
                        "clip_index": clip_index,
                        "num_frames": len(render_poses),
                        "requested_num_frames": requested_count,
                        "pre_filter_num_frames": pre_filter_count,
                        "post_filter_num_frames": post_filter_count,
                        "post_resample_num_frames": post_resample_count,
                        "resolution": list(resolution),
                        "fps": fps,
                        "video_path": str(output.video_path),
                        "depth_video_path": str(output.depth_video_path),
                        "camera_path": str(render_dir / f"{clip_name}_camera_path.json"),
                        "initial_camera": initial_camera,
                        "path_context": path_context,
                        "expected_focus_text": _build_expected_focus_text(
                            path_type=str(planned_spec.type),
                            path_context=path_context,
                        ),
                        "duplicate_clip_detected": bool(dedupe_info.get("is_duplicate", False)),
                        "duplicate_clip_regen_attempts": int(dedupe_regen_attempts),
                        "duplicate_clip_similarity": (
                            None
                            if dedupe_info.get("max_similarity") is None
                            else float(dedupe_info["max_similarity"])
                        ),
                        "video_sha256": str(dedupe_info.get("sha256", "")),
                        "candidate_count_evaluated": int(candidate_count_evaluated),
                        **_manifest_vlm_probe_fields(probe_meta),
                        **planner_metrics,
                    }
                    quality_passed = self._annotate_entry_quality(
                        config=config,
                        clip_entry=entry,
                        quality_retries_used=attempt,
                        candidate_count_evaluated=int(candidate_count_evaluated),
                    )
                    if (
                        quality_passed
                        or not bool(config.render.stage1_quality_planner_enabled)
                        or not bool(config.render.stage1_quality_autoretry_enabled)
                        or attempt >= int(config.render.stage1_quality_max_regen_attempts)
                    ):
                        break
                    attempt += 1
                    planned_spec = _retry_adjusted_spec(planned_spec, attempt)

                if entry is None:
                    clip_index += 1
                    continue
                seen_render_fingerprints.append(
                    {
                        "sha256": str(entry.get("video_sha256", "")),
                        "samples": dedupe_info.get("samples", []),
                        "clip_name": str(entry.get("clip_name", "")),
                    }
                )
                _update_quality_summary_from_entry(
                    quality_summary=quality_summary,
                    clip_entry=entry,
                    enforce_fail=bool(config.render.stage1_quality_planner_enabled),
                )
                manifest_entries.append(entry)
                clip_index += 1

        quality_summary["quality_gate_passed"] = bool(
            int(quality_summary["num_quality_failures"]) == 0
            and int(quality_summary["num_vlm_probe_failures"]) == 0
        )
        return manifest_entries, corrected_poses_total, corrected_clip_count, quality_summary

    def _build_render_poses(
        self,
        *,
        config: ValidationConfig,
        path_spec: CameraPathSpec,
        scene_center: np.ndarray,
        occupancy: Optional[OccupancyGrid],
        num_frames: int,
        camera_height: float,
        look_down_deg: float,
        resolution: tuple,
        start_offset: np.ndarray,
    ) -> tuple[List[object], int, int, int, int]:
        if _is_scene_locked_spec(path_spec):
            poses = _build_scene_locked_poses(
                path_spec=path_spec,
                num_frames=int(num_frames),
                resolution=resolution,
            )
            pre_filter_count = len(poses)
            if not poses:
                return [], pre_filter_count, 0, 0, 0
            poses, corrected_count = correct_upside_down_camera_poses(poses)
            post_count = len(poses)
            return poses, pre_filter_count, post_count, post_count, corrected_count

        poses = generate_path_from_spec(
            spec=path_spec,
            scene_center=scene_center,
            num_frames=num_frames,
            camera_height=camera_height,
            look_down_deg=look_down_deg,
            resolution=resolution,
            start_offset=start_offset,
            manipulation_target_z_bias_m=config.render.manipulation_target_z_bias_m,
        )
        pre_filter_count = len(poses)
        if occupancy is not None:
            target = np.array(path_spec.approach_point or scene_center[:3])
            retarget = None
            if isinstance(path_spec.approach_point, list) and len(path_spec.approach_point) >= 3:
                try:
                    candidate = np.asarray(path_spec.approach_point[:3], dtype=np.float64)
                    if np.all(np.isfinite(candidate)):
                        retarget = candidate
                except Exception:
                    retarget = None
            poses = filter_and_fix_poses(
                poses,
                occupancy,
                target,
                config.render.min_clearance_m,
                retarget_point=retarget,
                reorient_nudged_poses=True,
            )
            if not poses:
                return [], pre_filter_count, 0, 0, 0
        post_filter_count = len(poses)
        if (
            bool(config.render.preserve_num_frames_after_collision_filter)
            and poses
            and num_frames > 0
            and len(poses) != num_frames
        ):
            poses = _resample_poses_nearest(poses, num_frames)
        post_resample_count = len(poses)
        poses, corrected_count = correct_upside_down_camera_poses(poses)
        return poses, pre_filter_count, post_filter_count, post_resample_count, corrected_count

    def _annotate_entry_quality(
        self,
        *,
        config: ValidationConfig,
        clip_entry: Dict[str, object],
        quality_retries_used: int,
        candidate_count_evaluated: int,
    ) -> bool:
        path_context = clip_entry.get("path_context") or {}
        if not isinstance(path_context, dict):
            path_context = {}
        target_xyz = path_context.get("approach_point")
        if not (isinstance(target_xyz, list) and len(target_xyz) == 3):
            target_xyz = None

        require_target = str(clip_entry.get("path_type", "")).strip().lower() == "manipulation"
        metrics = evaluate_clip_quality(
            clip_entry=clip_entry,
            target_xyz=target_xyz,
            blur_laplacian_min=float(config.render.stage1_coverage_blur_laplacian_min),
            min_visible_frame_ratio=float(config.render.stage1_coverage_min_visible_frame_ratio),
            min_center_band_ratio=float(config.render.stage1_coverage_min_center_band_ratio),
            min_approach_angle_bins=int(config.render.stage1_coverage_min_approach_angle_bins),
            angle_bin_deg=float(config.render.stage1_coverage_angle_bin_deg),
            center_band_x=config.render.stage1_coverage_center_band_x,
            center_band_y=config.render.stage1_coverage_center_band_y,
            blur_sample_every_n_frames=int(config.render.stage1_coverage_blur_sample_every_n_frames),
            blur_max_samples=int(config.render.stage1_coverage_blur_max_samples_per_clip),
            min_clip_score=float(config.render.stage1_quality_min_clip_score),
            require_target=bool(require_target),
        )
        clip_entry.update(metrics)
        clip_entry["quality_retries_used"] = int(max(0, quality_retries_used))
        clip_entry["candidate_count_evaluated"] = int(max(1, candidate_count_evaluated))
        clip_entry["degenerate_mix_detected"] = False
        return bool(metrics.get("quality_gate_passed", False))

    def _run_active_perception_probe(
        self,
        *,
        config: ValidationConfig,
        splat,
        clip_name: str,
        initial_spec: CameraPathSpec,
        ranked_candidates: List[object],
        scene_center: np.ndarray,
        occupancy: Optional[OccupancyGrid],
        render_dir: Path,
        camera_height: float,
        look_down_deg: float,
        resolution: tuple,
        start_offset: np.ndarray,
        fps: int,
        scene_T: Optional[np.ndarray],
        facility_description: str,
        target_presence_enforced: bool = False,
        probe_orientation_fix: str = "none",
        probe_scores_path: Optional[Path] = None,
        locked_mode: bool = False,
    ) -> tuple[CameraPathSpec, Dict[str, object]]:
        budget = resolve_probe_budget(
            candidate_budget=config.render.stage1_quality_candidate_budget,
            max_loops_cap=int(config.render.stage1_active_perception_max_loops),
            probe_frames_override=int(config.render.stage1_probe_frames_override),
            probe_resolution_scale_override=float(config.render.stage1_probe_resolution_scale),
        )
        probe_resolution = compute_probe_resolution(
            base_resolution=(int(resolution[0]), int(resolution[1])),
            scale=float(budget.probe_resolution_scale),
        )
        selected_fps = float(config.eval_policy.vlm_judge.video_metadata_fps)
        probe_meta = _default_vlm_probe_fields(
            selected_fps=selected_fps,
            candidate_count=max(1, int(min(len(ranked_candidates), int(budget.top_k)))),
        )
        seen_probe_fingerprints: List[Dict[str, object]] = []
        fallback_target_point = [float(scene_center[0]), float(scene_center[1]), float(scene_center[2])]

        if bool(locked_mode):
            ranked_specs = [initial_spec]
            geometric_scores = [0.0]
        elif ranked_candidates:
            ranked_specs = [row.spec for row in ranked_candidates[: max(1, int(budget.top_k))]]
            geometric_scores = [
                float(getattr(row, "score", 0.0))
                for row in ranked_candidates[: max(1, int(budget.top_k))]
            ]
        else:
            ranked_specs = [initial_spec]
            geometric_scores = [0.0]

        current_spec = initial_spec
        passed = False
        last_issue_tags: List[str] = []
        last_fail_reason: Optional[str] = None
        retries_used = 0
        type_conversion_locked = bool(locked_mode)

        # max_loops is "correction loops"; total rounds include the initial probe round.
        total_rounds = 1 if bool(locked_mode) else max(1, int(budget.max_loops) + 1)
        for round_idx in range(total_rounds):
            if round_idx == 0:
                candidate_specs = ranked_specs
                candidate_geometric = geometric_scores
            else:
                candidate_specs = [current_spec]
                candidate_geometric = [0.0]

            best_row: Dict[str, object] | None = None
            for cand_idx, candidate_spec in enumerate(candidate_specs):
                probe_meta["vlm_probe_attempts"] = int(probe_meta["vlm_probe_attempts"]) + 1
                probe_meta["vlm_probe_evaluated"] = True
                candidate_offset = np.asarray(start_offset, dtype=np.float64).copy()
                candidate_current_spec = candidate_spec
                dedupe_regen_attempts = 0
                pose_regen_attempted = False
                row: Dict[str, object] | None = None
                while True:
                    path_context = _path_context_from_spec(candidate_current_spec)
                    candidate_target_grounded = bool(target_presence_enforced) and bool(
                        _is_target_grounded_path_context(path_context)
                    )
                    candidate_target_xyz = _path_context_target_xyz(path_context)
                    candidate_target_extents = _path_context_target_extents(path_context)
                    (
                        poses,
                        pre_count,
                        post_filter_count,
                        _post_resample_count,
                        corrected_count,
                    ) = self._build_render_poses(
                        config=config,
                        path_spec=candidate_current_spec,
                        scene_center=scene_center,
                        occupancy=occupancy,
                        num_frames=int(budget.probe_frames),
                        camera_height=camera_height,
                        look_down_deg=look_down_deg,
                        resolution=probe_resolution,
                        start_offset=candidate_offset,
                    )
                    probe_clip_name = f"{clip_name}_probe_l{round_idx}_c{cand_idx:02d}"
                    if not poses:
                        row = {
                            "spec": candidate_current_spec,
                            "combined_score": -1e9,
                            "passed": False,
                            "issue_tags": ["target_missing"],
                            "reasoning": "probe_pose_generation_failed",
                            "target_grounded": bool(candidate_target_grounded),
                        }
                        _append_probe_score_row(
                            probe_scores_path,
                            {
                                "clip_name": clip_name,
                                "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                "path_type": str(candidate_current_spec.type),
                                "loop": int(round_idx),
                                "candidate": int(cand_idx),
                                "status": "pose_generation_failed",
                                "target_grounded": bool(candidate_target_grounded),
                            },
                        )
                        break

                    viable_ratio = float(post_filter_count) / max(1.0, float(int(budget.probe_frames)))
                    min_viable_ratio = float(config.render.stage1_probe_min_viable_pose_ratio)
                    if viable_ratio < min_viable_ratio:
                        probe_meta["num_probe_viability_rejects"] = int(
                            probe_meta.get("num_probe_viability_rejects", 0)
                        ) + 1
                        row = {
                            "spec": candidate_current_spec,
                            "combined_score": -1e9,
                            "passed": False,
                            "issue_tags": ["camera_motion_too_fast", "target_missing"],
                            "reasoning": (
                                f"probe_viability_reject ratio={viable_ratio:.3f} "
                                f"threshold={min_viable_ratio:.3f}"
                            ),
                            "target_grounded": bool(candidate_target_grounded),
                        }
                        _append_probe_score_row(
                            probe_scores_path,
                            {
                                "clip_name": clip_name,
                                "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                "path_type": str(candidate_current_spec.type),
                                "loop": int(round_idx),
                                "candidate": int(cand_idx),
                                "status": "viability_reject",
                                "viable_pose_ratio": round(viable_ratio, 6),
                                "pre_filter_count": int(pre_count),
                                "post_filter_count": int(post_filter_count),
                                "target_grounded": bool(candidate_target_grounded),
                            },
                        )
                        break

                    min_unique_positions = int(config.render.stage1_probe_min_unique_positions)
                    if bool(locked_mode):
                        # Locked target clips intentionally use near-static motion.
                        min_unique_positions = min(min_unique_positions, 1)
                    unique_positions = _count_unique_camera_positions(poses)
                    if unique_positions < min_unique_positions:
                        if not pose_regen_attempted:
                            pose_regen_attempted = True
                            candidate_current_spec = _apply_diversity_kick(
                                candidate_current_spec, round_idx + 1
                            )
                            continue
                        probe_meta["num_probe_viability_rejects"] = int(
                            probe_meta.get("num_probe_viability_rejects", 0)
                        ) + 1
                        row = {
                            "spec": candidate_current_spec,
                            "combined_score": -1e9,
                            "passed": False,
                            "issue_tags": ["unstable_view", "target_missing"],
                            "reasoning": (
                                f"probe_degenerate_positions unique={unique_positions} "
                                f"min_required={min_unique_positions}"
                            ),
                            "target_grounded": bool(candidate_target_grounded),
                        }
                        _append_probe_score_row(
                            probe_scores_path,
                            {
                                "clip_name": clip_name,
                                "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                "path_type": str(candidate_current_spec.type),
                                "loop": int(round_idx),
                                "candidate": int(cand_idx),
                                "status": "degenerate_positions",
                                "unique_positions": int(unique_positions),
                                "min_unique_positions": int(min_unique_positions),
                                "target_grounded": bool(candidate_target_grounded),
                            },
                        )
                        break

                    if candidate_target_grounded:
                        if candidate_target_xyz is None:
                            row = {
                                "spec": candidate_current_spec,
                                "combined_score": -1e9,
                                "passed": False,
                                "issue_tags": ["target_missing"],
                                "reasoning": "target_presence_reject missing_target_annotation",
                                "target_grounded": True,
                            }
                            _append_probe_score_row(
                                probe_scores_path,
                                {
                                    "clip_name": clip_name,
                                    "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                    "path_type": str(candidate_current_spec.type),
                                    "loop": int(round_idx),
                                    "candidate": int(cand_idx),
                                    "status": "target_presence_reject",
                                    "reason": "missing_target_annotation",
                                    "target_grounded": True,
                                },
                            )
                            break

                        total_frames, visible_samples = project_target_to_poses(
                            poses,
                            candidate_target_xyz,
                        )
                        (
                            visible_frames,
                            total_frames,
                            center_frames,
                            _angle_bins,
                        ) = analyze_target_visibility(
                            total_frames=total_frames,
                            visible_samples=visible_samples,
                            angle_bin_deg=float(config.render.stage1_coverage_angle_bin_deg),
                            center_band_x=config.render.stage1_coverage_center_band_x,
                            center_band_y=config.render.stage1_coverage_center_band_y,
                        )
                        visible_ratio = (
                            float(visible_frames) / float(total_frames) if total_frames > 0 else 0.0
                        )
                        center_ratio = (
                            float(center_frames) / float(total_frames) if total_frames > 0 else 0.0
                        )
                        min_visible_ratio = max(
                            float(config.render.stage1_coverage_min_visible_frame_ratio),
                            float(_TARGET_PRESENCE_STRICT_MIN_VISIBLE_RATIO),
                        )
                        min_center_ratio = max(
                            float(config.render.stage1_coverage_min_center_band_ratio),
                            float(_TARGET_PRESENCE_STRICT_MIN_CENTER_RATIO),
                        )
                        if visible_ratio < min_visible_ratio or center_ratio < min_center_ratio:
                            issue_tags = ["target_missing"]
                            if center_ratio < min_center_ratio:
                                issue_tags.append("target_off_center")
                            row = {
                                "spec": candidate_current_spec,
                                "combined_score": -1e9,
                                "passed": False,
                                "issue_tags": issue_tags,
                                "reasoning": (
                                    "target_presence_reject "
                                    f"visible={visible_ratio:.3f}/{min_visible_ratio:.3f} "
                                    f"center={center_ratio:.3f}/{min_center_ratio:.3f}"
                                ),
                                "target_grounded": True,
                            }
                            _append_probe_score_row(
                                probe_scores_path,
                                {
                                    "clip_name": clip_name,
                                    "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                    "path_type": str(candidate_current_spec.type),
                                    "loop": int(round_idx),
                                    "candidate": int(cand_idx),
                                    "status": "target_presence_reject",
                                    "target_visible_ratio": round(visible_ratio, 6),
                                    "target_center_ratio": round(center_ratio, 6),
                                    "min_visible_ratio": round(min_visible_ratio, 6),
                                    "min_center_ratio": round(min_center_ratio, 6),
                                    "target_grounded": True,
                                },
                            )
                            break
                        if candidate_target_extents is not None:
                            projected_size_ratio = _estimate_target_projected_size_ratio(
                                poses=poses,
                                target_xyz=candidate_target_xyz,
                                target_extents_m=candidate_target_extents,
                            )
                            if projected_size_ratio < float(_TARGET_PRESENCE_STRICT_MIN_SIZE_RATIO):
                                row = {
                                    "spec": candidate_current_spec,
                                    "combined_score": -1e9,
                                    "passed": False,
                                    "issue_tags": ["target_missing", "target_off_center"],
                                    "reasoning": (
                                        "target_presence_reject "
                                        f"size={projected_size_ratio:.3f}/"
                                        f"{float(_TARGET_PRESENCE_STRICT_MIN_SIZE_RATIO):.3f}"
                                    ),
                                    "target_grounded": True,
                                }
                                _append_probe_score_row(
                                    probe_scores_path,
                                    {
                                        "clip_name": clip_name,
                                        "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                        "path_type": str(candidate_current_spec.type),
                                        "loop": int(round_idx),
                                        "candidate": int(cand_idx),
                                        "status": "target_presence_reject",
                                        "reason": "target_too_small",
                                        "target_size_ratio": round(projected_size_ratio, 6),
                                        "min_target_size_ratio": round(
                                            float(_TARGET_PRESENCE_STRICT_MIN_SIZE_RATIO), 6
                                        ),
                                        "target_grounded": True,
                                    },
                                )
                                break
                        los_ratio = _compute_target_line_of_sight_ratio(
                            poses=poses,
                            target_xyz=candidate_target_xyz,
                            occupancy=occupancy,
                            min_clearance_m=float(config.render.min_clearance_m),
                        )
                        min_target_los_ratio = (
                            0.0
                            if bool(locked_mode)
                            else float(_TARGET_PRESENCE_STRICT_MIN_LOS_RATIO)
                        )
                        if los_ratio < float(min_target_los_ratio):
                            row = {
                                "spec": candidate_current_spec,
                                "combined_score": -1e9,
                                "passed": False,
                                "issue_tags": ["target_missing", "target_occluded"],
                                "reasoning": (
                                    "target_presence_reject "
                                    f"line_of_sight={los_ratio:.3f}/"
                                    f"{float(min_target_los_ratio):.3f}"
                                ),
                                "target_grounded": True,
                            }
                            _append_probe_score_row(
                                probe_scores_path,
                                {
                                    "clip_name": clip_name,
                                    "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                    "path_type": str(candidate_current_spec.type),
                                    "loop": int(round_idx),
                                    "candidate": int(cand_idx),
                                    "status": "target_presence_reject",
                                    "reason": "line_of_sight_occluded",
                                    "target_los_ratio": round(los_ratio, 6),
                                    "min_target_los_ratio": round(
                                        float(min_target_los_ratio), 6
                                    ),
                                    "target_grounded": True,
                                },
                            )
                            break

                    render_poses = poses
                    if scene_T is not None:
                        from ..rendering.camera_paths import CameraPose

                        render_poses = [
                            CameraPose(
                                c2w=transform_c2w(p.c2w, scene_T),
                                fx=p.fx,
                                fy=p.fy,
                                cx=p.cx,
                                cy=p.cy,
                                width=p.width,
                                height=p.height,
                            )
                            for p in poses
                        ]
                    output = render_video(
                        splat=splat,
                        poses=render_poses,
                        output_dir=render_dir,
                        clip_name=probe_clip_name,
                        fps=fps,
                    )
                    path_context = _path_context_from_spec(candidate_current_spec)
                    expected_focus_text = _build_expected_focus_text(
                        path_type=str(candidate_current_spec.type),
                        path_context=path_context,
                    )
                    probe_video_path = Path(str(output.video_path))
                    probe_oriented_path: Optional[Path] = None
                    probe_scoring_path: Optional[Path] = None

                    _orientation_mode = str(probe_orientation_fix or "none").strip().lower()
                    if _orientation_mode and _orientation_mode != "none":
                        from ..evaluation.video_orientation import (
                            normalize_video_orientation_fix,
                            transform_video_orientation,
                        )

                        _norm_mode = normalize_video_orientation_fix(_orientation_mode)
                        if _norm_mode != "none":
                            probe_oriented_path = render_dir / f"{probe_clip_name}_oriented.mp4"
                            try:
                                transform_video_orientation(
                                    input_path=probe_video_path,
                                    output_path=probe_oriented_path,
                                    orientation_fix=_norm_mode,
                                    force_grayscale=False,
                                )
                                probe_video_path = probe_oriented_path
                            except Exception as _orient_exc:
                                logger.warning(
                                    "Probe orientation fix failed for %s (%s): %s — "
                                    "scoring unoriented video",
                                    probe_clip_name,
                                    _norm_mode,
                                    _orient_exc,
                                )
                                probe_oriented_path = None

                    dedupe_info = _detect_duplicate_clip(
                        video_path=probe_video_path,
                        seen_fingerprints=seen_probe_fingerprints,
                        similarity_threshold=float(config.render.stage1_repeat_similarity_ssim_threshold),
                    )
                    if bool(config.render.stage1_probe_dedupe_enabled) and bool(
                        dedupe_info.get("is_duplicate", False)
                    ):
                        probe_meta["num_probe_duplicate_detected"] = int(
                            probe_meta.get("num_probe_duplicate_detected", 0)
                        ) + 1
                        if dedupe_regen_attempts < int(
                            config.render.stage1_probe_dedupe_max_regen_attempts
                        ):
                            dedupe_regen_attempts += 1
                            probe_meta["num_probe_duplicate_regenerated"] = int(
                                probe_meta.get("num_probe_duplicate_regenerated", 0)
                            ) + 1
                            jitter_base = float(
                                max(
                                    float(config.render.stage1_repeat_min_xy_jitter_m),
                                    float(config.render.stage1_probe_dedupe_center_dist_m),
                                )
                            )
                            if not bool(candidate_target_grounded):
                                candidate_offset = _offset_with_dedupe_jitter(
                                    offset=candidate_offset,
                                    attempt=dedupe_regen_attempts,
                                    min_jitter=jitter_base,
                                )
                            if str(candidate_current_spec.type).strip().lower() == "manipulation":
                                span_scale = 0.85 if dedupe_regen_attempts >= 2 else 1.0
                                candidate_current_spec = replace(
                                    candidate_current_spec,
                                    arc_phase_offset_deg=float(
                                        candidate_current_spec.arc_phase_offset_deg
                                    )
                                    + float(27.0 * dedupe_regen_attempts),
                                    arc_span_deg=float(
                                        max(
                                            30.0,
                                            min(
                                                220.0,
                                                float(candidate_current_spec.arc_span_deg) * span_scale,
                                            ),
                                        )
                                    ),
                                )
                            if not bool(config.render.stage1_keep_probe_videos):
                                _cleanup_probe_outputs(output)
                                if probe_oriented_path is not None:
                                    try:
                                        probe_oriented_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                            continue
                        probe_meta["num_probe_duplicate_unresolved"] = int(
                            probe_meta.get("num_probe_duplicate_unresolved", 0)
                        ) + 1
                        row = {
                            "spec": candidate_current_spec,
                            "combined_score": -1e9,
                            "passed": False,
                            "issue_tags": ["target_missing"],
                            "reasoning": "probe_duplicate_unresolved",
                            "target_grounded": bool(candidate_target_grounded),
                        }
                        _append_probe_score_row(
                            probe_scores_path,
                            {
                                "clip_name": clip_name,
                                "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                "path_type": str(candidate_current_spec.type),
                                "loop": int(round_idx),
                                "candidate": int(cand_idx),
                                "status": "duplicate_unresolved",
                                "duplicate_reason": str(dedupe_info.get("reason", "")),
                                "duplicate_similarity": dedupe_info.get("max_similarity"),
                                "target_grounded": bool(candidate_target_grounded),
                            },
                        )
                        if not bool(config.render.stage1_keep_probe_videos):
                            _cleanup_probe_outputs(output)
                            if probe_oriented_path is not None:
                                try:
                                    probe_oriented_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                        break

                    seen_probe_fingerprints.append(
                        {
                            "sha256": str(dedupe_info.get("sha256", "")),
                            "samples": list(dedupe_info.get("samples", []) or []),
                            "clip_name": probe_clip_name,
                        }
                    )
                    consensus_status = "scoring_failed"
                    try:
                        validated_probe = _ensure_probe_h264_for_scoring(
                            video_path=probe_video_path,
                            min_frames=max(1, int(budget.probe_frames)),
                        )
                        probe_scoring_path = Path(validated_probe.path)
                        probe_meta["probe_codec"] = str(validated_probe.codec_name)
                        probe_meta["probe_decoded_frames"] = int(validated_probe.decoded_frames)
                        probe_meta["probe_resolution"] = [
                            int(validated_probe.width or 0),
                            int(validated_probe.height or 0),
                        ]
                        probe_meta["num_probe_monochrome_warnings"] = int(
                            probe_meta.get("num_probe_monochrome_warnings", 0)
                        ) + int(
                            1 if bool(getattr(validated_probe, "content_monochrome_warning", False)) else 0
                        )
                        if candidate_target_grounded and bool(
                            getattr(validated_probe, "content_monochrome_warning", False)
                        ):
                            consensus = {
                                "score": None,
                                "error": "probe_media_invalid_monochrome",
                                "num_api_failures": 0,
                                "num_parse_failures": 0,
                                "votes_effective": 0,
                                "score_spread": None,
                                "active_model_used": str(config.eval_policy.vlm_judge.model),
                                "vote_rows": [],
                            }
                            consensus_status = "probe_media_invalid"
                        else:
                            consensus = _score_stage1_probe_consensus(
                                video_path=probe_scoring_path,
                                expected_focus_text=expected_focus_text,
                                config=config.eval_policy.vlm_judge,
                                facility_description=facility_description,
                                votes=max(1, int(config.render.stage1_probe_consensus_votes)),
                                primary_model_only=bool(config.render.stage1_probe_primary_model_only),
                                tiebreak_extra_votes=max(
                                    0, int(config.render.stage1_probe_tiebreak_extra_votes)
                                ),
                                tiebreak_spread_threshold=float(
                                    config.render.stage1_probe_tiebreak_spread_threshold
                                ),
                            )
                    except Exception as exc:
                        consensus = {
                            "score": None,
                            "error": str(exc),
                            "num_api_failures": 1,
                            "num_parse_failures": 0,
                            "votes_effective": 0,
                            "score_spread": None,
                            "active_model_used": str(config.eval_policy.vlm_judge.model),
                            "vote_rows": [],
                        }
                        consensus_status = "scoring_failed"

                    probe_meta["num_vlm_probe_parse_failures"] = int(
                        probe_meta["num_vlm_probe_parse_failures"]
                    ) + int(consensus.get("num_parse_failures", 0))
                    probe_meta["num_vlm_probe_api_failures"] = int(
                        probe_meta["num_vlm_probe_api_failures"]
                    ) + int(consensus.get("num_api_failures", 0))
                    probe_meta["active_model_used"] = str(
                        consensus.get("active_model_used") or config.eval_policy.vlm_judge.model
                    )
                    probe_meta["vlm_probe_consensus_votes_configured"] = int(
                        max(1, int(config.render.stage1_probe_consensus_votes))
                    )
                    probe_meta["vlm_probe_consensus_votes_effective"] = int(
                        max(0, int(consensus.get("votes_effective", 0)))
                    )
                    probe_meta["vlm_probe_score_spread"] = consensus.get("score_spread")
                    spread = consensus.get("score_spread")
                    if spread is not None and float(spread) >= float(
                        config.render.stage1_probe_consensus_high_variance_delta
                    ):
                        probe_meta["num_vlm_probe_high_variance"] = int(
                            probe_meta.get("num_vlm_probe_high_variance", 0)
                        ) + 1

                    if consensus.get("score") is None:
                        row = {
                            "spec": candidate_current_spec,
                            "combined_score": -1e9,
                            "passed": False,
                            "issue_tags": ["target_missing"],
                            "reasoning": str(consensus.get("error", "probe_consensus_failed")),
                            "target_grounded": bool(candidate_target_grounded),
                        }
                        _append_probe_score_row(
                            probe_scores_path,
                            {
                                "clip_name": clip_name,
                                "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                "path_type": str(candidate_current_spec.type),
                                "loop": int(round_idx),
                                "candidate": int(cand_idx),
                                "status": str(consensus_status),
                                "error": str(consensus.get("error", "probe_consensus_failed")),
                                "score_spread": consensus.get("score_spread"),
                                "votes_effective": int(consensus.get("votes_effective", 0)),
                                "votes": list(consensus.get("vote_rows", []) or []),
                                "probe_codec": probe_meta.get("probe_codec"),
                                "probe_resolution": probe_meta.get("probe_resolution"),
                                "probe_decoded_frames": probe_meta.get("probe_decoded_frames"),
                                "target_grounded": bool(candidate_target_grounded),
                            },
                        )
                    else:
                        probe_score = consensus["score"]
                        issue_tags = list(probe_score.issue_tags)
                        target_missing_detected = bool(
                            candidate_target_grounded and "target_missing" in issue_tags
                        )
                        if target_missing_detected:
                            row = {
                                "spec": candidate_current_spec,
                                "combined_score": -1e9,
                                "passed": False,
                                "issue_tags": issue_tags,
                                "reasoning": (
                                    "target_presence_reject vlm_target_missing "
                                    f"task={float(probe_score.task_score):.1f} "
                                    f"visual={float(probe_score.visual_score):.1f} "
                                    f"spatial={float(probe_score.spatial_score):.1f}"
                                ),
                                # Keep task_score internal so correction policy can lock
                                # path-type conversion when partial target evidence exists.
                                "task_score": float(probe_score.task_score),
                                "target_grounded": True,
                            }
                            _append_probe_score_row(
                                probe_scores_path,
                                {
                                    "clip_name": clip_name,
                                    "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                    "path_type": str(candidate_current_spec.type),
                                    "loop": int(round_idx),
                                    "candidate": int(cand_idx),
                                    "status": "target_presence_reject",
                                    "reason": "vlm_target_missing",
                                    "observed_task_score": float(probe_score.task_score),
                                    "observed_visual_score": float(probe_score.visual_score),
                                    "observed_spatial_score": float(probe_score.spatial_score),
                                    "observed_issue_tags": issue_tags,
                                    "reasoning": str(probe_score.reasoning or ""),
                                    "score_spread": consensus.get("score_spread"),
                                    "votes_effective": int(consensus.get("votes_effective", 0)),
                                    "votes": list(consensus.get("vote_rows", []) or []),
                                    "active_model_used": str(
                                        consensus.get("active_model_used")
                                        or config.eval_policy.vlm_judge.model
                                    ),
                                    "probe_codec": probe_meta.get("probe_codec"),
                                    "probe_resolution": probe_meta.get("probe_resolution"),
                                    "probe_decoded_frames": probe_meta.get("probe_decoded_frames"),
                                    "target_grounded": True,
                                },
                            )
                            logger.info(
                                "probe_target_presence_reject clip=%s loop=%d cand=%d "
                                "reason=vlm_target_missing scores=%.1f/%.1f/%.1f tags=%s",
                                clip_name,
                                round_idx,
                                cand_idx,
                                float(probe_score.task_score),
                                float(probe_score.visual_score),
                                float(probe_score.spatial_score),
                                issue_tags,
                            )
                        else:
                            passes = probe_passes_thresholds(
                                task_score=float(probe_score.task_score),
                                visual_score=float(probe_score.visual_score),
                                spatial_score=float(probe_score.spatial_score),
                                min_task=float(config.render.stage1_vlm_min_task_score),
                                min_visual=float(config.render.stage1_vlm_min_visual_score),
                                min_spatial=float(config.render.stage1_vlm_min_spatial_score),
                            )
                            geom = candidate_geometric[min(cand_idx, len(candidate_geometric) - 1)]
                            combined = combined_probe_score(
                                geometric_score=float(geom),
                                task_score=float(probe_score.task_score),
                                visual_score=float(probe_score.visual_score),
                                spatial_score=float(probe_score.spatial_score),
                            )
                            row = {
                                "spec": candidate_current_spec,
                                "combined_score": float(combined),
                                "passed": bool(passes),
                                "issue_tags": issue_tags,
                                "reasoning": str(probe_score.reasoning or ""),
                                "task_score": float(probe_score.task_score),
                                "target_grounded": bool(candidate_target_grounded),
                            }
                            logger.info(
                                "probe_score clip=%s loop=%d cand=%d task=%.1f visual=%.1f spatial=%.1f "
                                "spread=%s tags=%s",
                                clip_name,
                                round_idx,
                                cand_idx,
                                float(probe_score.task_score),
                                float(probe_score.visual_score),
                                float(probe_score.spatial_score),
                                (
                                    "n/a"
                                    if consensus.get("score_spread") is None
                                    else f"{float(consensus.get('score_spread')):.3f}"
                                ),
                                issue_tags,
                            )
                            _append_probe_score_row(
                                probe_scores_path,
                                {
                                    "clip_name": clip_name,
                                    "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                    "path_type": str(candidate_current_spec.type),
                                    "loop": int(round_idx),
                                    "candidate": int(cand_idx),
                                    "task_score": float(probe_score.task_score),
                                    "visual_score": float(probe_score.visual_score),
                                    "spatial_score": float(probe_score.spatial_score),
                                    "issue_tags": issue_tags,
                                    "reasoning": str(probe_score.reasoning or ""),
                                    "passed": bool(passes),
                                    "score_spread": consensus.get("score_spread"),
                                    "votes_effective": int(consensus.get("votes_effective", 0)),
                                    "votes": list(consensus.get("vote_rows", []) or []),
                                    "active_model_used": str(
                                        consensus.get("active_model_used")
                                        or config.eval_policy.vlm_judge.model
                                    ),
                                    "probe_codec": probe_meta.get("probe_codec"),
                                    "probe_resolution": probe_meta.get("probe_resolution"),
                                    "probe_decoded_frames": probe_meta.get("probe_decoded_frames"),
                                    "target_grounded": bool(candidate_target_grounded),
                                },
                            )

                    probe_meta["vlm_probe_evaluated"] = True
                    if not bool(config.render.stage1_keep_probe_videos):
                        _cleanup_probe_outputs(output)
                        if probe_oriented_path is not None:
                            try:
                                probe_oriented_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                    if probe_scoring_path is not None and probe_scoring_path != probe_video_path:
                        try:
                            probe_scoring_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                    if corrected_count > 0:
                        logger.debug(
                            "Probe clip %s had roll-corrected poses: %d",
                            probe_clip_name,
                            corrected_count,
                        )
                    break

                if row is None:
                    continue
                if best_row is None or float(row["combined_score"]) > float(best_row["combined_score"]):
                    best_row = row

            if best_row is None:
                last_fail_reason = "probe_no_candidate_results"
                break

            best_task_score = float(best_row.get("task_score", 0.0) or 0.0)
            last_issue_tags = list(best_row.get("issue_tags", []))
            # Lock orbit→manipulation conversion only when the target is genuinely
            # visible (task≥1) AND not flagged as missing.  If target_missing is
            # still present the zoom-in path makes framing worse (camera_too_close,
            # camera_motion_too_fast) without resolving the underlying problem.
            if (
                bool(best_row.get("target_grounded", False))
                and best_task_score >= 1.0
                and "target_missing" not in last_issue_tags
            ):
                type_conversion_locked = True
            current_spec = best_row["spec"]
            if bool(best_row.get("passed", False)):
                passed = True
                break
            if round_idx >= total_rounds - 1:
                last_fail_reason = "probe_threshold_not_met"
                break

            retries_used += 1
            prev_spec = current_spec
            current_spec = apply_issue_tag_corrections(
                spec=current_spec,
                issue_tags=last_issue_tags,
                default_camera_height=float(camera_height),
                default_look_down_deg=float(look_down_deg),
                loop_idx=int(round_idx + 1),
                fallback_target_point=fallback_target_point,
                best_task_score=best_task_score,
                allow_type_conversion=not bool(type_conversion_locked),
            )
            if _spec_is_effectively_unchanged(prev_spec, current_spec):
                logger.info(
                    "Probe loop round %d for %s: spec delta below epsilon after "
                    "corrections — applying diversity kick",
                    round_idx + 1,
                    clip_name,
                )
                current_spec = _apply_diversity_kick(current_spec, round_idx + 1)

        probe_meta["vlm_probe_passed"] = bool(passed)
        probe_meta["vlm_probe_retries_used"] = int(max(0, retries_used))
        probe_meta["vlm_probe_issue_tags_final"] = list(last_issue_tags)
        if not passed:
            probe_meta["vlm_probe_fail_reason"] = str(last_fail_reason or "probe_failed")
        # Return the best-effort corrected spec regardless of pass/fail so that
        # full renders benefit from the corrections even when no round passed threshold.
        return current_spec, probe_meta


# ---------------------------------------------------------------------------
# Probe-loop diversity helpers
# ---------------------------------------------------------------------------

_SPEC_EPSILON: float = 0.02  # 2% relative change threshold for no-op detection


def _spec_is_effectively_unchanged(
    prev: CameraPathSpec, curr: CameraPathSpec
) -> bool:
    """Return True if all numeric spec fields changed by less than _SPEC_EPSILON.

    Used to detect when apply_issue_tag_corrections hit every min-clamp and
    produced a spec that will render an identical video.
    """
    # A path-type change (e.g. orbit→manipulation) is always a meaningful
    # transition — never treat it as a no-op regardless of numeric field overlap.
    if str(getattr(prev, "type", "")).strip().lower() != str(
        getattr(curr, "type", "")
    ).strip().lower():
        return False

    import dataclasses

    prev_d = dataclasses.asdict(prev)
    curr_d = dataclasses.asdict(curr)
    for key in ("radius_m", "num_orbits", "length_m", "arc_radius_m", "arc_span_deg"):
        pv = prev_d.get(key)
        cv = curr_d.get(key)
        if pv is not None and cv is not None:
            ref = max(abs(float(pv)), 1e-6)
            if abs(float(cv) - float(pv)) / ref > _SPEC_EPSILON:
                return False
    return True


def _apply_diversity_kick(spec: CameraPathSpec, round_idx: int) -> CameraPathSpec:
    """Adjust standoff/height to explore a new viewpoint after a no-op correction.

    Alternates between closer/farther (even/odd round_idx) so repeated kicks
    don't oscillate to the same position.
    """
    from dataclasses import replace

    stype = str(spec.type).strip().lower()
    if stype == "orbit":
        cur_h = spec.height_override_m
        new_h = (cur_h if cur_h is not None else 1.2) * (
            0.7 if round_idx % 2 == 0 else 1.3
        )
        return replace(spec, height_override_m=float(max(0.4, min(new_h, 2.4))))
    if stype == "manipulation":
        cur_r = float(spec.arc_radius_m)
        new_r = cur_r * (1.4 if round_idx % 2 == 0 else 0.6)
        return replace(spec, arc_radius_m=float(max(0.15, min(new_r, 3.0))))
    if stype == "sweep":
        cur_len = float(spec.length_m)
        new_len = cur_len * (1.4 if round_idx % 2 == 0 else 0.6)
        return replace(spec, length_m=float(max(0.25, min(new_len, 6.0))))
    return spec


def _count_unique_camera_positions(poses: List[object], precision: int = 3) -> int:
    if not poses:
        return 0
    rounded = set()
    for pose in poses:
        try:
            pos = np.asarray(pose.position, dtype=np.float64).reshape(3)
        except Exception:
            continue
        rounded.add(tuple(np.round(pos, int(precision)).tolist()))
    return int(len(rounded))


def _compute_target_line_of_sight_ratio(
    *,
    poses: List[object],
    target_xyz: List[float],
    occupancy: Optional[OccupancyGrid],
    min_clearance_m: float,
) -> float:
    if not poses or occupancy is None:
        return 1.0
    target = np.asarray(target_xyz, dtype=np.float64).reshape(-1)
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return 0.0
    target = target[:3]
    clear = 0
    total = 0
    clearance = float(max(0.02, min(0.10, float(min_clearance_m) * 0.5)))
    for pose in poses:
        try:
            start = np.asarray(getattr(pose, "position"), dtype=np.float64).reshape(-1)[:3]
        except Exception:
            continue
        if start.size < 3 or not np.all(np.isfinite(start)):
            continue
        endpoint = _resolve_target_los_endpoint(
            start=start,
            target=target,
            occupancy=occupancy,
            min_clearance_m=float(min_clearance_m),
        )
        total += 1
        if occupancy.has_line_of_sight(
            start,
            endpoint,
            clearance_m=clearance,
            endpoint_margin_m=0.08,
        ):
            clear += 1
    if total <= 0:
        return 0.0
    return float(clear) / float(total)


def _resolve_target_los_endpoint(
    *,
    start: np.ndarray,
    target: np.ndarray,
    occupancy: Optional[OccupancyGrid],
    min_clearance_m: float,
) -> np.ndarray:
    """Choose a LOS endpoint near the visible target surface instead of center.

    Target centers are frequently inside occupied voxels for OBB-derived metadata.
    We back off from center toward the camera until we find a minimally free
    endpoint; LOS is then evaluated against that endpoint.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)[:3]
    if occupancy is None:
        return target
    start = np.asarray(start, dtype=np.float64).reshape(-1)[:3]
    if start.size < 3 or target.size < 3:
        return target
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(target))):
        return target

    ray = start - target
    dist = float(np.linalg.norm(ray))
    if dist <= 1e-6:
        return target
    direction = ray / dist

    max_backoff = float(np.clip(max(0.25, dist * 0.45), 0.10, 0.95))
    max_backoff = float(min(max_backoff, max(0.05, dist - 0.20)))
    endpoint_clearance = float(max(0.005, min(0.03, float(min_clearance_m) * 0.20)))

    for step in range(0, 13):
        frac = float(step) / 12.0
        backoff = max_backoff * frac
        endpoint = target + direction * backoff
        if occupancy.is_free(endpoint, min_clearance_m=endpoint_clearance):
            return endpoint

    # Best effort: return the furthest camera-facing endpoint tested.
    return target + direction * max_backoff


def _append_probe_score_row(path: Optional[Path], row: Dict[str, object]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    except Exception:
        logger.debug("Failed appending probe score row to %s", path, exc_info=True)


def _resample_poses_nearest(poses: List, requested_count: int) -> List:
    """Resample camera poses to a target count via deterministic nearest-neighbor indices."""
    if requested_count <= 0 or not poses:
        return poses
    if len(poses) == requested_count:
        return poses
    if len(poses) == 1:
        return [poses[0] for _ in range(requested_count)]

    sample_points = np.linspace(0, len(poses) - 1, num=requested_count, dtype=np.float64)
    indices = np.rint(sample_points).astype(np.int64)
    indices = np.clip(indices, 0, len(poses) - 1)
    return [poses[int(i)] for i in indices.tolist()]


def _camera_pose_metadata(pose) -> dict:
    """Serialize key camera pose fields for downstream task-conditioned selection."""
    c2w = np.asarray(pose.c2w, dtype=np.float64)
    forward = -c2w[:3, 2]
    right = c2w[:3, 0]
    up = c2w[:3, 1]
    return {
        "position": c2w[:3, 3].astype(float).tolist(),
        "forward": forward.astype(float).tolist(),
        "right": right.astype(float).tolist(),
        "up": up.astype(float).tolist(),
        "c2w": c2w.astype(float).tolist(),
        "fx": float(pose.fx),
        "fy": float(pose.fy),
        "cx": float(pose.cx),
        "cy": float(pose.cy),
        "width": int(pose.width),
        "height": int(pose.height),
    }


def _path_context_from_spec(path_spec: CameraPathSpec) -> dict:
    """Serialize minimal path-spec context into render manifest entries."""
    context = {
        "type": path_spec.type,
        "source_tag": path_spec.source_tag or "default",
        "height_override_m": (
            float(path_spec.height_override_m) if path_spec.height_override_m is not None else None
        ),
        "look_down_override_deg": (
            float(path_spec.look_down_override_deg)
            if path_spec.look_down_override_deg is not None
            else None
        ),
    }
    if path_spec.approach_point is not None:
        context["approach_point"] = [float(v) for v in path_spec.approach_point]
    if path_spec.type == "orbit":
        context["radius_m"] = float(path_spec.radius_m)
        context["num_orbits"] = int(path_spec.num_orbits)
    if path_spec.type == "sweep":
        context["length_m"] = float(path_spec.length_m)
    if path_spec.type == "manipulation":
        context["arc_radius_m"] = float(path_spec.arc_radius_m)
    if path_spec.target_instance_id is not None:
        context["target_instance_id"] = str(path_spec.target_instance_id)
    if path_spec.target_label is not None:
        context["target_label"] = str(path_spec.target_label)
    if path_spec.target_category is not None:
        context["target_category"] = str(path_spec.target_category)
    if path_spec.target_role is not None:
        context["target_role"] = str(path_spec.target_role)
    if path_spec.target_extents_m is not None:
        context["target_extents_m"] = [float(v) for v in path_spec.target_extents_m[:3]]
    if path_spec.locked_eye_point is not None:
        context["locked_eye_point"] = [float(v) for v in path_spec.locked_eye_point[:3]]
    if path_spec.locked_look_at_point is not None:
        context["locked_look_at_point"] = [float(v) for v in path_spec.locked_look_at_point[:3]]
    if path_spec.locked_probe_motion_radius_m is not None:
        context["locked_probe_motion_radius_m"] = float(path_spec.locked_probe_motion_radius_m)
    return context


def _path_context_target_xyz(path_context: dict | None) -> list[float] | None:
    if not isinstance(path_context, dict):
        return None
    point = path_context.get("approach_point")
    if not isinstance(point, list) or len(point) < 3:
        return None
    try:
        out = [float(point[0]), float(point[1]), float(point[2])]
    except Exception:
        return None
    if not np.all(np.isfinite(np.asarray(out, dtype=np.float64))):
        return None
    return out


def _path_context_target_extents(path_context: dict | None) -> list[float] | None:
    if not isinstance(path_context, dict):
        return None
    extents = path_context.get("target_extents_m")
    if not isinstance(extents, (list, tuple)) or len(extents) < 3:
        return None
    try:
        arr = np.asarray([float(extents[0]), float(extents[1]), float(extents[2])], dtype=np.float64)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    arr = np.maximum(arr, 0.01)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _is_target_grounded_path_context(path_context: dict | None) -> bool:
    if not isinstance(path_context, dict):
        return False
    role = str(path_context.get("target_role", "")).strip().lower()
    if role in {"targets", "context"}:
        return True
    for key in ("target_label", "target_instance_id", "target_category"):
        value = path_context.get(key)
        if value is not None and str(value).strip():
            return True
    return _path_context_target_xyz(path_context) is not None


def _build_task_prompt_pool(
    config: ValidationConfig,
    facility: FacilityConfig,
    profile: str,
) -> List[str]:
    tasks: List[str] = []
    for t in list(config.eval_policy.tasks or []) + list(
        config.eval_policy.manipulation_tasks or []
    ):
        task = str(t).strip()
        if task:
            tasks.append(task)

    if facility.task_hints_path is not None and facility.task_hints_path.exists():
        try:
            for task in tasks_from_task_hints(facility.task_hints_path, profile=profile):
                t = str(task).strip()
                if t:
                    tasks.append(t)
        except Exception as exc:
            logger.warning(
                "Failed to load task hints for task-scoped scene-aware selection: %s",
                exc,
            )

    return _dedupe_tasks(tasks)


def _dedupe_tasks(tasks: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for task in tasks:
        key = task.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(task.strip())
    return out


def _is_kitchen_0787_scene(facility: FacilityConfig) -> bool:
    text = " ".join(
        [
            str(getattr(facility, "name", "") or ""),
            str(getattr(facility, "description", "") or ""),
            str(getattr(facility, "ply_path", "") or ""),
            str(getattr(facility, "task_hints_path", "") or ""),
        ]
    ).lower()
    if "0787_841244" in text:
        return True
    return "0787" in text and "kitchen" in text


def _resolve_scene_locked_profile(
    config: ValidationConfig,
    facility: FacilityConfig,
) -> str | None:
    raw = str(getattr(config.render, "scene_locked_profile", "auto") or "auto").strip().lower()
    if raw in {"", "none"}:
        return None
    if raw == "auto":
        return "kitchen_0787" if _is_kitchen_0787_scene(facility) else None
    return raw


def _scene_locked_defaults(profile: str) -> Dict[str, object]:
    return dict(_SCENE_LOCKED_DEFAULTS.get(str(profile).strip().lower(), {}))


def _scene_locked_source_tag(profile: str) -> str:
    defaults = _scene_locked_defaults(profile)
    source_tag = str(defaults.get("source_tag", "")).strip()
    if source_tag:
        return source_tag
    return f"{_SCENE_LOCKED_SOURCE_TAG_PREFIX}{str(profile).strip().lower()}"


def _resolve_scene_locked_profile_from_spec(path_spec: CameraPathSpec) -> str | None:
    source_tag = str(getattr(path_spec, "source_tag", "") or "").strip().lower()
    if source_tag == "kitchen_0787_locked":
        return "kitchen_0787"
    if source_tag.startswith(_SCENE_LOCKED_SOURCE_TAG_PREFIX):
        suffix = source_tag[len(_SCENE_LOCKED_SOURCE_TAG_PREFIX) :].strip()
        return suffix or None
    return None


def _is_scene_locked_spec(path_spec: CameraPathSpec) -> bool:
    return _resolve_scene_locked_profile_from_spec(path_spec) is not None


def _stable_phase_from_key(text: str) -> float:
    key = str(text or "locked").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(key).hexdigest()
    unit = int(digest[:8], 16) / float(0xFFFFFFFF)
    return float(2.0 * np.pi * unit)


def _resolve_locked_pose_table_entry(
    profile: str,
    target_instance_id: object,
) -> Dict[str, object]:
    profile_key = str(profile or "").strip().lower()
    if profile_key == "facility_a":
        table = _FACILITY_A_LOCKED_TARGET_POSE_TABLE
    else:
        table = _KITCHEN_0787_LOCKED_TARGET_POSE_TABLE
    raw = str(target_instance_id or "").strip()
    if raw and raw in table:
        return dict(table[raw])
    m = re.search(r"([0-9]+)$", raw)
    if m:
        key = m.group(1)
        if key in table:
            return dict(table[key])
    return {}


def _as_valid_point3(value: object) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    arr = arr[:3]
    if not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float64)


def _transform_scene_point(point: np.ndarray, scene_transform: Optional[np.ndarray]) -> np.ndarray:
    if scene_transform is None:
        return np.asarray(point, dtype=np.float64).reshape(-1)[:3]
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    if p.size < 3:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    T = np.asarray(scene_transform, dtype=np.float64)
    if T.shape != (4, 4):
        return p[:3].astype(np.float64)
    ph = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0], dtype=np.float64)
    out = T @ ph
    return out[:3].astype(np.float64)


def _resolve_kitchen_0787_locked_pose_params(
    path_spec: CameraPathSpec,
) -> tuple[np.ndarray, np.ndarray, float]:
    profile = _resolve_scene_locked_profile_from_spec(path_spec) or "kitchen_0787"
    defaults = _scene_locked_defaults(profile)
    default_eye_offset = tuple(defaults.get("eye_offset_m", _KITCHEN_0787_LOCKED_DEFAULT_EYE_OFFSET_M))
    default_look_offset = tuple(defaults.get("look_at_offset_m", _KITCHEN_0787_LOCKED_DEFAULT_LOOK_AT_OFFSET_M))
    default_motion_radius = float(
        defaults.get("probe_motion_radius_m", _KITCHEN_0787_LOCKED_DEFAULT_PROBE_MOTION_RADIUS_M)
    )
    entry = _resolve_locked_pose_table_entry(profile, getattr(path_spec, "target_instance_id", None))
    motion_radius_raw = (
        getattr(path_spec, "locked_probe_motion_radius_m", None)
        if getattr(path_spec, "locked_probe_motion_radius_m", None) is not None
        else entry.get(
            "probe_motion_radius_m",
            default_motion_radius,
        )
    )
    try:
        motion_radius_m = float(motion_radius_raw)
    except Exception:
        motion_radius_m = default_motion_radius
    motion_radius_m = float(np.clip(motion_radius_m, 0.0, 0.05))

    target = _as_valid_point3(getattr(path_spec, "approach_point", None))
    eye = _as_valid_point3(getattr(path_spec, "locked_eye_point", None))
    look_at = _as_valid_point3(getattr(path_spec, "locked_look_at_point", None))
    if eye is None:
        eye = _as_valid_point3(entry.get("eye_world_m"))
    if look_at is None:
        look_at = _as_valid_point3(entry.get("look_at_world_m"))

    if eye is None or look_at is None:
        if target is None:
            eye = np.asarray([0.0, 0.0, 1.2], dtype=np.float64)
            look_at = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            eye = target + np.asarray(default_eye_offset, dtype=np.float64)
            look_at = target + np.asarray(default_look_offset, dtype=np.float64)

    if float(np.linalg.norm(look_at - eye)) <= 1e-4:
        if target is not None and float(np.linalg.norm(target - eye)) > 1e-4:
            look_at = target
        else:
            eye = eye + np.asarray([0.25, -0.10, 0.18], dtype=np.float64)
    return eye, look_at, motion_radius_m


def _build_scene_locked_poses(
    *,
    path_spec: CameraPathSpec,
    num_frames: int,
    resolution: tuple[int, int],
) -> List[CameraPose]:
    base_eye, look_at, motion_radius_m = _resolve_kitchen_0787_locked_pose_params(path_spec)
    if not (np.all(np.isfinite(base_eye)) and np.all(np.isfinite(look_at))):
        return []

    num = max(1, int(num_frames))
    height = max(2, int(resolution[0]))
    width = max(2, int(resolution[1]))
    fx = fy = float(width) / float(2.0 * np.tan(np.deg2rad(60.0 / 2.0)))
    cx, cy = float(width) / 2.0, float(height) / 2.0

    key = (
        str(getattr(path_spec, "target_instance_id", "")).strip()
        or str(getattr(path_spec, "target_label", "")).strip()
        or "locked"
    )
    phase = _stable_phase_from_key(key)
    turn_span = float(np.deg2rad(22.0))
    poses: List[CameraPose] = []
    for i in range(num):
        if num <= 1 or motion_radius_m <= 1e-6:
            theta = phase
        else:
            frac = float(i) / float(max(1, num - 1))
            theta = phase + turn_span * frac
        eye = base_eye.copy()
        eye[0] += motion_radius_m * np.cos(theta)
        eye[1] += 0.5 * motion_radius_m * np.sin(theta)
        eye[2] += 0.2 * motion_radius_m * np.sin(0.5 * theta)
        c2w = _look_at(eye, look_at)
        poses.append(
            CameraPose(
                c2w=c2w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
            )
        )
    return poses


def _build_kitchen_0787_locked_poses(
    *,
    path_spec: CameraPathSpec,
    num_frames: int,
    resolution: tuple[int, int],
) -> List[CameraPose]:
    return _build_scene_locked_poses(
        path_spec=path_spec,
        num_frames=num_frames,
        resolution=resolution,
    )


def _project_world_point_to_image(pose: object, xyz: np.ndarray) -> tuple[float, float] | None:
    try:
        c2w = np.asarray(getattr(pose, "c2w"), dtype=np.float64)
        fx = float(getattr(pose, "fx"))
        fy = float(getattr(pose, "fy"))
        cx = float(getattr(pose, "cx"))
        cy = float(getattr(pose, "cy"))
    except Exception:
        return None
    if c2w.shape != (4, 4):
        return None
    world = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
    cam = np.linalg.inv(c2w) @ world
    z = float(cam[2])
    # Camera convention in this pipeline is forward=-Z in camera space.
    if z >= -1e-6:
        return None
    u = fx * float(cam[0]) / (-z) + cx
    v = fy * float(cam[1]) / (-z) + cy
    return float(u), float(v)


def _estimate_target_projected_size_ratio(
    *,
    poses: List[object],
    target_xyz: list[float],
    target_extents_m: list[float],
) -> float:
    if not poses:
        return 0.0
    center = np.asarray(target_xyz, dtype=np.float64).reshape(-1)
    ext = np.asarray(target_extents_m, dtype=np.float64).reshape(-1)
    if center.size < 3 or ext.size < 3:
        return 0.0
    center = center[:3]
    ext = np.maximum(np.abs(ext[:3]), 0.01)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(ext)):
        return 0.0
    half = 0.5 * ext
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(center + np.array([sx * half[0], sy * half[1], sz * half[2]]))

    ratios: List[float] = []
    for pose in poses:
        try:
            width = max(1.0, float(getattr(pose, "width")))
            height = max(1.0, float(getattr(pose, "height")))
        except Exception:
            continue
        points = [_project_world_point_to_image(pose, c) for c in corners]
        points = [p for p in points if p is not None]
        if len(points) < 4:
            continue
        u_vals = [float(p[0]) for p in points]
        v_vals = [float(p[1]) for p in points]
        u0 = max(0.0, min(width, min(u_vals)))
        u1 = max(0.0, min(width, max(u_vals)))
        v0 = max(0.0, min(height, min(v_vals)))
        v1 = max(0.0, min(height, max(v_vals)))
        box_w = max(0.0, u1 - u0)
        box_h = max(0.0, v1 - v0)
        if box_w <= 0.0 or box_h <= 0.0:
            continue
        ratios.append(float(max(box_w / width, box_h / height)))
    if not ratios:
        return 0.0
    return float(np.median(np.asarray(ratios, dtype=np.float64)))


def _build_scene_locked_specs(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    scene_transform: Optional[np.ndarray],
    profile: str,
) -> List[CameraPathSpec]:
    """Build deterministic target-grounded specs for fixed eye/look-at capture."""
    task_hints_path = Path(facility.task_hints_path) if facility.task_hints_path else None
    if task_hints_path is None or not task_hints_path.exists():
        logger.warning(
            "%s scene-locked mode skipped: missing task_hints_path (%s)",
            profile,
            facility.task_hints_path,
        )
        return []

    try:
        obbs = load_obbs_from_task_targets(task_hints_path)
    except Exception:
        logger.warning(
            "%s scene-locked mode failed to load OBBs from %s",
            profile,
            task_hints_path,
            exc_info=True,
        )
        return []
    if not obbs:
        return []
    if scene_transform is not None:
        obbs = transform_obbs(obbs, scene_transform)

    selected_obbs: List[OrientedBoundingBox]
    role_by_instance: Dict[str, str]
    tasks = _build_task_prompt_pool(
        config=config,
        facility=facility,
        profile=config.render.task_scoped_profile,
    )
    selected_obbs, scoped_stats, role_by_instance = _select_task_scoped_obbs(
        obbs=obbs,
        tasks=tasks,
        max_specs=int(_scene_locked_defaults(profile).get("max_targets", _KITCHEN_0787_LOCKED_MAX_TARGETS)),
        context_per_target=1,
        overview_specs=0,
        fallback_specs=int(_scene_locked_defaults(profile).get("max_targets", _KITCHEN_0787_LOCKED_MAX_TARGETS)),
        center_dedupe_dist_m=float(max(0.0, config.render.stage1_probe_dedupe_center_dist_m)),
    )
    if not selected_obbs:
        return []

    role_priority = {"targets": 0, "context": 1, "fallback": 2, "overview": 3}
    cat_priority = {"manipulation": 0, "articulation": 1, "navigation": 2}
    ordered = sorted(
        selected_obbs,
        key=lambda obb: (
            role_priority.get(role_by_instance.get(str(obb.instance_id), "fallback"), 4),
            cat_priority.get(str(obb.category).strip().lower(), 9),
            _label_key(obb.label),
            str(obb.instance_id),
        ),
    )

    specs: List[CameraPathSpec] = []
    for obb in ordered:
        role = str(role_by_instance.get(str(obb.instance_id), "targets") or "targets").strip().lower()
        if role not in {"targets", "context", "fallback", "overview"}:
            role = "targets"
        ext_arr = np.asarray(obb.extents, dtype=np.float64).reshape(-1)
        if ext_arr.size < 3:
            ext_arr = np.pad(ext_arr, (0, 3 - ext_arr.size), constant_values=0.1)
        table_entry = _resolve_locked_pose_table_entry(profile, obb.instance_id)
        eye_abs = _as_valid_point3(table_entry.get("eye_world_m"))
        look_abs = _as_valid_point3(table_entry.get("look_at_world_m"))
        if eye_abs is not None and look_abs is not None:
            eye_abs = _transform_scene_point(eye_abs, scene_transform)
            look_abs = _transform_scene_point(look_abs, scene_transform)
        else:
            eye_abs = np.asarray(obb.center, dtype=np.float64).reshape(-1)[:3] + np.asarray(
                _KITCHEN_0787_LOCKED_DEFAULT_EYE_OFFSET_M, dtype=np.float64
            )
            look_abs = np.asarray(obb.center, dtype=np.float64).reshape(-1)[:3] + np.asarray(
                _KITCHEN_0787_LOCKED_DEFAULT_LOOK_AT_OFFSET_M, dtype=np.float64
            )
        try:
            probe_motion_radius_m = float(
                table_entry.get(
                    "probe_motion_radius_m", _KITCHEN_0787_LOCKED_DEFAULT_PROBE_MOTION_RADIUS_M
                )
            )
        except Exception:
            probe_motion_radius_m = float(_KITCHEN_0787_LOCKED_DEFAULT_PROBE_MOTION_RADIUS_M)
        probe_motion_radius_m = float(np.clip(probe_motion_radius_m, 0.0, 0.05))

        meta = {
            "source_tag": _scene_locked_source_tag(profile),
            "target_instance_id": str(obb.instance_id) if str(obb.instance_id).strip() else None,
            "target_label": str(obb.label) if str(obb.label).strip() else None,
            "target_category": str(obb.category) if str(obb.category).strip() else None,
            "target_role": role,
            "locked_eye_point": [float(eye_abs[0]), float(eye_abs[1]), float(eye_abs[2])],
            "locked_look_at_point": [float(look_abs[0]), float(look_abs[1]), float(look_abs[2])],
            "locked_probe_motion_radius_m": float(probe_motion_radius_m),
            "target_extents_m": [
                float(max(0.01, abs(float(ext_arr[0])))),
                float(max(0.01, abs(float(ext_arr[1])))),
                float(max(0.01, abs(float(ext_arr[2])))),
            ],
        }

        approach = [float(obb.center[0]), float(obb.center[1]), float(obb.center[2])]
        specs.append(
            CameraPathSpec(
                type="file",
                radius_m=1.0,
                num_orbits=1,
                approach_point=approach,
                height_override_m=None,
                look_down_override_deg=None,
                **meta,
            )
        )

    logger.info(
        "%s scene-locked spec builder selected %d OBBs (targets=%d context=%d fallback=%d) -> %d locked specs",
        profile,
        len(selected_obbs),
        int(scoped_stats.get("targets", 0)),
        int(scoped_stats.get("context", 0)),
        int(scoped_stats.get("fallback", 0)),
        len(specs),
    )
    return specs


def _build_kitchen_0787_locked_specs(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    scene_transform: Optional[np.ndarray],
) -> List[CameraPathSpec]:
    return _build_scene_locked_specs(
        config=config,
        facility=facility,
        scene_transform=scene_transform,
        profile="kitchen_0787",
    )


def _select_task_scoped_obbs(
    *,
    obbs: List[OrientedBoundingBox],
    tasks: List[str],
    max_specs: int,
    context_per_target: int,
    overview_specs: int,
    fallback_specs: int,
    center_dedupe_dist_m: float = 0.08,
) -> tuple[List[OrientedBoundingBox], Dict[str, int], Dict[str, str]]:
    stats = {"targets": 0, "context": 0, "overview": 0, "fallback": 0}
    if not obbs or max_specs <= 0:
        return [], stats, {}

    max_specs = max(1, int(max_specs))
    fallback_specs = max(1, int(fallback_specs))
    obbs = _dedupe_task_scoped_obbs(
        obbs,
        center_dedupe_dist_m=float(max(0.0, center_dedupe_dist_m)),
    )
    by_instance, by_label = _build_obb_lookup(obbs)

    ordered_indices: List[int] = []
    role_by_index: Dict[int, str] = {}
    selected = set()

    def add_idx(idx: int, role: str) -> None:
        if idx in selected:
            return
        selected.add(idx)
        ordered_indices.append(idx)
        role_by_index[idx] = role

    # 1) Primary task targets from explicit prompts.
    for task in tasks:
        idx = _resolve_task_target_index(task, by_instance, by_label)
        if idx is not None:
            add_idx(idx, "targets")

    # 2) Fallback: if task parsing finds nothing, keep a small useful subset.
    if not ordered_indices:
        for idx in _fallback_obb_indices(
            obbs,
            limit=min(max_specs, fallback_specs),
        ):
            add_idx(idx, "fallback")
        selected_obbs = [obbs[i] for i in ordered_indices[:max_specs]]
        stats["fallback"] = len(selected_obbs)
        role_by_instance = {
            str(obbs[i].instance_id): str(role_by_index.get(i, "")).strip() for i in ordered_indices[:max_specs]
        }
        return selected_obbs, stats, role_by_instance

    # 3) Add local context objects near each primary target.
    primary_indices = list(ordered_indices)
    for pidx in primary_indices:
        if len(ordered_indices) >= max_specs:
            break
        pcenter = obbs[pidx].center
        candidates = []
        for idx, obb in enumerate(obbs):
            if idx in selected:
                continue
            dist = float(np.linalg.norm(obb.center - pcenter))
            candidates.append((dist, idx))
        candidates.sort(key=lambda x: (x[0], x[1]))
        for _, idx in candidates[: max(0, int(context_per_target))]:
            add_idx(idx, "context")
            if len(ordered_indices) >= max_specs:
                break

    # 4) Add a few overview/navigation anchors for broader coverage.
    if overview_specs > 0 and len(ordered_indices) < max_specs:
        anchors = [obbs[idx].center for idx in ordered_indices]
        for _ in range(int(overview_specs)):
            if len(ordered_indices) >= max_specs:
                break
            remaining = [idx for idx in range(len(obbs)) if idx not in selected]
            if not remaining:
                break
            best_idx = None
            best_score = -1e9
            for idx in remaining:
                obb = obbs[idx]
                if anchors:
                    dmin = min(float(np.linalg.norm(obb.center - a)) for a in anchors)
                else:
                    dmin = 0.0
                cat_bonus = {
                    "navigation": 0.25,
                    "articulation": 0.15,
                    "manipulation": 0.0,
                }.get(str(obb.category).strip().lower(), 0.0)
                score = dmin + cat_bonus
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            add_idx(best_idx, "overview")
            anchors.append(obbs[best_idx].center)

    final_indices = ordered_indices[:max_specs]
    selected_obbs = [obbs[i] for i in final_indices]
    for idx in final_indices:
        role = role_by_index.get(idx, "")
        if role in stats:
            stats[role] += 1
    role_by_instance = {
        str(obbs[i].instance_id): str(role_by_index.get(i, "")).strip() for i in final_indices
    }
    return selected_obbs, stats, role_by_instance


def _dedupe_task_scoped_obbs(
    obbs: List[OrientedBoundingBox],
    *,
    center_dedupe_dist_m: float,
) -> List[OrientedBoundingBox]:
    if not obbs or float(center_dedupe_dist_m) <= 0.0:
        return list(obbs)

    kept: List[OrientedBoundingBox] = []
    for obb in obbs:
        label_key = _label_key(obb.label)
        center = np.asarray(obb.center, dtype=np.float64)
        duplicate_idx: Optional[int] = None
        for idx, prev in enumerate(kept):
            if _label_key(prev.label) != label_key:
                continue
            prev_center = np.asarray(prev.center, dtype=np.float64)
            if float(np.linalg.norm(center - prev_center)) < float(center_dedupe_dist_m):
                duplicate_idx = idx
                break
        if duplicate_idx is None:
            kept.append(obb)
            continue
        prev = kept[duplicate_idx]
        if float(getattr(obb, "confidence", 0.0)) > float(getattr(prev, "confidence", 0.0)):
            kept[duplicate_idx] = obb
    return kept


def _build_obb_lookup(
    obbs: List[OrientedBoundingBox],
) -> tuple[Dict[str, int], Dict[str, List[int]]]:
    by_instance: Dict[str, int] = {}
    by_label: Dict[str, List[int]] = {}
    for idx, obb in enumerate(obbs):
        iid = str(obb.instance_id).strip()
        if iid and iid not in by_instance:
            by_instance[iid] = idx
        lkey = _label_key(obb.label)
        by_label.setdefault(lkey, []).append(idx)
    return by_instance, by_label


def _label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


def _resolve_task_target_index(
    task: str,
    by_instance: Dict[str, int],
    by_label: Dict[str, List[int]],
) -> Optional[int]:
    lowered = str(task).strip().lower()
    if not lowered:
        return None

    # explicit object token near action verbs (e.g. "pick up bowl_101 ...")
    token_match = re.search(
        r"(?:pick up|open and close|turn on|turn off|toggle|approach|go to|move toward|navigate to)\s+([a-z0-9_]+)",
        lowered,
    )
    if token_match:
        resolved = _resolve_token_to_index(token_match.group(1), by_instance, by_label)
        if resolved is not None:
            return resolved

    # any explicit label_123 token anywhere in the prompt
    explicit_match = re.search(r"\b([a-z][a-z0-9_]*_[0-9]{1,})\b", lowered)
    if explicit_match:
        resolved = _resolve_token_to_index(explicit_match.group(1), by_instance, by_label)
        if resolved is not None:
            return resolved

    # navigation/object phrase with spaces
    nav_match = re.search(
        r"(?:navigate to|approach|go to|move toward)\s+(?:the\s+)?([a-z0-9_ ]+)", lowered
    )
    if nav_match:
        label_key = _label_key(nav_match.group(1))
        options = by_label.get(label_key, [])
        if options:
            return options[0]
    return None


def _resolve_token_to_index(
    token: str,
    by_instance: Dict[str, int],
    by_label: Dict[str, List[int]],
) -> Optional[int]:
    token = str(token).strip().strip("_")
    if not token:
        return None

    if token in by_instance:
        return by_instance[token]

    m = re.match(r"(.+?)_([0-9]+)$", token)
    if m:
        instance_id = m.group(2)
        if instance_id in by_instance:
            return by_instance[instance_id]
        options = by_label.get(_label_key(m.group(1)), [])
        if options:
            return options[0]

    options = by_label.get(_label_key(token), [])
    if options:
        return options[0]
    return None


def _fallback_obb_indices(obbs: List[OrientedBoundingBox], limit: int) -> List[int]:
    def key_fn(item: tuple[int, OrientedBoundingBox]) -> tuple:
        idx, obb = item
        cat_pri = {
            "manipulation": 0,
            "articulation": 1,
            "navigation": 2,
        }.get(str(obb.category).strip().lower(), 3)
        return (cat_pri, -float(obb.confidence), _label_key(obb.label), idx)

    ranked = sorted(enumerate(obbs), key=key_fn)
    return [idx for idx, _ in ranked[: max(1, int(limit))]]


def _sample_start_offset(
    config: ValidationConfig,
    path_spec: CameraPathSpec,
    rng: np.random.Generator,
    *,
    clip_repeat_index: int = 0,
    is_task_scoped: bool = False,
    target_grounded: bool = False,
) -> np.ndarray:
    """Sample deterministic XY offset with stricter defaults for manipulation clips."""
    if bool(target_grounded):
        return np.zeros(3, dtype=np.float64)
    span = (
        float(config.render.manipulation_random_xy_offset_m)
        if str(path_spec.type).strip().lower() == "manipulation"
        else float(config.render.non_manipulation_random_xy_offset_m)
    )
    offset = np.zeros(3, dtype=np.float64)
    if span > 0.0:
        offset[:2] = rng.uniform(-span, span, size=2)
    # Repeated task-scoped clips need deterministic diversity even when span=0.
    elif is_task_scoped and int(clip_repeat_index) > 0:
        jitter = float(max(0.0, config.render.stage1_repeat_min_xy_jitter_m))
        if jitter > 0.0:
            theta = float(clip_repeat_index) * 1.1137
            offset[0] = jitter * np.cos(theta)
            offset[1] = jitter * np.sin(theta)
    return offset


def _offset_with_dedupe_jitter(
    *,
    offset: np.ndarray,
    attempt: int,
    min_jitter: float,
) -> np.ndarray:
    updated = np.asarray(offset, dtype=np.float64).copy()
    jitter = float(max(0.0, min_jitter))
    if jitter <= 0.0:
        return updated
    scale = 1.0 + 0.5 * float(max(1, int(attempt)))
    theta = 0.73 * float(max(1, int(attempt)))
    updated[0] += jitter * scale * np.cos(theta)
    updated[1] += jitter * scale * np.sin(theta)
    return updated


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _retry_adjusted_spec(path_spec: CameraPathSpec, attempt: int) -> CameraPathSpec:
    """Deterministic per-attempt spec adjustment used by bounded Stage-1 retries."""
    step = max(1, int(attempt))
    if str(path_spec.type).strip().lower() == "manipulation":
        return replace(
            path_spec,
            arc_radius_m=max(0.15, float(path_spec.arc_radius_m) * (1.0 + 0.08 * step)),
            height_override_m=(
                None
                if path_spec.height_override_m is None
                else max(0.25, float(path_spec.height_override_m) + 0.04 * step)
            ),
            look_down_override_deg=(
                None
                if path_spec.look_down_override_deg is None
                else max(5.0, min(80.0, float(path_spec.look_down_override_deg) + 2.0 * step))
            ),
        )
    if str(path_spec.type).strip().lower() == "orbit":
        return replace(
            path_spec,
            radius_m=max(0.30, float(path_spec.radius_m) * (1.0 + 0.08 * step)),
            height_override_m=(
                None
                if path_spec.height_override_m is None
                else max(0.25, float(path_spec.height_override_m) + 0.03 * step)
            ),
            look_down_override_deg=(
                None
                if path_spec.look_down_override_deg is None
                else max(2.0, min(80.0, float(path_spec.look_down_override_deg) + 1.5 * step))
            ),
        )
    if str(path_spec.type).strip().lower() == "sweep":
        return replace(
            path_spec,
            length_m=max(0.30, float(path_spec.length_m) * (1.0 + 0.08 * step)),
            height_override_m=(
                None
                if path_spec.height_override_m is None
                else max(0.25, float(path_spec.height_override_m) + 0.03 * step)
            ),
            look_down_override_deg=(
                None
                if path_spec.look_down_override_deg is None
                else max(2.0, min(80.0, float(path_spec.look_down_override_deg) + 1.5 * step))
            ),
        )
    return path_spec


def _cache_path_context(clip_data: Dict[str, object]) -> dict:
    context = clip_data.get("path_context")
    if isinstance(context, dict) and context:
        out = dict(context)
        out.setdefault("source", "warmup_cache")
        return out
    fallback = {"source": "warmup_cache"}
    for key in (
        "target_instance_id",
        "target_label",
        "target_category",
        "target_role",
        "approach_point",
    ):
        if key in clip_data:
            fallback[key] = clip_data[key]
    return fallback


def _quality_cache_key(
    *,
    config: ValidationConfig,
    task_hints_path: Optional[Path],
) -> str:
    hints_sig = ""
    if task_hints_path is not None and Path(task_hints_path).exists():
        try:
            st = Path(task_hints_path).stat()
            hints_sig = f"{int(st.st_mtime_ns)}:{int(st.st_size)}"
        except Exception:
            hints_sig = "unknown"
    bits = [
        str(bool(config.render.stage1_quality_planner_enabled)),
        str(config.render.stage1_quality_candidate_budget),
        str(bool(config.render.stage1_quality_autoretry_enabled)),
        str(int(config.render.stage1_quality_max_regen_attempts)),
        f"{float(config.render.stage1_quality_min_clip_score):.6f}",
        f"{float(config.render.stage1_coverage_min_visible_frame_ratio):.6f}",
        f"{float(config.render.stage1_coverage_min_center_band_ratio):.6f}",
        str(int(config.render.stage1_coverage_min_approach_angle_bins)),
        f"{float(config.render.stage1_coverage_angle_bin_deg):.6f}",
        f"{float(config.render.stage1_coverage_blur_laplacian_min):.6f}",
        str(bool(config.render.stage1_active_perception_enabled)),
        str(config.render.stage1_active_perception_scope),
        str(config.render.scene_locked_profile),
        str(int(config.render.stage1_active_perception_max_loops)),
        str(bool(config.render.stage1_active_perception_fail_closed)),
        str(int(config.render.stage1_probe_frames_override)),
        f"{float(config.render.stage1_probe_resolution_scale):.6f}",
        f"{float(config.render.stage1_probe_min_viable_pose_ratio):.6f}",
        str(int(config.render.stage1_probe_min_unique_positions)),
        str(bool(config.render.stage1_probe_dedupe_enabled)),
        str(int(config.render.stage1_probe_dedupe_max_regen_attempts)),
        f"{float(config.render.stage1_probe_dedupe_center_dist_m):.6f}",
        str(int(config.render.stage1_probe_consensus_votes)),
        f"{float(config.render.stage1_probe_consensus_high_variance_delta):.6f}",
        str(int(config.render.stage1_probe_tiebreak_extra_votes)),
        f"{float(config.render.stage1_probe_tiebreak_spread_threshold):.6f}",
        str(bool(config.render.stage1_probe_primary_model_only)),
        f"{float(config.render.stage1_vlm_min_task_score):.6f}",
        f"{float(config.render.stage1_vlm_min_visual_score):.6f}",
        f"{float(config.render.stage1_vlm_min_spatial_score):.6f}",
        str(bool(config.render.stage1_keep_probe_videos)),
        str(bool(config.render.stage1_repeat_dedupe_enabled)),
        str(int(config.render.stage1_repeat_dedupe_max_regen_attempts)),
        f"{float(config.render.stage1_repeat_min_xy_jitter_m):.6f}",
        f"{float(config.render.stage1_repeat_similarity_ssim_threshold):.6f}",
        f"{float(config.eval_policy.vlm_judge.video_metadata_fps):.6f}",
        str(bool(config.render.vlm_fallback)),
        hints_sig,
    ]
    return "|".join(bits)


def _cleanup_probe_outputs(output) -> None:
    for attr in ("video_path", "depth_video_path"):
        value = getattr(output, attr, None)
        if value is None:
            continue
        path = Path(str(value))
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed deleting probe output: %s", path, exc_info=True)


def _empty_quality_summary() -> Dict[str, object]:
    return {
        "quality_gate_passed": True,
        "num_quality_failures": 0,
        "num_quality_retries": 0,
        "num_quality_recovered": 0,
        "num_vlm_probe_evaluated": 0,
        "num_vlm_probe_failures": 0,
        "num_vlm_probe_retries": 0,
        "num_vlm_probe_recovered": 0,
        "num_vlm_probe_api_failures": 0,
        "num_vlm_probe_parse_failures": 0,
        "num_vlm_probe_high_variance": 0,
        "num_probe_duplicate_detected": 0,
        "num_probe_duplicate_regenerated": 0,
        "num_probe_duplicate_unresolved": 0,
        "num_probe_viability_rejects": 0,
        "num_probe_monochrome_warnings": 0,
        "num_repeat_duplicate_detected": 0,
        "num_repeat_duplicate_regenerated": 0,
        "num_repeat_duplicate_unresolved": 0,
        "num_clips_with_target_metadata": 0,
        "num_missing_target_annotations": 0,
    }


def _default_vlm_probe_fields(*, selected_fps: float, candidate_count: int) -> Dict[str, object]:
    return {
        "vlm_probe_attempts": 0,
        "vlm_probe_passed": True,
        "vlm_probe_retries_used": 0,
        "vlm_probe_issue_tags_final": [],
        "vlm_probe_candidate_count": int(max(1, candidate_count)),
        "vlm_probe_selected_fps": (
            None if float(selected_fps) <= 0.0 else round(float(selected_fps), 3)
        ),
        "active_model_used": "",
        "vlm_probe_consensus_votes_configured": 1,
        "vlm_probe_consensus_votes_effective": 0,
        "vlm_probe_score_spread": None,
        "vlm_probe_fail_reason": None,
        "vlm_probe_evaluated": False,
        "num_vlm_probe_api_failures": 0,
        "num_vlm_probe_parse_failures": 0,
        "num_vlm_probe_high_variance": 0,
        "num_probe_duplicate_detected": 0,
        "num_probe_duplicate_regenerated": 0,
        "num_probe_duplicate_unresolved": 0,
        "num_probe_viability_rejects": 0,
        "num_probe_monochrome_warnings": 0,
        "probe_codec": "",
        "probe_resolution": None,
        "probe_decoded_frames": 0,
    }


def _manifest_vlm_probe_fields(probe_meta: Dict[str, object]) -> Dict[str, object]:
    return {
        "vlm_probe_attempts": int(max(0, int(probe_meta.get("vlm_probe_attempts", 0) or 0))),
        "vlm_probe_passed": bool(probe_meta.get("vlm_probe_passed", True)),
        "vlm_probe_retries_used": int(
            max(0, int(probe_meta.get("vlm_probe_retries_used", 0) or 0))
        ),
        "vlm_probe_issue_tags_final": list(probe_meta.get("vlm_probe_issue_tags_final", []) or []),
        "vlm_probe_candidate_count": int(
            max(1, int(probe_meta.get("vlm_probe_candidate_count", 1) or 1))
        ),
        "vlm_probe_selected_fps": probe_meta.get("vlm_probe_selected_fps"),
        "active_model_used": str(probe_meta.get("active_model_used", "") or ""),
        "vlm_probe_consensus_votes_configured": int(
            max(1, int(probe_meta.get("vlm_probe_consensus_votes_configured", 1) or 1))
        ),
        "vlm_probe_consensus_votes_effective": int(
            max(0, int(probe_meta.get("vlm_probe_consensus_votes_effective", 0) or 0))
        ),
        "vlm_probe_score_spread": probe_meta.get("vlm_probe_score_spread"),
        "vlm_probe_fail_reason": probe_meta.get("vlm_probe_fail_reason"),
        "num_probe_duplicate_detected": int(
            max(0, int(probe_meta.get("num_probe_duplicate_detected", 0) or 0))
        ),
        "num_probe_duplicate_regenerated": int(
            max(0, int(probe_meta.get("num_probe_duplicate_regenerated", 0) or 0))
        ),
        "num_probe_duplicate_unresolved": int(
            max(0, int(probe_meta.get("num_probe_duplicate_unresolved", 0) or 0))
        ),
        "num_probe_viability_rejects": int(
            max(0, int(probe_meta.get("num_probe_viability_rejects", 0) or 0))
        ),
        "num_probe_monochrome_warnings": int(
            max(0, int(probe_meta.get("num_probe_monochrome_warnings", 0) or 0))
        ),
        "probe_codec": str(probe_meta.get("probe_codec", "") or ""),
        "probe_resolution": probe_meta.get("probe_resolution"),
        "probe_decoded_frames": int(max(0, int(probe_meta.get("probe_decoded_frames", 0) or 0))),
    }


def _update_vlm_probe_summary(
    *,
    quality_summary: Dict[str, object],
    probe_meta: Dict[str, object],
) -> None:
    if not bool(probe_meta.get("vlm_probe_evaluated", False)):
        return
    quality_summary["num_vlm_probe_evaluated"] = int(quality_summary["num_vlm_probe_evaluated"]) + 1
    retries_used = int(max(0, int(probe_meta.get("vlm_probe_retries_used", 0) or 0)))
    quality_summary["num_vlm_probe_retries"] = int(quality_summary["num_vlm_probe_retries"]) + retries_used
    if bool(probe_meta.get("vlm_probe_passed", False)):
        if retries_used > 0:
            quality_summary["num_vlm_probe_recovered"] = int(
                quality_summary["num_vlm_probe_recovered"]
            ) + 1
    else:
        quality_summary["num_vlm_probe_failures"] = int(
            quality_summary["num_vlm_probe_failures"]
        ) + 1
    quality_summary["num_vlm_probe_api_failures"] = int(
        quality_summary["num_vlm_probe_api_failures"]
    ) + int(max(0, int(probe_meta.get("num_vlm_probe_api_failures", 0) or 0)))
    quality_summary["num_vlm_probe_parse_failures"] = int(
        quality_summary["num_vlm_probe_parse_failures"]
    ) + int(max(0, int(probe_meta.get("num_vlm_probe_parse_failures", 0) or 0)))
    quality_summary["num_vlm_probe_high_variance"] = int(
        quality_summary["num_vlm_probe_high_variance"]
    ) + int(max(0, int(probe_meta.get("num_vlm_probe_high_variance", 0) or 0)))
    quality_summary["num_probe_duplicate_detected"] = int(
        quality_summary["num_probe_duplicate_detected"]
    ) + int(max(0, int(probe_meta.get("num_probe_duplicate_detected", 0) or 0)))
    quality_summary["num_probe_duplicate_regenerated"] = int(
        quality_summary["num_probe_duplicate_regenerated"]
    ) + int(max(0, int(probe_meta.get("num_probe_duplicate_regenerated", 0) or 0)))
    quality_summary["num_probe_duplicate_unresolved"] = int(
        quality_summary["num_probe_duplicate_unresolved"]
    ) + int(max(0, int(probe_meta.get("num_probe_duplicate_unresolved", 0) or 0)))
    quality_summary["num_probe_viability_rejects"] = int(
        quality_summary["num_probe_viability_rejects"]
    ) + int(max(0, int(probe_meta.get("num_probe_viability_rejects", 0) or 0)))
    quality_summary["num_probe_monochrome_warnings"] = int(
        quality_summary["num_probe_monochrome_warnings"]
    ) + int(max(0, int(probe_meta.get("num_probe_monochrome_warnings", 0) or 0)))


def _update_quality_summary_from_entry(
    *,
    quality_summary: Dict[str, object],
    clip_entry: Dict[str, object],
    enforce_fail: bool,
) -> None:
    path_context = clip_entry.get("path_context") or {}
    target_xyz = path_context.get("approach_point") if isinstance(path_context, dict) else None
    if isinstance(target_xyz, list) and len(target_xyz) == 3:
        quality_summary["num_clips_with_target_metadata"] = int(
            quality_summary["num_clips_with_target_metadata"]
        ) + 1
    else:
        quality_summary["num_missing_target_annotations"] = int(
            quality_summary["num_missing_target_annotations"]
        ) + 1

    retries_used = int(max(0, int(clip_entry.get("quality_retries_used", 0) or 0)))
    quality_summary["num_quality_retries"] = int(quality_summary["num_quality_retries"]) + retries_used
    passed = bool(clip_entry.get("quality_gate_passed", True))
    if retries_used > 0 and passed:
        quality_summary["num_quality_recovered"] = int(quality_summary["num_quality_recovered"]) + 1
    if enforce_fail and not passed:
        quality_summary["num_quality_failures"] = int(quality_summary["num_quality_failures"]) + 1


def _ensure_probe_h264_for_scoring(video_path: Path, min_frames: int):
    """Transcode probe clips to H.264 when needed for stable VLM ingestion."""
    return ensure_h264_video(
        input_path=video_path,
        min_decoded_frames=max(1, int(min_frames)),
        output_path=video_path.with_name(f"{video_path.stem}_h264.mp4"),
        replace_source=False,
        crf=14,
        preset="slow",
    )


def _score_stage1_probe_consensus(
    *,
    video_path: Path,
    expected_focus_text: str,
    config: VLMJudgeConfig,
    facility_description: str,
    votes: int,
    primary_model_only: bool,
    tiebreak_extra_votes: int = 0,
    tiebreak_spread_threshold: float = 3.0,
) -> Dict[str, object]:
    configured_votes = max(1, int(votes))
    extra_votes = max(0, int(tiebreak_extra_votes))
    tiebreak_threshold = float(tiebreak_spread_threshold)
    primary_cfg = replace(config, fallback_models=[]) if primary_model_only else config

    def _collect_votes(
        judge_cfg: VLMJudgeConfig,
        *,
        phase: str,
        num_votes: int,
        vote_rows: List[Dict[str, object]],
    ) -> Dict[str, object]:
        phase_successes: List[Stage1ProbeScore] = []
        phase_api_failures = 0
        phase_parse_failures = 0
        phase_errors: List[str] = []
        for vote_idx in range(max(1, int(num_votes))):
            try:
                score = score_stage1_probe(
                    video_path=video_path,
                    expected_focus_text=expected_focus_text,
                    config=judge_cfg,
                    facility_description=facility_description,
                )
                phase_successes.append(score)
                vote_rows.append(
                    {
                        "phase": phase,
                        "vote_index": int(vote_idx),
                        "model_used": str(getattr(score, "model_used", "") or str(judge_cfg.model)),
                        "task_score": float(score.task_score),
                        "visual_score": float(score.visual_score),
                        "spatial_score": float(score.spatial_score),
                        "issue_tags": list(score.issue_tags),
                        "reasoning": str(score.reasoning or ""),
                        "error": None,
                    }
                )
            except Exception as exc:
                msg = str(exc)
                phase_errors.append(f"{phase}:vote={vote_idx + 1}: {msg}")
                lowered = msg.lower()
                if "json" in lowered or "parse" in lowered:
                    phase_parse_failures += 1
                else:
                    phase_api_failures += 1
                vote_rows.append(
                    {
                        "phase": phase,
                        "vote_index": int(vote_idx),
                        "model_used": str(judge_cfg.model),
                        "task_score": None,
                        "visual_score": None,
                        "spatial_score": None,
                        "issue_tags": [],
                        "reasoning": "",
                        "error": msg,
                    }
                )
        return {
            "successes": phase_successes,
            "api_failures": int(phase_api_failures),
            "parse_failures": int(phase_parse_failures),
            "errors": phase_errors,
        }

    vote_rows: List[Dict[str, object]] = []
    active_cfg = primary_cfg

    primary_out = _collect_votes(
        primary_cfg,
        phase="primary",
        num_votes=configured_votes,
        vote_rows=vote_rows,
    )
    successes = list(primary_out["successes"])
    api_failures = int(primary_out["api_failures"])
    parse_failures = int(primary_out["parse_failures"])
    errors: List[str] = list(primary_out["errors"])

    # Strict runs prefer the primary model; only allow fallback if the primary
    # model failed every vote (hard-failure path).
    if (
        primary_model_only
        and not successes
        and bool(config.fallback_models)
    ):
        active_cfg = config
        fallback_out = _collect_votes(
            config,
            phase="fallback",
            num_votes=configured_votes,
            vote_rows=vote_rows,
        )
        successes = list(fallback_out["successes"])
        api_failures += int(fallback_out["api_failures"])
        parse_failures += int(fallback_out["parse_failures"])
        errors.extend(list(fallback_out["errors"]))

    if not successes:
        return {
            "score": None,
            "error": "all_probe_votes_failed: " + "; ".join(errors),
            "num_api_failures": int(api_failures),
            "num_parse_failures": int(parse_failures),
            "votes_effective": 0,
            "score_spread": None,
            "active_model_used": str(config.model),
            "vote_rows": vote_rows,
        }

    if extra_votes > 0 and tiebreak_threshold > 0.0:
        task_vals = np.array([float(s.task_score) for s in successes], dtype=np.float64)
        visual_vals = np.array([float(s.visual_score) for s in successes], dtype=np.float64)
        spatial_vals = np.array([float(s.spatial_score) for s in successes], dtype=np.float64)
        sums = task_vals + visual_vals + spatial_vals
        spread = float(np.max(sums) - np.min(sums)) if len(sums) > 1 else 0.0
        if spread >= tiebreak_threshold:
            tiebreak_out = _collect_votes(
                active_cfg,
                phase="tiebreak",
                num_votes=extra_votes,
                vote_rows=vote_rows,
            )
            successes.extend(list(tiebreak_out["successes"]))
            api_failures += int(tiebreak_out["api_failures"])
            parse_failures += int(tiebreak_out["parse_failures"])
            errors.extend(list(tiebreak_out["errors"]))
            if not successes:
                return {
                    "score": None,
                    "error": "all_probe_votes_failed: " + "; ".join(errors),
                    "num_api_failures": int(api_failures),
                    "num_parse_failures": int(parse_failures),
                    "votes_effective": 0,
                    "score_spread": None,
                    "active_model_used": str(config.model),
                    "vote_rows": vote_rows,
                }

    task_vals = np.array([float(s.task_score) for s in successes], dtype=np.float64)
    visual_vals = np.array([float(s.visual_score) for s in successes], dtype=np.float64)
    spatial_vals = np.array([float(s.spatial_score) for s in successes], dtype=np.float64)
    sums = task_vals + visual_vals + spatial_vals
    idx_best = int(np.argmax(sums))
    reasoning = str(successes[idx_best].reasoning or "")

    tag_votes: Dict[str, int] = {}
    for rec in successes:
        for tag in rec.issue_tags:
            key = str(tag).strip().lower()
            if not key:
                continue
            tag_votes[key] = int(tag_votes.get(key, 0)) + 1
    min_votes = max(1, (len(successes) + 1) // 2)
    tags = sorted([tag for tag, count in tag_votes.items() if count >= min_votes])

    consensus_score = Stage1ProbeScore(
        task_score=float(np.median(task_vals)),
        visual_score=float(np.median(visual_vals)),
        spatial_score=float(np.median(spatial_vals)),
        issue_tags=tags,
        reasoning=reasoning,
        raw_response=json.dumps(
            {
                "mode": "consensus",
                "votes_configured": configured_votes,
                "votes_effective": len(successes),
                "errors": errors,
            },
            sort_keys=True,
        ),
        model_used=str(getattr(successes[idx_best], "model_used", "") or str(config.model)),
    )
    spread = float(np.max(sums) - np.min(sums)) if len(sums) > 1 else 0.0
    return {
        "score": consensus_score,
        "error": None,
        "num_api_failures": int(api_failures),
        "num_parse_failures": int(parse_failures),
        "votes_effective": int(len(successes)),
        "score_spread": round(spread, 6),
        "active_model_used": str(getattr(consensus_score, "model_used", "") or str(config.model)),
        "vote_rows": vote_rows,
    }


def _detect_duplicate_clip(
    *,
    video_path: Path,
    seen_fingerprints: List[Dict[str, object]],
    similarity_threshold: float,
) -> Dict[str, object]:
    fingerprint = _clip_fingerprint(video_path)
    sha256 = str(fingerprint.get("sha256", ""))
    max_similarity = None
    for prev in seen_fingerprints:
        if sha256 and sha256 == str(prev.get("sha256", "")):
            return {
                "is_duplicate": True,
                "reason": "sha256",
                "sha256": sha256,
                "max_similarity": 1.0,
                "samples": fingerprint.get("samples", []),
            }
        sim = _sample_similarity(
            prev_samples=list(prev.get("samples", []) or []),
            cur_samples=list(fingerprint.get("samples", []) or []),
        )
        if sim is None:
            continue
        if max_similarity is None or float(sim) > float(max_similarity):
            max_similarity = float(sim)
        if float(sim) >= float(similarity_threshold):
            return {
                "is_duplicate": True,
                "reason": "frame_similarity",
                "sha256": sha256,
                "max_similarity": float(sim),
                "samples": fingerprint.get("samples", []),
            }
    return {
        "is_duplicate": False,
        "reason": "",
        "sha256": sha256,
        "max_similarity": max_similarity,
        "samples": fingerprint.get("samples", []),
    }


def _clip_fingerprint(video_path: Path) -> Dict[str, object]:
    return {
        "sha256": _sha256_file(video_path),
        "samples": _sample_video_frames(video_path),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _sample_video_frames(video_path: Path, sample_count: int = 3) -> List[np.ndarray]:
    try:
        import cv2
    except Exception:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, max(0, total - 1), num=max(1, int(sample_count)), dtype=np.int64)
    frames: List[np.ndarray] = []
    for idx in indices.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        frames.append(small)
    cap.release()
    return frames


def _sample_similarity(
    *,
    prev_samples: List[np.ndarray],
    cur_samples: List[np.ndarray],
) -> Optional[float]:
    if not prev_samples or not cur_samples:
        return None
    count = min(len(prev_samples), len(cur_samples))
    vals: List[float] = []
    for i in range(count):
        a = prev_samples[i].astype(np.float32)
        b = cur_samples[i].astype(np.float32)
        diff = float(np.mean(np.abs(a - b)))
        vals.append(max(0.0, min(1.0, 1.0 - diff / 255.0)))
    if not vals:
        return None
    return float(np.mean(vals))


def _build_stage1_run_metadata(*, config: ValidationConfig) -> Dict[str, str]:
    return {
        "git_commit": _resolve_git_commit(),
        "config_hash": _stable_hash_payload(asdict(config)),
        "stage1_code_hash": _stage1_code_hash(),
        "active_model_used": str(config.eval_policy.vlm_judge.model),
    }


def _resolve_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _stable_hash_payload(payload: object) -> str:
    def _default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    text = json.dumps(payload, sort_keys=True, default=_default, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stage1_code_hash() -> str:
    files = [
        Path(__file__),
        Path(__file__).resolve().parents[1] / "rendering" / "stage1_active_perception.py",
        Path(__file__).resolve().parents[1] / "rendering" / "camera_paths.py",
    ]
    h = hashlib.sha256()
    for file_path in files:
        try:
            data = file_path.read_bytes()
            h.update(data)
        except Exception:
            h.update(str(file_path).encode("utf-8"))
    return h.hexdigest()
