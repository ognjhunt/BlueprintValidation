"""Stage 1: Render Gaussian splat PLY to video clips via gsplat."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, List, Optional

import numpy as np

from ..common import StageResult, get_logger, write_json
from ..config import CameraPathSpec, FacilityConfig, VLMJudgeConfig, ValidationConfig
from ..evaluation.camera_quality import evaluate_clip_quality
from ..evaluation.task_hints import tasks_from_task_hints
from ..evaluation.vlm_judge import Stage1ProbeScore, score_stage1_probe
from ..rendering.camera_paths import generate_path_from_spec, save_path_to_json
from ..rendering.camera_quality_planner import (
    plan_best_camera_spec,
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
from .base import PipelineStage

logger = get_logger("stages.s1_render")


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
                        "task_hints_path": (
                            str(facility.task_hints_path) if facility.task_hints_path is not None else None
                        )
                    },
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
                    outputs={"missing_tools": missing},
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

            # Scene-aware camera placement
            extra_specs, occupancy = self._build_scene_aware_specs(
                config, facility, splat_means_np, scene_center, scene_T if has_transform else None
            )
            extra_specs_count = len(extra_specs)

            base_specs = list(config.render.camera_paths)
            if has_transform:
                base_specs = transform_camera_path_specs(base_specs, scene_T)
            all_path_specs = base_specs + extra_specs
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
            clip_repeats = int(config.render.num_clips_per_path)
            # Task-scoped paths are already object-specific; render once to control cost.
            if str(getattr(path_spec, "source_tag", "")).strip().lower() == "task_scoped":
                clip_repeats = int(config.render.task_scoped_num_clips_per_path)
            clip_repeats = max(1, clip_repeats)
            for clip_num in range(clip_repeats):
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
                )
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
                if bool(config.render.stage1_quality_planner_enabled):
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
                path_context_for_scope = _path_context_from_spec(path_spec)
                if (
                    bool(config.render.stage1_active_perception_enabled)
                    and should_probe_clip(
                        scope=config.render.stage1_active_perception_scope,
                        path_type=str(path_spec.type),
                        path_context=path_context_for_scope,
                    )
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
                        start_offset=offset,
                        fps=fps,
                        scene_T=scene_T,
                        facility_description=facility_description,
                        probe_orientation_fix=probe_orientation_fix,
                        probe_scores_path=probe_scores_path,
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
            poses = filter_and_fix_poses(poses, occupancy, target, config.render.min_clearance_m)
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
        probe_orientation_fix: str = "none",
        probe_scores_path: Optional[Path] = None,
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

        if ranked_candidates:
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

        # max_loops is "correction loops"; total rounds include the initial probe round.
        total_rounds = max(1, int(budget.max_loops) + 1)
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
                            },
                        )
                        break

                    min_unique_positions = int(config.render.stage1_probe_min_unique_positions)
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
                        }
                        _append_probe_score_row(
                            probe_scores_path,
                            {
                                "clip_name": clip_name,
                                "clip_id": int(clip_name.split("_")[1]) if "_" in clip_name else -1,
                                "path_type": str(candidate_current_spec.type),
                                "loop": int(round_idx),
                                "candidate": int(cand_idx),
                                "status": "scoring_failed",
                                "error": str(consensus.get("error", "probe_consensus_failed")),
                                "score_spread": consensus.get("score_spread"),
                                "votes_effective": int(consensus.get("votes_effective", 0)),
                                "votes": list(consensus.get("vote_rows", []) or []),
                                "probe_codec": probe_meta.get("probe_codec"),
                                "probe_resolution": probe_meta.get("probe_resolution"),
                                "probe_decoded_frames": probe_meta.get("probe_decoded_frames"),
                            },
                        )
                    else:
                        probe_score = consensus["score"]
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
                            "issue_tags": list(probe_score.issue_tags),
                            "reasoning": str(probe_score.reasoning or ""),
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
                            list(probe_score.issue_tags),
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
                                "issue_tags": list(probe_score.issue_tags),
                                "reasoning": str(probe_score.reasoning or ""),
                                "passed": bool(passes),
                                "score_spread": consensus.get("score_spread"),
                                "votes_effective": int(consensus.get("votes_effective", 0)),
                                "votes": list(consensus.get("vote_rows", []) or []),
                                "active_model_used": str(
                                    consensus.get("active_model_used") or config.eval_policy.vlm_judge.model
                                ),
                                "probe_codec": probe_meta.get("probe_codec"),
                                "probe_resolution": probe_meta.get("probe_resolution"),
                                "probe_decoded_frames": probe_meta.get("probe_decoded_frames"),
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

            last_issue_tags = list(best_row.get("issue_tags", []))
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
    return context


def _target_focus_label(path_context: dict | None) -> str | None:
    if not isinstance(path_context, dict):
        return None
    for key in ("target_label", "target_instance_id", "target_category"):
        value = path_context.get(key)
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return None


def _build_expected_focus_text(*, path_type: str, path_context: dict | None) -> str:
    """Return a deterministic, human/VLM-readable description of clip intent."""
    path_key = str(path_type or "").strip().lower()
    role = ""
    if isinstance(path_context, dict):
        role = str(path_context.get("target_role", "")).strip().lower()
    target_label = _target_focus_label(path_context)

    if role == "targets":
        if target_label:
            return (
                f"Primary target focus: keep {target_label} centered and clearly visible "
                "for most of the clip."
            )
        return "Primary target focus: keep the task target centered and clearly visible."
    if role == "context":
        if target_label:
            return (
                f"Context focus: keep {target_label} visible alongside nearby objects and "
                "interaction affordances."
            )
        return (
            "Context focus: keep the task region visible alongside nearby objects and "
            "interaction affordances."
        )
    if role == "overview":
        return (
            "Overview focus: capture broad scene layout and navigation anchors while preserving "
            "task-relevant context."
        )
    if role == "fallback":
        return (
            "Fallback focus: capture clear, stable scene coverage when explicit task-target "
            "mapping is unavailable."
        )

    if path_key == "manipulation":
        if target_label:
            return (
                f"Manipulation focus: keep {target_label} and its interaction zone in frame with "
                "a stable close-range viewpoint."
            )
        return (
            "Manipulation focus: keep the task object and interaction zone in frame with a stable "
            "close-range viewpoint."
        )
    if path_key == "orbit":
        return "Orbit focus: provide stable global scene coverage for spatial orientation."
    if path_key == "sweep":
        return "Sweep focus: scan across the scene to expose spatial relationships and task regions."
    if path_key == "file":
        return "Path-file focus: follow the predefined camera path with stable, clear framing."
    return "General focus: produce clear, stable scene coverage useful for downstream evaluation."


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
) -> np.ndarray:
    """Sample deterministic XY offset with stricter defaults for manipulation clips."""
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
