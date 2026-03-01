"""Stage 1: Render Gaussian splat PLY to video clips via gsplat."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np

from ..common import StageResult, get_logger, write_json
from ..config import CameraPathSpec, FacilityConfig, ValidationConfig
from ..evaluation.task_hints import tasks_from_task_hints
from ..rendering.camera_paths import generate_path_from_spec, save_path_to_json
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
                    selected_obbs, scoped_stats = _select_task_scoped_obbs(
                        obbs=obbs,
                        tasks=task_pool,
                        max_specs=max(1, int(config.render.task_scoped_max_specs)),
                        context_per_target=max(
                            0, int(config.render.task_scoped_context_per_target)
                        ),
                        overview_specs=max(0, int(config.render.task_scoped_overview_specs)),
                        fallback_specs=max(1, int(config.render.task_scoped_fallback_specs)),
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
                extra_specs = generate_scene_aware_specs(selected_obbs, occupancy)
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
        render_dir = work_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)

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
        extra_specs_count = 0
        num_poses_roll_corrected = 0
        num_clips_with_roll_correction = 0

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
            ) = self._render_from_cache(
                cached_clips,
                splat,
                render_dir,
                resolution,
                fps,
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
            )

        # Write manifest
        manifest_path = render_dir / "render_manifest.json"
        manifest = {
            "facility": facility.name,
            "ply_path": str(facility.ply_path),
            "num_clips": len(manifest_entries),
            "scene_aware": config.render.scene_aware,
            "scene_aware_specs": extra_specs_count,
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
            },
        )

    def _render_from_cache(
        self,
        cached_clips: List[Dict],
        splat,
        render_dir: Path,
        resolution: tuple,
        fps: int,
        scene_T: Optional[np.ndarray] = None,
    ) -> tuple[List[Dict], int, int]:
        """Render clips using pre-computed camera paths from warmup cache."""
        from ..warmup import _deserialize_camera_poses

        manifest_entries: List[Dict] = []
        corrected_poses_total = 0
        corrected_clip_count = 0
        for clip_data in cached_clips:
            clip_name = clip_data["clip_name"]
            poses = _deserialize_camera_poses(clip_data["poses"])
            poses, corrected_count = correct_upside_down_camera_poses(poses)
            corrected_poses_total += corrected_count
            if corrected_count > 0:
                corrected_clip_count += 1

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
            manifest_entries.append(
                {
                    "clip_name": clip_name,
                    "path_type": clip_data["path_type"],
                    "clip_index": clip_data["clip_index"],
                    "num_frames": len(poses),
                    "resolution": list(resolution),
                    "fps": fps,
                    "video_path": str(output.video_path),
                    "depth_video_path": str(output.depth_video_path),
                    "camera_path": str(render_dir / f"{clip_name}_camera_path.json"),
                    "initial_camera": initial_camera,
                        "path_context": {"source": "warmup_cache"},
                    }
                )
        return manifest_entries, corrected_poses_total, corrected_clip_count

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
    ) -> tuple[List[Dict], int, int]:
        """Original path: generate camera paths from scratch, then render."""
        manifest_entries: List[Dict] = []
        clip_index = 0
        corrected_poses_total = 0
        corrected_clip_count = 0

        for path_spec in all_path_specs:
            clip_repeats = int(config.render.num_clips_per_path)
            # Task-scoped paths are already object-specific; render once to control cost.
            if str(getattr(path_spec, "source_tag", "")).strip().lower() == "task_scoped":
                clip_repeats = 1
            clip_repeats = max(1, clip_repeats)
            for clip_num in range(clip_repeats):
                # Add random offset for variety between clips
                rng = np.random.default_rng(seed=clip_index * 42)
                offset = _sample_start_offset(config, path_spec, rng)

                poses = generate_path_from_spec(
                    spec=path_spec,
                    scene_center=scene_center,
                    num_frames=num_frames,
                    camera_height=camera_height,
                    look_down_deg=look_down_deg,
                    resolution=resolution,
                    start_offset=offset,
                    manipulation_target_z_bias_m=config.render.manipulation_target_z_bias_m,
                )

                # Collision filter
                if occupancy is not None:
                    target = np.array(path_spec.approach_point or scene_center[:3])
                    poses = filter_and_fix_poses(
                        poses, occupancy, target, config.render.min_clearance_m
                    )
                    if not poses:
                        logger.warning(
                            "All poses rejected for %s clip %d — skipping",
                            path_spec.type,
                            clip_num,
                        )
                        clip_index += 1
                        continue
                poses, corrected_count = correct_upside_down_camera_poses(poses)
                corrected_poses_total += corrected_count
                if corrected_count > 0:
                    corrected_clip_count += 1

                # Convert cameras from corrected frame to original PLY frame
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

                clip_name = f"clip_{clip_index:03d}_{path_spec.type}"

                # Save camera path
                save_path_to_json(
                    poses,
                    render_dir / f"{clip_name}_camera_path.json",
                )

                # Render
                output = render_video(
                    splat=splat,
                    poses=poses,
                    output_dir=render_dir,
                    clip_name=clip_name,
                    fps=fps,
                )
                initial_camera = _camera_pose_metadata(poses[0]) if poses else None

                manifest_entries.append(
                    {
                        "clip_name": clip_name,
                        "path_type": path_spec.type,
                        "clip_index": clip_index,
                        "num_frames": len(poses),
                        "resolution": list(resolution),
                        "fps": fps,
                        "video_path": str(output.video_path),
                        "depth_video_path": str(output.depth_video_path),
                        "camera_path": str(render_dir / f"{clip_name}_camera_path.json"),
                        "initial_camera": initial_camera,
                        "path_context": _path_context_from_spec(path_spec),
                    }
                )
                clip_index += 1

        return manifest_entries, corrected_poses_total, corrected_clip_count


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
    return context


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
) -> tuple[List[OrientedBoundingBox], Dict[str, int]]:
    stats = {"targets": 0, "context": 0, "overview": 0, "fallback": 0}
    if not obbs or max_specs <= 0:
        return [], stats

    max_specs = max(1, int(max_specs))
    fallback_specs = max(1, int(fallback_specs))
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
        return selected_obbs, stats

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
    return selected_obbs, stats


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
) -> np.ndarray:
    """Sample deterministic XY offset with stricter defaults for manipulation clips."""
    span = (
        float(config.render.manipulation_random_xy_offset_m)
        if str(path_spec.type).strip().lower() == "manipulation"
        else float(config.render.non_manipulation_random_xy_offset_m)
    )
    offset = np.zeros(3, dtype=np.float64)
    if span <= 0.0:
        return offset
    offset[:2] = rng.uniform(-span, span, size=2)
    return offset


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
