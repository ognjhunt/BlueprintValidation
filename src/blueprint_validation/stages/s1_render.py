"""Stage 1: Render Gaussian splat PLY to video clips via gsplat."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..common import StageResult, get_logger, write_json
from ..config import CameraPathSpec, FacilityConfig, ValidationConfig
from ..rendering.camera_paths import generate_path_from_spec, save_path_to_json
from ..rendering.gsplat_renderer import render_video
from ..rendering.ply_loader import load_splat
from ..rendering.scene_geometry import (
    OccupancyGrid,
    auto_populate_manipulation_zones,
    build_occupancy_grid,
    compute_scene_transform,
    detect_up_axis,
    filter_and_fix_poses,
    generate_scene_aware_specs,
    is_identity_transform,
    load_obbs_from_task_targets,
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
                extra_specs = generate_scene_aware_specs(obbs, occupancy)
                facility.manipulation_zones = auto_populate_manipulation_zones(
                    facility.manipulation_zones, obbs
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
            logger.warning("VLM fallback detection failed; continuing with naive paths", exc_info=True)
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

        # Resolve auto up-axis: prefer warmup-detected value, else detect now
        if facility.up_axis.lower().strip() == "auto":
            if warmup_cache and "detected_up_axis" in warmup_cache:
                resolved = warmup_cache["detected_up_axis"]
                logger.info("Using warmup-detected up_axis='%s'", resolved)
            elif warmup_cache and "resolved_up_axis" in warmup_cache:
                resolved = warmup_cache["resolved_up_axis"]
                logger.info("Using warmup-resolved up_axis='%s'", resolved)
            else:
                splat_means_for_detect = splat.means.cpu().numpy()
                resolved = detect_up_axis(splat_means_for_detect)
            facility = replace(facility, up_axis=resolved)

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
            manifest_entries = self._render_from_cache(
                cached_clips, splat, render_dir, resolution, fps,
                scene_T if has_transform else None,
            )
        else:
            splat_means_np = splat.means.cpu().numpy()

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

            manifest_entries = self._generate_and_render(
                config, splat, all_path_specs, scene_center,
                occupancy, render_dir, num_frames, camera_height,
                look_down_deg, resolution, fps,
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
            },
            metrics={
                "num_clips": len(manifest_entries),
                "total_frames": sum(e["num_frames"] for e in manifest_entries),
                "resolution": list(resolution),
                "scene_aware_specs": extra_specs_count,
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
    ) -> List[Dict]:
        """Render clips using pre-computed camera paths from warmup cache."""
        from ..warmup import _deserialize_camera_poses

        manifest_entries: List[Dict] = []
        for clip_data in cached_clips:
            clip_name = clip_data["clip_name"]
            poses = _deserialize_camera_poses(clip_data["poses"])

            # Convert cameras from corrected frame back to original PLY frame
            if scene_T is not None:
                from ..rendering.camera_paths import CameraPose

                poses = [
                    CameraPose(
                        c2w=transform_c2w(p.c2w, scene_T),
                        fx=p.fx, fy=p.fy, cx=p.cx, cy=p.cy,
                        width=p.width, height=p.height,
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
                manifest_entries.append({
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
                })
        return manifest_entries

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
    ) -> List[Dict]:
        """Original path: generate camera paths from scratch, then render."""
        manifest_entries: List[Dict] = []
        clip_index = 0

        for path_spec in all_path_specs:
            for clip_num in range(config.render.num_clips_per_path):
                # Add random offset for variety between clips
                rng = np.random.default_rng(seed=clip_index * 42)
                offset = rng.uniform(-1.0, 1.0, size=3)

                poses = generate_path_from_spec(
                    spec=path_spec,
                    scene_center=scene_center,
                    num_frames=num_frames,
                    camera_height=camera_height,
                    look_down_deg=look_down_deg,
                    resolution=resolution,
                    start_offset=offset,
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

                # Convert cameras from corrected frame to original PLY frame
                if scene_T is not None:
                    from ..rendering.camera_paths import CameraPose

                    poses = [
                        CameraPose(
                            c2w=transform_c2w(p.c2w, scene_T),
                            fx=p.fx, fy=p.fy, cx=p.cx, cy=p.cy,
                            width=p.width, height=p.height,
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

                manifest_entries.append({
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
                })
                clip_index += 1

        return manifest_entries


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
        "height_override_m": (
            float(path_spec.height_override_m)
            if path_spec.height_override_m is not None
            else None
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


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
