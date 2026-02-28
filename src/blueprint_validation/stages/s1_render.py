"""Stage 1: Render Gaussian splat PLY to video clips via gsplat."""

from __future__ import annotations

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
    filter_and_fix_poses,
    generate_scene_aware_specs,
    load_obbs_from_task_targets,
)
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

            return detect_and_generate_specs(
                splat_means_np=splat_means_np,
                scene_center=scene_center,
                num_views=config.render.vlm_fallback_num_views,
                model=config.render.vlm_fallback_model,
                resolution=config.render.resolution,
            )
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

        scene_center = splat.center.cpu().numpy()
        splat_means_np = splat.means.cpu().numpy()
        resolution = config.render.resolution
        num_frames = config.render.num_frames
        camera_height = config.render.camera_height_m
        look_down_deg = config.render.camera_look_down_deg
        fps = config.render.fps

        # Scene-aware camera placement
        extra_specs, occupancy = self._build_scene_aware_specs(
            config, facility, splat_means_np, scene_center
        )

        all_path_specs = list(config.render.camera_paths) + extra_specs
        if extra_specs:
            logger.info(
                "Added %d scene-aware camera paths (total: %d)",
                len(extra_specs),
                len(all_path_specs),
            )

        # Generate camera paths and render clips
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
                })
                clip_index += 1

        # Write manifest
        manifest_path = render_dir / "render_manifest.json"
        manifest = {
            "facility": facility.name,
            "ply_path": str(facility.ply_path),
            "num_clips": len(manifest_entries),
            "scene_aware": config.render.scene_aware,
            "scene_aware_specs": len(extra_specs),
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
                "scene_aware_specs": len(extra_specs),
            },
        )


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
