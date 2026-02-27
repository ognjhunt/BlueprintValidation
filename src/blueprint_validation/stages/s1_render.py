"""Stage 1: Render Gaussian splat PLY to video clips via gsplat."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, write_json
from ..config import FacilityConfig, ValidationConfig
from ..rendering.camera_paths import generate_path_from_spec, save_path_to_json
from ..rendering.gsplat_renderer import render_video
from ..rendering.ply_loader import load_splat
from .base import PipelineStage

logger = get_logger("stages.s1_render")


class RenderStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1_render"

    @property
    def description(self) -> str:
        return "Render PLY Gaussian splat to video clips at robot-height perspectives"

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
        resolution = config.render.resolution
        num_frames = config.render.num_frames
        camera_height = config.render.camera_height_m
        look_down_deg = config.render.camera_look_down_deg
        fps = config.render.fps

        # Generate camera paths and render clips
        manifest_entries: List[Dict] = []
        clip_index = 0

        for path_spec in config.render.camera_paths:
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
            },
        )


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
