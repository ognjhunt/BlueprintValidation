"""Stage 5: Visual fidelity metrics (PSNR/SSIM/LPIPS)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.metrics import compute_video_metrics
from .base import PipelineStage

logger = get_logger("stages.s5_visual_fidelity")


class VisualFidelityStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s5_visual_fidelity"

    @property
    def description(self) -> str:
        return "Compute visual fidelity metrics between rendered and enriched video"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        import cv2

        fidelity_dir = work_dir / "visual_fidelity"
        fidelity_dir.mkdir(parents=True, exist_ok=True)

        # Load manifests
        render_manifest_path = work_dir / "renders" / "render_manifest.json"
        enriched_manifest_path = work_dir / "enriched" / "enriched_manifest.json"

        if not render_manifest_path.exists() or not enriched_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Render or enriched manifest not found. Run Stages 1-2 first.",
            )

        render_manifest = read_json(render_manifest_path)
        enriched_manifest = read_json(enriched_manifest_path)

        # Build a map from clip_name to rendered video
        render_map: Dict[str, Path] = {}
        for clip in render_manifest["clips"]:
            render_map[clip["clip_name"]] = Path(clip["video_path"])

        all_metrics: List[Dict] = []

        for enriched_clip in enriched_manifest["clips"]:
            clip_name = enriched_clip["clip_name"]
            enriched_path = Path(enriched_clip["output_video_path"])
            render_path = render_map.get(clip_name)

            if not render_path or not render_path.exists() or not enriched_path.exists():
                continue

            # Extract frames from both videos
            render_frames = _extract_frames(render_path)
            enriched_frames = _extract_frames(enriched_path)

            if not render_frames or not enriched_frames:
                continue

            # Compute metrics
            video_metrics = compute_video_metrics(
                render_frames,
                enriched_frames,
                metrics=config.eval_visual.metrics,
                lpips_backbone=config.eval_visual.lpips_backbone,
            )

            all_metrics.append({
                "clip_name": clip_name,
                "variant": enriched_clip.get("variant_name", ""),
                **video_metrics.to_dict(),
            })

        # Aggregate
        if all_metrics:
            agg = {
                "num_comparisons": len(all_metrics),
            }
            for key in ["mean_psnr", "mean_ssim", "mean_lpips"]:
                vals = [m[key] for m in all_metrics if key in m and m[key] is not None]
                if vals:
                    agg[f"overall_{key}"] = round(float(np.mean(vals)), 4)
        else:
            agg = {"num_comparisons": 0}

        result_data = {"aggregate": agg, "per_clip": all_metrics}
        write_json(result_data, fidelity_dir / "visual_fidelity.json")

        return StageResult(
            stage_name=self.name,
            status="success" if all_metrics else "failed",
            elapsed_seconds=0,
            outputs={"fidelity_dir": str(fidelity_dir)},
            metrics=agg,
        )


def _extract_frames(video_path: Path, max_frames: int = 49) -> List[np.ndarray]:
    """Extract frames from a video file."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
