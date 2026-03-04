"""Stage 5: Visual fidelity metrics (PSNR/SSIM/LPIPS)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.metrics import compute_video_metrics
from .manifest_resolution import ManifestCandidate, ManifestSource, resolve_manifest_source
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
        del facility
        fidelity_dir = work_dir / "visual_fidelity"
        fidelity_dir.mkdir(parents=True, exist_ok=True)

        source = _resolve_source_manifest_source(work_dir, previous_results)
        enriched_manifest_path = work_dir / "enriched" / "enriched_manifest.json"
        if source is None or not enriched_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Source Stage-1 lineage manifest or Stage-2 enriched manifest not found. "
                    "Run Stages 1-2 first."
                ),
            )

        source_manifest = read_json(source.source_manifest_path)
        enriched_manifest = read_json(enriched_manifest_path)
        source_map = _build_source_video_map(source_manifest)

        all_metrics: List[Dict] = []
        skipped_missing_enriched = 0
        skipped_missing_reference = 0
        skipped_empty_frames = 0
        num_reference_from_input = 0
        num_reference_from_manifest = 0

        for enriched_clip in enriched_manifest.get("clips", []):
            clip_name = str(enriched_clip.get("clip_name", "")).strip()
            enriched_raw = str(enriched_clip.get("output_video_path", "")).strip()
            if not enriched_raw:
                skipped_missing_enriched += 1
                continue
            enriched_path = Path(enriched_raw)
            if not enriched_path.exists():
                skipped_missing_enriched += 1
                continue

            reference_path, reference_mode = _resolve_reference_path(enriched_clip, source_map)
            if reference_path is None or not reference_path.exists():
                skipped_missing_reference += 1
                continue

            if reference_mode == "input_video_path":
                num_reference_from_input += 1
            elif reference_mode == "source_manifest":
                num_reference_from_manifest += 1

            # Extract frames from both videos
            reference_frames = _extract_frames(reference_path)
            enriched_frames = _extract_frames(enriched_path)
            if not reference_frames or not enriched_frames:
                skipped_empty_frames += 1
                continue

            # Compute metrics
            video_metrics = compute_video_metrics(
                reference_frames,
                enriched_frames,
                metrics=config.eval_visual.metrics,
                lpips_backbone=config.eval_visual.lpips_backbone,
            )

            all_metrics.append(
                {
                    "clip_name": clip_name,
                    "variant": enriched_clip.get("variant_name", ""),
                    "reference_video_path": str(reference_path),
                    "reference_resolution_mode": reference_mode,
                    **video_metrics.to_dict(),
                }
            )

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
        agg.update(
            {
                "num_missing_enriched_outputs": skipped_missing_enriched,
                "num_missing_reference_videos": skipped_missing_reference,
                "num_empty_frame_pairs": skipped_empty_frames,
                "num_reference_from_input_video_path": num_reference_from_input,
                "num_reference_from_source_manifest": num_reference_from_manifest,
                **source.to_metadata(),
            }
        )

        result_data = {
            "aggregate": agg,
            "per_clip": all_metrics,
            "lineage_source": source.to_metadata(),
        }
        write_json(result_data, fidelity_dir / "visual_fidelity.json")

        return StageResult(
            stage_name=self.name,
            status="success" if all_metrics else "failed",
            elapsed_seconds=0,
            detail=(
                ""
                if all_metrics
                else (
                    "No valid visual-fidelity comparisons could be computed "
                    "(missing reference/enriched videos or empty decoded frames)."
                )
            ),
            outputs={"fidelity_dir": str(fidelity_dir), **source.to_metadata()},
            metrics=agg,
        )


def _resolve_source_manifest_source(
    work_dir: Path,
    previous_results: Dict[str, StageResult] | None = None,
) -> ManifestSource | None:
    return resolve_manifest_source(
        work_dir=work_dir,
        previous_results=previous_results or {},
        candidates=[
            ManifestCandidate(
                stage_name="s1e_splatsim_interaction",
                manifest_relpath=Path("splatsim/interaction_manifest.json"),
            ),
            ManifestCandidate(
                stage_name="s1d_gaussian_augment",
                manifest_relpath=Path("gaussian_augment/augmented_manifest.json"),
            ),
            ManifestCandidate(
                stage_name="s1c_gemini_polish",
                manifest_relpath=Path("gemini_polish/polished_manifest.json"),
            ),
            ManifestCandidate(
                stage_name="s1b_robot_composite",
                manifest_relpath=Path("robot_composite/composited_manifest.json"),
            ),
            ManifestCandidate(
                stage_name="s1_render",
                manifest_relpath=Path("renders/render_manifest.json"),
            ),
        ],
    )


def _build_source_video_map(source_manifest: dict) -> Dict[str, Path]:
    source_map: Dict[str, Path] = {}
    for clip in source_manifest.get("clips", []):
        clip_name = str(clip.get("clip_name", "")).strip()
        if not clip_name or clip_name in source_map:
            continue
        source_video_raw = str(clip.get("video_path") or clip.get("output_video_path") or "").strip()
        if not source_video_raw:
            continue
        source_map[clip_name] = Path(source_video_raw)
    return source_map


def _resolve_reference_path(
    enriched_clip: dict,
    source_map: Dict[str, Path],
) -> Tuple[Path | None, str | None]:
    input_video_raw = str(enriched_clip.get("input_video_path", "")).strip()
    if input_video_raw:
        input_path = Path(input_video_raw)
        if input_path.exists():
            return input_path, "input_video_path"

    clip_name = str(enriched_clip.get("clip_name", "")).strip()
    fallback = source_map.get(clip_name)
    if fallback is not None and fallback.exists():
        return fallback, "source_manifest"

    return None, None


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
