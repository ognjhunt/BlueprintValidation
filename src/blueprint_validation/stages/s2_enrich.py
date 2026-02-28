"""Stage 2: Cosmos Transfer 2.5 enrichment of rendered clips."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..enrichment.cosmos_runner import enrich_clip
from ..enrichment.variant_specs import get_variants
from ..warmup import load_cached_variants
from .base import PipelineStage

logger = get_logger("stages.s2_enrich")


class EnrichStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s2_enrich"

    @property
    def description(self) -> str:
        return "Enrich rendered clips with Cosmos Transfer 2.5 visual variants"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        enrich_dir = work_dir / "enriched"
        enrich_dir.mkdir(parents=True, exist_ok=True)

        # Load render manifest
        render_manifest_path = _resolve_render_manifest(work_dir)
        if render_manifest_path is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Render/composite/polish manifest not found. Run Stage 1 first "
                    "(and Stage 1b/1c if enabled)."
                ),
            )

        render_manifest = read_json(render_manifest_path)

        # Check for warmup-cached variant prompts before calling Gemini
        cached_variants = load_cached_variants(work_dir)
        if cached_variants:
            logger.info("Using %d cached variant prompts from warmup", len(cached_variants))
            variants = cached_variants
        else:
            # Extract a sample frame for dynamic variant generation
            sample_frame_path = _extract_sample_frame(render_manifest, work_dir)
            variants = get_variants(
                custom_variants=config.enrich.variants or None,
                dynamic=config.enrich.dynamic_variants,
                dynamic_model=config.enrich.dynamic_variants_model,
                sample_frame_path=sample_frame_path,
                num_variants=config.enrich.num_variants_per_render,
                facility_description=facility.description,
            )

        # Limit variants to configured count
        variants = variants[: config.enrich.num_variants_per_render]

        manifest_entries: List[Dict] = []
        total_enriched = 0
        total_failed = 0

        for clip_entry in render_manifest["clips"]:
            video_path = Path(clip_entry["video_path"])
            depth_path = Path(clip_entry["depth_video_path"])
            clip_name = clip_entry["clip_name"]

            if not video_path.exists():
                logger.warning("Video not found: %s", video_path)
                continue

            outputs = enrich_clip(
                video_path=video_path,
                depth_path=depth_path if depth_path.exists() else None,
                variants=variants,
                output_dir=enrich_dir,
                clip_name=clip_name,
                config=config.enrich,
            )

            for out in outputs:
                manifest_entries.append({
                    "clip_name": clip_name,
                    "variant_name": out.variant_name,
                    "prompt": out.prompt,
                    "output_video_path": str(out.output_video_path),
                    "input_video_path": str(out.input_video_path),
                })
                total_enriched += 1

            expected = len(variants)
            actual = len(outputs)
            if actual < expected:
                total_failed += expected - actual

        # Write enriched manifest
        manifest_path = enrich_dir / "enriched_manifest.json"
        manifest = {
            "facility": facility.name,
            "num_clips": len(manifest_entries),
            "variants_per_clip": len(variants),
            "variant_names": [v.name for v in variants],
            "clips": manifest_entries,
        }
        write_json(manifest, manifest_path)

        return StageResult(
            stage_name=self.name,
            status="success" if total_enriched > 0 else "failed",
            elapsed_seconds=0,
            outputs={
                "enrich_dir": str(enrich_dir),
                "manifest_path": str(manifest_path),
                "num_enriched": total_enriched,
            },
            metrics={
                "num_enriched_clips": total_enriched,
                "num_failed": total_failed,
                "variants_used": [v.name for v in variants],
            },
        )


def _resolve_render_manifest(work_dir: Path) -> Path | None:
    for candidate in [
        work_dir / "splatsim" / "interaction_manifest.json",
        work_dir / "gaussian_augment" / "augmented_manifest.json",
        work_dir / "gemini_polish" / "polished_manifest.json",
        work_dir / "robot_composite" / "composited_manifest.json",
        work_dir / "renders" / "render_manifest.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def _extract_sample_frame(manifest: dict, work_dir: Path) -> Path | None:
    """Extract a single frame from the first clip for dynamic variant generation."""
    clips = manifest.get("clips", [])
    if not clips:
        return None

    video_path = Path(clips[0].get("video_path", ""))
    if not video_path.exists():
        return None

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        # Seek to ~25% through for a representative frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 4))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        sample_path = work_dir / "enriched" / "_sample_frame.png"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_path), frame)
        return sample_path
    except Exception:
        logger.debug("Failed to extract sample frame for dynamic variants", exc_info=True)
        return None
