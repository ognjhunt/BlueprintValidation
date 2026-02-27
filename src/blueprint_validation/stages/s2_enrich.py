"""Stage 2: Cosmos Transfer 2.5 enrichment of rendered clips."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..enrichment.cosmos_runner import enrich_clip
from ..enrichment.variant_specs import get_variants
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
        render_manifest_path = work_dir / "renders" / "render_manifest.json"
        if not render_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Render manifest not found at {render_manifest_path}. Run Stage 1 first.",
            )

        render_manifest = read_json(render_manifest_path)
        variants = get_variants(config.enrich.variants or None)

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
