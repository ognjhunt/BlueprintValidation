"""Stage 1d: Full RoboSplat-default Gaussian augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..augmentation.robosplat_engine import run_robosplat_augmentation
from ..common import StageResult, get_logger
from ..config import FacilityConfig, ValidationConfig
from .manifest_resolution import ManifestCandidate, ManifestSource, resolve_manifest_source
from .base import PipelineStage

logger = get_logger("stages.s1d_gaussian_augment")


class GaussianAugmentStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1d_gaussian_augment"

    @property
    def description(self) -> str:
        return "Full RoboSplat-default augmentation over rendered clips"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        if not config.robosplat.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="robosplat.enabled=false",
            )

        source = _resolve_source_manifest(work_dir, previous_results)
        if source is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Render/composite/polish manifest not found. Run Stage 1 first "
                    "(and Stage 1b/1c if enabled)."
                ),
            )

        stage_dir = work_dir / "gaussian_augment"
        stage_dir.mkdir(parents=True, exist_ok=True)
        result = run_robosplat_augmentation(
            config=config,
            facility=facility,
            work_dir=work_dir,
            stage_dir=stage_dir,
            source_manifest_path=source.source_manifest_path,
        )

        status = result.status
        return StageResult(
            stage_name=self.name,
            status=status,
            elapsed_seconds=0,
            outputs={
                "augment_dir": str(stage_dir),
                "manifest_path": str(result.manifest_path),
                **source.to_metadata(),
            },
            metrics={
                "num_source_clips": result.num_source_clips,
                "num_augmented_clips": result.num_augmented_clips,
                "num_total_clips": result.num_total_clips,
                "num_rejected_quality": result.num_rejected_quality,
                "backend_used": result.backend_used,
                "fallback_backend": result.fallback_backend,
                "object_source": result.object_source,
                "demo_source": result.demo_source,
                **source.to_metadata(),
            },
            detail=result.detail,
        )


def _resolve_source_manifest(
    work_dir: Path,
    previous_results: Dict[str, StageResult] | None = None,
) -> ManifestSource | None:
    return resolve_manifest_source(
        work_dir=work_dir,
        previous_results=previous_results or {},
        candidates=[
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
