"""Stage 1e: Minimal SplatSim-style interaction clip generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..synthetic.splatsim_pybullet_backend import run_splatsim_pybullet_backend
from .base import PipelineStage


class SplatSimInteractionStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1e_splatsim_interaction"

    @property
    def description(self) -> str:
        return "Generate minimal physics-validated interaction clips with PyBullet"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del previous_results

        if not config.splatsim.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="splatsim.enabled=false",
            )

        source_manifest = _resolve_source_manifest(work_dir)
        if source_manifest is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No upstream manifest found (S1/S1b/S1c/S1d).",
            )

        stage_dir = work_dir / "splatsim"
        stage_dir.mkdir(parents=True, exist_ok=True)

        backend_result = run_splatsim_pybullet_backend(
            config=config,
            facility=facility,
            stage_dir=stage_dir,
            source_manifest_path=source_manifest,
        )

        if backend_result.get("status") == "success":
            manifest_path = Path(str(backend_result.get("manifest_path", "")))
            if not manifest_path.exists():
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail="SplatSim backend reported success but manifest is missing.",
                )
            return StageResult(
                stage_name=self.name,
                status="success",
                elapsed_seconds=0,
                outputs={
                    "splatsim_dir": str(stage_dir),
                    "manifest_path": str(manifest_path),
                },
                metrics={
                    "mode": config.splatsim.mode,
                    "num_source_clips": int(backend_result.get("num_source_clips", 0)),
                    "num_generated_clips": int(backend_result.get("num_generated_clips", 0)),
                    "num_successful_rollouts": int(
                        backend_result.get("num_successful_rollouts", 0)
                    ),
                    "fallback_used": False,
                },
            )

        if config.splatsim.mode == "hybrid" and config.splatsim.fallback_to_prior_manifest:
            fallback_manifest = stage_dir / "interaction_manifest.json"
            source_data = read_json(source_manifest)
            clips = list(source_data.get("clips", []))
            write_json(
                {
                    "facility": facility.name,
                    "source_manifest": str(source_manifest),
                    "augmentation_type": "splatsim_fallback",
                    "backend_used": "none",
                    "fallback_used": True,
                    "num_source_clips": len(clips),
                    "num_generated_clips": 0,
                    "num_successful_rollouts": 0,
                    "clips": clips,
                },
                fallback_manifest,
            )
            return StageResult(
                stage_name=self.name,
                status="success",
                elapsed_seconds=0,
                outputs={
                    "splatsim_dir": str(stage_dir),
                    "manifest_path": str(fallback_manifest),
                },
                metrics={
                    "mode": config.splatsim.mode,
                    "num_source_clips": len(clips),
                    "num_generated_clips": 0,
                    "num_successful_rollouts": 0,
                    "fallback_used": True,
                },
                detail=str(backend_result.get("reason", "fallback")),
            )

        return StageResult(
            stage_name=self.name,
            status="failed",
            elapsed_seconds=0,
            detail=str(backend_result.get("reason", "splatsim_failed")),
            metrics={
                "mode": config.splatsim.mode,
                "fallback_used": False,
            },
        )


def _resolve_source_manifest(work_dir: Path) -> Path | None:
    for candidate in [
        work_dir / "gaussian_augment" / "augmented_manifest.json",
        work_dir / "gemini_polish" / "polished_manifest.json",
        work_dir / "robot_composite" / "composited_manifest.json",
        work_dir / "renders" / "render_manifest.json",
    ]:
        if candidate.exists():
            return candidate
    return None
