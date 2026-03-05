"""Stage 1f: Ingest external interaction manifests (e.g., PolaRiS outputs)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, write_json
from ..config import FacilityConfig, ValidationConfig
from ..validation import ManifestValidationError, load_and_validate_manifest
from .base import PipelineStage


class ExternalInteractionIngestStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1f_external_interaction_ingest"

    @property
    def description(self) -> str:
        return "Ingest external interaction manifest into stage-1 source schema"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility, previous_results

        ext_cfg = config.external_interaction
        if not bool(ext_cfg.enabled):
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="external_interaction.enabled=false",
            )

        manifest_path = ext_cfg.manifest_path
        if manifest_path is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "external_interaction.enabled=true but manifest_path is not set. "
                    "Set external_interaction.manifest_path to a valid stage1_source manifest."
                ),
            )

        if not manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"External interaction manifest not found: {manifest_path}",
            )

        try:
            manifest = load_and_validate_manifest(
                manifest_path,
                manifest_type="stage1_source",
                require_existing_paths=True,
            )
        except ManifestValidationError as exc:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Invalid external interaction manifest: {exc}",
            )

        stage_dir = work_dir / "external_interaction"
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_manifest = stage_dir / "interaction_manifest.json"

        source_name = str(ext_cfg.source_name or "external").strip() or "external"
        normalized_clips = []
        for clip in manifest.get("clips", []):
            video_path = str(clip.get("video_path") or clip.get("output_video_path") or "").strip()
            normalized = dict(clip)
            normalized["video_path"] = video_path
            normalized.setdefault("depth_video_path", str(clip.get("depth_video_path", "")))
            normalized.setdefault("augmentation_type", "external_interaction")
            normalized.setdefault("source_name", source_name)
            normalized_clips.append(normalized)

        normalized_manifest = {
            "source_name": source_name,
            "source_manifest": str(manifest_path),
            "augmentation_type": "external_interaction",
            "num_source_clips": len(normalized_clips),
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "clips": normalized_clips,
        }
        write_json(normalized_manifest, output_manifest)

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "manifest_path": str(output_manifest),
                "source_manifest": str(manifest_path),
                "source_name": source_name,
            },
            metrics={
                "num_clips": len(normalized_clips),
                "source_name": source_name,
            },
        )
