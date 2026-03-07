"""Stage 1f: Ingest external interaction manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..common import StageResult, write_json
from ..config import FacilityConfig, ValidationConfig
from ..validation import ManifestValidationError, load_and_validate_manifest
from .base import PipelineStage


def _normalize_clip(
    clip: dict,
    *,
    source_name: str,
    source_stage: str,
    augmentation_type: str,
) -> dict:
    video_path = str(clip.get("video_path") or clip.get("output_video_path") or "").strip()
    normalized = dict(clip)
    normalized["video_path"] = video_path
    normalized.setdefault("depth_video_path", str(clip.get("depth_video_path", "")))
    normalized.setdefault("augmentation_type", augmentation_type)
    normalized.setdefault("source_name", source_name)
    normalized.setdefault("source_stage", source_stage)
    return normalized


def _normalize_external_clips(
    *,
    external_clips: List[dict],
    external_source_name: str,
) -> Tuple[List[dict], int, int]:
    merged: List[dict] = []
    seen_name_video: set[Tuple[str, str]] = set()
    used_clip_names: set[str] = set()
    renamed_count = 0
    duplicate_count = 0

    def _ingest(rows: List[dict], *, source_tag: str, stage_name: str, default_source_name: str) -> None:
        nonlocal renamed_count, duplicate_count
        for row in rows:
            normalized = _normalize_clip(
                row,
                source_name=str(row.get("source_name") or default_source_name).strip() or default_source_name,
                source_stage=stage_name,
                augmentation_type=str(row.get("augmentation_type") or source_tag).strip() or source_tag,
            )
            base_name = str(normalized.get("clip_name", "")).strip() or f"{source_tag}_clip_{len(merged):03d}"
            video_path = str(normalized.get("video_path", "")).strip()
            dedupe_key = (base_name, video_path)
            if dedupe_key in seen_name_video:
                duplicate_count += 1
                continue

            clip_name = base_name
            if clip_name in used_clip_names:
                suffix = 1
                while True:
                    candidate = f"{base_name}__{source_tag}_{suffix:02d}"
                    if candidate not in used_clip_names:
                        clip_name = candidate
                        renamed_count += 1
                        break
                    suffix += 1

            normalized["clip_name"] = clip_name
            seen_name_video.add((clip_name, video_path))
            used_clip_names.add(clip_name)
            merged.append(normalized)

    _ingest(
        external_clips,
        source_tag="external",
        stage_name="s1f_external_interaction_ingest",
        default_source_name=external_source_name,
    )
    return merged, renamed_count, duplicate_count


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
            external_manifest = load_and_validate_manifest(
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
        external_clips = list(external_manifest.get("clips", []))
        normalized_clips, renamed_count, duplicate_count = _normalize_external_clips(
            external_clips=external_clips,
            external_source_name=source_name,
        )

        normalized_manifest = {
            "source_name": source_name,
            "source_manifest": str(manifest_path),
            "source_manifests": [str(manifest_path)],
            "augmentation_type": "external_interaction",
            "num_source_clips": len(normalized_clips),
            "num_external_source_clips": len(external_clips),
            "num_duplicate_rows_dropped": duplicate_count,
            "num_clip_names_renamed": renamed_count,
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
                "source_manifests": normalized_manifest["source_manifests"],
                "source_name": source_name,
            },
            metrics={
                "num_clips": len(normalized_clips),
                "num_external_source_clips": len(external_clips),
                "num_duplicate_rows_dropped": duplicate_count,
                "num_clip_names_renamed": renamed_count,
                "source_name": source_name,
            },
        )
