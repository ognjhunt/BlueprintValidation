"""Stage 1g: Ingest external action-labeled teleop manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult
from ..config import FacilityConfig, ValidationConfig
from ..teleop.contracts import TeleopManifestError, load_and_validate_teleop_manifest
from ..training.external_rollouts import (
    convert_teleop_sessions_to_rollout_rows,
    external_rollouts_enabled_for_policy,
    write_external_rollout_rows,
)
from .base import PipelineStage


class ExternalRolloutIngestStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1g_external_rollout_ingest"

    @property
    def description(self) -> str:
        return "Ingest external teleop sessions into action-labeled rollout rows"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility, previous_results
        ext_cfg = config.external_rollouts
        if not external_rollouts_enabled_for_policy(ext_cfg):
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="external_rollouts disabled or mode excludes policy ingestion",
            )

        manifest_path = ext_cfg.manifest_path
        if manifest_path is None:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="external_rollouts.enabled=true but manifest_path is not set; Stage 1g auto-skipped.",
            )
        if not manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"External rollout manifest not found: {manifest_path}",
            )

        try:
            payload = load_and_validate_teleop_manifest(manifest_path, require_existing_paths=True)
        except TeleopManifestError as exc:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Invalid external rollout manifest: {exc}",
            )

        rows = convert_teleop_sessions_to_rollout_rows(payload)
        stage_dir = work_dir / "external_rollouts"
        stage_dir.mkdir(parents=True, exist_ok=True)
        rows_path = stage_dir / "rollout_rows.json"
        write_external_rollout_rows(rows_path, rows)

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "manifest_path": str(manifest_path),
                "rollout_rows_path": str(rows_path),
                "source_name": str(payload.get("source_name", "teleop") or "teleop"),
            },
            metrics={
                "num_sessions": len(payload.get("sessions", [])),
                "num_rollout_rows": len(rows),
            },
        )
