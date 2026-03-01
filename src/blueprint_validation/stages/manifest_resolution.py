"""Helpers for resolving upstream stage manifests with lineage metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..common import StageResult, get_logger

logger = get_logger("stages.manifest_resolution")


@dataclass(frozen=True)
class ManifestCandidate:
    """Possible upstream stage manifest source."""

    stage_name: str
    manifest_relpath: Path


@dataclass(frozen=True)
class ManifestSource:
    """Resolved manifest source and lineage metadata."""

    source_stage: str
    source_manifest_path: Path
    source_mode: str  # "previous_results" | "filesystem_fallback"

    def to_metadata(self) -> dict:
        return {
            "source_stage": self.source_stage,
            "source_manifest_path": str(self.source_manifest_path),
            "source_mode": self.source_mode,
        }


def resolve_manifest_source(
    *,
    work_dir: Path,
    previous_results: Dict[str, StageResult],
    candidates: List[ManifestCandidate],
) -> Optional[ManifestSource]:
    """Resolve a source manifest from upstream stages.

    Resolution order:
    1) Prefer successful entries from ``previous_results``.
    2) Only if ``previous_results`` is empty (standalone stage runs), fall back to filesystem probing.
    """
    for candidate in candidates:
        stage_result = previous_results.get(candidate.stage_name)
        if stage_result is None or stage_result.status != "success":
            continue
        manifest_raw = stage_result.outputs.get("manifest_path")
        if manifest_raw:
            manifest_path = Path(str(manifest_raw))
        else:
            manifest_path = work_dir / candidate.manifest_relpath
        if manifest_path.exists():
            return ManifestSource(
                source_stage=candidate.stage_name,
                source_manifest_path=manifest_path,
                source_mode="previous_results",
            )
        logger.warning(
            "Stage %s reported success but manifest missing: %s",
            candidate.stage_name,
            manifest_path,
        )

    if previous_results:
        return None

    for candidate in candidates:
        manifest_path = work_dir / candidate.manifest_relpath
        if manifest_path.exists():
            return ManifestSource(
                source_stage=candidate.stage_name,
                source_manifest_path=manifest_path,
                source_mode="filesystem_fallback",
            )
    return None

