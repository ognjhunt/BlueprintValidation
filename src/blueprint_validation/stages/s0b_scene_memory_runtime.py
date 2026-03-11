"""Stage S0b: choose the active scene-memory runtime adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, write_json
from ..config import FacilityConfig, ValidationConfig
from ..scene_memory_runtime import resolve_scene_memory_runtime_plan
from .base import PipelineStage


class SceneMemoryRuntimeStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s0b_scene_memory_runtime"

    @property
    def description(self) -> str:
        return "Resolve the preferred scene-memory runtime adapters for downstream world-model work"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del previous_results
        if not bool(config.scene_memory_runtime.enabled):
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="scene_memory_runtime.enabled=false",
            )
        if not facility.has_scene_memory_bundle:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="No scene_memory_bundle configured for this evaluation target.",
            )

        runtime_dir = work_dir / "scene_memory_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_plan = resolve_scene_memory_runtime_plan(config, facility)
        selection_path = runtime_dir / "runtime_selection.json"
        write_json(runtime_plan, selection_path)

        selected_backend = runtime_plan.get("selected_backend")
        available_backends = list(runtime_plan.get("available_backends", []) or [])
        status = "success" if selected_backend else "skipped"
        detail = (
            runtime_plan.get("selection_reason")
            or "Resolved scene-memory runtime selection."
        )
        return StageResult(
            stage_name=self.name,
            status=status,
            elapsed_seconds=0,
            detail=str(detail),
            outputs={
                "runtime_selection_path": str(selection_path),
                "scene_memory_runtime": runtime_plan,
                "selected_backend": selected_backend,
                "secondary_backend": runtime_plan.get("secondary_backend"),
                "fallback_backend": runtime_plan.get("fallback_backend"),
                "available_backends": available_backends,
                "skipped_watchlist_backends": list(
                    runtime_plan.get("skipped_watchlist_backends", []) or []
                ),
            },
            metrics={
                "num_available_backends": len(available_backends),
                "has_selected_backend": bool(selected_backend),
                "scene_memory_runtime_backend": selected_backend,
                "num_skipped_watchlist_backends": len(
                    list(runtime_plan.get("skipped_watchlist_backends", []) or [])
                ),
            },
        )
