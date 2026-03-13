"""Compatibility pipeline orchestrator for older build-oriented workflows."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .common import StageResult, get_logger, write_json
from .config import ValidationConfig
from .stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage
from .stages.s0a_scene_package import ScenePackageStage
from .stages.s0b_scene_memory_runtime import SceneMemoryRuntimeStage
from .stages.s1_isaac_render import IsaacRenderStage
from .stages.s1_render import RenderStage
from .stages.s1b_robot_composite import RobotCompositeStage
from .stages.s1c_gemini_polish import GeminiPolishStage
from .stages.s1d_gaussian_augment import GaussianAugmentStage
from .stages.s1f_external_interaction_ingest import ExternalInteractionIngestStage
from .stages.s1g_external_rollout_ingest import ExternalRolloutIngestStage

logger = get_logger("pipeline")


class ValidationPipeline:
    """Runs the legacy compatibility pipeline for older build-oriented workflows."""

    def __init__(self, config: ValidationConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _run_post_stage_sync(self, *, stage_key: str, result: StageResult) -> None:
        cmd = str(os.environ.get("BLUEPRINT_POST_STAGE_SYNC_CMD", "") or "").strip()
        if not cmd:
            return
        env = os.environ.copy()
        env["BLUEPRINT_SYNC_STAGE_KEY"] = stage_key
        env["BLUEPRINT_SYNC_STAGE_STATUS"] = result.status
        env["BLUEPRINT_SYNC_STAGE_NAME"] = result.stage_name
        subprocess.run(shlex.split(cmd), check=False, shell=False, env=env)

    def _maybe_trigger_auto_shutdown(self, reason: str) -> None:
        if not bool(getattr(self.config.cloud, "auto_shutdown", False)):
            return
        cmd = str(os.environ.get("BLUEPRINT_AUTO_SHUTDOWN_CMD", "") or "").strip()
        if not cmd:
            return
        env = os.environ.copy()
        env["BLUEPRINT_AUTO_SHUTDOWN_REASON"] = str(reason or "").strip()
        subprocess.run(shlex.split(cmd), check=False, shell=False, env=env)

    def run_all(
        self,
        fail_fast: bool = True,
        resume_from_results: bool = False,
    ) -> Dict[str, StageResult]:
        all_results: Dict[str, StageResult] = {}
        failed_stage_keys: list[str] = []
        stage_provenance: Dict[str, Dict[str, str | None]] = {}
        dry_run = os.environ.get("BLUEPRINT_DRY_RUN", "0") == "1"
        pipeline_start = time.monotonic()
        run_started_at = datetime.now(timezone.utc).isoformat()
        run_mode = "resume" if resume_from_results else "fresh"

        per_facility_stages = [
            TaskHintsBootstrapStage(),
            SceneMemoryRuntimeStage(),
            ScenePackageStage(),
            IsaacRenderStage(),
            RenderStage(),
            RobotCompositeStage(),
            GeminiPolishStage(),
            GaussianAugmentStage(),
            ExternalInteractionIngestStage(),
            ExternalRolloutIngestStage(),
        ]

        for facility_id, facility in self.config.facilities.items():
            facility_dir = self.work_dir / facility_id
            facility_dir.mkdir(parents=True, exist_ok=True)
            facility_results: Dict[str, StageResult] = {}
            for stage in per_facility_stages:
                stage_key = f"{facility_id}/{stage.name}"
                result_path = facility_dir / f"{stage.name}_result.json"
                if resume_from_results and result_path.exists():
                    try:
                        payload = StageResult.from_dict(
                            json.loads(result_path.read_text(encoding="utf-8"))
                        )
                    except Exception as exc:
                        payload = StageResult(
                            stage_name=stage.name,
                            status="failed",
                            elapsed_seconds=0.0,
                            detail=f"Corrupt resume artifact: {exc}",
                        )
                        facility_results[stage.name] = payload
                        all_results[stage_key] = payload
                        stage_provenance[stage_key] = {
                            "source": "resume_corrupt",
                            "result_path": str(result_path),
                        }
                        failed_stage_keys.append(stage_key)
                        if fail_fast:
                            break
                        continue
                    if payload.status == "success":
                        facility_results[stage.name] = payload
                        all_results[stage_key] = payload
                        stage_provenance[stage_key] = {
                            "source": "resumed",
                            "result_path": str(result_path),
                        }
                        self._run_post_stage_sync(stage_key=stage_key, result=payload)
                        continue

                result = stage.execute(
                    config=self.config,
                    facility=facility,
                    work_dir=facility_dir,
                    previous_results=facility_results,
                )
                result.save(result_path)
                facility_results[stage.name] = result
                all_results[stage_key] = result
                stage_provenance[stage_key] = {"source": "executed", "result_path": str(result_path)}
                self._run_post_stage_sync(stage_key=stage_key, result=result)
                if result.status == "failed":
                    failed_stage_keys.append(stage_key)
                    if fail_fast:
                        break
            if fail_fast and failed_stage_keys:
                break

        summary = {
            "num_facilities": len(self.config.facilities),
            "facility_ids": list(self.config.facilities.keys()),
            "overall_status": "failed" if failed_stage_keys else "success",
            "fail_fast": fail_fast,
            "resume_from_results": resume_from_results,
            "run_mode": run_mode,
            "run_started_at": run_started_at,
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - pipeline_start, 3),
            "failed_stage_keys": failed_stage_keys,
            "dry_run": dry_run,
            "stages": {
                stage_key: {
                    **result.to_dict(),
                    "provenance": stage_provenance.get(stage_key, {"source": "executed", "result_path": None}),
                }
                for stage_key, result in all_results.items()
            },
        }
        write_json(summary, self.work_dir / "pipeline_summary.json")
        if failed_stage_keys:
            self._maybe_trigger_auto_shutdown("pipeline_failed")
        logger.info("Pipeline complete. Summary at %s", self.work_dir / "pipeline_summary.json")
        return all_results
