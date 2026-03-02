"""Full pipeline orchestrator — chains all stages sequentially."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .common import StageResult, get_logger, read_json, write_json
from .config import ValidationConfig
from .stages.s0_task_hints_bootstrap import TaskHintsBootstrapStage
from .stages.s1_render import RenderStage
from .stages.s1b_robot_composite import RobotCompositeStage
from .stages.s1c_gemini_polish import GeminiPolishStage
from .stages.s1d_gaussian_augment import GaussianAugmentStage
from .stages.s1e_splatsim_interaction import SplatSimInteractionStage
from .stages.s2_enrich import EnrichStage
from .stages.s3_finetune import FinetuneStage
from .stages.s3b_policy_finetune import PolicyFinetuneStage
from .stages.s3c_policy_rl_loop import PolicyRLLoopStage
from .stages.s3d_wm_refresh_loop import WorldModelRefreshLoopStage
from .stages.s4_policy_eval import PolicyEvalStage
from .stages.s4a_rlds_export import RLDSExportStage
from .stages.s4b_rollout_dataset import RolloutDatasetStage
from .stages.s4c_policy_pair_train import PolicyPairTrainStage
from .stages.s4d_policy_pair_eval import PolicyPairEvalStage
from .stages.s4e_trained_eval import TrainedPolicyEvalStage
from .stages.s5_visual_fidelity import VisualFidelityStage
from .stages.s6_spatial_accuracy import SpatialAccuracyStage
from .stages.s7_cross_site import CrossSiteStage

logger = get_logger("pipeline")

_WM_ONLY_DEFERRED_STAGES = {
    "s3b_policy_finetune",
    "s3c_policy_rl_loop",
    "s4a_rlds_export",
    "s4b_rollout_dataset",
    "s4c_policy_pair_train",
    "s4d_policy_pair_eval",
    "s4e_trained_eval",
}

_ACTION_BOOST_REQUIRED_STAGES = {
    "s4_policy_eval",
    "s4a_rlds_export",
    "s3b_policy_finetune",
    "s3c_policy_rl_loop",
    "s4e_trained_eval",
}


class ValidationPipeline:
    """Orchestrates the full validation pipeline across all facilities."""

    def __init__(self, config: ValidationConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run_all(
        self,
        fail_fast: bool = True,
        resume_from_results: bool = False,
    ) -> Dict[str, StageResult]:
        """Run all stages for all facilities, then cross-site test."""
        all_results: Dict[str, StageResult] = {}
        stage_provenance: Dict[str, Dict[str, str | None]] = {}
        failed_stage_keys: list[str] = []
        aborted_early = False
        pipeline_start = time.monotonic()
        run_started_at = datetime.now(timezone.utc).isoformat()
        run_mode = "resume" if resume_from_results else "fresh"
        dry_run = os.environ.get("BLUEPRINT_DRY_RUN", "0") == "1"
        action_boost = self._apply_action_boost_runtime_overrides()
        wm_only_scope = (
            (getattr(self.config.eval_policy, "headline_scope", "wm_only") or "wm_only")
            .strip()
            .lower()
            == "wm_only"
        )

        strict_fresh_guard = (
            not resume_from_results
            and bool(getattr(self.config.eval_policy.reliability, "enforce_stage_success", False))
        )
        if (
            resume_from_results
            and bool(getattr(self.config.eval_policy.reliability, "enforce_stage_success", False))
        ):
            detail = (
                "Strict reliability mode forbids resume runs for canonical metrics. "
                "Re-run without --resume in a fresh work directory."
            )
            key = "pipeline/run_mode_guard"
            all_results[key] = StageResult(
                stage_name="run_mode_guard",
                status="failed",
                elapsed_seconds=0,
                detail=detail,
            )
            stage_provenance[key] = {"source": "executed", "result_path": None}
            failed_stage_keys.append(key)
            aborted_early = True
        if strict_fresh_guard:
            stale_result_paths: list[str] = []
            for fid in self.config.facilities:
                fac_dir = self.work_dir / fid
                if not fac_dir.exists():
                    continue
                stale_result_paths.extend(
                    str(path) for path in sorted(fac_dir.glob("*_result.json"))
                )
            if stale_result_paths:
                detail = (
                    "Fresh strict mode requires an empty facility result set when --resume is disabled. "
                    f"Found {len(stale_result_paths)} pre-existing *_result.json files."
                )
                key = "pipeline/fresh_workdir_guard"
                all_results[key] = StageResult(
                    stage_name="fresh_workdir_guard",
                    status="failed",
                    elapsed_seconds=0,
                    detail=detail,
                    outputs={"stale_result_paths": stale_result_paths},
                )
                stage_provenance[key] = {"source": "executed", "result_path": None}
                failed_stage_keys.append(key)
                aborted_early = True

        if (
            not aborted_early
            and not dry_run
            and self.config.cloud.max_cost_usd > 0
            and self._hourly_rate_usd() is None
        ):
            detail = (
                "cloud.max_cost_usd is set but BLUEPRINT_GPU_HOURLY_RATE_USD is unset; "
                "refusing to run without enforceable budget guard."
            )
            logger.error(detail)
            key = "pipeline/cloud_budget_guard"
            failed_stage_keys.append(key)
            all_results[key] = StageResult(
                stage_name="cloud_budget_guard",
                status="failed",
                elapsed_seconds=0,
                detail=detail,
            )
            stage_provenance[key] = {"source": "executed", "result_path": None}
            self._maybe_trigger_auto_shutdown(detail)
            aborted_early = True

        # Per-facility stages (1-6)
        per_facility_stages = [
            TaskHintsBootstrapStage(),  # S0: bootstrap synthetic task hints if missing
            RenderStage(),  # S1: splat -> video clips
            RobotCompositeStage(),  # S1b: URDF robot arm composite
            GeminiPolishStage(),  # S1c: optional Gemini photorealism polish
            GaussianAugmentStage(),  # S1d: Full RoboSplat-default augmentation
            SplatSimInteractionStage(),  # S1e: Optional PyBullet interaction augmentation
            EnrichStage(),  # S2: Cosmos Transfer variants
            FinetuneStage(),  # S3: DreamDojo LoRA fine-tune
            PolicyEvalStage(),  # S4: frozen policy eval (baseline + adapted)
            WorldModelRefreshLoopStage(),  # S3d: WM-only near-miss/success world refresh loop
            RLDSExportStage(),  # S4a: export adapted rollouts -> RLDS TFRecords
            PolicyFinetuneStage(),  # S3b: OpenVLA-OFT fine-tune on pipeline-generated data
            PolicyRLLoopStage(),  # S3c: iterative RL loop + world-model refresh
            TrainedPolicyEvalStage(),  # S4e: evaluate trained vs frozen baselines
            RolloutDatasetStage(),  # S4b: export paired rollouts -> JSONL datasets
            PolicyPairTrainStage(),  # S4c: train policy_base + policy_site
            PolicyPairEvalStage(),  # S4d: heldout paired evaluation
            VisualFidelityStage(),  # S5: PSNR/SSIM/LPIPS metrics
            SpatialAccuracyStage(),  # S6: VLM spatial scoring
        ]

        for fid, fconfig in self.config.facilities.items():
            if aborted_early:
                break
            logger.info("=== Processing facility: %s ===", fid)
            fac_dir = self.work_dir / fid
            fac_dir.mkdir(parents=True, exist_ok=True)

            facility_results: Dict[str, StageResult] = {}

            for stage in per_facility_stages:
                stage_key = f"{fid}/{stage.name}"
                result_path = fac_dir / f"{stage.name}_result.json"
                if wm_only_scope and stage.name in _WM_ONLY_DEFERRED_STAGES:
                    result = StageResult(
                        stage_name=stage.name,
                        status="skipped",
                        elapsed_seconds=0,
                        detail=(
                            "Skipped by policy: eval_policy.headline_scope=wm_only "
                            "(OpenVLA stages deferred)."
                        ),
                    )
                    result = self._enforce_action_boost_required_stage_result(
                        result=result,
                        stage_name=stage.name,
                        action_boost=action_boost,
                    )
                    result.save(result_path)
                    facility_results[stage.name] = result
                    all_results[stage_key] = result
                    stage_provenance[stage_key] = {
                        "source": "executed",
                        "result_path": str(result_path),
                    }
                    if result.status == "failed":
                        failed_stage_keys.append(stage_key)
                        if fail_fast:
                            aborted_early = True
                            break
                    continue

                if resume_from_results:
                    existing = self._load_existing_stage_result(result_path)
                    if existing is not None and existing.status in {"success", "skipped"}:
                        logger.info(
                            "Resume mode: reusing %s result from %s (status=%s)",
                            stage_key,
                            result_path,
                            existing.status,
                        )
                        existing = self._enforce_action_boost_required_stage_result(
                            result=existing,
                            stage_name=stage.name,
                            action_boost=action_boost,
                        )
                        facility_results[stage.name] = existing
                        all_results[stage_key] = existing
                        stage_provenance[stage_key] = {
                            "source": "resumed",
                            "result_path": str(result_path),
                        }
                        if existing.status == "failed":
                            failed_stage_keys.append(stage_key)
                            if fail_fast:
                                aborted_early = True
                                break
                        continue

                result = stage.execute(
                    config=self.config,
                    facility=fconfig,
                    work_dir=fac_dir,
                    previous_results=facility_results,
                )
                # Save stage outputs first so sync hooks can rely on this file existing.
                result.save(result_path)
                post_sync_result = self._maybe_run_post_stage_sync_hook(
                    result=result,
                    stage_key=stage_key,
                    facility_id=fid,
                    facility_work_dir=fac_dir,
                    result_path=result_path,
                )
                if post_sync_result != result:
                    post_sync_result.save(result_path)
                result = post_sync_result
                result = self._enforce_action_boost_required_stage_result(
                    result=result,
                    stage_name=stage.name,
                    action_boost=action_boost,
                )
                if result != post_sync_result:
                    result.save(result_path)
                facility_results[stage.name] = result
                all_results[stage_key] = result
                stage_provenance[stage_key] = {
                    "source": "executed",
                    "result_path": str(result_path),
                }

                if (
                    stage.name == "s3d_wm_refresh_loop"
                    and result.status == "success"
                    and not resume_from_results
                ):
                    s4_result_path = fac_dir / "s4_policy_eval_result.json"
                    s4_archive_path = fac_dir / "s4_policy_eval_pre_refresh_result.json"
                    if s4_result_path.exists():
                        shutil.copy2(s4_result_path, s4_archive_path)
                    post_refresh_s4 = PolicyEvalStage().execute(
                        config=self.config,
                        facility=fconfig,
                        work_dir=fac_dir,
                        previous_results=facility_results,
                    )
                    post_refresh_s4.save(s4_result_path)
                    synced_post_refresh_s4 = self._maybe_run_post_stage_sync_hook(
                        result=post_refresh_s4,
                        stage_key=f"{fid}/s4_policy_eval",
                        facility_id=fid,
                        facility_work_dir=fac_dir,
                        result_path=s4_result_path,
                    )
                    if synced_post_refresh_s4 != post_refresh_s4:
                        synced_post_refresh_s4.save(s4_result_path)
                    post_refresh_s4 = self._enforce_action_boost_required_stage_result(
                        result=synced_post_refresh_s4,
                        stage_name="s4_policy_eval",
                        action_boost=action_boost,
                    )
                    if post_refresh_s4 != synced_post_refresh_s4:
                        post_refresh_s4.save(s4_result_path)
                    facility_results["s4_policy_eval"] = post_refresh_s4
                    all_results[f"{fid}/s4_policy_eval"] = post_refresh_s4
                    stage_provenance[f"{fid}/s4_policy_eval"] = {
                        "source": "executed",
                        "result_path": str(s4_result_path),
                    }

                    if post_refresh_s4.status == "failed":
                        failed_key = f"{fid}/s4_policy_eval"
                        if failed_key not in failed_stage_keys:
                            failed_stage_keys.append(failed_key)
                        logger.error(
                            "Post-refresh Stage s4_policy_eval failed for %s: %s.",
                            fid,
                            post_refresh_s4.detail,
                        )
                        if fail_fast:
                            logger.error(
                                "Aborting remaining pipeline stages because fail_fast=true "
                                "(post-refresh failure: %s).",
                                failed_key,
                            )
                            aborted_early = True
                            break

                if result.status == "failed":
                    failed_key = stage_key
                    failed_stage_keys.append(failed_key)
                    logger.error(
                        "Stage %s failed for %s: %s.",
                        stage.name,
                        fid,
                        result.detail,
                    )
                    if fail_fast:
                        logger.error(
                            "Aborting remaining pipeline stages because fail_fast=true "
                            "(first failure: %s).",
                            failed_key,
                        )
                        aborted_early = True
                        break
                    logger.warning("Continuing to next stage because fail_fast=false.")

                if self._is_budget_exceeded(pipeline_start):
                    detail = self._budget_failure_detail(pipeline_start)
                    logger.error("Cloud budget guard triggered: %s", detail)
                    budget_key = "pipeline/cloud_budget_guard"
                    failed_stage_keys.append(budget_key)
                    all_results[budget_key] = StageResult(
                        stage_name="cloud_budget_guard",
                        status="failed",
                        elapsed_seconds=0,
                        detail=detail,
                    )
                    stage_provenance[budget_key] = {"source": "executed", "result_path": None}
                    self._maybe_trigger_auto_shutdown(detail)
                    aborted_early = True
                    break

            if aborted_early:
                break

        # Cross-site stage (requires all facilities)
        if aborted_early:
            logger.info("Skipping cross-site test because pipeline aborted early on stage failure.")
        elif len(self.config.facilities) >= 2:
            logger.info("=== Running cross-site discrimination test ===")
            cross_site = CrossSiteStage()
            first_fac = list(self.config.facilities.values())[0]
            cs_result = cross_site.execute(
                config=self.config,
                facility=first_fac,
                work_dir=self.work_dir,
                previous_results=all_results,
            )
            cross_site_result_path = self.work_dir / "s7_cross_site_result.json"
            cs_result.save(cross_site_result_path)
            post_sync_cs_result = self._maybe_run_post_stage_sync_hook(
                result=cs_result,
                stage_key="cross_site/s7_cross_site",
                facility_id="cross_site",
                facility_work_dir=self.work_dir,
                result_path=cross_site_result_path,
            )
            if post_sync_cs_result != cs_result:
                post_sync_cs_result.save(cross_site_result_path)
            cs_result = post_sync_cs_result
            all_results["cross_site/s7_cross_site"] = cs_result
            stage_provenance["cross_site/s7_cross_site"] = {
                "source": "executed",
                "result_path": str(cross_site_result_path),
            }
            if cs_result.status == "failed":
                failed_stage_keys.append("cross_site/s7_cross_site")
        else:
            logger.info("Skipping cross-site test (requires 2+ facilities)")

        # Write pipeline summary
        run_finished_at = datetime.now(timezone.utc).isoformat()
        stages_summary = {}
        for stage_key, stage_result in all_results.items():
            payload = stage_result.to_dict()
            payload["provenance"] = stage_provenance.get(
                stage_key,
                {"source": "executed", "result_path": None},
            )
            stages_summary[stage_key] = payload
        summary = {
            "num_facilities": len(self.config.facilities),
            "facility_ids": list(self.config.facilities.keys()),
            "stages": stages_summary,
            "overall_status": "failed" if failed_stage_keys else "success",
            "fail_fast": fail_fast,
            "resume_from_results": resume_from_results,
            "run_mode": run_mode,
            "run_started_at": run_started_at,
            "run_finished_at": run_finished_at,
            "aborted_early": aborted_early,
            "failed_stage_keys": failed_stage_keys,
        }
        write_json(summary, self.work_dir / "pipeline_summary.json")

        logger.info("Pipeline complete. Summary at %s", self.work_dir / "pipeline_summary.json")
        return all_results

    def _apply_action_boost_runtime_overrides(self) -> dict:
        cfg = getattr(self.config, "action_boost", None)
        state = {
            "enabled": bool(getattr(cfg, "enabled", False)),
            "require_full_pipeline": bool(getattr(cfg, "require_full_pipeline", False)),
        }
        if not state["enabled"]:
            return state

        if bool(getattr(cfg, "auto_switch_headline_scope_to_dual", True)):
            scope = (getattr(self.config.eval_policy, "headline_scope", "wm_only") or "wm_only").strip().lower()
            if scope == "wm_only":
                logger.warning(
                    "ActionBoost enabled: overriding eval_policy.headline_scope from wm_only to dual."
                )
                self.config.eval_policy.headline_scope = "dual"

        if bool(getattr(cfg, "auto_enable_rollout_dataset", True)):
            self.config.rollout_dataset.enabled = True
        if bool(getattr(cfg, "auto_enable_policy_finetune", True)):
            self.config.policy_finetune.enabled = True
        if bool(getattr(cfg, "auto_enable_policy_rl_loop", True)):
            self.config.policy_rl_loop.enabled = True

        self._apply_action_boost_compute_profile(str(getattr(cfg, "compute_profile", "standard")))
        return state

    def _apply_action_boost_compute_profile(self, profile: str) -> None:
        mode = (profile or "standard").strip().lower()
        if mode == "lean":
            self.config.policy_rl_loop.iterations = 1
            self.config.policy_rl_loop.rollouts_per_task = 6
            self.config.policy_rl_loop.policy_refine_steps_per_iter = 750
            self.config.policy_rl_loop.world_model_refresh_epochs = 2
            return
        if mode == "aggressive":
            self.config.policy_rl_loop.iterations = 3
            self.config.policy_rl_loop.rollouts_per_task = 12
            self.config.policy_rl_loop.policy_refine_steps_per_iter = 1500
            self.config.policy_rl_loop.world_model_refresh_epochs = 4
            return
        # standard
        self.config.policy_rl_loop.iterations = 2
        self.config.policy_rl_loop.rollouts_per_task = 8
        self.config.policy_rl_loop.policy_refine_steps_per_iter = 1000
        self.config.policy_rl_loop.world_model_refresh_epochs = 3

    def _enforce_action_boost_required_stage_result(
        self,
        *,
        result: StageResult,
        stage_name: str,
        action_boost: dict,
    ) -> StageResult:
        if not bool(action_boost.get("enabled", False)):
            return result
        if not bool(action_boost.get("require_full_pipeline", False)):
            return result
        if stage_name not in _ACTION_BOOST_REQUIRED_STAGES:
            return result
        if result.status != "skipped":
            return result
        detail = (
            f"ActionBoost require_full_pipeline=true forbids skipped required stage '{stage_name}'. "
            f"Original skip detail: {result.detail}"
        )
        return StageResult(
            stage_name=result.stage_name,
            status="failed",
            elapsed_seconds=result.elapsed_seconds,
            outputs=result.outputs,
            metrics=result.metrics,
            detail=detail,
            timestamp=result.timestamp,
        )

    def _load_existing_stage_result(self, result_path: Path) -> StageResult | None:
        if not result_path.exists():
            return None
        try:
            payload = read_json(result_path)
        except Exception:
            logger.warning("Failed reading existing stage result: %s", result_path, exc_info=True)
            return None

        try:
            return StageResult(
                stage_name=str(
                    payload.get("stage_name") or result_path.stem.replace("_result", "")
                ),
                status=str(payload.get("status", "failed")),
                elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
                outputs=dict(payload.get("outputs") or {}),
                metrics=dict(payload.get("metrics") or {}),
                detail=str(payload.get("detail") or ""),
                timestamp=str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            )
        except Exception:
            logger.warning("Malformed stage result payload at %s", result_path, exc_info=True)
            return None

    def _maybe_run_post_stage_sync_hook(
        self,
        *,
        result: StageResult,
        stage_key: str,
        facility_id: str,
        facility_work_dir: Path,
        result_path: Path,
    ) -> StageResult:
        hook_cmd = (os.environ.get("BLUEPRINT_POST_STAGE_SYNC_CMD") or "").strip()
        if not hook_cmd:
            return result

        env = os.environ.copy()
        env.update(
            {
                "BLUEPRINT_SYNC_STAGE_KEY": stage_key,
                "BLUEPRINT_SYNC_STAGE_NAME": result.stage_name,
                "BLUEPRINT_SYNC_STAGE_STATUS": result.status,
                "BLUEPRINT_SYNC_STAGE_DETAIL": result.detail,
                "BLUEPRINT_SYNC_FACILITY_ID": facility_id,
                "BLUEPRINT_SYNC_FACILITY_WORK_DIR": str(facility_work_dir),
                "BLUEPRINT_SYNC_RESULT_PATH": str(result_path),
                "BLUEPRINT_SYNC_PIPELINE_WORK_DIR": str(self.work_dir),
            }
        )

        strict = os.environ.get("BLUEPRINT_POST_STAGE_SYNC_STRICT", "0") == "1"
        try:
            proc = subprocess.run(
                hook_cmd,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        except Exception as exc:
            msg = f"Post-stage sync hook exception for {stage_key}: {exc}"
            logger.error(msg)
            if not strict:
                return result
            return StageResult(
                stage_name=result.stage_name,
                status="failed",
                elapsed_seconds=result.elapsed_seconds,
                outputs=result.outputs,
                metrics=result.metrics,
                detail=_append_detail(result.detail, msg),
            )

        if proc.returncode == 0:
            logger.info("Post-stage sync hook succeeded for %s", stage_key)
            return result

        stderr_tail = (proc.stderr or "")[-300:].strip()
        msg = (
            f"Post-stage sync hook failed for {stage_key} (exit={proc.returncode}). {stderr_tail}"
        ).strip()
        if strict:
            logger.error(msg)
            return StageResult(
                stage_name=result.stage_name,
                status="failed",
                elapsed_seconds=result.elapsed_seconds,
                outputs=result.outputs,
                metrics=result.metrics,
                detail=_append_detail(result.detail, msg),
            )
        logger.warning(msg)
        return result

    def _hourly_rate_usd(self) -> float | None:
        raw = (os.environ.get("BLUEPRINT_GPU_HOURLY_RATE_USD") or "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value

    def _estimated_cost_usd(self, pipeline_start: float) -> float | None:
        hourly_rate = self._hourly_rate_usd()
        if hourly_rate is None:
            return None
        elapsed_hours = max(0.0, (time.monotonic() - pipeline_start) / 3600.0)
        return hourly_rate * elapsed_hours

    def _is_budget_exceeded(self, pipeline_start: float) -> bool:
        # Allow cloud.max_cost_usd <= 0 to disable the guard explicitly.
        if float(self.config.cloud.max_cost_usd) <= 0:
            return False
        estimated = self._estimated_cost_usd(pipeline_start)
        if estimated is None:
            return False
        return estimated > float(self.config.cloud.max_cost_usd)

    def _budget_failure_detail(self, pipeline_start: float) -> str:
        estimated = self._estimated_cost_usd(pipeline_start)
        if estimated is None:
            return (
                "cloud.max_cost_usd is set but BLUEPRINT_GPU_HOURLY_RATE_USD is unset; "
                "cannot enforce budget."
            )
        return (
            f"Estimated spend ${estimated:.2f} exceeded cloud.max_cost_usd="
            f"${self.config.cloud.max_cost_usd:.2f}."
        )

    def _maybe_trigger_auto_shutdown(self, reason: str) -> None:
        if not self.config.cloud.auto_shutdown:
            return
        shutdown_cmd = (os.environ.get("BLUEPRINT_AUTO_SHUTDOWN_CMD") or "").strip()
        if not shutdown_cmd:
            logger.warning(
                "cloud.auto_shutdown=true but BLUEPRINT_AUTO_SHUTDOWN_CMD is not set. "
                "Skipping shutdown trigger."
            )
            return
        logger.warning("Executing auto-shutdown command due to budget guard trigger.")
        try:
            subprocess.run(
                shutdown_cmd,
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            logger.error("Failed to execute auto-shutdown command (%s): %s", shutdown_cmd, exc)


def _append_detail(detail: str, msg: str) -> str:
    detail = str(detail or "").strip()
    msg = str(msg or "").strip()
    if detail and msg:
        return f"{detail}\n{msg}"
    return detail or msg
