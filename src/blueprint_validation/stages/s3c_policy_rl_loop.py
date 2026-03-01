"""Stage 3c: World-VLA-Loop-style policy RL loop in adapted world model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..common import StageResult, get_logger
from ..config import FacilityConfig, ValidationConfig
from ..policy_adapters import get_policy_adapter
from ..training.policy_rl_loop import run_policy_rl_iterations
from .base import PipelineStage

logger = get_logger("stages.s3c_policy_rl_loop")


class PolicyRLLoopStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s3c_policy_rl_loop"

    @property
    def description(self) -> str:
        return "World-VLA-Loop-style iterative policy/world-model refinement"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        if (
            (getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only")
            .strip()
            .lower()
            == "wm_only"
        ):
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail=(
                    "Skipped by policy: eval_policy.headline_scope=wm_only "
                    "(OpenVLA stages deferred)."
                ),
            )
        if not config.policy_rl_loop.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_rl_loop.enabled=false",
            )
        if not config.policy_finetune.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_finetune.enabled=false; RL loop requires policy updates",
            )

        adapted_world_checkpoint = _resolve_adapted_world_checkpoint(previous_results, work_dir)
        if adapted_world_checkpoint is None or not adapted_world_checkpoint.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Adapted DreamDojo checkpoint not found. Run Stage 3 first.",
            )

        policy_adapter = get_policy_adapter(config.policy_adapter)
        initial_policy_checkpoint = _resolve_initial_policy_checkpoint(
            previous_results=previous_results,
            work_dir=work_dir,
            policy_adapter=policy_adapter,
        )
        stage_dir = work_dir / "policy_rl_loop"
        stage_dir.mkdir(parents=True, exist_ok=True)

        result = run_policy_rl_iterations(
            config=config,
            facility=facility,
            work_dir=work_dir,
            output_dir=stage_dir,
            initial_policy_checkpoint=initial_policy_checkpoint,
            adapted_world_checkpoint=adapted_world_checkpoint,
        )
        status = result.get("status", "failed")
        return StageResult(
            stage_name=self.name,
            status=status,
            elapsed_seconds=0,
            outputs={
                "policy_rl_loop_dir": str(stage_dir),
                "adapted_policy_checkpoint_rl": result.get("final_policy_checkpoint", ""),
                "adapted_openvla_checkpoint_rl": result.get("final_policy_checkpoint", ""),
                "adapted_world_checkpoint_rl": result.get("final_world_checkpoint", ""),
                "loop_log": str(stage_dir / "policy_rl_loop_log.json"),
            },
            metrics={
                "iterations_completed": result.get("iterations_completed", 0),
                "reward_mode": config.policy_rl_loop.reward_mode,
                "horizon_steps": config.policy_rl_loop.horizon_steps,
                "rollouts_per_task": config.policy_rl_loop.rollouts_per_task,
            },
            detail=result.get("detail", ""),
        )


def _resolve_initial_policy_checkpoint(
    previous_results: Dict[str, StageResult],
    work_dir: Path,
    policy_adapter,
) -> Optional[Path]:
    # Prefer Stage 3b fine-tuned checkpoint.
    s3b = previous_results.get("s3b_policy_finetune")
    if s3b and s3b.status == "success":
        candidate = s3b.outputs.get("adapted_policy_checkpoint") or s3b.outputs.get(
            "adapted_openvla_checkpoint"
        )
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    # Fallback to known policy_finetune run root.
    fallback = work_dir / "policy_finetune" / "runs"
    checkpoint = policy_adapter.resolve_latest_checkpoint(fallback)
    if checkpoint is not None:
        return checkpoint
    return None


def _resolve_adapted_world_checkpoint(
    previous_results: Dict[str, StageResult],
    work_dir: Path,
) -> Optional[Path]:
    s3 = previous_results.get("s3_finetune")
    if s3 and s3.status == "success":
        candidate = s3.outputs.get("adapted_checkpoint_path") or s3.outputs.get("lora_weights_path")
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    for candidate in [
        work_dir / "finetune" / "adapted_checkpoint",
        work_dir / "finetune" / "lora_weights",
    ]:
        if candidate.exists():
            return candidate
    return None
