"""Stage 4f: PolaRiS policy evaluation for deployment-gate ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..common import StageResult, get_logger
from ..config import FacilityConfig, ValidationConfig
from ..policy_adapters import get_policy_adapter
from ..polaris import run_polaris_comparison
from ..polaris.runtime import polaris_primary_gate_enabled, resolve_polaris_scene_spec
from .base import PipelineStage
from .s4e_trained_eval import _resolve_trained_checkpoint

logger = get_logger("stages.s4f_polaris_eval")


class PolarisEvalStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4f_polaris_eval"

    @property
    def description(self) -> str:
        return "Evaluate frozen vs adapted OpenVLA in PolaRiS and rank the deployment candidate"

    def preflight(self, config: ValidationConfig):
        if not bool(config.eval_polaris.enabled):
            return []
        return super().preflight(config)

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        if not bool(config.eval_polaris.enabled):
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0.0,
                detail="eval_polaris.enabled=false",
            )
        if (
            getattr(config.eval_policy, "headline_scope", "wm_only") or "wm_only"
        ).strip().lower() == "wm_only":
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0.0,
                detail="Skipped by policy: eval_policy.headline_scope=wm_only",
            )

        scene_spec = resolve_polaris_scene_spec(config, facility)
        if not scene_spec.primary_eligible and polaris_primary_gate_enabled(config):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0.0,
                detail=(
                    "PolaRiS is configured as the default primary gate but the current scene "
                    f"handoff is not primary-eligible ({scene_spec.detail})."
                ),
            )

        policy_adapter = get_policy_adapter(config.policy_adapter)
        frozen_model_name, frozen_checkpoint = policy_adapter.base_model_ref(config.eval_policy)
        adapted_checkpoint = _resolve_trained_checkpoint(
            previous_results=previous_results,
            work_dir=work_dir,
            policy_adapter=policy_adapter,
        )
        if adapted_checkpoint is None or not Path(adapted_checkpoint).exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0.0,
                detail="No adapted policy checkpoint available for PolaRiS comparison.",
            )

        result = run_polaris_comparison(
            config=config,
            facility=facility,
            work_dir=work_dir,
            frozen_model_name=frozen_model_name,
            frozen_checkpoint=frozen_checkpoint,
            adapted_checkpoint=Path(adapted_checkpoint),
        )
        summary_path = Path(result["summary_path"])
        metrics = {
            "winner": result.get("winner", "unknown"),
            "scene_mode": result.get("scene_mode", "unknown"),
            "frozen_success_rate": (result.get("frozen_openvla", {}) or {}).get("success_rate"),
            "adapted_success_rate": (result.get("adapted_openvla", {}) or {}).get("success_rate"),
            "frozen_mean_progress": (result.get("frozen_openvla", {}) or {}).get("mean_progress"),
            "adapted_mean_progress": (result.get("adapted_openvla", {}) or {}).get("mean_progress"),
            "delta_vs_frozen": result.get("delta_vs_frozen"),
            "num_rollouts": (result.get("adapted_openvla", {}) or {}).get("num_rollouts"),
            "primary_gate_candidate": bool(config.eval_polaris.default_as_primary_gate),
        }
        outputs = {
            "polaris_summary_path": str(summary_path),
            "frozen_report_path": (result.get("frozen_openvla", {}) or {}).get("report_path"),
            "adapted_report_path": (result.get("adapted_openvla", {}) or {}).get("report_path"),
            "frozen_csv_path": (result.get("frozen_openvla", {}) or {}).get("csv_path"),
            "adapted_csv_path": (result.get("adapted_openvla", {}) or {}).get("csv_path"),
        }
        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0.0,
            outputs=outputs,
            metrics=metrics,
            detail=(
                "PolaRiS ranked "
                f"{metrics['winner']} for the deployment scene using mode={metrics['scene_mode']}."
            ),
        )
