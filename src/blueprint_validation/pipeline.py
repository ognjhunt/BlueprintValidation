"""Full pipeline orchestrator — chains all stages sequentially."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .common import StageResult, get_logger, write_json
from .config import ValidationConfig
from .stages.s1_render import RenderStage
from .stages.s1b_robot_composite import RobotCompositeStage
from .stages.s1c_gemini_polish import GeminiPolishStage
from .stages.s2_enrich import EnrichStage
from .stages.s3_finetune import FinetuneStage
from .stages.s3b_policy_finetune import PolicyFinetuneStage
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


class ValidationPipeline:
    """Orchestrates the full validation pipeline across all facilities."""

    def __init__(self, config: ValidationConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> Dict[str, StageResult]:
        """Run all stages for all facilities, then cross-site test."""
        all_results: Dict[str, StageResult] = {}

        # Per-facility stages (1-6)
        per_facility_stages = [
            RenderStage(),              # S1: splat -> video clips
            RobotCompositeStage(),      # S1b: URDF robot arm composite
            GeminiPolishStage(),        # S1c: optional Gemini photorealism polish
            EnrichStage(),              # S2: Cosmos Transfer variants
            FinetuneStage(),            # S3: DreamDojo LoRA fine-tune
            PolicyEvalStage(),          # S4: frozen policy eval (baseline + adapted)
            RLDSExportStage(),          # S4a: export adapted rollouts -> RLDS TFRecords
            PolicyFinetuneStage(),      # S3b: OpenVLA LoRA on pipeline-generated data
            TrainedPolicyEvalStage(),   # S4e: evaluate trained vs frozen baselines
            RolloutDatasetStage(),      # S4b: export paired rollouts -> JSONL datasets
            PolicyPairTrainStage(),     # S4c: train policy_base + policy_site
            PolicyPairEvalStage(),      # S4d: heldout paired evaluation
            VisualFidelityStage(),      # S5: PSNR/SSIM/LPIPS metrics
            SpatialAccuracyStage(),     # S6: VLM spatial scoring
        ]

        for fid, fconfig in self.config.facilities.items():
            logger.info("=== Processing facility: %s ===", fid)
            fac_dir = self.work_dir / fid
            fac_dir.mkdir(parents=True, exist_ok=True)

            facility_results: Dict[str, StageResult] = {}

            for stage in per_facility_stages:
                result = stage.execute(
                    config=self.config,
                    facility=fconfig,
                    work_dir=fac_dir,
                    previous_results=facility_results,
                )
                result.save(fac_dir / f"{stage.name}_result.json")
                facility_results[stage.name] = result
                all_results[f"{fid}/{stage.name}"] = result

                if result.status == "failed":
                    logger.warning(
                        "Stage %s failed for %s: %s. Continuing to next stage.",
                        stage.name, fid, result.detail,
                    )

        # Cross-site stage (requires all facilities)
        if len(self.config.facilities) >= 2:
            logger.info("=== Running cross-site discrimination test ===")
            cross_site = CrossSiteStage()
            first_fac = list(self.config.facilities.values())[0]
            cs_result = cross_site.execute(
                config=self.config,
                facility=first_fac,
                work_dir=self.work_dir,
                previous_results=all_results,
            )
            cs_result.save(self.work_dir / "s7_cross_site_result.json")
            all_results["cross_site/s7_cross_site"] = cs_result
        else:
            logger.info("Skipping cross-site test (requires 2+ facilities)")

        # Write pipeline summary
        summary = {
            "num_facilities": len(self.config.facilities),
            "facility_ids": list(self.config.facilities.keys()),
            "stages": {k: v.to_dict() for k, v in all_results.items()},
            "overall_status": "success" if all(
                r.status != "failed" for r in all_results.values()
            ) else "partial",
        }
        write_json(summary, self.work_dir / "pipeline_summary.json")

        logger.info("Pipeline complete. Summary at %s", self.work_dir / "pipeline_summary.json")
        return all_results
