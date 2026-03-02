"""Stage 6: Spatial accuracy verification via VLM judge."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig, VLMJudgeConfig
from ..evaluation.vlm_judge import score_spatial_accuracy
from .base import PipelineStage

logger = get_logger("stages.s6_spatial_accuracy")

_REASONING_CONFLICT_TOKENS = (
    "cannot evaluate",
    "can't evaluate",
    "not visible",
    "too distorted",
    "distorted to evaluate",
    "impossible to evaluate",
    "unable to evaluate",
    "indiscernible",
)


class SpatialAccuracyStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s6_spatial_accuracy"

    @property
    def description(self) -> str:
        return "Verify generated videos match real facility layout via VLM judge"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        spatial_dir = work_dir / "spatial_accuracy"
        spatial_dir.mkdir(parents=True, exist_ok=True)

        # Find enriched or policy-generated videos to evaluate
        enriched_manifest_path = work_dir / "enriched" / "enriched_manifest.json"
        if not enriched_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Enriched manifest not found. Run Stage 2 first.",
            )

        enriched_manifest = read_json(enriched_manifest_path)

        vlm_config = VLMJudgeConfig(
            model=config.eval_spatial.vlm_model,
            api_key_env=config.eval_policy.vlm_judge.api_key_env,
            enable_agentic_vision=True,
        )

        scores: List[Dict] = []
        clips = enriched_manifest.get("clips", [])

        # Sample clips for evaluation
        sample_size = min(config.eval_spatial.num_sample_frames, len(clips))
        rng = np.random.default_rng(seed=42)
        sampled = rng.choice(len(clips), size=sample_size, replace=False)

        for idx in sampled:
            clip = clips[idx]
            video_path = Path(clip["output_video_path"])

            if not video_path.exists():
                continue

            try:
                score = score_spatial_accuracy(
                    video_path=video_path,
                    facility_description=facility.description,
                    landmarks=facility.landmarks,
                    config=vlm_config,
                )
                reasoning = str(score.reasoning or "")
                reasoning_lower = reasoning.lower()
                reasoning_conflict = any(
                    token in reasoning_lower for token in _REASONING_CONFLICT_TOKENS
                )
                scores.append(
                    {
                        "clip_name": clip.get("clip_name", ""),
                        "variant": clip.get("variant_name", ""),
                        "spatial_score": score.spatial_score,
                        "visual_score": score.visual_score,
                        "task_score": score.task_score,
                        "reasoning": reasoning,
                        "reasoning_conflict": reasoning_conflict,
                    }
                )
            except Exception as e:
                logger.warning("Spatial scoring failed for %s: %s", video_path.name, e)

        # Aggregate
        valid_scores = [row for row in scores if not bool(row.get("reasoning_conflict", False))]
        if valid_scores:
            mean_spatial = float(np.mean([s["spatial_score"] for s in valid_scores]))
            mean_visual = float(np.mean([s["visual_score"] for s in valid_scores]))
            mean_landmark = float(np.mean([s["task_score"] for s in valid_scores]))
        else:
            mean_spatial = mean_visual = mean_landmark = 0.0

        num_reasoning_conflicts = sum(
            1 for row in scores if bool(row.get("reasoning_conflict", False))
        )
        num_valid_samples = len(valid_scores)
        metrics = {
            "mean_spatial_score": round(mean_spatial, 3),
            "mean_visual_score": round(mean_visual, 3),
            "mean_landmark_score": round(mean_landmark, 3),
            "num_evaluated": len(scores),
            "num_valid_samples": num_valid_samples,
            "num_reasoning_conflicts": num_reasoning_conflicts,
        }

        write_json({"metrics": metrics, "scores": scores}, spatial_dir / "spatial_accuracy.json")

        detail_parts: List[str] = []
        if num_valid_samples < int(config.eval_spatial.min_valid_samples):
            detail_parts.append(
                "Insufficient valid spatial samples: "
                f"{num_valid_samples} < min_valid_samples={int(config.eval_spatial.min_valid_samples)}"
            )
        if bool(config.eval_spatial.fail_on_reasoning_conflict) and num_reasoning_conflicts > 0:
            detail_parts.append(
                f"Reasoning conflicts detected in {num_reasoning_conflicts} scored rows."
            )
        stage_failed = bool(detail_parts) or not bool(scores)
        if not scores and not detail_parts:
            detail_parts.append("No spatial scores were produced.")

        return StageResult(
            stage_name=self.name,
            status="failed" if stage_failed else "success",
            elapsed_seconds=0,
            outputs={"spatial_dir": str(spatial_dir)},
            metrics=metrics,
            detail=" ".join(detail_parts).strip(),
        )
