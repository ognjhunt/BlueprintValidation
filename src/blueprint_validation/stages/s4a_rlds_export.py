"""Stage 4a: Export successful rollouts to RLDS TFRecord format for policy training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from ..common import StageResult, get_logger, read_json
from ..config import FacilityConfig, ValidationConfig
from ..training.rlds_export import (
    convert_jsonl_to_tfrecord,
    export_rollouts_to_rlds_jsonl,
)
from .base import PipelineStage

logger = get_logger("stages.s4a_rlds_export")


class RLDSExportStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4a_rlds_export"

    @property
    def description(self) -> str:
        return "Export successful rollouts to RLDS TFRecords for OpenVLA-OFT fine-tuning"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility

        if not config.rollout_dataset.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="rollout_dataset.enabled=false",
            )

        if not config.policy_finetune.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_finetune.enabled=false; no need to export RLDS",
            )

        # Read Stage 4 scores
        prev_s4 = previous_results.get("s4_policy_eval")
        if not prev_s4 or prev_s4.status != "success":
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Stage 4 (policy_eval) did not succeed. Cannot export rollouts.",
            )

        scores_path = prev_s4.outputs.get("scores_path")
        if not scores_path or not Path(scores_path).exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Scores file not found: {scores_path}",
            )

        scores_data = read_json(Path(scores_path))
        all_rollouts: List[Dict] = scores_data.get("scores", [])

        # Filter: only adapted-condition rollouts (training data from site-adapted world model)
        adapted_rollouts = [r for r in all_rollouts if r.get("condition") == "adapted"]
        if not adapted_rollouts:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No adapted-condition rollouts found in Stage 4 output.",
            )

        # Shuffle and split into train/eval
        rng = random.Random(config.rollout_dataset.seed)
        rng.shuffle(adapted_rollouts)
        split_idx = int(len(adapted_rollouts) * config.rollout_dataset.train_split)
        train_rollouts = adapted_rollouts[:split_idx]
        eval_rollouts = adapted_rollouts[split_idx:]

        stage_dir = work_dir / "rlds_export"
        stage_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = config.rollout_dataset.adapted_dataset_name
        threshold = config.rollout_dataset.task_score_threshold
        min_steps = config.rollout_dataset.min_steps_per_rollout
        include_failed = config.rollout_dataset.include_failed_rollouts

        # Export train split to JSONL
        train_dir = stage_dir / "train"
        train_meta = export_rollouts_to_rlds_jsonl(
            rollouts=train_rollouts,
            output_dir=train_dir,
            condition="adapted",
            split="train",
            task_threshold=threshold,
            min_steps_per_rollout=min_steps,
            include_failed_rollouts=include_failed,
        )

        # Export eval split to JSONL
        eval_dir = stage_dir / "eval"
        eval_meta = export_rollouts_to_rlds_jsonl(
            rollouts=eval_rollouts,
            output_dir=eval_dir,
            condition="adapted",
            split="eval",
            task_threshold=threshold,
            min_steps_per_rollout=min_steps,
            include_failed_rollouts=include_failed,
        )

        num_train = train_meta.get("num_episodes", 0)
        num_eval = eval_meta.get("num_episodes", 0)

        if num_train == 0:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    f"No episodes passed filters (threshold={threshold}, "
                    f"min_steps={min_steps}). "
                    f"Total adapted rollouts: {len(adapted_rollouts)}"
                ),
            )

        # Convert JSONL to TFRecords for OpenVLA-OFT
        tfrecord_dir = config.rollout_dataset.export_dir / dataset_name
        convert_jsonl_to_tfrecord(
            train_jsonl_path=train_dir / "episodes.jsonl",
            eval_jsonl_path=eval_dir / "episodes.jsonl" if num_eval > 0 else None,
            output_dir=tfrecord_dir,
            dataset_name=dataset_name,
        )

        logger.info(
            "Exported %d train + %d eval episodes to %s",
            num_train, num_eval, tfrecord_dir,
        )

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "rlds_dataset_dir": str(tfrecord_dir),
                "dataset_name": dataset_name,
                "train_jsonl": str(train_dir / "episodes.jsonl"),
                "eval_jsonl": str(eval_dir / "episodes.jsonl"),
            },
            metrics={
                "num_train_episodes": num_train,
                "num_eval_episodes": num_eval,
                "num_train_successes": train_meta.get("num_successes", 0),
                "task_score_threshold": threshold,
                "total_adapted_rollouts": len(adapted_rollouts),
            },
        )
