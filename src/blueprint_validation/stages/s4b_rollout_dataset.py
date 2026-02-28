"""Stage 4b: Export paired rollouts to RLDS-style training datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..training.rlds_export import export_rollouts_to_rlds_jsonl
from .base import PipelineStage

logger = get_logger("stages.s4b_rollout_dataset")


class RolloutDatasetStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4b_rollout_dataset"

    @property
    def description(self) -> str:
        return "Export baseline/adapted policy rollouts into RLDS-style train+heldout datasets"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility, previous_results
        if not config.rollout_dataset.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="rollout_dataset.enabled=false",
            )

        scores_path = work_dir / "policy_eval" / "vlm_scores.json"
        if not scores_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Policy eval scores missing. Run Stage 4 first.",
            )
        scores = read_json(scores_path).get("scores", [])
        if not scores:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No rollout scores found to export.",
            )

        baseline, adapted = _paired_rollouts(scores)
        baseline = _filter_rollouts(baseline, config)
        adapted = _filter_rollouts(adapted, config)
        if not baseline or not adapted:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No paired rollouts passed consistency filters for dataset export.",
            )

        split_train_ids, split_heldout_ids = _split_pairs(
            pair_ids=[_pair_id(r) for r in baseline],
            seed=config.rollout_dataset.seed,
            train_split=config.rollout_dataset.train_split,
        )
        baseline_train, baseline_heldout = _split_by_ids(
            baseline, split_train_ids, split_heldout_ids
        )
        adapted_train, adapted_heldout = _split_by_ids(adapted, split_train_ids, split_heldout_ids)

        dataset_root = config.rollout_dataset.export_dir / work_dir.name
        baseline_root = dataset_root / "baseline"
        adapted_root = dataset_root / "adapted"
        baseline_train_meta = export_rollouts_to_rlds_jsonl(
            rollouts=baseline_train,
            output_dir=baseline_root / "train",
            condition="baseline",
            split="train",
            task_threshold=config.rollout_dataset.task_score_threshold,
            min_steps_per_rollout=config.rollout_dataset.min_steps_per_rollout,
            include_failed_rollouts=config.rollout_dataset.include_failed_rollouts,
        )
        baseline_heldout_meta = export_rollouts_to_rlds_jsonl(
            rollouts=baseline_heldout,
            output_dir=baseline_root / "heldout",
            condition="baseline",
            split="heldout",
            task_threshold=config.rollout_dataset.task_score_threshold,
            min_steps_per_rollout=config.rollout_dataset.min_steps_per_rollout,
            include_failed_rollouts=True,
        )
        adapted_train_meta = export_rollouts_to_rlds_jsonl(
            rollouts=adapted_train,
            output_dir=adapted_root / "train",
            condition="adapted",
            split="train",
            task_threshold=config.rollout_dataset.task_score_threshold,
            min_steps_per_rollout=config.rollout_dataset.min_steps_per_rollout,
            include_failed_rollouts=config.rollout_dataset.include_failed_rollouts,
        )
        adapted_heldout_meta = export_rollouts_to_rlds_jsonl(
            rollouts=adapted_heldout,
            output_dir=adapted_root / "heldout",
            condition="adapted",
            split="heldout",
            task_threshold=config.rollout_dataset.task_score_threshold,
            min_steps_per_rollout=config.rollout_dataset.min_steps_per_rollout,
            include_failed_rollouts=True,
        )

        summary = {
            "baseline_train": baseline_train_meta,
            "baseline_heldout": baseline_heldout_meta,
            "adapted_train": adapted_train_meta,
            "adapted_heldout": adapted_heldout_meta,
            "baseline_dataset_name": config.rollout_dataset.baseline_dataset_name,
            "adapted_dataset_name": config.rollout_dataset.adapted_dataset_name,
        }
        write_json(summary, dataset_root / "dataset_export_summary.json")

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "dataset_root": str(dataset_root),
                "baseline_dataset_root": str(baseline_root),
                "adapted_dataset_root": str(adapted_root),
                "summary_path": str(dataset_root / "dataset_export_summary.json"),
            },
            metrics={
                "num_pairs_input": min(len(baseline), len(adapted)),
                "num_pairs_train": len(split_train_ids),
                "num_pairs_heldout": len(split_heldout_ids),
                "baseline_train_episodes": baseline_train_meta["num_episodes"],
                "adapted_train_episodes": adapted_train_meta["num_episodes"],
            },
        )


def _paired_rollouts(scores: List[dict]) -> Tuple[List[dict], List[dict]]:
    by_cond: Dict[str, Dict[str, dict]] = {"baseline": {}, "adapted": {}}
    for entry in scores:
        key = _pair_id(entry)
        cond = entry.get("condition")
        if cond in by_cond:
            by_cond[cond][key] = entry
    keys = sorted(set(by_cond["baseline"].keys()) & set(by_cond["adapted"].keys()))
    baseline = [by_cond["baseline"][k] for k in keys]
    adapted = [by_cond["adapted"][k] for k in keys]
    return baseline, adapted


def _pair_id(entry: dict) -> str:
    return f"{entry.get('rollout_index')}::{entry.get('task', '')}"


def _split_pairs(pair_ids: List[str], seed: int, train_split: float) -> Tuple[set[str], set[str]]:
    if not pair_ids:
        return set(), set()
    ids = sorted(set(pair_ids))
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(len(ids))
    cutoff = int(round(len(ids) * train_split))
    train_ids = {ids[i] for i in perm[:cutoff]}
    heldout_ids = {ids[i] for i in perm[cutoff:]}
    if not heldout_ids:
        heldout_ids = {ids[perm[-1]]}
        train_ids = set(ids) - heldout_ids
    return train_ids, heldout_ids


def _split_by_ids(
    entries: List[dict], train_ids: set[str], heldout_ids: set[str]
) -> Tuple[List[dict], List[dict]]:
    train = []
    heldout = []
    for e in entries:
        pid = _pair_id(e)
        if pid in train_ids:
            train.append(e)
        elif pid in heldout_ids:
            heldout.append(e)
    return train, heldout


def _filter_rollouts(entries: List[dict], config: ValidationConfig) -> List[dict]:
    out = []
    for e in entries:
        actions = e.get("action_sequence") or []
        if len(actions) < config.rollout_dataset.min_steps_per_rollout:
            continue
        if config.rollout_dataset.require_consistent_action_dim:
            dims = {len(a) for a in actions if isinstance(a, list)}
            if len(dims) != 1:
                continue
        if not _action_smoothness_ok(actions, config.rollout_dataset.max_action_delta_norm):
            continue
        # deterministic hash for reproducibility in summaries/debugging
        digest = hashlib.md5(_pair_id(e).encode("utf-8")).hexdigest()[:8]
        item = dict(e)
        item["pair_hash"] = digest
        out.append(item)
    return out


def _action_smoothness_ok(actions: List[list], max_delta_norm: float) -> bool:
    if len(actions) < 2:
        return True
    try:
        arr = np.asarray(actions, dtype=np.float64)
    except Exception:
        return False
    if not np.isfinite(arr).all():
        return False
    deltas = np.diff(arr, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    return bool(np.max(norms) <= max_delta_norm)
