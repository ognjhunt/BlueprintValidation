"""Stage 4b: Export paired rollouts to RLDS-style training datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.claim_protocol import claim_protocol_enabled
from ..training.external_rollouts import load_external_rollouts_for_policy
from ..training.native_teacher import generate_correction_rollouts, generate_teacher_rollouts
from ..training.rlds_export import export_rollouts_to_rlds_jsonl
from ..validation import ManifestValidationError, load_and_validate_manifest
from .base import PipelineStage

logger = get_logger("stages.s4b_rollout_dataset")


def _eval_only_task_set(config: ValidationConfig) -> set[str]:
    if not bool(config.policy_compare.enabled):
        return set()
    return {
        str(task).strip()
        for task in (config.policy_compare.heldout_tasks or [])
        if str(task).strip()
    }


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
        del previous_results
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
        try:
            scores_data = load_and_validate_manifest(
                scores_path,
                manifest_type="policy_scores",
                require_existing_paths=True,
            )
        except ManifestValidationError as exc:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Invalid policy scores manifest: {exc}",
            )
        scores = scores_data.get("scores", [])
        if not scores:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No rollout scores found to export.",
            )
        eval_report_path = work_dir / "policy_eval" / "policy_eval_report.json"
        eval_report = read_json(eval_report_path) if eval_report_path.exists() else {}
        claim_mode = bool(eval_report.get("claim_mode", False))
        action_contract = eval_report.get("action_contract", {})
        reliability_gate = eval_report.get("reliability_gate", {})
        if claim_mode and not bool(action_contract.get("compliant", False)):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Claim mode action contract failed: {action_contract}",
            )
        if claim_mode and not bool(reliability_gate.get("passed", False)):
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Claim mode reliability gate failed: {reliability_gate}",
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

        eval_only_tasks = _eval_only_task_set(config)
        forced_heldout_ids = {
            _pair_id(entry)
            for entry in baseline
            if str(entry.get("task", "")).strip() in eval_only_tasks
        }
        claim_split_path = work_dir / "policy_eval" / "claim_split_manifest.json"
        claim_manifest_path = work_dir / "policy_eval" / "claim_manifest.json"
        claim_manifest = read_json(claim_manifest_path) if claim_manifest_path.exists() else {}
        if claim_protocol_enabled(config):
            if not claim_split_path.exists():
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail="Claim protocol requires policy_eval/claim_split_manifest.json.",
                )
            claim_split = read_json(claim_split_path)
            split_train_ids = set(str(v) for v in claim_split.get("train_eval_cell_ids", []))
            split_heldout_ids = set(str(v) for v in claim_split.get("eval_eval_cell_ids", []))
            split_heldout_ids |= forced_heldout_ids
        else:
            split_train_ids, split_heldout_ids = _split_pairs(
                entries=baseline,
                seed=config.rollout_dataset.seed,
                train_split=config.rollout_dataset.train_split,
                forced_heldout_ids=forced_heldout_ids,
            )
        baseline_train, baseline_heldout = _split_by_ids(
            baseline, split_train_ids, split_heldout_ids
        )
        adapted_train, adapted_heldout = _split_by_ids(adapted, split_train_ids, split_heldout_ids)
        external_rollouts = load_external_rollouts_for_policy(config, previous_results)
        if external_rollouts:
            adapted_train = _merge_augmented_rollouts(adapted_train, external_rollouts)
        native_teacher_summary: Dict[str, object] = {}
        generic_control_train = list(baseline_train)
        generic_control_mode = "baseline_only"
        if claim_protocol_enabled(config) and bool(getattr(config.native_teacher, "enabled", False)):
            teacher_dir = work_dir / "native_teacher"
            site_teacher_rows, site_teacher_meta = generate_teacher_rollouts(
                config=config,
                facility=facility,
                work_dir=work_dir,
                output_dir=teacher_dir / "site_teacher",
                condition="adapted",
                mode="site",
                max_steps=int(config.native_teacher.planner_horizon_steps),
            )
            generic_teacher_rows: List[dict] = []
            generic_teacher_meta: Dict[str, object] = {}
            if bool(config.native_teacher.include_generic_control):
                generic_teacher_rows, generic_teacher_meta = generate_teacher_rollouts(
                    config=config,
                    facility=facility,
                    work_dir=work_dir,
                    output_dir=teacher_dir / "generic_teacher",
                    condition="baseline",
                    mode="generic",
                    max_steps=int(config.native_teacher.planner_horizon_steps),
                )
            correction_rows: List[dict] = []
            correction_meta: Dict[str, object] = {}
            if bool(config.native_teacher.generate_corrections):
                correction_rows, correction_meta = generate_correction_rollouts(
                    config=config,
                    facility=facility,
                    work_dir=work_dir,
                    output_dir=teacher_dir / "site_corrections",
                    failed_rows=[
                        row
                        for row in adapted_train
                        if not bool(row.get("task_success", False))
                    ],
                    condition="adapted",
                    mode="site",
                    max_steps=int(config.native_teacher.planner_horizon_steps),
                )
            adapted_train = _merge_augmented_rollouts(adapted_train, site_teacher_rows + correction_rows)
            if generic_teacher_rows:
                generic_control_train = _merge_augmented_rollouts(
                    baseline_train,
                    generic_teacher_rows,
                )
                generic_control_mode = "provisional_generic_control"
            native_teacher_summary = {
                "site_teacher": site_teacher_meta,
                "generic_teacher": generic_teacher_meta,
                "site_corrections": correction_meta,
                "num_site_teacher_rows": len(site_teacher_rows),
                "num_generic_teacher_rows": len(generic_teacher_rows),
                "num_site_correction_rows": len(correction_rows),
            }
        if external_rollouts:
            native_teacher_summary["external_rollouts"] = {
                "num_rows": len(external_rollouts),
                "source_name": str(config.external_rollouts.source_name or "teleop"),
            }
        strict_generic_rows = _load_leave_one_facility_out_generic_pool(
            config=config,
            target_facility_id=str(work_dir.name),
            target_teacher_count=int(native_teacher_summary.get("num_site_teacher_rows", 0) or 0),
            target_correction_count=int(native_teacher_summary.get("num_site_correction_rows", 0) or 0),
            pipeline_work_root=work_dir.parent,
        )
        if strict_generic_rows:
            generic_control_train = strict_generic_rows
            generic_control_mode = "leave_one_facility_out"
        leakage_error = _claim_split_leakage_error(
            train_rollouts=baseline_train + generic_control_train + adapted_train,
            heldout_rollouts=baseline_heldout + adapted_heldout,
        )
        if leakage_error is not None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=leakage_error,
            )

        dataset_root = config.rollout_dataset.export_dir / work_dir.name
        baseline_root = dataset_root / "baseline"
        adapted_root = dataset_root / "adapted"
        generic_root = dataset_root / "generic_control"
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
        generic_control_train_meta = export_rollouts_to_rlds_jsonl(
            rollouts=generic_control_train,
            output_dir=generic_root / "train",
            condition="baseline",
            split="train",
            task_threshold=config.rollout_dataset.task_score_threshold,
            min_steps_per_rollout=config.rollout_dataset.min_steps_per_rollout,
            include_failed_rollouts=config.rollout_dataset.include_failed_rollouts,
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
            "generic_control_train": generic_control_train_meta,
            "adapted_train": adapted_train_meta,
            "adapted_heldout": adapted_heldout_meta,
            "baseline_dataset_name": config.rollout_dataset.baseline_dataset_name,
            "adapted_dataset_name": config.rollout_dataset.adapted_dataset_name,
            "claim_mode": claim_mode,
            "action_contract": action_contract,
            "reliability_gate": reliability_gate,
            "action_contract_hash": _action_contract_hash(action_contract),
            "eval_only_tasks": sorted(eval_only_tasks),
            "num_forced_heldout_pairs": len(forced_heldout_ids),
            "claim_protocol": claim_protocol_enabled(config),
            "claim_split_manifest_path": str(claim_split_path) if claim_split_path.exists() else "",
            "claim_manifest_path": str(claim_manifest_path) if claim_manifest_path.exists() else "",
            "generic_control_mode": generic_control_mode,
            "investor_grade_generic_control": generic_control_mode == "leave_one_facility_out",
            "dataset_lineage": {
                "world_snapshot_hash": str(
                    claim_manifest.get("world_snapshot_hash", "")
                    or eval_report.get("world_snapshot_hash", "")
                    or ""
                ),
                "claim_manifest_hash": _json_manifest_hash(claim_manifest_path),
                "claim_split_manifest_hash": _json_manifest_hash(claim_split_path),
                "train_eval_cell_ids_hash": _sorted_token_hash(split_train_ids),
                "heldout_eval_cell_ids_hash": _sorted_token_hash(split_heldout_ids),
            },
            "native_teacher": native_teacher_summary,
        }
        write_json(summary, dataset_root / "dataset_export_summary.json")

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "dataset_root": str(dataset_root),
                "baseline_dataset_root": str(baseline_root),
                "generic_control_dataset_root": str(generic_root),
                "adapted_dataset_root": str(adapted_root),
                "summary_path": str(dataset_root / "dataset_export_summary.json"),
            },
            metrics={
                "num_pairs_input": min(len(baseline), len(adapted)),
                "num_pairs_train": len(split_train_ids),
                "num_pairs_heldout": len(split_heldout_ids),
                "num_forced_heldout_pairs": len(forced_heldout_ids),
                "baseline_train_episodes": baseline_train_meta["num_episodes"],
                "generic_control_train_episodes": generic_control_train_meta["num_episodes"],
                "adapted_train_episodes": adapted_train_meta["num_episodes"],
                "investor_grade_generic_control": generic_control_mode == "leave_one_facility_out",
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
    eval_cell_id = str(entry.get("eval_cell_id", "")).strip()
    if eval_cell_id:
        return eval_cell_id
    return f"{entry.get('rollout_index')}::{entry.get('task', '')}"


def _split_pairs(
    entries: List[dict],
    seed: int,
    train_split: float,
    forced_heldout_ids: set[str] | None = None,
) -> Tuple[set[str], set[str]]:
    if not entries:
        return set(), set()
    forced_heldout_ids = set(forced_heldout_ids or set())
    ids = sorted({_pair_id(entry) for entry in entries})
    trainable_ids = [pair_id for pair_id in ids if pair_id not in forced_heldout_ids]
    if not trainable_ids:
        return set(), set(forced_heldout_ids)
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(len(trainable_ids))
    cutoff = int(round(len(trainable_ids) * train_split))
    train_ids = {trainable_ids[i] for i in perm[:cutoff]}
    heldout_ids = {trainable_ids[i] for i in perm[cutoff:]} | forced_heldout_ids
    if not heldout_ids:
        heldout_ids = {trainable_ids[perm[-1]]} | forced_heldout_ids
        train_ids = set(trainable_ids) - {trainable_ids[perm[-1]]}
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


def _action_contract_hash(action_contract: dict) -> str:
    try:
        payload = json.dumps(action_contract or {}, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = "{}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _merge_augmented_rollouts(base_rows: List[dict], extra_rows: List[dict]) -> List[dict]:
    merged: List[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for row in list(base_rows) + list(extra_rows):
        video_path = str(row.get("video_path", "")).strip()
        rollout_index = int(row.get("rollout_index", -1))
        source_type = str(row.get("source_type", "policy_rollout")).strip()
        key = (video_path, rollout_index, source_type)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(row))
    return merged


def _load_leave_one_facility_out_generic_pool(
    *,
    config: ValidationConfig,
    target_facility_id: str,
    target_teacher_count: int,
    target_correction_count: int,
    pipeline_work_root: Path,
) -> List[dict]:
    if not claim_protocol_enabled(config):
        return []
    if len(config.facilities) < int(config.claim_portfolio.min_facilities):
        return []
    teacher_rows: List[dict] = []
    correction_rows: List[dict] = []
    for facility_id in sorted(config.facilities.keys()):
        if facility_id == target_facility_id:
            continue
        facility_dir = pipeline_work_root / facility_id / "native_teacher"
        teacher_rows.extend(
            _load_row_manifest(facility_dir / "site_teacher" / "adapted_site_teacher_rows.json")
        )
        correction_rows.extend(
            _load_row_manifest(
                facility_dir / "site_corrections" / "adapted_site_correction_rows.json"
            )
        )
    if not teacher_rows and not correction_rows:
        return []
    teacher_rows = sorted(teacher_rows, key=_generic_row_sort_key)
    correction_rows = sorted(correction_rows, key=_generic_row_sort_key)
    if target_teacher_count > 0:
        teacher_rows = teacher_rows[:target_teacher_count]
    if target_correction_count > 0:
        correction_rows = correction_rows[:target_correction_count]
    return _merge_augmented_rollouts(teacher_rows, correction_rows)


def _load_row_manifest(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        payload = read_json(path)
    except Exception:
        return []
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _generic_row_sort_key(row: dict) -> tuple[str, int]:
    return (
        str(row.get("task_spec_id", "")).strip(),
        int(row.get("rollout_index", -1)),
    )


def _json_manifest_hash(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.dumps(read_json(path), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(path)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _sorted_token_hash(values: set[str]) -> str:
    payload = json.dumps(sorted(str(value) for value in values), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _claim_split_leakage_error(
    *,
    train_rollouts: List[dict],
    heldout_rollouts: List[dict],
) -> str | None:
    for field_name in ("eval_cell_id", "start_clip_id", "start_frame_hash"):
        train_values = {
            str(row.get(field_name, "")).strip()
            for row in train_rollouts
            if str(row.get(field_name, "")).strip()
        }
        heldout_values = {
            str(row.get(field_name, "")).strip()
            for row in heldout_rollouts
            if str(row.get(field_name, "")).strip()
        }
        overlap = sorted(train_values & heldout_values)
        if overlap:
            return (
                f"Claim split leakage detected on {field_name}: "
                f"{', '.join(overlap[:5])}"
            )
    return None
