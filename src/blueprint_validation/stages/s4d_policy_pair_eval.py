"""Stage 4d: Heldout paired evaluation of trained policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.claim_protocol import (
    build_task_specs,
    checkpoint_content_hash,
    claim_protocol_enabled,
)
from ..evaluation.claim_stats import bootstrap_site_vs_baseline, success_rate
from ..evaluation.openvla_runner import load_dreamdojo_world_model
from ..evaluation.rollout_utils import run_rollout_with_adapter
from ..evaluation.task_success import evaluate_task_success
from ..evaluation.video_orientation import (
    normalize_video_orientation_fix,
    transform_video_orientation,
)
from ..evaluation.vlm_judge import score_rollout, score_rollout_manipulation
from ..policy_adapters import get_policy_adapter
from .base import PipelineStage

logger = get_logger("stages.s4d_policy_pair_eval")


class PolicyPairEvalStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4d_policy_pair_eval"

    @property
    def description(self) -> str:
        return "Compare policy_base vs policy_site on heldout rollouts in same world model"

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
        claim_mode = (config.eval_policy.mode or "claim").strip().lower() == "claim"
        if claim_mode and config.policy_adapter.name.strip().lower() != "openvla_oft":
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Claim mode only supports policy_adapter.name=openvla_oft.",
            )
        if not config.policy_compare.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="policy_compare.enabled=false",
            )

        dataset_root = config.rollout_dataset.export_dir / work_dir.name
        pair_summary_path = work_dir / "policy_pair_train" / "policy_pair_train_summary.json"
        heldout_path = dataset_root / "adapted" / "heldout" / "episodes.jsonl"
        if not pair_summary_path.exists() or not heldout_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Missing policy_pair_train summary or heldout dataset. Run Stage 4b and 4c first.",
            )

        pair_summary = read_json(pair_summary_path)
        if claim_protocol_enabled(config):
            return _run_fixed_world_claim_eval(
                config=config,
                facility=facility,
                work_dir=work_dir,
                previous_results=previous_results,
                dataset_root=dataset_root,
                pair_summary=pair_summary,
                heldout_path=heldout_path,
            )
        base_ckpt = Path(pair_summary["policy_base"]["adapted_checkpoint_path"])
        site_ckpt = Path(pair_summary["policy_site"]["adapted_checkpoint_path"])
        if not base_ckpt.exists() or not site_ckpt.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Trained policy checkpoints missing.",
            )

        episodes = _load_heldout_episodes(heldout_path)
        if config.policy_compare.heldout_tasks:
            allow = {t.strip() for t in config.policy_compare.heldout_tasks if t.strip()}
            episodes = [ep for ep in episodes if ep["task"] in allow]
        if not episodes:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "No heldout episodes available for pair evaluation. "
                    "Check rollout dataset export and policy_compare.heldout_tasks."
                ),
            )
        episodes = _sample_episodes(
            episodes,
            num_rollouts=config.policy_compare.heldout_num_rollouts,
            seed=config.policy_compare.heldout_seed,
        )

        eval_world_checkpoint = _resolve_eval_world_checkpoint(config, work_dir)
        if config.policy_compare.eval_world_model == "adapted" and eval_world_checkpoint is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Requested adapted eval world model, but adapted checkpoint was not found.",
            )
        device = "cuda" if _has_cuda() else "cpu"
        world_model = load_dreamdojo_world_model(
            checkpoint_path=config.finetune.dreamdojo_checkpoint,
            adapted_checkpoint=eval_world_checkpoint,
            configured_experiment=(
                config.finetune.eval_world_experiment or config.finetune.experiment_config
            ),
            dreamdojo_repo=config.finetune.dreamdojo_repo,
            device=device,
        )
        adapter = get_policy_adapter(config.policy_adapter)
        base_model_name, _ = adapter.base_model_ref(config.eval_policy)
        base_handle = adapter.load_policy(
            model_name=base_model_name,
            checkpoint_path=base_ckpt,
            device=device,
        )
        site_handle = adapter.load_policy(
            model_name=base_model_name,
            checkpoint_path=site_ckpt,
            device=device,
        )

        eval_dir = work_dir / "policy_pair_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        score_rows: List[dict] = []

        for i, ep in enumerate(episodes):
            init_frame = _read_rgb_image(Path(ep["init_frame_path"]))
            task = ep["task"]
            for policy_name, handle in [("policy_base", base_handle), ("policy_site", site_handle)]:
                video_dir = eval_dir / f"{policy_name}_rollouts"
                video_dir.mkdir(parents=True, exist_ok=True)
                rollout_result = run_rollout_with_adapter(
                    world_model=world_model,
                    policy_adapter=adapter,
                    policy_handle=handle,
                    initial_frame=init_frame,
                    task_prompt=task,
                    max_steps=config.eval_policy.max_steps_per_rollout,
                    unnorm_key=config.eval_policy.unnorm_key,
                    output_dir=video_dir,
                    clip_name=f"heldout_{i:03d}",
                    device=device,
                )
                rollout_video = rollout_result.video_path
                actions = rollout_result.action_sequence
                num_steps = rollout_result.num_steps
                _orient_mode_4d = normalize_video_orientation_fix(
                    str(getattr(facility, "video_orientation_fix", "none"))
                )
                _oriented_4d_path = None
                if _orient_mode_4d != "none":
                    _oriented_4d_path = rollout_video.with_name(
                        rollout_video.stem + f"_oriented_{_orient_mode_4d}.mp4"
                    )
                    try:
                        transform_video_orientation(
                            input_path=rollout_video,
                            output_path=_oriented_4d_path,
                            orientation_fix=_orient_mode_4d,
                            force_grayscale=False,
                        )
                        _rollout_video_for_scoring = _oriented_4d_path
                    except Exception as _oe:
                        logger.warning(
                            "orientation transform failed for %s: %s", rollout_video, _oe
                        )
                        _rollout_video_for_scoring = rollout_video
                        _oriented_4d_path = None
                else:
                    _rollout_video_for_scoring = rollout_video
                try:
                    if _is_manipulation_task(task, config):
                        score = score_rollout_manipulation(
                            video_path=_rollout_video_for_scoring,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                        success = _manip_success(score, config)
                        manip_success = success
                    else:
                        score = score_rollout(
                            video_path=_rollout_video_for_scoring,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                        success = score.task_score >= config.policy_compare.task_score_success_threshold
                        manip_success = None
                finally:
                    if _oriented_4d_path is not None and _oriented_4d_path.exists():
                        try:
                            _oriented_4d_path.unlink()
                        except Exception:
                            pass

                score_rows.append(
                    {
                        "episode_id": ep["episode_id"],
                        "policy": policy_name,
                        "task": task,
                        "task_score": score.task_score,
                        "visual_score": score.visual_score,
                        "spatial_score": score.spatial_score,
                        "success": success,
                        "is_manipulation_task": _is_manipulation_task(task, config),
                        "manipulation_success": manip_success,
                        "num_steps": num_steps,
                        "video_path": str(rollout_video),
                        "action_sequence": actions,
                    }
                )

        write_json({"scores": score_rows}, eval_dir / "pair_scores.json")
        metrics = _compute_pair_metrics(score_rows)
        claim_failure_reasons: List[str] = []
        task_abs_diff = float(metrics.get("task_score_absolute_difference", 0.0) or 0.0)
        if task_abs_diff < float(config.eval_policy.min_absolute_difference):
            claim_failure_reasons.append(
                "Absolute task-score difference below threshold: "
                f"{task_abs_diff:.3f} < {config.eval_policy.min_absolute_difference:.3f}"
            )
        base_manip = metrics.get("policy_base_manipulation_success_rate")
        site_manip = metrics.get("policy_site_manipulation_success_rate")
        manip_delta_pp = None
        if base_manip is not None and site_manip is not None:
            manip_delta_pp = (float(site_manip) - float(base_manip)) * 100.0
            if manip_delta_pp < float(config.eval_policy.min_manip_success_delta_pp):
                claim_failure_reasons.append(
                    "Manipulation success delta below threshold: "
                    f"{manip_delta_pp:.2f}pp < {config.eval_policy.min_manip_success_delta_pp:.2f}pp"
                )
        claim_passed = len(claim_failure_reasons) == 0
        metrics["claim_mode"] = claim_mode
        metrics["claim_passed"] = claim_passed
        metrics["claim_failure_reasons"] = claim_failure_reasons
        metrics["manipulation_success_delta_pp"] = (
            round(float(manip_delta_pp), 6) if manip_delta_pp is not None else None
        )
        write_json(metrics, eval_dir / "pair_eval_report.json")

        return StageResult(
            stage_name=self.name,
            status=(
                "success"
                if score_rows and (not claim_mode or claim_passed)
                else "failed"
            ),
            elapsed_seconds=0,
            outputs={
                "pair_eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "pair_scores.json"),
                "report_path": str(eval_dir / "pair_eval_report.json"),
            },
            metrics=metrics,
        )


def _run_fixed_world_claim_eval(
    *,
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    previous_results: Dict[str, StageResult],
    dataset_root: Path,
    pair_summary: dict,
    heldout_path: Path,
) -> StageResult:
    for moving_stage in ("s3c_policy_rl_loop", "s3d_wm_refresh_loop"):
        stage_result = previous_results.get(moving_stage)
        if stage_result and stage_result.status == "success":
            return StageResult(
                stage_name="s4d_policy_pair_eval",
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Fixed-world claim protocol rejects refreshed-world artifacts from "
                    f"{moving_stage}."
                ),
            )

    policy_eval_dir = work_dir / "policy_eval"
    claim_manifest_path = policy_eval_dir / "claim_manifest.json"
    claim_split_path = policy_eval_dir / "claim_split_manifest.json"
    task_specs_path = policy_eval_dir / "task_specs.json"
    if not claim_manifest_path.exists() or not claim_split_path.exists() or not task_specs_path.exists():
        return StageResult(
            stage_name="s4d_policy_pair_eval",
            status="failed",
            elapsed_seconds=0,
            detail=(
                "Missing fixed-world claim artifacts. Run Stage 4 with "
                "eval_policy.claim_protocol=fixed_same_facility_uplift first."
            ),
        )
    claim_manifest = read_json(claim_manifest_path)
    claim_split = read_json(claim_split_path)
    task_specs = read_json(task_specs_path)
    task_specs_by_id = {str(spec.get("task_spec_id", "")): spec for spec in task_specs}
    task_specs_by_prompt = {str(spec.get("task_prompt", "")): spec for spec in task_specs}

    episodes = _load_heldout_episodes(heldout_path)
    eval_ids = {str(v) for v in claim_split.get("eval_eval_cell_ids", [])}
    if eval_ids:
        episodes = [ep for ep in episodes if str(ep.get("eval_cell_id", "")) in eval_ids]
    if config.policy_compare.heldout_tasks:
        allow = {t.strip() for t in config.policy_compare.heldout_tasks if t.strip()}
        episodes = [ep for ep in episodes if ep["task"] in allow]
    if not episodes:
        return StageResult(
            stage_name="s4d_policy_pair_eval",
            status="failed",
            elapsed_seconds=0,
            detail="No heldout claim episodes available for fixed-world evaluation.",
        )

    eval_world_checkpoint = _resolve_eval_world_checkpoint(config, work_dir)
    if eval_world_checkpoint is None or not eval_world_checkpoint.exists():
        return StageResult(
            stage_name="s4d_policy_pair_eval",
            status="failed",
            elapsed_seconds=0,
            detail="Fixed-world claim protocol requires an adapted eval world checkpoint.",
        )
    world_snapshot_hash = checkpoint_content_hash(eval_world_checkpoint)
    if world_snapshot_hash != str(claim_manifest.get("world_snapshot_hash", "")):
        return StageResult(
            stage_name="s4d_policy_pair_eval",
            status="failed",
            elapsed_seconds=0,
            detail=(
                "World snapshot hash drift detected between claim manifest and S4d eval world. "
                "Refusing to continue."
            ),
        )

    device = "cuda" if _has_cuda() else "cpu"
    world_model = load_dreamdojo_world_model(
        checkpoint_path=config.finetune.dreamdojo_checkpoint,
        adapted_checkpoint=eval_world_checkpoint,
        configured_experiment=(
            config.finetune.eval_world_experiment or config.finetune.experiment_config
        ),
        dreamdojo_repo=config.finetune.dreamdojo_repo,
        device=device,
    )
    adapter = get_policy_adapter(config.policy_adapter)
    base_model_name, base_checkpoint = adapter.base_model_ref(config.eval_policy)

    expected_arms = [str(v) for v in list(config.policy_compare.control_arms or [])]
    arm_entries = _claim_arm_entries(
        pair_summary=pair_summary,
        base_checkpoint=base_checkpoint,
        expected_arms=expected_arms,
    )
    missing_arms = [arm for arm in expected_arms if arm not in {entry["arm"] for entry in arm_entries}]
    if missing_arms:
        return StageResult(
            stage_name="s4d_policy_pair_eval",
            status="failed",
            elapsed_seconds=0,
            detail=f"Missing control arms for claim protocol: {', '.join(sorted(missing_arms))}",
        )

    handles: Dict[tuple[str, int | None], object] = {}
    for entry in arm_entries:
        handles[(str(entry["arm"]), entry.get("seed"))] = adapter.load_policy(
            model_name=base_model_name,
            checkpoint_path=entry.get("checkpoint_path"),
            device=device,
        )

    eval_dir = work_dir / "policy_pair_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    score_rows: List[dict] = []
    state_failures: List[str] = []

    for i, ep in enumerate(episodes):
        init_frame = _read_rgb_image(Path(ep["init_frame_path"]))
        task = ep["task"]
        task_spec = task_specs_by_id.get(str(ep.get("task_spec_id", ""))) or task_specs_by_prompt.get(task)
        if task_spec is None:
            return StageResult(
                stage_name="s4d_policy_pair_eval",
                status="failed",
                elapsed_seconds=0,
                detail=f"Missing task spec for heldout episode task='{task}'.",
            )
        for entry in arm_entries:
            arm = str(entry["arm"])
            seed = entry.get("seed")
            policy_label = str(entry["policy_label"])
            handle = handles[(arm, seed)]
            video_dir = eval_dir / f"{policy_label}_rollouts"
            video_dir.mkdir(parents=True, exist_ok=True)
            rollout_result = run_rollout_with_adapter(
                world_model=world_model,
                policy_adapter=adapter,
                policy_handle=handle,
                initial_frame=init_frame,
                task_prompt=task,
                max_steps=config.eval_policy.max_steps_per_rollout,
                unnorm_key=config.eval_policy.unnorm_key,
                output_dir=video_dir,
                clip_name=f"{policy_label}_{i:03d}",
                device=device,
            )
            score = _score_rollout_video(
                facility=facility,
                config=config,
                video_path=rollout_result.video_path,
                task=task,
            )
            row = {
                "episode_id": ep["episode_id"],
                "eval_cell_id": ep.get("eval_cell_id"),
                "task_spec_id": ep.get("task_spec_id"),
                "start_region_id": ep.get("start_region_id"),
                "world_snapshot_hash": world_snapshot_hash,
                "policy": policy_label,
                "arm": arm,
                "seed": seed,
                "task": task,
                "task_score": score.task_score,
                "visual_score": score.visual_score,
                "spatial_score": score.spatial_score,
                "num_steps": rollout_result.num_steps,
                "video_path": str(rollout_result.video_path),
                "action_sequence": rollout_result.action_sequence,
                "state_trace": list(getattr(rollout_result, "state_trace", []) or []),
                "is_manipulation_task": _is_manipulation_task(task, config),
                "grasp_acquired": getattr(score, "grasp_acquired", None),
                "lifted_clear": getattr(score, "lifted_clear", None),
                "placed_in_target": getattr(score, "placed_in_target", None),
                "stable_after_place": getattr(score, "stable_after_place", None),
            }
            row.update(
                evaluate_task_success(
                    task_spec=task_spec,
                    rollout_row=row,
                    state_trace=row["state_trace"],
                )
            )
            if not bool(row.get("task_success_available", False)):
                state_failures.append(
                    f"Missing deterministic task-state evidence arm={arm} seed={seed} eval_cell={ep.get('eval_cell_id')}"
                )
            row["success"] = bool(row.get("task_success", False))
            row["manipulation_success"] = (
                bool(row.get("task_success", False)) if row["is_manipulation_task"] else None
            )
            score_rows.append(row)

    write_json({"scores": score_rows}, eval_dir / "pair_scores.json")
    legacy_rows = [row for row in score_rows if row["policy"] in {"policy_base", "policy_site"}]
    pair_metrics = _compute_pair_metrics(legacy_rows) if legacy_rows else {"num_pairs": 0}
    write_json(pair_metrics, eval_dir / "pair_eval_report.json")

    baseline_rows = [row for row in score_rows if row["arm"] == "frozen_baseline"]
    site_rows_by_seed = {
        int(seed): [row for row in score_rows if row["arm"] == "site_trained" and int(row.get("seed", -1)) == int(seed)]
        for seed in [int(v) for v in list(config.eval_policy.claim_replication.training_seeds)]
    }
    generic_rows_by_seed = {
        int(seed): [row for row in score_rows if row["arm"] == "generic_control" and int(row.get("seed", -1)) == int(seed)]
        for seed in [int(v) for v in list(config.eval_policy.claim_replication.training_seeds)]
    }
    bootstrap = bootstrap_site_vs_baseline(
        baseline_rows=baseline_rows,
        site_rows_by_seed=site_rows_by_seed,
        bootstrap_seed=int(config.policy_compare.heldout_seed),
    )
    required_positive = max(1, len(list(config.eval_policy.claim_replication.training_seeds)) - 1)
    claim_failure_reasons: List[str] = []
    if state_failures:
        claim_failure_reasons.append(
            f"Deterministic primary endpoint unavailable for {len(state_failures)} scored rows."
        )
    if bootstrap.get("num_common_eval_cells", 0) <= 0:
        claim_failure_reasons.append("No common eval cells across frozen baseline and site-trained seeds.")
    if bootstrap.get("ci_low_pp") is None or float(bootstrap["ci_low_pp"]) <= 0.0:
        claim_failure_reasons.append("Lower 95% CI bound for site-trained uplift is not > 0.")
    if bootstrap.get("mean_lift_pp") is None or float(bootstrap["mean_lift_pp"]) < float(
        config.eval_policy.min_practical_success_lift_pp
    ):
        claim_failure_reasons.append(
            "Mean success-rate uplift below practical threshold: "
            f"{bootstrap.get('mean_lift_pp')}pp < {config.eval_policy.min_practical_success_lift_pp}pp"
        )
    if int(bootstrap.get("positive_seed_count", 0)) < int(required_positive):
        claim_failure_reasons.append(
            f"Positive uplift direction not observed in enough seeds: "
            f"{bootstrap.get('positive_seed_count', 0)} < {required_positive}."
        )

    arm_summary = {
        "frozen_baseline": {
            "success_rate": round(success_rate(baseline_rows), 6),
            "num_rows": len(baseline_rows),
        },
        "generic_control": {
            "per_seed_success_rate": {
                int(seed): round(success_rate(rows), 6) for seed, rows in generic_rows_by_seed.items()
            },
        },
        "site_trained": {
            "per_seed_success_rate": {
                int(seed): round(success_rate(rows), 6) for seed, rows in site_rows_by_seed.items()
            },
        },
    }
    claim_metrics = {
        "claim_protocol": "fixed_same_facility_uplift",
        "primary_endpoint": "task_success",
        "world_snapshot_hash": world_snapshot_hash,
        "claim_manifest_path": str(claim_manifest_path),
        "claim_split_manifest_path": str(claim_split_path),
        "task_specs_path": str(task_specs_path),
        "num_eval_cells": len(eval_ids) if eval_ids else len({str(ep.get("eval_cell_id", "")) for ep in episodes}),
        "required_positive_seed_count": int(required_positive),
        "min_practical_success_lift_pp": float(config.eval_policy.min_practical_success_lift_pp),
        "bootstrap_site_vs_frozen": bootstrap,
        "arm_summary": arm_summary,
        "claim_passed": len(claim_failure_reasons) == 0,
        "claim_failure_reasons": claim_failure_reasons,
        "num_state_failures": len(state_failures),
    }
    write_json(claim_metrics, eval_dir / "claim_eval_report.json")
    return StageResult(
        stage_name="s4d_policy_pair_eval",
        status="success" if score_rows and not claim_failure_reasons else "failed",
        elapsed_seconds=0,
        outputs={
            "pair_eval_dir": str(eval_dir),
            "scores_path": str(eval_dir / "pair_scores.json"),
            "report_path": str(eval_dir / "pair_eval_report.json"),
            "claim_report_path": str(eval_dir / "claim_eval_report.json"),
        },
        metrics=claim_metrics,
        detail=" ".join(claim_failure_reasons).strip(),
    )


def _load_heldout_episodes(path: Path) -> List[dict]:
    episodes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        steps = payload.get("steps", [])
        if not steps:
            continue
        init_path = steps[0]["observation"]["image_path"]
        episodes.append(
            {
                "episode_id": payload["episode_id"],
                "task": steps[0].get("language_instruction", payload.get("task", "")),
                "init_frame_path": init_path,
                "eval_cell_id": payload.get("eval_cell_id", ""),
                "task_spec_id": payload.get("task_spec_id", ""),
                "start_region_id": payload.get("start_region_id", ""),
                "world_snapshot_hash": payload.get("world_snapshot_hash", ""),
            }
        )
    return episodes


def _claim_arm_entries(
    *,
    pair_summary: dict,
    base_checkpoint: Path | None,
    expected_arms: List[str],
) -> List[dict]:
    entries: List[dict] = []
    if "frozen_baseline" in expected_arms:
        entries.append(
            {
                "arm": "frozen_baseline",
                "seed": None,
                "policy_label": "frozen_baseline",
                "checkpoint_path": base_checkpoint if base_checkpoint and base_checkpoint.exists() else None,
            }
        )

    replicates = pair_summary.get("replicates", {}) if isinstance(pair_summary, dict) else {}
    for arm_name, policy_label in (
        ("generic_control", "policy_base"),
        ("site_trained", "policy_site"),
    ):
        if arm_name not in expected_arms:
            continue
        rows = replicates.get(arm_name)
        if isinstance(rows, list) and rows:
            for row in rows:
                ckpt = Path(str(row.get("adapted_checkpoint_path", "") or ""))
                if not ckpt.exists():
                    continue
                entries.append(
                    {
                        "arm": arm_name,
                        "seed": int(row.get("seed", 0)),
                        "policy_label": policy_label,
                        "checkpoint_path": ckpt,
                    }
                )
        else:
            legacy_key = "policy_base" if arm_name == "generic_control" else "policy_site"
            legacy = pair_summary.get(legacy_key, {})
            ckpt = Path(str(legacy.get("adapted_checkpoint_path", "") or ""))
            if ckpt.exists():
                entries.append(
                    {
                        "arm": arm_name,
                        "seed": 0,
                        "policy_label": policy_label,
                        "checkpoint_path": ckpt,
                    }
                )
    return entries


def _score_rollout_video(
    *,
    facility: FacilityConfig,
    config: ValidationConfig,
    video_path: Path,
    task: str,
):
    orient_mode = normalize_video_orientation_fix(str(getattr(facility, "video_orientation_fix", "none")))
    oriented_path = None
    try:
        path_for_scoring = video_path
        if orient_mode != "none":
            oriented_path = video_path.with_name(video_path.stem + f"_oriented_{orient_mode}.mp4")
            try:
                transform_video_orientation(
                    input_path=video_path,
                    output_path=oriented_path,
                    orientation_fix=orient_mode,
                    force_grayscale=False,
                )
                path_for_scoring = oriented_path
            except Exception as exc:
                logger.warning("orientation transform failed for %s: %s", video_path, exc)
                path_for_scoring = video_path
                oriented_path = None
        if _is_manipulation_task(task, config):
            return score_rollout_manipulation(
                video_path=path_for_scoring,
                task_prompt=task,
                config=config.eval_policy.vlm_judge,
                facility_description=facility.description,
            )
        return score_rollout(
            video_path=path_for_scoring,
            task_prompt=task,
            config=config.eval_policy.vlm_judge,
            facility_description=facility.description,
        )
    finally:
        if oriented_path is not None and oriented_path.exists():
            try:
                oriented_path.unlink()
            except Exception:
                pass


def _sample_episodes(episodes: List[dict], num_rollouts: int, seed: int) -> List[dict]:
    if len(episodes) <= num_rollouts:
        return episodes
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(episodes), size=num_rollouts, replace=False)
    return [episodes[i] for i in sorted(idx.tolist())]


def _resolve_eval_world_checkpoint(config: ValidationConfig, work_dir: Path) -> Path | None:
    if config.policy_compare.eval_world_model == "baseline":
        return None
    candidates = [
        work_dir / "finetune" / "adapted_checkpoint",
        work_dir / "finetune" / "lora_weights",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_rgb_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read heldout frame image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _is_manipulation_task(task: str, config: ValidationConfig) -> bool:
    text = task.lower()
    return any(k.lower() in text for k in config.policy_compare.manipulation_task_keywords)


def _manip_success(score, config: ValidationConfig) -> bool:
    grasp = bool(getattr(score, "grasp_acquired", False))
    lifted = bool(getattr(score, "lifted_clear", False))
    placed = bool(getattr(score, "placed_in_target", False))
    stable = bool(getattr(score, "stable_after_place", True))
    if config.policy_compare.require_grasp_for_manipulation and not grasp:
        return False
    if config.policy_compare.require_lift_for_manipulation and not lifted:
        return False
    if config.policy_compare.require_place_for_manipulation and not placed:
        return False
    return stable


def _compute_pair_metrics(rows: List[dict]) -> dict:
    base = [r for r in rows if r["policy"] == "policy_base"]
    site = [r for r in rows if r["policy"] == "policy_site"]
    has_explicit_pairing = any(str(row.get("eval_cell_id", "")).strip() for row in rows)
    if has_explicit_pairing:
        grouped: Dict[str, Dict[str, dict]] = {}
        for row in rows:
            policy = str(row.get("policy", "")).strip()
            if policy not in {"policy_base", "policy_site"}:
                continue
            grouped.setdefault(str(row.get("eval_cell_id", "")).strip(), {})[policy] = row
        paired = [
            (payload["policy_base"], payload["policy_site"])
            for payload in grouped.values()
            if "policy_base" in payload and "policy_site" in payload
        ]
    else:
        paired = list(zip(base, site))
    if not paired:
        return {"num_pairs": 0}

    base_scores = [p[0]["task_score"] for p in paired]
    site_scores = [p[1]["task_score"] for p in paired]
    base_success = [1.0 if p[0]["success"] else 0.0 for p in paired]
    site_success = [1.0 if p[1]["success"] else 0.0 for p in paired]
    wins = sum(1 for b, s in zip(base_scores, site_scores) if s > b)

    p_value = None
    if len(base_scores) >= 2:
        try:
            from scipy import stats

            _, p_value = stats.ttest_rel(base_scores, site_scores)
            p_value = float(p_value)
        except Exception:
            p_value = None

    base_manip = [
        p[0]["manipulation_success"] for p in paired if p[0]["manipulation_success"] is not None
    ]
    site_manip = [
        p[1]["manipulation_success"] for p in paired if p[1]["manipulation_success"] is not None
    ]
    base_manip_rate = float(np.mean(base_manip)) if base_manip else None
    site_manip_rate = float(np.mean(site_manip)) if site_manip else None

    return {
        "num_pairs": len(paired),
        "policy_base_mean_task_score": round(float(np.mean(base_scores)), 3),
        "policy_site_mean_task_score": round(float(np.mean(site_scores)), 3),
        "task_score_improvement_pct": round(
            (
                (float(np.mean(site_scores)) - float(np.mean(base_scores)))
                / max(float(np.mean(base_scores)), 1e-8)
            )
            * 100.0,
            2,
        ),
        "task_score_absolute_difference": round(
            float(np.mean(site_scores)) - float(np.mean(base_scores)), 3
        ),
        "policy_base_success_rate": round(float(np.mean(base_success)), 3),
        "policy_site_success_rate": round(float(np.mean(site_success)), 3),
        "win_rate_site_over_base": round(wins / len(paired), 3),
        "p_value_task_score": round(p_value, 6) if p_value is not None else None,
        "policy_base_manipulation_success_rate": round(base_manip_rate, 3)
        if base_manip_rate is not None
        else None,
        "policy_site_manipulation_success_rate": round(site_manip_rate, 3)
        if site_manip_rate is not None
        else None,
    }


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
