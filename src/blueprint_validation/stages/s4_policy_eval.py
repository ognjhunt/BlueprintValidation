"""Stage 4: Policy evaluation — baseline vs site-adapted DreamDojo."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.openvla_runner import (
    load_dreamdojo_world_model,
)
from ..evaluation.judge_audit import write_judge_audit_csv
from ..evaluation.task_hints import (
    balance_eval_tasks,
    recommended_rollouts_per_condition,
    tasks_from_task_hints,
)
from ..evaluation.task_start_selector import (
    build_task_start_assignments,
    load_initial_frames_for_assignments,
    load_shared_task_start_manifest,
    save_shared_task_start_manifest,
    shared_manifest_is_compatible,
)
from ..evaluation.vlm_judge import (
    JudgeScore,
    ManipulationJudgeScore,
    score_rollout,
    score_rollout_manipulation,
)
from ..evaluation.rollout_utils import run_rollout_with_adapter
from ..policy_adapters import get_policy_adapter
from .base import PipelineStage

logger = get_logger("stages.s4_policy_eval")


class PolicyEvalStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4_policy_eval"

    @property
    def description(self) -> str:
        return "Evaluate selected policy adapter in baseline vs site-adapted DreamDojo world model"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        eval_dir = work_dir / "policy_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Check for fine-tuned checkpoint produced by Stage 3.
        adapted_dir = None
        prev_stage = previous_results.get("s3_finetune")
        if prev_stage:
            adapted_candidate = (
                prev_stage.outputs.get("adapted_checkpoint_path")
                or prev_stage.outputs.get("lora_weights_path")
            )
            if adapted_candidate:
                adapted_dir = Path(adapted_candidate)
        if adapted_dir is None:
            # Backward-compatible fallbacks.
            for candidate in [
                work_dir / "finetune" / "adapted_checkpoint",
                work_dir / "finetune" / "lora_weights",
            ]:
                if candidate.exists():
                    adapted_dir = candidate
                    break
        if adapted_dir is None:
            adapted_dir = work_dir / "finetune" / "adapted_checkpoint"
        if not adapted_dir.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Adapted checkpoint not found at {adapted_dir}. Run Stage 3 first.",
            )

        # Load render manifest for initial frames
        render_manifest_path = work_dir / "renders" / "render_manifest.json"
        if not render_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Render manifest not found. Run Stage 1 first.",
            )

        render_manifest = read_json(render_manifest_path)

        tasks, hint_count = _build_task_list(config, facility)

        requested_rollouts = int(config.eval_policy.num_rollouts)
        planned_rollouts = recommended_rollouts_per_condition(
            num_unique_tasks=len(tasks),
            requested=requested_rollouts,
            profile="dreamdojo",
        )

        shared_manifest_path = eval_dir / "shared_task_start_manifest.json"
        shared_manifest = load_shared_task_start_manifest(shared_manifest_path)
        reused_shared_manifest = False
        rollout_assignments: List[dict] = []
        if shared_manifest and shared_manifest_is_compatible(
            shared_manifest,
            facility_name=facility.name,
            render_manifest_path=render_manifest_path,
        ):
            rollout_assignments = list(shared_manifest.get("assignments", []))
            reused_shared_manifest = bool(rollout_assignments)

        if not rollout_assignments:
            rollout_assignments = build_task_start_assignments(
                tasks=tasks,
                num_rollouts=planned_rollouts,
                render_manifest=render_manifest,
                task_hints_path=facility.task_hints_path,
            )
            save_shared_task_start_manifest(
                path=shared_manifest_path,
                facility_name=facility.name,
                render_manifest_path=render_manifest_path,
                task_profile="dreamdojo",
                requested_rollouts=requested_rollouts,
                planned_rollouts=planned_rollouts,
                tasks=tasks,
                assignments=rollout_assignments,
            )

        frame_cache = load_initial_frames_for_assignments(rollout_assignments)
        if not frame_cache:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Could not extract initial frames for rollout assignments.",
            )

        num_rollouts = len(rollout_assignments)
        max_steps = config.eval_policy.max_steps_per_rollout

        device = "cuda" if _has_cuda() else "cpu"
        policy_adapter = get_policy_adapter(config.policy_adapter)
        base_model_name, base_checkpoint = policy_adapter.base_model_ref(config.eval_policy)
        adapted_policy_checkpoint = _resolve_adapted_policy_checkpoint(
            previous_results=previous_results,
            work_dir=work_dir,
            policy_adapter=policy_adapter,
        )

        all_scores: List[Dict] = []
        scoring_failures: List[str] = []

        conditions = list(config.eval_policy.conditions)
        for condition in conditions:
            logger.info("Running %s condition rollouts", condition)
            condition_dir = eval_dir / f"{condition}_rollouts"
            condition_dir.mkdir(exist_ok=True)

            # Load policy checkpoint for this condition.
            policy_checkpoint = base_checkpoint
            if condition == "adapted" and adapted_policy_checkpoint is not None:
                policy_checkpoint = adapted_policy_checkpoint
            policy_handle = policy_adapter.load_policy(
                model_name=base_model_name,
                checkpoint_path=policy_checkpoint,
                device=device,
            )

            # Load world model
            adapted = adapted_dir if condition == "adapted" else None
            world_model = load_dreamdojo_world_model(
                checkpoint_path=config.finetune.dreamdojo_checkpoint,
                adapted_checkpoint=adapted,
                device=device,
            )

            for assignment in rollout_assignments:
                rollout_idx = int(assignment.get("rollout_index", 0))
                task = str(assignment.get("task", ""))
                clip_index = int(assignment.get("clip_index", -1))
                init_frame = frame_cache.get(clip_index)
                if init_frame is None:
                    msg = f"Initial frame missing for clip_index={clip_index} task='{task}'"
                    scoring_failures.append(msg)
                    logger.warning(msg)
                    continue
                clip_stub = str(assignment.get("clip_name", f"clip_{clip_index:03d}"))
                clip_name = (
                    f"{condition}_{clip_stub}_{rollout_idx:03d}"
                    .replace("/", "_")
                    .replace(" ", "_")
                )

                rollout = run_rollout_with_adapter(
                    world_model=world_model,
                    policy_adapter=policy_adapter,
                    policy_handle=policy_handle,
                    initial_frame=init_frame,
                    task_prompt=task,
                    max_steps=max_steps,
                    unnorm_key=config.eval_policy.unnorm_key,
                    output_dir=condition_dir,
                    clip_name=clip_name,
                    device=device,
                )

                if not rollout.video_path or not rollout.video_path.exists():
                    msg = f"Rollout video missing for {clip_name}"
                    scoring_failures.append(msg)
                    logger.warning(msg)
                    continue

                try:
                    if _is_manipulation_task(task):
                        score = score_rollout_manipulation(
                            video_path=rollout.video_path,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                    else:
                        score = score_rollout(
                            video_path=rollout.video_path,
                            task_prompt=task,
                            config=config.eval_policy.vlm_judge,
                            facility_description=facility.description,
                        )
                except Exception as e:
                    msg = f"VLM scoring failed for {clip_name}: {e}"
                    logger.warning(msg)
                    scoring_failures.append(msg)
                    score = JudgeScore(0, 0, 0, str(e), "")

                all_scores.append({
                    "condition": condition,
                    "task": task,
                    "rollout_index": rollout_idx,
                    "task_score": score.task_score,
                    "visual_score": score.visual_score,
                    "spatial_score": score.spatial_score,
                    "reasoning": score.reasoning,
                    "video_path": str(rollout.video_path),
                    "num_steps": rollout.num_steps,
                    "action_sequence": getattr(rollout, "action_sequence", []),
                    "start_clip_index": clip_index,
                    "start_clip_name": clip_stub,
                    "start_path_type": str(assignment.get("path_type", "unknown")),
                    "target_instance_id": assignment.get("target_instance_id"),
                    "target_label": assignment.get("target_label"),
                    "is_manipulation_task": _is_manipulation_task(task),
                    "grasp_acquired": (
                        score.grasp_acquired
                        if isinstance(score, ManipulationJudgeScore)
                        else None
                    ),
                    "lifted_clear": (
                        score.lifted_clear
                        if isinstance(score, ManipulationJudgeScore)
                        else None
                    ),
                    "placed_in_target": (
                        score.placed_in_target
                        if isinstance(score, ManipulationJudgeScore)
                        else None
                    ),
                    "stable_after_place": (
                        score.stable_after_place
                        if isinstance(score, ManipulationJudgeScore)
                        else None
                    ),
                })

        # Compute aggregate metrics per condition
        write_json({"scores": all_scores}, eval_dir / "vlm_scores.json")
        audit_csv_path = eval_dir / "judge_audit.csv"
        write_judge_audit_csv(all_scores, audit_csv_path)

        per_condition: Dict[str, Dict] = {}
        for cond in conditions:
            cond_scores = [s for s in all_scores if s["condition"] == cond]
            cond_manip = [s for s in cond_scores if s["is_manipulation_task"]]
            cond_mean = float(np.mean([s["task_score"] for s in cond_scores])) if cond_scores else 0
            per_condition[cond] = {
                "mean_task_score": round(cond_mean, 3),
                "num_rollouts": len(cond_scores),
                "manipulation_success_rate": _manipulation_success_rate(cond_manip),
            }

        # Pairwise comparisons between all condition pairs
        pairwise = _build_pairwise_metrics(all_scores, conditions)

        # Backward-compatible top-level metrics (baseline vs adapted)
        baseline_scores = [s for s in all_scores if s["condition"] == "baseline"]
        adapted_scores = [s for s in all_scores if s["condition"] == "adapted"]
        baseline_mean = per_condition.get("baseline", {}).get("mean_task_score", 0)
        adapted_mean = per_condition.get("adapted", {}).get("mean_task_score", 0)
        improvement = ((adapted_mean - baseline_mean) / max(baseline_mean, 1e-8)) * 100
        absolute_difference = adapted_mean - baseline_mean

        min_len = min(len(baseline_scores), len(adapted_scores))
        wins = sum(
            1 for b, a in zip(baseline_scores[:min_len], adapted_scores[:min_len])
            if a["task_score"] > b["task_score"]
        )
        win_rate = wins / max(min_len, 1)

        p_value = None
        if min_len >= 2:
            try:
                from scipy import stats
                b_vals = [s["task_score"] for s in baseline_scores[:min_len]]
                a_vals = [s["task_score"] for s in adapted_scores[:min_len]]
                _, p_value = stats.ttest_rel(b_vals, a_vals)
                p_value = float(p_value)
            except ImportError:
                logger.warning("scipy not available; skipping p-value computation")

        metrics = {
            "baseline_mean_task_score": round(float(baseline_mean), 3),
            "adapted_mean_task_score": round(float(adapted_mean), 3),
            "improvement_pct": round(float(improvement), 2),
            "absolute_difference": round(float(absolute_difference), 3),
            "win_rate": round(float(win_rate), 3),
            "p_value": round(p_value, 6) if p_value is not None else None,
            "num_rollouts_baseline": len(baseline_scores),
            "num_rollouts_adapted": len(adapted_scores),
            "num_scoring_failures": len(scoring_failures),
            "used_adapted_policy_checkpoint": adapted_policy_checkpoint is not None,
            "adapted_policy_checkpoint": (
                str(adapted_policy_checkpoint) if adapted_policy_checkpoint else None
            ),
            "requested_rollouts_per_condition": requested_rollouts,
            "planned_rollouts_per_condition": planned_rollouts,
            "executed_rollouts_per_condition": num_rollouts,
            "num_unique_task_templates": len(tasks),
            "shared_task_start_manifest": str(shared_manifest_path),
            "shared_task_start_manifest_reused": reused_shared_manifest,
            "per_condition": per_condition,
            "pairwise": pairwise,
            "baseline_manipulation_success_rate": per_condition.get(
                "baseline", {}
            ).get("manipulation_success_rate", 0.0),
            "adapted_manipulation_success_rate": per_condition.get(
                "adapted", {}
            ).get("manipulation_success_rate", 0.0),
            "task_hints_injected": hint_count,
            "judge_audit_csv": str(audit_csv_path),
        }

        write_json(metrics, eval_dir / "policy_eval_report.json")

        return StageResult(
            stage_name=self.name,
            status="success" if all_scores else "failed",
            elapsed_seconds=0,
            outputs={
                "eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "vlm_scores.json"),
                "report_path": str(eval_dir / "policy_eval_report.json"),
                "shared_task_start_manifest": str(shared_manifest_path),
                "judge_audit_csv": str(audit_csv_path),
            },
            metrics=metrics,
            detail="\n".join(scoring_failures[:5]),
        )


def _extract_initial_frames(render_manifest: dict) -> List[np.ndarray]:
    """Extract first frames from rendered video clips."""
    import cv2

    frames = []
    for clip in render_manifest.get("clips", []):
        video_path = Path(clip["video_path"])
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def _build_task_list(config: ValidationConfig, facility: FacilityConfig) -> tuple[List[str], int]:
    tasks = list(config.eval_policy.tasks or [])
    for task in config.eval_policy.manipulation_tasks:
        if task not in tasks:
            tasks.append(task)

    hint_tasks: List[str] = []
    if facility.task_hints_path is not None:
        try:
            hint_tasks = tasks_from_task_hints(
                facility.task_hints_path,
                profile="dreamdojo",
            )
        except Exception as exc:
            logger.warning("Failed loading task hints from %s: %s", facility.task_hints_path, exc)
    for task in hint_tasks:
        if task not in tasks:
            tasks.append(task)

    if not tasks:
        tasks = [
            "Navigate forward through the corridor",
            "Turn left at the intersection",
            "Approach the nearest obstacle",
        ]
    return balance_eval_tasks(tasks, profile="dreamdojo"), len(hint_tasks)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_rollout_plan(tasks: List[str], num_rollouts: int) -> List[str]:
    """Build exactly num_rollouts task prompts in round-robin order."""
    if num_rollouts <= 0:
        return []
    if not tasks:
        return []
    return [tasks[i % len(tasks)] for i in range(num_rollouts)]


def _is_manipulation_task(task: str) -> bool:
    lowered = task.lower()
    keywords = ["pick", "grasp", "lift", "place", "stack", "regrasp", "tote", "bin"]
    return any(k in lowered for k in keywords)


def _manipulation_success_rate(scores: List[Dict]) -> float:
    if not scores:
        return 0.0
    successes = 0
    for s in scores:
        has_flags = s.get("grasp_acquired") is not None
        if has_flags:
            if (
                bool(s.get("grasp_acquired"))
                and bool(s.get("lifted_clear"))
                and bool(s.get("placed_in_target"))
            ):
                successes += 1
        elif float(s.get("task_score", 0.0)) >= 7.0:
            successes += 1
    return round(successes / len(scores), 3)


def _resolve_adapted_policy_checkpoint(
    previous_results: Dict[str, StageResult],
    work_dir: Path,
    policy_adapter=None,
) -> Path | None:
    if policy_adapter is None:
        from ..training.openvla_finetune import resolve_latest_openvla_checkpoint

        class _LegacyResolver:
            def resolve_latest_checkpoint(self, run_root: Path):
                return resolve_latest_openvla_checkpoint(run_root)

        policy_adapter = _LegacyResolver()

    prev_rl = previous_results.get("s3c_policy_rl_loop")
    if prev_rl:
        candidate = (
            prev_rl.outputs.get("adapted_policy_checkpoint_rl")
            or prev_rl.outputs.get("adapted_openvla_checkpoint_rl")
        )
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    prev = previous_results.get("s3b_policy_finetune")
    if prev:
        candidate = (
            prev.outputs.get("adapted_policy_checkpoint")
            or prev.outputs.get("adapted_openvla_checkpoint")
        )
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    rl_root = work_dir / "policy_rl_loop"
    if rl_root.exists():
        for iter_dir in sorted(rl_root.glob("iter_*"), reverse=True):
            candidate = policy_adapter.resolve_latest_checkpoint(
                iter_dir / "policy_refine" / "runs"
            )
            if candidate is not None:
                return candidate
    fallback = policy_adapter.resolve_latest_checkpoint(work_dir / "policy_finetune" / "runs")
    if fallback is not None:
        return fallback
    return None


def _build_pairwise_metrics(all_scores: List[Dict], conditions: List[str]) -> Dict:
    """Compute improvement, win rate, and p-value for each pair of conditions."""
    pairwise = {}
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            s1 = [s for s in all_scores if s["condition"] == c1]
            s2 = [s for s in all_scores if s["condition"] == c2]
            if not s1 or not s2:
                continue
            mean1 = float(np.mean([s["task_score"] for s in s1]))
            mean2 = float(np.mean([s["task_score"] for s in s2]))
            improvement = ((mean2 - mean1) / max(mean1, 1e-8)) * 100
            abs_diff = mean2 - mean1
            min_len = min(len(s1), len(s2))
            wins = sum(
                1 for a, b in zip(s1[:min_len], s2[:min_len])
                if b["task_score"] > a["task_score"]
            )
            win_rate = wins / max(min_len, 1)
            p_value = None
            if min_len >= 2:
                try:
                    from scipy import stats
                    v1 = [s["task_score"] for s in s1[:min_len]]
                    v2 = [s["task_score"] for s in s2[:min_len]]
                    _, p_value = stats.ttest_rel(v1, v2)
                    p_value = float(p_value)
                except ImportError:
                    pass
            pairwise[f"{c1}_vs_{c2}"] = {
                f"{c1}_mean": round(mean1, 3),
                f"{c2}_mean": round(mean2, 3),
                "improvement_pct": round(improvement, 2),
                "absolute_difference": round(abs_diff, 3),
                "win_rate": round(win_rate, 3),
                "p_value": round(p_value, 6) if p_value is not None else None,
            }
    return pairwise
