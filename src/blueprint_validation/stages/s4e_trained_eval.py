"""Stage 4e: Evaluate trained policy adapter in the adapted world model."""

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

logger = get_logger("stages.s4e_trained_eval")


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


def _build_rollout_plan(tasks: List[str], num_rollouts: int) -> List[str]:
    """Backward-compatible helper retained for existing tests and callers."""
    if num_rollouts <= 0 or not tasks:
        return []
    return [tasks[i % len(tasks)] for i in range(num_rollouts)]


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class TrainedPolicyEvalStage(PipelineStage):
    """Evaluate the trained policy against frozen baselines.

    Runs the ``trained`` condition: fine-tuned selected policy adapter in the adapted world
    model.  Then merges with Stage 4 scores to produce pairwise comparisons
    (baseline-vs-trained, adapted-vs-trained).
    """

    @property
    def name(self) -> str:
        return "s4e_trained_eval"

    @property
    def description(self) -> str:
        return "Evaluate trained policy adapter in site-adapted DreamDojo world model"

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
        # Only run if either S3b or S3c produced a trained checkpoint.
        policy_adapter = get_policy_adapter(config.policy_adapter)
        trained_checkpoint = _resolve_trained_checkpoint(
            previous_results=previous_results,
            work_dir=work_dir,
            policy_adapter=policy_adapter,
        )
        if trained_checkpoint is None:
            if _has_successful_training_stage(previous_results):
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail="Trained checkpoint not found from S3c/S3b outputs.",
                )
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="Neither S3c nor S3b produced a trained checkpoint. Skipping trained eval.",
            )

        # Need adapted DreamDojo checkpoint from Stage 3
        adapted_dir = None
        prev_s3 = previous_results.get("s3_finetune")
        if prev_s3:
            adapted_candidate = prev_s3.outputs.get(
                "adapted_checkpoint_path"
            ) or prev_s3.outputs.get("lora_weights_path")
            if adapted_candidate:
                adapted_dir = Path(adapted_candidate)
        if adapted_dir is None:
            for candidate in [
                work_dir / "finetune" / "adapted_checkpoint",
                work_dir / "finetune" / "lora_weights",
            ]:
                if candidate.exists():
                    adapted_dir = candidate
                    break
        if adapted_dir is None or not adapted_dir.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Adapted DreamDojo checkpoint not found. Run Stage 3 first.",
            )

        # Need initial frames from Stage 1
        render_manifest_path = work_dir / "renders" / "render_manifest.json"
        if not render_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Render manifest not found. Run Stage 1 first.",
            )

        render_manifest = read_json(render_manifest_path)

        # Build task list (merge config + task hints)
        tasks, hint_count = _build_task_list(config, facility)

        requested_rollouts = int(config.eval_policy.num_rollouts)
        planned_rollouts = recommended_rollouts_per_condition(
            num_unique_tasks=len(tasks),
            requested=requested_rollouts,
            profile="policy",
        )
        max_steps = config.eval_policy.max_steps_per_rollout
        device = "cuda" if _has_cuda() else "cpu"

        eval_dir = work_dir / "trained_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        condition_dir = eval_dir / "trained_rollouts"
        condition_dir.mkdir(exist_ok=True)

        shared_manifest_path = work_dir / "policy_eval" / "shared_task_start_manifest.json"
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
                task_profile="policy",
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
        base_model_name, _ = policy_adapter.base_model_ref(config.eval_policy)
        policy_handle = policy_adapter.load_policy(
            model_name=base_model_name,
            checkpoint_path=trained_checkpoint,
            device=device,
        )

        # Load adapted world model (always use adapted for trained eval)
        world_model = load_dreamdojo_world_model(
            checkpoint_path=config.finetune.dreamdojo_checkpoint,
            adapted_checkpoint=adapted_dir,
            configured_experiment=(
                config.finetune.eval_world_experiment or config.finetune.experiment_config
            ),
            dreamdojo_repo=config.finetune.dreamdojo_repo,
            device=device,
        )

        trained_scores: List[Dict] = []
        scoring_failures: List[str] = []

        for assignment in rollout_assignments:
            rollout_idx = int(assignment.get("rollout_index", 0))
            task = str(assignment.get("task", ""))
            clip_index = int(assignment.get("clip_index", -1))
            init_frame = frame_cache.get(clip_index)
            if init_frame is None:
                scoring_failures.append(
                    f"Initial frame missing for clip_index={clip_index} task='{task}'"
                )
                continue
            clip_stub = str(assignment.get("clip_name", f"clip_{clip_index:03d}"))
            clip_name = f"trained_{clip_stub}_{rollout_idx:03d}".replace("/", "_").replace(" ", "_")

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
                scoring_failures.append(f"Rollout video missing for {clip_name}")
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
                scoring_failures.append(f"VLM scoring failed for {clip_name}: {e}")
                score = JudgeScore(0, 0, 0, str(e), "")

            trained_scores.append(
                {
                    "condition": "trained",
                    "task": task,
                    "rollout_index": rollout_idx,
                    "task_score": score.task_score,
                    "visual_score": score.visual_score,
                    "spatial_score": score.spatial_score,
                    "reasoning": score.reasoning,
                    "video_path": str(rollout.video_path),
                    "num_steps": rollout.num_steps,
                    "action_sequence": rollout.action_sequence,
                    "start_clip_index": clip_index,
                    "start_clip_name": clip_stub,
                    "start_path_type": str(assignment.get("path_type", "unknown")),
                    "target_instance_id": assignment.get("target_instance_id"),
                    "target_label": assignment.get("target_label"),
                    "is_manipulation_task": _is_manipulation_task(task),
                    "grasp_acquired": (
                        score.grasp_acquired if isinstance(score, ManipulationJudgeScore) else None
                    ),
                    "lifted_clear": (
                        score.lifted_clear if isinstance(score, ManipulationJudgeScore) else None
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
                }
            )

        # Load Stage 4 scores for pairwise comparison
        prev_s4 = previous_results.get("s4_policy_eval")
        prior_scores: List[Dict] = []
        if prev_s4 and prev_s4.status == "success":
            scores_path = prev_s4.outputs.get("scores_path")
            if scores_path and Path(scores_path).exists():
                prior_scores = read_json(Path(scores_path)).get("scores", [])

        all_scores = prior_scores + trained_scores

        # Save combined scores
        write_json({"scores": all_scores}, eval_dir / "vlm_scores_combined.json")
        audit_csv_path = eval_dir / "judge_audit.csv"
        write_judge_audit_csv(all_scores, audit_csv_path)

        # Compute pairwise metrics
        pairwise = _build_pairwise_metrics(all_scores)

        trained_mean = np.mean([s["task_score"] for s in trained_scores]) if trained_scores else 0
        trained_manip = [s for s in trained_scores if s["is_manipulation_task"]]

        metrics = {
            "trained_mean_task_score": round(float(trained_mean), 3),
            "trained_manipulation_success_rate": _manipulation_success_rate(trained_manip),
            "num_rollouts_trained": len(trained_scores),
            "num_scoring_failures": len(scoring_failures),
            "task_hints_injected": hint_count,
            "requested_rollouts_trained": requested_rollouts,
            "planned_rollouts_trained": planned_rollouts,
            "executed_rollouts_trained": num_rollouts,
            "num_unique_task_templates": len(tasks),
            "shared_task_start_manifest": str(shared_manifest_path),
            "shared_task_start_manifest_reused": reused_shared_manifest,
            "trained_checkpoint": str(trained_checkpoint),
            "pairwise": pairwise,
            "judge_audit_csv": str(audit_csv_path),
        }
        claim_failure_reasons: List[str] = []
        claim_pair = pairwise.get("baseline_vs_trained") or pairwise.get("trained_vs_baseline")
        abs_diff = 0.0
        if claim_pair is not None:
            abs_diff = float(claim_pair.get("absolute_difference", 0.0) or 0.0)
        if abs_diff < float(config.eval_policy.min_absolute_difference):
            claim_failure_reasons.append(
                "Absolute task-score difference below threshold: "
                f"{abs_diff:.3f} < {config.eval_policy.min_absolute_difference:.3f}"
            )
        baseline_manip = None
        for row in prior_scores:
            if row.get("condition") == "baseline":
                baseline_manip = row
                break
        baseline_manip_rate = None
        if baseline_manip is not None:
            baseline_manip_rows = [s for s in prior_scores if s.get("condition") == "baseline" and s.get("is_manipulation_task")]
            baseline_manip_rate = _manipulation_success_rate(baseline_manip_rows)
        trained_manip_rate = _manipulation_success_rate(trained_manip)
        manip_delta_pp = None
        if baseline_manip_rate is not None:
            manip_delta_pp = (trained_manip_rate - baseline_manip_rate) * 100.0
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

        write_json(metrics, eval_dir / "trained_eval_report.json")

        return StageResult(
            stage_name=self.name,
            status=(
                "success"
                if trained_scores and (not claim_mode or claim_passed)
                else "failed"
            ),
            elapsed_seconds=0,
            outputs={
                "eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "vlm_scores_combined.json"),
                "report_path": str(eval_dir / "trained_eval_report.json"),
                "shared_task_start_manifest": str(shared_manifest_path),
                "judge_audit_csv": str(audit_csv_path),
            },
            metrics=metrics,
            detail="\n".join(scoring_failures[:5]),
        )


def _extract_initial_frames(render_manifest: dict) -> list:
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
                profile="policy",
            )
        except Exception as exc:
            logger.warning("Failed loading task hints from %s: %s", facility.task_hints_path, exc)
    for task in hint_tasks:
        if task not in tasks:
            tasks.append(task)

    if not tasks:
        tasks = [
            "Navigate forward through the corridor",
            "Pick up the tote from the shelf",
        ]
    return balance_eval_tasks(tasks, profile="policy"), len(hint_tasks)


def _resolve_trained_checkpoint(
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

    # Prefer RL loop output.
    s3c = previous_results.get("s3c_policy_rl_loop")
    if s3c and s3c.status == "success":
        candidate = s3c.outputs.get("adapted_policy_checkpoint_rl") or s3c.outputs.get(
            "adapted_openvla_checkpoint_rl"
        )
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    # Fallback to supervised policy fine-tune output.
    s3b = previous_results.get("s3b_policy_finetune")
    if s3b and s3b.status == "success":
        candidate = s3b.outputs.get("adapted_policy_checkpoint") or s3b.outputs.get(
            "adapted_openvla_checkpoint"
        )
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    # Backward-compatible filesystem fallback.
    rl_root = work_dir / "policy_rl_loop"
    if rl_root.exists():
        for iter_dir in sorted(rl_root.glob("iter_*"), reverse=True):
            candidate = policy_adapter.resolve_latest_checkpoint(
                iter_dir / "policy_refine" / "runs"
            )
            if candidate is not None:
                return candidate

    candidate = policy_adapter.resolve_latest_checkpoint(work_dir / "policy_finetune" / "runs")
    if candidate is not None:
        return candidate
    return None


def _has_successful_training_stage(previous_results: Dict[str, StageResult]) -> bool:
    for key in ("s3c_policy_rl_loop", "s3b_policy_finetune"):
        stage = previous_results.get(key)
        if stage and stage.status == "success":
            return True
    return False


def _build_pairwise_metrics(all_scores: List[Dict]) -> Dict:
    """Compute improvement, win rate, and p-value for each pair of conditions."""
    conditions = sorted({s["condition"] for s in all_scores})
    pairwise = {}

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1 :]:
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
                1 for a, b in zip(s1[:min_len], s2[:min_len]) if b["task_score"] > a["task_score"]
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
