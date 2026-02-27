"""Stage 4e: Evaluate the trained (fine-tuned) OpenVLA policy in the adapted world model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.openvla_runner import (
    load_dreamdojo_world_model,
    load_openvla,
    run_rollout,
)
from ..evaluation.vlm_judge import (
    JudgeScore,
    ManipulationJudgeScore,
    score_rollout,
    score_rollout_manipulation,
)
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

    Runs the ``trained`` condition: fine-tuned OpenVLA in the adapted world
    model.  Then merges with Stage 4 scores to produce pairwise comparisons
    (baseline-vs-trained, adapted-vs-trained).
    """

    @property
    def name(self) -> str:
        return "s4e_trained_eval"

    @property
    def description(self) -> str:
        return "Evaluate trained OpenVLA policy in site-adapted DreamDojo world model"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        # Only run if S3b produced a trained checkpoint
        s3b = previous_results.get("s3b_policy_finetune")
        if not s3b or s3b.status != "success":
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="S3b (policy_finetune) did not succeed. Skipping trained eval.",
            )

        trained_checkpoint = s3b.outputs.get("adapted_openvla_checkpoint")
        if not trained_checkpoint or not Path(trained_checkpoint).exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"Trained checkpoint not found: {trained_checkpoint}",
            )

        # Need adapted DreamDojo checkpoint from Stage 3
        adapted_dir = None
        prev_s3 = previous_results.get("s3_finetune")
        if prev_s3:
            adapted_candidate = (
                prev_s3.outputs.get("adapted_checkpoint_path")
                or prev_s3.outputs.get("lora_weights_path")
            )
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
        initial_frames = _extract_initial_frames(render_manifest)
        if not initial_frames:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Could not extract initial frames from rendered clips.",
            )

        # Build task list (merge navigation + manipulation)
        tasks = list(config.eval_policy.tasks or [])
        for mt in config.eval_policy.manipulation_tasks:
            if mt not in tasks:
                tasks.append(mt)
        if not tasks:
            tasks = [
                "Navigate forward through the corridor",
                "Pick up the tote from the shelf",
            ]

        num_rollouts = config.eval_policy.num_rollouts
        max_steps = config.eval_policy.max_steps_per_rollout
        device = "cuda" if _has_cuda() else "cpu"

        eval_dir = work_dir / "trained_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        condition_dir = eval_dir / "trained_rollouts"
        condition_dir.mkdir(exist_ok=True)

        # Load trained policy
        openvla_model, openvla_processor = load_openvla(
            config.eval_policy.openvla_model,
            Path(trained_checkpoint),
            device=device,
        )

        # Load adapted world model (always use adapted for trained eval)
        world_model = load_dreamdojo_world_model(
            checkpoint_path=config.finetune.dreamdojo_checkpoint,
            adapted_checkpoint=adapted_dir,
            device=device,
        )

        rollout_plan = _build_rollout_plan(tasks, num_rollouts)
        trained_scores: List[Dict] = []
        scoring_failures: List[str] = []

        for rollout_idx, task in enumerate(rollout_plan):
            init_frame = initial_frames[rollout_idx % len(initial_frames)]
            clip_name = f"trained_{task[:20].replace(' ', '_')}_{rollout_idx:03d}"

            rollout = run_rollout(
                world_model=world_model,
                openvla_model=openvla_model,
                openvla_processor=openvla_processor,
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

            trained_scores.append({
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

        # Compute pairwise metrics
        pairwise = _build_pairwise_metrics(all_scores)

        trained_mean = (
            np.mean([s["task_score"] for s in trained_scores]) if trained_scores else 0
        )
        trained_manip = [s for s in trained_scores if s["is_manipulation_task"]]

        metrics = {
            "trained_mean_task_score": round(float(trained_mean), 3),
            "trained_manipulation_success_rate": _manipulation_success_rate(trained_manip),
            "num_rollouts_trained": len(trained_scores),
            "num_scoring_failures": len(scoring_failures),
            "trained_checkpoint": str(trained_checkpoint),
            "pairwise": pairwise,
        }

        write_json(metrics, eval_dir / "trained_eval_report.json")

        return StageResult(
            stage_name=self.name,
            status="success" if trained_scores else "failed",
            elapsed_seconds=0,
            outputs={
                "eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "vlm_scores_combined.json"),
                "report_path": str(eval_dir / "trained_eval_report.json"),
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


def _build_pairwise_metrics(all_scores: List[Dict]) -> Dict:
    """Compute improvement, win rate, and p-value for each pair of conditions."""
    conditions = sorted({s["condition"] for s in all_scores})
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

            min_len = min(len(s1), len(s2))
            wins = sum(
                1
                for a, b in zip(s1[:min_len], s2[:min_len])
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
                "win_rate": round(win_rate, 3),
                "p_value": round(p_value, 6) if p_value is not None else None,
            }

    return pairwise
