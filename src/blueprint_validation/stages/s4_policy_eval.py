"""Stage 4: OpenVLA policy evaluation — baseline vs site-adapted DreamDojo."""

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
from ..evaluation.vlm_judge import JudgeScore, score_rollout
from .base import PipelineStage

logger = get_logger("stages.s4_policy_eval")


class PolicyEvalStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s4_policy_eval"

    @property
    def description(self) -> str:
        return "Evaluate OpenVLA policy in baseline vs site-adapted DreamDojo world model"

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

        # Extract initial frames from rendered clips
        initial_frames = _extract_initial_frames(render_manifest)
        if not initial_frames:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Could not extract initial frames from rendered clips.",
            )

        tasks = config.eval_policy.tasks
        if not tasks:
            tasks = [
                "Navigate forward through the corridor",
                "Turn left at the intersection",
                "Approach the nearest obstacle",
            ]

        num_rollouts = config.eval_policy.num_rollouts
        max_steps = config.eval_policy.max_steps_per_rollout

        device = "cuda" if _has_cuda() else "cpu"
        adapted_policy_checkpoint = _resolve_adapted_policy_checkpoint(previous_results, work_dir)

        all_scores: List[Dict] = []
        scoring_failures: List[str] = []
        rollout_plan = _build_rollout_plan(tasks, num_rollouts)

        for condition in ["baseline", "adapted"]:
            logger.info("Running %s condition rollouts", condition)
            condition_dir = eval_dir / f"{condition}_rollouts"
            condition_dir.mkdir(exist_ok=True)

            # Load policy checkpoint for this condition.
            policy_checkpoint = config.eval_policy.openvla_checkpoint
            if condition == "adapted" and adapted_policy_checkpoint is not None:
                policy_checkpoint = adapted_policy_checkpoint
            openvla_model, openvla_processor = load_openvla(
                config.eval_policy.openvla_model,
                policy_checkpoint,
                device=device,
            )

            # Load world model
            adapted = adapted_dir if condition == "adapted" else None
            world_model = load_dreamdojo_world_model(
                checkpoint_path=config.finetune.dreamdojo_checkpoint,
                adapted_checkpoint=adapted,
                device=device,
            )

            for rollout_idx, task in enumerate(rollout_plan):
                # Pick initial frame (cycle through available)
                init_frame = initial_frames[rollout_idx % len(initial_frames)]
                clip_name = f"{condition}_{task[:20].replace(' ', '_')}_{rollout_idx:03d}"

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
                    msg = f"Rollout video missing for {clip_name}"
                    scoring_failures.append(msg)
                    logger.warning(msg)
                    continue

                try:
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
                })

        # Compute aggregate metrics
        baseline_scores = [s for s in all_scores if s["condition"] == "baseline"]
        adapted_scores = [s for s in all_scores if s["condition"] == "adapted"]

        baseline_mean = np.mean([s["task_score"] for s in baseline_scores]) if baseline_scores else 0
        adapted_mean = np.mean([s["task_score"] for s in adapted_scores]) if adapted_scores else 0
        improvement = ((adapted_mean - baseline_mean) / max(baseline_mean, 1e-8)) * 100

        # Win rate
        min_len = min(len(baseline_scores), len(adapted_scores))
        wins = sum(
            1 for b, a in zip(baseline_scores[:min_len], adapted_scores[:min_len])
            if a["task_score"] > b["task_score"]
        )
        win_rate = wins / max(min_len, 1)

        # Statistical significance (paired t-test)
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

        # Save scores
        write_json({"scores": all_scores}, eval_dir / "vlm_scores.json")

        metrics = {
            "baseline_mean_task_score": round(float(baseline_mean), 3),
            "adapted_mean_task_score": round(float(adapted_mean), 3),
            "improvement_pct": round(float(improvement), 2),
            "win_rate": round(float(win_rate), 3),
            "p_value": round(p_value, 6) if p_value is not None else None,
            "num_rollouts_baseline": len(baseline_scores),
            "num_rollouts_adapted": len(adapted_scores),
            "num_scoring_failures": len(scoring_failures),
            "used_adapted_policy_checkpoint": adapted_policy_checkpoint is not None,
            "adapted_policy_checkpoint": str(adapted_policy_checkpoint) if adapted_policy_checkpoint else None,
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


def _resolve_adapted_policy_checkpoint(
    previous_results: Dict[str, StageResult],
    work_dir: Path,
) -> Path | None:
    prev = previous_results.get("s3b_policy_finetune")
    if prev:
        candidate = prev.outputs.get("adapted_openvla_checkpoint")
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    fallback = work_dir / "policy_finetune" / "adapters"
    if fallback.exists() and any(fallback.iterdir()):
        return fallback
    return None
