"""World-VLA-Loop-style policy RL loop orchestration helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..common import get_logger, write_json
from ..config import FacilityConfig, PolicyFinetuneConfig, ValidationConfig
from ..evaluation.openvla_runner import load_dreamdojo_world_model, run_rollout
from ..evaluation.rollout_utils import run_rollout_with_adapter
from ..evaluation.vlm_judge import (
    JudgeScore,
    ManipulationJudgeScore,
    score_rollout,
    score_rollout_manipulation,
)
from ..policy_adapters import get_policy_adapter
from .dataset_builder import build_dreamdojo_dataset
from .dreamdojo_finetune import run_dreamdojo_finetune
from .openvla_finetune import run_openvla_finetune
from .rlds_export import convert_jsonl_to_tfrecord, export_rollouts_to_rlds_jsonl

logger = get_logger("training.policy_rl_loop")


def run_policy_rl_iterations(
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    output_dir: Path,
    initial_policy_checkpoint: Optional[Path],
    adapted_world_checkpoint: Path,
) -> Dict:
    """Run iterative policy/world-model co-adaptation loop."""
    loop_cfg = config.policy_rl_loop
    if not loop_cfg.enabled:
        return {"status": "skipped", "detail": "policy_rl_loop.enabled=false"}

    render_manifest = _resolve_render_manifest(work_dir)
    if not render_manifest.exists():
        raise RuntimeError(f"Render manifest not found: {render_manifest}")

    from ..common import read_json

    initial_frames = _extract_initial_frames(read_json(render_manifest))
    if not initial_frames:
        raise RuntimeError("Could not extract initial frames for RL loop rollouts")

    tasks = _build_task_list(config)
    if not tasks:
        raise RuntimeError("No tasks configured for RL loop")

    device = "cuda" if _has_cuda() else "cpu"
    policy_adapter = get_policy_adapter(config.policy_adapter.name)

    current_policy_checkpoint = initial_policy_checkpoint
    current_world_checkpoint = adapted_world_checkpoint
    iteration_summaries: List[Dict] = []

    for iteration in range(loop_cfg.iterations):
        iter_dir = output_dir / f"iter_{iteration:02d}"
        rollouts_dir = iter_dir / "rollouts"
        iter_dir.mkdir(parents=True, exist_ok=True)
        rollouts_dir.mkdir(parents=True, exist_ok=True)

        policy_handle = policy_adapter.load_policy(
            model_name=config.eval_policy.openvla_model,
            checkpoint_path=current_policy_checkpoint,
            device=device,
        )
        world_model = load_dreamdojo_world_model(
            checkpoint_path=config.finetune.dreamdojo_checkpoint,
            adapted_checkpoint=current_world_checkpoint,
            device=device,
        )

        rollout_rows = _collect_rollouts(
            config=config,
            facility=facility,
            world_model=world_model,
            policy_adapter=policy_adapter,
            policy_handle=policy_handle,
            initial_frames=initial_frames,
            tasks=tasks,
            rollouts_per_task=loop_cfg.rollouts_per_task,
            max_steps=loop_cfg.horizon_steps,
            output_dir=rollouts_dir,
            iteration=iteration,
            device=device,
        )
        if not rollout_rows:
            raise RuntimeError(f"No rollouts generated in RL iteration {iteration}")

        selected, near_miss = _select_rollouts(
            rollout_rows=rollout_rows,
            group_size=loop_cfg.group_size,
            top_quantile=loop_cfg.top_quantile,
            near_miss_min_quantile=loop_cfg.near_miss_min_quantile,
            near_miss_max_quantile=loop_cfg.near_miss_max_quantile,
        )

        dataset_name = f"{config.policy_finetune.dataset_name}_rl_iter{iteration:02d}"
        policy_dataset_root = _export_selected_rollouts(
            selected=selected,
            output_root=iter_dir / "policy_dataset",
            dataset_name=dataset_name,
            task_threshold=config.rollout_dataset.task_score_threshold,
            min_steps=config.rollout_dataset.min_steps_per_rollout,
        )

        current_policy_checkpoint = _refine_policy(
            config=config,
            dataset_root=policy_dataset_root,
            dataset_name=dataset_name,
            iteration=iteration,
            output_dir=iter_dir / "policy_refine",
            current_policy_checkpoint=current_policy_checkpoint,
        )

        world_refresh = {
            "status": "skipped",
            "adapted_checkpoint_path": str(current_world_checkpoint),
        }
        if loop_cfg.world_model_refresh_enabled and near_miss:
            world_refresh = _refresh_world_model(
                config=config,
                facility=facility,
                near_miss_rows=near_miss,
                output_dir=iter_dir / "world_refresh",
                iteration=iteration,
            )
            world_candidate = world_refresh.get("adapted_checkpoint_path")
            if world_candidate and Path(world_candidate).exists():
                current_world_checkpoint = Path(world_candidate)

        iter_summary = {
            "iteration": iteration,
            "num_rollouts": len(rollout_rows),
            "num_selected_for_policy_refine": len(selected),
            "num_near_miss_for_world_refresh": len(near_miss),
            "policy_checkpoint_after_iter": str(current_policy_checkpoint or ""),
            "world_checkpoint_after_iter": str(current_world_checkpoint),
            "policy_refine": {
                "status": "success" if current_policy_checkpoint else "failed",
            },
            "world_refresh": world_refresh,
            "mean_reward": float(np.mean([r["rl_reward"] for r in rollout_rows])),
            "mean_task_score": float(np.mean([r.get("task_score", 0.0) for r in rollout_rows])),
        }
        write_json(iter_summary, iter_dir / "iteration_summary.json")
        iteration_summaries.append(iter_summary)

    result = {
        "status": "success",
        "iterations_completed": len(iteration_summaries),
        "final_policy_checkpoint": str(current_policy_checkpoint or ""),
        "final_world_checkpoint": str(current_world_checkpoint),
        "iteration_summaries": iteration_summaries,
    }
    write_json(result, output_dir / "policy_rl_loop_log.json")
    return result


def _collect_rollouts(
    config: ValidationConfig,
    facility: FacilityConfig,
    world_model,
    policy_adapter,
    policy_handle,
    initial_frames: List[np.ndarray],
    tasks: List[str],
    rollouts_per_task: int,
    max_steps: int,
    output_dir: Path,
    iteration: int,
    device: str,
) -> List[Dict]:
    rows: List[Dict] = []
    rollout_idx = 0
    for task in tasks:
        for _ in range(rollouts_per_task):
            init_frame = initial_frames[rollout_idx % len(initial_frames)]
            clip_name = f"iter{iteration:02d}_r{rollout_idx:04d}_{task[:20].replace(' ', '_')}"

            if policy_adapter.name == "openvla":
                rollout = run_rollout(
                    world_model=world_model,
                    openvla_model=policy_handle.model,
                    openvla_processor=policy_handle.processor,
                    initial_frame=init_frame,
                    task_prompt=task,
                    max_steps=max_steps,
                    unnorm_key=config.eval_policy.unnorm_key,
                    output_dir=output_dir,
                    clip_name=clip_name,
                    device=device,
                )
            else:
                rollout = run_rollout_with_adapter(
                    world_model=world_model,
                    policy_adapter=policy_adapter,
                    policy_handle=policy_handle,
                    initial_frame=init_frame,
                    task_prompt=task,
                    max_steps=max_steps,
                    unnorm_key=config.eval_policy.unnorm_key,
                    output_dir=output_dir,
                    clip_name=clip_name,
                    device=device,
                )

            if not rollout.video_path or not rollout.video_path.exists():
                rollout_idx += 1
                continue

            reward_payload = _score_rollout_reward(
                config=config,
                facility=facility,
                task=task,
                rollout_index=rollout_idx,
                video_path=Path(rollout.video_path),
                action_sequence=rollout.action_sequence,
            )
            rows.append(
                {
                    "condition": "adapted",
                    "task": task,
                    "rollout_index": rollout_idx,
                    "video_path": str(rollout.video_path),
                    "num_steps": rollout.num_steps,
                    "action_sequence": rollout.action_sequence,
                    "is_manipulation_task": _is_manipulation_task(task),
                    **reward_payload,
                }
            )
            rollout_idx += 1
    return rows


def _score_rollout_reward(
    config: ValidationConfig,
    facility: FacilityConfig,
    task: str,
    rollout_index: int,
    video_path: Path,
    action_sequence: List[List[float]],
) -> Dict:
    loop_cfg = config.policy_rl_loop
    should_query_vlm = False
    if loop_cfg.reward_mode == "vlm_only":
        should_query_vlm = True
    elif loop_cfg.reward_mode == "hybrid":
        # Deterministic sampling using rollout index.
        period = max(1, int(round(1.0 / max(loop_cfg.vlm_reward_fraction, 1e-3))))
        should_query_vlm = (rollout_index % period) == 0

    vlm_score: Optional[JudgeScore] = None
    if should_query_vlm:
        try:
            if _is_manipulation_task(task):
                vlm_score = score_rollout_manipulation(
                    video_path=video_path,
                    task_prompt=task,
                    config=config.eval_policy.vlm_judge,
                    facility_description=facility.description,
                )
            else:
                vlm_score = score_rollout(
                    video_path=video_path,
                    task_prompt=task,
                    config=config.eval_policy.vlm_judge,
                    facility_description=facility.description,
                )
        except Exception as exc:
            logger.warning("VLM reward scoring failed for %s: %s", video_path.name, exc)

    action_arr = np.asarray(action_sequence, dtype=np.float32) if action_sequence else np.zeros((0, 1))
    smooth_reward = 0.0
    stability_penalty = 0.0
    if len(action_arr) >= 2:
        deltas = np.diff(action_arr, axis=0)
        mean_delta = float(np.mean(np.linalg.norm(deltas, axis=1)))
        smooth_reward = float(max(0.0, 1.0 - (mean_delta / max(config.rollout_dataset.max_action_delta_norm, 1e-6))))
        stability_penalty = float(min(1.0, mean_delta / max(config.rollout_dataset.max_action_delta_norm, 1e-6)))
    elif len(action_arr) == 1:
        smooth_reward = 0.5

    step_reward = min(1.0, len(action_sequence) / max(config.policy_rl_loop.horizon_steps, 1))
    heuristic_reward = float(0.6 * smooth_reward + 0.4 * step_reward)

    task_score = float(vlm_score.task_score) if vlm_score else 0.0
    visual_score = float(vlm_score.visual_score) if vlm_score else 0.0
    spatial_score = float(vlm_score.spatial_score) if vlm_score else 0.0
    vlm_reward = float(task_score / 10.0) if vlm_score else 0.0

    if loop_cfg.reward_mode == "vlm_only":
        rl_reward = vlm_reward if vlm_score else heuristic_reward * 0.25
    elif loop_cfg.reward_mode == "heuristic_only":
        rl_reward = heuristic_reward
    else:
        rl_reward = (0.7 * heuristic_reward) + (0.3 * vlm_reward)

    grasp = None
    lifted = None
    placed = None
    stable = None
    if isinstance(vlm_score, ManipulationJudgeScore):
        grasp = vlm_score.grasp_acquired
        lifted = vlm_score.lifted_clear
        placed = vlm_score.placed_in_target
        stable = vlm_score.stable_after_place

    return {
        "task_score": task_score,
        "visual_score": visual_score,
        "spatial_score": spatial_score,
        "reasoning": vlm_score.reasoning if vlm_score else "heuristic_reward_only",
        "grasp_acquired": grasp,
        "lifted_clear": lifted,
        "placed_in_target": placed,
        "stable_after_place": stable,
        "heuristic_reward": round(heuristic_reward, 6),
        "vlm_reward": round(vlm_reward, 6),
        "stability_penalty": round(stability_penalty, 6),
        "rl_reward": round(float(rl_reward), 6),
    }


def _select_rollouts(
    rollout_rows: List[Dict],
    group_size: int,
    top_quantile: float,
    near_miss_min_quantile: float,
    near_miss_max_quantile: float,
) -> tuple[List[Dict], List[Dict]]:
    task_groups: Dict[str, List[Dict]] = {}
    for row in rollout_rows:
        task_groups.setdefault(row["task"], []).append(row)

    selected: List[Dict] = []
    near_miss: List[Dict] = []

    for task, rows in task_groups.items():
        rows = sorted(rows, key=lambda r: float(r["rl_reward"]), reverse=True)
        # Group-relative advantage (GRPO-style normalization by group mean).
        for i in range(0, len(rows), max(1, group_size)):
            group = rows[i:i + max(1, group_size)]
            g_mean = float(np.mean([float(r["rl_reward"]) for r in group]))
            for row in group:
                row["advantage"] = round(float(row["rl_reward"]) - g_mean, 6)

        rewards = np.asarray([float(r["rl_reward"]) for r in rows], dtype=np.float32)
        top_thr = float(np.quantile(rewards, max(0.0, 1.0 - top_quantile)))
        near_low = float(np.quantile(rewards, max(0.0, 1.0 - near_miss_max_quantile)))
        near_high = float(np.quantile(rewards, max(0.0, 1.0 - near_miss_min_quantile)))

        for row in rows:
            reward = float(row["rl_reward"])
            if reward >= top_thr:
                selected.append(row)
            elif near_low <= reward <= near_high:
                near_miss.append(row)

        logger.info(
            "Task=%s selected=%d near_miss=%d (n=%d)",
            task,
            sum(1 for r in rows if float(r["rl_reward"]) >= top_thr),
            sum(1 for r in rows if near_low <= float(r["rl_reward"]) <= near_high),
            len(rows),
        )

    return selected, near_miss


def _export_selected_rollouts(
    selected: List[Dict],
    output_root: Path,
    dataset_name: str,
    task_threshold: float,
    min_steps: int,
) -> Path:
    jsonl_root = output_root / "jsonl"
    train_dir = jsonl_root / "train"
    eval_dir = jsonl_root / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    export_rollouts_to_rlds_jsonl(
        rollouts=selected,
        output_dir=train_dir,
        condition="adapted",
        split="train",
        task_threshold=task_threshold,
        min_steps_per_rollout=min_steps,
        include_failed_rollouts=True,
    )
    # No eval split for iterative refine; generate empty placeholder.
    (eval_dir / "episodes.jsonl").write_text("")

    tfrecord_root = output_root / "tfrecord"
    convert_jsonl_to_tfrecord(
        train_jsonl_path=train_dir / "episodes.jsonl",
        eval_jsonl_path=None,
        output_dir=tfrecord_root,
        dataset_name=dataset_name,
    )
    dataset_dir = tfrecord_root / dataset_name
    if not dataset_dir.exists():
        # Fallback path when TensorFlow is unavailable and only JSONL is emitted.
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "train_episodes.jsonl").write_text((train_dir / "episodes.jsonl").read_text())
    return tfrecord_root


def _refine_policy(
    config: ValidationConfig,
    dataset_root: Path,
    dataset_name: str,
    iteration: int,
    output_dir: Path,
    current_policy_checkpoint: Optional[Path],
) -> Optional[Path]:
    local_cfg: PolicyFinetuneConfig = replace(
        config.policy_finetune,
        data_root_dir=dataset_root,
        dataset_name=dataset_name,
        max_steps=config.policy_rl_loop.policy_refine_steps_per_iter,
    )
    vla_path = (
        str(current_policy_checkpoint)
        if current_policy_checkpoint and current_policy_checkpoint.exists()
        else config.eval_policy.openvla_model
    )
    result = run_openvla_finetune(
        config=local_cfg,
        vla_path=vla_path,
        facility_id=f"rl_iter_{iteration:02d}",
        output_dir=output_dir,
    )
    adapted = result.get("adapted_checkpoint_path")
    if adapted and Path(adapted).exists():
        return Path(adapted)
    logger.warning("Policy refine failed for iter=%d: %s", iteration, result.get("stderr", ""))
    return current_policy_checkpoint


def _refresh_world_model(
    config: ValidationConfig,
    facility: FacilityConfig,
    near_miss_rows: List[Dict],
    output_dir: Path,
    iteration: int,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "refresh_manifest.json"
    clips = []
    for idx, row in enumerate(near_miss_rows):
        clips.append(
            {
                "clip_name": f"near_miss_{idx:04d}",
                "variant_name": "near_miss",
                "prompt": f"Near-miss rollout for task: {row['task']}",
                "output_video_path": row["video_path"],
                "input_video_path": row["video_path"],
            }
        )
    write_json({"facility": facility.name, "clips": clips}, manifest_path)

    dataset_dir = build_dreamdojo_dataset(
        enriched_manifest_path=manifest_path,
        output_dir=output_dir,
        facility_name=f"{facility.name}_rl_iter_{iteration:02d}",
    )

    ft_cfg = replace(
        config.finetune,
        num_epochs=config.policy_rl_loop.world_model_refresh_epochs,
        learning_rate=config.policy_rl_loop.world_model_refresh_learning_rate,
        max_training_hours=min(config.finetune.max_training_hours, 8.0),
    )
    result = run_dreamdojo_finetune(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=ft_cfg,
        facility_id=f"rl_iter_{iteration:02d}",
    )
    return result


def _resolve_render_manifest(work_dir: Path) -> Path:
    for candidate in [
        work_dir / "gaussian_augment" / "augmented_manifest.json",
        work_dir / "gemini_polish" / "polished_manifest.json",
        work_dir / "robot_composite" / "composited_manifest.json",
        work_dir / "renders" / "render_manifest.json",
    ]:
        if candidate.exists():
            return candidate
    return work_dir / "renders" / "render_manifest.json"


def _extract_initial_frames(render_manifest: dict) -> List[np.ndarray]:
    import cv2

    frames: List[np.ndarray] = []
    for clip in render_manifest.get("clips", []):
        video_path = Path(str(clip.get("video_path", "")))
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def _build_task_list(config: ValidationConfig) -> List[str]:
    tasks = list(config.eval_policy.tasks or [])
    for task in config.eval_policy.manipulation_tasks:
        if task not in tasks:
            tasks.append(task)
    return tasks


def _is_manipulation_task(task: str) -> bool:
    lowered = task.lower()
    return any(k in lowered for k in ["pick", "grasp", "lift", "place", "stack", "regrasp", "tote", "bin"])


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


