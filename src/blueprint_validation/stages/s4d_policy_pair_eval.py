"""Stage 4d: Heldout paired evaluation of trained policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.openvla_runner import load_dreamdojo_world_model
from ..evaluation.rollout_utils import run_rollout_with_adapter
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
        del previous_results
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
                if _is_manipulation_task(task, config):
                    score = score_rollout_manipulation(
                        video_path=rollout_video,
                        task_prompt=task,
                        config=config.eval_policy.vlm_judge,
                        facility_description=facility.description,
                    )
                    success = _manip_success(score, config)
                    manip_success = success
                else:
                    score = score_rollout(
                        video_path=rollout_video,
                        task_prompt=task,
                        config=config.eval_policy.vlm_judge,
                        facility_description=facility.description,
                    )
                    success = score.task_score >= config.policy_compare.task_score_success_threshold
                    manip_success = None

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
        write_json(metrics, eval_dir / "pair_eval_report.json")

        return StageResult(
            stage_name=self.name,
            status="success" if score_rows else "failed",
            elapsed_seconds=0,
            outputs={
                "pair_eval_dir": str(eval_dir),
                "scores_path": str(eval_dir / "pair_scores.json"),
                "report_path": str(eval_dir / "pair_eval_report.json"),
            },
            metrics=metrics,
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
            }
        )
    return episodes


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

    base_manip = [p[0]["manipulation_success"] for p in paired if p[0]["manipulation_success"] is not None]
    site_manip = [p[1]["manipulation_success"] for p in paired if p[1]["manipulation_success"] is not None]
    base_manip_rate = float(np.mean(base_manip)) if base_manip else None
    site_manip_rate = float(np.mean(site_manip)) if site_manip else None

    return {
        "num_pairs": len(paired),
        "policy_base_mean_task_score": round(float(np.mean(base_scores)), 3),
        "policy_site_mean_task_score": round(float(np.mean(site_scores)), 3),
        "task_score_improvement_pct": round(
            ((float(np.mean(site_scores)) - float(np.mean(base_scores))) / max(float(np.mean(base_scores)), 1e-8))
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
