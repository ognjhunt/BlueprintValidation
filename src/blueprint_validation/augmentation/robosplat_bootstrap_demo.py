"""Synthetic bootstrap demo generation for RoboSplat hybrid mode."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..common import get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..evaluation.task_hints import tasks_from_task_hints

logger = get_logger("augmentation.robosplat_bootstrap_demo")


def resolve_or_create_demo_manifest(
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    source_clips: List[Dict],
) -> Optional[Path]:
    """Resolve demo manifest from prior artifacts or synthesize lightweight pseudo-demo."""
    demo_dir = work_dir / "gaussian_augment" / "bootstrap_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_manifest_path = demo_dir / "demo_manifest.json"
    if demo_manifest_path.exists():
        return demo_manifest_path

    # Prefer prior rollout artifacts when available.
    prior = _build_from_prior_rollouts(work_dir, config.robosplat.bootstrap_tasks_limit)
    if prior:
        write_json({"facility": facility.name, "demo_source": "synthetic_prior", "entries": prior}, demo_manifest_path)
        return demo_manifest_path

    if not config.robosplat.bootstrap_if_missing_demo:
        return None

    # Lightweight bootstrap prepass: synthesize pseudo-demos from source clips + tasks.
    tasks = _build_bootstrap_tasks(config, facility, limit=config.robosplat.bootstrap_tasks_limit)
    if not tasks:
        tasks = ["Approach the nearest manipulation target"]

    entries: List[Dict] = []
    for i, clip in enumerate(source_clips[: max(1, config.robosplat.bootstrap_num_rollouts)]):
        video_path = str(clip.get("video_path", ""))
        if not video_path:
            continue
        entries.append(
            {
                "task": tasks[i % len(tasks)],
                "video_path": video_path,
                "action_sequence": [],
                "origin": "bootstrap_source_clip_stub",
            }
        )

    if not entries:
        return None

    write_json(
        {
            "facility": facility.name,
            "demo_source": "synthetic_bootstrap",
            "bootstrap_horizon_steps": config.robosplat.bootstrap_horizon_steps,
            "entries": entries,
        },
        demo_manifest_path,
    )
    return demo_manifest_path


def _build_from_prior_rollouts(work_dir: Path, limit: int) -> List[Dict]:
    candidates = [
        work_dir / "trained_policy_eval" / "vlm_scores_combined.json",
        work_dir / "policy_eval" / "vlm_scores.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = read_json(path)
            scores = list(payload.get("scores", []))
            if not scores:
                continue
            scores = sorted(scores, key=lambda s: float(s.get("task_score", 0.0)), reverse=True)
            out: List[Dict] = []
            for row in scores[: max(1, limit)]:
                video_path = str(row.get("video_path", ""))
                if not video_path:
                    continue
                out.append(
                    {
                        "task": str(row.get("task", "")),
                        "video_path": video_path,
                        "action_sequence": row.get("action_sequence", []),
                        "origin": f"prior:{path.name}",
                    }
                )
            if out:
                return out
        except Exception as exc:
            logger.warning("Failed reading prior rollout scores from %s: %s", path, exc)
    return []


def _build_bootstrap_tasks(
    config: ValidationConfig,
    facility: FacilityConfig,
    limit: int,
) -> List[str]:
    tasks = list(config.eval_policy.tasks or [])
    for task in config.eval_policy.manipulation_tasks:
        if task not in tasks:
            tasks.append(task)

    if facility.task_hints_path is not None and facility.task_hints_path.exists():
        try:
            for task in tasks_from_task_hints(facility.task_hints_path):
                if task not in tasks:
                    tasks.append(task)
        except Exception as exc:
            logger.warning("Task-hints bootstrap load failed from %s: %s", facility.task_hints_path, exc)

    if limit > 0:
        return tasks[:limit]
    return tasks

