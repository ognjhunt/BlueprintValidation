"""Build policy-evaluation matrix artifacts across facilities.

This utility is intentionally lightweight and stage-free. It aggregates existing
S4d artifacts into a policy-uplift matrix with three axes:

1) seen_task_seen_env
2) unseen_object_seen_env
3) seen_task_novel_env
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..common import get_logger, read_json, write_json
from ..config import ValidationConfig

logger = get_logger("evaluation.policy_eval_matrix")


def build_policy_eval_matrix_artifact(config: ValidationConfig, work_dir: Path) -> Path | None:
    """Aggregate available S4d outputs and emit policy_eval/matrix_report.json."""
    facility_ids = list(config.facilities.keys())
    if not facility_ids:
        return None

    primary_facility = facility_ids[0]
    novel_facility = facility_ids[1] if len(facility_ids) > 1 else None

    primary = _load_pair_eval_payload(work_dir / primary_facility)
    if primary is None:
        logger.info("Skipping policy matrix: missing S4d payload for %s", primary_facility)
        return None

    novel = _load_pair_eval_payload(work_dir / novel_facility) if novel_facility else None

    seen_axis = _axis_from_stage_metrics(primary["metrics"])
    unseen_tasks = [str(t).strip() for t in (config.policy_compare.heldout_tasks or []) if str(t).strip()]
    unseen_axis = _axis_from_filtered_tasks(primary["rows"], unseen_tasks)
    novel_axis = _axis_seen_task_novel_env(
        primary_rows=primary["rows"],
        novel_rows=(novel["rows"] if novel else []),
        seen_tasks=unseen_tasks,
    )

    forgetting_ratio = novel_axis.get("forgetting_ratio")
    gate = 0.15
    matrix = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_facility": primary_facility,
        "novel_facility": novel_facility,
        "axes": {
            "seen_task_seen_env": seen_axis,
            "unseen_object_seen_env": unseen_axis,
            "seen_task_novel_env": novel_axis,
        },
        "forgetting_ratio": forgetting_ratio,
        "forgetting_ratio_gate": gate,
        "forgetting_ratio_pass": (
            bool(forgetting_ratio <= gate) if isinstance(forgetting_ratio, (int, float)) else None
        ),
    }

    output_path = work_dir / "policy_eval" / "matrix_report.json"
    write_json(matrix, output_path)
    logger.info("Policy eval matrix written to %s", output_path)
    return output_path


def _load_pair_eval_payload(facility_dir: Path | None) -> Dict | None:
    if facility_dir is None:
        return None
    stage_path = facility_dir / "s4d_policy_pair_eval_result.json"
    score_path = facility_dir / "policy_pair_eval" / "pair_scores.json"
    if not stage_path.exists() or not score_path.exists():
        return None
    stage_payload = read_json(stage_path)
    score_payload = read_json(score_path)
    rows = score_payload.get("scores", [])
    if not isinstance(rows, list):
        rows = []
    return {"metrics": stage_payload.get("metrics", {}), "rows": rows}


def _axis_from_stage_metrics(metrics: Dict) -> Dict:
    if not isinstance(metrics, dict):
        return {"available": False}
    return {
        "available": True,
        "task_score_absolute_difference": metrics.get("task_score_absolute_difference"),
        "p_value_task_score": metrics.get("p_value_task_score"),
        "policy_base_mean_task_score": metrics.get("policy_base_mean_task_score"),
        "policy_site_mean_task_score": metrics.get("policy_site_mean_task_score"),
        "num_pairs": metrics.get("num_pairs"),
    }


def _axis_from_filtered_tasks(rows: List[Dict], tasks: List[str]) -> Dict:
    if not tasks:
        return {
            "available": False,
            "detail": "policy_compare.heldout_tasks is empty; cannot form unseen-object axis.",
        }
    paired = _build_pairs([r for r in rows if str(r.get("task", "")) in set(tasks)])
    if not paired:
        return {"available": False, "detail": "No paired rows matched heldout task filter."}
    metrics = _pair_metrics_from_pairs(paired)
    metrics["available"] = True
    metrics["task_filter"] = list(tasks)
    return metrics


def _axis_seen_task_novel_env(
    *,
    primary_rows: List[Dict],
    novel_rows: List[Dict],
    seen_tasks: List[str],
) -> Dict:
    if not novel_rows:
        return {"available": False, "detail": "No second-facility S4d payload for novel-env axis."}

    task_set = set(seen_tasks) if seen_tasks else _task_intersection(primary_rows, novel_rows)
    primary_site = _site_task_scores(primary_rows, task_set)
    novel_site = _site_task_scores(novel_rows, task_set)
    if not primary_site or not novel_site:
        return {"available": False, "detail": "Insufficient policy_site scores for novel-env axis."}

    primary_mean = float(np.mean(primary_site))
    novel_mean = float(np.mean(novel_site))
    p_value = _independent_ttest_p_value(primary_site, novel_site)
    forgetting_ratio = max(0.0, (primary_mean - novel_mean) / max(primary_mean, 1e-8))

    return {
        "available": True,
        "task_score_absolute_difference": round(novel_mean - primary_mean, 3),
        "p_value_task_score": round(p_value, 6) if p_value is not None else None,
        "primary_site_mean_task_score": round(primary_mean, 3),
        "novel_site_mean_task_score": round(novel_mean, 3),
        "forgetting_ratio": round(forgetting_ratio, 6),
        "num_primary_site_scores": len(primary_site),
        "num_novel_site_scores": len(novel_site),
    }


def _build_pairs(rows: Iterable[Dict]) -> List[Tuple[Dict, Dict]]:
    grouped: Dict[str, Dict[str, Dict]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id", "")).strip()
        policy = str(row.get("policy", "")).strip()
        if not episode_id or policy not in {"policy_base", "policy_site"}:
            continue
        grouped.setdefault(episode_id, {})[policy] = row

    pairs: List[Tuple[Dict, Dict]] = []
    for entry in grouped.values():
        base = entry.get("policy_base")
        site = entry.get("policy_site")
        if base is None or site is None:
            continue
        pairs.append((base, site))
    return pairs


def _pair_metrics_from_pairs(pairs: List[Tuple[Dict, Dict]]) -> Dict:
    base_scores = [float(base.get("task_score", 0.0)) for base, _ in pairs]
    site_scores = [float(site.get("task_score", 0.0)) for _, site in pairs]
    p_value = _paired_ttest_p_value(base_scores, site_scores)
    wins = sum(1 for b, s in zip(base_scores, site_scores) if s > b)
    return {
        "num_pairs": len(pairs),
        "policy_base_mean_task_score": round(float(np.mean(base_scores)), 3),
        "policy_site_mean_task_score": round(float(np.mean(site_scores)), 3),
        "task_score_absolute_difference": round(float(np.mean(site_scores)) - float(np.mean(base_scores)), 3),
        "p_value_task_score": round(p_value, 6) if p_value is not None else None,
        "win_rate_site_over_base": round(wins / max(len(pairs), 1), 3),
    }


def _task_intersection(rows_a: List[Dict], rows_b: List[Dict]) -> set[str]:
    tasks_a = {str(r.get("task", "")).strip() for r in rows_a if str(r.get("task", "")).strip()}
    tasks_b = {str(r.get("task", "")).strip() for r in rows_b if str(r.get("task", "")).strip()}
    return tasks_a & tasks_b


def _site_task_scores(rows: List[Dict], task_set: set[str]) -> List[float]:
    scores: List[float] = []
    for row in rows:
        if str(row.get("policy", "")) != "policy_site":
            continue
        task = str(row.get("task", "")).strip()
        if task_set and task not in task_set:
            continue
        scores.append(float(row.get("task_score", 0.0)))
    return scores


def _paired_ttest_p_value(a: List[float], b: List[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    try:
        from scipy import stats

        _, p_value = stats.ttest_rel(a, b)
        return float(p_value)
    except Exception:
        return None


def _independent_ttest_p_value(a: List[float], b: List[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy import stats

        _, p_value = stats.ttest_ind(a, b, equal_var=False)
        return float(p_value)
    except Exception:
        return None
