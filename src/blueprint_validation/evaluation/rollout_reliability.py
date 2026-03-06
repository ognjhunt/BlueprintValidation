"""Shared rollout reliability helpers for claim and evaluation stages."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import ValidationConfig


def single_or_none(values: set[int]) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return None


def build_reliability_gate(
    config: ValidationConfig,
    scores: List[Dict],
    *,
    scoring_failure_rate: float = 0.0,
) -> Dict[str, object]:
    if not scores:
        return {
            "replay_pass_rate": 0.0,
            "controllability_pass_rate": 0.0,
            "passed": False,
            "reason": "No valid scored rollouts available.",
            "scoring_failure_rate": round(float(scoring_failure_rate), 6),
            "max_scoring_failure_rate": round(
                float(config.eval_policy.reliability.max_scoring_failure_rate), 6
            ),
            "num_valid_scores": 0,
        }

    replay_passes = 0
    controllability_passes = 0
    for row in scores:
        visual = float(row.get("visual_score", 0.0))
        spatial = float(row.get("spatial_score", 0.0))
        if visual >= 5.0 and spatial >= 5.0:
            replay_passes += 1
        actions = row.get("action_sequence") or []
        if len(actions) < 2:
            continue
        try:
            arr = np.asarray(actions, dtype=np.float32)
            deltas = np.diff(arr, axis=0)
            if float(np.max(np.linalg.norm(deltas, axis=1))) > 1e-4:
                controllability_passes += 1
        except Exception:
            continue

    replay_rate = replay_passes / max(len(scores), 1)
    ctrl_rate = controllability_passes / max(len(scores), 1)
    failure_rate_ok = float(scoring_failure_rate) <= float(
        config.eval_policy.reliability.max_scoring_failure_rate
    )
    passed = (
        replay_rate >= float(config.eval_policy.reliability.min_replay_pass_rate)
        and ctrl_rate >= float(config.eval_policy.reliability.min_controllability_pass_rate)
        and failure_rate_ok
    )
    reasons: List[str] = []
    if replay_rate < float(config.eval_policy.reliability.min_replay_pass_rate):
        reasons.append("replay_pass_rate below threshold")
    if ctrl_rate < float(config.eval_policy.reliability.min_controllability_pass_rate):
        reasons.append("controllability_pass_rate below threshold")
    if not failure_rate_ok:
        reasons.append("scoring_failure_rate above threshold")
    return {
        "replay_pass_rate": round(float(replay_rate), 6),
        "controllability_pass_rate": round(float(ctrl_rate), 6),
        "passed": bool(passed),
        "reason": "" if passed else "; ".join(reasons),
        "scoring_failure_rate": round(float(scoring_failure_rate), 6),
        "max_scoring_failure_rate": round(
            float(config.eval_policy.reliability.max_scoring_failure_rate), 6
        ),
        "num_valid_scores": len(scores),
    }
