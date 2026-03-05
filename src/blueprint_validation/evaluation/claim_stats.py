"""Statistical helpers for fixed-world claim evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def success_rate(rows: Iterable[Dict[str, object]]) -> float:
    vals = [1.0 if bool(row.get("task_success", False)) else 0.0 for row in rows]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def paired_binary_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    paired: Dict[str, int] = {}
    for row in rows:
        cell_id = str(row.get("eval_cell_id", "")).strip()
        if not cell_id:
            continue
        paired[cell_id] = 1 if bool(row.get("task_success", False)) else 0
    return paired


def bootstrap_site_vs_baseline(
    *,
    baseline_rows: List[Dict[str, object]],
    site_rows_by_seed: Dict[int, List[Dict[str, object]]],
    bootstrap_seed: int,
    num_bootstrap: int = 2000,
) -> Dict[str, object]:
    baseline_map = paired_binary_rows(baseline_rows)
    per_seed_maps = {
        int(seed): paired_binary_rows(rows)
        for seed, rows in site_rows_by_seed.items()
    }
    common_ids = set(baseline_map.keys())
    for rows in per_seed_maps.values():
        common_ids &= set(rows.keys())
    common = sorted(common_ids)
    if not common:
        return {
            "mean_lift_pp": None,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "num_common_eval_cells": 0,
            "per_seed_lift_pp": {},
            "positive_seed_count": 0,
        }

    per_seed_lift_pp: Dict[int, float] = {}
    seed_deltas: Dict[int, np.ndarray] = {}
    baseline_vals = np.asarray([baseline_map[cell_id] for cell_id in common], dtype=np.float32)
    for seed, rows in per_seed_maps.items():
        site_vals = np.asarray([rows[cell_id] for cell_id in common], dtype=np.float32)
        deltas = site_vals - baseline_vals
        seed_deltas[int(seed)] = deltas
        per_seed_lift_pp[int(seed)] = round(float(np.mean(deltas) * 100.0), 6)

    rng = np.random.default_rng(seed=int(bootstrap_seed))
    seed_keys = sorted(seed_deltas.keys())
    bootstrap_means: List[float] = []
    if len(common) == 1:
        pooled = float(np.mean([float(seed_deltas[seed][0]) for seed in seed_keys]) * 100.0)
        bootstrap_means = [pooled]
    else:
        for _ in range(max(100, int(num_bootstrap))):
            sample_idx = rng.integers(0, len(common), size=len(common))
            sampled_seed_means = []
            for seed in seed_keys:
                sampled_seed_means.append(float(np.mean(seed_deltas[seed][sample_idx])))
            bootstrap_means.append(float(np.mean(sampled_seed_means) * 100.0))
    mean_lift_pp = float(np.mean(list(per_seed_lift_pp.values()))) if per_seed_lift_pp else None
    ci_low_pp = float(np.quantile(bootstrap_means, 0.025)) if bootstrap_means else None
    ci_high_pp = float(np.quantile(bootstrap_means, 0.975)) if bootstrap_means else None
    return {
        "mean_lift_pp": round(mean_lift_pp, 6) if mean_lift_pp is not None else None,
        "ci_low_pp": round(ci_low_pp, 6) if ci_low_pp is not None else None,
        "ci_high_pp": round(ci_high_pp, 6) if ci_high_pp is not None else None,
        "num_common_eval_cells": len(common),
        "per_seed_lift_pp": per_seed_lift_pp,
        "positive_seed_count": sum(1 for val in per_seed_lift_pp.values() if float(val) > 0.0),
    }
