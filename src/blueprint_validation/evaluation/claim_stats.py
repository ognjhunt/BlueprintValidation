"""Statistical helpers for fixed-world claim evaluation."""

from __future__ import annotations

from itertools import product
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
            "p_value_two_sided": None,
            "num_common_eval_cells": 0,
            "per_seed_lift_pp": {},
            "positive_seed_count": 0,
        }

    seed_deltas = _seed_delta_arrays(
        baseline_map=baseline_map,
        arm_rows_by_seed=per_seed_maps,
        common_eval_ids=common,
    )
    return _summarize_seed_deltas(
        seed_deltas=seed_deltas,
        bootstrap_seed=int(bootstrap_seed),
        num_bootstrap=int(num_bootstrap),
    )


def compare_seeded_lifts(
    *,
    baseline_rows: List[Dict[str, object]],
    left_rows_by_seed: Dict[int, List[Dict[str, object]]],
    right_rows_by_seed: Dict[int, List[Dict[str, object]]],
    bootstrap_seed: int,
    num_bootstrap: int = 2000,
) -> Dict[str, object]:
    baseline_map = paired_binary_rows(baseline_rows)
    left_maps = {
        int(seed): paired_binary_rows(rows)
        for seed, rows in left_rows_by_seed.items()
    }
    right_maps = {
        int(seed): paired_binary_rows(rows)
        for seed, rows in right_rows_by_seed.items()
    }
    common_seed_keys = sorted(set(left_maps.keys()) & set(right_maps.keys()))
    if not common_seed_keys or not baseline_map:
        return {
            "mean_lift_delta_pp": None,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "p_value_two_sided": None,
            "positive_seed_count": 0,
            "num_common_eval_cells": 0,
            "per_seed_lift_delta_pp": {},
        }

    common_ids = set(baseline_map.keys())
    for seed in common_seed_keys:
        common_ids &= set(left_maps[seed].keys())
        common_ids &= set(right_maps[seed].keys())
    common = sorted(common_ids)
    if not common:
        return {
            "mean_lift_delta_pp": None,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "p_value_two_sided": None,
            "positive_seed_count": 0,
            "num_common_eval_cells": 0,
            "per_seed_lift_delta_pp": {},
        }

    delta_by_seed: Dict[int, np.ndarray] = {}
    baseline_vals = np.asarray([baseline_map[cell_id] for cell_id in common], dtype=np.float32)
    for seed in common_seed_keys:
        left_vals = np.asarray([left_maps[seed][cell_id] for cell_id in common], dtype=np.float32)
        right_vals = np.asarray([right_maps[seed][cell_id] for cell_id in common], dtype=np.float32)
        delta_by_seed[int(seed)] = (left_vals - baseline_vals) - (right_vals - baseline_vals)

    summary = _summarize_seed_deltas(
        seed_deltas=delta_by_seed,
        bootstrap_seed=int(bootstrap_seed),
        num_bootstrap=int(num_bootstrap),
    )
    return {
        "mean_lift_delta_pp": summary.get("mean_lift_pp"),
        "ci_low_pp": summary.get("ci_low_pp"),
        "ci_high_pp": summary.get("ci_high_pp"),
        "p_value_two_sided": summary.get("p_value_two_sided"),
        "positive_seed_count": summary.get("positive_seed_count"),
        "num_common_eval_cells": summary.get("num_common_eval_cells"),
        "per_seed_lift_delta_pp": summary.get("per_seed_lift_pp", {}),
    }


def _seed_delta_arrays(
    *,
    baseline_map: Dict[str, int],
    arm_rows_by_seed: Dict[int, Dict[str, int]],
    common_eval_ids: List[str],
) -> Dict[int, np.ndarray]:
    seed_deltas: Dict[int, np.ndarray] = {}
    baseline_vals = np.asarray([baseline_map[cell_id] for cell_id in common_eval_ids], dtype=np.float32)
    for seed, rows in arm_rows_by_seed.items():
        arm_vals = np.asarray([rows[cell_id] for cell_id in common_eval_ids], dtype=np.float32)
        seed_deltas[int(seed)] = arm_vals - baseline_vals
    return seed_deltas


def _summarize_seed_deltas(
    *,
    seed_deltas: Dict[int, np.ndarray],
    bootstrap_seed: int,
    num_bootstrap: int,
) -> Dict[str, object]:
    if not seed_deltas:
        return {
            "mean_lift_pp": None,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "p_value_two_sided": None,
            "num_common_eval_cells": 0,
            "per_seed_lift_pp": {},
            "positive_seed_count": 0,
        }

    seed_keys = sorted(seed_deltas.keys())
    num_common = int(seed_deltas[seed_keys[0]].shape[0]) if seed_keys else 0
    per_seed_lift_pp = {
        int(seed): round(float(np.mean(seed_deltas[seed]) * 100.0), 6)
        for seed in seed_keys
    }
    bootstrap_means = _hierarchical_bootstrap_means(
        seed_deltas=seed_deltas,
        bootstrap_seed=int(bootstrap_seed),
        num_bootstrap=int(num_bootstrap),
    )
    ci_low_pp = float(np.quantile(bootstrap_means, 0.025)) if bootstrap_means else None
    ci_high_pp = float(np.quantile(bootstrap_means, 0.975)) if bootstrap_means else None
    p_value_two_sided = _seed_sign_flip_p_value(seed_deltas)
    mean_lift_pp = float(np.mean(list(per_seed_lift_pp.values()))) if per_seed_lift_pp else None
    return {
        "mean_lift_pp": round(mean_lift_pp, 6) if mean_lift_pp is not None else None,
        "ci_low_pp": round(ci_low_pp, 6) if ci_low_pp is not None else None,
        "ci_high_pp": round(ci_high_pp, 6) if ci_high_pp is not None else None,
        "p_value_two_sided": round(p_value_two_sided, 6)
        if p_value_two_sided is not None
        else None,
        "num_common_eval_cells": num_common,
        "per_seed_lift_pp": per_seed_lift_pp,
        "positive_seed_count": sum(1 for val in per_seed_lift_pp.values() if float(val) > 0.0),
    }


def _hierarchical_bootstrap_means(
    *,
    seed_deltas: Dict[int, np.ndarray],
    bootstrap_seed: int,
    num_bootstrap: int,
) -> List[float]:
    seed_keys = sorted(seed_deltas.keys())
    if not seed_keys:
        return []
    rng = np.random.default_rng(seed=int(bootstrap_seed))
    bootstrap_means: List[float] = []
    for _ in range(max(100, int(num_bootstrap))):
        sampled_seed_indices = rng.integers(0, len(seed_keys), size=len(seed_keys))
        sampled_seed_means: List[float] = []
        for idx in sampled_seed_indices:
            deltas = np.asarray(seed_deltas[seed_keys[int(idx)]], dtype=np.float32)
            if deltas.size <= 1:
                sampled_seed_means.append(float(np.mean(deltas)))
                continue
            sample_idx = rng.integers(0, deltas.size, size=deltas.size)
            sampled_seed_means.append(float(np.mean(deltas[sample_idx])))
        bootstrap_means.append(float(np.mean(sampled_seed_means) * 100.0))
    return bootstrap_means


def _seed_sign_flip_p_value(seed_deltas: Dict[int, np.ndarray]) -> float | None:
    seed_means = np.asarray(
        [float(np.mean(np.asarray(deltas, dtype=np.float32))) for deltas in seed_deltas.values()],
        dtype=np.float64,
    )
    if seed_means.size == 0:
        return None
    observed = abs(float(np.mean(seed_means)))
    if seed_means.size <= 12:
        permuted = [
            abs(float(np.mean(seed_means * np.asarray(signs, dtype=np.float64))))
            for signs in product((-1.0, 1.0), repeat=int(seed_means.size))
        ]
    else:
        rng = np.random.default_rng(seed=0)
        permuted = []
        for _ in range(5000):
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=seed_means.size)
            permuted.append(abs(float(np.mean(seed_means * signs))))
    extreme = sum(1 for value in permuted if float(value) >= observed)
    return float((extreme + 1) / (len(permuted) + 1))
