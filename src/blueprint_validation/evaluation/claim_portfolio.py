"""Aggregate investor-grade multi-facility claim signals."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import get_logger, read_json, write_json
from ..config import ValidationConfig
from .claim_benchmark import claim_benchmark_manifest_hash

logger = get_logger("evaluation.claim_portfolio")


def build_claim_portfolio_artifact(config: ValidationConfig, work_dir: Path) -> Path | None:
    if len(config.facilities) < 2:
        return None
    if str(getattr(config.eval_policy, "claim_protocol", "none") or "none").strip().lower() != "fixed_same_facility_uplift":
        return None

    facility_payloads = [
        _facility_claim_payload(
            config=config,
            work_dir=work_dir,
            facility_id=facility_id,
        )
        for facility_id in config.facilities
    ]
    eligible = [payload for payload in facility_payloads if bool(payload.get("eligible", False))]
    portfolio_cfg = config.claim_portfolio

    site_vs_frozen = [
        float(payload["site_vs_frozen_lift_pp"])
        for payload in eligible
        if payload.get("site_vs_frozen_lift_pp") is not None
    ]
    site_vs_generic = [
        float(payload["site_vs_generic_lift_pp"])
        for payload in eligible
        if payload.get("site_vs_generic_lift_pp") is not None
    ]
    frozen_summary = _bootstrap_facility_means(site_vs_frozen)
    generic_summary = _bootstrap_facility_means(site_vs_generic)
    task_family_summary = _aggregate_task_family_summary(eligible)

    gate_failures: List[str] = []
    if len(eligible) < int(portfolio_cfg.min_facilities):
        gate_failures.append(
            f"eligible_facilities={len(eligible)} < min_facilities={int(portfolio_cfg.min_facilities)}"
        )
    if frozen_summary["mean_lift_pp"] is None or float(frozen_summary["mean_lift_pp"]) < float(
        portfolio_cfg.min_mean_site_vs_frozen_lift_pp
    ):
        gate_failures.append(
            "mean site_vs_frozen lift below threshold: "
            f"{frozen_summary['mean_lift_pp']} < {float(portfolio_cfg.min_mean_site_vs_frozen_lift_pp)}"
        )
    if generic_summary["mean_lift_pp"] is None or float(generic_summary["mean_lift_pp"]) < float(
        portfolio_cfg.min_mean_site_vs_generic_lift_pp
    ):
        gate_failures.append(
            "mean site_vs_generic lift below threshold: "
            f"{generic_summary['mean_lift_pp']} < {float(portfolio_cfg.min_mean_site_vs_generic_lift_pp)}"
        )
    if frozen_summary["ci_low_pp"] is None or float(frozen_summary["ci_low_pp"]) <= 0.0:
        gate_failures.append("pooled site_vs_frozen lower CI bound is not > 0")
    if generic_summary["ci_low_pp"] is None or float(generic_summary["ci_low_pp"]) <= 0.0:
        gate_failures.append("pooled site_vs_generic lower CI bound is not > 0")
    if bool(portfolio_cfg.require_manipulation_nonzero):
        manip = task_family_summary.get("manipulation", {})
        if float(manip.get("site_success_rate", 0.0) or 0.0) <= 0.0:
            gate_failures.append("manipulation site_success_rate is zero")
    negative_families = [
        family
        for family, payload in task_family_summary.items()
        if float(payload.get("site_minus_frozen_pp", 0.0) or 0.0)
        < float(portfolio_cfg.max_negative_task_family_delta_pp)
        or float(payload.get("site_minus_generic_pp", 0.0) or 0.0)
        < float(portfolio_cfg.max_negative_task_family_delta_pp)
    ]
    if negative_families:
        gate_failures.append(
            "materially negative task families: " + ", ".join(sorted(negative_families))
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_protocol": "fixed_same_facility_uplift",
        "min_facilities": int(portfolio_cfg.min_facilities),
        "eligible_facility_count": len(eligible),
        "facility_claims": facility_payloads,
        "pooled_site_vs_frozen": frozen_summary,
        "pooled_site_vs_generic": generic_summary,
        "task_family_summary": task_family_summary,
        "go_to_robot_gate": {
            "passed": len(gate_failures) == 0,
            "failures": gate_failures,
            "thresholds": {
                "min_facilities": int(portfolio_cfg.min_facilities),
                "min_mean_site_vs_frozen_lift_pp": float(
                    portfolio_cfg.min_mean_site_vs_frozen_lift_pp
                ),
                "min_mean_site_vs_generic_lift_pp": float(
                    portfolio_cfg.min_mean_site_vs_generic_lift_pp
                ),
                "max_negative_task_family_delta_pp": float(
                    portfolio_cfg.max_negative_task_family_delta_pp
                ),
                "require_manipulation_nonzero": bool(portfolio_cfg.require_manipulation_nonzero),
            },
        },
    }
    output_path = work_dir / "claim_portfolio_report.json"
    write_json(payload, output_path)
    logger.info("Claim portfolio report written to %s", output_path)
    return output_path


def _facility_claim_payload(
    *,
    config: ValidationConfig,
    work_dir: Path,
    facility_id: str,
) -> Dict[str, object]:
    facility = config.facilities[facility_id]
    fac_dir = work_dir / facility_id
    s4d_path = fac_dir / "s4d_policy_pair_eval_result.json"
    s4_path = fac_dir / "s4_policy_eval_result.json"
    dataset_summary_path = config.rollout_dataset.export_dir / facility_id / "dataset_export_summary.json"
    claim_manifest_path = fac_dir / "policy_eval" / "claim_manifest.json"
    claim_split_path = fac_dir / "policy_eval" / "claim_split_manifest.json"
    payload: Dict[str, object] = {
        "facility_id": facility_id,
        "eligible": False,
        "eligibility_failures": [],
    }
    failures: List[str] = []
    if not s4d_path.exists():
        failures.append("missing_s4d")
        payload["eligibility_failures"] = failures
        return payload
    s4d = read_json(s4d_path)
    s4d_metrics = dict(s4d.get("metrics", {}) or {})
    if str(s4d_metrics.get("claim_protocol", "")).strip().lower() != "fixed_same_facility_uplift":
        failures.append("claim_protocol_not_enabled")
    if not bool(s4d_metrics.get("headline_eligible", False)):
        failures.append("headline_ineligible")
    if not bool(s4d_metrics.get("investor_grade_generic_control", False)):
        failures.append("generic_control_not_investor_grade")

    benchmark_hash_reported = ""
    benchmark_hash_current = ""
    if s4_path.exists():
        s4_metrics = dict(read_json(s4_path).get("metrics", {}) or {})
        benchmark_hash_reported = str(s4_metrics.get("claim_benchmark_manifest_hash", "") or "")
        benchmark_path_raw = str(s4_metrics.get("claim_benchmark_path", "") or "")
        benchmark_path = Path(benchmark_path_raw) if benchmark_path_raw else facility.claim_benchmark_path
        if benchmark_path:
            benchmark_hash_current = claim_benchmark_manifest_hash(Path(benchmark_path))
        if benchmark_hash_reported and benchmark_hash_current and benchmark_hash_reported != benchmark_hash_current:
            failures.append("benchmark_hash_drift")
    else:
        failures.append("missing_s4_policy_eval")

    if not claim_manifest_path.exists() or not claim_split_path.exists():
        failures.append("missing_claim_manifests")
    else:
        current_manifest_hash = _json_manifest_hash(claim_manifest_path)
        current_split_hash = _json_manifest_hash(claim_split_path)
        dataset_summary = read_json(dataset_summary_path) if dataset_summary_path.exists() else {}
        dataset_lineage = dict(dataset_summary.get("dataset_lineage", {}) or {})
        if current_manifest_hash != str(dataset_lineage.get("claim_manifest_hash", "") or ""):
            failures.append("claim_manifest_lineage_drift")
        if current_split_hash != str(dataset_lineage.get("claim_split_manifest_hash", "") or ""):
            failures.append("claim_split_lineage_drift")
        current_world_hash = str(read_json(claim_manifest_path).get("world_snapshot_hash", "") or "")
        if current_world_hash != str(s4d_metrics.get("world_snapshot_hash", "") or ""):
            failures.append("moving_world_violation")

    payload.update(
        {
            "claim_outcome": s4d_metrics.get("claim_outcome"),
            "site_vs_frozen_lift_pp": (s4d_metrics.get("bootstrap_site_vs_frozen", {}) or {}).get("mean_lift_pp"),
            "site_vs_generic_lift_pp": (s4d_metrics.get("site_vs_generic_attribution", {}) or {}).get("mean_lift_delta_pp"),
            "generic_control_mode": s4d_metrics.get("generic_control_mode"),
            "task_family_summary": dict(s4d_metrics.get("task_family_summary", {}) or {}),
            "negative_task_families": list(s4d_metrics.get("negative_task_families", []) or []),
            "benchmark_manifest_hash": benchmark_hash_reported,
            "benchmark_manifest_hash_current": benchmark_hash_current,
            "eligible": len(failures) == 0,
            "eligibility_failures": failures,
        }
    )
    return payload


def _bootstrap_facility_means(values: List[float]) -> Dict[str, object]:
    if not values:
        return {"mean_lift_pp": None, "ci_low_pp": None, "ci_high_pp": None, "num_facilities": 0}
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed=0)
    means = [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) for _ in range(2000)]
    return {
        "mean_lift_pp": round(float(np.mean(arr)), 6),
        "ci_low_pp": round(float(np.quantile(means, 0.025)), 6),
        "ci_high_pp": round(float(np.quantile(means, 0.975)), 6),
        "num_facilities": int(len(arr)),
    }


def _aggregate_task_family_summary(eligible_payloads: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    families: Dict[str, List[Dict[str, float]]] = {}
    for payload in eligible_payloads:
        task_family_summary = dict(payload.get("task_family_summary", {}) or {})
        for family, summary in task_family_summary.items():
            if not isinstance(summary, dict):
                continue
            families.setdefault(str(family), []).append(
                {
                    "frozen": float(summary.get("frozen_baseline", 0.0) or 0.0),
                    "generic": float(summary.get("generic_control", 0.0) or 0.0),
                    "site": float(summary.get("site_trained", 0.0) or 0.0),
                }
            )
    aggregated: Dict[str, Dict[str, object]] = {}
    for family, rows in families.items():
        frozen = float(np.mean([row["frozen"] for row in rows])) if rows else 0.0
        generic = float(np.mean([row["generic"] for row in rows])) if rows else 0.0
        site = float(np.mean([row["site"] for row in rows])) if rows else 0.0
        aggregated[family] = {
            "frozen_success_rate": round(frozen, 6),
            "generic_success_rate": round(generic, 6),
            "site_success_rate": round(site, 6),
            "site_minus_frozen_pp": round((site - frozen) * 100.0, 6),
            "site_minus_generic_pp": round((site - generic) * 100.0, 6),
            "num_facilities": len(rows),
        }
    return aggregated


def _json_manifest_hash(path: Path) -> str:
    try:
        payload = json.dumps(read_json(path), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(path)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
