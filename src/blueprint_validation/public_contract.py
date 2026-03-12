"""Customer-facing contract helpers for evaluation and hosted-session surfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


_PUBLIC_RUNTIME_LABELS = {
    "neoverse": "Hosted site runtime",
    "neoverse_service": "NeoVerse runtime service",
    "gen3c": "Scene-memory runtime",
    "cosmos_transfer": "Transfer-backed runtime",
    "isaac_scene": "Isaac Sim strict scene",
    "gsplat": "Geometry preview runtime",
    "robosplat": "Geometry preview runtime",
}


def public_runtime_label(backend: object) -> str:
    key = str(backend or "").strip().lower()
    if not key:
        return "Qualified site runtime"
    return _PUBLIC_RUNTIME_LABELS.get(key, key.replace("_", " ").title())


def _failure_taxonomy(stage_name: str, stage_payload: Mapping[str, Any]) -> List[str]:
    metrics = stage_payload.get("metrics") if isinstance(stage_payload.get("metrics"), Mapping) else {}
    detail = str(stage_payload.get("detail") or "").strip()
    failure_reasons: List[str] = []
    for key in ("claim_failure_reasons", "structural_failure_reasons"):
        value = metrics.get(key)
        if isinstance(value, list):
            failure_reasons.extend(str(item).strip() for item in value if str(item).strip())
    if detail:
        failure_reasons.append(detail)
    return failure_reasons[:6] or [f"{stage_name}: no detailed failure taxonomy emitted"]


def build_standardized_eval_report(report_data: Mapping[str, Any]) -> Dict[str, Any]:
    facilities = report_data.get("facilities") if isinstance(report_data.get("facilities"), Mapping) else {}
    comparisons: List[Dict[str, Any]] = []

    for facility_id, facility_payload in facilities.items():
        if not isinstance(facility_payload, Mapping):
            continue
        runtime_stage = facility_payload.get("s0b_scene_memory_runtime")
        runtime_outputs = runtime_stage.get("outputs", {}) if isinstance(runtime_stage, Mapping) else {}
        runtime_backend = runtime_outputs.get("selected_backend")
        runtime_label = public_runtime_label(runtime_backend)

        for stage_name, metric_map in (
            ("s4_policy_eval", {
                "mean_task_score_a": "baseline_mean_task_score",
                "mean_task_score_b": "adapted_mean_task_score",
                "win_rate": "win_rate",
                "p_value": "p_value",
                "task_success_proxy": "adapted_mean_task_score",
            }),
            ("s4e_trained_eval", {
                "mean_task_score_a": "frozen_mean_task_score",
                "mean_task_score_b": "trained_mean_task_score",
                "win_rate": "win_rate",
                "p_value": "p_value",
                "task_success_proxy": "trained_manipulation_success_rate",
            }),
            ("s4d_policy_pair_eval", {
                "mean_task_score_a": "policy_base_mean_task_score",
                "mean_task_score_b": "policy_site_mean_task_score",
                "win_rate": "win_rate",
                "p_value": "p_value",
                "task_success_proxy": "task_success",
            }),
            ("s4f_polaris_eval", {
                "mean_task_score_a": "frozen_policy_success",
                "mean_task_score_b": "adapted_policy_success",
                "win_rate": "delta_vs_frozen",
                "p_value": "p_value",
                "task_success_proxy": "winner",
            }),
        ):
            payload = facility_payload.get(stage_name)
            if not isinstance(payload, Mapping):
                continue
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
            outputs = payload.get("outputs") if isinstance(payload.get("outputs"), Mapping) else {}
            comparisons.append(
                {
                    "comparison_id": f"{facility_id}:{stage_name}",
                    "facility_id": str(facility_id),
                    "stage_name": stage_name,
                    "status": str(payload.get("status") or "unknown"),
                    "runtime_backend_internal": str(runtime_backend or ""),
                    "runtime_backend_public": runtime_label,
                    "report_path": str(
                        outputs.get("report_path")
                        or outputs.get("claim_report_path")
                        or outputs.get("polaris_summary_path")
                        or ""
                    ),
                    "metrics": {
                        "mean_task_score_a": metrics.get(metric_map["mean_task_score_a"]),
                        "mean_task_score_b": metrics.get(metric_map["mean_task_score_b"]),
                        "win_rate": metrics.get(metric_map["win_rate"]),
                        "p_value": metrics.get(metric_map["p_value"]),
                        "task_success_proxy": metrics.get(metric_map["task_success_proxy"]),
                    },
                    "failure_taxonomy": _failure_taxonomy(stage_name, payload),
                }
            )

    return {
        "schema_version": "v1",
        "generated_at": report_data.get("generated_at"),
        "project_name": report_data.get("project_name"),
        "comparisons": comparisons,
        "failure_taxonomy_reference": [
            "claim_failure_reasons",
            "structural_failure_reasons",
            "stage_detail",
        ],
    }
