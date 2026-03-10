"""Validation helpers for post-qualification opportunity handoffs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from ..common import read_json


class QualifiedOpportunityValidationError(RuntimeError):
    """Raised when a qualified opportunity handoff is invalid."""


_ALLOWED_QUALIFICATION_STATES = {"ready", "risky", "not_ready_yet"}


def _as_mapping(payload: Any, *, manifest_path: Path | None) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        where = f" ({manifest_path})" if manifest_path is not None else ""
        raise QualifiedOpportunityValidationError(
            f"Qualified opportunity handoff must be a JSON object{where}"
        )
    return dict(payload)


def _require_text(payload: Mapping[str, Any], key: str, *, where: str) -> str:
    value = str(payload.get(key, "") or "").strip()
    if not value:
        raise QualifiedOpportunityValidationError(f"Missing non-empty '{key}'{where}")
    return value


def _require_bool(payload: Mapping[str, Any], key: str, *, where: str) -> bool:
    if key not in payload or not isinstance(payload.get(key), bool):
        raise QualifiedOpportunityValidationError(f"Missing boolean '{key}'{where}")
    return bool(payload[key])


def _require_mapping(payload: Mapping[str, Any], key: str, *, where: str) -> Dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise QualifiedOpportunityValidationError(f"Missing object '{key}'{where}")
    return dict(value)


def _require_present(value: Any, key: str, *, where: str) -> None:
    if value is None:
        raise QualifiedOpportunityValidationError(f"Missing '{key}'{where}")
    if isinstance(value, str) and not value.strip():
        raise QualifiedOpportunityValidationError(f"Missing non-empty '{key}'{where}")
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        raise QualifiedOpportunityValidationError(f"Missing non-empty '{key}'{where}")


def _normalize_qualification_state(raw_value: Any, *, where: str) -> str:
    qualification_state = str(raw_value or "").strip().lower()
    if qualification_state not in _ALLOWED_QUALIFICATION_STATES:
        allowed = ", ".join(sorted(_ALLOWED_QUALIFICATION_STATES))
        raise QualifiedOpportunityValidationError(
            f"qualification_state must be one of: {allowed}{where}"
        )
    return qualification_state


def _optional_mapping(payload: Mapping[str, Any], key: str, *, where: str) -> Dict[str, Any] | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, Mapping):
        raise QualifiedOpportunityValidationError(f"Missing object '{key}'{where}")
    return dict(value)


def _looks_like_capture_pipeline_handoff(payload: Mapping[str, Any]) -> bool:
    return all(key in payload for key in ("scene_id", "capture_id", "readiness_state", "match_ready"))


def _looks_like_rich_handoff(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "qualification_state",
            "downstream_evaluation_eligibility",
            "scoped_task_definition",
            "target_robot_team",
        )
    )


def _validate_rich_handoff(data: Mapping[str, Any], *, where: str) -> Dict[str, Any]:
    normalized = dict(data)
    site_submission_id = _require_text(normalized, "site_submission_id", where=where)
    opportunity_id = _require_text(normalized, "opportunity_id", where=where)
    qualification_state = _normalize_qualification_state(
        _require_text(normalized, "qualification_state", where=where),
        where=where,
    )
    downstream_evaluation_eligibility = _require_bool(
        normalized,
        "downstream_evaluation_eligibility",
        where=where,
    )
    operator_approved_summary = _require_text(normalized, "operator_approved_summary", where=where)

    scoped_task = _require_mapping(normalized, "scoped_task_definition", where=where)
    _require_text(scoped_task, "task_id", where=f"{where} in scoped_task_definition")
    _require_text(
        scoped_task,
        "scoped_task_statement",
        where=f"{where} in scoped_task_definition",
    )
    _require_present(
        scoped_task.get("success_criteria"),
        "success_criteria",
        where=f"{where} in scoped_task_definition",
    )
    _require_present(
        scoped_task.get("in_scope_zone"),
        "in_scope_zone",
        where=f"{where} in scoped_task_definition",
    )

    site_constraints = _require_mapping(normalized, "site_constraints", where=where)
    for key in (
        "operating_constraints",
        "privacy_security_constraints",
        "known_blockers",
    ):
        _require_present(
            site_constraints.get(key),
            key,
            where=f"{where} in site_constraints",
        )

    target_robot_team = _require_mapping(normalized, "target_robot_team", where=where)
    for key in ("team_name_or_id", "robot_platform", "embodiment_notes"):
        _require_text(
            target_robot_team,
            key,
            where=f"{where} in target_robot_team",
        )

    for optional_mapping in ("geometry_package", "scene_package"):
        _optional_mapping(normalized, optional_mapping, where=where)

    normalized["site_submission_id"] = site_submission_id
    normalized["opportunity_id"] = opportunity_id
    normalized["qualification_state"] = qualification_state
    normalized["downstream_evaluation_eligibility"] = downstream_evaluation_eligibility
    normalized["operator_approved_summary"] = operator_approved_summary
    return normalized


def _validate_capture_pipeline_handoff(data: Mapping[str, Any], *, where: str) -> Dict[str, Any]:
    normalized = dict(data)
    scene_id = _require_text(normalized, "scene_id", where=where)
    capture_id = _require_text(normalized, "capture_id", where=where)
    qualification_state = _normalize_qualification_state(
        _require_text(normalized, "readiness_state", where=where),
        where=where,
    )
    downstream_evaluation_eligibility = _require_bool(normalized, "match_ready", where=where)
    operator_approved_summary = (
        str(normalized.get("summary", "") or "").strip()
        or f"BlueprintCapturePipeline handoff for scene {scene_id} capture {capture_id}"
    )

    for optional_mapping in ("constraints", "geometry_package", "scene_package"):
        _optional_mapping(normalized, optional_mapping, where=where)

    normalized["site_submission_id"] = capture_id
    normalized["opportunity_id"] = scene_id
    normalized["qualification_state"] = qualification_state
    normalized["downstream_evaluation_eligibility"] = downstream_evaluation_eligibility
    normalized["operator_approved_summary"] = operator_approved_summary
    return normalized


def validate_qualified_opportunity_handoff(
    payload: Any,
    *,
    manifest_path: Path | None = None,
) -> Dict[str, Any]:
    """Validate the qualified opportunity handoff contract."""
    where = f" ({manifest_path})" if manifest_path is not None else ""
    data = _as_mapping(payload, manifest_path=manifest_path)

    schema_version = _require_text(data, "schema_version", where=where)
    if schema_version != "v1":
        raise QualifiedOpportunityValidationError(
            f"Unsupported qualified opportunity schema_version '{schema_version}'{where}"
        )
    if _looks_like_rich_handoff(data):
        return _validate_rich_handoff(data, where=where)
    if _looks_like_capture_pipeline_handoff(data):
        return _validate_capture_pipeline_handoff(data, where=where)
    raise QualifiedOpportunityValidationError(
        "Qualified opportunity handoff must include either the rich downstream fields "
        "(qualification_state, downstream_evaluation_eligibility, scoped_task_definition, "
        "site_constraints, target_robot_team) or the BlueprintCapturePipeline fields "
        f"(scene_id, capture_id, readiness_state, match_ready){where}"
    )


def load_and_validate_qualified_opportunity_handoff(path: Path) -> Dict[str, Any]:
    """Load a handoff JSON file and validate the contract."""
    payload = read_json(path)
    return validate_qualified_opportunity_handoff(payload, manifest_path=path)
