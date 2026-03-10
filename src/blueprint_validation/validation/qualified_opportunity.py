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

    _require_text(data, "site_submission_id", where=where)
    _require_text(data, "opportunity_id", where=where)
    qualification_state = _require_text(data, "qualification_state", where=where).lower()
    if qualification_state not in _ALLOWED_QUALIFICATION_STATES:
        allowed = ", ".join(sorted(_ALLOWED_QUALIFICATION_STATES))
        raise QualifiedOpportunityValidationError(
            f"qualification_state must be one of: {allowed}{where}"
        )
    _require_bool(data, "downstream_evaluation_eligibility", where=where)
    _require_text(data, "operator_approved_summary", where=where)

    scoped_task = _require_mapping(data, "scoped_task_definition", where=where)
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

    site_constraints = _require_mapping(data, "site_constraints", where=where)
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

    target_robot_team = _require_mapping(data, "target_robot_team", where=where)
    for key in ("team_name_or_id", "robot_platform", "embodiment_notes"):
        _require_text(
            target_robot_team,
            key,
            where=f"{where} in target_robot_team",
        )

    for optional_mapping in ("geometry_package", "scene_package"):
        if optional_mapping in data and data[optional_mapping] is not None:
            _require_mapping(data, optional_mapping, where=where)

    return data


def load_and_validate_qualified_opportunity_handoff(path: Path) -> Dict[str, Any]:
    """Load a handoff JSON file and validate the contract."""
    payload = read_json(path)
    return validate_qualified_opportunity_handoff(payload, manifest_path=path)
