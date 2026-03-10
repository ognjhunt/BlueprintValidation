"""Tests for qualified opportunity handoff validation."""

from __future__ import annotations

import json

import pytest


def _valid_handoff() -> dict:
    return {
        "schema_version": "v1",
        "site_submission_id": "site-sub-001",
        "opportunity_id": "opp-001",
        "qualification_state": "ready",
        "downstream_evaluation_eligibility": True,
        "operator_approved_summary": "Qualified warehouse tote-pick lane",
        "scoped_task_definition": {
            "task_id": "task-001",
            "scoped_task_statement": "Pick tote from shelf bay 3",
            "success_criteria": ["grasp tote", "clear shelf"],
            "in_scope_zone": "bay-3",
        },
        "site_constraints": {
            "operating_constraints": ["night shift only"],
            "privacy_security_constraints": ["no worker faces"],
            "known_blockers": ["reflective wrap on two pallets"],
        },
        "target_robot_team": {
            "team_name_or_id": "team-alpha",
            "robot_platform": "franka_panda",
            "embodiment_notes": "fixed-base arm on mobile cart",
        },
    }


def test_validate_qualified_opportunity_handoff_accepts_ready_payload():
    from blueprint_validation.validation import validate_qualified_opportunity_handoff

    payload = validate_qualified_opportunity_handoff(_valid_handoff())
    assert payload["qualification_state"] == "ready"


def test_validate_qualified_opportunity_handoff_accepts_non_ready_states():
    from blueprint_validation.validation import validate_qualified_opportunity_handoff

    for state in ("risky", "not_ready_yet"):
        payload = _valid_handoff()
        payload["qualification_state"] = state
        validated = validate_qualified_opportunity_handoff(payload)
        assert validated["qualification_state"] == state


def test_validate_qualified_opportunity_handoff_rejects_missing_scoped_task_fields():
    from blueprint_validation.validation import QualifiedOpportunityValidationError
    from blueprint_validation.validation import validate_qualified_opportunity_handoff

    payload = _valid_handoff()
    del payload["scoped_task_definition"]["task_id"]

    with pytest.raises(QualifiedOpportunityValidationError, match="task_id"):
        validate_qualified_opportunity_handoff(payload)


def test_validate_qualified_opportunity_handoff_rejects_missing_constraints():
    from blueprint_validation.validation import QualifiedOpportunityValidationError
    from blueprint_validation.validation import validate_qualified_opportunity_handoff

    payload = _valid_handoff()
    del payload["site_constraints"]["privacy_security_constraints"]

    with pytest.raises(QualifiedOpportunityValidationError, match="privacy_security_constraints"):
        validate_qualified_opportunity_handoff(payload)


def test_validate_qualified_opportunity_handoff_rejects_missing_robot_team_fields():
    from blueprint_validation.validation import QualifiedOpportunityValidationError
    from blueprint_validation.validation import validate_qualified_opportunity_handoff

    payload = _valid_handoff()
    del payload["target_robot_team"]["robot_platform"]

    with pytest.raises(QualifiedOpportunityValidationError, match="robot_platform"):
        validate_qualified_opportunity_handoff(payload)


def test_load_and_validate_qualified_opportunity_handoff_from_disk(tmp_path):
    from blueprint_validation.validation import load_and_validate_qualified_opportunity_handoff

    handoff_path = tmp_path / "opportunity_handoff.json"
    handoff_path.write_text(json.dumps(_valid_handoff()))

    payload = load_and_validate_qualified_opportunity_handoff(handoff_path)
    assert payload["opportunity_id"] == "opp-001"
