"""Compatibility wrappers for the shared qualified opportunity contract."""

from __future__ import annotations

from blueprint_contracts.handoff_contract import (
    QualifiedOpportunityValidationError,
    load_and_validate_qualified_opportunity_handoff,
    validate_qualified_opportunity_handoff,
)

__all__ = [
    "QualifiedOpportunityValidationError",
    "load_and_validate_qualified_opportunity_handoff",
    "validate_qualified_opportunity_handoff",
]
