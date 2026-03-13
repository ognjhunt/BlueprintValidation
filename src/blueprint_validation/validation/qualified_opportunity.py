"""Compatibility wrappers for the shared qualified opportunity contract.

Ownership for the handoff schema and validation logic stays in BlueprintContracts. This
module exists only to preserve older BlueprintValidation import paths.
"""

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
