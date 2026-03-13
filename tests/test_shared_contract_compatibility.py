from __future__ import annotations

from blueprint_contracts.handoff_contract import (
    validate_qualified_opportunity_handoff as shared_validate_handoff,
)
from blueprint_contracts.site_world_contract import (
    load_site_world_bundle as shared_load_site_world_bundle,
)
from blueprint_validation.site_world_intake import load_site_world_bundle
from blueprint_validation.validation.qualified_opportunity import (
    validate_qualified_opportunity_handoff,
)


def test_validation_handoff_wrapper_points_at_shared_contract() -> None:
    assert validate_qualified_opportunity_handoff is shared_validate_handoff


def test_validation_site_world_wrapper_points_at_shared_contract() -> None:
    assert load_site_world_bundle is shared_load_site_world_bundle
