"""Compatibility wrappers for the shared site-world contract.

Ownership for portable site-world loading/validation stays in BlueprintContracts. This
module exists only to preserve older BlueprintValidation import paths.
"""

from __future__ import annotations

from blueprint_contracts.site_world_contract import (
    SiteWorldBundle,
    SiteWorldIntakeError,
    adjacent_site_world_paths,
    grounding_summary,
    load_site_world_bundle,
    merge_site_world_definition,
    normalize_trajectory_payload,
)

__all__ = [
    "SiteWorldBundle",
    "SiteWorldIntakeError",
    "adjacent_site_world_paths",
    "grounding_summary",
    "load_site_world_bundle",
    "merge_site_world_definition",
    "normalize_trajectory_payload",
]
