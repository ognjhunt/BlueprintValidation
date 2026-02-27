"""Built-in visual variant definitions for Cosmos Transfer enrichment."""

from __future__ import annotations

from ..config import VariantSpec

BUILTIN_VARIANTS = [
    VariantSpec(
        name="daylight_empty",
        prompt="Indoor industrial facility, bright natural daylight, empty clean corridors, polished concrete floors, overhead LED lighting",
    ),
    VariantSpec(
        name="daylight_occupied",
        prompt="Indoor industrial facility, bright daylight, workers in safety vests moving through space, forklifts, active workspace",
    ),
    VariantSpec(
        name="evening_dim",
        prompt="Indoor industrial facility, dim evening lighting, overhead fluorescent lights casting shadows, quiet empty space",
    ),
    VariantSpec(
        name="wet_floor",
        prompt="Indoor industrial facility, wet reflective concrete floors, caution signs, overhead lighting reflecting off puddles",
    ),
    VariantSpec(
        name="cluttered",
        prompt="Indoor industrial facility, boxes and pallets scattered throughout, equipment and supplies visible, busy workspace",
    ),
]


def get_variants(custom_variants: list[VariantSpec] | None = None) -> list[VariantSpec]:
    """Return variant specs, preferring custom variants over builtins."""
    if custom_variants:
        return custom_variants
    return BUILTIN_VARIANTS
