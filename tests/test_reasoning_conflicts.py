from __future__ import annotations

from blueprint_validation.evaluation.reasoning_conflicts import has_reasoning_conflict


def test_has_reasoning_conflict_detects_unusable_reasoning():
    assert has_reasoning_conflict("The sample is too distorted and cannot evaluate landmarks.")


def test_has_reasoning_conflict_ignores_clean_reasoning():
    assert not has_reasoning_conflict("Task target remains visible and spatially consistent.")
