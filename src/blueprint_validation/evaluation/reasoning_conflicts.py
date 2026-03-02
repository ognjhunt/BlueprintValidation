"""Helpers for detecting reasoning/score consistency conflicts in VLM outputs."""

from __future__ import annotations

from typing import Iterable, Tuple

# Keep this token family explicit and conservative so fail-closed behavior is deterministic.
DEFAULT_REASONING_CONFLICT_TOKENS: Tuple[str, ...] = (
    "cannot evaluate",
    "can't evaluate",
    "unable to evaluate",
    "impossible to evaluate",
    "not visible",
    "no landmarks visible",
    "target not visible",
    "too distorted",
    "distorted to evaluate",
    "indiscernible",
    "unclear",
    "unusable",
)


def has_reasoning_conflict(
    reasoning: str | None,
    *,
    extra_tokens: Iterable[str] | None = None,
) -> bool:
    """Return True when reasoning text indicates the sample is not reliably evaluable."""
    text = str(reasoning or "").strip().lower()
    if not text:
        return False
    tokens = list(DEFAULT_REASONING_CONFLICT_TOKENS)
    if extra_tokens is not None:
        tokens.extend(str(token).strip().lower() for token in extra_tokens if str(token).strip())
    return any(token in text for token in tokens)
