"""Shared helpers for deterministic expected-focus text in manifests and audits."""

from __future__ import annotations


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _target_focus_label(path_context: dict | None) -> str | None:
    if not isinstance(path_context, dict):
        return None
    for key in ("target_label", "target_instance_id", "target_category"):
        text = _clean_text(path_context.get(key))
        if text:
            return text
    return None


def build_expected_focus_text(*, path_type: str, path_context: dict | None) -> str:
    """Return a deterministic, human/VLM-readable description of clip intent."""
    path_key = str(path_type or "").strip().lower()
    role = ""
    if isinstance(path_context, dict):
        role = str(path_context.get("target_role", "")).strip().lower()
    target_label = _target_focus_label(path_context)

    if role == "targets":
        if target_label:
            return (
                f"Primary target focus: keep {target_label} centered and clearly visible "
                "for most of the clip."
            )
        return "Primary target focus: keep the task target centered and clearly visible."
    if role == "context":
        if target_label:
            return (
                f"Context focus: keep {target_label} visible alongside nearby objects and "
                "interaction affordances."
            )
        return (
            "Context focus: keep the task region visible alongside nearby objects and "
            "interaction affordances."
        )
    if role == "overview":
        return (
            "Overview focus: capture broad scene layout and navigation anchors while preserving "
            "task-relevant context."
        )
    if role == "fallback":
        return (
            "Fallback focus: capture clear, stable scene coverage when explicit task-target "
            "mapping is unavailable."
        )

    if target_label and path_key in {"orbit", "sweep"}:
        return (
            f"Target focus: keep {target_label} centered and clearly visible for most of the clip "
            "while preserving stable scene context."
        )

    if path_key == "manipulation":
        if target_label:
            return (
                f"Manipulation focus: keep {target_label} and its interaction zone in frame with "
                "a stable close-range viewpoint."
            )
        return (
            "Manipulation focus: keep the task object and interaction zone in frame with a stable "
            "close-range viewpoint."
        )
    if path_key == "orbit":
        return "Orbit focus: provide stable global scene coverage for spatial orientation."
    if path_key == "sweep":
        return "Sweep focus: scan across the scene to expose spatial relationships and task regions."
    if path_key == "file":
        return "Path-file focus: follow the predefined camera path with stable, clear framing."
    return "General focus: produce clear, stable scene coverage useful for downstream evaluation."


def resolve_expected_focus_text(clip_entry: dict) -> tuple[str | None, str]:
    """Resolve explicit or derived expected-focus text for a manifest clip."""
    source_text = _clean_text(clip_entry.get("expected_focus_text"))
    if source_text:
        return source_text, source_text
    path_type = str(clip_entry.get("path_type", "") or "")
    path_context = clip_entry.get("path_context", {})
    return (
        None,
        build_expected_focus_text(
            path_type=path_type,
            path_context=path_context if isinstance(path_context, dict) else {},
        ),
    )
