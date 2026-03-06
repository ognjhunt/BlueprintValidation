"""Helpers for capturing rollout task state from world-model runtimes."""

from __future__ import annotations

from typing import Any


def world_model_supports_native_task_state(world_model: object) -> bool:
    """Return true when the world model exposes a native task-state interface."""
    return callable(getattr(world_model, "capture_rollout_state", None)) or callable(
        getattr(world_model, "extract_task_state", None)
    )


def capture_task_state(
    *,
    world_model: object,
    frame: Any,
    action: Any,
    step_idx: int,
    phase: str,
    task_prompt: str | None = None,
    fallback_proxy: object | None = None,
    require_native_task_state: bool = False,
) -> dict | None:
    """Capture task state, optionally refusing non-native fallbacks."""
    capture = getattr(world_model, "capture_rollout_state", None)
    payload = _call_capture_fn(
        capture,
        frame=frame,
        action=action,
        step_idx=step_idx,
        phase=phase,
        task_prompt=task_prompt,
        state_source="native_world_model.capture_rollout_state",
    )
    if payload is not None:
        return payload

    extract = getattr(world_model, "extract_task_state", None)
    payload = _call_capture_fn(
        extract,
        frame=frame,
        action=action,
        step_idx=step_idx,
        phase=phase,
        task_prompt=task_prompt,
        state_source="native_world_model.extract_task_state",
    )
    if payload is not None:
        return payload

    if require_native_task_state or fallback_proxy is None:
        return None

    try:
        payload = fallback_proxy.capture(action=action, step_idx=step_idx, phase=phase)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    normalized = dict(payload)
    normalized.setdefault("state_source", "deterministic_rollout_state_proxy")
    return normalized


def _call_capture_fn(
    fn: object,
    *,
    frame: Any,
    action: Any,
    step_idx: int,
    phase: str,
    task_prompt: str | None,
    state_source: str,
) -> dict | None:
    if not callable(fn):
        return None
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    if task_prompt is not None:
        attempts.append(
            (
                tuple(),
                {
                    "frame": frame,
                    "action": action,
                    "step_idx": step_idx,
                    "phase": phase,
                    "task_prompt": task_prompt,
                },
            )
        )
    attempts.append(
        (
            tuple(),
            {
                "frame": frame,
                "action": action,
                "step_idx": step_idx,
                "phase": phase,
            },
        )
    )
    attempts.append((tuple(), {"frame": frame, "action": action, "step_idx": step_idx}))

    for args, kwargs in attempts:
        try:
            payload = fn(*args, **kwargs)
        except TypeError:
            continue
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        normalized = dict(payload)
        normalized.setdefault("state_source", state_source)
        return normalized
    return None
