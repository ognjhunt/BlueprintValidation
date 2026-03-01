"""Deterministic scripted rollout driver for world-model-only evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from ..common import get_logger

logger = get_logger("evaluation.scripted_rollout_driver")


def build_scripted_trace_manifest(
    assignments: List[dict],
    *,
    action_dim: int,
    max_steps: int,
) -> Dict[str, dict]:
    """Build deterministic scripted traces keyed by rollout assignment id."""
    manifest: Dict[str, dict] = {}
    for assignment in assignments:
        key = _assignment_key(assignment)
        task = str(assignment.get("task", ""))
        seed = _stable_seed_from_key(key)
        actions, trace_type = build_scripted_action_trace(
            task=task,
            action_dim=action_dim,
            max_steps=max_steps,
            seed=seed,
        )
        trace_id = _trace_id(
            key=key,
            action_dim=action_dim,
            max_steps=max_steps,
            trace_type=trace_type,
        )
        manifest[key] = {
            "trace_id": trace_id,
            "trace_type": trace_type,
            "action_sequence": actions,
            "seed": seed,
        }
    return manifest


def build_scripted_action_trace(
    *,
    task: str,
    action_dim: int,
    max_steps: int,
    seed: int,
) -> tuple[List[list[float]], str]:
    """Return deterministic scripted actions for a task family."""
    dim = int(action_dim)
    if dim <= 0:
        raise ValueError(f"action_dim must be > 0, got {action_dim}")
    steps = max(1, int(max_steps))
    rng = np.random.default_rng(seed)
    is_manip = _is_manipulation_task(task)
    trace_type = "manipulation_scripted" if is_manip else "navigation_scripted"

    yaw_axis = 5 if dim > 5 else min(2, dim - 1)
    grip_axis = 6 if dim > 6 else dim - 1
    x_axis = 0
    y_axis = 1 if dim > 1 else 0
    z_axis = 2 if dim > 2 else y_axis

    phase = float(rng.uniform(0.0, np.pi))
    base = float(rng.uniform(0.012, 0.03))
    lateral = float(rng.uniform(0.004, 0.015))
    yaw_gain = float(rng.uniform(0.01, 0.03))
    z_gain = float(rng.uniform(0.005, 0.02))
    noise_scale = float(rng.uniform(0.0, 0.0015))

    actions: List[list[float]] = []
    for step in range(steps):
        vec = np.zeros((dim,), dtype=np.float32)
        t = float(step)
        vec[x_axis] = base * np.cos(0.25 * t + phase)
        vec[y_axis] = lateral * np.sin(0.2 * t + phase / 2.0)
        vec[yaw_axis] = yaw_gain * np.sin(0.18 * t + phase)

        if is_manip:
            half = max(1, steps // 2)
            vec[z_axis] = z_gain if step < half else -z_gain
            if dim > 6:
                vec[grip_axis] = -1.0 if step < (steps // 3) else 1.0
        else:
            vec[z_axis] = 0.003 * np.sin(0.4 * t + phase)
            if dim > 6:
                vec[grip_axis] = -1.0

        # Keep deterministic but non-degenerate motion in high dimensions.
        if dim > 8:
            tail = min(dim, 16)
            vec[7:tail] = noise_scale * rng.standard_normal((tail - 7,))

        actions.append(vec.astype(np.float32).tolist())
    return actions, trace_type


def run_scripted_rollout(
    *,
    world_model,
    initial_frame: np.ndarray,
    action_sequence: List[list[float]],
    output_dir: Path,
    clip_name: str,
    trace_id: str,
    reanchor_every: int | None = None,
) -> SimpleNamespace:
    """Execute scripted action sequence in world model and write rollout video."""
    import cv2

    frames = [initial_frame.copy()]
    current = initial_frame
    for step_idx, action in enumerate(action_sequence):
        next_frame = world_model.predict_next_frame(current, action)
        frames.append(next_frame)
        current = next_frame
        if reanchor_every and reanchor_every > 0 and (step_idx + 1) % reanchor_every == 0:
            current = frames[-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{clip_name}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    world_dim = getattr(world_model, "expected_action_dim", None)
    if world_dim is None:
        world_dim = getattr(world_model, "_expected_action_dim", None)
    dataset_dim = len(action_sequence[0]) if action_sequence else None
    compliant = (
        world_dim is None or dataset_dim is None or int(world_dim) == int(dataset_dim)
    )

    return SimpleNamespace(
        video_path=video_path,
        action_sequence=action_sequence,
        num_steps=len(action_sequence),
        trace_id=trace_id,
        driver_type="scripted",
        action_contract={
            "policy_dim": None,
            "world_dim": int(world_dim) if world_dim is not None else None,
            "dataset_dim": int(dataset_dim) if dataset_dim is not None else None,
            "compliant": bool(compliant),
            "reason": "" if compliant else "scripted trace dim does not match world-model dim",
        },
    )


def _assignment_key(assignment: dict) -> str:
    return (
        f"{assignment.get('rollout_index', 0)}::"
        f"{assignment.get('clip_index', -1)}::"
        f"{assignment.get('task', '')}"
    )


def _stable_seed_from_key(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def _trace_id(*, key: str, action_dim: int, max_steps: int, trace_type: str) -> str:
    digest = hashlib.md5(
        f"{key}|{action_dim}|{max_steps}|{trace_type}".encode("utf-8")
    ).hexdigest()
    return f"trace_{digest[:12]}"


def _is_manipulation_task(task: str) -> bool:
    lowered = (task or "").lower()
    keywords = ("pick", "grasp", "lift", "place", "stack", "regrasp", "tote", "bin")
    return any(k in lowered for k in keywords)

