"""Shared rollout execution utilities for adapter-based policy rollouts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np


def run_rollout_with_adapter(
    world_model,
    policy_adapter,
    policy_handle,
    initial_frame: np.ndarray,
    task_prompt: str,
    max_steps: int,
    unnorm_key: str,
    output_dir: Path,
    clip_name: str,
    device: str,
    expected_action_dim: Optional[int] = None,
    reanchor_every: Optional[int] = None,
) -> SimpleNamespace:
    """Run a policy rollout in a world model using a generic policy adapter.

    Returns a SimpleNamespace with ``video_path``, ``action_sequence``, and
    ``num_steps`` — matching the contract of ``run_rollout()`` from
    ``openvla_runner.py``.
    """
    import cv2

    frames = [initial_frame.copy()]
    actions: List[list] = []
    current = initial_frame
    for step_idx in range(max_steps):
        action = policy_adapter.predict_action(
            handle=policy_handle,
            frame=current,
            task_prompt=task_prompt,
            unnorm_key=unnorm_key,
            device=device,
        )
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        if expected_action_dim is not None and len(action_list) != int(expected_action_dim):
            raise RuntimeError(
                "Action-space mismatch at rollout boundary: "
                f"expected action_dim={int(expected_action_dim)}, got {len(action_list)} "
                f"(step={step_idx}, clip={clip_name})."
            )
        actions.append(action_list)
        next_frame = world_model.predict_next_frame(current, action)
        frames.append(next_frame)
        current = next_frame
        # Placeholder for keyframe re-anchoring bookkeeping in claim mode.
        if reanchor_every and reanchor_every > 0 and (step_idx + 1) % reanchor_every == 0:
            current = frames[-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{clip_name}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    policy_dim = len(actions[0]) if actions else None
    world_dim = getattr(world_model, "expected_action_dim", None)
    if world_dim is None:
        world_dim = getattr(world_model, "_expected_action_dim", None)
    dataset_dim = policy_dim
    compliant = (
        policy_dim is not None
        and world_dim is not None
        and dataset_dim is not None
        and policy_dim == world_dim == dataset_dim
    )
    return SimpleNamespace(
        video_path=video_path,
        action_sequence=actions,
        num_steps=len(actions),
        action_contract={
            "policy_dim": policy_dim,
            "world_dim": world_dim,
            "dataset_dim": dataset_dim,
            "compliant": bool(compliant),
            "reason": "" if compliant else "policy/world/dataset dims differ or missing",
        },
    )
