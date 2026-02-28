"""Shared rollout execution utilities for adapter-based policy rollouts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

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
    for _ in range(max_steps):
        action = policy_adapter.predict_action(
            handle=policy_handle,
            frame=current,
            task_prompt=task_prompt,
            unnorm_key=unnorm_key,
            device=device,
        )
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        actions.append(action_list)
        next_frame = world_model.predict_next_frame(current, action)
        frames.append(next_frame)
        current = next_frame

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{clip_name}.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return SimpleNamespace(
        video_path=video_path, action_sequence=actions, num_steps=len(actions)
    )
