"""Shared rollout execution utilities for adapter-based policy rollouts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from .rollout_state_proxy import DeterministicRolloutStateProxy
from .task_state_capture import capture_task_state, world_model_supports_native_task_state
from ..video_io import ensure_h264_video, open_mp4_writer


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
    rollout_context: Optional[Dict[str, object]] = None,
    task_spec: Optional[Dict[str, object]] = None,
    require_native_task_state: bool = False,
) -> SimpleNamespace:
    """Run a policy rollout in a world model using a generic policy adapter.

    Returns a SimpleNamespace with ``video_path``, ``action_sequence``, and
    ``num_steps`` — matching the contract of ``run_rollout()`` from
    ``openvla_runner.py``.
    """
    import cv2

    frames = [initial_frame.copy()]
    actions: List[list] = []
    state_trace: List[dict] = []
    state_proxy = DeterministicRolloutStateProxy.from_context(
        task_prompt=task_prompt,
        task_spec=task_spec,
        rollout_context=rollout_context,
    )
    native_task_state_supported = world_model_supports_native_task_state(world_model)
    if require_native_task_state and not native_task_state_supported:
        raise RuntimeError(
            "Claim mode requires native world-model task state, but the loaded world model "
            "does not expose capture_rollout_state or extract_task_state."
        )
    current = initial_frame
    initial_state = capture_task_state(
        world_model=world_model,
        frame=current,
        action=None,
        step_idx=0,
        phase="initial",
        task_prompt=task_prompt,
        fallback_proxy=state_proxy,
        require_native_task_state=require_native_task_state,
    )
    if initial_state is not None:
        state_trace.append(initial_state)
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
        captured = capture_task_state(
            world_model=world_model,
            frame=next_frame,
            action=action_list,
            step_idx=step_idx + 1,
            phase="post_step",
            task_prompt=task_prompt,
            fallback_proxy=state_proxy,
            require_native_task_state=require_native_task_state,
        )
        if captured is not None:
            state_trace.append(captured)
        current = next_frame
        # Placeholder for keyframe re-anchoring bookkeeping in claim mode.
        if reanchor_every and reanchor_every > 0 and (step_idx + 1) % reanchor_every == 0:
            current = frames[-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{clip_name}.mp4"
    h, w = frames[0].shape[:2]
    writer = open_mp4_writer(
        output_path=video_path,
        fps=10.0,
        frame_size=(w, h),
        is_color=True,
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    checked_video = ensure_h264_video(
        input_path=video_path,
        min_decoded_frames=len(frames),
        replace_source=True,
    )
    video_path = checked_video.path
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
        state_trace=state_trace,
        action_contract={
            "policy_dim": policy_dim,
            "world_dim": world_dim,
            "dataset_dim": dataset_dim,
            "compliant": bool(compliant),
            "reason": "" if compliant else "policy/world/dataset dims differ or missing",
        },
        native_task_state_required=bool(require_native_task_state),
        native_task_state_supported=bool(native_task_state_supported),
    )
