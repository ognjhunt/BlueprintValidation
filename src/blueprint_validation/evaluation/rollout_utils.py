"""Shared rollout execution utilities for adapter-based policy rollouts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from .rollout_state_proxy import DeterministicRolloutStateProxy
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
    current = initial_frame
    initial_state = _capture_task_state(
        world_model=world_model,
        frame=current,
        action=None,
        step_idx=0,
        phase="initial",
        task_prompt=task_prompt,
        fallback_proxy=state_proxy,
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
        captured = _capture_task_state(
            world_model=world_model,
            frame=next_frame,
            action=action_list,
            step_idx=step_idx + 1,
            phase="post_step",
            task_prompt=task_prompt,
            fallback_proxy=state_proxy,
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
    )


def _capture_task_state(
    *,
    world_model,
    frame,
    action,
    step_idx: int,
    phase: str,
    task_prompt: str,
    fallback_proxy: DeterministicRolloutStateProxy | None = None,
) -> dict | None:
    capture = getattr(world_model, "capture_rollout_state", None)
    if callable(capture):
        try:
            payload = capture(
                frame=frame,
                action=action,
                step_idx=step_idx,
                phase=phase,
                task_prompt=task_prompt,
            )
            if isinstance(payload, dict):
                return payload
        except TypeError:
            try:
                payload = capture(frame=frame, action=action, step_idx=step_idx)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        except Exception:
            pass
    extract = getattr(world_model, "extract_task_state", None)
    if callable(extract):
        try:
            payload = extract(
                frame=frame,
                action=action,
                step_idx=step_idx,
                phase=phase,
                task_prompt=task_prompt,
            )
            if isinstance(payload, dict):
                return payload
        except TypeError:
            try:
                payload = extract(frame=frame, action=action, step_idx=step_idx)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        except Exception:
            pass
    if fallback_proxy is not None:
        try:
            return fallback_proxy.capture(action=action, step_idx=step_idx, phase=phase)
        except Exception:
            return None
    return None
