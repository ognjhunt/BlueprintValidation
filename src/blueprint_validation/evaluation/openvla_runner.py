"""OpenVLA-OFT inference loop for policy evaluation inside DreamDojo world model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..common import get_logger

logger = get_logger("evaluation.openvla_runner")


@dataclass
class RolloutResult:
    task_prompt: str
    condition: str  # "baseline" or "adapted"
    video_path: Optional[Path]
    action_sequence: List[List[float]]
    num_steps: int
    success: bool


def load_openvla(model_name: str, checkpoint_path: Optional[Path] = None, device: str = "cuda"):
    """Load OpenVLA-OFT model for inference."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import torch

    model_id = str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else model_name

    logger.info("Loading OpenVLA-OFT from %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    if not hasattr(model, "predict_action"):
        raise RuntimeError(
            f"Loaded OpenVLA-OFT model from {model_id} does not expose predict_action(). "
            "Ensure you are using an OpenVLA-OFT-compatible checkpoint."
        )

    return model, processor


def load_dreamdojo_world_model(
    checkpoint_path: Path,
    adapted_checkpoint: Optional[Path] = None,
    dreamdojo_repo: Optional[Path] = None,
    device: str = "cuda",
):
    """Load DreamDojo world model for action-conditioned video prediction.

    DreamDojo is built on cosmos_predict2 and uses the action-conditioned
    inference API. If adapted_checkpoint is provided, loads the fine-tuned
    model instead of the baseline.

    Returns an object with a predict_next_frame(frame, action) method.
    """
    effective_checkpoint = adapted_checkpoint if adapted_checkpoint else checkpoint_path

    if not effective_checkpoint.exists():
        raise RuntimeError(f"DreamDojo checkpoint path does not exist: {effective_checkpoint}")

    logger.info("Loading DreamDojo from %s", effective_checkpoint)
    if adapted_checkpoint:
        logger.info("Using adapted (fine-tuned) checkpoint")
    else:
        logger.info("Using baseline (pretrained) checkpoint")

    import sys

    def _load_action_conditioned_api():
        # DreamDojo has shipped both:
        # 1) cosmos_predict2.action_conditioned.inference (package style)
        # 2) cosmos_predict2.action_conditioned (module style)
        try:
            from cosmos_predict2.action_conditioned import inference as ac_inference
            from cosmos_predict2.action_conditioned.inference import (
                ActionConditionedInferenceArguments,
            )

            return ac_inference, ActionConditionedInferenceArguments
        except Exception:
            from cosmos_predict2 import action_conditioned as ac_inference
            from cosmos_predict2.action_conditioned import ActionConditionedInferenceArguments

            return ac_inference, ActionConditionedInferenceArguments

    # Prefer already-installed cosmos_predict2 first, then explicit repo fallback.
    try:
        ac_inference, ActionConditionedInferenceArguments = _load_action_conditioned_api()
    except Exception as e:
        if dreamdojo_repo is None:
            raise RuntimeError(
                "DreamDojo/cosmos_predict2 is not importable. Install DreamDojo in the current "
                "environment or pass finetune.dreamdojo_repo to enable repo-path fallback."
            ) from e
        if not dreamdojo_repo.exists():
            raise RuntimeError(
                "DreamDojo/cosmos_predict2 is not importable and finetune.dreamdojo_repo "
                f"does not exist: {dreamdojo_repo}"
            ) from e
        dreamdojo_repo_str = str(dreamdojo_repo)
        if dreamdojo_repo_str not in sys.path:
            sys.path.insert(0, dreamdojo_repo_str)
        try:
            ac_inference, ActionConditionedInferenceArguments = _load_action_conditioned_api()
        except Exception as inner:
            raise RuntimeError(
                "DreamDojo/cosmos_predict2 is not importable after repo-path fallback. "
                f"Attempted finetune.dreamdojo_repo={dreamdojo_repo}"
            ) from inner

    args = ActionConditionedInferenceArguments(
        checkpoint_dir=str(effective_checkpoint),
    )
    model = ac_inference.setup(args)
    if hasattr(model, "to"):
        model = model.to(device)
    if not hasattr(model, "predict_next_frame"):
        raise RuntimeError(
            "DreamDojo model does not expose predict_next_frame(). "
            "Update adapter to match the installed DreamDojo API."
        )
    return model


def run_rollout(
    world_model,
    openvla_model,
    openvla_processor,
    initial_frame: np.ndarray,
    task_prompt: str,
    max_steps: int = 100,
    unnorm_key: Optional[str] = "bridge_orig",
    output_dir: Optional[Path] = None,
    clip_name: str = "rollout",
    device: str = "cuda",
) -> RolloutResult:
    """Run a single policy rollout: OpenVLA-OFT predicts actions, DreamDojo generates frames."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - exercised in lightweight test envs
        Image = None

    try:
        import torch

        torch_dtype = torch.bfloat16
    except ImportError:  # pragma: no cover - exercised in lightweight test envs
        torch = None
        torch_dtype = None

    frames = [initial_frame.copy()]
    actions = []
    current_frame = initial_frame

    logger.info("Running rollout for task: %s (max %d steps)", task_prompt, max_steps)

    for step in range(max_steps):
        # OpenVLA-OFT predicts action from current observation
        image = Image.fromarray(current_frame) if Image else current_frame
        prompt = f"In: What action should the robot take to {task_prompt}?\nOut:"

        inputs = openvla_processor(prompt, image, return_tensors="pt")
        if hasattr(inputs, "to"):
            if torch_dtype is not None:
                inputs = inputs.to(device, dtype=torch_dtype)
            else:
                inputs = inputs.to(device)
        else:
            inputs = {
                key: (
                    value.to(device, dtype=torch_dtype)
                    if hasattr(value, "to") and torch_dtype is not None
                    else value.to(device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }

        predict_kwargs = dict(inputs)
        predict_kwargs["do_sample"] = False
        if unnorm_key:
            predict_kwargs["unnorm_key"] = unnorm_key

        try:
            action = openvla_model.predict_action(**predict_kwargs)
        except TypeError:
            predict_kwargs.pop("unnorm_key", None)
            action = openvla_model.predict_action(**predict_kwargs)

        actions.append(action.tolist() if hasattr(action, "tolist") else list(action))

        # DreamDojo generates next frame from action
        if not hasattr(world_model, "predict_next_frame"):
            raise RuntimeError("World model missing predict_next_frame()")
        next_frame = world_model.predict_next_frame(current_frame, action)

        frames.append(next_frame)
        current_frame = next_frame

    # Save rollout video
    video_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{clip_name}.mp4"

        import cv2

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    return RolloutResult(
        task_prompt=task_prompt,
        condition="",
        video_path=video_path,
        action_sequence=actions,
        num_steps=len(actions),
        success=True,
    )
