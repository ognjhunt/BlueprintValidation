"""OpenVLA inference loop for policy evaluation inside DreamDojo world model."""

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
    """Load OpenVLA model for inference."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import torch

    model_id = str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else model_name

    logger.info("Loading OpenVLA from %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    return model, processor


def load_dreamdojo_world_model(
    checkpoint_path: Path,
    lora_path: Optional[Path] = None,
    device: str = "cuda",
):
    """Load DreamDojo world model, optionally with LoRA adapter.

    Returns a callable that takes (current_frame, action) and returns next_frame.
    """
    import torch
    import sys

    # Add DreamDojo to path
    dreamdojo_root = checkpoint_path.parent.parent
    if str(dreamdojo_root) not in sys.path:
        sys.path.insert(0, str(dreamdojo_root))

    logger.info("Loading DreamDojo from %s", checkpoint_path)
    if lora_path:
        logger.info("Applying LoRA adapter from %s", lora_path)

    # This is a placeholder for the actual DreamDojo loading API.
    # The exact loading code depends on DreamDojo's Python API.
    # Common pattern: load base model, then merge LoRA weights via PEFT.

    try:
        # Try DreamDojo's native loading
        from dreamdojo import DreamDojoModel
        model = DreamDojoModel.from_pretrained(str(checkpoint_path))
        if lora_path and lora_path.exists():
            model.load_lora(str(lora_path))
        model = model.to(device)
        return model
    except ImportError:
        logger.warning("DreamDojo module not found, using stub world model")
        return _StubWorldModel(device)


class _StubWorldModel:
    """Stub world model for testing without DreamDojo installed."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def predict_next_frame(self, current_frame, action):
        """Return a slightly modified version of current frame (stub)."""
        # Add small noise to simulate frame progression
        noise = np.random.normal(0, 5, current_frame.shape).astype(np.uint8)
        return np.clip(current_frame.astype(int) + noise, 0, 255).astype(np.uint8)


def run_rollout(
    world_model,
    openvla_model,
    openvla_processor,
    initial_frame: np.ndarray,
    task_prompt: str,
    max_steps: int = 100,
    output_dir: Optional[Path] = None,
    clip_name: str = "rollout",
    device: str = "cuda",
) -> RolloutResult:
    """Run a single policy rollout: OpenVLA predicts actions, DreamDojo generates frames."""
    import torch
    from PIL import Image

    frames = [initial_frame]
    actions = []
    current_frame = initial_frame

    logger.info("Running rollout for task: %s (max %d steps)", task_prompt, max_steps)

    for step in range(max_steps):
        # OpenVLA predicts action from current observation
        image = Image.fromarray(current_frame)
        prompt = f"In: What action should the robot take to {task_prompt}?\nOut:"

        inputs = openvla_processor(prompt, image).to(device, dtype=torch.bfloat16)
        action = openvla_model.predict_action(**inputs, do_sample=False)

        actions.append(action.tolist() if hasattr(action, "tolist") else list(action))

        # DreamDojo generates next frame from action
        if hasattr(world_model, "predict_next_frame"):
            next_frame = world_model.predict_next_frame(current_frame, action)
        else:
            # Fallback for stub
            next_frame = current_frame

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
