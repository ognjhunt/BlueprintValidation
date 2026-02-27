"""Optional Gemini image-edit polishing for composited clips."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List

import numpy as np

from ..common import get_logger

logger = get_logger("synthetic.gemini_image_polish")


def _load_video_frames(video_path: Path) -> tuple[List, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _save_video_frames(frames, output_path: Path, fps: float) -> None:
    import cv2

    if not frames:
        raise RuntimeError("No frames to save")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def polish_clip_with_gemini(
    input_video: Path,
    output_video: Path,
    model: str,
    api_key_env: str,
    prompt: str,
    sample_every_n_frames: int = 1,
) -> dict:
    """Polish composited video with Gemini image editing.

    Applies edits to sampled frames and keeps unsampled frames unchanged.
    """
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing Gemini API key environment variable: {api_key_env}")

    from google import genai

    client = genai.Client(api_key=api_key)
    frames, fps = _load_video_frames(input_video)
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {input_video}")

    edited = 0
    output_frames = []
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required for Gemini image polishing. Install dependency 'Pillow'."
        ) from e

    for idx, frame in enumerate(frames):
        if idx % max(sample_every_n_frames, 1) != 0:
            output_frames.append(frame)
            continue
        pil_img = Image.fromarray(frame)
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, pil_img],
        )
        generated = None
        for candidate in getattr(resp, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    generated = Image.open(io.BytesIO(inline.data)).convert("RGB")
                    break
            if generated is not None:
                break
        if generated is None:
            output_frames.append(frame)
            continue
        output_frames.append(np.array(generated))
        edited += 1

    _save_video_frames(output_frames, output_video, fps=fps)
    return {
        "num_frames": len(frames),
        "num_frames_edited": edited,
        "model": model,
        "prompt": prompt,
    }
