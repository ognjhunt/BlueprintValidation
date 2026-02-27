"""VLM judge scoring using Gemini 3 Flash with Agentic Vision."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..common import get_logger
from ..config import VLMJudgeConfig

logger = get_logger("evaluation.vlm_judge")


@dataclass
class JudgeScore:
    task_score: float
    visual_score: float
    spatial_score: float
    reasoning: str
    raw_response: str


def _extract_json_from_response(text: str) -> dict:
    """Extract JSON from a VLM response that may contain markdown or extra text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    json_match = re.search(r"\{[^{}]*\"task_score\"[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from VLM response: %s", text[:200])
    return {"task_score": 0, "visual_score": 0, "spatial_score": 0, "reasoning": text}


def _get_gemini_client(config: VLMJudgeConfig):
    """Initialize Gemini client."""
    from google import genai

    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        raise ValueError(f"API key not found in env var: {config.api_key_env}")

    client = genai.Client(api_key=api_key)
    return client


def _encode_video_frames(video_path: Path, max_frames: int = 16) -> List[dict]:
    """Extract and encode frames from a video for VLM input."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)

    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode("utf-8")
        frames.append({
            "mime_type": "image/jpeg",
            "data": b64,
        })
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


def score_rollout(
    video_path: Path,
    task_prompt: str,
    config: VLMJudgeConfig,
    facility_description: str = "",
) -> JudgeScore:
    """Score a rollout video using Gemini 3 Flash with Agentic Vision.

    Uses the Think-Act-Observe loop to actively inspect video frames.
    """
    from google import genai
    from google.genai import types

    client = _get_gemini_client(config)

    # Build the scoring prompt
    prompt = config.scoring_prompt.replace("{task}", task_prompt)
    if facility_description:
        prompt += f"\n\nFacility context: {facility_description}"

    # Encode video frames for multimodal input
    frames = _encode_video_frames(video_path)

    # Build content parts
    parts = []
    parts.append(types.Part.from_text(
        f"I'm providing {len(frames)} frames from a robot policy rollout video. "
        f"The robot was given the task: \"{task_prompt}\"\n\n{prompt}"
    ))

    for i, frame_data in enumerate(frames):
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(frame_data["data"]),
            mime_type="image/jpeg",
        ))

    # Configure for Agentic Vision with code execution
    tools = []
    if config.enable_agentic_vision:
        tools = [types.Tool(code_execution=types.ToolCodeExecution())]

    response = client.models.generate_content(
        model=config.model,
        contents=[types.Content(parts=parts, role="user")],
        config=types.GenerateContentConfig(
            tools=tools,
            temperature=0.1,
        ),
    )

    raw_text = response.text or ""
    parsed = _extract_json_from_response(raw_text)

    return JudgeScore(
        task_score=float(parsed.get("task_score", 0)),
        visual_score=float(parsed.get("visual_score", 0)),
        spatial_score=float(parsed.get("spatial_score", 0)),
        reasoning=parsed.get("reasoning", ""),
        raw_response=raw_text,
    )


def score_spatial_accuracy(
    video_path: Path,
    facility_description: str,
    landmarks: List[str],
    config: VLMJudgeConfig,
) -> JudgeScore:
    """Score spatial accuracy of generated video against facility description."""
    from google import genai
    from google.genai import types

    client = _get_gemini_client(config)

    landmarks_str = "\n".join(f"- {lm}" for lm in landmarks)
    prompt = (
        f"Analyze these video frames from a generated environment.\n"
        f"The target facility is: {facility_description}\n"
        f"Expected landmarks:\n{landmarks_str}\n\n"
        f"Score on a 1-10 scale:\n"
        f"1. spatial_score: How well does the layout match the description?\n"
        f"2. visual_score: How photorealistic is the environment?\n"
        f"3. task_score: How many of the expected landmarks are visible?\n"
        f'Return JSON: {{"task_score": N, "visual_score": N, "spatial_score": N, "reasoning": "..."}}'
    )

    frames = _encode_video_frames(video_path, max_frames=8)
    parts = [types.Part.from_text(prompt)]
    for frame_data in frames:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(frame_data["data"]),
            mime_type="image/jpeg",
        ))

    tools = []
    if config.enable_agentic_vision:
        tools = [types.Tool(code_execution=types.ToolCodeExecution())]

    response = client.models.generate_content(
        model=config.model,
        contents=[types.Content(parts=parts, role="user")],
        config=types.GenerateContentConfig(tools=tools, temperature=0.1),
    )

    raw_text = response.text or ""
    parsed = _extract_json_from_response(raw_text)

    return JudgeScore(
        task_score=float(parsed.get("task_score", 0)),
        visual_score=float(parsed.get("visual_score", 0)),
        spatial_score=float(parsed.get("spatial_score", 0)),
        reasoning=parsed.get("reasoning", ""),
        raw_response=raw_text,
    )


def classify_facility(
    video_path: Path,
    facility_descriptions: dict[str, str],
    config: VLMJudgeConfig,
) -> dict:
    """Classify which facility a generated video depicts (for cross-site test)."""
    from google import genai
    from google.genai import types

    client = _get_gemini_client(config)

    desc_str = "\n".join(
        f"- Facility {fid}: {desc}" for fid, desc in facility_descriptions.items()
    )
    prompt = (
        f"Look at these video frames from a generated environment.\n"
        f"Which facility does this most closely match?\n\n{desc_str}\n\n"
        f"Return JSON: {{\"predicted_facility\": \"<facility_id>\", \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    )

    frames = _encode_video_frames(video_path, max_frames=8)
    parts = [types.Part.from_text(prompt)]
    for frame_data in frames:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(frame_data["data"]),
            mime_type="image/jpeg",
        ))

    response = client.models.generate_content(
        model=config.model,
        contents=[types.Content(parts=parts, role="user")],
        config=types.GenerateContentConfig(temperature=0.1),
    )

    raw_text = response.text or ""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = _extract_json_from_response(raw_text)

    return {
        "predicted_facility": parsed.get("predicted_facility", "unknown"),
        "confidence": float(parsed.get("confidence", 0)),
        "reasoning": parsed.get("reasoning", ""),
        "raw_response": raw_text,
    }
