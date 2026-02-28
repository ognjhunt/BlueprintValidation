"""VLM judge scoring using Gemini 3 Flash with Agentic Vision."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

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


@dataclass
class ManipulationJudgeScore(JudgeScore):
    grasp_acquired: bool
    lifted_clear: bool
    placed_in_target: bool
    stable_after_place: bool


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


def _extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
        if chunks:
            return "\n".join(chunks)
    return ""


def _build_code_execution_tool(types):
    # SDKs differ on whether ToolCodeExecution should be passed as a class or instance.
    try:
        return types.Tool(code_execution=types.ToolCodeExecution())
    except TypeError:
        return types.Tool(code_execution=types.ToolCodeExecution)


def _build_generate_config(types, enable_agentic_vision: bool, temperature: float):
    kwargs = {"temperature": temperature}
    if enable_agentic_vision:
        kwargs["tools"] = [_build_code_execution_tool(types)]
    return types.GenerateContentConfig(**kwargs)


def _validate_numeric_score(payload: dict, key: str) -> float:
    if key not in payload:
        raise ValueError(f"Missing required key: {key}")
    value = float(payload[key])
    if value < 0 or value > 10:
        raise ValueError(f"{key} out of range [0, 10]: {value}")
    return value


def _parse_judge_payload(payload: dict) -> tuple[float, float, float, str]:
    task = _validate_numeric_score(payload, "task_score")
    visual = _validate_numeric_score(payload, "visual_score")
    spatial = _validate_numeric_score(payload, "spatial_score")
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("Missing reasoning in judge payload")
    return task, visual, spatial, reasoning


def _parse_bool(payload: dict, key: str) -> bool:
    if key not in payload:
        raise ValueError(f"Missing required key: {key}")
    value = payload[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "yes", "y", "1"}:
            return True
        if val in {"false", "no", "n", "0"}:
            return False
    raise ValueError(f"{key} must be boolean-like, got {value!r}")


def _parse_manipulation_payload(payload: dict) -> tuple[float, float, float, bool, bool, bool, bool, str]:
    task, visual, spatial, reasoning = _parse_judge_payload(payload)
    grasp = _parse_bool(payload, "grasp_acquired")
    lifted = _parse_bool(payload, "lifted_clear")
    placed = _parse_bool(payload, "placed_in_target")
    stable = _parse_bool(payload, "stable_after_place")
    return task, visual, spatial, grasp, lifted, placed, stable, reasoning


def _parse_classify_payload(payload: dict) -> tuple[str, float, str]:
    predicted = str(payload.get("predicted_facility", "")).strip()
    confidence = float(payload.get("confidence", 0))
    reasoning = str(payload.get("reasoning", "")).strip()
    if not predicted:
        raise ValueError("Missing predicted_facility in classify payload")
    if confidence < 0 or confidence > 1:
        raise ValueError(f"confidence out of range [0, 1]: {confidence}")
    if not reasoning:
        raise ValueError("Missing reasoning in classify payload")
    return predicted, confidence, reasoning


def _get_gemini_client(config: VLMJudgeConfig):
    """Initialize Gemini client."""
    from google import genai

    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        raise ValueError(f"API key not found in env var: {config.api_key_env}")

    client = genai.Client(api_key=api_key)
    return client


def _generate_with_retry(client, *, model, contents, config, max_retries: int = 3):
    """Call generate_content with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config,
            )
        except Exception as exc:
            exc_text = str(exc).lower()
            transient = any(
                kw in exc_text
                for kw in ("rate limit", "429", "500", "503", "timeout", "unavailable", "deadline")
            )
            if not transient or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Transient API error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, max_retries, wait, exc,
            )
            time.sleep(wait)


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
    errors = []
    for attempt in range(3):
        if attempt > 0:
            parts.append(
                types.Part.from_text(
                    "Retry: return JSON only with numeric task_score/visual_score/spatial_score "
                    "between 0 and 10 and a non-empty reasoning field."
                )
            )

        response = _generate_with_retry(
            client,
            model=config.model,
            contents=[types.Content(parts=parts, role="user")],
            config=_build_generate_config(
                types,
                enable_agentic_vision=config.enable_agentic_vision,
                temperature=0.1,
            ),
        )
        raw_text = _extract_response_text(response)
        parsed = _extract_json_from_response(raw_text)
        try:
            task_score, visual_score, spatial_score, reasoning = _parse_judge_payload(parsed)
            return JudgeScore(
                task_score=task_score,
                visual_score=visual_score,
                spatial_score=spatial_score,
                reasoning=reasoning,
                raw_response=raw_text,
            )
        except Exception as e:
            errors.append(f"attempt={attempt + 1}: {e}")
            continue

    raise RuntimeError(
        "VLM judge failed to return valid scoring JSON after retries: "
        + "; ".join(errors)
    )


def score_rollout_manipulation(
    video_path: Path,
    task_prompt: str,
    config: VLMJudgeConfig,
    facility_description: str = "",
) -> ManipulationJudgeScore:
    """Score manipulation rollout with explicit grasp/lift/place/stability checks."""
    from google.genai import types

    client = _get_gemini_client(config)
    prompt = (
        "You are evaluating a robot manipulation rollout.\n"
        f"Task: {task_prompt}\n"
        "Score and return JSON with:\n"
        '- "task_score" (0-10)\n'
        '- "visual_score" (0-10)\n'
        '- "spatial_score" (0-10)\n'
        '- "grasp_acquired" (bool)\n'
        '- "lifted_clear" (bool)\n'
        '- "placed_in_target" (bool)\n'
        '- "stable_after_place" (bool)\n'
        '- "reasoning" (string)\n'
    )
    if facility_description:
        prompt += f"\nFacility context: {facility_description}\n"

    frames = _encode_video_frames(video_path)
    parts = [
        types.Part.from_text(
            f"Manipulation rollout frames ({len(frames)}). {prompt}"
        )
    ]
    for frame_data in frames:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(frame_data["data"]),
            mime_type="image/jpeg",
        ))

    errors = []
    for attempt in range(3):
        if attempt > 0:
            parts.append(
                types.Part.from_text(
                    "Retry: return strict JSON only, include all required boolean manipulation fields."
                )
            )

        response = _generate_with_retry(
            client,
            model=config.model,
            contents=[types.Content(parts=parts, role="user")],
            config=_build_generate_config(
                types,
                enable_agentic_vision=config.enable_agentic_vision,
                temperature=0.1,
            ),
        )
        raw_text = _extract_response_text(response)
        parsed = _extract_json_from_response(raw_text)
        try:
            (
                task_score,
                visual_score,
                spatial_score,
                grasp,
                lifted,
                placed,
                stable,
                reasoning,
            ) = _parse_manipulation_payload(parsed)
            return ManipulationJudgeScore(
                task_score=task_score,
                visual_score=visual_score,
                spatial_score=spatial_score,
                grasp_acquired=grasp,
                lifted_clear=lifted,
                placed_in_target=placed,
                stable_after_place=stable,
                reasoning=reasoning,
                raw_response=raw_text,
            )
        except Exception as e:
            errors.append(f"attempt={attempt + 1}: {e}")
            continue

    raise RuntimeError(
        "VLM manipulation judge failed to return valid JSON after retries: "
        + "; ".join(errors)
    )


def score_spatial_accuracy(
    video_path: Path,
    facility_description: str,
    landmarks: List[str],
    config: VLMJudgeConfig,
) -> JudgeScore:
    """Score spatial accuracy of generated video against facility description."""
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

    errors = []
    for attempt in range(3):
        if attempt > 0:
            parts.append(
                types.Part.from_text(
                    "Retry: return JSON only with task_score/visual_score/spatial_score in [0,10] "
                    "and a non-empty reasoning string."
                )
            )
        response = _generate_with_retry(
            client,
            model=config.model,
            contents=[types.Content(parts=parts, role="user")],
            config=_build_generate_config(
                types,
                enable_agentic_vision=config.enable_agentic_vision,
                temperature=0.1,
            ),
        )
        raw_text = _extract_response_text(response)
        parsed = _extract_json_from_response(raw_text)
        try:
            task_score, visual_score, spatial_score, reasoning = _parse_judge_payload(parsed)
            return JudgeScore(
                task_score=task_score,
                visual_score=visual_score,
                spatial_score=spatial_score,
                reasoning=reasoning,
                raw_response=raw_text,
            )
        except Exception as e:
            errors.append(f"attempt={attempt + 1}: {e}")
            continue

    raise RuntimeError(
        "VLM spatial scorer failed to return valid JSON after retries: "
        + "; ".join(errors)
    )


def classify_facility(
    video_path: Path,
    facility_descriptions: dict[str, str],
    config: VLMJudgeConfig,
) -> dict:
    """Classify which facility a generated video depicts (for cross-site test)."""
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

    errors = []
    for attempt in range(3):
        if attempt > 0:
            parts.append(
                types.Part.from_text(
                    "Retry: return JSON only with predicted_facility, confidence (0-1), reasoning."
                )
            )
        response = _generate_with_retry(
            client,
            model=config.model,
            contents=[types.Content(parts=parts, role="user")],
            config=_build_generate_config(
                types,
                enable_agentic_vision=config.enable_agentic_vision,
                temperature=0.1,
            ),
        )

        raw_text = _extract_response_text(response)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = _extract_json_from_response(raw_text)

        try:
            predicted, confidence, reasoning = _parse_classify_payload(parsed)
            return {
                "predicted_facility": predicted,
                "confidence": confidence,
                "reasoning": reasoning,
                "raw_response": raw_text,
            }
        except Exception as e:
            errors.append(f"attempt={attempt + 1}: {e}")
            continue

    raise RuntimeError(
        "VLM facility classifier failed to return valid JSON after retries: "
        + "; ".join(errors)
    )
