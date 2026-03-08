"""VLM judge scoring using Gemini 3 Flash with Agentic Vision."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from ..common import get_logger
from ..config import VLMJudgeConfig

logger = get_logger("evaluation.vlm_judge")

_USAGE_LOCK = threading.Lock()
_USAGE_CALL_COUNT = 0
_USAGE_TOTAL_TOKENS = 0
_VIDEO_TOOL_DISABLE_WARNED = False


@dataclass
class JudgeScore:
    task_score: float
    visual_score: float
    spatial_score: float
    reasoning: str
    raw_response: str
    model_used: str = field(default="", kw_only=True)


@dataclass
class ManipulationJudgeScore(JudgeScore):
    grasp_acquired: bool
    lifted_clear: bool
    placed_in_target: bool
    stable_after_place: bool


@dataclass
class Stage1ProbeScore(JudgeScore):
    issue_tags: List[str]


@dataclass
class Stage2QualityScore(JudgeScore):
    issue_tags: List[str]


_STAGE1_ALLOWED_ISSUE_TAGS = {
    "target_missing",
    "target_off_center",
    "target_occluded",
    "camera_too_far",
    "camera_too_close",
    "camera_motion_too_fast",
    "blur_or_soft_focus",
    "unstable_view",
}

_STAGE2_ALLOWED_ISSUE_TAGS = _STAGE1_ALLOWED_ISSUE_TAGS | {
    "off_task",
    "semantic_mismatch",
    "green_cast",
    "low_motion",
    "temporal_flicker",
    "depth_artifacts",
    "compression_artifacts",
    "hallucinated_objects",
    "style_drift",
}


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

    snippet = str(text[:200]).replace("\n", " ")
    logger.warning("Could not parse JSON from VLM response: %s", snippet)
    raise ValueError("Could not parse JSON from VLM response")


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


def _extract_usage_int(usage_obj, *keys: str) -> int | None:
    for key in keys:
        if hasattr(usage_obj, key):
            try:
                return int(getattr(usage_obj, key))
            except Exception:
                continue
        if isinstance(usage_obj, dict) and key in usage_obj:
            try:
                return int(usage_obj[key])
            except Exception:
                continue
    return None


def _log_usage_metadata(model_name: str, response) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = getattr(response, "usageMetadata", None)
    if usage is None:
        return

    prompt_tokens = _extract_usage_int(usage, "prompt_token_count", "promptTokenCount")
    candidate_tokens = _extract_usage_int(
        usage, "candidates_token_count", "candidatesTokenCount"
    )
    total_tokens = _extract_usage_int(usage, "total_token_count", "totalTokenCount")
    if total_tokens is None:
        if prompt_tokens is None and candidate_tokens is None:
            return
        total_tokens = int((prompt_tokens or 0) + (candidate_tokens or 0))

    with _USAGE_LOCK:
        global _USAGE_CALL_COUNT, _USAGE_TOTAL_TOKENS
        _USAGE_CALL_COUNT += 1
        _USAGE_TOTAL_TOKENS += int(total_tokens)
        avg_tokens = _USAGE_TOTAL_TOKENS / max(1, _USAGE_CALL_COUNT)

    logger.info(
        "VLM token usage model=%s prompt=%s candidates=%s total=%s avg_total_per_call=%.1f calls=%d",
        model_name,
        prompt_tokens if prompt_tokens is not None else "n/a",
        candidate_tokens if candidate_tokens is not None else "n/a",
        total_tokens,
        avg_tokens,
        _USAGE_CALL_COUNT,
    )


def _build_code_execution_tool(types):
    # SDKs differ on whether ToolCodeExecution should be passed as a class or instance.
    try:
        return types.Tool(code_execution=types.ToolCodeExecution())
    except TypeError:
        return types.Tool(code_execution=types.ToolCodeExecution)


def _build_generate_config(
    types,
    *,
    enable_agentic_vision: bool,
    temperature: float,
    includes_video: bool,
):
    global _VIDEO_TOOL_DISABLE_WARNED
    kwargs = {"temperature": temperature}
    if enable_agentic_vision and not includes_video:
        kwargs["tools"] = [_build_code_execution_tool(types)]
    elif enable_agentic_vision and includes_video:
        if not _VIDEO_TOOL_DISABLE_WARNED:
            logger.warning(
                "Agentic code-execution tools auto-disabled for video Gemini request to avoid "
                "unsupported video/text/timestamp MIME path."
            )
            _VIDEO_TOOL_DISABLE_WARNED = True
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


def _parse_manipulation_payload(
    payload: dict,
) -> tuple[float, float, float, bool, bool, bool, bool, str]:
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


def _parse_issue_tags(payload: dict, *, allowed_tags: set[str]) -> List[str]:
    if "issue_tags" not in payload:
        raise ValueError("Missing required key: issue_tags")
    raw = payload.get("issue_tags")
    if not isinstance(raw, list):
        raise ValueError("issue_tags must be a JSON array")
    deduped: List[str] = []
    seen = set()
    for item in raw:
        tag = str(item).strip().lower()
        if not tag:
            continue
        if tag not in allowed_tags:
            raise ValueError(f"Unknown issue tag: {tag}")
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped


def _parse_stage1_probe_payload(payload: dict) -> tuple[float, float, float, List[str], str]:
    task, visual, spatial, reasoning = _parse_judge_payload(payload)
    issue_tags = _parse_issue_tags(payload, allowed_tags=_STAGE1_ALLOWED_ISSUE_TAGS)
    return task, visual, spatial, issue_tags, reasoning


def _parse_stage2_quality_payload(payload: dict) -> tuple[float, float, float, List[str], str]:
    task, visual, spatial, reasoning = _parse_judge_payload(payload)
    issue_tags = _parse_issue_tags(payload, allowed_tags=_STAGE2_ALLOWED_ISSUE_TAGS)
    return task, visual, spatial, issue_tags, reasoning


def _get_gemini_client(config: VLMJudgeConfig):
    """Initialize Gemini client."""
    from google import genai

    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        raise ValueError(f"API key not found in env var: {config.api_key_env}")

    client = genai.Client(api_key=api_key)
    return client


def _file_state_name(file_obj) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return ""
    name = getattr(state, "name", None)
    if name:
        return str(name).strip().upper()
    return str(state).strip().upper()


def _wait_for_uploaded_file_active(
    client,
    file_obj,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 2.0,
):
    """Poll Gemini Files API until uploaded media is ACTIVE (or fails)."""
    name = str(getattr(file_obj, "name", "")).strip()
    if not name:
        return file_obj

    deadline = time.time() + max(1.0, float(timeout_s))
    current = file_obj
    while True:
        state = _file_state_name(current)
        if "ACTIVE" in state or "READY" in state:
            return current
        if "FAILED" in state or "ERROR" in state:
            raise RuntimeError(f"Uploaded Gemini file failed processing: {name} state={state}")
        if state and "PROCESSING" not in state:
            # Unknown non-terminal state; treat as transient and keep polling until timeout.
            logger.info("Waiting for Gemini file state transition: %s state=%s", name, state)

        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Gemini file to become ACTIVE: {name} "
                f"(last_state={state or 'unknown'})"
            )

        time.sleep(max(0.1, float(poll_interval_s)))
        current = client.files.get(name=name)


def _is_video_file_path(video_path: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(str(video_path))
    return bool(mime_type and mime_type.startswith("video/"))


def _upload_video_file(client, video_path: Path):
    """Upload a local video file to Gemini Files API and wait until ACTIVE."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")
    if not _is_video_file_path(video_path):
        raise ValueError(f"Video path does not have a recognized video MIME type: {video_path}")
    uploaded = client.files.upload(file=str(video_path))
    return _wait_for_uploaded_file_active(client, uploaded)


def _delete_uploaded_file(client, file_obj) -> None:
    """Best-effort cleanup of uploaded Gemini file handles."""
    name = str(getattr(file_obj, "name", "")).strip()
    if not name:
        return
    try:
        client.files.delete(name=name)
    except Exception:
        logger.warning("Failed to delete uploaded Gemini file: %s", name, exc_info=True)


def _uploaded_file_uri(file_obj) -> str:
    uri = getattr(file_obj, "uri", None)
    if uri:
        return str(uri).strip()
    file_uri = getattr(file_obj, "file_uri", None)
    if file_uri:
        return str(file_uri).strip()
    return ""


def _uploaded_file_mime_type(file_obj) -> str:
    mime = getattr(file_obj, "mime_type", None)
    if mime:
        return str(mime).strip()
    mime = getattr(file_obj, "mimeType", None)
    if mime:
        return str(mime).strip()
    return "video/mp4"


def _build_uploaded_video_part(types, file_obj, *, video_metadata_fps: float):
    """Build a Gemini video part with optional explicit fps metadata."""
    fps = float(video_metadata_fps)
    if fps <= 0.0:
        return file_obj

    uri = _uploaded_file_uri(file_obj)
    if not uri:
        logger.warning(
            "Uploaded file URI unavailable; falling back to default video-part behavior."
        )
        return file_obj

    mime_type = _uploaded_file_mime_type(file_obj)
    return types.Part(
        file_data=types.FileData(file_uri=uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=fps),
    )


def _is_quota_exhausted_error(exc_text: str) -> bool:
    text = exc_text.lower()
    quota_markers = (
        "quota exceeded",
        "resource_exhausted",
        "free_tier_requests",
        "generativelanguage.googleapis.com/generate_content_free_tier_requests",
        "retry in",
    )
    return "429" in text and any(marker in text for marker in quota_markers)


def _generate_with_retry(
    client,
    *,
    model: str,
    fallback_models: Sequence[str],
    contents,
    config,
    max_retries: int = 3,
) -> tuple[object, str]:
    """Call generate_content with exponential backoff on transient errors."""
    model_candidates: List[str] = [str(model).strip()]
    for fallback in fallback_models:
        candidate = str(fallback).strip()
        if candidate and candidate not in model_candidates:
            model_candidates.append(candidate)

    last_exc: Exception | None = None
    for model_idx, candidate_model in enumerate(model_candidates):
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=config,
                )
                _log_usage_metadata(candidate_model, response)
                return response, candidate_model
            except Exception as exc:
                last_exc = exc
                exc_text = str(exc).lower()
                quota_exhausted = _is_quota_exhausted_error(exc_text)
                transient = any(
                    kw in exc_text
                    for kw in (
                        "rate limit",
                        "429",
                        "500",
                        "503",
                        "timeout",
                        "unavailable",
                        "deadline",
                    )
                )
                has_fallback = model_idx < len(model_candidates) - 1

                # If this model's quota is exhausted, immediately try the next fallback model.
                if quota_exhausted and has_fallback:
                    logger.warning(
                        "Quota exhausted for model %s; falling back to %s.",
                        candidate_model,
                        model_candidates[model_idx + 1],
                    )
                    break

                if not transient:
                    raise

                if attempt == max_retries - 1:
                    if has_fallback:
                        logger.warning(
                            "Model %s failed after %d retries; falling back to %s. Last error: %s",
                            candidate_model,
                            max_retries,
                            model_candidates[model_idx + 1],
                            exc,
                        )
                        break
                    raise

                wait = 2**attempt
                logger.warning(
                    "Transient API error (model=%s attempt %d/%d), retrying in %ds: %s",
                    candidate_model,
                    attempt + 1,
                    max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("VLM generation failed without returning or raising an exception.")


def score_rollout(
    video_path: Path,
    task_prompt: str,
    config: VLMJudgeConfig,
    facility_description: str = "",
    *,
    max_frames: int = 16,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> JudgeScore:
    """Score a rollout video using Gemini 3 Flash with Agentic Vision.

    Uses Gemini native video input (Files API) and returns strict JSON scores.
    """
    del max_frames, start_frame, end_frame  # Deprecated args; native video path always used.
    from google.genai import types

    client = _get_gemini_client(config)

    # Build the scoring prompt
    prompt = config.scoring_prompt.replace("{task}", task_prompt)
    if facility_description:
        prompt += f"\n\nFacility context: {facility_description}"

    # Configure for Agentic Vision with code execution
    errors = []
    uploaded_video = _upload_video_file(client, video_path)
    try:
        video_part = _build_uploaded_video_part(
            types,
            uploaded_video,
            video_metadata_fps=float(config.video_metadata_fps),
        )
        effective_fps = (
            float(config.video_metadata_fps) if float(config.video_metadata_fps) > 0.0 else None
        )
        logger.info(
            "Scoring rollout with native Gemini video input: path=%s metadata_fps=%s",
            video_path,
            "default" if effective_fps is None else f"{effective_fps:.3f}",
        )
        for attempt in range(3):
            contents = [
                (
                    "I am providing a robot policy rollout video file. "
                    f'The robot was given the task: "{task_prompt}"\n\n{prompt}'
                ),
                video_part,
            ]
            if attempt > 0:
                contents.append(
                    "Retry: return JSON only with numeric task_score/visual_score/spatial_score "
                    "between 0 and 10 and a non-empty reasoning field."
                )

            response, model_used = _generate_with_retry(
                client,
                model=config.model,
                fallback_models=config.fallback_models,
                contents=contents,
                config=_build_generate_config(
                    types,
                    enable_agentic_vision=config.enable_agentic_vision,
                    temperature=0.1,
                    includes_video=True,
                ),
            )
            try:
                raw_text = _extract_response_text(response)
                parsed = _extract_json_from_response(raw_text)
                task_score, visual_score, spatial_score, reasoning = _parse_judge_payload(parsed)
                return JudgeScore(
                    task_score=task_score,
                    visual_score=visual_score,
                    spatial_score=spatial_score,
                    reasoning=reasoning,
                    raw_response=raw_text,
                    model_used=model_used,
                )
            except Exception as e:
                errors.append(f"attempt={attempt + 1}: {e}")
                continue
    finally:
        _delete_uploaded_file(client, uploaded_video)

    raise RuntimeError(
        "VLM judge failed to return valid scoring JSON after retries: " + "; ".join(errors)
    )


def score_stage1_probe(
    video_path: Path,
    expected_focus_text: str,
    config: VLMJudgeConfig,
    facility_description: str = "",
) -> Stage1ProbeScore:
    """Score Stage-1 probe clips and return structured issue tags."""
    from google.genai import types

    client = _get_gemini_client(config)
    expected_focus = str(expected_focus_text).strip() or "task-relevant scene region"
    prompt = (
        "You are evaluating a Stage-1 camera probe clip for robot-world-model data quality.\n"
        f'Expected focus: "{expected_focus}"\n\n'
        "Return strict JSON with keys:\n"
        '- "task_score" (0-10): how well clip focus/framing matches expected focus\n'
        '- "visual_score" (0-10): clarity/sharpness and color quality\n'
        '- "spatial_score" (0-10): usefulness for downstream world-model/policy training\n'
        '- "issue_tags" (array): subset of ['
        + ", ".join(sorted(_STAGE1_ALLOWED_ISSUE_TAGS))
        + "]\n"
        '- "reasoning" (string)\n'
        "Use issue_tags=[] when no issues are detected."
    )
    if facility_description:
        prompt += f"\nFacility context: {facility_description}"

    errors = []
    uploaded_video = _upload_video_file(client, video_path)
    try:
        video_part = _build_uploaded_video_part(
            types,
            uploaded_video,
            video_metadata_fps=float(config.video_metadata_fps),
        )
        effective_fps = (
            float(config.video_metadata_fps) if float(config.video_metadata_fps) > 0.0 else None
        )
        logger.info(
            "Scoring Stage-1 probe with native Gemini video input: path=%s metadata_fps=%s",
            video_path,
            "default" if effective_fps is None else f"{effective_fps:.3f}",
        )
        for attempt in range(3):
            contents = [prompt, video_part]
            if attempt > 0:
                contents.append(
                    "Retry: return JSON only with required numeric fields, issue_tags array, and reasoning."
                )
            response, model_used = _generate_with_retry(
                client,
                model=config.model,
                fallback_models=config.fallback_models,
                contents=contents,
                config=_build_generate_config(
                    types,
                    enable_agentic_vision=config.enable_agentic_vision,
                    temperature=0.0,
                    includes_video=True,
                ),
            )
            try:
                raw_text = _extract_response_text(response)
                parsed = _extract_json_from_response(raw_text)
                task_score, visual_score, spatial_score, issue_tags, reasoning = (
                    _parse_stage1_probe_payload(parsed)
                )
                return Stage1ProbeScore(
                    task_score=task_score,
                    visual_score=visual_score,
                    spatial_score=spatial_score,
                    issue_tags=issue_tags,
                    reasoning=reasoning,
                    raw_response=raw_text,
                    model_used=model_used,
                )
            except Exception as e:
                errors.append(f"attempt={attempt + 1}: {e}")
                continue
    finally:
        _delete_uploaded_file(client, uploaded_video)

    raise RuntimeError(
        "Stage-1 probe judge failed to return valid JSON after retries: " + "; ".join(errors)
    )


def score_stage2_enriched_clip(
    video_path: Path,
    expected_focus_text: str,
    variant_prompt: str,
    config: VLMJudgeConfig,
    facility_description: str = "",
) -> Stage2QualityScore:
    """Score Stage-2 enriched clips with structured issue tags."""
    from google.genai import types

    client = _get_gemini_client(config)
    expected_focus = str(expected_focus_text).strip() or "task-relevant scene region"
    variant = str(variant_prompt).strip()
    prompt = (
        "You are evaluating a Stage-2 enriched robot-world-model training clip.\n"
        f'Expected focus: "{expected_focus}"\n'
    )
    if variant:
        prompt += f'Variant intent: "{variant}"\n'
    prompt += (
        "\nReturn strict JSON with keys:\n"
        '- "task_score" (0-10): whether focus/task intent is preserved\n'
        '- "visual_score" (0-10): realism, sharpness, and artifact/color quality\n'
        '- "spatial_score" (0-10): temporal/spatial coherence for model training\n'
        '- "issue_tags" (array): subset of ['
        + ", ".join(sorted(_STAGE2_ALLOWED_ISSUE_TAGS))
        + "]\n"
        '- "reasoning" (string)\n'
        "Use issue_tags=[] only when no issues are present."
    )
    if facility_description:
        prompt += f"\nFacility context: {facility_description}"

    errors = []
    uploaded_video = _upload_video_file(client, video_path)
    try:
        video_part = _build_uploaded_video_part(
            types,
            uploaded_video,
            video_metadata_fps=float(config.video_metadata_fps),
        )
        effective_fps = (
            float(config.video_metadata_fps) if float(config.video_metadata_fps) > 0.0 else None
        )
        logger.info(
            "Scoring Stage-2 enriched clip with native Gemini video input: path=%s metadata_fps=%s",
            video_path,
            "default" if effective_fps is None else f"{effective_fps:.3f}",
        )
        for attempt in range(3):
            contents = [prompt, video_part]
            if attempt > 0:
                contents.append(
                    "Retry: return JSON only with required numeric fields, issue_tags array, and reasoning."
                )
            response, model_used = _generate_with_retry(
                client,
                model=config.model,
                fallback_models=config.fallback_models,
                contents=contents,
                config=_build_generate_config(
                    types,
                    enable_agentic_vision=config.enable_agentic_vision,
                    temperature=0.0,
                    includes_video=True,
                ),
            )
            try:
                raw_text = _extract_response_text(response)
                parsed = _extract_json_from_response(raw_text)
                task_score, visual_score, spatial_score, issue_tags, reasoning = (
                    _parse_stage2_quality_payload(parsed)
                )
                return Stage2QualityScore(
                    task_score=task_score,
                    visual_score=visual_score,
                    spatial_score=spatial_score,
                    issue_tags=issue_tags,
                    reasoning=reasoning,
                    raw_response=raw_text,
                    model_used=model_used,
                )
            except Exception as e:
                errors.append(f"attempt={attempt + 1}: {e}")
                continue
    finally:
        _delete_uploaded_file(client, uploaded_video)

    raise RuntimeError(
        "Stage-2 enriched clip judge failed to return valid JSON after retries: "
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

    errors = []
    uploaded_video = _upload_video_file(client, video_path)
    try:
        video_part = _build_uploaded_video_part(
            types,
            uploaded_video,
            video_metadata_fps=float(config.video_metadata_fps),
        )
        effective_fps = (
            float(config.video_metadata_fps) if float(config.video_metadata_fps) > 0.0 else None
        )
        logger.info(
            "Scoring manipulation rollout with native Gemini video input: path=%s metadata_fps=%s",
            video_path,
            "default" if effective_fps is None else f"{effective_fps:.3f}",
        )
        for attempt in range(3):
            contents = [f"Manipulation rollout video. {prompt}", video_part]
            if attempt > 0:
                contents.append(
                    "Retry: return strict JSON only, include all required boolean manipulation fields."
                )

            response, model_used = _generate_with_retry(
                client,
                model=config.model,
                fallback_models=config.fallback_models,
                contents=contents,
                config=_build_generate_config(
                    types,
                    enable_agentic_vision=config.enable_agentic_vision,
                    temperature=0.1,
                    includes_video=True,
                ),
            )
            try:
                raw_text = _extract_response_text(response)
                parsed = _extract_json_from_response(raw_text)
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
                    model_used=model_used,
                )
            except Exception as e:
                errors.append(f"attempt={attempt + 1}: {e}")
                continue
    finally:
        _delete_uploaded_file(client, uploaded_video)

    raise RuntimeError(
        "VLM manipulation judge failed to return valid JSON after retries: " + "; ".join(errors)
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
        f"Analyze this generated-environment video.\n"
        f"The target facility is: {facility_description}\n"
        f"Expected landmarks:\n{landmarks_str}\n\n"
        f"Score on a 1-10 scale:\n"
        f"1. spatial_score: How well does the layout match the description?\n"
        f"2. visual_score: How photorealistic is the environment?\n"
        f"3. task_score: How many of the expected landmarks are visible?\n"
        f'Return JSON: {{"task_score": N, "visual_score": N, "spatial_score": N, "reasoning": "..."}}'
    )

    errors = []
    uploaded_video = _upload_video_file(client, video_path)
    try:
        video_part = _build_uploaded_video_part(
            types,
            uploaded_video,
            video_metadata_fps=float(config.video_metadata_fps),
        )
        effective_fps = (
            float(config.video_metadata_fps) if float(config.video_metadata_fps) > 0.0 else None
        )
        logger.info(
            "Scoring spatial accuracy with native Gemini video input: path=%s metadata_fps=%s",
            video_path,
            "default" if effective_fps is None else f"{effective_fps:.3f}",
        )
        for attempt in range(3):
            contents = [prompt, video_part]
            if attempt > 0:
                contents.append(
                    "Retry: return JSON only with task_score/visual_score/spatial_score in [0,10] "
                    "and a non-empty reasoning string."
                )
            response, model_used = _generate_with_retry(
                client,
                model=config.model,
                fallback_models=config.fallback_models,
                contents=contents,
                config=_build_generate_config(
                    types,
                    enable_agentic_vision=config.enable_agentic_vision,
                    temperature=0.1,
                    includes_video=True,
                ),
            )
            try:
                raw_text = _extract_response_text(response)
                parsed = _extract_json_from_response(raw_text)
                task_score, visual_score, spatial_score, reasoning = _parse_judge_payload(parsed)
                return JudgeScore(
                    task_score=task_score,
                    visual_score=visual_score,
                    spatial_score=spatial_score,
                    reasoning=reasoning,
                    raw_response=raw_text,
                    model_used=model_used,
                )
            except Exception as e:
                errors.append(f"attempt={attempt + 1}: {e}")
                continue
    finally:
        _delete_uploaded_file(client, uploaded_video)

    raise RuntimeError(
        "VLM spatial scorer failed to return valid JSON after retries: " + "; ".join(errors)
    )


def classify_facility(
    video_path: Path,
    facility_descriptions: dict[str, str],
    config: VLMJudgeConfig,
) -> dict:
    """Classify which facility a generated video depicts (for cross-site test)."""
    from google.genai import types

    client = _get_gemini_client(config)

    desc_str = "\n".join(f"- Facility {fid}: {desc}" for fid, desc in facility_descriptions.items())
    prompt = (
        f"Look at this generated-environment video.\n"
        f"Which facility does this most closely match?\n\n{desc_str}\n\n"
        f'Return JSON: {{"predicted_facility": "<facility_id>", "confidence": 0.0-1.0, "reasoning": "..."}}'
    )

    errors = []
    uploaded_video = _upload_video_file(client, video_path)
    try:
        video_part = _build_uploaded_video_part(
            types,
            uploaded_video,
            video_metadata_fps=float(config.video_metadata_fps),
        )
        effective_fps = (
            float(config.video_metadata_fps) if float(config.video_metadata_fps) > 0.0 else None
        )
        logger.info(
            "Classifying facility with native Gemini video input: path=%s metadata_fps=%s",
            video_path,
            "default" if effective_fps is None else f"{effective_fps:.3f}",
        )
        for attempt in range(3):
            contents = [prompt, video_part]
            if attempt > 0:
                contents.append(
                    "Retry: return JSON only with predicted_facility, confidence (0-1), reasoning."
                )
            response, model_used = _generate_with_retry(
                client,
                model=config.model,
                fallback_models=config.fallback_models,
                contents=contents,
                config=_build_generate_config(
                    types,
                    enable_agentic_vision=config.enable_agentic_vision,
                    temperature=0.1,
                    includes_video=True,
                ),
            )

            try:
                raw_text = _extract_response_text(response)
                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError:
                    parsed = _extract_json_from_response(raw_text)
                predicted, confidence, reasoning = _parse_classify_payload(parsed)
                return {
                    "predicted_facility": predicted,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "raw_response": raw_text,
                    "model_used": model_used,
                }
            except Exception as e:
                errors.append(f"attempt={attempt + 1}: {e}")
                continue
    finally:
        _delete_uploaded_file(client, uploaded_video)

    raise RuntimeError(
        "VLM facility classifier failed to return valid JSON after retries: " + "; ".join(errors)
    )
