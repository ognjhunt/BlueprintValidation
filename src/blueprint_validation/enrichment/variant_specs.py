"""Built-in and dynamic visual variant definitions for Cosmos Transfer enrichment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from ..common import get_logger
from ..config import VariantSpec

logger = get_logger("enrichment.variant_specs")


class DynamicVariantGenerationError(RuntimeError):
    """Raised when dynamic variant generation is required but unavailable."""


BUILTIN_VARIANTS = [
    VariantSpec(
        name="daylight_clean",
        prompt=(
            "Preserve the exact same scene layout, camera motion, object identities, and visible "
            "robot arm trajectory from the input video. Bright natural daylight, clean appearance."
        ),
    ),
    VariantSpec(
        name="warm_indoor_lighting",
        prompt=(
            "Preserve the exact same scene layout, camera motion, object identities, and visible "
            "robot arm trajectory from the input video. Warm indoor lighting with soft shadows."
        ),
    ),
    VariantSpec(
        name="evening_dim",
        prompt=(
            "Preserve the exact same scene layout, camera motion, object identities, and visible "
            "robot arm trajectory from the input video. Dim evening lighting and low ambient light."
        ),
    ),
    VariantSpec(
        name="overcast_soft_light",
        prompt=(
            "Preserve the exact same scene layout, camera motion, object identities, and visible "
            "robot arm trajectory from the input video. Neutral overcast-style soft lighting."
        ),
    ),
    VariantSpec(
        name="bright_task_lighting",
        prompt=(
            "Preserve the exact same scene layout, camera motion, object identities, and visible "
            "robot arm trajectory from the input video. Bright task lighting with clear object edges."
        ),
    ),
]


_DYNAMIC_VARIANT_PROMPT = """\
Analyze this rendered frame from a 3D scene scan. The scene may be a warehouse, \
office, lab, outdoor area, retail space, kitchen, or any other environment.

Based on what you see, generate exactly {num_variants} visually diverse variant \
prompts for a video-to-video diffusion model (Cosmos Transfer). Each prompt should:
1. Preserve the exact scene identity, spatial layout, fixed objects, and camera trajectory.
2. If a robot arm/tool/end-effector appears, preserve its geometry and trajectory.
3. Keep the same environment type (do not change apartment/kitchen into warehouse/factory/etc.).
4. Only vary lighting, mood, activity level, and subtle surface conditions.
5. Be specific and descriptive (15-30 words).

Return ONLY a JSON array with this exact format:
[
  {{"name": "short_snake_case_name", "prompt": "Detailed visual description..."}},
  {{"name": "another_name", "prompt": "Another detailed visual description..."}}
]
"""


def _candidate_dynamic_models(primary_model: str) -> list[str]:
    """Build ordered candidate model list for dynamic variant generation."""
    primary = (primary_model or "").strip() or "gemini-3-flash-preview"
    fallback = "gemini-2.5-flash"
    candidates = [primary]
    if fallback not in candidates:
        candidates.append(fallback)
    return candidates


def _parse_variant_response_text(text: str) -> list[dict]:
    """Parse model text response into a JSON list of variant dicts."""
    normalized = text.strip()
    if "```" in normalized:
        start = normalized.index("```") + 3
        if normalized[start:].startswith("json"):
            start += 4
        end = normalized.index("```", start)
        normalized = normalized[start:end].strip()
    data = json.loads(normalized)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Expected non-empty JSON array")
    return data


def generate_dynamic_variants(
    sample_frame_path: Optional[Path] = None,
    sample_frame_rgb: Optional[object] = None,
    num_variants: int = 5,
    model: str = "gemini-3-flash-preview",
    facility_description: str = "",
    allow_fallback: bool = True,
) -> List[VariantSpec]:
    """Use Gemini to generate scene-appropriate variant prompts.

    Accepts either a path to a sample frame image or an RGB numpy array.
    Falls back to BUILTIN_VARIANTS on any failure.
    """
    def _fallback_or_raise(message: str, exc: Exception | None = None) -> List[VariantSpec]:
        if allow_fallback:
            logger.warning("%s; using builtin variants", message)
            return BUILTIN_VARIANTS[:num_variants]
        if exc is not None:
            raise DynamicVariantGenerationError(message) from exc
        raise DynamicVariantGenerationError(message)

    api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        return _fallback_or_raise("GOOGLE_GENAI_API_KEY not set")

    try:
        from google import genai
        from PIL import Image
    except ImportError as exc:
        return _fallback_or_raise("google-genai or Pillow not installed", exc=exc)

    try:
        client = genai.Client(api_key=api_key)

        # Build the image input
        if sample_frame_path is not None and sample_frame_path.exists():
            image = Image.open(sample_frame_path)
        elif sample_frame_rgb is not None:
            import numpy as np

            image = Image.fromarray(np.asarray(sample_frame_rgb))
        else:
            return _fallback_or_raise("No sample frame available")

        prompt = _DYNAMIC_VARIANT_PROMPT.format(num_variants=num_variants)
        if facility_description:
            prompt += f"\n\nAdditional context about the environment: {facility_description}"

        last_exc: Exception | None = None
        model_candidates = _candidate_dynamic_models(model)
        for idx, candidate_model in enumerate(model_candidates):
            if idx > 0:
                logger.info("Retrying dynamic variant generation with fallback model: %s", candidate_model)
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=[prompt, image],
                )

                raw_variants = _parse_variant_response_text(response.text or "")

                variants = []
                for v in raw_variants[:num_variants]:
                    name = v.get("name", f"variant_{len(variants)}")
                    prompt_text = v.get("prompt", "")
                    if not prompt_text:
                        continue
                    # Sanitize name to be filesystem-safe
                    name = name.lower().replace(" ", "_").replace("-", "_")
                    name = "".join(c for c in name if c.isalnum() or c == "_")
                    variants.append(VariantSpec(name=name, prompt=prompt_text))

                if variants:
                    logger.info(
                        "Generated %d dynamic variants via %s: %s",
                        len(variants),
                        candidate_model,
                        [v.name for v in variants],
                    )
                    return variants
                raise ValueError("Parsed zero usable variants")
            except Exception as exc:
                last_exc = exc
                if idx + 1 < len(model_candidates):
                    logger.warning(
                        "Dynamic variant generation failed via %s; trying next candidate",
                        candidate_model,
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "Dynamic variant generation failed via %s; no more model fallbacks",
                        candidate_model,
                        exc_info=True,
                    )

        if last_exc is not None:
            return _fallback_or_raise(
                f"Dynamic variant generation failed for models: {', '.join(model_candidates)}",
                exc=last_exc,
            )
        return _fallback_or_raise("Dynamic variant generation failed for unknown reason")
    except Exception as exc:
        if allow_fallback:
            logger.warning("Dynamic variant generation failed; using builtin variants", exc_info=True)
        return _fallback_or_raise("Dynamic variant generation failed", exc=exc)

    return BUILTIN_VARIANTS[:num_variants]


def get_variants(
    custom_variants: list[VariantSpec] | None = None,
    dynamic: bool = False,
    dynamic_model: str = "gemini-3-flash-preview",
    sample_frame_path: Optional[Path] = None,
    sample_frame_rgb: Optional[object] = None,
    num_variants: int = 5,
    facility_description: str = "",
    allow_dynamic_fallback: bool = True,
) -> list[VariantSpec]:
    """Return variant specs, preferring custom > dynamic > builtins."""
    if custom_variants:
        return custom_variants
    if dynamic:
        return generate_dynamic_variants(
            sample_frame_path=sample_frame_path,
            sample_frame_rgb=sample_frame_rgb,
            num_variants=num_variants,
            model=dynamic_model,
            facility_description=facility_description,
            allow_fallback=allow_dynamic_fallback,
        )
    return BUILTIN_VARIANTS
