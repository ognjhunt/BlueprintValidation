"""Built-in and dynamic visual variant definitions for Cosmos Transfer enrichment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from ..common import get_logger
from ..config import VariantSpec

logger = get_logger("enrichment.variant_specs")

BUILTIN_VARIANTS = [
    VariantSpec(
        name="daylight_empty",
        prompt="Indoor industrial facility, bright natural daylight, empty clean corridors, polished concrete floors, overhead LED lighting",
    ),
    VariantSpec(
        name="daylight_occupied",
        prompt="Indoor industrial facility, bright daylight, workers in safety vests moving through space, forklifts, active workspace",
    ),
    VariantSpec(
        name="evening_dim",
        prompt="Indoor industrial facility, dim evening lighting, overhead fluorescent lights casting shadows, quiet empty space",
    ),
    VariantSpec(
        name="wet_floor",
        prompt="Indoor industrial facility, wet reflective concrete floors, caution signs, overhead lighting reflecting off puddles",
    ),
    VariantSpec(
        name="cluttered",
        prompt="Indoor industrial facility, boxes and pallets scattered throughout, equipment and supplies visible, busy workspace",
    ),
]


_DYNAMIC_VARIANT_PROMPT = """\
Analyze this rendered frame from a 3D scene scan. The scene may be a warehouse, \
office, lab, outdoor area, retail space, kitchen, or any other environment.

Based on what you see, generate exactly {num_variants} visually diverse variant \
prompts for a video-to-video diffusion model (Cosmos Transfer). Each prompt should:
1. Preserve the spatial layout and objects in the scene
2. Change lighting, weather, activity level, or surface conditions
3. Be realistic for the type of environment shown
4. Be specific and descriptive (15-30 words)

Return ONLY a JSON array with this exact format:
[
  {{"name": "short_snake_case_name", "prompt": "Detailed visual description..."}},
  {{"name": "another_name", "prompt": "Another detailed visual description..."}}
]
"""


def generate_dynamic_variants(
    sample_frame_path: Optional[Path] = None,
    sample_frame_rgb: Optional[object] = None,
    num_variants: int = 5,
    model: str = "gemini-3-flash-preview",
    facility_description: str = "",
) -> List[VariantSpec]:
    """Use Gemini to generate scene-appropriate variant prompts.

    Accepts either a path to a sample frame image or an RGB numpy array.
    Falls back to BUILTIN_VARIANTS on any failure.
    """
    api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_GENAI_API_KEY not set; using builtin variants")
        return BUILTIN_VARIANTS[:num_variants]

    try:
        from google import genai
        from PIL import Image
    except ImportError:
        logger.warning("google-genai or Pillow not installed; using builtin variants")
        return BUILTIN_VARIANTS[:num_variants]

    try:
        client = genai.Client(api_key=api_key)

        # Build the image input
        if sample_frame_path is not None and sample_frame_path.exists():
            image = Image.open(sample_frame_path)
        elif sample_frame_rgb is not None:
            import numpy as np

            image = Image.fromarray(np.asarray(sample_frame_rgb))
        else:
            logger.warning("No sample frame available; using builtin variants")
            return BUILTIN_VARIANTS[:num_variants]

        prompt = _DYNAMIC_VARIANT_PROMPT.format(num_variants=num_variants)
        if facility_description:
            prompt += f"\n\nAdditional context about the environment: {facility_description}"

        response = client.models.generate_content(
            model=model,
            contents=[prompt, image],
        )

        text = response.text.strip()
        # Extract JSON from response (may be wrapped in markdown code block)
        if "```" in text:
            start = text.index("```") + 3
            if text[start:].startswith("json"):
                start += 4
            end = text.index("```", start)
            text = text[start:end].strip()

        raw_variants = json.loads(text)
        if not isinstance(raw_variants, list) or len(raw_variants) == 0:
            raise ValueError("Expected non-empty JSON array")

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
                model,
                [v.name for v in variants],
            )
            return variants

    except Exception:
        logger.warning("Dynamic variant generation failed; using builtin variants", exc_info=True)

    return BUILTIN_VARIANTS[:num_variants]


def get_variants(
    custom_variants: list[VariantSpec] | None = None,
    dynamic: bool = False,
    dynamic_model: str = "gemini-3-flash-preview",
    sample_frame_path: Optional[Path] = None,
    sample_frame_rgb: Optional[object] = None,
    num_variants: int = 5,
    facility_description: str = "",
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
        )
    return BUILTIN_VARIANTS
