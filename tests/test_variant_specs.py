from __future__ import annotations

import sys
import types

import pytest

from blueprint_validation.enrichment.variant_specs import (
    DynamicVariantGenerationError,
    _candidate_dynamic_models,
    generate_dynamic_variants,
    get_variants,
)


def test_generate_dynamic_variants_no_api_key_allows_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    variants = generate_dynamic_variants(num_variants=2, allow_fallback=True)
    assert len(variants) == 2


def test_generate_dynamic_variants_no_api_key_raises_when_fallback_disabled(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    with pytest.raises(DynamicVariantGenerationError):
        generate_dynamic_variants(num_variants=2, allow_fallback=False)


def test_get_variants_dynamic_no_fallback_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    with pytest.raises(DynamicVariantGenerationError):
        get_variants(
            dynamic=True,
            allow_dynamic_fallback=False,
            num_variants=1,
        )


def test_candidate_dynamic_models_includes_2_5_fallback():
    assert _candidate_dynamic_models("gemini-3-flash-preview") == [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ]
    assert _candidate_dynamic_models("gemini-2.5-flash") == ["gemini-2.5-flash"]


def test_generate_dynamic_variants_retries_with_2_5_flash(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "test-key")

    tried_models: list[str] = []

    class FakeModels:
        def generate_content(self, model, contents):
            tried_models.append(model)
            if model == "gemini-3-flash-preview":
                raise RuntimeError("model unavailable")
            return types.SimpleNamespace(
                text='[{"name":"kitchen_day","prompt":"Keep same kitchen and robot arm; warm daylight."}]'
            )

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.models = FakeModels()

    fake_genai = types.SimpleNamespace(Client=FakeClient)
    fake_pil_image = types.SimpleNamespace(fromarray=lambda arr: arr, open=lambda path: path)
    fake_pil = types.SimpleNamespace(Image=fake_pil_image)

    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    variants = generate_dynamic_variants(
        sample_frame_rgb=[[[0, 0, 0]]],
        num_variants=1,
        model="gemini-3-flash-preview",
        allow_fallback=False,
    )
    assert len(variants) == 1
    assert variants[0].name == "kitchen_day"
    assert tried_models == ["gemini-3-flash-preview", "gemini-2.5-flash"]
