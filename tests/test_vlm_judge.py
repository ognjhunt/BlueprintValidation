"""Tests for VLM judge scoring (with mocked responses)."""

import pytest


def test_extract_json_from_clean():
    from blueprint_validation.evaluation.vlm_judge import _extract_json_from_response

    text = '{"task_score": 8, "visual_score": 7, "spatial_score": 6, "reasoning": "good"}'
    result = _extract_json_from_response(text)
    assert result["task_score"] == 8
    assert result["visual_score"] == 7
    assert result["spatial_score"] == 6


def test_extract_json_from_markdown():
    from blueprint_validation.evaluation.vlm_judge import _extract_json_from_response

    text = """Here is my analysis:

```json
{"task_score": 9, "visual_score": 8, "spatial_score": 7, "reasoning": "excellent"}
```

The robot performed well."""
    result = _extract_json_from_response(text)
    assert result["task_score"] == 9


def test_extract_json_from_mixed():
    from blueprint_validation.evaluation.vlm_judge import _extract_json_from_response

    text = 'Some preamble text {"task_score": 5, "visual_score": 4, "spatial_score": 3, "reasoning": "ok"} more text'
    result = _extract_json_from_response(text)
    assert result["task_score"] == 5


def test_extract_json_invalid_raises():
    from blueprint_validation.evaluation.vlm_judge import _extract_json_from_response

    text = "I could not analyze the video properly."
    with pytest.raises(ValueError, match="Could not parse JSON"):
        _extract_json_from_response(text)


def test_judge_score_dataclass():
    from blueprint_validation.evaluation.vlm_judge import JudgeScore

    score = JudgeScore(
        task_score=7.5,
        visual_score=8.0,
        spatial_score=6.5,
        reasoning="Solid performance",
        raw_response="...",
    )
    assert score.task_score == 7.5
    assert score.reasoning == "Solid performance"


def test_parse_manipulation_payload():
    from blueprint_validation.evaluation.vlm_judge import _parse_manipulation_payload

    payload = {
        "task_score": 9,
        "visual_score": 8,
        "spatial_score": 7,
        "grasp_acquired": True,
        "lifted_clear": True,
        "placed_in_target": False,
        "stable_after_place": True,
        "reasoning": "close but missed placement",
    }
    out = _parse_manipulation_payload(payload)
    assert out[0] == 9.0
    assert out[3] is True
    assert out[5] is False


def test_is_quota_exhausted_error_detection():
    from blueprint_validation.evaluation.vlm_judge import _is_quota_exhausted_error

    msg = "429 RESOURCE_EXHAUSTED quota exceeded for generate_content_free_tier_requests"
    assert _is_quota_exhausted_error(msg) is True
    assert _is_quota_exhausted_error("500 internal server error") is False


def test_generate_with_retry_falls_back_on_quota():
    from blueprint_validation.evaluation.vlm_judge import _generate_with_retry

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, *, model, contents, config):
            self.calls.append(model)
            if model == "gemini-3-flash-preview":
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED quota exceeded "
                    "generativelanguage.googleapis.com/generate_content_free_tier_requests"
                )
            return {"ok": True, "model": model}

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    client = FakeClient()
    out = _generate_with_retry(
        client,
        model="gemini-3-flash-preview",
        fallback_models=["gemini-2.5-flash"],
        contents=[],
        config={},
        max_retries=1,
    )
    assert out["ok"] is True
    assert out["model"] == "gemini-2.5-flash"
    assert client.models.calls == ["gemini-3-flash-preview", "gemini-2.5-flash"]
