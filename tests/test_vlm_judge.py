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


def test_extract_json_fallback():
    from blueprint_validation.evaluation.vlm_judge import _extract_json_from_response

    text = "I could not analyze the video properly."
    result = _extract_json_from_response(text)
    assert result["task_score"] == 0
    assert "could not analyze" in result["reasoning"]


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
