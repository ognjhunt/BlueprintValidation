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
    out, model_used = _generate_with_retry(
        client,
        model="gemini-3-flash-preview",
        fallback_models=["gemini-2.5-flash"],
        contents=[],
        config={},
        max_retries=1,
    )
    assert out["ok"] is True
    assert out["model"] == "gemini-2.5-flash"
    assert model_used == "gemini-2.5-flash"
    assert client.models.calls == ["gemini-3-flash-preview", "gemini-2.5-flash"]


def test_build_uploaded_video_part_uses_metadata_fps():
    from blueprint_validation.evaluation.vlm_judge import _build_uploaded_video_part

    class _FakeVideoMetadata:
        def __init__(self, *, fps):
            self.fps = fps

    class _FakeFileData:
        def __init__(self, *, file_uri, mime_type):
            self.file_uri = file_uri
            self.mime_type = mime_type

    class _FakePart:
        def __init__(self, *, file_data, video_metadata):
            self.file_data = file_data
            self.video_metadata = video_metadata

    class _FakeTypes:
        Part = _FakePart
        FileData = _FakeFileData
        VideoMetadata = _FakeVideoMetadata

    class Uploaded:
        uri = "gs://bucket/sample.mp4"
        mime_type = "video/mp4"

    part = _build_uploaded_video_part(_FakeTypes, Uploaded(), video_metadata_fps=10.0)
    assert getattr(part, "video_metadata", None) is not None
    assert float(part.video_metadata.fps) == 10.0
    assert str(part.file_data.file_uri) == "gs://bucket/sample.mp4"


def test_build_uploaded_video_part_disables_when_fps_zero():
    from blueprint_validation.evaluation.vlm_judge import _build_uploaded_video_part

    class _FakeTypes:
        class Part:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FileData:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class VideoMetadata:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    class Uploaded:
        uri = "gs://bucket/sample.mp4"
        mime_type = "video/mp4"

    uploaded = Uploaded()
    out = _build_uploaded_video_part(_FakeTypes, uploaded, video_metadata_fps=0.0)
    assert out is uploaded


def test_build_generate_config_disables_agentic_tools_for_video():
    from blueprint_validation.evaluation.vlm_judge import _build_generate_config

    class _FakeGenerateContentConfig(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs

    class _FakeToolCodeExecution:
        pass

    class _FakeTool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeTypes:
        GenerateContentConfig = _FakeGenerateContentConfig
        ToolCodeExecution = _FakeToolCodeExecution
        Tool = _FakeTool

    out = _build_generate_config(
        _FakeTypes,
        enable_agentic_vision=True,
        temperature=0.1,
        includes_video=True,
    )
    assert "tools" not in out.kwargs
    assert out.kwargs["temperature"] == 0.1


def test_build_generate_config_keeps_agentic_tools_without_video():
    from blueprint_validation.evaluation.vlm_judge import _build_generate_config

    class _FakeGenerateContentConfig(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.kwargs = kwargs

    class _FakeToolCodeExecution:
        pass

    class _FakeTool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeTypes:
        GenerateContentConfig = _FakeGenerateContentConfig
        ToolCodeExecution = _FakeToolCodeExecution
        Tool = _FakeTool

    out = _build_generate_config(
        _FakeTypes,
        enable_agentic_vision=True,
        temperature=0.1,
        includes_video=False,
    )
    assert "tools" in out.kwargs
    assert len(out.kwargs["tools"]) == 1


def test_parse_stage1_probe_payload():
    from blueprint_validation.evaluation.vlm_judge import _parse_stage1_probe_payload

    payload = {
        "task_score": 8,
        "visual_score": 7,
        "spatial_score": 6,
        "issue_tags": ["target_off_center", "blur_or_soft_focus"],
        "reasoning": "target drifts and frames are a bit soft",
    }
    task, visual, spatial, tags, reasoning = _parse_stage1_probe_payload(payload)
    assert task == 8.0
    assert visual == 7.0
    assert spatial == 6.0
    assert tags == ["target_off_center", "blur_or_soft_focus"]
    assert "soft" in reasoning


def test_parse_stage2_quality_payload():
    from blueprint_validation.evaluation.vlm_judge import _parse_stage2_quality_payload

    payload = {
        "task_score": 8,
        "visual_score": 7,
        "spatial_score": 6,
        "issue_tags": ["semantic_mismatch", "green_cast"],
        "reasoning": "content drifts and a mild green tint is visible",
    }
    task, visual, spatial, tags, reasoning = _parse_stage2_quality_payload(payload)
    assert task == 8.0
    assert visual == 7.0
    assert spatial == 6.0
    assert tags == ["semantic_mismatch", "green_cast"]
    assert "green" in reasoning


def test_parse_stage2_quality_payload_rejects_unknown_issue_tag():
    from blueprint_validation.evaluation.vlm_judge import _parse_stage2_quality_payload

    payload = {
        "task_score": 8,
        "visual_score": 7,
        "spatial_score": 6,
        "issue_tags": ["totally_unknown_tag"],
        "reasoning": "unknown tag should fail strict parser",
    }
    with pytest.raises(ValueError, match="Unknown issue tag"):
        _parse_stage2_quality_payload(payload)


def test_upload_video_file_rejects_non_video_path(tmp_path):
    from blueprint_validation.evaluation.vlm_judge import _upload_video_file

    text_file = tmp_path / "secret.txt"
    text_file.write_text("top-secret", encoding="utf-8")

    class _Files:
        def upload(self, *, file):  # pragma: no cover - should not run
            raise AssertionError("upload should not be called")

    class _Client:
        files = _Files()

    with pytest.raises(ValueError, match="video MIME type"):
        _upload_video_file(_Client(), text_file)


def test_upload_video_file_uploads_video_path(tmp_path, monkeypatch):
    from blueprint_validation.evaluation import vlm_judge

    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"fake")

    uploaded_marker = object()
    waited_marker = object()

    class _Files:
        def __init__(self):
            self.uploaded = None

        def upload(self, *, file):
            self.uploaded = file
            return uploaded_marker

    class _Client:
        def __init__(self):
            self.files = _Files()

    client = _Client()

    monkeypatch.setattr(
        vlm_judge,
        "_wait_for_uploaded_file_active",
        lambda client_arg, uploaded: waited_marker,
    )

    out = vlm_judge._upload_video_file(client, video_file)
    assert client.files.uploaded == str(video_file)
    assert out is waited_marker
