from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from blueprint_validation.stages.s6_spatial_accuracy import SpatialAccuracyStage


def test_spatial_accuracy_fails_on_reasoning_conflict(sample_config, tmp_path, monkeypatch):
    sample_config.eval_spatial.min_valid_samples = 1
    sample_config.eval_spatial.fail_on_reasoning_conflict = True
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "fac"
    enriched_dir = work_dir / "enriched"
    enriched_dir.mkdir(parents=True, exist_ok=True)

    clip_path = work_dir / "clip.mp4"
    clip_path.write_bytes(b"00")
    (enriched_dir / "enriched_manifest.json").write_text(
        '{"clips":[{"clip_name":"clip_000","variant_name":"v0","output_video_path":"%s"}]}'
        % str(clip_path)
    )

    monkeypatch.setattr(
        "blueprint_validation.stages.s6_spatial_accuracy.score_spatial_accuracy",
        lambda **kwargs: SimpleNamespace(
            spatial_score=10.0,
            visual_score=10.0,
            task_score=10.0,
            reasoning="The scene is too distorted and cannot evaluate landmarks.",
        ),
    )

    result = SpatialAccuracyStage().execute(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert result.metrics["num_reasoning_conflicts"] == 1
    assert result.metrics["num_valid_samples"] == 0


def test_spatial_accuracy_enforces_min_valid_samples(sample_config, tmp_path, monkeypatch):
    sample_config.eval_spatial.min_valid_samples = 2
    sample_config.eval_spatial.fail_on_reasoning_conflict = True
    fac = sample_config.facilities["test_facility"]
    work_dir = tmp_path / "fac"
    enriched_dir = work_dir / "enriched"
    enriched_dir.mkdir(parents=True, exist_ok=True)

    clip_a = work_dir / "clip_a.mp4"
    clip_b = work_dir / "clip_b.mp4"
    clip_a.write_bytes(b"00")
    clip_b.write_bytes(b"00")
    (enriched_dir / "enriched_manifest.json").write_text(
        '{"clips":['
        '{"clip_name":"clip_a","variant_name":"v0","output_video_path":"%s"},'
        '{"clip_name":"clip_b","variant_name":"v0","output_video_path":"%s"}'
        "]}"
        % (str(clip_a), str(clip_b))
    )

    def _fake_score(*, video_path: Path, **kwargs):
        if video_path == clip_a:
            return SimpleNamespace(
                spatial_score=8.0,
                visual_score=8.0,
                task_score=8.0,
                reasoning="Layout is clear and landmarks are visible.",
            )
        return SimpleNamespace(
            spatial_score=8.0,
            visual_score=8.0,
            task_score=8.0,
            reasoning="Target object not visible; cannot evaluate spatial layout.",
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s6_spatial_accuracy.score_spatial_accuracy",
        _fake_score,
    )

    result = SpatialAccuracyStage().execute(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert result.metrics["num_reasoning_conflicts"] == 1
    assert result.metrics["num_valid_samples"] == 1
