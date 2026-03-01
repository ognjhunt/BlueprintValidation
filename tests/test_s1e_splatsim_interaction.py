"""Tests for Stage 1e minimal SplatSim interaction stage."""

from __future__ import annotations

from pathlib import Path


def _write_source_manifest(path: Path) -> None:
    from blueprint_validation.common import write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "facility": "Test Facility",
            "clips": [
                {
                    "clip_name": "clip_000",
                    "path_type": "orbit",
                    "clip_index": 0,
                    "num_frames": 4,
                    "resolution": [48, 64],
                    "fps": 5,
                    "video_path": "/tmp/placeholder.mp4",
                    "depth_video_path": "",
                }
            ],
        },
        path,
    )


def test_stage_s1e_skips_when_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = False
    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"


def test_stage_s1e_success_path(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = True
    sample_config.splatsim.mode = "hybrid"

    source_manifest = tmp_path / "renders" / "render_manifest.json"
    _write_source_manifest(source_manifest)

    def _fake_backend(**kwargs):
        stage_dir = kwargs["stage_dir"]
        manifest_path = stage_dir / "interaction_manifest.json"
        write_json(
            {
                "facility": "Test Facility",
                "clips": [],
                "backend_used": "pybullet",
                "fallback_used": False,
            },
            manifest_path,
        )
        return {
            "status": "success",
            "reason": "ok",
            "manifest_path": str(manifest_path),
            "num_source_clips": 1,
            "num_generated_clips": 1,
            "num_successful_rollouts": 1,
            "fallback_used": False,
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s1e_splatsim_interaction.run_splatsim_pybullet_backend",
        _fake_backend,
    )

    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["fallback_used"] is False
    assert result.metrics["num_generated_clips"] == 1


def test_stage_s1e_hybrid_fallback(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import read_json
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = True
    sample_config.splatsim.mode = "hybrid"
    sample_config.splatsim.fallback_to_prior_manifest = True

    source_manifest = tmp_path / "renders" / "render_manifest.json"
    _write_source_manifest(source_manifest)

    def _fake_backend(**kwargs):
        del kwargs
        return {
            "status": "failed",
            "reason": "pybullet_unavailable",
            "manifest_path": "",
            "num_source_clips": 1,
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s1e_splatsim_interaction.run_splatsim_pybullet_backend",
        _fake_backend,
    )

    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "success"
    assert result.metrics["fallback_used"] is True

    manifest = read_json(tmp_path / "splatsim" / "interaction_manifest.json")
    assert manifest["fallback_used"] is True
    assert len(manifest["clips"]) == 1


def test_stage_s1e_prefers_previous_results_manifest(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult, write_json
    from blueprint_validation.stages.s1e_splatsim_interaction import SplatSimInteractionStage

    sample_config.splatsim.enabled = True
    sample_config.splatsim.mode = "hybrid"

    render_manifest = tmp_path / "renders" / "render_manifest.json"
    _write_source_manifest(render_manifest)

    # Stale higher-priority manifest on disk should be ignored in pipeline execution.
    stale_aug = tmp_path / "gaussian_augment" / "augmented_manifest.json"
    stale_aug.parent.mkdir(parents=True, exist_ok=True)
    write_json({"clips": []}, stale_aug)

    captured: dict = {}

    def _fake_backend(**kwargs):
        captured["source_manifest_path"] = kwargs["source_manifest_path"]
        stage_dir = kwargs["stage_dir"]
        manifest_path = stage_dir / "interaction_manifest.json"
        write_json({"clips": []}, manifest_path)
        return {
            "status": "success",
            "reason": "ok",
            "manifest_path": str(manifest_path),
            "num_source_clips": 1,
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    monkeypatch.setattr(
        "blueprint_validation.stages.s1e_splatsim_interaction.run_splatsim_pybullet_backend",
        _fake_backend,
    )

    previous_results = {
        "s1_render": StageResult(
            stage_name="s1_render",
            status="success",
            elapsed_seconds=0,
            outputs={"manifest_path": str(render_manifest)},
        )
    }
    stage = SplatSimInteractionStage()
    fac = list(sample_config.facilities.values())[0]
    result = stage.run(sample_config, fac, tmp_path, previous_results)
    assert result.status == "success"
    assert captured["source_manifest_path"] == render_manifest
    assert result.metrics["source_stage"] == "s1_render"
    assert result.metrics["source_mode"] == "previous_results"
