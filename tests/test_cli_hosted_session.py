from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_hosted_session_cli_flow(monkeypatch, tmp_path, sample_config):
    import blueprint_validation.cli as cli_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: v1\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "load_config", lambda _path: sample_config)

    runtime_dir = tmp_path / "runtime"
    task_anchor_path = runtime_dir / "task_anchor_manifest.json"
    task_run_path = runtime_dir / "task_run_manifest.json"
    scene_memory_manifest_path = runtime_dir / "scene_memory_manifest.json"
    conditioning_bundle_path = runtime_dir / "conditioning_bundle.json"

    _write_json(
        task_anchor_path,
        {
            "schema_version": "v1",
            "tasks": [
                {
                    "task_id": "task-1",
                    "task_text": "Walk to shelf staging and pick the blue tote",
                }
            ],
        },
    )
    _write_json(task_run_path, {"schema_version": "v1"})
    _write_json(scene_memory_manifest_path, {"schema_version": "v1"})
    _write_json(conditioning_bundle_path, {"schema_version": "v1"})
    runtime_manifest_path = runtime_dir / "hosted_session_runtime_manifest.json"
    _write_json(
        runtime_manifest_path,
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "cap-1",
            "site_submission_id": "site-sub-1",
            "pipeline_prefix": "scenes/scene-1/captures/cap-1/pipeline",
            "scene_memory_manifest_uri": str(scene_memory_manifest_path),
            "conditioning_bundle_uri": str(conditioning_bundle_path),
            "preview_simulation_manifest_uri": None,
            "task_anchor_manifest_uri": str(task_anchor_path),
            "task_run_manifest_uri": str(task_run_path),
            "available_backends": ["neoverse"],
            "default_backend": "neoverse",
            "task_ids": ["task-1"],
            "task_texts": ["Walk to shelf staging and pick the blue tote"],
            "start_states": ["Dock-side tote stack"],
            "scenario_variants": ["Normal lighting"],
            "launchable": True,
            "launch_blockers": [],
        },
    )

    work_dir = tmp_path / "session-work"
    runner = CliRunner()
    create = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "create",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--runtime-manifest",
            str(runtime_manifest_path),
            "--robot",
            "Unitree G1",
            "--task",
            "Walk to shelf staging and pick the blue tote",
            "--scenario",
            "Normal lighting",
            "--policy-json",
            json.dumps({"adapter_name": "mock", "model_name": "mock-policy"}),
        ],
    )
    assert create.exit_code == 0
    create_payload = json.loads(create.output)
    assert create_payload["runtime_backend_selected"] == "neoverse"

    reset = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "reset",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
        ],
    )
    assert reset.exit_code == 0
    reset_payload = json.loads(reset.output)
    assert reset_payload["episode"]["episodeId"].startswith("episode-")

    step = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "step",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--episode-id",
            reset_payload["episode"]["episodeId"],
        ],
    )
    assert step.exit_code == 0
    step_payload = json.loads(step.output)
    assert step_payload["episode"]["stepIndex"] == 1

    batch = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "run-batch",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
            "--num-episodes",
            "3",
        ],
    )
    assert batch.exit_code == 0
    batch_payload = json.loads(batch.output)
    assert batch_payload["summary"]["numEpisodes"] == 3

    export = runner.invoke(
        cli_module.cli,
        [
            "--config",
            str(config_path),
            "session",
            "export",
            "--session-id",
            "session-1",
            "--session-work-dir",
            str(work_dir),
        ],
    )
    assert export.exit_code == 0
    export_payload = json.loads(export.output)
    assert export_payload["artifact_uris"]["export_manifest"].endswith("export_manifest.json")
