"""Tests for the PolaRiS evaluation stage."""

from __future__ import annotations

import json
from pathlib import Path


def _write_scene_package(root: Path) -> None:
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "usd").mkdir(parents=True, exist_ok=True)
    (root / "geniesim").mkdir(parents=True, exist_ok=True)
    (root / "isaac_lab" / "scene_task").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "scene_manifest.json").write_text("{}")
    (root / "usd" / "scene.usda").write_text("#usda 1.0")
    (root / "geniesim" / "task_config.json").write_text(
        json.dumps(
            {
                "scene_id": "test_scene",
                "suggested_tasks": [
                    {"description_hint": "Pick up the mug and place it on the table"}
                ],
            }
        )
    )
    (root / "isaac_lab" / "scene_task" / "__init__.py").write_text("")
    (root / "isaac_lab" / "scene_task" / "blueprint_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "runtime_kind": "blueprint_scene_env",
                "task_package": "scene_task",
                "env_factory": "scene_task.create_env",
                "env_cfg_class": "TeleopEnvCfg",
                "action_dim": 7,
                "camera_keys": ["wrist_rgb"],
                "state_keys": ["policy"],
            }
        )
    )


def _write_fake_polaris_repo(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "src" / "polaris" / "policy").mkdir(parents=True, exist_ok=True)
    (root / "src" / "polaris" / "__init__.py").write_text("")
    (root / "src" / "polaris" / "config.py").write_text("class PolicyArgs:\n    pass\n")
    (root / "src" / "polaris" / "policy" / "__init__.py").write_text("")
    (root / "src" / "polaris" / "policy" / "abstract_client.py").write_text(
        "class InferenceClient:\n"
        "    registry = {}\n"
        "    @classmethod\n"
        "    def register(cls, client_name):\n"
        "        def _decorator(subcls):\n"
        "            cls.registry[client_name] = subcls\n"
        "            return subcls\n"
        "        return _decorator\n"
    )
    (root / "scripts" / "eval.py").write_text(
        "import argparse, csv\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--environment-name')\n"
        "parser.add_argument('--bundle-root')\n"
        "parser.add_argument('--output-dir')\n"
        "parser.add_argument('--num-rollouts', type=int, default=1)\n"
        "parser.add_argument('--policy-client')\n"
        "parser.add_argument('--policy-host')\n"
        "parser.add_argument('--policy-port')\n"
        "parser.add_argument('--observation-mode')\n"
        "parser.add_argument('--action-mode')\n"
        "parser.add_argument('--device')\n"
        "args = parser.parse_args()\n"
        "out = Path(args.output_dir)\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "with (out / 'eval_results.csv').open('w', newline='') as handle:\n"
        "    writer = csv.DictWriter(handle, fieldnames=['episode','success_score','progress'])\n"
        "    writer.writeheader()\n"
        "    for idx in range(int(args.num_rollouts)):\n"
        "        writer.writerow({'episode': idx, 'success_score': 0.8, 'progress': 0.9})\n"
    )
    return root


def test_s4f_polaris_eval_stage_uses_fake_backend(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4f_polaris_eval import PolarisEvalStage

    monkeypatch.setenv("BLUEPRINT_POLARIS_FAKE_BACKEND", "1")
    scene_root = tmp_path / "scene_pkg"
    _write_scene_package(scene_root)
    adapted_ckpt = tmp_path / "adapted"
    adapted_ckpt.mkdir()

    sample_config.eval_polaris.enabled = True
    sample_config.eval_polaris.default_as_primary_gate = True
    sample_config.facilities["test_facility"].scene_package_path = scene_root
    previous = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_policy_checkpoint": str(adapted_ckpt)},
        )
    }

    result = PolarisEvalStage().execute(
        config=sample_config,
        facility=sample_config.facilities["test_facility"],
        work_dir=tmp_path / "facility_work",
        previous_results=previous,
    )
    assert result.status == "success"
    assert result.metrics["winner"] in {"frozen_openvla", "adapted_openvla"}
    assert Path(result.outputs["polaris_summary_path"]).exists()


def test_s4f_polaris_eval_fails_without_primary_eligible_scene(sample_config, tmp_path):
    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4f_polaris_eval import PolarisEvalStage

    adapted_ckpt = tmp_path / "adapted"
    adapted_ckpt.mkdir()
    sample_config.eval_polaris.enabled = True
    sample_config.eval_polaris.default_as_primary_gate = True

    previous = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_policy_checkpoint": str(adapted_ckpt)},
        )
    }
    result = PolarisEvalStage().execute(
        config=sample_config,
        facility=sample_config.facilities["test_facility"],
        work_dir=tmp_path / "facility_work",
        previous_results=previous,
    )
    assert result.status == "failed"
    assert "primary-eligible" in result.detail


def test_s4f_polaris_eval_stage_runs_live_native_bundle(sample_config, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from blueprint_validation.common import StageResult
    from blueprint_validation.stages.s4f_polaris_eval import PolarisEvalStage

    adapted_ckpt = tmp_path / "adapted"
    adapted_ckpt.mkdir()
    repo_root = _write_fake_polaris_repo(tmp_path / "PolaRiS")
    native_root = tmp_path / "polaris_hub" / "DemoScene"
    native_root.mkdir(parents=True, exist_ok=True)
    (native_root / "scene.usda").write_text("#usda 1.0")
    (native_root / "initial_conditions.json").write_text("{}")

    sample_config.eval_polaris.enabled = True
    sample_config.eval_polaris.default_as_primary_gate = True
    sample_config.eval_polaris.environment_mode = "native_bundle"
    sample_config.eval_polaris.environment_name = "DemoScene"
    sample_config.eval_polaris.repo_path = repo_root
    sample_config.eval_polaris.hub_path = tmp_path / "polaris_hub"

    import blueprint_validation.polaris.runner as polaris_runner

    monkeypatch.setattr(
        polaris_runner,
        "_launch_policy_server",
        lambda **kwargs: SimpleNamespace(poll=lambda: 0),
    )
    monkeypatch.setattr(polaris_runner, "_terminate_process", lambda proc: None)

    previous = {
        "s3b_policy_finetune": StageResult(
            stage_name="s3b_policy_finetune",
            status="success",
            elapsed_seconds=0.0,
            outputs={"adapted_policy_checkpoint": str(adapted_ckpt)},
        )
    }
    result = PolarisEvalStage().execute(
        config=sample_config,
        facility=sample_config.facilities["test_facility"],
        work_dir=tmp_path / "facility_work",
        previous_results=previous,
    )
    assert result.status == "success"
    assert result.metrics["scene_mode"] == "native_bundle"
    assert result.metrics["adapted_success_rate"] == 0.8
