"""Tests for OpenVLA policy fine-tuning adapter."""

from __future__ import annotations

import types
from pathlib import Path

import pytest


def test_build_openvla_finetune_command_contract(tmp_path):
    from blueprint_validation.config import PolicyFinetuneConfig
    from blueprint_validation.training.openvla_finetune import (
        build_openvla_finetune_command,
    )

    data_root = tmp_path / "datasets"
    cfg = PolicyFinetuneConfig(
        enabled=True,
        data_root_dir=data_root,
        dataset_name="bridge_orig",
        nproc_per_node=2,
        lora_rank=16,
        batch_size=4,
        grad_accumulation_steps=3,
        learning_rate=1e-4,
        save_steps=50,
        max_steps=200,
        image_aug=False,
    )
    script_path = Path("/opt/openvla/vla-scripts/finetune.py")
    cmd = build_openvla_finetune_command(
        script_path=script_path,
        config=cfg,
        vla_path="openvla/openvla-7b",
        run_root_dir=tmp_path / "runs",
        adapter_tmp_dir=tmp_path / "adapters",
    )

    assert cmd[0] == "torchrun"
    assert "--vla_path" in cmd
    assert "--data_root_dir" in cmd
    assert "--dataset_name" in cmd
    assert "--max_steps" in cmd
    assert "200" in cmd
    assert "--image_aug" in cmd
    assert "False" in cmd


def test_run_openvla_finetune_fails_without_dataset(tmp_path):
    from blueprint_validation.config import PolicyFinetuneConfig
    from blueprint_validation.training.openvla_finetune import run_openvla_finetune

    repo = tmp_path / "openvla"
    repo.mkdir()
    (repo / "vla-scripts").mkdir()
    (repo / "vla-scripts" / "finetune.py").write_text("print('stub')")

    cfg = PolicyFinetuneConfig(
        enabled=True,
        openvla_repo=repo,
        finetune_script="vla-scripts/finetune.py",
        data_root_dir=tmp_path / "datasets",
        dataset_name="bridge_orig",
    )
    cfg.data_root_dir.mkdir()

    with pytest.raises(RuntimeError, match="dataset directory missing"):
        run_openvla_finetune(
            config=cfg,
            vla_path="openvla/openvla-7b",
            facility_id="facility_a",
            output_dir=tmp_path / "out",
        )


def test_run_openvla_finetune_success_with_adapter_artifact(tmp_path, monkeypatch):
    from blueprint_validation.config import PolicyFinetuneConfig
    from blueprint_validation.training.openvla_finetune import run_openvla_finetune

    repo = tmp_path / "openvla"
    script = repo / "vla-scripts" / "finetune.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('stub')")
    data_root = tmp_path / "datasets"
    (data_root / "bridge_orig").mkdir(parents=True)

    cfg = PolicyFinetuneConfig(
        enabled=True,
        openvla_repo=repo,
        finetune_script="vla-scripts/finetune.py",
        data_root_dir=data_root,
        dataset_name="bridge_orig",
        max_steps=5,
    )

    monkeypatch.setattr(
        "blueprint_validation.training.openvla_finetune.shutil.which",
        lambda cmd: "/usr/bin/torchrun",
    )

    def fake_run(cmd, capture_output, text, cwd):
        del capture_output, text, cwd
        adapter_dir = Path(cmd[cmd.index("--adapter_tmp_dir") + 1])
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x")
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "blueprint_validation.training.openvla_finetune.subprocess.run",
        fake_run,
    )

    result = run_openvla_finetune(
        config=cfg,
        vla_path="openvla/openvla-7b",
        facility_id="facility_a",
        output_dir=tmp_path / "out",
    )
    assert result["status"] == "success"
    assert "adapted_checkpoint_path" in result
