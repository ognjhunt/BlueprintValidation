"""Tests for pi0.5 fine-tuning orchestration."""

from __future__ import annotations

import types
from pathlib import Path


def test_build_pi05_commands(tmp_path):
    from blueprint_validation.config import PolicyFinetuneConfig
    from blueprint_validation.training.pi05_finetune import (
        build_pi05_norm_stats_command,
        build_pi05_train_command,
    )

    repo = tmp_path / "openpi"
    (repo / "scripts").mkdir(parents=True)
    dataset_root = tmp_path / "datasets"
    cfg = PolicyFinetuneConfig(batch_size=4, learning_rate=1e-4, max_steps=123)

    norm_cmd = build_pi05_norm_stats_command(
        openpi_repo=repo,
        norm_stats_script="scripts/compute_norm_stats.py",
        dataset_root=dataset_root,
        dataset_name="bridge_orig",
        profile="pi05_libero",
    )
    train_cmd = build_pi05_train_command(
        openpi_repo=repo,
        train_script="scripts/train_pytorch.py",
        dataset_root=dataset_root,
        dataset_name="bridge_orig",
        profile="pi05_libero",
        exp_name="exp123",
        run_root_dir=tmp_path / "runs",
        base_model_name="pi/model",
        base_checkpoint=None,
        finetune_config=cfg,
        extra_args=["--foo", "bar"],
    )

    assert "compute_norm_stats.py" in " ".join(norm_cmd)
    assert "--dataset_name" in norm_cmd and "bridge_orig" in norm_cmd
    assert "train_pytorch.py" in " ".join(train_cmd)
    assert "--exp_name" in train_cmd and "exp123" in train_cmd
    assert "--max_steps" in train_cmd and "123" in train_cmd
    assert "--foo" in train_cmd


def test_resolve_latest_pi05_checkpoint(tmp_path):
    from blueprint_validation.training.pi05_finetune import resolve_latest_pi05_checkpoint

    run_root = tmp_path / "runs"
    old_dir = run_root / "exp_old" / "checkpoints" / "step_001"
    new_dir = run_root / "exp_new" / "checkpoints" / "step_002"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "model.safetensors").write_bytes(b"old")
    (new_dir / "model.safetensors").write_bytes(b"new")

    resolved = resolve_latest_pi05_checkpoint(run_root)
    assert resolved == new_dir


def test_run_pi05_finetune_failure_from_norm_stats(tmp_path, monkeypatch):
    from blueprint_validation.config import Pi05AdapterBackendConfig, PolicyFinetuneConfig
    from blueprint_validation.training.pi05_finetune import run_pi05_finetune

    repo = tmp_path / "openpi"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "train_pytorch.py").write_text("print('stub')")
    (repo / "scripts" / "compute_norm_stats.py").write_text("print('stub')")
    dataset_root = tmp_path / "datasets"
    (dataset_root / "bridge_orig").mkdir(parents=True)

    backend = Pi05AdapterBackendConfig(openpi_repo=repo)
    cfg = PolicyFinetuneConfig(enabled=True)

    monkeypatch.setattr(
        "blueprint_validation.training.pi05_finetune.subprocess.run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=1, stdout="", stderr="norm stats failed"
        ),
    )

    result = run_pi05_finetune(
        config=cfg,
        backend=backend,
        base_model_name="pi/model",
        base_checkpoint=None,
        dataset_root=dataset_root,
        dataset_name="bridge_orig",
        facility_id="fac_a",
        output_dir=tmp_path / "out",
    )
    assert result["status"] == "failed"
    assert result["phase"] == "norm_stats"


def test_run_pi05_finetune_success(tmp_path, monkeypatch):
    from blueprint_validation.config import Pi05AdapterBackendConfig, PolicyFinetuneConfig
    from blueprint_validation.training.pi05_finetune import run_pi05_finetune

    repo = tmp_path / "openpi"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "train_pytorch.py").write_text("print('stub')")
    (repo / "scripts" / "compute_norm_stats.py").write_text("print('stub')")
    dataset_root = tmp_path / "datasets"
    (dataset_root / "bridge_orig").mkdir(parents=True)
    backend = Pi05AdapterBackendConfig(openpi_repo=repo)
    cfg = PolicyFinetuneConfig(enabled=True)

    call_count = {"n": 0}

    def fake_run(cmd, cwd, capture_output, text):
        del cwd, capture_output, text
        call_count["n"] += 1
        if call_count["n"] == 1:
            return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
        run_root = Path(cmd[cmd.index("--run_root_dir") + 1])
        ckpt = run_root / "exp" / "checkpoints" / "step_001"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "model.safetensors").write_bytes(b"x")
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("blueprint_validation.training.pi05_finetune.subprocess.run", fake_run)

    result = run_pi05_finetune(
        config=cfg,
        backend=backend,
        base_model_name="pi/model",
        base_checkpoint=None,
        dataset_root=dataset_root,
        dataset_name="bridge_orig",
        facility_id="fac_a",
        output_dir=tmp_path / "out",
    )
    assert result["status"] == "success"
    assert "adapted_checkpoint_path" in result
