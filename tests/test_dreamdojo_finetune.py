"""Tests for DreamDojo fine-tune command construction."""

from __future__ import annotations

from pathlib import Path


def test_resolve_experiment_config_by_name(tmp_path):
    from blueprint_validation.training.dreamdojo_finetune import _resolve_experiment_config

    root = tmp_path / "DreamDojo"
    cfg = root / "configs" / "post-training" / "site_adapt.sh"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("EXP_NAME=test\n")

    resolved = _resolve_experiment_config(root, "post-training/site_adapt")
    assert resolved == cfg.resolve()


def test_render_dreamdojo_config_script(tmp_path):
    from blueprint_validation.config import FinetuneConfig
    from blueprint_validation.training.dreamdojo_finetune import render_dreamdojo_config_script

    base_cfg = tmp_path / "base.sh"
    base_cfg.write_text("EXP_NAME=base\n")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    script = render_dreamdojo_config_script(
        base_config=base_cfg,
        dataset_dir=dataset,
        output_dir=out,
        config=FinetuneConfig(),
        facility_id="facility_a",
    )
    text = script.read_text()
    assert f'source "{base_cfg}"' in text
    assert "LEARNING_RATE" in text
    assert "DATASET_PATH" in text
    assert "LORA_RANK" in text
    assert "USE_LORA" in text
    assert "TRAIN_ARCHITECTURE" in text


def test_build_dreamdojo_launch_command(tmp_path):
    from blueprint_validation.training.dreamdojo_finetune import build_dreamdojo_launch_command

    root = tmp_path / "DreamDojo"
    launch = root / "launch.sh"
    launch.parent.mkdir(parents=True)
    launch.write_text("#!/usr/bin/env bash\n")
    cfg = tmp_path / "config.sh"
    cfg.write_text("EXP_NAME=test\n")

    cmd = build_dreamdojo_launch_command(root, cfg)
    assert cmd[0] == "bash"
    assert cmd[1] == str(launch)
    assert cmd[2] == str(cfg)
