"""Tests for DreamDojo fine-tune command construction."""

from __future__ import annotations

import sys


def test_resolve_experiment_name_by_short_name(tmp_path):
    from blueprint_validation.training.dreamdojo_finetune import resolve_dreamdojo_experiment_name

    root = tmp_path / "DreamDojo"
    config_file = root / "configs" / "site_adapt.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("name: test\n")

    resolved = resolve_dreamdojo_experiment_name(root, "site_adapt")
    assert resolved == "dreamdojo_site_adapt"


def test_resolve_experiment_name_by_yaml_path(tmp_path):
    from blueprint_validation.training.dreamdojo_finetune import resolve_dreamdojo_experiment_name

    root = tmp_path / "DreamDojo"
    config_file = root / "configs" / "site_adapt.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("name: test\n")

    resolved = resolve_dreamdojo_experiment_name(root, "site_adapt.yaml")
    assert resolved == "dreamdojo_site_adapt"


def test_build_dreamdojo_launch_command(tmp_path):
    from blueprint_validation.config import FinetuneConfig
    from blueprint_validation.training.dreamdojo_finetune import build_dreamdojo_launch_command

    root = tmp_path / "DreamDojo"
    train_script = root / "scripts" / "train.py"
    train_script.parent.mkdir(parents=True)
    train_script.write_text("print('ok')\n")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    cfg = FinetuneConfig(
        dreamdojo_checkpoint=tmp_path / "checkpoints" / "2B_pretrain",
        batch_size=2,
        gradient_accumulation_steps=3,
        learning_rate=1e-4,
        num_epochs=4,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="q_proj,v_proj",
    )

    cmd = build_dreamdojo_launch_command(
        dreamdojo_root=root,
        experiment_name="dreamdojo_site_adapt",
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=cfg,
        facility_id="facility_a",
    )
    text = " ".join(cmd)
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in text
    assert "experiment=dreamdojo_site_adapt" in text
    assert "job.project=blueprint_validation" in text
    assert "job.group=facility_a" in text
    assert f"dataloader_train.dataset.dataset_path={dataset_dir}" in text
    assert f"checkpoint.load_path={cfg.dreamdojo_checkpoint}" in text
    assert "model.config.use_lora=true" in text
    assert 'model.config.lora_target_modules="q_proj,v_proj"' in text


def test_quote_hydra_string():
    from blueprint_validation.training.dreamdojo_finetune import _quote_hydra_string

    assert _quote_hydra_string("q_proj,v_proj") == '"q_proj,v_proj"'
    assert _quote_hydra_string(" q_proj , v_proj ") == '"q_proj , v_proj"'
    assert _quote_hydra_string('a"b') == '"a\\"b"'


def test_resolve_latest_checkpoint_recurses(tmp_path):
    from blueprint_validation.training.dreamdojo_finetune import _resolve_latest_checkpoint

    lora_dir = tmp_path / "lora_weights"
    ckpt = lora_dir / "blueprint_validation" / "facility_a" / "run_1" / "checkpoints" / "iter_000001"
    ckpt.mkdir(parents=True)

    resolved = _resolve_latest_checkpoint(lora_dir)
    assert resolved == ckpt
