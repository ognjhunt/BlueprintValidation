"""Tests for scripts/generate_pilot_config.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def _write_min_capture_run(capture_root: Path) -> None:
    ply = capture_root / "runs" / "run_a" / "export_last_refined.ply"
    ply.parent.mkdir(parents=True, exist_ok=True)
    ply.write_bytes(b"ply\n")


def test_generate_pilot_config_rejects_pi05_without_explicit_eval_ref(tmp_path):
    capture_root = tmp_path / "capture"
    _write_min_capture_run(capture_root)
    output_config = tmp_path / "pilot.yaml"
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_pilot_config.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--capture-pipeline-root",
            str(capture_root),
            "--output-config",
            str(output_config),
            "--policy-adapter",
            "pi05",
        ],
        capture_output=True,
        text=True,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode != 0
    assert "provide --pi05-model-ref and/or --pi05-checkpoint-path" in combined


def test_generate_pilot_config_pi05_with_model_ref_succeeds(tmp_path):
    capture_root = tmp_path / "capture"
    _write_min_capture_run(capture_root)
    output_config = tmp_path / "pilot.yaml"
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_pilot_config.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--capture-pipeline-root",
            str(capture_root),
            "--output-config",
            str(output_config),
            "--policy-adapter",
            "pi05",
            "--pi05-model-ref",
            "openpi/pi05-base",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    raw = yaml.safe_load(output_config.read_text())
    assert raw["policy_adapter"]["name"] == "pi05"
    assert raw["eval_policy"]["model_name"] == "openpi/pi05-base"
    assert "openvla" not in str(raw["eval_policy"]["checkpoint_path"]).lower()
