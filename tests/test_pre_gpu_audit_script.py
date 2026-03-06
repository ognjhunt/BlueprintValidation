from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

if os.environ.get("BLUEPRINT_IN_PRE_GPU_AUDIT") == "1":
    pytest.skip("Skipped inside nested pre_gpu_audit.sh pytest run.", allow_module_level=True)


@pytest.mark.parametrize("scope", ["quick", "full"])
def test_pre_gpu_audit_script_supports_quick_and_full_scopes(tmp_path: Path, scope: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_dir = tmp_path / "fixture"

    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "BLUEPRINT_IN_PRE_GPU_AUDIT": "1",
        }
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_cpu_audit_fixture.py"),
            "--output-dir",
            str(fixture_dir),
            "--facility-id",
            "ci_facility",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "LOCAL_CONFIG": str(fixture_dir / "config.yaml"),
            "CLOUD_CONFIG": str(fixture_dir / "config.yaml"),
            "WORK_DIR": str(fixture_dir / "work"),
            "RUN_LOCAL_PREFLIGHT": "false",
            "RUN_CLOUD_PREFLIGHT": "false",
            "RUN_SECRET_SCAN": "false",
            "RUN_LINT": "false",
            "RUN_FORMAT_CHECK": "false",
            "AUDIT_SCOPE": scope,
        }
    )

    result = subprocess.run(
        ["bash", "scripts/pre_gpu_audit.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert f"=== Pytest ({scope}) ===" in result.stdout


def test_pre_gpu_audit_script_runs_local_preflight_on_cpu_fixture(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_dir = tmp_path / "fixture"

    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "BLUEPRINT_IN_PRE_GPU_AUDIT": "1",
        }
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_cpu_audit_fixture.py"),
            "--output-dir",
            str(fixture_dir),
            "--facility-id",
            "ci_facility",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "LOCAL_CONFIG": str(fixture_dir / "config.yaml"),
            "CLOUD_CONFIG": str(fixture_dir / "config.yaml"),
            "WORK_DIR": str(fixture_dir / "work"),
            "RUN_CLOUD_PREFLIGHT": "false",
            "RUN_SECRET_SCAN": "false",
            "RUN_TARGETED_PYTEST": "false",
            "RUN_LINT": "false",
            "RUN_FORMAT_CHECK": "false",
            "AUDIT_SCOPE": "quick",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/pre_gpu_audit.sh"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "[PASS] Local preflight (--profile audit)" in result.stdout
