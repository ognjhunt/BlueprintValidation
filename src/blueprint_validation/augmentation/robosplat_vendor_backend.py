"""Vendor-backed RoboSplat integration wrapper (pinned repo strategy)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

from ..common import get_logger, write_json
from ..config import FacilityConfig, ValidationConfig

logger = get_logger("augmentation.robosplat_vendor_backend")


def vendor_backend_available(config: ValidationConfig) -> tuple[bool, str]:
    repo = config.robosplat.vendor_repo_path
    if not repo.exists():
        return False, f"vendor_repo_missing:{repo}"
    if not repo.is_dir():
        return False, f"vendor_repo_not_dir:{repo}"
    return True, "ok"


def run_vendor_backend(
    config: ValidationConfig,
    facility: FacilityConfig,
    stage_dir: Path,
    source_manifest_path: Path,
    output_manifest_path: Path,
) -> Dict:
    """Run vendor RoboSplat script when a known entrypoint exists.

    This wrapper is intentionally conservative: if no stable entrypoint is found,
    we report unavailability and let hybrid mode fall back to native/legacy paths.
    """
    del facility
    repo = config.robosplat.vendor_repo_path
    candidates: List[Path] = [
        repo / "scripts" / "augment.py",
        repo / "tools" / "augment.py",
    ]
    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        return {
            "status": "unavailable",
            "reason": "vendor_entrypoint_not_found",
            "backend_used": "vendor",
            "generated": [],
        }

    # Stable JSON handoff to keep the wrapper resilient to minor CLI changes.
    handoff_path = stage_dir / "vendor_handoff.json"
    write_json(
        {
            "source_manifest_path": str(source_manifest_path),
            "output_manifest_path": str(output_manifest_path),
            "variants_per_input": int(config.robosplat.variants_per_input),
            "parity_mode": config.robosplat.parity_mode,
        },
        handoff_path,
    )

    cmd = [
        "python",
        str(script),
        "--handoff_json",
        str(handoff_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("Vendor RoboSplat command failed: %s", proc.stderr[-4000:])
        return {
            "status": "failed",
            "reason": "vendor_command_failed",
            "backend_used": "vendor",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "generated": [],
        }

    if not output_manifest_path.exists():
        return {
            "status": "failed",
            "reason": "vendor_manifest_missing",
            "backend_used": "vendor",
            "generated": [],
        }

    return {
        "status": "success",
        "reason": "ok",
        "backend_used": "vendor",
        "generated_manifest_path": str(output_manifest_path),
        "generated": [],
    }

