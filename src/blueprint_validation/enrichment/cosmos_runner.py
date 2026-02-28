"""Cosmos Transfer 2.5 inference wrapper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..common import get_logger
from ..config import EnrichConfig, VariantSpec

logger = get_logger("enrichment.cosmos_runner")


@dataclass
class CosmosOutput:
    variant_name: str
    prompt: str
    output_video_path: Path
    input_video_path: Path
    depth_video_path: Optional[Path]


def build_controlnet_spec(
    video_path: Path,
    depth_path: Optional[Path],
    prompt: str,
    output_path: Path,
    guidance: float = 7.0,
    controlnet_inputs: List[str] = None,
) -> dict:
    """Build an inference config matching Cosmos Transfer's JSON CLI schema."""
    if controlnet_inputs is None:
        controlnet_inputs = ["rgb", "depth"]

    controls = set(controlnet_inputs)
    spec: dict = {
        "name": output_path.stem,
        "prompt": prompt,
        "video_path": str(video_path),
        "guidance": guidance,
    }

    if "depth" in controls and depth_path and depth_path.exists():
        spec["depth"] = {
            "control_path": str(depth_path),
            "control_weight": 0.6,
        }
    if "edge" in controls:
        spec["edge"] = {"control_weight": 0.5}
    if "seg" in controls:
        spec["seg"] = {"control_weight": 0.5}
    if "vis" in controls:
        spec["vis"] = {"control_weight": 0.5}

    if "rgb" in controls:
        # Transfer already conditions on input RGB via `video_path`.
        logger.debug("Ignoring explicit 'rgb' control input; Cosmos uses video_path directly")

    return spec


def build_cosmos_inference_command(
    spec_path: Path,
    output_dir: Path,
) -> list[str]:
    """Build the Cosmos inference command."""
    cmd = [
        "python",
        "examples/inference.py",
        "-i",
        str(spec_path),
        "-o",
        str(output_dir),
    ]
    return cmd


def resolve_cosmos_repo(cosmos_repo: Path) -> Path:
    """Resolve Cosmos repo path and verify the inference script exists."""
    candidates = [
        cosmos_repo,
        Path(os.environ["COSMOS_ROOT"]) if os.environ.get("COSMOS_ROOT") else None,
        Path("/opt/cosmos-transfer"),
        Path("/opt/cosmos-transfer2.5"),
        Path.home() / "cosmos-transfer2.5",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        inference_script = candidate / "examples" / "inference.py"
        if inference_script.exists():
            return candidate

    checked = ", ".join(str(c) for c in candidates if c)
    raise RuntimeError(
        "Cosmos Transfer inference script not found. Checked: "
        f"{checked}. Set enrich.cosmos_repo to your cosmos-transfer2.5 checkout."
    )


def _resolve_generated_video(expected_path: Path) -> Path:
    if expected_path.exists():
        return expected_path

    output_dir = expected_path.parent
    stem_matches = sorted(
        output_dir.glob(f"{expected_path.stem}*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if stem_matches:
        return stem_matches[0]

    candidates = sorted(
        output_dir.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"Cosmos inference produced no MP4 output in {output_dir}")
    return candidates[0]


def run_cosmos_inference(
    spec: dict,
    expected_output_path: Path,
    cosmos_checkpoint: Path,
    cosmos_model: str = "nvidia/Cosmos-Transfer2.5-2B",
    cosmos_repo: Optional[Path] = None,
) -> Path:
    """Run Cosmos Transfer 2.5 inference and return generated output video path."""
    del (
        cosmos_checkpoint
    )  # checkpoint presence is preflighted; CLI resolves model assets internally.

    expected_output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = resolve_cosmos_repo(cosmos_repo or Path("/opt/cosmos-transfer"))
    if not shutil.which("python"):
        raise RuntimeError("Python interpreter not found in PATH for Cosmos inference")

    # Write spec to temp file in output directory for traceability.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        dir=expected_output_path.parent,
    ) as f:
        json.dump(spec, f, indent=2)
        spec_path = Path(f.name)

    cmd = build_cosmos_inference_command(
        spec_path=spec_path,
        output_dir=expected_output_path.parent,
    )
    logger.info("Running Cosmos inference (%s): %s", cosmos_model, " ".join(cmd))
    env = os.environ.copy()
    repo_root_str = str(repo_root)
    cosmos_pythonpath_entries = [
        repo_root_str,
        str(repo_root / "packages" / "cosmos-cuda"),
        str(repo_root / "packages" / "cosmos-oss"),
    ]
    current_pythonpath = env.get("PYTHONPATH", "")
    merged_entries = cosmos_pythonpath_entries + ([current_pythonpath] if current_pythonpath else [])
    env["PYTHONPATH"] = os.pathsep.join(merged_entries)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
        timeout=1800,
    )

    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-1000:]
        stdout_tail = (result.stdout or "")[-1000:]
        raise RuntimeError(
            "Cosmos inference failed "
            f"(returncode={result.returncode}). stdout_tail={stdout_tail!r} "
            f"stderr_tail={stderr_tail!r}"
        )

    generated = _resolve_generated_video(expected_output_path)
    logger.info("Cosmos inference complete: %s", generated)
    return generated


def enrich_clip(
    video_path: Path,
    depth_path: Optional[Path],
    variants: List[VariantSpec],
    output_dir: Path,
    clip_name: str,
    config: EnrichConfig,
) -> List[CosmosOutput]:
    """Enrich a single rendered clip with multiple visual variants."""
    outputs = []

    for variant in variants:
        expected_video = output_dir / f"{clip_name}_{variant.name}.mp4"
        spec = build_controlnet_spec(
            video_path=video_path,
            depth_path=depth_path,
            prompt=variant.prompt,
            output_path=expected_video,
            guidance=config.guidance,
            controlnet_inputs=config.controlnet_inputs,
        )

        generated_video = run_cosmos_inference(
            spec=spec,
            expected_output_path=expected_video,
            cosmos_checkpoint=config.cosmos_checkpoint,
            cosmos_model=config.cosmos_model,
            cosmos_repo=config.cosmos_repo,
        )
        outputs.append(
            CosmosOutput(
                variant_name=variant.name,
                prompt=variant.prompt,
                output_video_path=generated_video,
                input_video_path=video_path,
                depth_video_path=depth_path,
            )
        )

    return outputs
