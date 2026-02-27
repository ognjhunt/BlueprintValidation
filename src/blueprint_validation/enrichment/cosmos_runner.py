"""Cosmos Transfer 2.5 inference wrapper."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..common import get_logger, write_json
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
    """Build a Cosmos Transfer 2.5 controlnet spec JSON."""
    if controlnet_inputs is None:
        controlnet_inputs = ["rgb", "depth"]

    spec = {
        "video_path": str(video_path),
        "prompt_path": None,  # We'll write prompt to file
        "prompt": prompt,
        "output_path": str(output_path),
        "guidance": guidance,
        "controlnet_specs": [],
    }

    if "rgb" in controlnet_inputs:
        spec["controlnet_specs"].append({
            "control_path": str(video_path),
            "control_type": "rgb",
            "control_weight": 0.8,
        })

    if "depth" in controlnet_inputs and depth_path and depth_path.exists():
        spec["controlnet_specs"].append({
            "control_path": str(depth_path),
            "control_type": "depth",
            "control_weight": 0.6,
        })

    return spec


def run_cosmos_inference(
    spec: dict,
    cosmos_checkpoint: Path,
    cosmos_model: str = "nvidia/Cosmos-Transfer2.5-2B",
    num_gpus: int = 1,
) -> Path:
    """Run Cosmos Transfer 2.5 inference via CLI.

    Returns the path to the generated output video.
    """
    output_path = Path(spec["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write spec to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=output_path.parent
    ) as f:
        json.dump(spec, f, indent=2)
        spec_path = f.name

    # Write prompt to file if needed
    prompt_path = output_path.parent / f"{output_path.stem}_prompt.txt"
    prompt_path.write_text(spec.get("prompt", ""))
    spec["prompt_path"] = str(prompt_path)

    # Re-write spec with prompt_path
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    # Build command
    cosmos_root = cosmos_checkpoint.parent.parent
    inference_script = cosmos_root / "examples" / "inference.py"

    if not inference_script.exists():
        # Try finding in standard install locations
        for candidate in [
            Path("/opt/cosmos-transfer/examples/inference.py"),
            Path.home() / "cosmos-transfer2.5" / "examples" / "inference.py",
        ]:
            if candidate.exists():
                inference_script = candidate
                break

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--master_port=12341",
            "-m", "examples.inference",
            "--params_file", spec_path,
            f"--num_gpus={num_gpus}",
        ]
    else:
        cmd = [
            "python",
            str(inference_script),
            "--params_file", spec_path,
        ]

    logger.info("Running Cosmos inference: %s", " ".join(cmd))
    logger.info("Output: %s", output_path)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cosmos_root) if cosmos_root.exists() else None,
        timeout=1800,  # 30 min timeout
    )

    if result.returncode != 0:
        logger.error("Cosmos inference failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
        raise RuntimeError(f"Cosmos inference failed: {result.stderr[-500:]}")

    logger.info("Cosmos inference complete: %s", output_path)
    return output_path


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
        output_video = output_dir / f"{clip_name}_{variant.name}.mp4"

        spec = build_controlnet_spec(
            video_path=video_path,
            depth_path=depth_path,
            prompt=variant.prompt,
            output_path=output_video,
            guidance=config.guidance,
            controlnet_inputs=config.controlnet_inputs,
        )

        try:
            run_cosmos_inference(
                spec=spec,
                cosmos_checkpoint=config.cosmos_checkpoint,
                cosmos_model=config.cosmos_model,
            )
            outputs.append(CosmosOutput(
                variant_name=variant.name,
                prompt=variant.prompt,
                output_video_path=output_video,
                input_video_path=video_path,
                depth_video_path=depth_path,
            ))
        except Exception as e:
            logger.error("Failed to enrich %s with variant %s: %s", clip_name, variant.name, e)

    return outputs
