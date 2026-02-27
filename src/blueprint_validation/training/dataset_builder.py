"""Build DreamDojo-compatible training datasets from enriched videos."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

from ..common import get_logger, read_json, write_json

logger = get_logger("training.dataset_builder")


def build_dreamdojo_dataset(
    enriched_manifest_path: Path,
    output_dir: Path,
    facility_name: str,
) -> Path:
    """Convert enriched video manifest into DreamDojo training dataset format.

    DreamDojo expects a dataset directory with:
    - videos/ directory containing .mp4 files
    - metadata.json with video paths and descriptions
    - dataset_info.json with dataset-level metadata

    Returns the path to the dataset directory.
    """
    manifest = read_json(enriched_manifest_path)
    dataset_dir = output_dir / "dreamdojo_dataset"
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    video_entries: List[Dict] = []

    for entry in manifest.get("clips", []):
        src = Path(entry["output_video_path"])
        if not src.exists():
            logger.warning("Enriched video not found: %s", src)
            continue

        # Copy video to dataset directory
        dst = videos_dir / src.name
        shutil.copy2(src, dst)

        video_entries.append({
            "video_path": f"videos/{src.name}",
            "prompt": entry.get("prompt", ""),
            "variant": entry.get("variant_name", ""),
            "source_clip": entry.get("clip_name", ""),
            "facility": facility_name,
        })

    # Write metadata
    metadata = {
        "dataset_name": f"blueprint_{facility_name}",
        "num_videos": len(video_entries),
        "facility": facility_name,
        "videos": video_entries,
    }
    write_json(metadata, dataset_dir / "metadata.json")

    # Write dataset info
    dataset_info = {
        "name": f"blueprint_{facility_name}",
        "description": f"Site-adapted training data for {facility_name}",
        "num_videos": len(video_entries),
        "source": "BlueprintValidation pipeline",
    }
    write_json(dataset_info, dataset_dir / "dataset_info.json")

    logger.info("Built dataset with %d videos at %s", len(video_entries), dataset_dir)
    return dataset_dir
