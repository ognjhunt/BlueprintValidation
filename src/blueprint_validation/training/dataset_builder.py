"""Build DreamDojo-compatible training datasets from enriched videos."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import List

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
    - metas/ directory containing .txt files (one per video, with the text prompt)
    - metadata.csv with columns: video_path, prompt

    Returns the path to the dataset directory.
    """
    manifest = read_json(enriched_manifest_path)
    dataset_dir = output_dir / "dreamdojo_dataset"
    videos_dir = dataset_dir / "videos"
    metas_dir = dataset_dir / "metas"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metas_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[dict] = []

    for entry in manifest.get("clips", []):
        src = Path(entry["output_video_path"])
        if not src.exists():
            logger.warning("Enriched video not found: %s", src)
            continue

        # Copy video to dataset directory
        dst = videos_dir / src.name
        shutil.copy2(src, dst)

        # Write corresponding meta text file (prompt)
        prompt = entry.get("prompt", "")
        meta_name = src.stem + ".txt"
        meta_path = metas_dir / meta_name
        meta_path.write_text(prompt)

        csv_rows.append(
            {
                "video_path": f"videos/{src.name}",
                "meta_path": f"metas/{meta_name}",
                "prompt": prompt,
                "variant": entry.get("variant_name", ""),
                "source_clip": entry.get("clip_name", ""),
                "facility": facility_name,
            }
        )

    # Write metadata.csv
    csv_path = dataset_dir / "metadata.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # Write dataset info JSON for our own tracking
    dataset_info = {
        "name": f"blueprint_{facility_name}",
        "description": f"Site-adapted training data for {facility_name}",
        "num_videos": len(csv_rows),
        "source": "BlueprintValidation pipeline",
    }
    write_json(dataset_info, dataset_dir / "dataset_info.json")

    logger.info("Built dataset with %d videos at %s", len(csv_rows), dataset_dir)
    return dataset_dir
