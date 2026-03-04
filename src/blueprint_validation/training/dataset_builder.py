"""Build DreamDojo-compatible training datasets from enriched videos."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path
from typing import List

from ..common import get_logger, read_json, write_json
from ..video_io import ensure_h264_video

logger = get_logger("training.dataset_builder")


def _sanitize_name_component(value: object, *, fallback: str) -> str:
    text = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return cleaned or fallback


def _derive_dataset_stem(entry: dict, src: Path, index: int) -> str:
    clip_component = _sanitize_name_component(entry.get("clip_name"), fallback=src.stem or "clip")
    variant_component = _sanitize_name_component(entry.get("variant_name"), fallback="variant")
    try:
        source_fingerprint_input = str(src.resolve(strict=False))
    except Exception:
        source_fingerprint_input = str(src)
    source_hash = hashlib.sha1(source_fingerprint_input.encode("utf-8")).hexdigest()[:10]
    return f"{clip_component}__{variant_component}__{index:05d}__{source_hash}"


def build_dreamdojo_dataset(
    enriched_manifest_path: Path,
    output_dir: Path,
    facility_name: str,
    min_decoded_frames: int = 13,
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

    for idx, entry in enumerate(manifest.get("clips", [])):
        src = Path(entry["output_video_path"])
        if not src.exists():
            logger.warning("Enriched video not found: %s", src)
            continue

        # Collision-proof naming keeps prompt/video pairings stable even when source basenames collide.
        stem = _derive_dataset_stem(entry, src, idx)
        video_suffix = src.suffix or ".mp4"
        dst = videos_dir / f"{stem}{video_suffix}"
        dedupe_counter = 1
        while dst.exists():
            dst = videos_dir / f"{stem}__dup{dedupe_counter}{video_suffix}"
            dedupe_counter += 1

        # Copy video to dataset directory
        shutil.copy2(src, dst)
        checked = ensure_h264_video(
            input_path=dst,
            min_decoded_frames=max(1, int(min_decoded_frames)),
            replace_source=True,
        )
        if checked.path != dst:
            dst = checked.path
        if bool(checked.content_monochrome_warning):
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed cleaning rejected monochrome clip: %s", dst, exc_info=True)
            max_std = checked.content_max_std_dev
            raise RuntimeError(
                "Strict monochrome-content check failed for dataset clip "
                f"{dst} (source={src}, max_std_dev={max_std})."
            )

        # Write corresponding meta text file (prompt)
        prompt = entry.get("prompt", "")
        meta_name = dst.stem + ".txt"
        meta_path = metas_dir / meta_name
        meta_dedupe_counter = 1
        while meta_path.exists():
            meta_name = f"{dst.stem}__dup{meta_dedupe_counter}.txt"
            meta_path = metas_dir / meta_name
            meta_dedupe_counter += 1
        meta_path.write_text(prompt, encoding="utf-8")

        csv_rows.append(
            {
                "video_path": f"videos/{dst.name}",
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
