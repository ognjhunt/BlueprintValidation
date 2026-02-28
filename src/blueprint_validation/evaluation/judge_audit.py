"""Helpers for exporting human-review audit sheets for VLM judge outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def write_judge_audit_csv(rows: Iterable[dict], output_path: Path) -> None:
    """Write a compact CSV for manual review of judge correctness.

    Each row contains rollout/task context and judge outputs, plus blank
    reviewer columns:
      - reviewer_agree
      - reviewer_disagree
      - reviewer_notes
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "rollout_index",
        "task",
        "video_path",
        "start_clip_name",
        "start_clip_index",
        "start_path_type",
        "target_label",
        "target_instance_id",
        "task_score",
        "visual_score",
        "spatial_score",
        "reasoning",
        "reviewer_agree",
        "reviewer_disagree",
        "reviewer_notes",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "condition": str(item.get("condition", "")),
                    "rollout_index": item.get("rollout_index", ""),
                    "task": str(item.get("task", "")),
                    "video_path": str(item.get("video_path", "")),
                    "start_clip_name": str(item.get("start_clip_name", "")),
                    "start_clip_index": item.get("start_clip_index", ""),
                    "start_path_type": str(item.get("start_path_type", "")),
                    "target_label": str(item.get("target_label", "")),
                    "target_instance_id": str(item.get("target_instance_id", "")),
                    "task_score": item.get("task_score", ""),
                    "visual_score": item.get("visual_score", ""),
                    "spatial_score": item.get("spatial_score", ""),
                    "reasoning": str(item.get("reasoning", "")),
                    "reviewer_agree": "",
                    "reviewer_disagree": "",
                    "reviewer_notes": "",
                }
            )
