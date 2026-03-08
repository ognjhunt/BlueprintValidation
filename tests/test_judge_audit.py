"""Tests for judge-audit CSV export."""

from __future__ import annotations

import csv


def test_write_judge_audit_csv(tmp_path):
    from blueprint_validation.evaluation.judge_audit import write_judge_audit_csv

    rows = [
        {
            "condition": "baseline",
            "rollout_index": 3,
            "task": "Pick up bowl_101 and place it in the target zone",
            "video_path": "/tmp/baseline_003.mp4",
            "start_clip_name": "clip_007_manipulation",
            "start_clip_index": 7,
            "start_path_type": "manipulation",
            "target_label": "bowl",
            "target_instance_id": "101",
            "task_score": 8,
            "visual_score": 7,
            "spatial_score": 8,
            "reasoning": "Looks correct",
        }
    ]
    out = tmp_path / "judge_audit.csv"
    write_judge_audit_csv(rows, out)

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        got = list(reader)

    assert len(got) == 1
    row = got[0]
    assert row["task"] == rows[0]["task"]
    assert row["start_clip_name"] == "clip_007_manipulation"
    assert row["reviewer_agree"] == ""
    assert row["reviewer_disagree"] == ""
    assert row["reviewer_notes"] == ""


def test_write_judge_audit_csv_sanitizes_formula_cells(tmp_path):
    from blueprint_validation.evaluation.judge_audit import write_judge_audit_csv

    rows = [
        {
            "condition": "baseline",
            "rollout_index": 1,
            "task": "=HYPERLINK(\"http://example.com\",\"click\")",
            "video_path": "+video.mp4",
            "start_clip_name": "-clip",
            "start_clip_index": 2,
            "start_path_type": "@manipulation",
            "target_label": "=bowl",
            "target_instance_id": "+101",
            "task_score": 1,
            "visual_score": 1,
            "spatial_score": 1,
            "reasoning": "-SUM(1,1)",
        }
    ]
    out = tmp_path / "judge_audit.csv"
    write_judge_audit_csv(rows, out)

    with open(out, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["task"].startswith("'=")
    assert row["video_path"].startswith("'+")
    assert row["start_clip_name"].startswith("'-")
    assert row["start_path_type"].startswith("'@")
    assert row["target_label"].startswith("'=")
    assert row["target_instance_id"].startswith("'+")
    assert row["reasoning"].startswith("'-")
