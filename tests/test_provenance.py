"""Tests for provenance stamp helpers."""

from __future__ import annotations

from pathlib import Path

from blueprint_validation.provenance import build_provenance_stamp, canonical_json_hash, file_sha256


def test_canonical_json_hash_is_order_stable() -> None:
    left = {"b": 2, "a": 1, "nested": {"z": 3, "y": [2, 1]}}
    right = {"nested": {"y": [2, 1], "z": 3}, "a": 1, "b": 2}
    assert canonical_json_hash(left) == canonical_json_hash(right)


def test_build_provenance_stamp_includes_config_and_file_hashes(tmp_path: Path) -> None:
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.csv"
    input_path.write_text('{"ok": true}', encoding="utf-8")
    output_path.write_text("id,value\n1,2\n", encoding="utf-8")

    stamp = build_provenance_stamp(
        stage="unit-test",
        config_obj={"alpha": 1, "beta": {"x": True}},
        input_paths=[input_path],
        output_paths=[output_path],
        extra={"dataset_tag": "facility:s3"},
    )

    assert stamp["stage"] == "unit-test"
    assert stamp["config_hash"] == canonical_json_hash({"alpha": 1, "beta": {"x": True}})
    assert stamp["input_hashes"][str(input_path)] == file_sha256(input_path)
    assert stamp["output_hashes"][str(output_path)] == file_sha256(output_path)
    assert isinstance(stamp["stamp_hash"], str) and len(stamp["stamp_hash"]) == 64
