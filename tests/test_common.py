"""Tests for common utility helpers."""

from __future__ import annotations

from pathlib import Path


def test_write_json_atomic_overwrite_and_tmp_cleanup(tmp_path: Path) -> None:
    from blueprint_validation.common import read_json, write_json

    out = tmp_path / "artifact.json"
    write_json({"v": 1}, out)
    assert read_json(out) == {"v": 1}

    write_json({"v": 2, "name": "updated"}, out)
    assert read_json(out) == {"v": 2, "name": "updated"}

    leftovers = list(tmp_path.glob(f".{out.name}.*.tmp"))
    assert leftovers == []


def test_sanitize_filename_component_with_hash_avoids_collisions() -> None:
    from blueprint_validation.common import (
        sanitize_filename_component,
        sanitize_filename_component_with_hash,
    )

    raw_a = "clip/a"
    raw_b = "clip\\a"
    assert sanitize_filename_component(raw_a) == sanitize_filename_component(raw_b)

    hashed_a = sanitize_filename_component_with_hash(raw_a)
    hashed_b = sanitize_filename_component_with_hash(raw_b)
    assert hashed_a != hashed_b
    assert hashed_a.startswith("clip_a-")
    assert hashed_b.startswith("clip_a-")
