"""Unit tests for RoboSplat scan augmentation helpers."""

from blueprint_validation.augmentation.robosplat_scan import _sanitize_output_clip_name


def test_sanitize_output_clip_name_blocks_path_traversal() -> None:
    assert _sanitize_output_clip_name("../escape/evil") == "evil"
    assert _sanitize_output_clip_name("/tmp/pwn") == "pwn"
    assert _sanitize_output_clip_name("..\\escape\\evil") == "escape_evil"
    assert _sanitize_output_clip_name("") == "clip"
