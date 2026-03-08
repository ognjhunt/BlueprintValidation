from __future__ import annotations

from pathlib import Path

from blueprint_validation.video_io import ensure_h264_video


def test_ensure_h264_video_replace_source_avoids_predictable_symlink(monkeypatch, tmp_path: Path):
    src = tmp_path / "clip.mp4"
    src.write_text("SOURCE")

    victim = tmp_path / "victim.txt"
    victim.write_text("SAFE")

    predictable_tmp = tmp_path / "clip.__tmp_h264__.mp4"
    predictable_tmp.symlink_to(victim)

    def _probe_video_stream(path: Path):
        codec = "mp4v" if path == src and src.read_text() == "SOURCE" else "h264"
        return {"codec_name": codec, "width": 64, "height": 48, "duration": "0.5"}

    def _decode_video_frame_count(_path: Path):
        return 5

    def _transcode_mp4_to_h264(*, input_path: Path, output_path: Path, **_kwargs):
        assert input_path == src
        output_path.write_text("TRANSCODED")

    monkeypatch.setattr("blueprint_validation.video_io._probe_video_stream", _probe_video_stream)
    monkeypatch.setattr(
        "blueprint_validation.video_io.decode_video_frame_count", _decode_video_frame_count
    )
    monkeypatch.setattr(
        "blueprint_validation.video_io.transcode_mp4_to_h264", _transcode_mp4_to_h264
    )

    result = ensure_h264_video(input_path=src, replace_source=True, min_decoded_frames=1)

    assert result.path == src
    assert src.read_text() == "TRANSCODED"
    assert victim.read_text() == "SAFE"
    assert predictable_tmp.is_symlink()
