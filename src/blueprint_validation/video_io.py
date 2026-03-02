"""Shared video helpers for writing, codec enforcement, and strict frame validation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .common import get_logger

logger = get_logger("video_io")

_DEFAULT_MP4_CODECS: tuple[str, ...] = ("avc1", "H264", "mp4v")


@dataclass(frozen=True)
class VideoValidationResult:
    path: Path
    codec_name: str
    decoded_frames: int
    duration_seconds: float | None
    transcoded: bool


def open_mp4_writer(
    *,
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    is_color: bool = True,
    codec_candidates: Sequence[str] = _DEFAULT_MP4_CODECS,
):
    """Open an MP4 writer with a codec fallback chain for broad player compatibility."""
    import cv2

    width = int(frame_size[0])
    height = int(frame_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size for video writer: {(width, height)}")

    resolved_fps = float(fps) if fps and fps > 0 else 10.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tried: list[str] = []
    for codec in codec_candidates:
        code = str(codec or "").strip()
        if len(code) != 4:
            continue
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*code),
            resolved_fps,
            (width, height),
            isColor=bool(is_color),
        )
        if writer.isOpened():
            if code != _DEFAULT_MP4_CODECS[0]:
                logger.warning(
                    "Using fallback MP4 codec '%s' for %s (attempted: %s)",
                    code,
                    output_path,
                    ",".join(tried + [code]),
                )
            return writer
        writer.release()
        tried.append(code)

    raise RuntimeError(
        f"Could not open MP4 video writer for {output_path}; tried codecs={','.join(tried)}"
    )


def decode_video_frame_count(video_path: Path) -> int:
    """Decode frames with OpenCV and return the exact decoded count."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            count += 1
    finally:
        cap.release()
    return int(count)


def transcode_mp4_to_h264(
    *,
    input_path: Path,
    output_path: Path,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Transcode an MP4 file to H.264 (libx264/yuv420p) via ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH; required for strict H.264 enforcement")
    if not input_path.exists():
        raise RuntimeError(f"Input video not found for transcode: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-1000:]
        stdout_tail = (result.stdout or "")[-1000:]
        raise RuntimeError(
            f"ffmpeg transcode failed for {input_path} -> {output_path} "
            f"(returncode={result.returncode}, stdout_tail={stdout_tail!r}, stderr_tail={stderr_tail!r})"
        )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced empty output: {output_path}")


def ensure_h264_video(
    *,
    input_path: Path,
    min_decoded_frames: int = 1,
    output_path: Path | None = None,
    replace_source: bool = False,
    crf: int = 18,
    preset: str = "medium",
) -> VideoValidationResult:
    """Ensure a video is H.264 and decodes to at least ``min_decoded_frames``.

    Behavior:
    - If already H.264, reuses the source file path.
    - If not H.264, transcodes to H.264:
      - in-place when ``replace_source=True``
      - otherwise writes to ``output_path`` (or ``<stem>_h264.mp4`` beside source)
    - Always validates resulting codec and exact decoded frame count.
    """
    if int(min_decoded_frames) < 1:
        raise ValueError("min_decoded_frames must be >= 1")
    if output_path is not None and replace_source:
        raise ValueError("output_path and replace_source are mutually exclusive")
    if not input_path.exists():
        raise RuntimeError(f"Video not found: {input_path}")

    initial_probe = _probe_video_stream(input_path)
    initial_codec = str(initial_probe.get("codec_name", "") or "").strip().lower()

    resolved_path = input_path
    transcoded = False
    if initial_codec != "h264":
        transcoded = True
        if replace_source:
            tmp_path = input_path.with_name(f"{input_path.stem}.__tmp_h264__.mp4")
            transcode_mp4_to_h264(
                input_path=input_path,
                output_path=tmp_path,
                crf=crf,
                preset=preset,
            )
            os.replace(tmp_path, input_path)
            resolved_path = input_path
        else:
            resolved_path = output_path or input_path.with_name(f"{input_path.stem}_h264.mp4")
            transcode_mp4_to_h264(
                input_path=input_path,
                output_path=resolved_path,
                crf=crf,
                preset=preset,
            )

    final_probe = _probe_video_stream(resolved_path)
    final_codec = str(final_probe.get("codec_name", "") or "").strip().lower()
    if final_codec != "h264":
        raise RuntimeError(
            f"Strict codec enforcement failed for {resolved_path}: codec={final_codec or 'unknown'} (expected h264)"
        )

    decoded_frames = decode_video_frame_count(resolved_path)
    if decoded_frames < int(min_decoded_frames):
        raise RuntimeError(
            f"Strict frame-count check failed for {resolved_path}: "
            f"decoded_frames={decoded_frames}, required_min_frames={int(min_decoded_frames)}"
        )

    return VideoValidationResult(
        path=resolved_path,
        codec_name=final_codec,
        decoded_frames=int(decoded_frames),
        duration_seconds=_parse_optional_float(final_probe.get("duration")),
        transcoded=bool(transcoded),
    )


def _probe_video_stream(video_path: Path) -> dict[str, Any]:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        raise RuntimeError("ffprobe not found in PATH; required for strict H.264 enforcement")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,nb_frames,duration,width,height,pix_fmt",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-1000:]
        stdout_tail = (result.stdout or "")[-1000:]
        raise RuntimeError(
            f"ffprobe failed for {video_path} (returncode={result.returncode}, "
            f"stdout_tail={stdout_tail!r}, stderr_tail={stderr_tail!r})"
        )

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe produced invalid JSON for {video_path}") from exc
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"ffprobe found no video stream for {video_path}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError(f"ffprobe returned malformed stream payload for {video_path}")
    return stream


def _parse_optional_float(raw: Any) -> float | None:
    try:
        if raw in {None, "N/A", ""}:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None
