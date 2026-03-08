"""Shared video helpers for writing, codec enforcement, and strict frame validation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .common import get_logger

logger = get_logger("video_io")

_DEFAULT_MP4_CODECS: tuple[str, ...] = ("avc1", "H264", "X264", "x264", "mp4v")


@dataclass(frozen=True)
class VideoValidationResult:
    path: Path
    codec_name: str
    decoded_frames: int
    duration_seconds: float | None
    transcoded: bool
    width: int | None = None
    height: int | None = None
    content_monochrome_warning: bool = False
    content_max_std_dev: float | None = None


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


_MIN_CONTENT_STD_DEV: float = 6.0  # frames with global std-dev below this are flagged


def _warn_if_content_monochrome(video_path: Path, *, label: str = "") -> tuple[bool, float | None]:
    """Sample ~3 frames from the video and warn if all appear monochromatic (e.g. all-green).

    OpenCV mp4v on some Linux builds produces solid-colour frames when the
    codec falls back without error.  A valid 3DGS render should always have
    significant pixel variance.  This check does NOT raise — it only logs a
    WARNING so the pipeline continues while making the defect visible.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("content-sanity: could not open %s%s", video_path, f" ({label})" if label else "")
        return False, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_positions = [max(0, total // 4), max(0, total // 2), max(0, 3 * total // 4)]
    std_devs: list[float] = []
    try:
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            # Global std-dev across all channels / pixels
            std_devs.append(float(np.std(frame.astype(np.float32))))
    finally:
        cap.release()

    if not std_devs:
        logger.warning(
            "content-sanity: no frames decoded from %s%s",
            video_path,
            f" ({label})" if label else "",
        )
        return False, None

    max_std = max(std_devs)
    if max_std < _MIN_CONTENT_STD_DEV:
        logger.warning(
            "content-sanity: %s%s appears MONOCHROMATIC — max pixel std-dev=%.2f "
            "(threshold=%.1f). Gemini may be scoring a green/blank video. "
            "Check codec fallback chain and gsplat renderer output.",
            video_path,
            f" ({label})" if label else "",
            max_std,
            _MIN_CONTENT_STD_DEV,
        )
        return True, float(max_std)
    else:
        logger.debug(
            "content-sanity: %s OK max_std=%.2f%s",
            video_path,
            max_std,
            f" ({label})" if label else "",
        )
        return False, float(max_std)


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
            tmp_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{input_path.stem}.",
                    suffix=".h264_tmp",
                    dir=str(input_path.parent),
                )
            )
            tmp_path = tmp_dir / f"{input_path.name}.h264.mp4"
            try:
                transcode_mp4_to_h264(
                    input_path=input_path,
                    output_path=tmp_path,
                    crf=crf,
                    preset=preset,
                )
                os.replace(tmp_path, input_path)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
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

    # Pixel-content sanity: warn if frames are monochromatic (all-green/blank).
    # This catches the OpenCV mp4v silent codec fallback that produces green frames.
    content_warning = False
    content_max_std = None
    try:
        content_warning, content_max_std = _warn_if_content_monochrome(
            resolved_path,
            label=f"transcoded={transcoded}",
        )
    except Exception as _ce:
        logger.debug("content-sanity check error for %s: %s", resolved_path, _ce)

    return VideoValidationResult(
        path=resolved_path,
        codec_name=final_codec,
        decoded_frames=int(decoded_frames),
        width=_parse_optional_int(final_probe.get("width")),
        height=_parse_optional_int(final_probe.get("height")),
        duration_seconds=_parse_optional_float(final_probe.get("duration")),
        transcoded=bool(transcoded),
        content_monochrome_warning=bool(content_warning),
        content_max_std_dev=content_max_std,
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


def _parse_optional_int(raw: Any) -> int | None:
    try:
        if raw in {None, "N/A", ""}:
            return None
        return int(raw)
    except (TypeError, ValueError):
        return None
