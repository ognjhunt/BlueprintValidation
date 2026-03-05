"""Strict manifest schema checks used at stage boundaries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..common import get_logger, read_json

logger = get_logger("validation.manifest_validation")


class ManifestValidationError(RuntimeError):
    """Raised when a manifest violates a strict schema contract."""


def _as_mapping(payload: Any, *, manifest_type: str, manifest_path: Path | None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        where = f" ({manifest_path})" if manifest_path is not None else ""
        raise ManifestValidationError(
            f"{manifest_type} manifest must be a JSON object{where}; got {type(payload).__name__}"
        )
    return payload


def _coerce_clips(payload: Dict[str, Any], *, manifest_type: str, manifest_path: Path | None) -> List[Dict]:
    clips = payload.get("clips")
    if not isinstance(clips, list):
        where = f" ({manifest_path})" if manifest_path is not None else ""
        raise ManifestValidationError(f"{manifest_type} manifest missing list field 'clips'{where}")
    out: List[Dict] = []
    for idx, clip in enumerate(clips):
        if not isinstance(clip, dict):
            where = f" ({manifest_path})" if manifest_path is not None else ""
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{idx}] must be object{where}; got {type(clip).__name__}"
            )
        out.append(clip)
    return out


def _require_nonempty_text(
    clip: Dict[str, Any],
    key: str,
    *,
    manifest_type: str,
    clip_index: int,
    manifest_path: Path | None,
) -> str:
    value = str(clip.get(key, "")).strip()
    if not value:
        where = f" ({manifest_path})" if manifest_path is not None else ""
        raise ManifestValidationError(
            f"{manifest_type} manifest clip[{clip_index}] missing non-empty '{key}'{where}"
        )
    return value


def _validate_paths_exist(
    *,
    path_values: Iterable[tuple[str, str]],
    manifest_type: str,
    clip_index: int,
    manifest_path: Path | None,
) -> None:
    for key, raw in path_values:
        value = str(raw or "").strip()
        if not value:
            continue
        if not Path(value).exists():
            where = f" ({manifest_path})" if manifest_path is not None else ""
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] path '{key}' does not exist: {value}{where}"
            )


def _validate_optional_numeric_fields(
    clip: Dict[str, Any],
    *,
    manifest_type: str,
    clip_index: int,
    manifest_path: Path | None,
) -> None:
    """Validate optional numeric metadata fields for stage-1 style clip rows."""
    where = f" ({manifest_path})" if manifest_path is not None else ""

    fps_val = clip.get("fps")
    if fps_val is not None:
        try:
            fps = float(fps_val)
        except Exception as exc:  # pragma: no cover - defensive
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] invalid fps={fps_val!r}{where}"
            ) from exc
        if not math.isfinite(fps) or fps <= 0.0:
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] fps must be > 0; got {fps_val!r}{where}"
            )

    num_frames_val = clip.get("num_frames")
    if num_frames_val is not None:
        try:
            num_frames = int(num_frames_val)
        except Exception as exc:  # pragma: no cover - defensive
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] invalid num_frames={num_frames_val!r}{where}"
            ) from exc
        if num_frames < 1:
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] num_frames must be >= 1; "
                f"got {num_frames}{where}"
            )

    resolution_val = clip.get("resolution")
    if resolution_val is not None:
        if not isinstance(resolution_val, (list, tuple)) or len(resolution_val) != 2:
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] resolution must be [H, W]{where}"
            )
        try:
            height = int(resolution_val[0])
            width = int(resolution_val[1])
        except Exception as exc:  # pragma: no cover - defensive
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] resolution values must be ints; "
                f"got {resolution_val!r}{where}"
            ) from exc
        if not (1 <= height <= 16384 and 1 <= width <= 16384):
            raise ManifestValidationError(
                f"{manifest_type} manifest clip[{clip_index}] resolution out of bounds: "
                f"{resolution_val!r}{where}"
            )


def validate_manifest_schema(
    payload: Any,
    *,
    manifest_type: str,
    manifest_path: Path | None = None,
    require_existing_paths: bool = True,
) -> Dict[str, Any]:
    """Validate a known manifest schema and return the mapping if valid."""
    data = _as_mapping(payload, manifest_type=manifest_type, manifest_path=manifest_path)

    if manifest_type in {"stage1_source", "render", "source"}:
        clips = _coerce_clips(data, manifest_type=manifest_type, manifest_path=manifest_path)
        declared_num_clips = data.get("num_clips")
        if declared_num_clips is not None:
            where = f" ({manifest_path})" if manifest_path is not None else ""
            try:
                declared_num_clips_int = int(declared_num_clips)
            except Exception as exc:  # pragma: no cover - defensive
                raise ManifestValidationError(
                    f"{manifest_type} manifest field 'num_clips' must be int{where}; "
                    f"got {declared_num_clips!r}"
                ) from exc
            if declared_num_clips_int != len(clips):
                raise ManifestValidationError(
                    f"{manifest_type} manifest num_clips mismatch: "
                    f"{declared_num_clips_int} != len(clips)={len(clips)}{where}"
                )
        seen_clip_names: set[str] = set()
        for idx, clip in enumerate(clips):
            clip_name = _require_nonempty_text(
                clip,
                "clip_name",
                manifest_type=manifest_type,
                clip_index=idx,
                manifest_path=manifest_path,
            )
            if clip_name in seen_clip_names:
                where = f" ({manifest_path})" if manifest_path is not None else ""
                raise ManifestValidationError(
                    f"{manifest_type} manifest has duplicate clip_name '{clip_name}'{where}"
                )
            seen_clip_names.add(clip_name)
            source_video = str(clip.get("video_path") or clip.get("output_video_path") or "").strip()
            if not source_video:
                where = f" ({manifest_path})" if manifest_path is not None else ""
                raise ManifestValidationError(
                    f"{manifest_type} manifest clip[{idx}] missing 'video_path'/'output_video_path'{where}"
                )
            _validate_optional_numeric_fields(
                clip,
                manifest_type=manifest_type,
                clip_index=idx,
                manifest_path=manifest_path,
            )
            if require_existing_paths:
                _validate_paths_exist(
                    path_values=[
                        ("video_path", str(clip.get("video_path", "") or source_video)),
                        ("depth_video_path", str(clip.get("depth_video_path", "") or "")),
                        ("camera_path", str(clip.get("camera_path", "") or "")),
                    ],
                    manifest_type=manifest_type,
                    clip_index=idx,
                    manifest_path=manifest_path,
                )
        return data

    if manifest_type == "enriched":
        clips = _coerce_clips(data, manifest_type=manifest_type, manifest_path=manifest_path)
        for idx, clip in enumerate(clips):
            _require_nonempty_text(
                clip,
                "clip_name",
                manifest_type=manifest_type,
                clip_index=idx,
                manifest_path=manifest_path,
            )
            _require_nonempty_text(
                clip,
                "output_video_path",
                manifest_type=manifest_type,
                clip_index=idx,
                manifest_path=manifest_path,
            )
            if require_existing_paths:
                _validate_paths_exist(
                    path_values=[
                        ("output_video_path", str(clip.get("output_video_path", "") or "")),
                        ("input_video_path", str(clip.get("input_video_path", "") or "")),
                    ],
                    manifest_type=manifest_type,
                    clip_index=idx,
                    manifest_path=manifest_path,
                )
        return data

    if manifest_type == "policy_scores":
        scores = data.get("scores")
        if not isinstance(scores, list):
            where = f" ({manifest_path})" if manifest_path is not None else ""
            raise ManifestValidationError(f"policy_scores manifest missing list field 'scores'{where}")
        for idx, row in enumerate(scores):
            if not isinstance(row, dict):
                where = f" ({manifest_path})" if manifest_path is not None else ""
                raise ManifestValidationError(
                    f"policy_scores manifest row[{idx}] must be object{where}; got {type(row).__name__}"
                )
            _require_nonempty_text(
                row,
                "video_path",
                manifest_type=manifest_type,
                clip_index=idx,
                manifest_path=manifest_path,
            )
            if require_existing_paths:
                _validate_paths_exist(
                    path_values=[("video_path", str(row.get("video_path", "") or ""))],
                    manifest_type=manifest_type,
                    clip_index=idx,
                    manifest_path=manifest_path,
                )
        return data

    raise ManifestValidationError(f"Unsupported manifest_type '{manifest_type}'")


def load_and_validate_manifest(
    path: Path,
    *,
    manifest_type: str,
    require_existing_paths: bool = True,
) -> Dict[str, Any]:
    """Load JSON from disk and apply strict schema validation."""
    payload = read_json(path)
    data = validate_manifest_schema(
        payload,
        manifest_type=manifest_type,
        manifest_path=path,
        require_existing_paths=bool(require_existing_paths),
    )
    logger.debug(
        "Validated %s manifest at %s (clips=%d)",
        manifest_type,
        path,
        len(data.get("clips", [])) if isinstance(data.get("clips"), list) else -1,
    )
    return data
