"""Strict manifest schema checks used at stage boundaries."""

from __future__ import annotations

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
        for idx, clip in enumerate(clips):
            _require_nonempty_text(
                clip,
                "clip_name",
                manifest_type=manifest_type,
                clip_index=idx,
                manifest_path=manifest_path,
            )
            source_video = str(clip.get("video_path") or clip.get("output_video_path") or "").strip()
            if not source_video:
                where = f" ({manifest_path})" if manifest_path is not None else ""
                raise ManifestValidationError(
                    f"{manifest_type} manifest clip[{idx}] missing 'video_path'/'output_video_path'{where}"
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
