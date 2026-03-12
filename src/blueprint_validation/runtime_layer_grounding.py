"""Runtime-layer grounding helpers for the NeoVerse runtime service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


PROTECTED_OBSERVED_THRESHOLD = 0.85
PROTECTED_RECONSTRUCTED_THRESHOLD = 0.80
EDITABLE_LOW_CONFIDENCE_THRESHOLD = 0.65
TASK_CRITICAL_OVERRIDE_THRESHOLD = 0.70
TASK_CRITICAL_DILATION_PX = 3
DEGRADED_EDITABLE_RATIO_THRESHOLD = 0.40
LOCK_VIOLATION_RETRY_BUDGET = 1


def normalized_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for debug mask outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def _save_rgb(path: Path, frame: np.ndarray) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for debug frame outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _presentation_policy_paths(spec: Mapping[str, Any]) -> Dict[str, Path]:
    policy = dict(spec.get("runtime_layer_policy") or {}) if isinstance(spec.get("runtime_layer_policy"), Mapping) else {}
    out: Dict[str, Path] = {}
    for key in (
        "protected_regions_manifest_path",
        "canonical_render_policy_path",
        "presentation_variance_policy_path",
    ):
        value = str(policy.get(key) or "").strip()
        if value:
            out[key] = Path(value).resolve()
    return out


def validate_runtime_layer_spec(spec: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(spec.get("canonical_package_version") or "").strip():
        errors.append("missing_canonical_package_version")
    policy = dict(spec.get("runtime_layer_policy") or {}) if isinstance(spec.get("runtime_layer_policy"), Mapping) else {}
    for key in (
        "protected_regions_manifest_uri",
        "canonical_render_policy_uri",
        "presentation_variance_policy_uri",
        "protected_regions_manifest_path",
        "canonical_render_policy_path",
        "presentation_variance_policy_path",
    ):
        if not str(policy.get(key) or "").strip():
            errors.append(f"missing_runtime_layer_policy:{key}")
    for path in _presentation_policy_paths(spec).values():
        if not path.is_file():
            errors.append(f"missing_runtime_layer_policy_file:{path.name}")
    return errors


def load_runtime_layer_bundle(spec: Mapping[str, Any]) -> Dict[str, Any]:
    paths = _presentation_policy_paths(spec)
    return {
        "protected_regions_manifest": _read_json(paths["protected_regions_manifest_path"]),
        "canonical_render_policy": _read_json(paths["canonical_render_policy_path"]),
        "presentation_variance_policy": _read_json(paths["presentation_variance_policy_path"]),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def compute_canonical_package_version(
    *,
    scene_memory_manifest: Mapping[str, Any],
    conditioning_bundle: Mapping[str, Any],
    object_geometry_manifest: Mapping[str, Any],
    task_anchor_manifest: Mapping[str, Any],
    site_world_spec: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    canonical_render_policy: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
) -> str:
    normalized_spec = dict(site_world_spec)
    normalized_spec.pop("canonical_package_version", None)
    digest = hashlib.sha256()
    for payload in (
        scene_memory_manifest,
        conditioning_bundle,
        object_geometry_manifest,
        task_anchor_manifest,
        normalized_spec,
        protected_regions_manifest,
        canonical_render_policy,
        presentation_variance_policy,
    ):
        digest.update(normalized_json_bytes(payload))
    return digest.hexdigest()


def verify_canonical_package_version(
    *,
    spec: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    canonical_render_policy: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
) -> Optional[str]:
    conditioning = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
    geometry = dict(spec.get("geometry") or {}) if isinstance(spec.get("geometry"), Mapping) else {}
    local_paths = dict(conditioning.get("local_paths") or {}) if isinstance(conditioning.get("local_paths"), Mapping) else {}

    def _optional_path(*values: Any) -> Optional[Path]:
        for value in values:
            text = str(value or "").strip()
            if text:
                return Path(text).resolve()
        return None

    scene_memory_manifest_path = _optional_path(conditioning.get("scene_memory_manifest_path"), local_paths.get("scene_memory_manifest_path"))
    conditioning_bundle_path = _optional_path(conditioning.get("conditioning_bundle_path"), local_paths.get("conditioning_bundle_path"))
    object_geometry_manifest_path = _optional_path(geometry.get("object_geometry_manifest_path"))
    task_anchor_manifest_path = _optional_path(spec.get("task_anchor_manifest_path"))
    if not all(path is not None and path.is_file() for path in (scene_memory_manifest_path, conditioning_bundle_path, object_geometry_manifest_path, task_anchor_manifest_path)):
        return "canonical_package_verification_inputs_missing"
    observed = compute_canonical_package_version(
        scene_memory_manifest=_read_json(scene_memory_manifest_path),
        conditioning_bundle=_read_json(conditioning_bundle_path),
        object_geometry_manifest=_read_json(object_geometry_manifest_path),
        task_anchor_manifest=_read_json(task_anchor_manifest_path),
        site_world_spec=spec,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
    )
    expected = str(spec.get("canonical_package_version") or "").strip()
    if expected and observed != expected:
        return f"canonical_package_version_mismatch:{observed}"
    return None


def snapshot_runtime_layer_bundle(bundle: Mapping[str, Any], cache_dir: Path) -> Dict[str, str]:
    snapshots: Dict[str, str] = {}
    for key, raw_path in bundle.get("paths", {}).items():
        source = Path(str(raw_path)).resolve()
        destination = cache_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        snapshots[key] = str(destination)
    return snapshots


def _bbox_extents(regions: list[Mapping[str, Any]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for region in regions:
        bbox = dict(region.get("geometry_refs", {}).get("placement_bbox") or {})
        center = bbox.get("center") if isinstance(bbox.get("center"), list) else None
        extents = bbox.get("extents") if isinstance(bbox.get("extents"), list) else None
        if not center or not extents or len(center) < 2 or len(extents) < 2:
            continue
        half_x = max(float(extents[0]) / 2.0, 0.1)
        half_y = max(float(extents[1]) / 2.0, 0.1)
        xs.extend([float(center[0]) - half_x, float(center[0]) + half_x])
        ys.extend([float(center[1]) - half_y, float(center[1]) + half_y])
    if not xs or not ys:
        return -1.0, 1.0, -1.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def _region_rect(region: Mapping[str, Any], shape: tuple[int, int], extents: tuple[float, float, float, float]) -> Optional[tuple[int, int, int, int]]:
    bbox = dict(region.get("geometry_refs", {}).get("placement_bbox") or {})
    center = bbox.get("center") if isinstance(bbox.get("center"), list) else None
    dims = bbox.get("extents") if isinstance(bbox.get("extents"), list) else None
    if not center or not dims or len(center) < 2 or len(dims) < 2:
        return None
    height, width = shape
    min_x, max_x, min_y, max_y = extents
    x_range = max(max_x - min_x, 1e-6)
    y_range = max(max_y - min_y, 1e-6)
    cx = (float(center[0]) - min_x) / x_range
    cy = (float(center[1]) - min_y) / y_range
    hw = max(float(dims[0]) / x_range, 0.06) * width * 0.5
    hh = max(float(dims[1]) / y_range, 0.06) * height * 0.5
    px = int(np.clip(cx * (width - 1), 0, width - 1))
    py = int(np.clip(cy * (height - 1), 0, height - 1))
    x0 = max(px - int(hw), 0)
    x1 = min(px + int(hw), width - 1)
    y0 = max(py - int(hh), 0)
    y1 = min(py + int(hh), height - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def build_target_view_masks(
    *,
    protected_regions_manifest: Mapping[str, Any],
    frame_shape: tuple[int, int],
) -> Dict[str, np.ndarray]:
    regions = [dict(item) for item in protected_regions_manifest.get("regions", []) if isinstance(item, Mapping)]
    extents = _bbox_extents(regions)
    locked = np.zeros(frame_shape, dtype=bool)
    uncertain = np.zeros(frame_shape, dtype=bool)
    editable = np.zeros(frame_shape, dtype=bool)
    for region in regions:
        rect = _region_rect(region, frame_shape, extents)
        classification = str(region.get("classification") or "editable").strip().lower()
        if rect is None:
            editable[:] = True
            continue
        x0, y0, x1, y1 = rect
        if classification == "locked":
            locked[y0 : y1 + 1, x0 : x1 + 1] = True
        elif classification == "uncertain":
            uncertain[y0 : y1 + 1, x0 : x1 + 1] = True
        else:
            editable[y0 : y1 + 1, x0 : x1 + 1] = True
        if classification == "locked" and bool(region.get("task_critical")) and cv2 is not None:
            kernel = np.ones((2 * TASK_CRITICAL_DILATION_PX + 1, 2 * TASK_CRITICAL_DILATION_PX + 1), dtype=np.uint8)
            locked = cv2.dilate(locked.astype(np.uint8), kernel, iterations=1).astype(bool)
    editable = np.logical_or(editable, ~(np.logical_or(locked, uncertain)))
    editable = np.logical_and(editable, ~locked)
    uncertain = np.logical_and(uncertain, ~locked)
    return {"locked_mask": locked, "uncertain_mask": uncertain, "editable_mask": editable}


def _effect_strength(config: Mapping[str, Any], step_index: int) -> int:
    seed = hashlib.sha256(
        "::".join(
            [
                str(config.get("prompt") or ""),
                str(config.get("presentation_model") or ""),
                json.dumps(config.get("trajectory") or {}, sort_keys=True),
                str(step_index),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return 8 + (int(seed[:2], 16) % 24)


def _generator_candidate(base_frame: np.ndarray, *, config: Mapping[str, Any], editable_mask: np.ndarray, step_index: int, strict_editable_only: bool) -> np.ndarray:
    candidate = base_frame.copy()
    strength = _effect_strength(config, step_index)
    mask = editable_mask if strict_editable_only else np.ones(editable_mask.shape, dtype=bool)
    tinted = candidate.astype(np.int16)
    tinted[..., 1] = np.clip(tinted[..., 1] + strength, 0, 255)
    tinted[..., 2] = np.clip(tinted[..., 2] - max(3, strength // 3), 0, 255)
    tinted = tinted.astype(np.uint8)
    candidate[mask] = tinted[mask]
    return candidate


def composite_runtime_layer(
    *,
    canonical_frame: np.ndarray,
    protected_regions_manifest: Mapping[str, Any],
    presentation_config: Mapping[str, Any],
    session_dir: Path,
    step_index: int,
    camera_id: str,
) -> Dict[str, Any]:
    masks = build_target_view_masks(
        protected_regions_manifest=protected_regions_manifest,
        frame_shape=canonical_frame.shape[:2],
    )
    locked_mask = masks["locked_mask"]
    uncertain_mask = masks["uncertain_mask"]
    editable_mask = masks["editable_mask"]

    attempt = 0
    violations: list[Dict[str, Any]] = []
    final_frame = canonical_frame.copy()
    while attempt <= LOCK_VIOLATION_RETRY_BUDGET:
        candidate = _generator_candidate(
            canonical_frame,
            config=presentation_config,
            editable_mask=editable_mask,
            step_index=step_index,
            strict_editable_only=attempt > 0,
        )
        touched_locked = bool(np.any(np.any(candidate != canonical_frame, axis=2) & locked_mask))
        if touched_locked:
            violations.append({"attempt": attempt + 1, "reason": "locked_region_modified"})
            attempt += 1
            if attempt > LOCK_VIOLATION_RETRY_BUDGET:
                final_frame = canonical_frame.copy()
                break
            continue
        final_frame = canonical_frame.copy()
        final_frame[editable_mask] = candidate[editable_mask]
        break

    editable_ratio = float(editable_mask.sum()) / float(max(1, editable_mask.size))
    quality_flags = {
        "presentation_quality": "degraded" if editable_ratio > DEGRADED_EDITABLE_RATIO_THRESHOLD else "normal",
        "editable_ratio": round(editable_ratio, 4),
        "locked_ratio": round(float(locked_mask.sum()) / float(max(1, locked_mask.size)), 4),
    }
    debug_artifacts: Dict[str, str] = {}
    if bool(presentation_config.get("debug_mode")):
        debug_dir = session_dir / "debug" / f"step_{step_index:03d}" / camera_id
        canonical_path = debug_dir / "canonical_only.png"
        locked_path = debug_dir / "locked_mask.png"
        editable_path = debug_dir / "editable_mask.png"
        composite_path = debug_dir / "final_composite.png"
        _save_rgb(canonical_path, canonical_frame)
        _save_mask_png(locked_path, locked_mask)
        _save_mask_png(editable_path, editable_mask)
        _save_rgb(composite_path, final_frame)
        debug_artifacts = {
            "canonical_only": str(canonical_path),
            "locked_mask": str(locked_path),
            "editable_mask": str(editable_path),
            "final_composite": str(composite_path),
        }

    return {
        "frame": final_frame,
        "locked_mask": locked_mask,
        "uncertain_mask": uncertain_mask,
        "editable_mask": editable_mask,
        "quality_flags": quality_flags,
        "protected_region_violations": violations,
        "debug_artifacts": debug_artifacts,
    }


def update_presentation_session_manifest(*, session_dir: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    path = session_dir / "presentation_session_manifest.json"
    existing = _read_json(path) if path.is_file() else {}
    updated = {**existing, **dict(payload)}
    _write_json(path, updated)
    return updated
