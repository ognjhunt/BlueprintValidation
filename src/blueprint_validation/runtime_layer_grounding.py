"""Downstream runtime-layer helpers for the NeoVerse runtime service.

Portable runtime-layer policy schemas, thresholds, and canonical version helpers stay in
BlueprintContracts. This module adds local runtime compositing and cache behavior only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from blueprint_contracts.canonical_package import (
    compute_canonical_package_version as shared_compute_canonical_package_version,
)
from blueprint_contracts.canonical_package import (
    normalized_json_bytes as shared_normalized_json_bytes,
)
from blueprint_contracts.canonical_package import (
    verify_canonical_package_version as shared_verify_canonical_package_version,
)
from blueprint_contracts.runtime_layer_contract import (
    DEGRADED_EDITABLE_RATIO_THRESHOLD,
    LOCK_VIOLATION_RETRY_BUDGET,
    TASK_CRITICAL_DILATION_PX,
    load_runtime_layer_bundle as shared_load_runtime_layer_bundle,
    validate_runtime_layer_spec as shared_validate_runtime_layer_spec,
)

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

from blueprint_contracts.site_world_contract import normalize_trajectory_payload


def normalized_json_bytes(payload: Any) -> bytes:
    return shared_normalized_json_bytes(payload)


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
    return shared_validate_runtime_layer_spec(spec)


def load_runtime_layer_bundle(spec: Mapping[str, Any]) -> Dict[str, Any]:
    return shared_load_runtime_layer_bundle(spec)


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
    return shared_compute_canonical_package_version(
        scene_memory_manifest=scene_memory_manifest,
        conditioning_bundle=conditioning_bundle,
        object_geometry_manifest=object_geometry_manifest,
        task_anchor_manifest=task_anchor_manifest,
        site_world_spec=site_world_spec,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
    )


def verify_canonical_package_version(
    *,
    spec: Mapping[str, Any],
    protected_regions_manifest: Mapping[str, Any],
    canonical_render_policy: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
) -> Optional[str]:
    return shared_verify_canonical_package_version(
        spec=spec,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
    )


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
    allow_ungrounded_editable: bool = False,
) -> Dict[str, np.ndarray]:
    regions = [dict(item) for item in protected_regions_manifest.get("regions", []) if isinstance(item, Mapping)]
    grounding_status = str(protected_regions_manifest.get("grounding_status") or "").strip().lower()
    if grounding_status == "ungrounded":
        locked = np.zeros(frame_shape, dtype=bool)
        blocked = np.zeros(frame_shape, dtype=bool)
        uncertain = np.zeros(frame_shape, dtype=bool)
        editable = np.zeros(frame_shape, dtype=bool)
        if allow_ungrounded_editable:
            editable[:] = True
        else:
            uncertain[:] = True
        return {
            "locked_mask": locked,
            "blocked_mask": blocked,
            "uncertain_mask": uncertain,
            "editable_mask": editable,
            "unprojectable_region_count": 0,
        }
    extents = _bbox_extents(regions)
    locked = np.zeros(frame_shape, dtype=bool)
    blocked = np.zeros(frame_shape, dtype=bool)
    uncertain = np.zeros(frame_shape, dtype=bool)
    editable = np.zeros(frame_shape, dtype=bool)
    unprojectable_regions = 0
    for region in regions:
        rect = _region_rect(region, frame_shape, extents)
        classification = str(region.get("classification") or "editable").strip().lower()
        if rect is None:
            unprojectable_regions += 1
            if classification == "locked" or bool(region.get("task_critical")):
                blocked[:] = True
            else:
                uncertain[:] = True
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
    non_editable = np.logical_or(np.logical_or(locked, blocked), uncertain)
    editable = np.logical_or(editable, ~non_editable)
    editable = np.logical_and(editable, ~np.logical_or(locked, blocked))
    uncertain = np.logical_and(uncertain, ~np.logical_or(locked, blocked))
    return {
        "locked_mask": locked,
        "blocked_mask": blocked,
        "uncertain_mask": uncertain,
        "editable_mask": editable,
        "unprojectable_region_count": unprojectable_regions,
    }


def _policy_retry_budget(canonical_render_policy: Mapping[str, Any]) -> int:
    fallback_behavior = (
        dict(canonical_render_policy.get("fallback_behavior") or {})
        if isinstance(canonical_render_policy.get("fallback_behavior"), Mapping)
        else {}
    )
    raw = fallback_behavior.get("retry_budget", canonical_render_policy.get("retry_budget", LOCK_VIOLATION_RETRY_BUDGET))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return LOCK_VIOLATION_RETRY_BUDGET


def _policy_fallback_mode(canonical_render_policy: Mapping[str, Any]) -> str:
    fallback_behavior = (
        dict(canonical_render_policy.get("fallback_behavior") or {})
        if isinstance(canonical_render_policy.get("fallback_behavior"), Mapping)
        else {}
    )
    value = str(
        fallback_behavior.get("on_locked_region_violation")
        or canonical_render_policy.get("fallback_mode")
        or "canonical_only"
    ).strip().lower()
    return value or "canonical_only"


def _policy_degraded_threshold(canonical_render_policy: Mapping[str, Any]) -> float:
    raw = canonical_render_policy.get("degraded_quality_threshold", DEGRADED_EDITABLE_RATIO_THRESHOLD)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEGRADED_EDITABLE_RATIO_THRESHOLD


def _sanitized_presentation_config(
    presentation_config: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    allowed = {
        str(item).strip()
        for item in presentation_variance_policy.get("allowed_variable_inputs", []) or []
        if str(item).strip()
    }
    if not allowed:
        allowed = {"prompt", "presentation_model", "trajectory"}
    return {
        "prompt": str(presentation_config.get("prompt") or "") if "prompt" in allowed else "",
        "presentation_model": (
            str(presentation_config.get("presentation_model") or "")
            if "presentation_model" in allowed
            else ""
        ),
        "trajectory": (
            normalize_trajectory_payload(presentation_config.get("trajectory"))
            if "trajectory" in allowed
            else {"trajectory": "static"}
        ),
        "debug_mode": bool(presentation_config.get("debug_mode")),
        "unsafe_allow_blocked_site_world": bool(presentation_config.get("unsafe_allow_blocked_site_world")),
    }


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
    canonical_render_policy: Mapping[str, Any],
    presentation_config: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
    session_dir: Path,
    step_index: int,
    camera_id: str,
) -> Dict[str, Any]:
    runtime_presentation_config = _sanitized_presentation_config(
        presentation_config,
        presentation_variance_policy,
    )
    allow_ungrounded_editable = bool(runtime_presentation_config.get("unsafe_allow_blocked_site_world"))
    masks = build_target_view_masks(
        protected_regions_manifest=protected_regions_manifest,
        frame_shape=canonical_frame.shape[:2],
        allow_ungrounded_editable=allow_ungrounded_editable,
    )
    locked_mask = masks["locked_mask"]
    blocked_mask = masks["blocked_mask"]
    uncertain_mask = masks["uncertain_mask"]
    editable_mask = masks["editable_mask"]
    unprojectable_region_count = int(masks.get("unprojectable_region_count", 0) or 0)
    grounding_status = str(protected_regions_manifest.get("grounding_status") or "grounded").strip().lower()
    ungrounded_reason = str(protected_regions_manifest.get("ungrounded_reason") or "ungrounded").strip()
    fallback_mode = _policy_fallback_mode(canonical_render_policy)
    retry_budget = _policy_retry_budget(canonical_render_policy)

    attempt = 0
    violations: list[Dict[str, Any]] = []
    final_frame = canonical_frame.copy()
    if grounding_status == "ungrounded" and not allow_ungrounded_editable:
        quality_flags = {
            "presentation_quality": "ungrounded",
            "editable_ratio": 0.0,
            "locked_ratio": 0.0,
            "fallback_mode": "ungrounded_canonical_only",
            "grounding_status": "ungrounded",
            "ungrounded_reason": ungrounded_reason,
            "unprojectable_region_count": unprojectable_region_count,
        }
    else:
        while attempt <= retry_budget:
            candidate = _generator_candidate(
                canonical_frame,
                config=runtime_presentation_config,
                editable_mask=editable_mask,
                step_index=step_index,
                strict_editable_only=attempt > 0,
            )
            changed_pixels = np.any(candidate != canonical_frame, axis=2)
            touched_locked = bool(np.any(changed_pixels & locked_mask))
            touched_blocked = bool(np.any(changed_pixels & blocked_mask))
            if touched_locked or touched_blocked:
                reason = "locked_region_modified" if touched_locked else "unprojectable_region_blocked"
                violations.append({"attempt": attempt + 1, "reason": reason, "fallback_mode": fallback_mode})
                attempt += 1
                if attempt > retry_budget:
                    final_frame = canonical_frame.copy()
                    break
                continue
            final_frame = canonical_frame.copy()
            final_frame[editable_mask] = candidate[editable_mask]
            break

        editable_ratio = float(editable_mask.sum()) / float(max(1, editable_mask.size))
        degraded_threshold = _policy_degraded_threshold(canonical_render_policy)
        quality_flags = {
            "presentation_quality": "degraded" if editable_ratio > degraded_threshold else "normal",
            "editable_ratio": round(editable_ratio, 4),
            "locked_ratio": round(float(locked_mask.sum()) / float(max(1, locked_mask.size)), 4),
            "fallback_mode": fallback_mode if violations else None,
            "grounding_status": grounding_status or "grounded",
            "unsafe_editable_override": bool(allow_ungrounded_editable and grounding_status == "ungrounded"),
            "unprojectable_region_count": unprojectable_region_count,
        }
    debug_artifacts: Dict[str, str] = {}
    if bool(runtime_presentation_config.get("debug_mode")):
        debug_dir = session_dir / "debug" / f"step_{step_index:03d}" / camera_id
        canonical_path = debug_dir / "canonical_only.png"
        locked_path = debug_dir / "locked_mask.png"
        blocked_path = debug_dir / "blocked_mask.png"
        editable_path = debug_dir / "editable_mask.png"
        composite_path = debug_dir / "final_composite.png"
        _save_rgb(canonical_path, canonical_frame)
        _save_mask_png(locked_path, locked_mask)
        _save_mask_png(blocked_path, blocked_mask)
        _save_mask_png(editable_path, editable_mask)
        _save_rgb(composite_path, final_frame)
        debug_artifacts = {
            "canonical_only": str(canonical_path),
            "locked_mask": str(locked_path),
            "blocked_mask": str(blocked_path),
            "editable_mask": str(editable_path),
            "final_composite": str(composite_path),
        }

    return {
        "frame": final_frame,
        "locked_mask": locked_mask,
        "blocked_mask": blocked_mask,
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
