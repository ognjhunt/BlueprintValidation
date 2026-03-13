"""Local grounding/compositing helpers for downstream NeoVerse runtime execution.

This module supports local runtime rendering behavior. It does not own the portable
site-world or runtime-layer contracts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for grounded runtime rendering")


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _copy_into_cache(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _load_frame(path: Path) -> np.ndarray:
    _require_cv2()
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"failed to load frame at {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _save_frame(path: Path, frame: np.ndarray) -> None:
    _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _read_mask(path: Path, *, shape: tuple[int, int]) -> np.ndarray:
    _require_cv2()
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to load mask at {path}")
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _resolve_local_path(raw: Any) -> Optional[Path]:
    value = str(raw or "").strip()
    if not value or value.startswith(("gs://", "http://", "https://")):
        return None
    path = Path(value).resolve()
    return path if path.exists() else None


def _artifact_path(spec: Mapping[str, Any], key: str) -> Optional[Path]:
    value = spec.get(key)
    if isinstance(value, Mapping):
        for subkey in ("local_path", "path", "uri"):
            path = _resolve_local_path(value.get(subkey))
            if path is not None:
                return path
        return None
    return _resolve_local_path(value)


def _conditioning_paths(spec: Mapping[str, Any]) -> Dict[str, Optional[Path]]:
    conditioning = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
    local_paths = dict(conditioning.get("local_paths") or {}) if isinstance(conditioning.get("local_paths"), Mapping) else {}
    geometry = dict(spec.get("geometry") or {}) if isinstance(spec.get("geometry"), Mapping) else {}
    return {
        "visual_source": (
            _resolve_local_path(local_paths.get("keyframe_path"))
            or _resolve_local_path(local_paths.get("raw_video_path"))
            or _resolve_local_path(conditioning.get("keyframe_uri"))
            or _resolve_local_path(conditioning.get("raw_video_uri"))
        ),
        "arkit_poses": _resolve_local_path(local_paths.get("arkit_poses_path")) or _resolve_local_path(conditioning.get("arkit_poses_uri")),
        "arkit_intrinsics": _resolve_local_path(local_paths.get("arkit_intrinsics_path")) or _resolve_local_path(conditioning.get("arkit_intrinsics_uri")),
        "depth": _resolve_local_path(local_paths.get("depth_path")) or _resolve_local_path(conditioning.get("depth_uri")),
        "occupancy": _resolve_local_path(local_paths.get("occupancy_path")) or _resolve_local_path(geometry.get("occupancy_path")),
        "object_index": _resolve_local_path(local_paths.get("object_index_path")) or _resolve_local_path(geometry.get("object_index_path")),
        "object_geometry": _resolve_local_path(local_paths.get("object_geometry_manifest_path")) or _resolve_local_path(geometry.get("object_geometry_manifest_path")),
        "collision": _resolve_local_path(local_paths.get("collision_path")) or _resolve_local_path(geometry.get("collision_path")),
    }


def _package_material(spec: Mapping[str, Any], artifact_payloads: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    conditioning_paths = _conditioning_paths(spec)
    material = {
        "scene_id": spec.get("scene_id"),
        "capture_id": spec.get("capture_id"),
        "site_submission_id": spec.get("site_submission_id"),
        "conditioning": {
            key: str(value) if value is not None else None
            for key, value in conditioning_paths.items()
        },
        "artifacts": {key: payload for key, payload in artifact_payloads.items()},
    }
    return material


def compute_canonical_package_version(
    spec: Mapping[str, Any],
    *,
    protected_regions_manifest: Mapping[str, Any],
    canonical_render_policy: Mapping[str, Any],
    presentation_variance_policy: Mapping[str, Any],
) -> str:
    material = _package_material(
        spec,
        artifact_payloads={
            "protected_regions_manifest": dict(protected_regions_manifest),
            "canonical_render_policy": dict(canonical_render_policy),
            "presentation_variance_policy": dict(presentation_variance_policy),
        },
    )
    return hashlib.sha256(_stable_json_bytes(material)).hexdigest()


def validate_grounding_spec(spec: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    version = str(spec.get("canonical_package_version") or "").strip()
    if not version:
        blockers.append("missing_canonical_package_version")

    for key in ("protected_regions_manifest", "canonical_render_policy", "presentation_variance_policy"):
        if _artifact_path(spec, key) is None:
            blockers.append(f"missing_runtime_artifact:{key}")

    conditioning_paths = _conditioning_paths(spec)
    for key in ("visual_source", "arkit_poses", "arkit_intrinsics"):
        if conditioning_paths[key] is None:
            blockers.append(f"missing_local_conditioning:{key}")
    for key in ("depth", "occupancy", "object_index", "object_geometry"):
        if conditioning_paths[key] is None:
            warnings.append(f"{key}_path_missing")

    if blockers:
        return blockers, warnings

    protected_regions_path = _artifact_path(spec, "protected_regions_manifest")
    canonical_render_policy_path = _artifact_path(spec, "canonical_render_policy")
    presentation_variance_policy_path = _artifact_path(spec, "presentation_variance_policy")
    assert protected_regions_path is not None
    assert canonical_render_policy_path is not None
    assert presentation_variance_policy_path is not None

    protected_regions_manifest = _read_json(protected_regions_path)
    canonical_render_policy = _read_json(canonical_render_policy_path)
    presentation_variance_policy = _read_json(presentation_variance_policy_path)
    computed = compute_canonical_package_version(
        spec,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
    )
    if version != computed:
        blockers.append("canonical_package_version_mismatch")
    return blockers, warnings


@dataclass(frozen=True)
class GroundingBundle:
    canonical_package_version: str
    canonical_package_uri: str
    canonical_frame_path: Path
    protected_regions_manifest_path: Path
    canonical_render_policy_path: Path
    presentation_variance_policy_path: Path
    protected_regions_manifest: Dict[str, Any]
    canonical_render_policy: Dict[str, Any]
    presentation_variance_policy: Dict[str, Any]


def load_grounding_bundle(
    spec: Mapping[str, Any],
    *,
    canonical_frame_path: Path,
    cache_dir: Path,
) -> GroundingBundle:
    blockers, warnings = validate_grounding_spec(spec)
    if blockers:
        raise RuntimeError(f"invalid canonical grounding spec: {', '.join(blockers)}")
    del warnings
    protected_regions_source = _artifact_path(spec, "protected_regions_manifest")
    canonical_render_policy_source = _artifact_path(spec, "canonical_render_policy")
    presentation_variance_policy_source = _artifact_path(spec, "presentation_variance_policy")
    assert protected_regions_source is not None
    assert canonical_render_policy_source is not None
    assert presentation_variance_policy_source is not None

    grounding_dir = cache_dir / "grounding"
    protected_regions_manifest_path = _copy_into_cache(protected_regions_source, grounding_dir)
    canonical_render_policy_path = _copy_into_cache(canonical_render_policy_source, grounding_dir)
    presentation_variance_policy_path = _copy_into_cache(presentation_variance_policy_source, grounding_dir)

    protected_regions_manifest = _read_json(protected_regions_manifest_path)
    canonical_render_policy = _read_json(canonical_render_policy_path)
    presentation_variance_policy = _read_json(presentation_variance_policy_path)

    return GroundingBundle(
        canonical_package_version=str(spec.get("canonical_package_version") or ""),
        canonical_package_uri=(
            str(spec.get("canonical_package_uri") or "").strip()
            or str((cache_dir / "canonical_package").resolve())
        ),
        canonical_frame_path=canonical_frame_path.resolve(),
        protected_regions_manifest_path=protected_regions_manifest_path,
        canonical_render_policy_path=canonical_render_policy_path,
        presentation_variance_policy_path=presentation_variance_policy_path,
        protected_regions_manifest=protected_regions_manifest,
        canonical_render_policy=canonical_render_policy,
        presentation_variance_policy=presentation_variance_policy,
    )


def render_capabilities() -> Dict[str, bool]:
    return {
        "protected_region_locking": True,
        "runtime_layer_compositing": True,
        "debug_render_outputs": True,
    }


def _trajectory_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        payload = {str(key): value[key] for key in value}
        if "trajectory" not in payload:
            payload["trajectory"] = "static"
        return payload
    token = str(value or "").strip()
    return {"trajectory": token or "static"}


def trajectory_from_action(action: Sequence[float]) -> Dict[str, str]:
    values = [float(item) for item in action]
    if not values:
        return {"trajectory": "static"}
    padded = values + [0.0] * max(0, 7 - len(values))
    labels = {
        0: ("move_right", "move_left", "distance", 0.28),
        1: ("push_in", "pull_out", "distance", 0.24),
        2: ("pan_right", "pan_left", "angle", 12.0),
        5: ("boom_up", "boom_down", "distance", 0.14),
    }
    axis = max(labels, key=lambda idx: abs(padded[idx]))
    magnitude = float(padded[axis])
    if abs(magnitude) < 1e-3:
        return {"trajectory": "static"}
    positive_name, negative_name, parameter_name, scale = labels[axis]
    if parameter_name == "angle":
        parameter_value = max(4.0, min(abs(magnitude) * scale, 18.0))
    else:
        parameter_value = max(0.05, min(abs(magnitude) * scale, scale))
    return {
        "trajectory": positive_name if magnitude >= 0 else negative_name,
        parameter_name: round(parameter_value, 4),
    }


def _affine_for_view(width: int, height: int, camera_id: str, trajectory: Mapping[str, Any]) -> np.ndarray:
    name = str(trajectory.get("trajectory") or "static").strip().lower()
    dx = 0.0
    dy = 0.0
    scale = 1.0
    angle = 0.0
    distance = float(trajectory.get("distance") or 0.0)
    angle_value = float(trajectory.get("angle") or 0.0)
    if name == "move_right":
        dx = width * min(distance, 0.24)
    elif name == "move_left":
        dx = -width * min(distance, 0.24)
    elif name == "push_in":
        scale = 1.0 + min(distance, 0.24)
    elif name == "pull_out":
        scale = 1.0 - min(distance, 0.20)
    elif name == "boom_up":
        dy = -height * min(distance, 0.18)
    elif name == "boom_down":
        dy = height * min(distance, 0.18)
    elif name == "pan_right":
        angle = min(angle_value or 10.0, 18.0)
    elif name == "pan_left":
        angle = -min(angle_value or 10.0, 18.0)

    if "wrist" in camera_id:
        scale *= 1.05
        dx += width * 0.02
        dy += height * 0.02
    elif "context" in camera_id:
        scale *= 0.96

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix.astype(np.float32)


def _warp_frame(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    _require_cv2()
    height, width = frame.shape[:2]
    return cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _warp_mask(mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    _require_cv2()
    height, width = mask.shape[:2]
    rendered = cv2.warpAffine(mask.astype(np.uint8) * 255, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    return rendered > 0


def _mask_from_bbox(shape: tuple[int, int], bbox: Sequence[Any], *, normalized: bool) -> np.ndarray:
    height, width = shape
    x, y, w, h = [float(value) for value in bbox[:4]]
    if normalized:
        x *= width
        y *= height
        w *= width
        h *= height
    left = max(0, min(width, int(round(x))))
    top = max(0, min(height, int(round(y))))
    right = max(left, min(width, int(round(x + w))))
    bottom = max(top, min(height, int(round(y + h))))
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _region_mask(bundle: GroundingBundle, region: Mapping[str, Any], *, shape: tuple[int, int]) -> np.ndarray:
    mask_path = _resolve_local_path(region.get("mask_path"))
    if mask_path is not None:
        return _read_mask(mask_path, shape=shape)
    bbox = region.get("bbox")
    if isinstance(bbox, Sequence) and len(bbox) >= 4:
        bbox_mode = str(region.get("bbox_mode") or "xywh_pixels").strip().lower()
        return _mask_from_bbox(shape, bbox, normalized="normalized" in bbox_mode)
    return np.zeros(shape, dtype=bool)


def _classify_region(region: Mapping[str, Any]) -> str:
    provenance = str(region.get("provenance") or "").strip().lower()
    confidence = float(region.get("confidence", 0.0) or 0.0)
    observed = float(region.get("observed_coverage", region.get("observation_coverage", 0.0)) or 0.0)
    reprojection_confidence = float(region.get("reprojection_confidence", 1.0) or 0.0)
    task_critical = bool(region.get("task_critical"))

    if not provenance or reprojection_confidence < 0.65 or bool(region.get("missing_geometry")):
        return "editable"
    if task_critical and confidence >= 0.70:
        return "locked"
    if provenance == "observed" and observed >= 0.85:
        return "locked"
    if provenance == "reconstructed" and confidence >= 0.80:
        return "locked"
    if provenance == "reconstructed" and 0.65 <= confidence < 0.80:
        return "uncertain"
    if provenance in {"generated", "inferred"} or confidence < 0.65:
        return "editable"
    classification = str(region.get("classification") or "").strip().lower()
    if classification in {"locked", "uncertain", "editable"}:
        return classification
    return "editable"


def _build_masks(bundle: GroundingBundle, *, shape: tuple[int, int], matrix: np.ndarray) -> Dict[str, np.ndarray]:
    locked = np.zeros(shape, dtype=bool)
    uncertain = np.zeros(shape, dtype=bool)
    editable = np.zeros(shape, dtype=bool)
    regions = list(bundle.protected_regions_manifest.get("regions", []) or [])
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        mask = _warp_mask(_region_mask(bundle, region, shape=shape), matrix)
        classification = _classify_region(region)
        if classification == "locked":
            locked |= mask
            if bool(region.get("task_critical")):
                kernel = np.ones((3, 3), dtype=np.uint8)
                locked = cv2.dilate((locked.astype(np.uint8) * 255), kernel, iterations=1) > 0
        elif classification == "uncertain":
            uncertain |= mask
        else:
            editable |= mask
    uncovered = ~(locked | uncertain | editable)
    editable |= uncovered
    uncertain &= ~locked
    editable &= ~(locked | uncertain)
    return {"locked": locked, "uncertain": uncertain, "editable": editable}


def _candidate_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _generate_editable_candidate(
    canonical_frame: np.ndarray,
    *,
    editable_mask: np.ndarray,
    prompt: str,
    presentation_model: str,
    trajectory: Mapping[str, Any],
    camera_id: str,
    step_index: int,
) -> np.ndarray:
    candidate = canonical_frame.copy()
    if not bool(np.any(editable_mask)):
        return candidate
    seed = _candidate_seed(prompt, presentation_model, json.dumps(dict(trajectory), sort_keys=True), camera_id, step_index)
    rng = np.random.default_rng(seed)
    variation = rng.integers(-18, 19, size=canonical_frame.shape, dtype=np.int16)
    tinted = np.clip(canonical_frame.astype(np.int16) + variation, 0, 255).astype(np.uint8)
    grid = (((np.indices(editable_mask.shape).sum(axis=0) + (seed % 7)) % 7) == 0)[..., None]
    blended = np.where(grid, np.clip(tinted.astype(np.int16) + 8, 0, 255).astype(np.uint8), tinted)
    candidate[editable_mask] = blended[editable_mask]
    return candidate


def _violation_mask(candidate: np.ndarray, canonical_frame: np.ndarray, locked_mask: np.ndarray) -> np.ndarray:
    pixel_diff = np.any(candidate != canonical_frame, axis=2)
    return locked_mask & pixel_diff


@dataclass(frozen=True)
class GroundedRenderResult:
    frame: np.ndarray
    canonical_frame: np.ndarray
    locked_mask: np.ndarray
    editable_mask: np.ndarray
    uncertain_mask: np.ndarray
    quality_flags: Dict[str, Any]
    protected_region_violations: Dict[str, Any]
    debug_artifacts: Dict[str, str]
    presentation_config: Dict[str, Any]


def render_grounded_frame(
    bundle: GroundingBundle,
    *,
    camera_id: str,
    prompt: str,
    trajectory: Mapping[str, Any] | str | None,
    presentation_model: str,
    debug_mode: bool,
    step_index: int,
    output_dir: Path,
) -> GroundedRenderResult:
    base_frame = _load_frame(bundle.canonical_frame_path)
    height, width = base_frame.shape[:2]
    trajectory_payload = _trajectory_payload(trajectory)
    matrix = _affine_for_view(width, height, camera_id, trajectory_payload)
    canonical_frame = _warp_frame(base_frame, matrix)
    masks = _build_masks(bundle, shape=(height, width), matrix=matrix)
    locked_mask = masks["locked"]
    uncertain_mask = masks["uncertain"]
    editable_mask = masks["editable"]

    retry_budget = int(bundle.canonical_render_policy.get("retry_budget", 1) or 1)
    violation_total = 0
    final_frame = canonical_frame.copy()
    fallback_mode = "grounded"
    for attempt in range(retry_budget + 1):
        candidate = _generate_editable_candidate(
            canonical_frame,
            editable_mask=editable_mask,
            prompt=prompt,
            presentation_model=presentation_model,
            trajectory=trajectory_payload,
            camera_id=camera_id,
            step_index=step_index + attempt,
        )
        if bool(bundle.presentation_variance_policy.get("test_force_locked_violation_once")) and attempt == 0 and np.any(locked_mask):
            y, x = np.argwhere(locked_mask)[0]
            candidate[y, x] = np.array([255, 0, 0], dtype=np.uint8)
        if bool(bundle.presentation_variance_policy.get("test_force_locked_violation_always")) and np.any(locked_mask):
            y, x = np.argwhere(locked_mask)[0]
            candidate[y, x] = np.array([255, 0, 0], dtype=np.uint8)
        violation = _violation_mask(candidate, canonical_frame, locked_mask)
        if not bool(np.any(violation)):
            final_frame = canonical_frame.copy()
            final_frame[uncertain_mask] = canonical_frame[uncertain_mask]
            final_frame[editable_mask] = candidate[editable_mask]
            break
        violation_total += 1
        if attempt >= retry_budget:
            fallback_mode = str(bundle.canonical_render_policy.get("fallback_mode") or "canonical_only")
            final_frame = canonical_frame.copy()

    editable_ratio = float(np.count_nonzero(editable_mask)) / float(max(height * width, 1))
    degraded_threshold = float(bundle.canonical_render_policy.get("degraded_quality_threshold", 0.40) or 0.40)
    presentation_quality = "degraded" if editable_ratio > degraded_threshold else "nominal"

    debug_artifacts: Dict[str, str] = {}
    if debug_mode:
        step_dir = output_dir / "debug" / camera_id / f"step_{step_index:03d}"
        canonical_path = step_dir / "canonical_only.png"
        locked_path = step_dir / "locked_mask.png"
        editable_path = step_dir / "editable_mask.png"
        final_path = step_dir / "final_composite.png"
        _save_frame(canonical_path, canonical_frame)
        _save_frame(locked_path, np.repeat((locked_mask.astype(np.uint8) * 255)[..., None], 3, axis=2))
        _save_frame(editable_path, np.repeat((editable_mask.astype(np.uint8) * 255)[..., None], 3, axis=2))
        _save_frame(final_path, final_frame)
        debug_artifacts = {
            "canonical_only": str(canonical_path),
            "locked_mask": str(locked_path),
            "editable_mask": str(editable_path),
            "final_composite": str(final_path),
        }

    return GroundedRenderResult(
        frame=final_frame,
        canonical_frame=canonical_frame,
        locked_mask=locked_mask,
        editable_mask=editable_mask,
        uncertain_mask=uncertain_mask,
        quality_flags={
            "presentation_quality": presentation_quality,
            "editable_ratio": round(editable_ratio, 4),
            "fallback_mode": fallback_mode,
        },
        protected_region_violations={
            "count": violation_total,
            "fallback_mode": fallback_mode,
            "retry_budget": retry_budget,
        },
        debug_artifacts=debug_artifacts,
        presentation_config={
            "prompt": prompt,
            "trajectory": trajectory_payload,
            "presentation_model": presentation_model,
            "debug_mode": bool(debug_mode),
        },
    )
