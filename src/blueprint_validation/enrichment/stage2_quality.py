"""CPU-side Stage-2 helpers for coverage gates, context selection, and clip sanity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List

from ..config import ValidationConfig
from ..evaluation.camera_quality import (
    analyze_target_visibility as shared_analyze_target_visibility,
    estimate_clip_blur_score as shared_estimate_clip_blur_score,
    project_target_to_camera_path as shared_project_target_to_camera_path,
    resolve_center_band_bounds as shared_resolve_center_band_bounds,
)


@dataclass(frozen=True)
class CoverageGateResult:
    """Result of Stage-1 manipulation coverage checks run before Stage 2."""

    passed: bool
    detail: str
    metrics: Dict[str, object]


def evaluate_stage1_coverage_gate(
    render_manifest: dict,
    config: ValidationConfig,
) -> CoverageGateResult | None:
    """Validate Stage-1 manipulation coverage before expensive enrichment."""
    if not bool(config.render.stage1_coverage_gate_enabled):
        return None

    import numpy as np

    clips = render_manifest.get("clips", [])
    manipulation_clips = []
    missing_target_annotations = 0
    for clip in clips:
        if str(clip.get("path_type", "")).strip().lower() != "manipulation":
            continue
        path_context = clip.get("path_context") or {}
        target_point = path_context.get("approach_point")
        if not isinstance(target_point, list) or len(target_point) != 3:
            missing_target_annotations += 1
            continue
        manipulation_clips.append((clip, np.asarray(target_point, dtype=np.float64)))

    base_metrics: Dict[str, object] = {
        "coverage_gate_enabled": True,
        "coverage_gate_passed": False,
        "coverage_clip_count": len(manipulation_clips),
        "coverage_missing_target_annotations": missing_target_annotations,
        "coverage_target_count": 0,
        "coverage_targets_passing": 0,
        "coverage_targets_failing": 0,
        "coverage_blurry_clip_count": 0,
        "coverage_blurry_clip_names": [],
        "coverage_targets": [],
        "coverage_min_visible_frame_ratio": float(
            config.render.stage1_coverage_min_visible_frame_ratio
        ),
        "coverage_min_approach_angle_bins": int(
            config.render.stage1_coverage_min_approach_angle_bins
        ),
        "coverage_angle_bin_deg": float(config.render.stage1_coverage_angle_bin_deg),
        "coverage_blur_laplacian_min": float(config.render.stage1_coverage_blur_laplacian_min),
        "coverage_min_center_band_ratio": float(config.render.stage1_coverage_min_center_band_ratio),
        "coverage_center_band_x": [float(v) for v in config.render.stage1_coverage_center_band_x],
        "coverage_center_band_y": [float(v) for v in config.render.stage1_coverage_center_band_y],
    }

    if not manipulation_clips:
        return CoverageGateResult(
            passed=False,
            detail=(
                "Stage 1 coverage gate failed: no manipulation clips with approach-point "
                "annotations were found in the render manifest."
            ),
            metrics=base_metrics,
        )

    min_visible_ratio = float(config.render.stage1_coverage_min_visible_frame_ratio)
    min_angle_bins = int(config.render.stage1_coverage_min_approach_angle_bins)
    angle_bin_deg = float(config.render.stage1_coverage_angle_bin_deg)
    blur_min = float(config.render.stage1_coverage_blur_laplacian_min)
    min_center_band_ratio = float(config.render.stage1_coverage_min_center_band_ratio)
    blur_every = max(1, int(config.render.stage1_coverage_blur_sample_every_n_frames))
    blur_max_samples = max(1, int(config.render.stage1_coverage_blur_max_samples_per_clip))

    by_target: Dict[str, Dict[str, object]] = {}
    blurred_clip_names: List[str] = []

    for clip_entry, target_xyz in manipulation_clips:
        clip_name = str(clip_entry.get("clip_name", "unknown"))
        target_key = target_key_for_xyz(target_xyz)
        target_stats = by_target.setdefault(
            target_key,
            {
                "target_point": [round(float(v), 4) for v in target_xyz.tolist()],
                "num_clips": 0,
                "num_blurry_clips": 0,
                "visible_frames": 0,
                "total_frames": 0,
                "center_band_frames": 0,
                "angle_bins": set(),
                "clip_names": [],
            },
        )
        target_stats["num_clips"] = int(target_stats["num_clips"]) + 1
        clip_names = target_stats.get("clip_names")
        if not isinstance(clip_names, list):
            clip_names = []
            target_stats["clip_names"] = clip_names
        clip_names.append(clip_name)

        blur_score = clip_entry.get("blur_laplacian_score")
        try:
            blur_score = float(blur_score) if blur_score is not None else None
        except Exception:
            blur_score = None
        if blur_score is None:
            blur_score = estimate_clip_blur_score(
                video_path=Path(str(clip_entry.get("video_path", ""))),
                sample_every_n_frames=blur_every,
                max_samples=blur_max_samples,
            )
        is_blurry = blur_score is None or blur_score < blur_min
        if is_blurry:
            target_stats["num_blurry_clips"] = int(target_stats["num_blurry_clips"]) + 1
            blurred_clip_names.append(clip_name)
            continue

        visible_frames, total_frames, center_band_frames, angle_bins = analyze_target_visibility(
            clip_entry=clip_entry,
            target_xyz=target_xyz,
            angle_bin_deg=angle_bin_deg,
            config=config,
        )
        target_stats["visible_frames"] = int(target_stats["visible_frames"]) + visible_frames
        target_stats["total_frames"] = int(target_stats["total_frames"]) + total_frames
        target_stats["center_band_frames"] = (
            int(target_stats["center_band_frames"]) + center_band_frames
        )
        current_bins = target_stats.get("angle_bins")
        if not isinstance(current_bins, set):
            current_bins = set()
            target_stats["angle_bins"] = current_bins
        current_bins.update(angle_bins)

    target_summaries: List[Dict[str, object]] = []
    failed_targets: List[str] = []
    for target_key, stats in by_target.items():
        total_frames = int(stats["total_frames"])
        visible_frames = int(stats["visible_frames"])
        center_band_frames = int(stats["center_band_frames"])
        visible_ratio = (visible_frames / total_frames) if total_frames > 0 else 0.0
        center_band_ratio = (center_band_frames / total_frames) if total_frames > 0 else 0.0
        angle_bin_count = len(stats["angle_bins"])
        blurry_count = int(stats["num_blurry_clips"])
        passed = (
            total_frames > 0
            and blurry_count == 0
            and visible_ratio >= min_visible_ratio
            and center_band_ratio >= min_center_band_ratio
            and angle_bin_count >= min_angle_bins
        )
        summary = {
            "target_key": target_key,
            "target_point": stats["target_point"],
            "num_clips": int(stats["num_clips"]),
            "num_blurry_clips": blurry_count,
            "visible_frame_ratio": round(float(visible_ratio), 4),
            "coverage_center_band_ratio": round(float(center_band_ratio), 4),
            "approach_angle_bins": angle_bin_count,
            "passes": passed,
        }
        target_summaries.append(summary)
        if not passed:
            failed_targets.append(target_key)

    coverage_passed = len(failed_targets) == 0
    metrics = {
        **base_metrics,
        "coverage_gate_passed": coverage_passed,
        "coverage_target_count": len(target_summaries),
        "coverage_targets_passing": len(target_summaries) - len(failed_targets),
        "coverage_targets_failing": len(failed_targets),
        "coverage_targets_center_band_failing": len(
            [
                s
                for s in target_summaries
                if float(s["coverage_center_band_ratio"]) < min_center_band_ratio
            ]
        ),
        "coverage_blurry_clip_count": len(blurred_clip_names),
        "coverage_blurry_clip_names": sorted(blurred_clip_names),
        "coverage_targets": target_summaries,
        "coverage_min_center_band_ratio": min_center_band_ratio,
        "coverage_center_band_x": [float(v) for v in config.render.stage1_coverage_center_band_x],
        "coverage_center_band_y": [float(v) for v in config.render.stage1_coverage_center_band_y],
    }

    if coverage_passed:
        detail = f"Stage 1 coverage gate passed for {len(target_summaries)} manipulation targets."
    else:
        detail_parts: List[str] = []
        if failed_targets:
            detail_parts.append(f"failed targets={', '.join(sorted(failed_targets))}")
        if blurred_clip_names:
            detail_parts.append(f"blurred clips={len(blurred_clip_names)}")
        detail = "Stage 1 coverage gate failed: " + "; ".join(detail_parts)

    return CoverageGateResult(passed=coverage_passed, detail=detail, metrics=metrics)


def target_key_for_xyz(target_xyz: object) -> str:
    """Stable key for grouping manipulation targets from floating-point points."""
    xyz = list(target_xyz) if not isinstance(target_xyz, list) else target_xyz
    rounded = [round(float(v), 3) for v in xyz[:3]]
    return ",".join(f"{v:.3f}" for v in rounded)


def analyze_target_visibility(
    clip_entry: dict,
    target_xyz: object,
    angle_bin_deg: float,
    config: ValidationConfig,
) -> tuple[int, int, int, set[int]]:
    """Estimate target visibility ratio and approach-angle diversity from camera poses."""
    cached_total = clip_entry.get("target_total_frames")
    cached_vis_ratio = clip_entry.get("target_visibility_ratio")
    cached_center_ratio = clip_entry.get("target_center_band_ratio")
    cached_bins = clip_entry.get("target_approach_angle_bins")
    try:
        if (
            cached_total is not None
            and cached_vis_ratio is not None
            and cached_center_ratio is not None
            and cached_bins is not None
        ):
            total_frames = int(cached_total)
            if total_frames > 0:
                visible_frames = int(round(float(cached_vis_ratio) * total_frames))
                center_band_frames = int(round(float(cached_center_ratio) * total_frames))
                angle_bin_count = max(0, int(cached_bins))
                return (
                    visible_frames,
                    total_frames,
                    center_band_frames,
                    set(range(angle_bin_count)),
                )
    except Exception:
        pass

    total_frames, visible_samples = project_target_to_camera_path(clip_entry, target_xyz)
    return shared_analyze_target_visibility(
        total_frames=total_frames,
        visible_samples=visible_samples,
        angle_bin_deg=angle_bin_deg,
        center_band_x=config.render.stage1_coverage_center_band_x,
        center_band_y=config.render.stage1_coverage_center_band_y,
    )


def project_target_to_camera_path(
    clip_entry: dict,
    target_xyz: object,
) -> tuple[int, List[Dict[str, float]]]:
    """Project a clip target point into all camera-path frames."""
    return shared_project_target_to_camera_path(clip_entry, target_xyz)


def resolve_center_band_bounds(config: ValidationConfig) -> tuple[float, float, float, float]:
    """Resolve normalized frame center-band bounds from config."""
    return shared_resolve_center_band_bounds(
        config.render.stage1_coverage_center_band_x,
        config.render.stage1_coverage_center_band_y,
    )


def resolve_clip_context_selection(
    clip_entry: dict,
    config: ValidationConfig,
) -> tuple[int | None, str, float | None]:
    """Resolve preferred context frame for a clip."""
    fixed_index = (
        int(config.enrich.context_frame_index)
        if config.enrich.context_frame_index is not None
        else None
    )
    mode = str(config.enrich.context_frame_mode or "target_centered").strip().lower()
    if mode != "target_centered":
        if fixed_index is not None:
            return fixed_index, "fixed", None
        return None, "deterministic", None

    if str(clip_entry.get("path_type", "")).strip().lower() == "manipulation":
        path_context = clip_entry.get("path_context") or {}
        target_point = path_context.get("approach_point")
        if isinstance(target_point, list) and len(target_point) == 3:
            _, visible_samples = project_target_to_camera_path(clip_entry, target_point)
            if visible_samples:
                x_lo, x_hi, y_lo, y_hi = resolve_center_band_bounds(config)
                in_band = [
                    s
                    for s in visible_samples
                    if x_lo <= float(s["u_norm"]) <= x_hi and y_lo <= float(s["v_norm"]) <= y_hi
                ]
                pool = in_band if in_band else visible_samples

                def _center_dist(sample: Dict[str, float]) -> float:
                    du = float(sample["u_norm"]) - 0.5
                    dv = float(sample["v_norm"]) - 0.5
                    return math.sqrt(du * du + dv * dv)

                best = min(pool, key=_center_dist)
                best_dist = _center_dist(best)
                max_dist = math.sqrt(0.5)
                center_score = max(0.0, 1.0 - (best_dist / max_dist))
                return int(best["frame_index"]), "target_centered", float(center_score)

    if fixed_index is not None:
        return fixed_index, "fixed", None
    return None, "deterministic", None


def estimate_clip_blur_score(
    video_path: Path,
    sample_every_n_frames: int,
    max_samples: int,
) -> float | None:
    """Estimate sharpness via Laplacian variance over sampled video frames."""
    return shared_estimate_clip_blur_score(
        video_path=video_path,
        sample_every_n_frames=sample_every_n_frames,
        max_samples=max_samples,
    )


def analyze_output_visual_quality(
    video_path: Path,
    *,
    max_frames: int = 24,
) -> Dict[str, float] | None:
    """Compute lightweight visual-collapse heuristics for a generated output clip."""
    import cv2
    import numpy as np

    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    blur_scores: List[float] = []
    green_ratios: List[float] = []
    interframe_deltas: List[float] = []
    prev_gray = None
    decoded = 0
    try:
        while decoded < max(1, int(max_frames)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            decoded += 1
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            b = frame[:, :, 0].astype(np.float32)
            g = frame[:, :, 1].astype(np.float32)
            r = frame[:, :, 2].astype(np.float32)
            green_mask = (g > (r + 20.0)) & (g > (b + 20.0)) & (g > 60.0)
            green_ratios.append(float(np.mean(green_mask)))

            if prev_gray is not None:
                delta = cv2.absdiff(gray, prev_gray)
                interframe_deltas.append(float(np.mean(delta)))
            prev_gray = gray
    finally:
        cap.release()

    if not blur_scores:
        return None
    return {
        "blur_laplacian_mean": float(np.mean(np.asarray(blur_scores, dtype=np.float64))),
        "green_dominance_ratio": float(np.mean(np.asarray(green_ratios, dtype=np.float64))),
        "interframe_delta_mean": (
            float(np.mean(np.asarray(interframe_deltas, dtype=np.float64)))
            if interframe_deltas
            else 0.0
        ),
    }


def read_video_frame(
    video_path: Path,
    preferred_index: int | None = None,
) -> tuple[object | None, int | None, int]:
    """Read a deterministic frame from a video clip."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    if preferred_index is None:
        target_index = max(0, total_frames // 4)
    else:
        target_index = max(0, min(int(preferred_index), total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        target_index = 0 if ok and frame is not None else None
    cap.release()
    return frame, target_index, total_frames
