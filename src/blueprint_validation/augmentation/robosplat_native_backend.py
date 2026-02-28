"""Native RoboSplat-style 3D Gaussian augmentation backend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..common import get_logger
from ..config import FacilityConfig, ValidationConfig
from ..rendering.camera_paths import load_path_from_json
from ..rendering.gsplat_renderer import render_video
from ..rendering.ply_loader import GaussianSplatData, load_splat
from ..rendering.scene_geometry import (
    cluster_scene_points,
    load_obbs_from_task_targets,
    select_gaussians_in_sphere,
)

logger = get_logger("augmentation.robosplat_native_backend")


def run_native_backend(
    config: ValidationConfig,
    facility: FacilityConfig,
    source_clips: List[Dict],
    stage_dir: Path,
    object_source_priority: Sequence[str],
) -> Dict:
    """Generate augmented clips by editing Gaussian subsets and re-rendering."""
    if not facility.ply_path.exists():
        return {
            "status": "failed",
            "reason": f"missing_ply:{facility.ply_path}",
            "backend_used": "native",
            "generated": [],
            "object_source": "none",
        }

    device = "cuda" if _has_cuda() else "cpu"
    try:
        splat = load_splat(facility.ply_path, device=device)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"load_splat_failed:{exc}",
            "backend_used": "native",
            "generated": [],
            "object_source": "none",
        }

    means_np = _to_numpy(splat.means)
    centers, object_source = _resolve_object_centers(
        facility=facility,
        means_np=means_np,
        object_source_priority=object_source_priority,
    )
    if not centers:
        centers = [means_np.mean(axis=0)]
        object_source = "cluster"

    generated: List[Dict] = []
    for clip in source_clips:
        source_clip_name = str(clip.get("clip_name", "clip"))
        camera_path_raw = str(clip.get("camera_path", ""))
        if not camera_path_raw:
            logger.warning(
                "Skipping native augmentation for %s: missing camera_path", source_clip_name
            )
            continue
        camera_path = Path(camera_path_raw)
        if not camera_path.exists():
            logger.warning(
                "Skipping native augmentation for %s: camera path not found", source_clip_name
            )
            continue

        resolution_raw = clip.get("resolution", [480, 640])
        resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
        poses = load_path_from_json(camera_path, resolution=resolution)
        if not poses:
            continue
        fps = int(clip.get("fps", config.render.fps))

        for variant_idx in range(max(1, int(config.robosplat.variants_per_input))):
            center = np.asarray(centers[variant_idx % len(centers)], dtype=np.float32)
            edited = _clone_splat(splat)
            variant_ops = _sample_variant_ops(source_clip_name, variant_idx)
            selected_idx = _select_subset_indices(
                means_np=_to_numpy(edited.means),
                center=center,
                radius_m=float(variant_ops["selection_radius_m"]),
            )
            if len(selected_idx) == 0:
                continue
            _apply_edit_ops(edited, selected_idx, variant_ops)

            out_clip_name = f"{source_clip_name}_rb{variant_idx:02d}"
            output = render_video(
                splat=edited,
                poses=poses,
                output_dir=stage_dir,
                clip_name=out_clip_name,
                fps=fps,
            )

            generated.append(
                {
                    "clip_name": out_clip_name,
                    "path_type": clip.get("path_type", "augmented"),
                    "clip_index": clip.get("clip_index", -1),
                    "num_frames": len(poses),
                    "resolution": list(resolution),
                    "fps": fps,
                    "video_path": str(output.video_path),
                    "depth_video_path": str(output.depth_video_path)
                    if output.depth_video_path
                    else "",
                    "source_clip_name": source_clip_name,
                    "source_video_path": str(clip.get("video_path", "")),
                    "source_depth_video_path": str(clip.get("depth_video_path", "")),
                    "source_camera_path": str(camera_path),
                    "source_scene_state_id": "base_scene",
                    "variant_id": f"rb-{variant_idx:02d}",
                    "variant_ops": variant_ops,
                    "object_source": object_source,
                    "augmentation_type": "robosplat_full",
                    "backend_used": "native",
                }
            )

    return {
        "status": "success" if generated else "failed",
        "reason": "ok" if generated else "no_native_variants_generated",
        "backend_used": "native",
        "generated": generated,
        "object_source": object_source,
    }


def _resolve_object_centers(
    facility: FacilityConfig,
    means_np: np.ndarray,
    object_source_priority: Sequence[str],
) -> tuple[List[np.ndarray], str]:
    for source in object_source_priority:
        source = str(source)
        if source == "task_hints_obb":
            if facility.task_hints_path is None or not facility.task_hints_path.exists():
                continue
            try:
                obbs = load_obbs_from_task_targets(facility.task_hints_path)
                centers = [np.asarray(obb.center, dtype=np.float32) for obb in obbs]
                if centers:
                    return centers, "task_hints_obb"
            except Exception as exc:
                logger.warning("OBB extraction failed: %s", exc)
        elif source == "cluster":
            try:
                centers = cluster_scene_points(means_np, num_clusters=8, max_points=30000)
                if len(centers) > 0:
                    return [np.asarray(c, dtype=np.float32) for c in centers], "cluster"
            except Exception as exc:
                logger.warning("Scene clustering failed: %s", exc)
        elif source == "vlm_detect":
            try:
                from ..rendering.vlm_scene_detector import detect_and_generate_specs

                scene_center = means_np.mean(axis=0)
                specs = detect_and_generate_specs(
                    splat_means_np=means_np,
                    scene_center=scene_center,
                    num_views=4,
                )
                if not isinstance(specs, list):
                    specs = list(getattr(specs, "specs", []))
                centers = []
                for spec in specs:
                    if spec.approach_point is None:
                        continue
                    centers.append(np.asarray(spec.approach_point, dtype=np.float32))
                if centers:
                    return centers, "vlm_detect"
            except Exception as exc:
                logger.warning("VLM object detection failed: %s", exc)
    return [], "none"


def _select_subset_indices(
    means_np: np.ndarray,
    center: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    return select_gaussians_in_sphere(
        means=means_np,
        center=center,
        radius_m=max(0.05, radius_m),
        max_points=12000,
    )


def _sample_variant_ops(source_clip_name: str, variant_idx: int) -> Dict[str, float]:
    seed = int.from_bytes(
        f"{source_clip_name}:{variant_idx}".encode("utf-8"),
        "little",
        signed=False,
    ) % (2**32)
    rng = np.random.default_rng(seed)
    return {
        "translate_x_m": float(rng.uniform(-0.12, 0.12)),
        "translate_y_m": float(rng.uniform(-0.12, 0.12)),
        "translate_z_m": float(rng.uniform(-0.04, 0.04)),
        "yaw_deg": float(rng.uniform(-15.0, 15.0)),
        "scale": float(rng.uniform(0.92, 1.08)),
        "relight_gain": float(rng.uniform(0.85, 1.20)),
        "selection_radius_m": float(rng.uniform(0.25, 0.55)),
    }


def _apply_edit_ops(splat: GaussianSplatData, indices: np.ndarray, ops: Dict[str, float]) -> None:
    import torch

    if len(indices) == 0:
        return
    idx = torch.as_tensor(indices, device=splat.means.device, dtype=torch.long)
    pts = splat.means.index_select(0, idx)
    center = pts.mean(dim=0, keepdim=True)

    theta = math.radians(float(ops["yaw_deg"]))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot = torch.tensor(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        device=splat.means.device,
        dtype=pts.dtype,
    )

    scale = float(ops["scale"])
    trans = torch.tensor(
        [ops["translate_x_m"], ops["translate_y_m"], ops["translate_z_m"]],
        device=splat.means.device,
        dtype=pts.dtype,
    )
    transformed = ((pts - center) @ rot.T) * scale + center + trans
    splat.means.index_copy_(0, idx, transformed)

    # Adjust log-scales and SH coefficients to emulate geometric/lighting perturbations.
    scale_log_delta = math.log(max(1e-3, scale))
    splat.scales.index_add_(
        0,
        idx,
        torch.full_like(splat.scales.index_select(0, idx), fill_value=scale_log_delta),
    )
    relight = float(ops["relight_gain"])
    sh = splat.sh_coeffs.index_select(0, idx) * relight
    splat.sh_coeffs.index_copy_(0, idx, sh)


def _clone_splat(splat: GaussianSplatData) -> GaussianSplatData:
    return GaussianSplatData(
        means=splat.means.clone(),
        scales=splat.scales.clone(),
        quats=splat.quats.clone(),
        opacities=splat.opacities.clone(),
        sh_coeffs=splat.sh_coeffs.clone(),
        num_points=splat.num_points,
    )


def _to_numpy(tensor_like) -> np.ndarray:
    if hasattr(tensor_like, "detach"):
        return tensor_like.detach().cpu().numpy()
    return np.asarray(tensor_like)


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
