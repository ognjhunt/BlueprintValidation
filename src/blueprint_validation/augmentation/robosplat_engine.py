"""RoboSplat augmentation orchestration (vendor/native/legacy fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..common import get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from .robosplat_bootstrap_demo import resolve_or_create_demo_manifest
from .robosplat_native_backend import run_native_backend
from .robosplat_quality_gate import validate_augmented_clip
from .robosplat_scan import augment_scan_only_clip
from .robosplat_vendor_backend import run_vendor_backend, vendor_backend_available

logger = get_logger("augmentation.robosplat_engine")


@dataclass
class RoboSplatRunResult:
    status: str
    backend_used: str
    manifest_path: Path
    num_source_clips: int
    num_augmented_clips: int
    num_total_clips: int
    num_rejected_quality: int
    object_source: str
    demo_source: str
    fallback_backend: str
    detail: str = ""


def run_robosplat_augmentation(
    config: ValidationConfig,
    facility: FacilityConfig,
    work_dir: Path,
    stage_dir: Path,
    source_manifest_path: Path,
) -> RoboSplatRunResult:
    source_manifest = read_json(source_manifest_path)
    source_clips: List[Dict] = list(source_manifest.get("clips", []))
    if not source_clips:
        return RoboSplatRunResult(
            status="failed",
            backend_used="none",
            manifest_path=stage_dir / "augmented_manifest.json",
            num_source_clips=0,
            num_augmented_clips=0,
            num_total_clips=0,
            num_rejected_quality=0,
            object_source="none",
            demo_source="none",
            fallback_backend="none",
            detail="No source clips found for augmentation",
        )

    demo_source = "none"
    if config.robosplat.parity_mode != "scan_only" and config.robosplat.demo_source in {
        "synthetic",
        "real",
        "required_real",
    }:
        demo_manifest = resolve_or_create_demo_manifest(
            config=config,
            facility=facility,
            work_dir=work_dir,
            source_clips=source_clips,
        )
        if demo_manifest and demo_manifest.exists():
            demo_source = config.robosplat.demo_source
        elif config.robosplat.demo_source == "required_real":
            return RoboSplatRunResult(
                status="failed",
                backend_used="none",
                manifest_path=stage_dir / "augmented_manifest.json",
                num_source_clips=len(source_clips),
                num_augmented_clips=0,
                num_total_clips=len(source_clips),
                num_rejected_quality=0,
                object_source="none",
                demo_source="missing_required_real",
                fallback_backend="none",
                detail="Demo source required but no demo manifest available",
            )
    elif config.robosplat.parity_mode == "scan_only":
        demo_source = "scan_only"

    backends = _backend_plan(config.robosplat.backend, config.robosplat.parity_mode)
    fallback_backend = "none"
    errors: List[str] = []

    for backend in backends:
        if backend == "vendor":
            ok, reason = vendor_backend_available(config)
            if not ok:
                errors.append(f"vendor_unavailable:{reason}")
                if config.robosplat.parity_mode == "strict":
                    break
                continue
            vendor_manifest_path = stage_dir / "vendor_augmented_manifest.json"
            result = run_vendor_backend(
                config=config,
                facility=facility,
                stage_dir=stage_dir,
                source_manifest_path=source_manifest_path,
                output_manifest_path=vendor_manifest_path,
            )
            if result.get("status") == "success" and vendor_manifest_path.exists():
                vendor_manifest = read_json(vendor_manifest_path)
                generated = list(vendor_manifest.get("clips", []))
                return _finalize_manifest(
                    config=config,
                    facility=facility,
                    stage_dir=stage_dir,
                    source_manifest_path=source_manifest_path,
                    source_clips=source_clips,
                    generated_entries=generated,
                    augmentation_type="robosplat_full",
                    backend_used="vendor",
                    object_source="vendor",
                    demo_source=demo_source,
                    fallback_backend=fallback_backend,
                )
            errors.append(f"vendor_failed:{result.get('reason', 'unknown')}")
            if config.robosplat.parity_mode == "strict":
                break
            if not config.robosplat.fallback_on_backend_error:
                break
            continue

        if backend == "native":
            result = run_native_backend(
                config=config,
                facility=facility,
                source_clips=source_clips,
                stage_dir=stage_dir,
                object_source_priority=config.robosplat.object_source_priority,
            )
            if result.get("status") == "success":
                return _finalize_manifest(
                    config=config,
                    facility=facility,
                    stage_dir=stage_dir,
                    source_manifest_path=source_manifest_path,
                    source_clips=source_clips,
                    generated_entries=list(result.get("generated", [])),
                    augmentation_type="robosplat_full",
                    backend_used="native",
                    object_source=str(result.get("object_source", "none")),
                    demo_source=demo_source,
                    fallback_backend=fallback_backend,
                )
            errors.append(f"native_failed:{result.get('reason', 'unknown')}")
            if config.robosplat.parity_mode == "strict":
                break
            if not config.robosplat.fallback_on_backend_error:
                break

        if backend == "legacy_scan":
            if (
                not config.robosplat.fallback_to_legacy_scan
                and config.robosplat.backend != "legacy_scan"
            ):
                continue
            fallback_backend = "legacy_scan"
            generated = _run_legacy_scan_backend(config, stage_dir, source_clips)
            return _finalize_manifest(
                config=config,
                facility=facility,
                stage_dir=stage_dir,
                source_manifest_path=source_manifest_path,
                source_clips=source_clips,
                generated_entries=generated,
                augmentation_type="legacy_scan",
                backend_used="legacy_scan",
                object_source="cluster",
                demo_source=demo_source,
                fallback_backend=fallback_backend,
            )

    return RoboSplatRunResult(
        status="failed",
        backend_used="none",
        manifest_path=stage_dir / "augmented_manifest.json",
        num_source_clips=len(source_clips),
        num_augmented_clips=0,
        num_total_clips=len(source_clips),
        num_rejected_quality=0,
        object_source="none",
        demo_source=demo_source,
        fallback_backend=fallback_backend,
        detail="; ".join(errors) if errors else "No augmentation backend succeeded",
    )


def _backend_plan(backend_cfg: str, parity_mode: str) -> List[str]:
    backend_cfg = str(backend_cfg or "auto")
    if backend_cfg == "auto":
        if parity_mode == "scan_only":
            return ["native", "legacy_scan"]
        return ["vendor", "native", "legacy_scan"]
    return [backend_cfg]


def _run_legacy_scan_backend(
    config: ValidationConfig,
    stage_dir: Path,
    source_clips: List[Dict],
) -> List[Dict]:
    generated: List[Dict] = []
    for clip in source_clips:
        source_clip_name = str(clip.get("clip_name", "clip"))
        rgb_path = Path(str(clip.get("video_path", "")))
        depth_val = clip.get("depth_video_path")
        depth_path = Path(str(depth_val)) if depth_val else None
        if not rgb_path.exists():
            continue
        for idx in range(max(1, int(config.robosplat.variants_per_input))):
            out = augment_scan_only_clip(
                video_path=rgb_path,
                depth_video_path=depth_path if depth_path and depth_path.exists() else None,
                output_dir=stage_dir,
                source_clip_name=source_clip_name,
                augment_index=idx,
                config=config.robosplat_scan,
            )
            generated.append(
                {
                    "clip_name": out.clip_name,
                    "path_type": clip.get("path_type", "augmented"),
                    "clip_index": clip.get("clip_index", -1),
                    "num_frames": clip.get("num_frames"),
                    "resolution": clip.get("resolution"),
                    "fps": clip.get("fps"),
                    "video_path": str(out.output_video_path),
                    "depth_video_path": (
                        str(out.output_depth_video_path) if out.output_depth_video_path else ""
                    ),
                    "source_clip_name": out.source_clip_name,
                    "source_video_path": str(out.source_video_path),
                    "source_depth_video_path": (
                        str(out.source_depth_video_path) if out.source_depth_video_path else ""
                    ),
                    "source_camera_path": str(clip.get("camera_path", "")),
                    "source_scene_state_id": "legacy_scan",
                    "variant_id": f"legacy-{idx:02d}",
                    "variant_ops": out.augment_ops,
                    "object_source": "cluster",
                    "augmentation_type": "legacy_scan",
                    "backend_used": "legacy_scan",
                }
            )
    return generated


def _finalize_manifest(
    config: ValidationConfig,
    facility: FacilityConfig,
    stage_dir: Path,
    source_manifest_path: Path,
    source_clips: List[Dict],
    generated_entries: List[Dict],
    augmentation_type: str,
    backend_used: str,
    object_source: str,
    demo_source: str,
    fallback_backend: str,
) -> RoboSplatRunResult:
    # Quality-gate and preserve source clips in manifest.
    manifest_clips: List[Dict] = [dict(c) for c in source_clips]
    accepted: List[Dict] = []
    rejected_quality = 0

    for entry in generated_entries:
        resolution = entry.get("resolution")
        expected_res = None
        if isinstance(resolution, list) and len(resolution) == 2:
            expected_res = (int(resolution[0]), int(resolution[1]))
        if config.robosplat.quality_gate_enabled:
            gate = validate_augmented_clip(
                video_path=Path(str(entry.get("video_path", ""))),
                depth_video_path=(
                    Path(str(entry.get("depth_video_path")))
                    if entry.get("depth_video_path")
                    else None
                ),
                expected_resolution=expected_res,
            )
            entry["quality_gate_passed"] = gate.accepted
            entry["quality_gate_reason"] = gate.reason
            if not gate.accepted:
                rejected_quality += 1
                continue
        else:
            entry["quality_gate_passed"] = True
            entry["quality_gate_reason"] = "disabled"
        accepted.append(entry)

    min_required = max(1, int(config.robosplat.min_variants_required_per_clip))
    min_total_required = len(source_clips) * min_required
    if len(accepted) < min_total_required and config.robosplat.backend != "legacy_scan":
        if config.robosplat.parity_mode == "strict":
            return RoboSplatRunResult(
                status="failed",
                backend_used=backend_used,
                manifest_path=stage_dir / "augmented_manifest.json",
                num_source_clips=len(source_clips),
                num_augmented_clips=len(accepted),
                num_total_clips=len(source_clips) + len(accepted),
                num_rejected_quality=rejected_quality,
                object_source=object_source,
                demo_source=demo_source,
                fallback_backend=fallback_backend,
                detail=(
                    f"Generated {len(accepted)} variants, required {min_total_required} "
                    f"for strict mode"
                ),
            )

    manifest_clips.extend(accepted)
    manifest_path = stage_dir / "augmented_manifest.json"
    write_json(
        {
            "facility": facility.name,
            "source_manifest": str(source_manifest_path),
            "augmentation_type": augmentation_type,
            "backend_used": backend_used,
            "fallback_backend": fallback_backend,
            "object_source": object_source,
            "demo_source": demo_source,
            "num_source_clips": len(source_clips),
            "num_augmented_clips": len(accepted),
            "num_rejected_quality": rejected_quality,
            "num_total_clips": len(manifest_clips),
            "clips": manifest_clips,
        },
        manifest_path,
    )
    return RoboSplatRunResult(
        status="success" if accepted else "failed",
        backend_used=backend_used,
        manifest_path=manifest_path,
        num_source_clips=len(source_clips),
        num_augmented_clips=len(accepted),
        num_total_clips=len(manifest_clips),
        num_rejected_quality=rejected_quality,
        object_source=object_source,
        demo_source=demo_source,
        fallback_backend=fallback_backend,
        detail=(
            f"Generated {len(accepted)} augmented clips ({rejected_quality} rejected)"
            if accepted
            else "No augmented clips passed quality gate"
        ),
    )
