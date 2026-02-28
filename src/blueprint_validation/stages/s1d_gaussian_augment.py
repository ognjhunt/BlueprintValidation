"""Stage 1d: RoboSplat-inspired scan-only Gaussian clip augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..augmentation.robosplat_scan import augment_scan_only_clip
from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from .base import PipelineStage

logger = get_logger("stages.s1d_gaussian_augment")


class GaussianAugmentStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1d_gaussian_augment"

    @property
    def description(self) -> str:
        return "RoboSplat-inspired scan-only augmentation over rendered clips"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del previous_results
        if not config.robosplat_scan.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="robosplat_scan.enabled=false",
            )

        source_manifest_path = _resolve_source_manifest(work_dir)
        if source_manifest_path is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Render/composite/polish manifest not found. Run Stage 1 first "
                    "(and Stage 1b/1c if enabled)."
                ),
            )

        source_manifest = read_json(source_manifest_path)
        source_clips: List[Dict] = list(source_manifest.get("clips", []))
        if not source_clips:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"No clips found in source manifest: {source_manifest_path}",
            )

        stage_dir = work_dir / "gaussian_augment"
        stage_dir.mkdir(parents=True, exist_ok=True)

        manifest_clips: List[Dict] = []
        augmented = 0
        failed = 0

        # Keep original clips in manifest to preserve baseline coverage.
        for clip in source_clips:
            manifest_clips.append(dict(clip))

        for clip in source_clips:
            source_clip_name = str(clip.get("clip_name", "clip"))
            rgb_path = Path(str(clip.get("video_path", "")))
            depth_val = clip.get("depth_video_path")
            depth_path = Path(str(depth_val)) if depth_val else None

            if not rgb_path.exists():
                logger.warning("Skipping augmentation for missing clip: %s", rgb_path)
                failed += config.robosplat_scan.num_augmented_clips_per_input
                continue

            for augment_idx in range(config.robosplat_scan.num_augmented_clips_per_input):
                try:
                    out = augment_scan_only_clip(
                        video_path=rgb_path,
                        depth_video_path=depth_path if depth_path and depth_path.exists() else None,
                        output_dir=stage_dir,
                        source_clip_name=source_clip_name,
                        augment_index=augment_idx,
                        config=config.robosplat_scan,
                    )
                    manifest_clips.append(
                        {
                            "clip_name": out.clip_name,
                            "path_type": clip.get("path_type", "augmented"),
                            "clip_index": clip.get("clip_index", -1),
                            "num_frames": clip.get("num_frames"),
                            "resolution": clip.get("resolution"),
                            "fps": clip.get("fps"),
                            "video_path": str(out.output_video_path),
                            "depth_video_path": (
                                str(out.output_depth_video_path)
                                if out.output_depth_video_path
                                else ""
                            ),
                            "source_clip_name": out.source_clip_name,
                            "source_video_path": str(out.source_video_path),
                            "source_depth_video_path": (
                                str(out.source_depth_video_path)
                                if out.source_depth_video_path
                                else ""
                            ),
                            "augment_ops": out.augment_ops,
                            "augmentation_type": "robosplat_scan_only",
                        }
                    )
                    augmented += 1
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "Failed scan-only augmentation for clip=%s idx=%d: %s",
                        source_clip_name,
                        augment_idx,
                        exc,
                    )

        manifest_path = stage_dir / "augmented_manifest.json"
        write_json(
            {
                "facility": facility.name,
                "source_manifest": str(source_manifest_path),
                "augmentation_type": "robosplat_scan_only",
                "num_source_clips": len(source_clips),
                "num_augmented_clips": augmented,
                "num_total_clips": len(manifest_clips),
                "clips": manifest_clips,
            },
            manifest_path,
        )

        status = "success" if augmented > 0 else "failed"
        return StageResult(
            stage_name=self.name,
            status=status,
            elapsed_seconds=0,
            outputs={
                "augment_dir": str(stage_dir),
                "manifest_path": str(manifest_path),
            },
            metrics={
                "num_source_clips": len(source_clips),
                "num_augmented_clips": augmented,
                "num_failed": failed,
                "num_total_clips": len(manifest_clips),
            },
            detail=(
                f"Generated {augmented} augmented clips from {len(source_clips)} sources"
                if status == "success"
                else "No augmented clips generated"
            ),
        )


def _resolve_source_manifest(work_dir: Path) -> Path | None:
    for candidate in [
        work_dir / "gemini_polish" / "polished_manifest.json",
        work_dir / "robot_composite" / "composited_manifest.json",
        work_dir / "renders" / "render_manifest.json",
    ]:
        if candidate.exists():
            return candidate
    return None
