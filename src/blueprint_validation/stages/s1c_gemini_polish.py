"""Stage 1c: Optional Gemini image editing polish for composited clips."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..common import StageResult, get_logger, write_json
from ..config import FacilityConfig, ValidationConfig
from ..synthetic.gemini_image_polish import polish_clip_with_gemini
from ..validation import load_and_validate_manifest
from .manifest_resolution import ManifestCandidate, ManifestSource, resolve_manifest_source
from .base import PipelineStage
from .render_backend import active_render_backend

logger = get_logger("stages.s1c_gemini_polish")


class GeminiPolishStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1c_gemini_polish"

    @property
    def description(self) -> str:
        return "Optional Gemini image editing polish on robot-composited clips"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        if active_render_backend(config, facility, previous_results) != "gsplat":
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail=(
                    "Skipped by render backend: isaac_scene already uses simulator-native robot "
                    "imagery."
                ),
            )
        del facility
        if not config.gemini_polish.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="gemini_polish.enabled=false",
            )

        source = _resolve_source_manifest(work_dir, previous_results)
        if source is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No source manifest found for Gemini polish. Run Stage 1 or Stage 1b first.",
            )
        source_manifest = load_and_validate_manifest(
            source.source_manifest_path,
            manifest_type="stage1_source",
            require_existing_paths=True,
        )

        out_dir = work_dir / "gemini_polish"
        out_dir.mkdir(parents=True, exist_ok=True)
        polished: List[dict] = []
        stats: List[dict] = []

        for clip in source_manifest.get("clips", []):
            in_video = Path(clip["video_path"])
            if not in_video.exists():
                continue
            out_video = out_dir / f"{in_video.stem}_polished.mp4"
            clip_stats = polish_clip_with_gemini(
                input_video=in_video,
                output_video=out_video,
                model=config.gemini_polish.model,
                api_key_env=config.gemini_polish.api_key_env,
                prompt=config.gemini_polish.prompt,
                sample_every_n_frames=config.gemini_polish.sample_every_n_frames,
            )
            updated = dict(clip)
            updated["video_path"] = str(out_video)
            polished.append(updated)
            stats.append({"clip_name": clip["clip_name"], **clip_stats})

        manifest = dict(source_manifest)
        manifest["clips"] = polished
        manifest["num_clips"] = len(polished)
        manifest["gemini_polish_stats"] = stats
        manifest_path = out_dir / "polished_manifest.json"
        write_json(manifest, manifest_path)

        return StageResult(
            stage_name=self.name,
            status="success" if polished else "failed",
            elapsed_seconds=0,
            outputs={
                "polish_dir": str(out_dir),
                "manifest_path": str(manifest_path),
                **source.to_metadata(),
            },
            metrics={
                "num_input_clips": len(source_manifest.get("clips", [])),
                "num_polished_clips": len(polished),
                **source.to_metadata(),
            },
        )


def _resolve_source_manifest(
    work_dir: Path,
    previous_results: Dict[str, StageResult] | None = None,
) -> ManifestSource | None:
    return resolve_manifest_source(
        work_dir=work_dir,
        previous_results=previous_results or {},
        candidates=[
            ManifestCandidate(
                stage_name="s1b_robot_composite",
                manifest_relpath=Path("robot_composite/composited_manifest.json"),
            ),
            ManifestCandidate(
                stage_name="s1_render",
                manifest_relpath=Path("renders/render_manifest.json"),
            ),
        ],
    )
