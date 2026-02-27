"""Stage 1c: Optional Gemini image editing polish for composited clips."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..synthetic.gemini_image_polish import polish_clip_with_gemini
from .base import PipelineStage

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
        del facility, previous_results
        if not config.gemini_polish.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="gemini_polish.enabled=false",
            )

        source_manifest_path = _resolve_source_manifest(work_dir)
        if source_manifest_path is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="No source manifest found for Gemini polish. Run Stage 1 or Stage 1b first.",
            )
        source_manifest = read_json(source_manifest_path)

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
            },
            metrics={
                "num_input_clips": len(source_manifest.get("clips", [])),
                "num_polished_clips": len(polished),
            },
        )


def _resolve_source_manifest(work_dir: Path) -> Path | None:
    for candidate in [
        work_dir / "robot_composite" / "composited_manifest.json",
        work_dir / "renders" / "render_manifest.json",
    ]:
        if candidate.exists():
            return candidate
    return None
