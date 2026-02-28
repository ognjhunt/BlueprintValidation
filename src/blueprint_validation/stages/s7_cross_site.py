"""Stage 7: Cross-site discrimination test."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig, VLMJudgeConfig
from ..evaluation.cross_site import compute_cross_site_metrics
from ..evaluation.vlm_judge import classify_facility
from .base import PipelineStage

logger = get_logger("stages.s7_cross_site")


class CrossSiteStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s7_cross_site"

    @property
    def description(self) -> str:
        return "Cross-site discrimination: verify models generate facility-specific imagery"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        cross_site_dir = work_dir / "cross_site"
        cross_site_dir.mkdir(parents=True, exist_ok=True)

        facility_ids = list(config.facilities.keys())
        if len(facility_ids) < 2:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="Cross-site test requires at least 2 facilities.",
            )

        # Build facility descriptions map
        facility_descriptions = {fid: config.facilities[fid].description for fid in facility_ids}

        vlm_config = VLMJudgeConfig(
            model=config.eval_crosssite.vlm_model,
            api_key_env=config.eval_policy.vlm_judge.api_key_env,
        )

        classifications: List[Dict] = []
        lpips_inter: List[float] = []
        lpips_intra: List[float] = []

        for source_fid in facility_ids:
            # Find enriched clips for this facility
            fac_work_dir = work_dir / source_fid if (work_dir / source_fid).exists() else work_dir
            enriched_manifest_path = fac_work_dir / "enriched" / "enriched_manifest.json"

            if not enriched_manifest_path.exists():
                logger.warning("No enriched manifest for facility %s", source_fid)
                continue

            manifest = read_json(enriched_manifest_path)
            clips = manifest.get("clips", [])

            # Sample clips
            num_clips = min(config.eval_crosssite.num_clips_per_model, len(clips))
            rng = np.random.default_rng(seed=hash(source_fid) % 2**32)
            sampled_indices = rng.choice(len(clips), size=num_clips, replace=False)

            for idx in sampled_indices:
                clip = clips[idx]
                video_path = Path(clip["output_video_path"])

                if not video_path.exists():
                    continue

                try:
                    result = classify_facility(
                        video_path=video_path,
                        facility_descriptions=facility_descriptions,
                        config=vlm_config,
                    )
                    classifications.append(
                        {
                            "source_model": source_fid,
                            "predicted_facility": result["predicted_facility"],
                            "confidence": result["confidence"],
                            "reasoning": result["reasoning"],
                            "video_path": str(video_path),
                        }
                    )
                except Exception as e:
                    logger.warning("Classification failed for %s: %s", video_path.name, e)

        # Compute LPIPS distances (simplified — using frame-level comparison)
        # In practice, extract frames from each model's outputs and compute pairwise LPIPS
        _compute_lpips_distances(config, work_dir, facility_ids, lpips_inter, lpips_intra)

        # Compute metrics
        cs_metrics = compute_cross_site_metrics(
            classifications=classifications,
            lpips_inter=lpips_inter,
            lpips_intra=lpips_intra,
            facility_ids=facility_ids,
        )

        write_json(
            {
                "metrics": cs_metrics.to_dict(),
                "classifications": classifications,
            },
            cross_site_dir / "cross_site.json",
        )

        return StageResult(
            stage_name=self.name,
            status="success" if classifications else "failed",
            elapsed_seconds=0,
            outputs={"cross_site_dir": str(cross_site_dir)},
            metrics=cs_metrics.to_dict(),
        )


def _compute_lpips_distances(
    config: ValidationConfig,
    work_dir: Path,
    facility_ids: List[str],
    lpips_inter: List[float],
    lpips_intra: List[float],
) -> None:
    """Compute LPIPS distances between facilities for discrimination analysis."""
    try:
        from ..evaluation.metrics import compute_lpips_batch
        import cv2

        facility_frames: Dict[str, List[np.ndarray]] = {}

        for fid in facility_ids:
            fac_dir = work_dir / fid if (work_dir / fid).exists() else work_dir
            render_manifest = fac_dir / "renders" / "render_manifest.json"
            if not render_manifest.exists():
                continue

            manifest = read_json(render_manifest)
            frames = []
            for clip in manifest.get("clips", [])[:5]:  # Sample 5 clips
                vpath = Path(clip["video_path"])
                if vpath.exists():
                    cap = cv2.VideoCapture(str(vpath))
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            facility_frames[fid] = frames

        # Compute pairwise LPIPS
        fids = list(facility_frames.keys())
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                f1 = facility_frames[fids[i]]
                f2 = facility_frames[fids[j]]
                min_len = min(len(f1), len(f2))
                if min_len > 0:
                    vals = compute_lpips_batch(f1[:min_len], f2[:min_len])
                    lpips_inter.extend(vals)

            # Intra-facility LPIPS (between clips from same facility)
            frames = facility_frames[fids[i]]
            if len(frames) >= 2:
                half = len(frames) // 2
                vals = compute_lpips_batch(frames[:half], frames[half : half + half])
                lpips_intra.extend(vals)

    except Exception as e:
        logger.warning("LPIPS distance computation failed: %s", e)
