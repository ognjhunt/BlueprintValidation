"""Stage 2: Cosmos Transfer 2.5 enrichment of rendered clips."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..enrichment.cosmos_runner import enrich_clip
from ..enrichment.variant_specs import get_variants
from ..warmup import load_cached_variants
from .manifest_resolution import ManifestCandidate, ManifestSource, resolve_manifest_source
from .base import PipelineStage

logger = get_logger("stages.s2_enrich")

# Leave empty by default. Opt in only after direct verification that a facility's
# depth control is inverted relative to its own RGB/depth render inputs.
_FORCE_ROTATE_180_DEPTH_FACILITIES: set[str] = set()


class EnrichStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s2_enrich"

    @property
    def description(self) -> str:
        return "Enrich rendered clips with Cosmos Transfer 2.5 visual variants"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        enrich_dir = work_dir / "enriched"
        enrich_dir.mkdir(parents=True, exist_ok=True)

        # Load render manifest
        source = _resolve_render_manifest_source(work_dir, previous_results)
        if source is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=(
                    "Render/composite/polish manifest not found. Run Stage 1 first "
                    "(and Stage 1b/1c if enabled)."
                ),
            )

        render_manifest = read_json(source.source_manifest_path)
        sample_context_frame_index = None

        # Check for warmup-cached variant prompts before calling Gemini
        cached_variants = load_cached_variants(work_dir)
        if cached_variants:
            logger.info("Using %d cached variant prompts from warmup", len(cached_variants))
            variants = cached_variants
        else:
            # Extract a sample frame for dynamic variant generation
            sample_frame_path, sample_context_frame_index = _extract_sample_frame(
                render_manifest,
                work_dir,
                context_frame_index=config.enrich.context_frame_index,
            )
            variants = get_variants(
                custom_variants=config.enrich.variants or None,
                dynamic=config.enrich.dynamic_variants,
                dynamic_model=config.enrich.dynamic_variants_model,
                sample_frame_path=sample_frame_path,
                num_variants=config.enrich.num_variants_per_render,
                facility_description=facility.description,
                allow_dynamic_fallback=config.enrich.allow_dynamic_variant_fallback,
            )

        # Limit variants to configured count
        variants = variants[: config.enrich.num_variants_per_render]

        manifest_entries: List[Dict] = []
        total_enriched = 0
        total_generated = 0
        total_failed = 0
        total_rejected_anchor_similarity = 0
        accepted_ssim_scores: List[float] = []
        ssim_gate_threshold = float(config.enrich.min_frame0_ssim)

        for clip_entry in render_manifest["clips"]:
            video_path = Path(clip_entry["video_path"])
            depth_path = Path(clip_entry["depth_video_path"])
            clip_name = clip_entry["clip_name"]

            if not video_path.exists():
                logger.warning("Video not found: %s", video_path)
                continue

            needs_anchor_frame = (
                config.enrich.context_frame_index is not None or ssim_gate_threshold > 0
            )
            anchor_frame = None
            context_frame_index = None
            total_frames = None
            if needs_anchor_frame:
                anchor_frame, context_frame_index, total_frames = _read_video_frame(
                    video_path=video_path,
                    preferred_index=config.enrich.context_frame_index,
                )
                if anchor_frame is None:
                    logger.warning("Could not read anchor frame from clip: %s", video_path)
                    continue

            depth_for_cosmos = _maybe_prepare_depth_control(
                facility_name=facility.name,
                clip_name=clip_name,
                depth_path=depth_path if depth_path.exists() else None,
                enrich_dir=enrich_dir,
            )

            outputs = enrich_clip(
                video_path=video_path,
                depth_path=depth_for_cosmos,
                variants=variants,
                output_dir=enrich_dir,
                clip_name=clip_name,
                config=config.enrich,
                context_frame_index=context_frame_index,
            )

            expected = len(variants)
            accepted_for_clip = 0
            total_generated += len(outputs)
            for out in outputs:
                frame0_ssim = None
                if ssim_gate_threshold > 0:
                    frame0_ssim = _compute_frame0_ssim(anchor_frame, out.output_video_path)
                    if frame0_ssim is None or frame0_ssim < ssim_gate_threshold:
                        total_rejected_anchor_similarity += 1
                        if config.enrich.delete_rejected_outputs:
                            try:
                                out.output_video_path.unlink(missing_ok=True)
                            except OSError:
                                logger.debug(
                                    "Failed deleting rejected output: %s",
                                    out.output_video_path,
                                    exc_info=True,
                                )
                        logger.info(
                            "Rejected enrich output by frame-0 SSIM gate for %s/%s: "
                            "ssim=%s threshold=%.3f",
                            clip_name,
                            out.variant_name,
                            "n/a" if frame0_ssim is None else f"{frame0_ssim:.4f}",
                            ssim_gate_threshold,
                        )
                        continue
                if frame0_ssim is not None:
                    accepted_ssim_scores.append(frame0_ssim)

                manifest_entries.append(
                    {
                        "clip_name": clip_name,
                        "variant_name": out.variant_name,
                        "prompt": out.prompt,
                        "output_video_path": str(out.output_video_path),
                        "input_video_path": str(out.input_video_path),
                        "context_frame_index": context_frame_index,
                        "input_total_frames": total_frames,
                        "frame0_ssim": frame0_ssim,
                    }
                )
                total_enriched += 1
                accepted_for_clip += 1

            if accepted_for_clip < expected:
                total_failed += expected - accepted_for_clip

        # Write enriched manifest
        manifest_path = enrich_dir / "enriched_manifest.json"
        manifest = {
            "facility": facility.name,
            "num_clips": len(manifest_entries),
            "variants_per_clip": len(variants),
            "variant_names": [v.name for v in variants],
            "context_frame_index": sample_context_frame_index,
            "min_frame0_ssim": ssim_gate_threshold,
            "clips": manifest_entries,
        }
        write_json(manifest, manifest_path)

        return StageResult(
            stage_name=self.name,
            status="success" if total_enriched > 0 else "failed",
            elapsed_seconds=0,
            outputs={
                "enrich_dir": str(enrich_dir),
                "manifest_path": str(manifest_path),
                "num_enriched": total_enriched,
                "num_generated": total_generated,
                "num_rejected_anchor_similarity": total_rejected_anchor_similarity,
                **source.to_metadata(),
            },
            metrics={
                "num_enriched_clips": total_enriched,
                "num_generated": total_generated,
                "num_failed": total_failed,
                "num_rejected_anchor_similarity": total_rejected_anchor_similarity,
                "min_frame0_ssim_threshold": ssim_gate_threshold,
                "mean_frame0_ssim": (
                    round(sum(accepted_ssim_scores) / len(accepted_ssim_scores), 4)
                    if accepted_ssim_scores
                    else None
                ),
                "variants_used": [v.name for v in variants],
                **source.to_metadata(),
            },
        )


def _resolve_render_manifest(work_dir: Path) -> Path | None:
    source = _resolve_render_manifest_source(work_dir, previous_results={})
    return source.source_manifest_path if source else None


def _resolve_render_manifest_source(
    work_dir: Path,
    previous_results: Dict[str, StageResult] | None = None,
) -> ManifestSource | None:
    return resolve_manifest_source(
        work_dir=work_dir,
        previous_results=previous_results or {},
        candidates=[
            ManifestCandidate(
                stage_name="s1e_splatsim_interaction",
                manifest_relpath=Path("splatsim/interaction_manifest.json"),
            ),
            ManifestCandidate(
                stage_name="s1d_gaussian_augment",
                manifest_relpath=Path("gaussian_augment/augmented_manifest.json"),
            ),
            ManifestCandidate(
                stage_name="s1c_gemini_polish",
                manifest_relpath=Path("gemini_polish/polished_manifest.json"),
            ),
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


def _extract_sample_frame(
    manifest: dict,
    work_dir: Path,
    context_frame_index: int | None = None,
) -> Tuple[Path | None, int | None]:
    """Extract a single frame from the first clip for dynamic variant generation."""
    clips = manifest.get("clips", [])
    if not clips:
        return None, None

    video_path = Path(clips[0].get("video_path", ""))
    if not video_path.exists():
        return None, None

    try:
        import cv2  # noqa: F401

        frame, resolved_index, _ = _read_video_frame(
            video_path=video_path,
            preferred_index=context_frame_index,
        )
        if frame is None:
            return None, None

        sample_path = work_dir / "enriched" / "_sample_frame.png"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_path), frame)
        return sample_path, resolved_index
    except Exception:
        logger.debug("Failed to extract sample frame for dynamic variants", exc_info=True)
        return None, None


def _read_video_frame(
    video_path: Path,
    preferred_index: int | None = None,
) -> Tuple[object | None, int | None, int]:
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


def _compute_frame0_ssim(reference_frame: object, generated_video_path: Path) -> float | None:
    """Compute SSIM between the reference anchor frame and generated frame 0."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(generated_video_path))
    ok, generated_frame = cap.read()
    cap.release()
    if not ok or generated_frame is None:
        return None

    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    gen_gray = cv2.cvtColor(generated_frame, cv2.COLOR_BGR2GRAY)
    if ref_gray.shape != gen_gray.shape:
        gen_gray = cv2.resize(gen_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    ref = ref_gray.astype(np.float64)
    gen = gen_gray.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_ref = cv2.GaussianBlur(ref, (11, 11), 1.5)
    mu_gen = cv2.GaussianBlur(gen, (11, 11), 1.5)
    mu_ref_sq = mu_ref * mu_ref
    mu_gen_sq = mu_gen * mu_gen
    mu_ref_gen = mu_ref * mu_gen

    sigma_ref_sq = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu_ref_sq
    sigma_gen_sq = cv2.GaussianBlur(gen * gen, (11, 11), 1.5) - mu_gen_sq
    sigma_ref_gen = cv2.GaussianBlur(ref * gen, (11, 11), 1.5) - mu_ref_gen

    denominator = (mu_ref_sq + mu_gen_sq + c1) * (sigma_ref_sq + sigma_gen_sq + c2)
    numerator = (2 * mu_ref_gen + c1) * (2 * sigma_ref_gen + c2)
    ssim_map = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return float(ssim_map.mean())


def _maybe_prepare_depth_control(
    facility_name: str,
    clip_name: str,
    depth_path: Path | None,
    enrich_dir: Path,
) -> Path | None:
    """Apply scene-specific depth-control fixes before Cosmos inference."""
    if depth_path is None:
        return None
    if facility_name not in _FORCE_ROTATE_180_DEPTH_FACILITIES:
        return depth_path
    try:
        fixed_dir = enrich_dir / "_depth_control_fixed"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        fixed_path = fixed_dir / f"{clip_name}_depth_rot180.mp4"
        if fixed_path.exists() and fixed_path.stat().st_mtime >= depth_path.stat().st_mtime:
            return fixed_path
        _rotate_video_180(depth_path, fixed_path)
        logger.info(
            "Applied scene-specific depth rotation (180°) for %s clip %s: %s",
            facility_name,
            clip_name,
            fixed_path,
        )
        return fixed_path
    except Exception:
        logger.warning(
            "Depth-control fix failed for %s clip %s; using original depth video",
            facility_name,
            clip_name,
            exc_info=True,
        )
        return depth_path


def _rotate_video_180(input_path: Path, output_path: Path) -> None:
    """Rotate a depth video by 180 degrees."""
    import cv2

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open depth video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid depth video dimensions for {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
        isColor=False,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {output_path}")

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
            writer.write(rotated)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames were read from depth video: {input_path}")
