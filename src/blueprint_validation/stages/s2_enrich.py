"""Stage 2: Cosmos Transfer 2.5 enrichment of rendered clips."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Dict, List, Tuple

from ..common import (
    StageResult,
    get_logger,
    sanitize_filename_component_with_hash,
    write_json,
)
from ..config import FacilityConfig, ValidationConfig
from ..enrichment.cosmos_runner import enrich_clip
from ..enrichment.scene_index import build_scene_index, query_nearest_context_candidates
from ..enrichment.variant_specs import get_variants
from ..evaluation.camera_quality import (
    analyze_target_visibility as _shared_analyze_target_visibility,
    estimate_clip_blur_score as _shared_estimate_clip_blur_score,
    project_target_to_camera_path as _shared_project_target_to_camera_path,
    resolve_center_band_bounds as _shared_resolve_center_band_bounds,
)
from ..evaluation.reasoning_conflicts import has_reasoning_conflict
from ..evaluation.task_start_selector import build_task_start_assignments
from ..evaluation.vlm_judge import score_stage2_enriched_clip
from ..evaluation.video_orientation import (
    apply_video_orientation_fix as _shared_apply_video_orientation_fix,
    normalize_video_orientation_fix as _shared_normalize_video_orientation_fix,
    transform_video_frame as _shared_transform_video_frame,
    transform_video_orientation as _shared_transform_video_orientation,
)
from ..video_io import ensure_h264_video, open_mp4_writer
from ..warmup import load_cached_variants
from ..validation import ManifestValidationError, load_and_validate_manifest
from .manifest_resolution import ManifestCandidate, ManifestSource, resolve_manifest_source
from .base import PipelineStage

logger = get_logger("stages.s2_enrich")

# Leave empty by default. Opt in only after direct verification that a facility's
# depth control is inverted relative to its own RGB/depth render inputs.
_FORCE_ROTATE_180_DEPTH_FACILITIES: set[str] = set()
_VISUAL_COLLAPSE_MIN_INTERFRAME_DELTA = 2.0


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return text


def _derive_expected_focus_text(path_type: str, path_context: dict | None) -> str:
    path_type_norm = str(path_type or "").strip().lower()
    ctx = path_context if isinstance(path_context, dict) else {}
    target_label = _clean_text(ctx.get("target_label"))
    target_instance = _clean_text(ctx.get("target_instance_id"))
    target_role = _clean_text(ctx.get("target_role"))
    if target_label:
        if target_role:
            return (
                f"Primary target focus: keep {target_label} ({target_role}) centered and clearly visible."
            )
        return f"Primary target focus: keep {target_label} centered and clearly visible."
    if target_instance:
        return (
            "Primary target focus: keep the task target instance "
            f"{target_instance} centered and clearly visible."
        )
    if path_type_norm == "manipulation":
        return "Primary target focus: keep the manipulation target centered and clearly visible."
    if path_type_norm == "orbit":
        return "Overview focus: maintain stable scene context with the task-relevant area centered."
    if path_type_norm == "sweep":
        return "Coverage focus: sweep across the task workspace while keeping key objects visible."
    return "Primary target focus: keep the task-relevant object/region centered and clearly visible."


def _resolve_expected_focus_text(clip_entry: dict) -> tuple[str | None, str]:
    source_text = _clean_text(clip_entry.get("expected_focus_text"))
    if source_text:
        return source_text, source_text
    path_type = str(clip_entry.get("path_type", "") or "")
    path_context = clip_entry.get("path_context", {})
    return None, _derive_expected_focus_text(path_type, path_context if isinstance(path_context, dict) else {})


def _resolve_retry_context_index(
    *,
    base_index: int | None,
    total_frames: int,
    stride: int,
    retry_number: int,
) -> int | None:
    if base_index is None or total_frames <= 0:
        return base_index
    stride = max(1, int(stride))
    retry_number = max(1, int(retry_number))
    if retry_number % 2 == 1:
        candidate = base_index - stride
    else:
        candidate = base_index + stride
    return int(max(0, min(total_frames - 1, candidate)))


@dataclass(frozen=True)
class CoverageGateResult:
    """Result of Stage-1 manipulation coverage checks run before Stage 2."""

    passed: bool
    detail: str
    metrics: Dict[str, object]


@dataclass(frozen=True)
class PreparedCosmosInput:
    """Prepared (possibly trimmed) video/depth inputs for Cosmos."""

    video_path: Path
    depth_path: Path | None
    preferred_context_frame_index: int | None
    input_total_frames: int
    input_trimmed: bool
    input_trim_start_frame: int | None
    input_trim_num_frames: int | None


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

        def _failed_result(
            *,
            detail: str,
            error_code: str,
            outputs: Dict[str, object] | None = None,
            metrics: Dict[str, object] | None = None,
        ) -> StageResult:
            out = dict(outputs or {})
            met = dict(metrics or {})
            out["error_code"] = error_code
            met["error_code"] = error_code
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=detail,
                outputs=out,
                metrics=met,
            )

        # Load render manifest
        source = _resolve_render_manifest_source(work_dir, previous_results)
        if source is None:
            return _failed_result(
                detail=(
                    "Render/composite/polish manifest not found. Run Stage 1 first "
                    "(and Stage 1b/1c if enabled)."
                ),
                error_code="s2_source_manifest_missing",
            )

        try:
            render_manifest = load_and_validate_manifest(
                source.source_manifest_path,
                manifest_type="stage1_source",
                require_existing_paths=True,
            )
        except ManifestValidationError as exc:
            return _failed_result(
                detail=f"Invalid source manifest for Stage 2 enrich: {exc}",
                error_code="s2_source_manifest_invalid",
                outputs=source.to_metadata(),
                metrics=source.to_metadata(),
            )
        sample_context_frame_index = None
        coverage_gate_result = _evaluate_stage1_coverage_gate(render_manifest, config)
        coverage_outputs: Dict[str, object] = {}
        coverage_metrics: Dict[str, object] = {}
        if coverage_gate_result is not None:
            coverage_outputs["coverage_gate_passed"] = coverage_gate_result.passed
            coverage_metrics.update(coverage_gate_result.metrics)
            if not coverage_gate_result.passed:
                return _failed_result(
                    detail=coverage_gate_result.detail,
                    error_code="s2_coverage_gate_failed",
                    outputs={
                        **source.to_metadata(),
                        **coverage_outputs,
                    },
                    metrics={
                        **source.to_metadata(),
                        **coverage_metrics,
                    },
                )

        source_clips, clip_selection_meta = _select_source_clips(
            render_manifest=render_manifest,
            config=config,
            facility=facility,
        )
        if not source_clips:
            selection_mode = str(clip_selection_meta.get("selection_mode") or "").strip().lower()
            fallback_reason = str(clip_selection_meta.get("fallback") or "").strip()
            fail_closed = bool(clip_selection_meta.get("fail_closed"))
            detail = "No source clips available for Stage 2 enrichment"
            if selection_mode == "task_targeted" and fallback_reason and fail_closed:
                detail = (
                    "Task-targeted source selection failed closed: "
                    f"{fallback_reason}. Provide enrich.source_clip_task + task_hints, "
                    "or set enrich.source_clip_selection_fail_closed=false to allow fallback."
                )
            elif selection_mode == "explicit" and fallback_reason and fail_closed:
                detail = (
                    "Explicit source selection failed closed: "
                    f"{fallback_reason}. Set enrich.source_clip_name to a valid Stage-1 clip_name, "
                    "or switch enrich.source_clip_selection_mode."
                )
            selection_error_code = "s2_source_selection_empty"
            if selection_mode == "task_targeted" and fail_closed and fallback_reason:
                selection_error_code = "s2_source_selection_task_targeted_fail_closed"
            elif selection_mode == "explicit" and fail_closed and fallback_reason:
                selection_error_code = "s2_source_selection_explicit_fail_closed"
            return _failed_result(
                detail=detail,
                error_code=selection_error_code,
                outputs={
                    **source.to_metadata(),
                    "source_clip_selection_mode": clip_selection_meta.get("selection_mode"),
                    "source_clip_selection_fallback": clip_selection_meta.get("fallback"),
                    "source_clip_selection_fail_closed": bool(clip_selection_meta.get("fail_closed")),
                },
                metrics={
                    **source.to_metadata(),
                    "num_selected_source_clips": 0,
                    "source_clip_selection_mode": clip_selection_meta.get("selection_mode"),
                    "source_clip_selection_fallback": clip_selection_meta.get("fallback"),
                    "source_clip_selection_fail_closed": bool(clip_selection_meta.get("fail_closed")),
                },
            )
        selected_clip_target_distribution = _summarize_target_distribution(source_clips)
        min_source_clips = max(0, int(config.enrich.min_source_clips))
        if min_source_clips > 0 and len(source_clips) < min_source_clips:
            return _failed_result(
                detail=(
                    "Insufficient selected source clips for Stage 2 strict gate: "
                    f"{len(source_clips)} < min_source_clips={min_source_clips}."
                ),
                error_code="s2_min_source_clips_not_met",
                outputs={
                    **source.to_metadata(),
                    "num_selected_source_clips": len(source_clips),
                    "selected_source_clips": [str(c.get("clip_name", "")) for c in source_clips],
                    "selected_source_target_distribution": selected_clip_target_distribution,
                    **coverage_outputs,
                },
                metrics={
                    **source.to_metadata(),
                    "num_selected_source_clips": len(source_clips),
                    "selected_source_target_distribution": selected_clip_target_distribution,
                    "source_clip_selection_mode": clip_selection_meta.get("selection_mode"),
                    "source_clip_selection_fallback": clip_selection_meta.get("fallback"),
                    "source_clip_selection_fail_closed": bool(clip_selection_meta.get("fail_closed")),
                    "min_source_clips": min_source_clips,
                    **coverage_metrics,
                },
            )

        scene_index_payload = None
        scene_index_path: Path | None = None
        if bool(config.enrich.scene_index_enabled) and bool(config.enrich.multi_view_context_enabled):
            try:
                scene_index_path = enrich_dir / "_scene_index" / "scene_index.json"
                scene_index_payload = build_scene_index(
                    render_manifest=render_manifest,
                    output_path=scene_index_path,
                    sample_every_n_frames=max(
                        1, int(config.enrich.scene_index_sample_every_n_frames)
                    ),
                )
            except Exception:
                logger.warning("Failed building scene index; continuing without retrieval", exc_info=True)
                scene_index_payload = None
                scene_index_path = None

        # Check for warmup-cached variant prompts before calling Gemini
        cached_variants = load_cached_variants(work_dir)
        if cached_variants:
            logger.info("Using %d cached variant prompts from warmup", len(cached_variants))
            variants = cached_variants
        else:
            # Extract a sample frame for dynamic variant generation.
            # Use selected source clips so debug/task-targeted runs anchor prompts correctly.
            sample_frame_path, sample_context_frame_index = _extract_sample_frame(
                clips=source_clips,
                work_dir=work_dir,
                context_frame_index=config.enrich.context_frame_index,
                facility=facility,
                enrich_dir=enrich_dir,
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
        total_rejected_blur = 0
        total_rejected_green_cast = 0
        total_rejected_low_motion = 0
        total_visual_gate_evaluated = 0
        num_vlm_quality_evaluated = 0
        num_vlm_quality_failures = 0
        num_vlm_quality_retries = 0
        num_vlm_quality_recovered = 0
        num_vlm_quality_api_failures = 0
        num_vlm_quality_parse_failures = 0
        num_vlm_quality_reasoning_conflicts = 0
        accepted_ssim_scores: List[float] = []
        accepted_blur_scores: List[float] = []
        accepted_green_ratios: List[float] = []
        accepted_interframe_deltas: List[float] = []
        total_trimmed_inputs = 0
        total_orientation_fixed_clips = 0
        total_multi_view_context_clips = 0
        total_scene_index_retrievals = 0
        context_mode_counts = {"target_centered": 0, "fixed": 0, "deterministic": 0}
        selected_center_scores: List[float] = []
        ssim_gate_threshold = float(config.enrich.min_frame0_ssim)
        min_valid_outputs = max(0, int(config.enrich.min_valid_outputs))
        visual_collapse_gate_enabled = bool(config.enrich.enable_visual_collapse_gate)
        max_blur_reject_rate = float(config.enrich.max_blur_reject_rate)
        green_frame_ratio_max = float(config.enrich.green_frame_ratio_max)
        blur_laplacian_min = float(config.render.stage1_coverage_blur_laplacian_min)
        vlm_quality_gate_enabled = bool(config.enrich.vlm_quality_gate_enabled)
        vlm_quality_fail_closed = bool(config.enrich.vlm_quality_fail_closed)
        vlm_quality_autoretry_enabled = bool(config.enrich.vlm_quality_autoretry_enabled)
        vlm_quality_max_regen_attempts = max(0, int(config.enrich.vlm_quality_max_regen_attempts))
        vlm_quality_min_task_score = float(config.enrich.vlm_quality_min_task_score)
        vlm_quality_min_visual_score = float(config.enrich.vlm_quality_min_visual_score)
        vlm_quality_min_spatial_score = float(config.enrich.vlm_quality_min_spatial_score)
        vlm_quality_require_reasoning_consistency = bool(
            config.enrich.vlm_quality_require_reasoning_consistency
        )
        vlm_quality_retry_context_frame_stride = max(
            1, int(config.enrich.vlm_quality_retry_context_frame_stride)
        )
        vlm_quality_disable_depth_on_final_retry = bool(
            config.enrich.vlm_quality_disable_depth_on_final_retry
        )
        vlm_selected_fps = (
            float(config.eval_policy.vlm_judge.video_metadata_fps)
            if float(config.eval_policy.vlm_judge.video_metadata_fps) > 0.0
            else None
        )
        unresolved_vlm_failures: List[str] = []
        # Stage 2 hard guard: clip must be decodable and non-trivially temporal.
        # World-model-specific minimum frame windows are enforced downstream where required.
        min_required_input_frames = 2

        for clip_entry in source_clips:
            source_video_raw = str(clip_entry.get("video_path", "")).strip()
            if not source_video_raw:
                logger.warning("Skipping clip with missing source video path: %s", clip_entry)
                continue
            source_video_path = Path(source_video_raw)
            source_depth_raw = str(clip_entry.get("depth_video_path", "")).strip()
            source_depth_path = Path(source_depth_raw) if source_depth_raw else None
            clip_name = str(clip_entry.get("clip_name", ""))
            if not clip_name:
                logger.warning("Skipping clip with missing clip_name in render manifest")
                continue
            clip_artifact_name = sanitize_filename_component_with_hash(
                clip_name,
                fallback="clip",
            )
            if not source_video_path.exists():
                logger.warning("Video not found: %s", source_video_path)
                continue

            oriented_video_path, oriented_depth_path, orientation_mode_applied = (
                _resolve_oriented_inputs_for_clip(
                    facility=facility,
                    clip_name=clip_name,
                    enrich_dir=enrich_dir,
                    video_path=source_video_path,
                    depth_path=(
                        source_depth_path
                        if source_depth_path is not None and source_depth_path.exists()
                        else None
                    ),
                )
            )
            if oriented_video_path != source_video_path:
                total_orientation_fixed_clips += 1

            (
                preferred_context_index,
                selected_context_mode,
                target_center_score,
            ) = _resolve_clip_context_selection(clip_entry, config)
            context_mode_counts[selected_context_mode] = (
                context_mode_counts.get(selected_context_mode, 0) + 1
            )
            if target_center_score is not None:
                selected_center_scores.append(float(target_center_score))

            prepared = _prepare_cosmos_input(
                video_path=oriented_video_path,
                depth_path=oriented_depth_path,
                clip_name=clip_artifact_name,
                enrich_dir=enrich_dir,
                preferred_context_frame_index=preferred_context_index,
                max_input_frames=int(config.enrich.max_input_frames),
                min_required_frames=min_required_input_frames,
            )
            if prepared is None:
                logger.warning("Failed to prepare Cosmos inputs for clip: %s", clip_name)
                continue
            if prepared.input_trimmed:
                total_trimmed_inputs += 1

            needs_anchor_frame = (
                prepared.preferred_context_frame_index is not None
                or ssim_gate_threshold > 0
                or bool(config.enrich.multi_view_context_enabled)
            )
            anchor_frame = None
            selected_context_frame_index = None
            total_frames = prepared.input_total_frames
            if needs_anchor_frame:
                anchor_frame, selected_context_frame_index, total_frames = _read_video_frame(
                    video_path=prepared.video_path,
                    preferred_index=prepared.preferred_context_frame_index,
                )
                if anchor_frame is None:
                    logger.warning(
                        "Could not read anchor frame from prepared clip: %s",
                        prepared.video_path,
                    )
                    continue

            multi_view_context_indices: List[int] = []
            image_context_path: Path | None = None
            scene_index_retrieval_count = 0
            context_frame_index_for_cosmos = selected_context_frame_index
            if bool(config.enrich.multi_view_context_enabled):
                total_multi_view_context_clips += 1
                if total_frames <= 0:
                    total_frames = _probe_video_frame_count(prepared.video_path)
                anchor_index = (
                    int(selected_context_frame_index)
                    if selected_context_frame_index is not None
                    else _resolve_context_frame_index(
                        total_frames, prepared.preferred_context_frame_index
                    )
                )
                selected_context_frame_index = anchor_index
                multi_view_context_indices = _resolve_multi_view_context_indices(
                    anchor_index=anchor_index,
                    total_frames=total_frames,
                    offsets=[int(v) for v in config.enrich.multi_view_context_offsets],
                )
                context_frames = _read_video_frames(prepared.video_path, multi_view_context_indices)
                if scene_index_payload is not None and int(config.enrich.scene_index_k) > 0:
                    retrieved = query_nearest_context_candidates(
                        scene_index=scene_index_payload,
                        anchor_clip_name=clip_name,
                        anchor_frame_index=anchor_index,
                        k=max(0, int(config.enrich.scene_index_k)),
                    )
                    scene_index_retrieval_count = len(retrieved)
                    total_scene_index_retrievals += scene_index_retrieval_count
                    for item in retrieved:
                        cand_video = Path(str(item.get("video_path", "")))
                        if _normalize_video_orientation_fix(facility.video_orientation_fix) != "none":
                            cand_video, _, _ = _resolve_oriented_inputs_for_clip(
                                facility=facility,
                                clip_name=str(item.get("clip_name", "retrieved")),
                                enrich_dir=enrich_dir,
                                video_path=cand_video,
                                depth_path=None,
                            )
                        cand_frame = _read_video_frame_at_index(
                            cand_video,
                            int(item.get("frame_index", 0)),
                        )
                        if cand_frame is not None:
                            context_frames.append(cand_frame)
                image_context_path = _write_context_montage(
                    frames=context_frames,
                    output_path=(
                        enrich_dir / "_context_montage" / f"{clip_artifact_name}_context.png"
                    ),
                )
                context_frame_index_for_cosmos = None

            source_expected_focus_text, expected_focus_text = _resolve_expected_focus_text(clip_entry)
            expected = len(variants)
            accepted_for_clip = 0

            for variant in variants:
                variant_max_attempts = 1
                if vlm_quality_gate_enabled and vlm_quality_autoretry_enabled:
                    variant_max_attempts += int(vlm_quality_max_regen_attempts)
                variant_resolved = False
                variant_has_vlm_unresolved_failure = False
                variant_last_fail_reason = ""
                variant_last_issue_tags: List[str] = []
                variant_last_reasoning = ""
                variant_last_reasoning_conflict = False
                variant_last_task_score: float | None = None
                variant_last_visual_score: float | None = None
                variant_last_spatial_score: float | None = None

                for attempt_idx in range(variant_max_attempts):
                    retry_active = attempt_idx > 0
                    if retry_active and vlm_quality_gate_enabled:
                        num_vlm_quality_retries += 1

                    attempt_context_index = context_frame_index_for_cosmos
                    attempt_selected_context_frame_index = selected_context_frame_index
                    attempt_image_context_path = image_context_path
                    attempt_multi_view_context_indices = list(multi_view_context_indices)
                    attempt_scene_index_retrieval_count = scene_index_retrieval_count

                    if retry_active:
                        shifted_context_index = _resolve_retry_context_index(
                            base_index=selected_context_frame_index,
                            total_frames=total_frames,
                            stride=vlm_quality_retry_context_frame_stride,
                            retry_number=attempt_idx,
                        )
                        attempt_selected_context_frame_index = shifted_context_index
                        if bool(config.enrich.multi_view_context_enabled):
                            if shifted_context_index is not None and total_frames > 0:
                                attempt_multi_view_context_indices = _resolve_multi_view_context_indices(
                                    anchor_index=int(shifted_context_index),
                                    total_frames=total_frames,
                                    offsets=[int(v) for v in config.enrich.multi_view_context_offsets],
                                )
                                attempt_frames = _read_video_frames(
                                    prepared.video_path,
                                    attempt_multi_view_context_indices,
                                )
                                if attempt_frames:
                                    safe_variant_name = sanitize_filename_component_with_hash(
                                        variant.name, fallback="variant"
                                    )
                                    attempt_image_context_path = _write_context_montage(
                                        frames=attempt_frames,
                                        output_path=(
                                            enrich_dir
                                            / "_context_montage"
                                            / (
                                                f"{clip_artifact_name}_{safe_variant_name}_retry"
                                                f"{attempt_idx:02d}_context.png"
                                            )
                                        ),
                                    )
                            attempt_context_index = None
                            attempt_scene_index_retrieval_count = 0
                        else:
                            attempt_context_index = shifted_context_index

                    attempt_controlnet_inputs = [str(v) for v in config.enrich.controlnet_inputs]
                    attempt_guidance = float(config.enrich.guidance)
                    if retry_active and attempt_idx == (variant_max_attempts - 1):
                        attempt_guidance = float(config.enrich.guidance) * 0.9
                        if vlm_quality_disable_depth_on_final_retry:
                            attempt_controlnet_inputs = [
                                inp
                                for inp in attempt_controlnet_inputs
                                if str(inp).strip().lower() != "depth"
                            ]
                            if not attempt_controlnet_inputs:
                                attempt_controlnet_inputs = ["rgb"]

                    attempt_cfg = replace(
                        config.enrich,
                        controlnet_inputs=attempt_controlnet_inputs,
                        guidance=attempt_guidance,
                    )

                    outputs = enrich_clip(
                        video_path=prepared.video_path,
                        depth_path=prepared.depth_path,
                        variants=[variant],
                        output_dir=enrich_dir,
                        clip_name=clip_name,
                        config=attempt_cfg,
                        context_frame_index=attempt_context_index,
                        image_context_path=attempt_image_context_path,
                        min_output_frames=min_required_input_frames,
                    )
                    total_generated += len(outputs)
                    if not outputs:
                        variant_last_fail_reason = "cosmos_generation_failed"
                        continue

                    out = None
                    for candidate in outputs:
                        if str(getattr(candidate, "variant_name", "")).strip() == str(variant.name):
                            out = candidate
                            break
                    if out is None:
                        out = outputs[0]
                    frame0_ssim = None
                    if ssim_gate_threshold > 0:
                        frame0_ssim = _compute_frame0_ssim(anchor_frame, out.output_video_path)
                        if frame0_ssim is None or frame0_ssim < ssim_gate_threshold:
                            total_rejected_anchor_similarity += 1
                            variant_last_fail_reason = "ssim_reject"
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

                    visual_stats = _analyze_output_visual_quality(out.output_video_path)
                    blur_laplacian_mean = None
                    green_dominance_ratio = None
                    interframe_delta_mean = None
                    if visual_stats is not None:
                        blur_laplacian_mean = visual_stats.get("blur_laplacian_mean")
                        green_dominance_ratio = visual_stats.get("green_dominance_ratio")
                        interframe_delta_mean = visual_stats.get("interframe_delta_mean")

                    if visual_collapse_gate_enabled:
                        total_visual_gate_evaluated += 1
                        reject_reasons: List[str] = []
                        if blur_laplacian_mean is None or float(blur_laplacian_mean) < blur_laplacian_min:
                            total_rejected_blur += 1
                            reject_reasons.append("blur")
                        if (
                            green_dominance_ratio is None
                            or float(green_dominance_ratio) > green_frame_ratio_max
                        ):
                            total_rejected_green_cast += 1
                            reject_reasons.append("green_cast")
                        if (
                            interframe_delta_mean is None
                            or float(interframe_delta_mean) < _VISUAL_COLLAPSE_MIN_INTERFRAME_DELTA
                        ):
                            total_rejected_low_motion += 1
                            reject_reasons.append("low_motion")
                        if reject_reasons:
                            variant_last_fail_reason = "visual_collapse_reject:" + ",".join(reject_reasons)
                            if config.enrich.delete_rejected_outputs:
                                try:
                                    out.output_video_path.unlink(missing_ok=True)
                                except OSError:
                                    logger.debug(
                                        "Failed deleting visual-collapse rejected output: %s",
                                        out.output_video_path,
                                        exc_info=True,
                                    )
                            logger.info(
                                "Rejected enrich output by visual-collapse gate for %s/%s: "
                                "reasons=%s blur_laplacian_mean=%s green_dominance_ratio=%s interframe_delta_mean=%s",
                                clip_name,
                                out.variant_name,
                                ",".join(reject_reasons),
                                "n/a"
                                if blur_laplacian_mean is None
                                else f"{float(blur_laplacian_mean):.4f}",
                                "n/a"
                                if green_dominance_ratio is None
                                else f"{float(green_dominance_ratio):.4f}",
                                "n/a"
                                if interframe_delta_mean is None
                                else f"{float(interframe_delta_mean):.4f}",
                            )
                            continue

                    vlm_quality_passed = True
                    vlm_quality_fail_reason: str | None = None
                    vlm_task_score: float | None = None
                    vlm_visual_score: float | None = None
                    vlm_spatial_score: float | None = None
                    vlm_reasoning = ""
                    vlm_issue_tags: List[str] = []
                    vlm_reasoning_conflict = False
                    vlm_attempts = 0
                    vlm_retries_used = max(0, attempt_idx)

                    if vlm_quality_gate_enabled:
                        vlm_attempts = attempt_idx + 1
                        num_vlm_quality_evaluated += 1
                        try:
                            vlm_score = score_stage2_enriched_clip(
                                video_path=out.output_video_path,
                                expected_focus_text=expected_focus_text,
                                variant_prompt=variant.prompt,
                                config=config.eval_policy.vlm_judge,
                                facility_description=facility.description,
                            )
                            vlm_task_score = float(vlm_score.task_score)
                            vlm_visual_score = float(vlm_score.visual_score)
                            vlm_spatial_score = float(vlm_score.spatial_score)
                            vlm_reasoning = str(vlm_score.reasoning or "")
                            vlm_issue_tags = [str(tag) for tag in vlm_score.issue_tags]
                            vlm_reasoning_conflict = bool(has_reasoning_conflict(vlm_reasoning))
                            if vlm_reasoning_conflict:
                                num_vlm_quality_reasoning_conflicts += 1
                            vlm_quality_passed = (
                                vlm_task_score >= vlm_quality_min_task_score
                                and vlm_visual_score >= vlm_quality_min_visual_score
                                and vlm_spatial_score >= vlm_quality_min_spatial_score
                                and (
                                    not vlm_quality_require_reasoning_consistency
                                    or not vlm_reasoning_conflict
                                )
                            )
                            if not vlm_quality_passed:
                                num_vlm_quality_failures += 1
                                fail_reasons: List[str] = []
                                if vlm_task_score < vlm_quality_min_task_score:
                                    fail_reasons.append("task_score")
                                if vlm_visual_score < vlm_quality_min_visual_score:
                                    fail_reasons.append("visual_score")
                                if vlm_spatial_score < vlm_quality_min_spatial_score:
                                    fail_reasons.append("spatial_score")
                                if (
                                    vlm_quality_require_reasoning_consistency
                                    and vlm_reasoning_conflict
                                ):
                                    fail_reasons.append("reasoning_conflict")
                                vlm_quality_fail_reason = "vlm_quality_fail:" + ",".join(fail_reasons)
                        except Exception as exc:
                            num_vlm_quality_failures += 1
                            num_vlm_quality_api_failures += 1
                            exc_text = str(exc).lower()
                            if "json" in exc_text or "parse" in exc_text:
                                num_vlm_quality_parse_failures += 1
                            vlm_quality_fail_reason = f"vlm_api_failure:{exc}"
                            vlm_quality_passed = not vlm_quality_fail_closed
                            if vlm_quality_fail_closed:
                                variant_has_vlm_unresolved_failure = True

                    if not vlm_quality_passed:
                        variant_last_fail_reason = str(vlm_quality_fail_reason or "vlm_quality_fail")
                        variant_last_issue_tags = list(vlm_issue_tags)
                        variant_last_reasoning = vlm_reasoning
                        variant_last_reasoning_conflict = bool(vlm_reasoning_conflict)
                        variant_last_task_score = vlm_task_score
                        variant_last_visual_score = vlm_visual_score
                        variant_last_spatial_score = vlm_spatial_score
                        if vlm_quality_fail_reason and str(vlm_quality_fail_reason).startswith("vlm_"):
                            variant_has_vlm_unresolved_failure = True
                        if config.enrich.delete_rejected_outputs:
                            try:
                                out.output_video_path.unlink(missing_ok=True)
                            except OSError:
                                logger.debug(
                                    "Failed deleting VLM-rejected output: %s",
                                    out.output_video_path,
                                    exc_info=True,
                                )
                        continue

                    manifest_entries.append(
                        {
                            "clip_name": clip_name,
                            "variant_name": out.variant_name,
                            "prompt": out.prompt,
                            "output_video_path": str(out.output_video_path),
                            "input_video_path": str(out.input_video_path),
                            "source_video_path": str(source_video_path),
                            "source_depth_video_path": source_depth_raw,
                            "orientation_fix": orientation_mode_applied,
                            "context_frame_index": attempt_context_index,
                            "selected_context_frame_index": attempt_selected_context_frame_index,
                            "selected_context_frame_mode": selected_context_mode,
                            "target_center_score": target_center_score,
                            "image_context_path": (
                                str(attempt_image_context_path) if attempt_image_context_path else None
                            ),
                            "multi_view_context_indices": attempt_multi_view_context_indices,
                            "scene_index_retrieval_count": attempt_scene_index_retrieval_count,
                            "input_total_frames": prepared.input_total_frames,
                            "input_trimmed": prepared.input_trimmed,
                            "input_trim_start_frame": prepared.input_trim_start_frame,
                            "input_trim_num_frames": prepared.input_trim_num_frames,
                            "source_clip_selection_mode": clip_selection_meta.get("selection_mode"),
                            "source_clip_selection_fail_closed": bool(
                                clip_selection_meta.get("fail_closed")
                            ),
                            "frame0_ssim": frame0_ssim,
                            "blur_laplacian_mean": blur_laplacian_mean,
                            "green_dominance_ratio": green_dominance_ratio,
                            "interframe_delta_mean": interframe_delta_mean,
                            "source_expected_focus_text": source_expected_focus_text,
                            "expected_focus_text": expected_focus_text,
                            "vlm_quality_task_score": vlm_task_score,
                            "vlm_quality_visual_score": vlm_visual_score,
                            "vlm_quality_spatial_score": vlm_spatial_score,
                            "vlm_quality_reasoning": vlm_reasoning,
                            "vlm_quality_reasoning_conflict": vlm_reasoning_conflict,
                            "vlm_quality_issue_tags": vlm_issue_tags,
                            "vlm_quality_passed": (
                                bool(vlm_quality_passed) if vlm_quality_gate_enabled else None
                            ),
                            "vlm_quality_attempts": vlm_attempts,
                            "vlm_quality_retries_used": vlm_retries_used,
                            "vlm_quality_selected_fps": vlm_selected_fps,
                            "vlm_quality_fail_reason": (
                                None if vlm_quality_passed else vlm_quality_fail_reason
                            ),
                        }
                    )
                    if blur_laplacian_mean is not None:
                        accepted_blur_scores.append(float(blur_laplacian_mean))
                    if green_dominance_ratio is not None:
                        accepted_green_ratios.append(float(green_dominance_ratio))
                    if interframe_delta_mean is not None:
                        accepted_interframe_deltas.append(float(interframe_delta_mean))
                    total_enriched += 1
                    accepted_for_clip += 1
                    if vlm_quality_gate_enabled and retry_active:
                        num_vlm_quality_recovered += 1
                    variant_resolved = True
                    break

                if not variant_resolved and variant_has_vlm_unresolved_failure:
                    unresolved_vlm_failures.append(
                        (
                            f"{clip_name}/{variant.name}: {variant_last_fail_reason or 'vlm_unresolved'} "
                            f"(task={variant_last_task_score}, visual={variant_last_visual_score}, "
                            f"spatial={variant_last_spatial_score}, reasoning_conflict="
                            f"{variant_last_reasoning_conflict}, issue_tags={variant_last_issue_tags}, "
                            f"reasoning={variant_last_reasoning[:120]})"
                        )
                    )

            if accepted_for_clip < expected:
                total_failed += expected - accepted_for_clip

        selected_clip_names = [str(c.get("clip_name", "")) for c in source_clips]
        blur_reject_rate = (
            float(total_rejected_blur) / float(total_visual_gate_evaluated)
            if total_visual_gate_evaluated > 0
            else 0.0
        )
        stage_fail_reasons: List[str] = []
        stage_fail_error_codes: List[str] = []
        if vlm_quality_gate_enabled and vlm_quality_fail_closed and unresolved_vlm_failures:
            sample_reason = unresolved_vlm_failures[0]
            stage_fail_reasons.append(
                "VLM quality gate unresolved failures under fail-closed policy: "
                f"{len(unresolved_vlm_failures)} clip variants. Example: {sample_reason}"
            )
            stage_fail_error_codes.append("s2_vlm_fail_closed_unresolved")
        if visual_collapse_gate_enabled and blur_reject_rate > max_blur_reject_rate:
            stage_fail_reasons.append(
                "Visual-collapse blur reject rate exceeded cap: "
                f"{blur_reject_rate:.3f} > max_blur_reject_rate={max_blur_reject_rate:.3f}."
            )
            stage_fail_error_codes.append("s2_blur_reject_rate_exceeded")
        if min_valid_outputs > 0 and total_enriched < min_valid_outputs:
            stage_fail_reasons.append(
                "Insufficient valid enriched outputs: "
                f"{total_enriched} < min_valid_outputs={min_valid_outputs}."
            )
            stage_fail_error_codes.append("s2_min_valid_outputs_not_met")

        stage_status = "success" if total_enriched > 0 and not stage_fail_reasons else "failed"
        if stage_status == "failed" and not stage_fail_error_codes:
            stage_fail_error_codes.append("s2_no_enriched_outputs")
        stage_error_code = stage_fail_error_codes[0] if stage_status == "failed" else None

        # Write enriched manifest
        manifest_path = enrich_dir / "enriched_manifest.json"
        manifest = {
            "facility": facility.name,
            "num_clips": len(manifest_entries),
            "variants_per_clip": len(variants),
            "variant_names": [v.name for v in variants],
            "context_frame_index": sample_context_frame_index,
            "context_frame_mode": config.enrich.context_frame_mode,
            "min_frame0_ssim": ssim_gate_threshold,
            "max_source_clips": int(config.enrich.max_source_clips),
            "source_clip_selection_mode": clip_selection_meta.get("selection_mode"),
            "source_clip_selection_fallback": clip_selection_meta.get("fallback"),
            "source_clip_selection_fail_closed": bool(clip_selection_meta.get("fail_closed")),
            "source_clip_task": config.enrich.source_clip_task,
            "source_clip_name": config.enrich.source_clip_name,
            "min_source_clips": min_source_clips,
            "min_valid_outputs": min_valid_outputs,
            "max_blur_reject_rate": max_blur_reject_rate,
            "green_frame_ratio_max": green_frame_ratio_max,
            "enable_visual_collapse_gate": visual_collapse_gate_enabled,
            "vlm_quality_gate_enabled": vlm_quality_gate_enabled,
            "vlm_quality_fail_closed": vlm_quality_fail_closed,
            "vlm_quality_autoretry_enabled": vlm_quality_autoretry_enabled,
            "vlm_quality_max_regen_attempts": vlm_quality_max_regen_attempts,
            "vlm_quality_min_task_score": vlm_quality_min_task_score,
            "vlm_quality_min_visual_score": vlm_quality_min_visual_score,
            "vlm_quality_min_spatial_score": vlm_quality_min_spatial_score,
            "vlm_quality_require_reasoning_consistency": (
                vlm_quality_require_reasoning_consistency
            ),
            "vlm_quality_retry_context_frame_stride": vlm_quality_retry_context_frame_stride,
            "vlm_quality_disable_depth_on_final_retry": vlm_quality_disable_depth_on_final_retry,
            "vlm_quality_selected_fps": vlm_selected_fps,
            "selected_source_clips": selected_clip_names,
            "selected_source_target_distribution": selected_clip_target_distribution,
            "video_orientation_fix": facility.video_orientation_fix,
            "multi_view_context_enabled": bool(config.enrich.multi_view_context_enabled),
            "multi_view_context_offsets": [int(v) for v in config.enrich.multi_view_context_offsets],
            "scene_index_enabled": bool(config.enrich.scene_index_enabled),
            "scene_index_path": str(scene_index_path) if scene_index_path is not None else None,
            "clips": manifest_entries,
        }
        write_json(manifest, manifest_path)

        return StageResult(
            stage_name=self.name,
            status=stage_status,
            elapsed_seconds=0,
            outputs={
                "enrich_dir": str(enrich_dir),
                "manifest_path": str(manifest_path),
                "num_enriched": total_enriched,
                "num_generated": total_generated,
                "num_rejected_anchor_similarity": total_rejected_anchor_similarity,
                "num_rejected_blur": total_rejected_blur,
                "num_rejected_green_cast": total_rejected_green_cast,
                "num_rejected_low_motion": total_rejected_low_motion,
                "num_vlm_quality_evaluated": num_vlm_quality_evaluated,
                "num_vlm_quality_failures": num_vlm_quality_failures,
                "num_vlm_quality_retries": num_vlm_quality_retries,
                "num_vlm_quality_recovered": num_vlm_quality_recovered,
                "num_vlm_quality_api_failures": num_vlm_quality_api_failures,
                "num_vlm_quality_parse_failures": num_vlm_quality_parse_failures,
                "num_vlm_quality_reasoning_conflicts": num_vlm_quality_reasoning_conflicts,
                "num_vlm_quality_unresolved": len(unresolved_vlm_failures),
                "num_trimmed_inputs": total_trimmed_inputs,
                "num_selected_source_clips": len(source_clips),
                "selected_source_clips": selected_clip_names,
                "selected_source_target_distribution": selected_clip_target_distribution,
                "error_code": stage_error_code,
                "error_codes": list(stage_fail_error_codes) if stage_status == "failed" else None,
                **coverage_outputs,
                **source.to_metadata(),
            },
            metrics={
                "num_enriched_clips": total_enriched,
                "num_generated": total_generated,
                "num_failed": total_failed,
                "num_rejected_anchor_similarity": total_rejected_anchor_similarity,
                "num_rejected_blur": total_rejected_blur,
                "num_rejected_green_cast": total_rejected_green_cast,
                "num_rejected_low_motion": total_rejected_low_motion,
                "num_vlm_quality_evaluated": num_vlm_quality_evaluated,
                "num_vlm_quality_failures": num_vlm_quality_failures,
                "num_vlm_quality_retries": num_vlm_quality_retries,
                "num_vlm_quality_recovered": num_vlm_quality_recovered,
                "num_vlm_quality_api_failures": num_vlm_quality_api_failures,
                "num_vlm_quality_parse_failures": num_vlm_quality_parse_failures,
                "num_vlm_quality_reasoning_conflicts": num_vlm_quality_reasoning_conflicts,
                "num_vlm_quality_unresolved": len(unresolved_vlm_failures),
                "blur_reject_rate": round(float(blur_reject_rate), 6),
                "max_blur_reject_rate": max_blur_reject_rate,
                "green_frame_ratio_max": green_frame_ratio_max,
                "enable_visual_collapse_gate": visual_collapse_gate_enabled,
                "vlm_quality_gate_enabled": vlm_quality_gate_enabled,
                "vlm_quality_fail_closed": vlm_quality_fail_closed,
                "vlm_quality_autoretry_enabled": vlm_quality_autoretry_enabled,
                "vlm_quality_max_regen_attempts": vlm_quality_max_regen_attempts,
                "vlm_quality_min_task_score": vlm_quality_min_task_score,
                "vlm_quality_min_visual_score": vlm_quality_min_visual_score,
                "vlm_quality_min_spatial_score": vlm_quality_min_spatial_score,
                "vlm_quality_require_reasoning_consistency": (
                    vlm_quality_require_reasoning_consistency
                ),
                "vlm_quality_retry_context_frame_stride": vlm_quality_retry_context_frame_stride,
                "vlm_quality_disable_depth_on_final_retry": (
                    vlm_quality_disable_depth_on_final_retry
                ),
                "vlm_quality_selected_fps": vlm_selected_fps,
                "min_source_clips": min_source_clips,
                "min_valid_outputs": min_valid_outputs,
                "num_trimmed_inputs": total_trimmed_inputs,
                "num_orientation_fixed_clips": total_orientation_fixed_clips,
                "num_multi_view_context_clips": total_multi_view_context_clips,
                "num_scene_index_retrievals": total_scene_index_retrievals,
                "num_selected_source_clips": len(source_clips),
                "selected_source_target_distribution": selected_clip_target_distribution,
                "source_clip_selection_mode": clip_selection_meta.get("selection_mode"),
                "source_clip_selection_fallback": clip_selection_meta.get("fallback"),
                "source_clip_selection_fail_closed": bool(clip_selection_meta.get("fail_closed")),
                "max_input_frames": int(config.enrich.max_input_frames),
                "min_required_input_frames": int(min_required_input_frames),
                "min_frame0_ssim_threshold": ssim_gate_threshold,
                "mean_frame0_ssim": (
                    round(sum(accepted_ssim_scores) / len(accepted_ssim_scores), 4)
                    if accepted_ssim_scores
                    else None
                ),
                "mean_blur_laplacian_mean": (
                    round(sum(accepted_blur_scores) / len(accepted_blur_scores), 4)
                    if accepted_blur_scores
                    else None
                ),
                "mean_green_dominance_ratio": (
                    round(sum(accepted_green_ratios) / len(accepted_green_ratios), 4)
                    if accepted_green_ratios
                    else None
                ),
                "mean_interframe_delta_mean": (
                    round(sum(accepted_interframe_deltas) / len(accepted_interframe_deltas), 4)
                    if accepted_interframe_deltas
                    else None
                ),
                "context_frame_mode": config.enrich.context_frame_mode,
                "context_selection_target_centered_count": context_mode_counts.get(
                    "target_centered", 0
                ),
                "context_selection_fixed_count": context_mode_counts.get("fixed", 0),
                "context_selection_deterministic_count": context_mode_counts.get(
                    "deterministic", 0
                ),
                "mean_target_center_score": (
                    round(sum(selected_center_scores) / len(selected_center_scores), 4)
                    if selected_center_scores
                    else None
                ),
                "variants_used": [v.name for v in variants],
                "error_code": stage_error_code,
                "error_codes": list(stage_fail_error_codes) if stage_status == "failed" else None,
                **coverage_metrics,
                **source.to_metadata(),
            },
            detail=" ".join(stage_fail_reasons),
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
    clips: List[dict],
    work_dir: Path,
    context_frame_index: int | None = None,
    facility: FacilityConfig | None = None,
    enrich_dir: Path | None = None,
) -> Tuple[Path | None, int | None]:
    """Extract a single frame from the first selected clip for dynamic variant generation."""
    if not clips:
        return None, None

    clip = clips[0]
    clip_name = str(clip.get("clip_name", "sample"))
    video_raw = str(clip.get("video_path", "")).strip()
    if not video_raw:
        return None, None
    video_path = Path(video_raw)
    if not video_path.exists():
        return None, None

    if facility is not None and enrich_dir is not None:
        source_depth_raw = str(clip.get("depth_video_path", "")).strip()
        source_depth = Path(source_depth_raw) if source_depth_raw else None
        video_path, _, _ = _resolve_oriented_inputs_for_clip(
            facility=facility,
            clip_name=clip_name,
            enrich_dir=enrich_dir,
            video_path=video_path,
            depth_path=source_depth if source_depth is not None and source_depth.exists() else None,
        )

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


def _select_source_clips(
    *,
    render_manifest: dict,
    config: ValidationConfig,
    facility: FacilityConfig,
) -> tuple[List[dict], Dict[str, object]]:
    clips = list(render_manifest.get("clips", []))
    if not clips:
        return [], {"selection_mode": "all", "fallback": None}

    requested_mode = str(config.enrich.source_clip_selection_mode or "all").strip().lower()
    fail_closed_task_targeted = bool(config.enrich.source_clip_selection_fail_closed)
    max_source_clips = int(config.enrich.max_source_clips)
    selection_limit = max_source_clips if max_source_clips > 0 else len(clips)
    requested_rollouts = max(
        1,
        selection_limit,
        max(0, int(config.enrich.min_source_clips)),
    )

    fallback_reason: str | None = None
    selected: List[dict] = []
    clip_by_name = {str(c.get("clip_name", "")): c for c in clips if c.get("clip_name")}

    if requested_mode == "task_targeted":
        source_task = str(config.enrich.source_clip_task or "").strip()
        hints_available = (
            facility.task_hints_path is not None and Path(facility.task_hints_path).exists()
        )
        if source_task and hints_available:
            try:
                assignments = build_task_start_assignments(
                    tasks=[source_task],
                    num_rollouts=requested_rollouts,
                    render_manifest={"clips": clips},
                    task_hints_path=facility.task_hints_path,
                )
                seen_names = set()
                for assignment in assignments:
                    name = str(assignment.get("clip_name", "")).strip()
                    if not name or name in seen_names:
                        continue
                    clip = clip_by_name.get(name)
                    if clip is None:
                        continue
                    selected.append(clip)
                    seen_names.add(name)
            except Exception:
                logger.warning("Task-targeted source-clip selection failed", exc_info=True)
                selected = []
        if not source_task:
            fallback_reason = "missing_task"
        elif not hints_available:
            fallback_reason = "missing_task_hints"
        elif not selected:
            fallback_reason = "selection_failed"
        if fallback_reason is not None:
            if fail_closed_task_targeted:
                selected = []
            else:
                selected = _fallback_task_targeted_clip_order(clips)
        if not (fail_closed_task_targeted and fallback_reason is not None):
            selected = _rebalance_task_targeted_selection(
                selected=selected,
                all_clips=clips,
                limit=selection_limit,
            )
    elif requested_mode == "explicit":
        explicit_name = str(config.enrich.source_clip_name or "").strip()
        if not explicit_name:
            fallback_reason = "explicit_clip_not_set"
            selected = []
        else:
            selected = [c for c in clips if str(c.get("clip_name", "")).strip() == explicit_name]
            if not selected:
                fallback_reason = "explicit_clip_not_found"
                selected = []
    else:
        selected = clips

    if (
        not selected
        and requested_mode != "explicit"
        and not (
            requested_mode == "task_targeted"
            and fallback_reason is not None
            and fail_closed_task_targeted
        )
    ):
        selected = clips
        fallback_reason = fallback_reason or "empty_selection"

    deduped: List[dict] = []
    seen = set()
    for clip in selected:
        name = str(clip.get("clip_name", "")).strip()
        key = name or str(id(clip))
        if key in seen:
            continue
        deduped.append(clip)
        seen.add(key)

    selected_limited = deduped[: max(1, selection_limit)]
    target_distribution = _summarize_target_distribution(selected_limited)
    explicit_fail_closed = requested_mode == "explicit" and fallback_reason is not None
    return selected_limited, {
        "selection_mode": requested_mode,
        "fallback": fallback_reason,
        "fail_closed": (
            (
                requested_mode == "task_targeted"
                and fallback_reason is not None
                and fail_closed_task_targeted
            )
            or explicit_fail_closed
        ),
        "target_distribution": target_distribution,
        "num_unique_targets": len(target_distribution),
    }


def _fallback_task_targeted_clip_order(clips: List[dict]) -> List[dict]:
    manipulation = []
    non_manipulation = []
    for clip in clips:
        path_type = str(clip.get("path_type", "")).strip().lower()
        if path_type == "manipulation":
            manipulation.append(clip)
        else:
            non_manipulation.append(clip)
    return manipulation + non_manipulation


def _rebalance_task_targeted_selection(
    *,
    selected: List[dict],
    all_clips: List[dict],
    limit: int,
) -> List[dict]:
    """Favor broad target coverage before selecting duplicate-target clips."""
    limit = max(1, int(limit))
    ordered_pool: List[dict] = []
    seen_names: set[str] = set()
    for clip in list(selected) + _fallback_task_targeted_clip_order(all_clips):
        name = str(clip.get("clip_name", "")).strip()
        if not name or name in seen_names:
            continue
        ordered_pool.append(clip)
        seen_names.add(name)

    if not ordered_pool:
        return []

    by_target: Dict[str, List[dict]] = {}
    target_order: List[str] = []
    for clip in ordered_pool:
        key = _clip_target_key(clip)
        if key not in by_target:
            by_target[key] = []
            target_order.append(key)
        by_target[key].append(clip)

    rebalanced: List[dict] = []
    # First pass: one clip per unique target.
    for key in target_order:
        clips_for_target = by_target.get(key, [])
        if not clips_for_target:
            continue
        rebalanced.append(clips_for_target.pop(0))
        if len(rebalanced) >= limit:
            return rebalanced
    # Second pass: fill remaining slots from residual clips.
    for key in target_order:
        for clip in by_target.get(key, []):
            rebalanced.append(clip)
            if len(rebalanced) >= limit:
                return rebalanced
    return rebalanced[:limit]


def _summarize_target_distribution(clips: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for clip in clips:
        key = _clip_target_key(clip)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _clip_target_key(clip: dict) -> str:
    path_context = clip.get("path_context") or {}
    if isinstance(path_context, dict):
        inst = str(path_context.get("target_instance_id", "")).strip()
        if inst:
            return f"instance:{inst}"
        label = str(path_context.get("target_label", "")).strip()
        if label:
            return f"label:{label}"
        point = path_context.get("approach_point")
        if isinstance(point, list) and len(point) >= 3:
            try:
                xyz = [round(float(v), 3) for v in point[:3]]
            except Exception:
                xyz = None
            if xyz is not None:
                return f"approach:{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}"

    inst = str(clip.get("target_instance_id", "")).strip()
    if inst:
        return f"instance:{inst}"
    label = str(clip.get("target_label", "")).strip()
    if label:
        return f"label:{label}"
    path_type = str(clip.get("path_type", "")).strip().lower()
    if path_type:
        return f"path:{path_type}"
    clip_name = str(clip.get("clip_name", "")).strip()
    return clip_name or "unknown"


def _resolve_oriented_inputs_for_clip(
    *,
    facility: FacilityConfig,
    clip_name: str,
    enrich_dir: Path,
    video_path: Path,
    depth_path: Path | None,
) -> tuple[Path, Path | None, str]:
    orientation_fix = _normalize_video_orientation_fix(getattr(facility, "video_orientation_fix", "none"))
    try:
        resolved_video = video_path
        resolved_depth = depth_path
        applied = "none"
        if orientation_fix != "none":
            resolved_video = _apply_video_orientation_fix(
                input_path=video_path,
                enrich_dir=enrich_dir,
                clip_name=clip_name,
                stream_tag="rgb",
                orientation_fix=orientation_fix,
                force_grayscale=False,
            )
            if depth_path is not None and depth_path.exists():
                resolved_depth = _apply_video_orientation_fix(
                    input_path=depth_path,
                    enrich_dir=enrich_dir,
                    clip_name=clip_name,
                    stream_tag="depth",
                    orientation_fix=orientation_fix,
                    force_grayscale=True,
                )
            applied = orientation_fix
        elif depth_path is not None and depth_path.exists():
            resolved_depth = _maybe_prepare_depth_control(
                facility_name=facility.name,
                clip_name=clip_name,
                depth_path=depth_path,
                enrich_dir=enrich_dir,
            )
            if resolved_depth != depth_path:
                applied = "rotate180_depth_legacy"
        return resolved_video, resolved_depth, applied
    except Exception:
        logger.warning(
            "Orientation fix failed for %s clip %s; continuing with original inputs",
            facility.name,
            clip_name,
            exc_info=True,
        )
        return video_path, depth_path, "none"


def _normalize_video_orientation_fix(raw: str | None) -> str:
    return _shared_normalize_video_orientation_fix(raw)


def _apply_video_orientation_fix(
    *,
    input_path: Path,
    enrich_dir: Path,
    clip_name: str,
    stream_tag: str,
    orientation_fix: str,
    force_grayscale: bool,
) -> Path:
    return _shared_apply_video_orientation_fix(
        input_path=input_path,
        cache_dir=enrich_dir / "_orientation_fixed",
        clip_name=clip_name,
        stream_tag=stream_tag,
        orientation_fix=orientation_fix,
        force_grayscale=force_grayscale,
    )


def _transform_video_orientation(
    *,
    input_path: Path,
    output_path: Path,
    orientation_fix: str,
    force_grayscale: bool,
) -> None:
    _shared_transform_video_orientation(
        input_path=input_path,
        output_path=output_path,
        orientation_fix=orientation_fix,
        force_grayscale=force_grayscale,
    )


def _transform_video_frame(frame, orientation_fix: str):
    return _shared_transform_video_frame(frame, orientation_fix)


def _resolve_multi_view_context_indices(
    *,
    anchor_index: int,
    total_frames: int,
    offsets: List[int],
) -> List[int]:
    if total_frames <= 0:
        return [max(0, int(anchor_index))]
    indices: List[int] = []
    seen = set()
    for raw_offset in offsets:
        idx = max(0, min(int(anchor_index) + int(raw_offset), total_frames - 1))
        if idx in seen:
            continue
        indices.append(idx)
        seen.add(idx)
    if not indices:
        idx = max(0, min(int(anchor_index), total_frames - 1))
        indices = [idx]
    return indices


def _read_video_frame_at_index(video_path: Path, frame_index: int):
    import cv2

    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _read_video_frames(video_path: Path, frame_indices: List[int]) -> List[object]:
    frames: List[object] = []
    for idx in frame_indices:
        frame = _read_video_frame_at_index(video_path, idx)
        if frame is not None:
            frames.append(frame)
    return frames


def _write_context_montage(frames: List[object], output_path: Path) -> Path | None:
    import cv2
    import numpy as np

    if not frames:
        return None
    processed = []
    target_h, target_w = None, None
    for frame in frames:
        if frame is None:
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if target_h is None or target_w is None:
            target_h, target_w = frame.shape[:2]
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        processed.append(frame)
    if not processed:
        return None

    cols = min(4, max(1, int(math.ceil(math.sqrt(len(processed))))))
    rows = int(math.ceil(len(processed) / cols))
    canvas = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
    for i, frame in enumerate(processed):
        r = i // cols
        c = i % cols
        canvas[r * target_h : (r + 1) * target_h, c * target_w : (c + 1) * target_w] = frame

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def _prepare_cosmos_input(
    video_path: Path,
    depth_path: Path | None,
    clip_name: str,
    enrich_dir: Path,
    preferred_context_frame_index: int | None,
    max_input_frames: int,
    min_required_frames: int,
) -> PreparedCosmosInput | None:
    """Prepare Cosmos inputs with optional pre-trim for long clips."""
    min_frames = max(1, int(min_required_frames))

    safe_clip_name = sanitize_filename_component_with_hash(clip_name, fallback="clip")

    def _enforce_h264(
        *,
        source_path: Path,
        label: str,
    ):
        return ensure_h264_video(
            input_path=source_path,
            min_decoded_frames=min_frames,
            output_path=enrich_dir / "_h264_inputs" / f"{safe_clip_name}_{label}_h264.mp4",
            replace_source=False,
        )

    total_frames = _probe_video_frame_count(video_path)
    if total_frames <= 0:
        return None

    resolved_index = _resolve_context_frame_index(total_frames, preferred_context_frame_index)
    untrimmed_context_index = (
        resolved_index if preferred_context_frame_index is not None else None
    )
    max_frames = max(0, int(max_input_frames))
    if max_frames <= 0 or total_frames <= max_frames:
        rgb_checked = _enforce_h264(
            source_path=video_path,
            label="rgb",
        )
        depth_checked: Path | None = None
        if depth_path is not None and depth_path.exists():
            depth_checked = _enforce_h264(
                source_path=depth_path,
                label="depth",
            ).path
        return PreparedCosmosInput(
            video_path=rgb_checked.path,
            depth_path=depth_checked,
            preferred_context_frame_index=untrimmed_context_index,
            input_total_frames=rgb_checked.decoded_frames,
            input_trimmed=False,
            input_trim_start_frame=None,
            input_trim_num_frames=None,
        )

    trim_len = max(1, min(max_frames, total_frames))
    trim_start = max(0, min(resolved_index - (trim_len // 2), total_frames - trim_len))

    trim_dir = enrich_dir / "_trimmed_inputs"
    trim_dir.mkdir(parents=True, exist_ok=True)
    trimmed_video = trim_dir / f"{safe_clip_name}_rgb_trim.mp4"
    written_video = _trim_video_window(
        input_path=video_path,
        output_path=trimmed_video,
        start_frame=trim_start,
        num_frames=trim_len,
        force_grayscale=False,
    )
    if written_video <= 0:
        return None

    trimmed_depth: Path | None = None
    if depth_path is not None and depth_path.exists():
        trimmed_depth = trim_dir / f"{safe_clip_name}_depth_trim.mp4"
        written_depth = _trim_video_window(
            input_path=depth_path,
            output_path=trimmed_depth,
            start_frame=trim_start,
            num_frames=trim_len,
            force_grayscale=True,
        )
        if written_depth <= 0:
            trimmed_depth = None

    rgb_checked = _enforce_h264(
        source_path=trimmed_video,
        label="rgb_trim",
    )
    depth_checked: Path | None = None
    if trimmed_depth is not None and trimmed_depth.exists():
        depth_checked = _enforce_h264(
            source_path=trimmed_depth,
            label="depth_trim",
        ).path

    context_in_trim = max(0, min(resolved_index - trim_start, rgb_checked.decoded_frames - 1))
    return PreparedCosmosInput(
        video_path=rgb_checked.path,
        depth_path=depth_checked,
        preferred_context_frame_index=context_in_trim,
        input_total_frames=rgb_checked.decoded_frames,
        input_trimmed=True,
        input_trim_start_frame=trim_start,
        input_trim_num_frames=rgb_checked.decoded_frames,
    )


def _resolve_context_frame_index(total_frames: int, preferred_index: int | None) -> int:
    """Resolve context frame index with existing Stage-2 deterministic behavior."""
    if total_frames <= 0:
        return 0
    if preferred_index is None:
        return max(0, total_frames // 4)
    return max(0, min(int(preferred_index), total_frames - 1))


def _probe_video_frame_count(video_path: Path) -> int:
    """Get video frame count with OpenCV probe."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, total_frames)


def _trim_video_window(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    num_frames: int,
    force_grayscale: bool,
) -> int:
    """Trim a contiguous frame window from a video and save as MP4."""
    import cv2

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 10.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        return 0

    if force_grayscale and first.ndim == 3:
        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    if first.ndim == 2:
        height, width = first.shape
    else:
        height, width = first.shape[:2]
    if width <= 0 or height <= 0:
        cap.release()
        return 0

    writer = open_mp4_writer(
        output_path=output_path,
        fps=float(fps),
        frame_size=(width, height),
        is_color=not force_grayscale,
    )
    if not writer.isOpened():
        cap.release()
        return 0

    written = 0
    try:
        frame = first
        while written < max(1, int(num_frames)):
            if force_grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            writer.write(frame)
            written += 1
            if written >= int(num_frames):
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
    finally:
        cap.release()
        writer.release()

    return written


def _evaluate_stage1_coverage_gate(
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
        target_key = _target_key(target_xyz)
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
            blur_score = _estimate_clip_blur_score(
                video_path=Path(str(clip_entry.get("video_path", ""))),
                sample_every_n_frames=blur_every,
                max_samples=blur_max_samples,
            )
        is_blurry = blur_score is None or blur_score < blur_min
        if is_blurry:
            target_stats["num_blurry_clips"] = int(target_stats["num_blurry_clips"]) + 1
            blurred_clip_names.append(clip_name)
            continue

        visible_frames, total_frames, center_band_frames, angle_bins = _analyze_target_visibility(
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


def _target_key(target_xyz: object) -> str:
    """Stable key for grouping manipulation targets from floating-point points."""
    xyz = list(target_xyz) if not isinstance(target_xyz, list) else target_xyz
    rounded = [round(float(v), 3) for v in xyz[:3]]
    return ",".join(f"{v:.3f}" for v in rounded)


def _analyze_target_visibility(
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

    total_frames, visible_samples = _project_target_to_camera_path(clip_entry, target_xyz)
    return _shared_analyze_target_visibility(
        total_frames=total_frames,
        visible_samples=visible_samples,
        angle_bin_deg=angle_bin_deg,
        center_band_x=config.render.stage1_coverage_center_band_x,
        center_band_y=config.render.stage1_coverage_center_band_y,
    )


def _project_target_to_camera_path(
    clip_entry: dict,
    target_xyz: object,
) -> tuple[int, List[Dict[str, float]]]:
    """Project a clip target point into all camera-path frames."""
    return _shared_project_target_to_camera_path(clip_entry, target_xyz)


def _resolve_center_band_bounds(config: ValidationConfig) -> tuple[float, float, float, float]:
    """Resolve normalized frame center-band bounds from config."""
    return _shared_resolve_center_band_bounds(
        config.render.stage1_coverage_center_band_x,
        config.render.stage1_coverage_center_band_y,
    )


def _resolve_clip_context_selection(
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
            _, visible_samples = _project_target_to_camera_path(clip_entry, target_point)
            if visible_samples:
                x_lo, x_hi, y_lo, y_hi = _resolve_center_band_bounds(config)
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


def _estimate_clip_blur_score(
    video_path: Path,
    sample_every_n_frames: int,
    max_samples: int,
) -> float | None:
    """Estimate sharpness via Laplacian variance over sampled video frames."""
    return _shared_estimate_clip_blur_score(
        video_path=video_path,
        sample_every_n_frames=sample_every_n_frames,
        max_samples=max_samples,
    )


def _analyze_output_visual_quality(
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
    """Legacy fallback depth-only rotation path for explicitly listed facilities."""
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
        _transform_video_orientation(
            input_path=depth_path,
            output_path=fixed_path,
            orientation_fix="rotate180",
            force_grayscale=True,
        )
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
    """Backward-compatible wrapper for legacy depth rotate helper."""
    _transform_video_orientation(
        input_path=input_path,
        output_path=output_path,
        orientation_fix="rotate180",
        force_grayscale=True,
    )
