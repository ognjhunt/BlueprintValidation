"""Build DreamDojo-compatible training datasets from enriched videos."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from ..common import get_logger, read_json, write_json
from ..config import DatasetQualityConfig
from ..provenance import build_provenance_stamp
from ..validation import load_and_validate_manifest
from ..video_io import ensure_h264_video
from .data_quality import (
    PromptLintConfig,
    TemporalGateConfig,
    analyze_temporal_quality,
    fingerprint_video_content,
    lint_prompt,
    normalize_prompt,
    summarize_distribution,
    temporal_reject_reasons,
)

logger = get_logger("training.dataset_builder")


def _sanitize_name_component(value: object, *, fallback: str) -> str:
    text = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return cleaned or fallback


def _derive_dataset_stem(entry: dict, src: Path, index: int) -> str:
    clip_component = _sanitize_name_component(entry.get("clip_name"), fallback=src.stem or "clip")
    variant_component = _sanitize_name_component(entry.get("variant_name"), fallback="variant")
    try:
        source_fingerprint_input = str(src.resolve(strict=False))
    except Exception:
        source_fingerprint_input = str(src)
    source_hash = hashlib.sha1(source_fingerprint_input.encode("utf-8")).hexdigest()[:10]
    return f"{clip_component}__{variant_component}__{index:05d}__{source_hash}"


def _load_leakage_index(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        logger.warning("Could not read leakage index at %s; starting fresh", path, exc_info=True)
        return {}
    fingerprints = payload.get("fingerprints", {})
    if isinstance(fingerprints, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in fingerprints.items():
            if isinstance(k, str) and isinstance(v, dict):
                out[k] = dict(v)
        return out
    return {}


def _write_leakage_index(path: Path | None, fingerprints: Dict[str, Dict[str, Any]]) -> None:
    if path is None:
        return
    write_json(
        {
            "num_fingerprints": len(fingerprints),
            "fingerprints": fingerprints,
        },
        path,
    )


def _build_prompt_lint_config(qcfg: DatasetQualityConfig) -> PromptLintConfig:
    src = qcfg.prompt_lint
    return PromptLintConfig(
        enabled=bool(src.enabled),
        min_chars=max(0, int(src.min_chars)),
        min_tokens=max(0, int(src.min_tokens)),
        min_unique_token_ratio=float(src.min_unique_token_ratio),
        allow_generic_substrings=bool(src.allow_generic_substrings),
    )


def _build_temporal_gate_config(qcfg: DatasetQualityConfig) -> TemporalGateConfig:
    src = qcfg.temporal_gates
    return TemporalGateConfig(
        enabled=bool(src.enabled),
        min_frames_for_check=max(1, int(src.min_frames_for_check)),
        max_frames_to_sample=max(1, int(src.max_frames_to_sample)),
        min_mean_interframe_delta=float(src.min_mean_interframe_delta),
        max_freeze_ratio=float(src.max_freeze_ratio),
        max_abrupt_cut_ratio=float(src.max_abrupt_cut_ratio),
        max_blockiness_score=float(src.max_blockiness_score),
    )


def _record_rejection(
    *,
    rejected_rows: List[Dict[str, Any]],
    entry: Dict[str, Any],
    src: Path,
    reason: str,
    detail: str,
    quarantine_videos_dir: Path,
    quarantine_enabled: bool,
    idx: int,
) -> None:
    quarantine_path: Path | None = None
    if quarantine_enabled and src.exists():
        stem = _sanitize_name_component(entry.get("clip_name"), fallback=src.stem or f"clip_{idx:05d}")
        qname = f"{stem}__reject_{idx:05d}{src.suffix or '.mp4'}"
        quarantine_path = quarantine_videos_dir / qname
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, quarantine_path)
        except Exception:
            logger.warning("Failed to copy rejected clip into quarantine: %s", src, exc_info=True)
            quarantine_path = None
    rejected_rows.append(
        {
            "clip_name": str(entry.get("clip_name", "") or ""),
            "variant_name": str(entry.get("variant_name", "") or ""),
            "output_video_path": str(src),
            "reason": reason,
            "detail": detail,
            "prompt": str(entry.get("prompt", "") or ""),
            "quarantine_video_path": str(quarantine_path) if quarantine_path is not None else None,
        }
    )


def _distribution_failures(
    *,
    csv_rows: List[Dict[str, Any]],
    prompt_counter: Counter[str],
    qcfg: DatasetQualityConfig,
) -> List[str]:
    failures: List[str] = []
    dcfg = qcfg.distribution
    if not bool(dcfg.enabled):
        return failures
    total = len(csv_rows)
    if total <= 0:
        failures.append("distribution_empty_dataset")
        return failures
    if total < int(dcfg.min_total_clips_for_caps):
        return failures

    variant_stats = summarize_distribution([str(r.get("variant", "") or "") for r in csv_rows])
    source_stats = summarize_distribution([str(r.get("source_clip", "") or "") for r in csv_rows])
    unique_variants = int(variant_stats["unique"])
    unique_sources = int(source_stats["unique"])
    if unique_variants < int(dcfg.min_unique_variants):
        failures.append(
            f"distribution_low_variant_diversity:{unique_variants}<{int(dcfg.min_unique_variants)}"
        )
    if unique_sources < int(dcfg.min_unique_source_clips):
        failures.append(
            f"distribution_low_source_diversity:{unique_sources}<{int(dcfg.min_unique_source_clips)}"
        )
    if float(variant_stats["dominant_fraction"]) > float(dcfg.max_single_variant_fraction):
        failures.append(
            "distribution_variant_dominance:"
            f"{variant_stats['dominant_fraction']:.3f}>{float(dcfg.max_single_variant_fraction):.3f}"
        )
    if float(source_stats["dominant_fraction"]) > float(dcfg.max_single_source_clip_fraction):
        failures.append(
            "distribution_source_dominance:"
            f"{source_stats['dominant_fraction']:.3f}>{float(dcfg.max_single_source_clip_fraction):.3f}"
        )

    if prompt_counter:
        top_prompt_count = max(prompt_counter.values())
        top_prompt_fraction = float(top_prompt_count) / float(total)
        if top_prompt_fraction > float(dcfg.max_prompt_dominance_fraction):
            failures.append(
                "distribution_prompt_dominance:"
                f"{top_prompt_fraction:.3f}>{float(dcfg.max_prompt_dominance_fraction):.3f}"
            )
    return failures


def build_dreamdojo_dataset(
    enriched_manifest_path: Path,
    output_dir: Path,
    facility_name: str,
    min_decoded_frames: int = 13,
    quality_config: DatasetQualityConfig | None = None,
    config_obj: Any | None = None,
    leakage_index_path: Path | None = None,
    dataset_tag: str = "",
    stage_name: str = "s3_finetune",
) -> Path:
    """Convert enriched video manifest into DreamDojo training dataset format."""
    qcfg = quality_config or DatasetQualityConfig()
    if bool(qcfg.strict_manifest_validation):
        manifest = load_and_validate_manifest(
            enriched_manifest_path,
            manifest_type="enriched",
            require_existing_paths=True,
        )
    else:
        manifest = read_json(enriched_manifest_path)
    dataset_dir = output_dir / "dreamdojo_dataset"
    videos_dir = dataset_dir / "videos"
    metas_dir = dataset_dir / "metas"
    quarantine_dir = dataset_dir / "quarantine"
    quarantine_videos_dir = quarantine_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metas_dir.mkdir(parents=True, exist_ok=True)
    if bool(qcfg.quarantine_rejections):
        quarantine_videos_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    prompt_lint_cfg = _build_prompt_lint_config(qcfg)
    temporal_cfg = _build_temporal_gate_config(qcfg)
    per_source_prompt_norms: Dict[str, set[str]] = defaultdict(set)
    prompt_counter: Counter[str] = Counter()
    exact_seen: set[str] = set()
    leakage_index = _load_leakage_index(leakage_index_path)
    new_fingerprints: Dict[str, Dict[str, Any]] = {}
    quality_rows: List[Dict[str, Any]] = []

    for idx, entry in enumerate(manifest.get("clips", [])):
        src = Path(str(entry.get("output_video_path", "") or ""))
        prompt = str(entry.get("prompt", "") or "")
        clip_name = str(entry.get("clip_name", "") or "").strip() or f"clip_{idx:05d}"
        variant_name = str(entry.get("variant_name", "") or "").strip() or "variant"
        prompt_norm = normalize_prompt(prompt)
        stem = _derive_dataset_stem(entry, src, idx)
        video_suffix = src.suffix or ".mp4"
        dst = videos_dir / f"{stem}{video_suffix}"
        dedupe_counter = 1
        while dst.exists():
            dst = videos_dir / f"{stem}__dup{dedupe_counter}{video_suffix}"
            dedupe_counter += 1

        try:
            if not src.exists():
                raise RuntimeError(f"source clip missing: {src}")

            prompt_reasons = lint_prompt(prompt, prompt_lint_cfg)
            if prompt_reasons:
                raise RuntimeError(",".join(prompt_reasons))
            if prompt_norm in per_source_prompt_norms[clip_name]:
                raise RuntimeError("variant_prompt_collapse")

            shutil.copy2(src, dst)
            checked = ensure_h264_video(
                input_path=dst,
                min_decoded_frames=max(1, int(min_decoded_frames)),
                replace_source=True,
            )
            if checked.path != dst:
                dst = checked.path
            if bool(checked.content_monochrome_warning):
                raise RuntimeError(
                    "monochrome-content:"
                    f"max_std_dev={checked.content_max_std_dev}"
                )

            temporal_metrics = analyze_temporal_quality(dst, temporal_cfg)
            temporal_reasons = temporal_reject_reasons(temporal_metrics, temporal_cfg)
            if temporal_reasons:
                raise RuntimeError(",".join(temporal_reasons))

            fingerprint = fingerprint_video_content(dst)
            exact_key = f"{fingerprint}::{prompt_norm}"
            if bool(qcfg.enable_duplicate_detection):
                if exact_key in exact_seen:
                    raise RuntimeError("duplicate_clip_prompt_pair")
                exact_seen.add(exact_key)
            if bool(qcfg.enable_leakage_detection):
                existing = leakage_index.get(fingerprint)
                if existing is not None and str(existing.get("dataset_tag", "")) != str(dataset_tag or ""):
                    raise RuntimeError(
                        "cross_dataset_leakage:"
                        f"{existing.get('dataset_tag', '')}"
                    )

            meta_name = dst.stem + ".txt"
            meta_path = metas_dir / meta_name
            meta_dedupe_counter = 1
            while meta_path.exists():
                meta_name = f"{dst.stem}__dup{meta_dedupe_counter}.txt"
                meta_path = metas_dir / meta_name
                meta_dedupe_counter += 1
            meta_path.write_text(prompt, encoding="utf-8")

            csv_rows.append(
                {
                    "video_path": f"videos/{dst.name}",
                    "meta_path": f"metas/{meta_name}",
                    "prompt": prompt,
                    "variant": variant_name,
                    "source_clip": clip_name,
                    "source_path_type": str(entry.get("source_path_type", "") or ""),
                    "facility": facility_name,
                    "video_fingerprint": fingerprint,
                }
            )
            quality_rows.append(
                {
                    "clip_name": clip_name,
                    "variant_name": variant_name,
                    "video_path": str(dst),
                    "prompt_norm": prompt_norm,
                    "video_fingerprint": fingerprint,
                    "temporal_metrics": temporal_metrics,
                    "content_max_std_dev": checked.content_max_std_dev,
                    "decoded_frames": checked.decoded_frames,
                }
            )
            per_source_prompt_norms[clip_name].add(prompt_norm)
            prompt_counter[prompt_norm] += 1
            new_fingerprints[fingerprint] = {
                "dataset_tag": str(dataset_tag or ""),
                "facility": facility_name,
                "clip_name": clip_name,
                "variant_name": variant_name,
                "prompt_norm": prompt_norm,
                "video_path": str(dst),
            }
        except Exception as exc:
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to clean rejected dataset clip: %s", dst, exc_info=True)
            _record_rejection(
                rejected_rows=rejected_rows,
                entry=entry,
                src=src,
                reason="quality_gate_reject",
                detail=str(exc),
                quarantine_videos_dir=quarantine_videos_dir,
                quarantine_enabled=bool(qcfg.quarantine_rejections),
                idx=idx,
            )

    distribution_failures = _distribution_failures(
        csv_rows=csv_rows,
        prompt_counter=prompt_counter,
        qcfg=qcfg,
    )

    csv_path = dataset_dir / "metadata.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    rejected_fraction = float(len(rejected_rows)) / float(len(csv_rows) + len(rejected_rows) or 1)
    reject_fraction_exceeded = rejected_fraction > float(qcfg.max_reject_fraction)
    quality_report = {
        "dataset_tag": str(dataset_tag or ""),
        "stage_name": stage_name,
        "num_input_clips": int(len(manifest.get("clips", []))),
        "num_accepted_clips": len(csv_rows),
        "num_rejected_clips": len(rejected_rows),
        "rejected_fraction": round(rejected_fraction, 6),
        "max_reject_fraction": float(qcfg.max_reject_fraction),
        "reject_fraction_exceeded": bool(reject_fraction_exceeded),
        "distribution_failures": distribution_failures,
        "quality_rows": quality_rows,
        "prompt_stats": {
            "num_unique_prompts": len(prompt_counter),
            "top_prompt_count": max(prompt_counter.values()) if prompt_counter else 0,
        },
        "quality_config": asdict(qcfg),
    }
    write_json(quality_report, dataset_dir / "dataset_quality_report.json")

    quarantine_manifest = {
        "dataset_tag": str(dataset_tag or ""),
        "num_rejected_clips": len(rejected_rows),
        "rejections": rejected_rows,
    }
    write_json(quarantine_manifest, quarantine_dir / "quarantine_manifest.json")

    if leakage_index_path is not None and new_fingerprints:
        merged = dict(leakage_index)
        merged.update(new_fingerprints)
        _write_leakage_index(leakage_index_path, merged)

    dataset_info = {
        "name": f"blueprint_{facility_name}",
        "description": f"Site-adapted training data for {facility_name}",
        "num_videos": len(csv_rows),
        "source": "BlueprintValidation pipeline",
        "num_rejected_videos": len(rejected_rows),
    }
    write_json(dataset_info, dataset_dir / "dataset_info.json")

    provenance = build_provenance_stamp(
        stage=stage_name,
        config_obj=config_obj,
        input_paths=[enriched_manifest_path],
        output_paths=[csv_path] if csv_path.exists() else [],
        extra={
            "dataset_tag": str(dataset_tag or ""),
            "quality_report_path": str(dataset_dir / "dataset_quality_report.json"),
            "quarantine_manifest_path": str(quarantine_dir / "quarantine_manifest.json"),
        },
    )
    write_json(provenance, dataset_dir / "provenance.json")

    if not csv_rows:
        sample = rejected_rows[0]["detail"] if rejected_rows else "no accepted clips"
        raise RuntimeError(f"Dataset assembly produced no valid clips. Example rejection: {sample}")

    hard_fail_reasons: List[str] = []
    if distribution_failures:
        hard_fail_reasons.append("distribution_checks_failed")
    if reject_fraction_exceeded:
        hard_fail_reasons.append("reject_fraction_exceeded")
    if rejected_rows and bool(qcfg.fail_on_rejections):
        hard_fail_reasons.append("quality_rejections_present")
    if hard_fail_reasons:
        example = rejected_rows[0]["detail"] if rejected_rows else distribution_failures[0]
        raise RuntimeError(
            "Dataset quality gate failed "
            f"(reasons={','.join(hard_fail_reasons)}). "
            f"Accepted={len(csv_rows)} Rejected={len(rejected_rows)}. "
            f"Example rejection: {example}"
        )

    logger.info(
        "Built dataset with %d videos at %s (rejected=%d)",
        len(csv_rows),
        dataset_dir,
        len(rejected_rows),
    )
    return dataset_dir
