"""CPU-only data quality checks for training dataset assembly."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..common import get_logger

logger = get_logger("training.data_quality")

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_GENERIC_PROMPT_SUBSTRINGS: tuple[str, ...] = (
    "placeholder",
    "describe the scene",
    "generic prompt",
    "lorem ipsum",
    "n/a",
)


@dataclass(frozen=True)
class PromptLintConfig:
    enabled: bool = True
    min_chars: int = 8
    min_tokens: int = 2
    min_unique_token_ratio: float = 0.35
    allow_generic_substrings: bool = False


@dataclass(frozen=True)
class TemporalGateConfig:
    enabled: bool = True
    min_frames_for_check: int = 8
    max_frames_to_sample: int = 96
    min_mean_interframe_delta: float = 1.5
    max_freeze_ratio: float = 0.70
    max_abrupt_cut_ratio: float = 0.35
    max_blockiness_score: float = 0.45


def normalize_prompt(prompt: str) -> str:
    text = str(prompt or "").strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def lint_prompt(prompt: str, cfg: PromptLintConfig) -> List[str]:
    if not bool(cfg.enabled):
        return []
    text = str(prompt or "").strip()
    norm = normalize_prompt(text)
    reasons: List[str] = []
    if len(text) < int(cfg.min_chars):
        reasons.append("prompt_too_short_chars")
    tokens = _TOKEN_RE.findall(norm)
    if len(tokens) < int(cfg.min_tokens):
        reasons.append("prompt_too_short_tokens")
    if tokens:
        unique_ratio = float(len(set(tokens))) / float(len(tokens))
        if unique_ratio < float(cfg.min_unique_token_ratio):
            reasons.append("prompt_low_token_diversity")
    if not bool(cfg.allow_generic_substrings):
        for needle in _GENERIC_PROMPT_SUBSTRINGS:
            if needle in norm:
                reasons.append("prompt_generic_template")
                break
    if not norm:
        reasons.append("prompt_empty")
    return reasons


def _average_hash(frame_rgb: np.ndarray, *, hash_size: int = 8) -> str:
    import cv2

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = float(np.mean(resized))
    bits = (resized >= mean_val).astype(np.uint8).flatten()
    # pack bits into a compact string
    return "".join("1" if int(v) else "0" for v in bits.tolist())


def fingerprint_video_content(video_path: Path, *, max_frames: int = 12) -> str:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"missing::{video_path}"
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            total = max(1, int(max_frames))
        n = max(1, min(int(max_frames), total))
        sample_positions = np.linspace(0, max(0, total - 1), n, dtype=np.int64).tolist()
        hashes: List[str] = []
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(pos)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hashes.append(_average_hash(rgb))
        payload = {
            "video_path": str(video_path),
            "total_frames": total,
            "sample_count": len(hashes),
            "hashes": hashes,
        }
        blob = str(payload).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()
    finally:
        cap.release()


def _estimate_blockiness(gray: np.ndarray, *, block: int = 8) -> float:
    # Approximate blockiness via boundary-edge energy at codec block boundaries.
    h, w = gray.shape[:2]
    if h < block * 2 or w < block * 2:
        return 0.0
    v_edges = []
    for x in range(block, w - 1, block):
        edge = np.abs(gray[:, x].astype(np.float32) - gray[:, x - 1].astype(np.float32))
        v_edges.append(float(np.mean(edge)))
    h_edges = []
    for y in range(block, h - 1, block):
        edge = np.abs(gray[y, :].astype(np.float32) - gray[y - 1, :].astype(np.float32))
        h_edges.append(float(np.mean(edge)))
    if not v_edges and not h_edges:
        return 0.0
    # normalize to [0,1]-ish range
    return float((np.mean(v_edges) + np.mean(h_edges)) / (2.0 * 255.0))


def analyze_temporal_quality(video_path: Path, cfg: TemporalGateConfig) -> Dict[str, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "decoded_frames": 0.0,
            "mean_interframe_delta": 0.0,
            "freeze_ratio": 1.0,
            "abrupt_cut_ratio": 0.0,
            "blockiness_score": 1.0,
        }
    inter_deltas: List[float] = []
    blockiness: List[float] = []
    decoded = 0
    prev_gray = None
    try:
        while decoded < max(1, int(cfg.max_frames_to_sample)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blockiness.append(_estimate_blockiness(gray))
            if prev_gray is not None:
                inter_deltas.append(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray))))
            prev_gray = gray
            decoded += 1
    finally:
        cap.release()

    if not inter_deltas:
        return {
            "decoded_frames": float(decoded),
            "mean_interframe_delta": 0.0,
            "freeze_ratio": 1.0,
            "abrupt_cut_ratio": 0.0,
            "blockiness_score": float(np.mean(blockiness)) if blockiness else 0.0,
        }
    inter_arr = np.asarray(inter_deltas, dtype=np.float32)
    freeze_ratio = float(np.mean(inter_arr < 0.75))
    abrupt_cut_ratio = float(np.mean(inter_arr > 35.0))
    return {
        "decoded_frames": float(decoded),
        "mean_interframe_delta": float(np.mean(inter_arr)),
        "freeze_ratio": freeze_ratio,
        "abrupt_cut_ratio": abrupt_cut_ratio,
        "blockiness_score": float(np.mean(blockiness)) if blockiness else 0.0,
    }


def temporal_reject_reasons(metrics: Dict[str, float], cfg: TemporalGateConfig) -> List[str]:
    if not bool(cfg.enabled):
        return []
    decoded = int(metrics.get("decoded_frames", 0.0) or 0.0)
    if decoded < int(cfg.min_frames_for_check):
        return []
    reasons: List[str] = []
    if float(metrics.get("mean_interframe_delta", 0.0)) < float(cfg.min_mean_interframe_delta):
        reasons.append("temporal_low_motion")
    if float(metrics.get("freeze_ratio", 0.0)) > float(cfg.max_freeze_ratio):
        reasons.append("temporal_frozen_video")
    if float(metrics.get("abrupt_cut_ratio", 0.0)) > float(cfg.max_abrupt_cut_ratio):
        reasons.append("temporal_abrupt_cuts")
    if float(metrics.get("blockiness_score", 0.0)) > float(cfg.max_blockiness_score):
        reasons.append("temporal_compression_artifacts")
    return reasons


def summarize_distribution(values: Sequence[str]) -> Dict[str, float]:
    if not values:
        return {"unique": 0.0, "dominant_fraction": 0.0}
    counts = Counter(str(v or "") for v in values)
    dominant = max(counts.values()) if counts else 0
    return {
        "unique": float(len(counts)),
        "dominant_fraction": float(dominant) / float(len(values)),
    }
