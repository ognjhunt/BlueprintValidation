"""Image quality metrics: PSNR, SSIM, LPIPS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..common import get_logger

logger = get_logger("evaluation.metrics")


@dataclass
class FrameMetrics:
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None


@dataclass
class VideoMetrics:
    mean_psnr: Optional[float] = None
    mean_ssim: Optional[float] = None
    mean_lpips: Optional[float] = None
    per_frame: List[FrameMetrics] = None

    def to_dict(self) -> dict:
        result = {}
        if self.mean_psnr is not None:
            result["mean_psnr"] = round(self.mean_psnr, 4)
        if self.mean_ssim is not None:
            result["mean_ssim"] = round(self.mean_ssim, 4)
        if self.mean_lpips is not None:
            result["mean_lpips"] = round(self.mean_lpips, 4)
        return result


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim_batch(
    frames1: List[np.ndarray], frames2: List[np.ndarray]
) -> List[float]:
    """Compute SSIM for a batch of frame pairs using torchmetrics."""
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    results = []
    for f1, f2 in zip(frames1, frames2):
        t1 = torch.from_numpy(f1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t2 = torch.from_numpy(f2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        val = ssim_metric(t1, t2)
        results.append(float(val))

    return results


def compute_lpips_batch(
    frames1: List[np.ndarray],
    frames2: List[np.ndarray],
    backbone: str = "alex",
) -> List[float]:
    """Compute LPIPS for a batch of frame pairs."""
    import torch
    import lpips

    loss_fn = lpips.LPIPS(net=backbone)
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    results = []
    for f1, f2 in zip(frames1, frames2):
        t1 = torch.from_numpy(f1).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
        t2 = torch.from_numpy(f2).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
        if torch.cuda.is_available():
            t1, t2 = t1.cuda(), t2.cuda()
        val = loss_fn(t1, t2)
        results.append(float(val.item()))

    return results


def compute_video_metrics(
    frames1: List[np.ndarray],
    frames2: List[np.ndarray],
    metrics: List[str] = None,
    lpips_backbone: str = "alex",
) -> VideoMetrics:
    """Compute all requested metrics between two sets of video frames.

    Both frames lists should have the same length and frame dimensions.
    """
    if metrics is None:
        metrics = ["psnr", "ssim", "lpips"]

    if len(frames1) != len(frames2):
        min_len = min(len(frames1), len(frames2))
        logger.warning(
            "Frame count mismatch (%d vs %d), truncating to %d",
            len(frames1), len(frames2), min_len,
        )
        frames1 = frames1[:min_len]
        frames2 = frames2[:min_len]

    per_frame = [FrameMetrics() for _ in range(len(frames1))]

    # PSNR
    psnr_values = None
    if "psnr" in metrics:
        psnr_values = [compute_psnr(f1, f2) for f1, f2 in zip(frames1, frames2)]
        for i, val in enumerate(psnr_values):
            per_frame[i].psnr = val

    # SSIM
    ssim_values = None
    if "ssim" in metrics:
        ssim_values = compute_ssim_batch(frames1, frames2)
        for i, val in enumerate(ssim_values):
            per_frame[i].ssim = val

    # LPIPS
    lpips_values = None
    if "lpips" in metrics:
        lpips_values = compute_lpips_batch(frames1, frames2, backbone=lpips_backbone)
        for i, val in enumerate(lpips_values):
            per_frame[i].lpips = val

    return VideoMetrics(
        mean_psnr=float(np.mean(psnr_values)) if psnr_values else None,
        mean_ssim=float(np.mean(ssim_values)) if ssim_values else None,
        mean_lpips=float(np.mean(lpips_values)) if lpips_values else None,
        per_frame=per_frame,
    )
