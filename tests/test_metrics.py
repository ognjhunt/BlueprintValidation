"""Tests for image quality metrics."""

import numpy as np


def test_psnr_identical():
    from blueprint_validation.evaluation.metrics import compute_psnr

    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    psnr = compute_psnr(img, img)
    assert psnr == float("inf")


def test_psnr_different():
    from blueprint_validation.evaluation.metrics import compute_psnr

    img1 = np.zeros((64, 64, 3), dtype=np.uint8)
    img2 = np.ones((64, 64, 3), dtype=np.uint8) * 255
    psnr = compute_psnr(img1, img2)
    assert psnr == 0.0  # Maximum difference, PSNR = 0


def test_psnr_similar():
    from blueprint_validation.evaluation.metrics import compute_psnr

    rng = np.random.default_rng(42)
    img1 = rng.integers(100, 200, (64, 64, 3), dtype=np.uint8)
    img2 = np.clip(img1.astype(int) + rng.integers(-5, 5, img1.shape), 0, 255).astype(np.uint8)
    psnr = compute_psnr(img1, img2)
    assert psnr > 30  # Similar images should have high PSNR


def test_compute_video_metrics_psnr_only():
    from blueprint_validation.evaluation.metrics import compute_video_metrics

    rng = np.random.default_rng(42)
    frames1 = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
    frames2 = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(5)]

    metrics = compute_video_metrics(frames1, frames2, metrics=["psnr"])
    assert metrics.mean_psnr is not None
    assert metrics.mean_psnr > 0
    assert metrics.mean_ssim is None
    assert metrics.mean_lpips is None


def test_video_metrics_to_dict():
    from blueprint_validation.evaluation.metrics import VideoMetrics

    vm = VideoMetrics(mean_psnr=25.123456, mean_ssim=0.951234)
    d = vm.to_dict()
    assert d["mean_psnr"] == 25.1235
    assert d["mean_ssim"] == 0.9512
    assert "mean_lpips" not in d
