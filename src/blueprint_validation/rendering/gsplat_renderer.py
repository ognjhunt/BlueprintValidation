"""gsplat-based Gaussian splat renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ..common import get_logger
from .camera_paths import CameraPose
from .ply_loader import GaussianSplatData

logger = get_logger("rendering.gsplat_renderer")


@dataclass
class RenderOutput:
    """Output from rendering a camera path."""

    rgb_frames: List[np.ndarray]  # List of (H, W, 3) uint8 arrays
    depth_frames: List[np.ndarray]  # List of (H, W) float32 arrays
    video_path: Optional[Path] = None
    depth_video_path: Optional[Path] = None


def render_frame(
    splat: GaussianSplatData,
    pose: CameraPose,
    background: Optional[torch.Tensor] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a single frame from the Gaussian splat.

    Returns (rgb, depth) where rgb is (H, W, 3) uint8 and depth is (H, W) float32.
    """
    from gsplat import rasterization

    device = splat.means.device

    viewmat = pose.viewmat().unsqueeze(0).to(device)  # (1, 4, 4)
    K = pose.K().unsqueeze(0).to(device)  # (1, 3, 3)

    if background is None:
        background = torch.ones(3, device=device)

    renders, alphas, info = rasterization(
        means=splat.means,
        quats=splat.quats,
        scales=torch.exp(splat.scales),
        opacities=torch.sigmoid(splat.opacities),
        colors=splat.sh_coeffs,
        viewmats=viewmat,
        Ks=K,
        width=pose.width,
        height=pose.height,
        sh_degree=int(np.sqrt(splat.sh_coeffs.shape[1]) - 1),
        backgrounds=background.unsqueeze(0),
        render_mode="RGB+ED",
    )

    # renders shape: (1, H, W, 4) — RGB + expected depth
    rgb = renders[0, :, :, :3].clamp(0, 1).cpu().numpy()
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    depth = renders[0, :, :, 3].cpu().numpy()

    return rgb_uint8, depth


def render_video(
    splat: GaussianSplatData,
    poses: List[CameraPose],
    output_dir: Path,
    clip_name: str = "render",
    fps: int = 10,
    save_frames: bool = False,
) -> RenderOutput:
    """Render a video from a sequence of camera poses.

    Saves RGB video and depth video as MP4. Returns RenderOutput.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_frames = []
    depth_frames = []

    logger.info("Rendering %d frames for clip '%s'", len(poses), clip_name)

    for i, pose in enumerate(poses):
        rgb, depth = render_frame(splat, pose)
        rgb_frames.append(rgb)
        depth_frames.append(depth)

        if save_frames:
            frame_dir = output_dir / f"{clip_name}_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(frame_dir / f"{i:05d}.png")

        if (i + 1) % 10 == 0 or i == len(poses) - 1:
            logger.debug("Rendered frame %d/%d", i + 1, len(poses))

    # Save RGB video
    h, w = rgb_frames[0].shape[:2]
    video_path = output_dir / f"{clip_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for frame in rgb_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    logger.info("Saved RGB video: %s", video_path)

    # Save depth video (normalized to 0-255 for visualization)
    depth_video_path = output_dir / f"{clip_name}_depth.mp4"
    depth_stack = np.stack(depth_frames)
    d_min, d_max = depth_stack.min(), depth_stack.max()
    if d_max > d_min:
        depth_norm = ((depth_stack - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_stack, dtype=np.uint8)

    writer = cv2.VideoWriter(str(depth_video_path), fourcc, fps, (w, h), isColor=False)
    for frame in depth_norm:
        writer.write(frame)
    writer.release()
    logger.info("Saved depth video: %s", depth_video_path)

    return RenderOutput(
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        video_path=video_path,
        depth_video_path=depth_video_path,
    )
