"""Blueprint video dataset adapter for DreamDojo Stage 3.

This mirrors DreamDojo's non-lerobot VideoDataset output schema but uses OpenCV
decode to avoid torchcodec runtime coupling.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _filter_video_files(file_names: Iterable[Path], xdof: bool = False) -> List[Path]:
    if xdof:
        return [
            f
            for f in file_names
            if "left" not in str(f).lower()
            and "right" not in str(f).lower()
            and "resize" not in str(f).lower()
            and "pad" not in str(f).lower()
            and "320_240" in str(f).lower()
        ]
    return [
        f
        for f in file_names
        if "left" not in str(f).lower()
        and "right" not in str(f).lower()
        and "resize" not in str(f).lower()
        and "pad" not in str(f).lower()
    ]


class BlueprintVideoActionDataset(Dataset):
    """OpenCV-backed replacement for DreamDojo's VideoDataset branch."""

    def __init__(
        self,
        args=None,
        dataset_path=None,
        num_frames: int = 13,
        data_split: str = "train",
        fps: int = 30,
        randomize: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if args is not None:
            dataset_path = getattr(args, "dataset_path", dataset_path)
            num_frames = int(getattr(args, "num_frames", num_frames))
        if dataset_path is None:
            raise ValueError("dataset_path is required for BlueprintVideoActionDataset")

        if isinstance(dataset_path, list):
            paths = [Path(str(p).strip()) for p in dataset_path]
        else:
            paths = [Path(str(p).strip()) for p in str(dataset_path).split(",") if str(p).strip()]
        if not paths:
            raise ValueError("No dataset paths provided")

        files: List[Path] = []
        for root in paths:
            files.extend(root.rglob("*.mp4"))
        if not files:
            raise ValueError(f"No video files found under: {paths}")

        first_path = str(paths[0]).lower()
        self.episodes = _filter_video_files(files, xdof="xdof" in first_path)
        if not self.episodes:
            raise ValueError(f"No usable MP4 files after filtering under: {paths}")

        self.dataset_label = "human_video"
        self.fps = int(fps)
        self.num_frames = int(num_frames)
        self.randomize = bool(randomize and data_split == "train")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return len(self.episodes)

    def _decode_contiguous_window(self, video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= self.num_frames:
                start = 0
            else:
                start = random.randint(0, frame_count - self.num_frames - 1) if self.randomize else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames: List[np.ndarray] = []
            for _ in range(self.num_frames):
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            if len(frames) != self.num_frames:
                raise RuntimeError(
                    f"Decoded {len(frames)} frames, expected {self.num_frames} from {video_path}"
                )
            return np.stack(frames, axis=0)  # T,H,W,C uint8
        finally:
            cap.release()

    def __getitem__(self, idx: int) -> Dict:
        last_error: Exception | None = None
        for _ in range(12):
            try:
                video_path = self.episodes[int(idx) % len(self.episodes)]
                frames = self._decode_contiguous_window(video_path)
                # T,H,W,C -> C,T,H,W
                video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).to(torch.float32)

                target_ratio = 640.0 / 480.0
                h = video_tensor.shape[2]
                w = video_tensor.shape[3]
                ratio = w / h
                if ratio > target_ratio:
                    target_h = h
                    target_w = int(h * target_ratio)
                elif ratio < target_ratio:
                    target_h = int(w / target_ratio)
                    target_w = w
                else:
                    target_h, target_w = h, w
                h_crop = (h - target_h) // 2
                w_crop = (w - target_w) // 2
                video_tensor = video_tensor[
                    :, :, h_crop : h_crop + target_h, w_crop : w_crop + target_w
                ]
                video_tensor = F.interpolate(
                    video_tensor, (480, 640), mode="bilinear", align_corners=False
                )
                video_tensor = torch.clamp(video_tensor, 0.0, 255.0).to(torch.uint8)

                lam_frames = F.interpolate(
                    video_tensor.to(torch.float32), (240, 320), mode="bilinear", align_corners=False
                )
                lam_frames = torch.clamp(lam_frames / 255.0, 0.0, 1.0)
                lam_frames = torch.repeat_interleave(lam_frames, 2, dim=1)[:, 1:-1, :, :]
                lam_frames = lam_frames.permute(1, 2, 3, 0).contiguous()  # T,H,W,C

                gt_actions = torch.zeros(self.num_frames - 1, 352, dtype=torch.float32)
                latent_actions = torch.ones(self.num_frames - 1, 32, dtype=torch.float32)
                action_seq = torch.cat([gt_actions, latent_actions], dim=-1)
                key = torch.ones((1, 29), dtype=torch.float32) * int(idx)

                return {
                    "video": video_tensor,
                    "lam_video": lam_frames,
                    "action": action_seq,
                    "dataset": self.dataset_label,
                    "fps": self.fps,
                    "num_frames": self.num_frames,
                    "__key__": key,
                    "padding_mask": torch.zeros(1, 256, 256, device=self._device),
                    "image_size": 256 * torch.ones(4, device=self._device),
                    "ai_caption": "",
                }
            except Exception as exc:  # pragma: no cover - stochastic retry branch
                last_error = exc
                idx = random.randint(0, len(self.episodes) - 1)
        raise RuntimeError(f"Failed to decode sample after retries: {last_error}")
