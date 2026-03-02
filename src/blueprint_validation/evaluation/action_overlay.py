"""Visualization helpers for WM-only manipulation evaluation overlays."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..common import get_logger

logger = get_logger("evaluation.action_overlay")


def overlay_scripted_trace_on_video(
    *,
    input_video_path: Path,
    output_video_path: Path,
    action_sequence: Sequence[Sequence[float]],
    target_label: str | None = None,
) -> Path:
    """Render a deterministic overlay of scripted action progression on a rollout video."""
    import cv2

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open rollout video for overlay: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 10.0

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(f"No frames available for overlay: {input_video_path}")

    height, width = first.shape[:2]
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open overlay writer: {output_video_path}")

    # Deterministic target marker position from label hash.
    digest = hashlib.md5(str(target_label or "target").encode("utf-8")).hexdigest()
    tx = int((int(digest[:6], 16) / 0xFFFFFF) * max(1, width - 1))
    ty = int((int(digest[6:12], 16) / 0xFFFFFF) * max(1, height - 1))
    target_px = np.asarray([tx, ty], dtype=np.float32)

    pos = np.asarray([width * 0.5, height * 0.65], dtype=np.float32)
    trail: List[np.ndarray] = []
    actions = list(action_sequence)
    frame_idx = 0
    try:
        frame = first
        while True:
            action = actions[min(frame_idx, max(0, len(actions) - 1))] if actions else []
            dx = float(action[0]) if len(action) > 0 else 0.0
            dy = float(action[1]) if len(action) > 1 else 0.0
            dz = float(action[2]) if len(action) > 2 else 0.0
            grip = float(action[6]) if len(action) > 6 else -1.0

            # 2D proxy motion from action deltas.
            pos[0] += np.clip(dx, -0.12, 0.12) * 120.0
            pos[1] += np.clip(dy, -0.12, 0.12) * 120.0
            pos[0] = float(np.clip(pos[0], 6.0, width - 6.0))
            pos[1] = float(np.clip(pos[1], 6.0, height - 6.0))
            trail.append(pos.copy())
            if len(trail) > 20:
                trail.pop(0)

            # Draw target marker.
            cv2.circle(frame, (int(target_px[0]), int(target_px[1])), 8, (0, 215, 255), 2)
            cv2.line(
                frame,
                (int(target_px[0]) - 10, int(target_px[1])),
                (int(target_px[0]) + 10, int(target_px[1])),
                (0, 215, 255),
                1,
            )
            cv2.line(
                frame,
                (int(target_px[0]), int(target_px[1]) - 10),
                (int(target_px[0]), int(target_px[1]) + 10),
                (0, 215, 255),
                1,
            )

            # Draw motion trail and end-effector proxy.
            if len(trail) >= 2:
                for i in range(1, len(trail)):
                    p0 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                    p1 = (int(trail[i][0]), int(trail[i][1]))
                    cv2.line(frame, p0, p1, (255, 180, 50), 1)

            # Gripper state proxy.
            grip_color = (0, 200, 0) if grip > 0 else (40, 40, 220)
            radius = int(7 + np.clip(abs(dz), 0.0, 0.12) * 35)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), radius, grip_color, 2)
            cv2.putText(
                frame,
                "WM overlay",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_idx += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                break
    finally:
        cap.release()
        writer.release()

    return output_video_path
