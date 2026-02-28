"""Minimal SplatSim-style interaction clip generation using PyBullet."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ..common import read_json, write_json
from ..config import FacilityConfig, ValidationConfig


def run_splatsim_pybullet_backend(
    config: ValidationConfig,
    facility: FacilityConfig,
    stage_dir: Path,
    source_manifest_path: Path,
) -> Dict:
    """Generate physics-validated interaction clips for manipulation zones."""
    source_manifest = read_json(source_manifest_path)
    source_clips = list(source_manifest.get("clips", []))
    manifest_path = stage_dir / "interaction_manifest.json"

    if not source_clips:
        return {
            "status": "failed",
            "reason": "no_source_clips",
            "manifest_path": str(manifest_path),
            "num_source_clips": 0,
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    zones = list(facility.manipulation_zones or [])
    if not zones:
        return {
            "status": "failed",
            "reason": "no_manipulation_zones",
            "manifest_path": str(manifest_path),
            "num_source_clips": len(source_clips),
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    try:
        import cv2
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"opencv_unavailable:{exc}",
            "manifest_path": str(manifest_path),
            "num_source_clips": len(source_clips),
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    try:
        import pybullet as pb
        import pybullet_data
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"pybullet_unavailable:{exc}",
            "manifest_path": str(manifest_path),
            "num_source_clips": len(source_clips),
            "num_generated_clips": 0,
            "num_successful_rollouts": 0,
            "fallback_used": False,
        }

    generated: List[Dict] = []
    successful_rollouts = 0
    per_zone_success = defaultdict(int)
    rollouts_per_zone = max(1, int(config.splatsim.per_zone_rollouts))

    for zone_idx, zone in enumerate(zones):
        for rollout_idx in range(rollouts_per_zone):
            source_clip = source_clips[(zone_idx * rollouts_per_zone + rollout_idx) % len(source_clips)]
            source_video = Path(str(source_clip.get("video_path", "")))
            if not source_video.exists():
                continue

            sim = _simulate_episode(
                pb=pb,
                pybullet_data=pybullet_data,
                approach_point=zone.approach_point,
                target_point=zone.target_point,
                horizon_steps=max(4, int(config.splatsim.horizon_steps)),
            )
            if sim["success"]:
                successful_rollouts += 1
                per_zone_success[zone.name] += 1

            clip_name = f"{source_clip.get('clip_name', 'clip')}_s1e_{zone.name}_{rollout_idx:02d}"
            out_video = stage_dir / f"{clip_name}.mp4"
            ok = _annotate_interaction_video(
                cv2=cv2,
                source_video=source_video,
                output_video=out_video,
                trajectory=sim["object_positions"],
                target_point=zone.target_point,
                zone_name=zone.name,
                success=sim["success"],
            )
            if not ok:
                continue

            depth_video = str(source_clip.get("depth_video_path", ""))
            generated.append(
                {
                    "clip_name": clip_name,
                    "path_type": source_clip.get("path_type", "interaction"),
                    "clip_index": source_clip.get("clip_index", -1),
                    "num_frames": source_clip.get("num_frames"),
                    "resolution": source_clip.get("resolution"),
                    "fps": source_clip.get("fps", config.render.fps),
                    "video_path": str(out_video),
                    "depth_video_path": depth_video,
                    "source_clip_name": source_clip.get("clip_name", ""),
                    "source_camera_path": source_clip.get("camera_path", ""),
                    "augmentation_type": "splatsim_interaction",
                    "zone_name": zone.name,
                    "backend_used": "pybullet",
                    "success_flags": {
                        "task_success": bool(sim["success"]),
                    },
                }
            )

    required_successes = max(0, int(config.splatsim.min_successful_rollouts_per_zone))
    zone_requirements_met = all(
        per_zone_success.get(zone.name, 0) >= required_successes for zone in zones
    )

    manifest = {
        "facility": facility.name,
        "source_manifest": str(source_manifest_path),
        "augmentation_type": "splatsim_interaction",
        "backend_used": "pybullet",
        "fallback_used": False,
        "num_source_clips": len(source_clips),
        "num_generated_clips": len(generated),
        "num_successful_rollouts": successful_rollouts,
        "clips": source_clips + generated,
    }
    write_json(manifest, manifest_path)

    if not generated:
        return {
            "status": "failed",
            "reason": "no_generated_clips",
            "manifest_path": str(manifest_path),
            "num_source_clips": len(source_clips),
            "num_generated_clips": 0,
            "num_successful_rollouts": successful_rollouts,
            "fallback_used": False,
        }
    if not zone_requirements_met:
        return {
            "status": "failed",
            "reason": "min_successful_rollouts_per_zone_not_met",
            "manifest_path": str(manifest_path),
            "num_source_clips": len(source_clips),
            "num_generated_clips": len(generated),
            "num_successful_rollouts": successful_rollouts,
            "fallback_used": False,
        }

    return {
        "status": "success",
        "reason": "ok",
        "manifest_path": str(manifest_path),
        "num_source_clips": len(source_clips),
        "num_generated_clips": len(generated),
        "num_successful_rollouts": successful_rollouts,
        "fallback_used": False,
    }


def _simulate_episode(
    pb,
    pybullet_data,
    approach_point: List[float],
    target_point: List[float],
    horizon_steps: int,
) -> Dict:
    client = pb.connect(pb.DIRECT)
    try:
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        pb.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        pb.loadURDF("plane.urdf", physicsClientId=client)

        half_obj = [0.05, 0.04, 0.04]
        col_obj = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_obj, physicsClientId=client)
        vis_obj = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=half_obj,
            rgbaColor=[0.9, 0.3, 0.2, 1.0],
            physicsClientId=client,
        )
        obj_start = [float(approach_point[0]), float(approach_point[1]), max(0.08, float(approach_point[2]))]
        obj_id = pb.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col_obj,
            baseVisualShapeIndex=vis_obj,
            basePosition=obj_start,
            physicsClientId=client,
        )

        col_gripper = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.03, physicsClientId=client)
        vis_gripper = pb.createVisualShape(
            pb.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[0.2, 0.8, 0.9, 1.0],
            physicsClientId=client,
        )
        grip_id = pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_gripper,
            baseVisualShapeIndex=vis_gripper,
            basePosition=[obj_start[0], obj_start[1], obj_start[2] + 0.1],
            physicsClientId=client,
        )

        trace: List[List[float]] = []
        for step in range(horizon_steps):
            alpha = step / max(horizon_steps - 1, 1)
            gx = float(approach_point[0]) * (1.0 - alpha) + float(target_point[0]) * alpha
            gy = float(approach_point[1]) * (1.0 - alpha) + float(target_point[1]) * alpha
            gz = max(0.1, float(target_point[2]) + 0.12)
            pb.resetBasePositionAndOrientation(
                grip_id, [gx, gy, gz], [0.0, 0.0, 0.0, 1.0], physicsClientId=client
            )

            op, _ = pb.getBasePositionAndOrientation(obj_id, physicsClientId=client)
            dx, dy, dz = gx - op[0], gy - op[1], gz - op[2]
            force = [35.0 * dx, 35.0 * dy, 45.0 * dz]
            pb.applyExternalForce(
                obj_id,
                -1,
                force,
                op,
                pb.WORLD_FRAME,
                physicsClientId=client,
            )

            for _ in range(4):
                pb.stepSimulation(physicsClientId=client)

            op, _ = pb.getBasePositionAndOrientation(obj_id, physicsClientId=client)
            trace.append([float(op[0]), float(op[1]), float(op[2])])

        final = trace[-1] if trace else obj_start
        dxy = ((final[0] - float(target_point[0])) ** 2 + (final[1] - float(target_point[1])) ** 2) ** 0.5
        success = dxy <= 0.08 and final[2] >= max(0.05, float(target_point[2]) - 0.05)
        return {"success": bool(success), "object_positions": trace}
    finally:
        pb.disconnect(physicsClientId=client)


def _annotate_interaction_video(
    cv2,
    source_video: Path,
    output_video: Path,
    trajectory: List[List[float]],
    target_point: List[float],
    zone_name: str,
    success: bool,
) -> bool:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        cap.release()
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    if width <= 0 or height <= 0:
        cap.release()
        return False

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        writer.release()
        return False

    frame_idx = 0
    n = max(1, len(trajectory))
    color = (60, 210, 90) if success else (40, 40, 220)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_idx = min(n - 1, int((frame_idx / max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)) * (n - 1)))
        obj = trajectory[t_idx] if trajectory else [float(target_point[0]), float(target_point[1]), 0.0]
        px, py = _project_to_frame(
            width=width,
            height=height,
            object_x=float(obj[0]),
            object_y=float(obj[1]),
            target_x=float(target_point[0]),
            target_y=float(target_point[1]),
        )
        cv2.circle(frame, (px, py), 8, color, -1)
        cv2.putText(
            frame,
            f"S1e zone={zone_name} {'OK' if success else 'MISS'}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


def _project_to_frame(
    width: int,
    height: int,
    object_x: float,
    object_y: float,
    target_x: float,
    target_y: float,
) -> tuple[int, int]:
    scale = 70.0
    px = int(round((width / 2.0) + (object_x - target_x) * scale))
    py = int(round((height * 0.65) - (object_y - target_y) * scale))
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    return px, py

