"""Stage 1b: Composite kinematic robot arm renders into scene videos."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..common import StageResult, get_logger, read_json, write_json
from ..config import FacilityConfig, ValidationConfig
from ..synthetic.robot_compositor import composite_robot_arm_into_clip
from .base import PipelineStage

logger = get_logger("stages.s1b_robot_composite")


class RobotCompositeStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s1b_robot_composite"

    @property
    def description(self) -> str:
        return "Composite URDF-driven robot arm into rendered clips with geometry checks"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del facility, previous_results
        if not config.robot_composite.enabled:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="robot_composite.enabled=false",
            )
        if config.robot_composite.urdf_path is None:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="robot_composite.urdf_path is required when robot_composite.enabled=true",
            )

        render_manifest_path = work_dir / "renders" / "render_manifest.json"
        if not render_manifest_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail="Render manifest missing. Run Stage 1 first.",
            )
        render_manifest = read_json(render_manifest_path)
        out_dir = work_dir / "robot_composite"
        out_dir.mkdir(parents=True, exist_ok=True)

        composited_clips: List[dict] = []
        metrics: List[dict] = []
        for clip in render_manifest.get("clips", []):
            clip_name = clip["clip_name"]
            in_video = Path(clip["video_path"])
            cam_json = Path(clip.get("camera_path", ""))
            if not in_video.exists() or not cam_json.exists():
                logger.warning("Skipping clip without video or camera path: %s", clip_name)
                continue
            out_video = out_dir / f"{clip_name}_robot.mp4"
            result = composite_robot_arm_into_clip(
                input_video=in_video,
                output_video=out_video,
                camera_path_json=cam_json,
                urdf_path=config.robot_composite.urdf_path,
                base_xyz=config.robot_composite.base_xyz,
                base_rpy=config.robot_composite.base_rpy,
                start_joints=config.robot_composite.start_joint_positions,
                end_joints=config.robot_composite.end_joint_positions,
                line_color_bgr=tuple(config.robot_composite.line_color_bgr),
                line_thickness=config.robot_composite.line_thickness,
                min_visible_joint_ratio=config.robot_composite.min_visible_joint_ratio,
                min_consistency_score=config.robot_composite.min_consistency_score,
                end_effector_link=config.robot_composite.end_effector_link,
            )
            metrics.append(result.to_dict())
            if not result.passed:
                continue
            updated = dict(clip)
            updated["video_path"] = str(out_video)
            updated["geometry_consistency_score"] = result.geometry_consistency_score
            updated["mean_visible_joint_ratio"] = result.mean_visible_joint_ratio
            composited_clips.append(updated)

        manifest = dict(render_manifest)
        manifest["clips"] = composited_clips
        manifest["num_clips"] = len(composited_clips)
        manifest["robot_composite_metrics"] = metrics
        manifest_path = out_dir / "composited_manifest.json"
        write_json(manifest, manifest_path)

        if not composited_clips:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                outputs={"manifest_path": str(manifest_path)},
                detail="No clips passed geometry-consistency checks in robot compositing stage.",
            )
        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "composite_dir": str(out_dir),
                "manifest_path": str(manifest_path),
            },
            metrics={
                "num_input_clips": len(render_manifest.get("clips", [])),
                "num_output_clips": len(composited_clips),
                "num_filtered_out": len(render_manifest.get("clips", [])) - len(composited_clips),
            },
        )
