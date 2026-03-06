"""Deterministic rollout-state sidecar for fixed-world claim evaluation.

This module provides a structured fallback when the underlying world-model
wrapper cannot expose task-state directly. The state is derived from rollout
actions plus task/start metadata so the canonical claim protocol can use a
deterministic primary endpoint instead of VLM-only heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def _xyz(value: object, default: list[float] | None = None) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
        except Exception:
            pass
    if default is None:
        default = [0.0, 0.0, 0.0]
    return np.asarray(default[:3], dtype=np.float64)


def _unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return np.asarray(fallback, dtype=np.float64)
    arr = arr[:3]
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-8:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass
class DeterministicRolloutStateProxy:
    task_prompt: str
    task_family: str
    goal_region_id: str
    start_region_id: str
    target_instance_id: str | None
    target_label: str | None
    goal_point: np.ndarray | None
    target_point: np.ndarray | None
    position_world: np.ndarray
    forward_world: np.ndarray
    right_world: np.ndarray
    up_world: np.ndarray
    ee_position_world: np.ndarray
    goal_radius_m: float
    grasp_radius_m: float
    lift_height_m: float
    joint_position: float = 0.0
    invalid_collision: bool = False
    target_dropped: bool = False
    attached: bool = False
    object_position_world: np.ndarray | None = None
    object_rest_position_world: np.ndarray | None = None
    object_start_z: float = 0.0
    grasp_acquired: bool = False
    lifted_clear: bool = False
    placed_in_target: bool = False
    stable_after_place: bool = False
    stable_steps: int = 0
    navigation_progress_m: float = 0.0

    @classmethod
    def from_context(
        cls,
        *,
        task_prompt: str,
        task_spec: Dict[str, object] | None,
        rollout_context: Dict[str, object] | None,
    ) -> "DeterministicRolloutStateProxy":
        spec = dict(task_spec or {})
        ctx = dict(rollout_context or {})
        predicate = dict(spec.get("success_predicate", {}) or {})
        initial_camera = dict(ctx.get("initial_camera", {}) or {})
        path_context = dict(ctx.get("path_context", {}) or {})

        position_world = _xyz(initial_camera.get("position"), [0.0, 0.0, 0.8])
        forward_world = _unit(_xyz(initial_camera.get("forward"), [1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]))
        right_world = _unit(_xyz(initial_camera.get("right"), [0.0, 1.0, 0.0]), np.asarray([0.0, 1.0, 0.0]))
        up_world = _unit(_xyz(initial_camera.get("up"), [0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, 1.0]))
        ee_position_world = position_world + (0.35 * forward_world) - (0.15 * up_world)

        target_point = _maybe_xyz(predicate.get("target_center_xyz"))
        if target_point is None:
            target_point = _maybe_xyz(spec.get("target_center_xyz"))
        if target_point is None:
            target_point = _maybe_xyz(path_context.get("approach_point"))
        if target_point is None:
            target_point = ee_position_world + (0.25 * forward_world)

        goal_point = _maybe_xyz(predicate.get("goal_point_xyz"))
        if goal_point is None:
            goal_point = _maybe_xyz(spec.get("goal_point_xyz"))
        if goal_point is None and str(spec.get("task_family", "")).strip().lower() == "navigation":
            goal_point = np.asarray(target_point, dtype=np.float64)
        if goal_point is None and str(spec.get("task_family", "")).strip().lower() == "manipulation":
            goal_point = np.asarray(target_point, dtype=np.float64) + np.asarray([0.35, 0.0, 0.10], dtype=np.float64)

        object_position_world = np.asarray(target_point, dtype=np.float64)
        return cls(
            task_prompt=str(task_prompt or ""),
            task_family=str(spec.get("task_family", "") or _infer_task_family(task_prompt)).strip().lower(),
            goal_region_id=str(predicate.get("goal_region_id", spec.get("goal_region_id", "")) or "").strip(),
            start_region_id=str(ctx.get("start_region_id", "") or "").strip(),
            target_instance_id=_clean_optional_text(spec.get("target_instance_id") or ctx.get("target_instance_id")),
            target_label=_clean_optional_text(spec.get("target_label") or ctx.get("target_label")),
            goal_point=goal_point,
            target_point=np.asarray(target_point, dtype=np.float64),
            position_world=np.asarray(position_world, dtype=np.float64),
            forward_world=np.asarray(forward_world, dtype=np.float64),
            right_world=np.asarray(right_world, dtype=np.float64),
            up_world=np.asarray(up_world, dtype=np.float64),
            ee_position_world=np.asarray(ee_position_world, dtype=np.float64),
            goal_radius_m=float(predicate.get("goal_radius_m", 0.75 if goal_point is not None else 0.0) or 0.75),
            grasp_radius_m=float(predicate.get("grasp_radius_m", 0.22) or 0.22),
            lift_height_m=float(predicate.get("lift_height_m", 0.14) or 0.14),
            joint_position=float(predicate.get("initial_joint_position", 0.0) or 0.0),
            object_position_world=np.asarray(object_position_world, dtype=np.float64),
            object_rest_position_world=np.asarray(object_position_world, dtype=np.float64),
            object_start_z=float(object_position_world[2]),
        )

    def capture(
        self,
        *,
        action: object | None,
        step_idx: int,
        phase: str,
    ) -> Dict[str, object]:
        del phase
        if action is not None:
            self._advance(action)
        return self._snapshot(step_idx=step_idx)

    def _advance(self, action: object) -> None:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return

        translation = np.zeros((3,), dtype=np.float64)
        translation[: min(3, arr.size)] = arr[: min(3, arr.size)]
        translation_scale = 0.18
        world_delta = (
            (translation[0] * self.forward_world)
            + (translation[1] * self.right_world)
            + (translation[2] * self.up_world)
        ) * translation_scale
        if float(np.linalg.norm(world_delta)) > 0.75:
            self.invalid_collision = True
            world_delta = world_delta / max(float(np.linalg.norm(world_delta)), 1.0e-6) * 0.75

        self.position_world = np.asarray(self.position_world + world_delta, dtype=np.float64)
        self.ee_position_world = np.asarray(self.ee_position_world + world_delta, dtype=np.float64)
        self.navigation_progress_m += max(0.0, float(np.dot(world_delta, self.forward_world)))

        if self.task_family == "articulation":
            drive = float(arr[0]) if arr.size > 0 else 0.0
            if arr.size > 5:
                drive += 0.35 * float(arr[5])
            if arr.size > 6:
                drive += -0.15 if float(arr[6]) < 0.0 else 0.15
            self.joint_position = _clip01(self.joint_position + (0.20 * np.tanh(drive)))

        if self.task_family == "manipulation":
            grip_value = float(arr[-1]) if arr.size > 0 else 0.0
            closed = grip_value <= 0.0
            if not self.attached and self.object_position_world is not None:
                if closed and float(np.linalg.norm(self.ee_position_world - self.object_position_world)) <= self.grasp_radius_m:
                    self.attached = True
                    self.grasp_acquired = True
            if self.attached:
                self.object_position_world = np.asarray(self.ee_position_world, dtype=np.float64)
            if self.object_position_world is not None and (
                float(self.object_position_world[2]) - float(self.object_start_z)
            ) >= self.lift_height_m:
                self.lifted_clear = True

            if self.attached and not closed and self.object_position_world is not None:
                if self.goal_point is not None and float(
                    np.linalg.norm(self.object_position_world - self.goal_point)
                ) <= self.goal_radius_m:
                    self.attached = False
                    self.placed_in_target = True
                    self.stable_steps = 1
                    self.object_rest_position_world = np.asarray(self.object_position_world, dtype=np.float64)
                else:
                    self.attached = False
                    self.target_dropped = True
                    self.object_rest_position_world = np.asarray(self.object_position_world, dtype=np.float64)

            if not self.attached and self.object_position_world is not None:
                reference = self.object_rest_position_world if self.object_rest_position_world is not None else self.object_position_world
                drift = float(np.linalg.norm(self.object_position_world - reference))
                in_goal = bool(
                    self.goal_point is not None
                    and float(np.linalg.norm(self.object_position_world - self.goal_point)) <= self.goal_radius_m
                )
                if self.placed_in_target and in_goal and drift <= 0.02:
                    self.stable_steps += 1
                    if self.stable_steps >= 2:
                        self.stable_after_place = True
                else:
                    self.stable_steps = 0

    def _snapshot(self, *, step_idx: int) -> Dict[str, object]:
        active_region_id = self.start_region_id
        if self.task_family == "navigation":
            reached_goal = False
            if self.goal_point is not None:
                reached_goal = float(np.linalg.norm(self.position_world - self.goal_point)) <= self.goal_radius_m
            else:
                reached_goal = self.navigation_progress_m >= 0.80
            active_region_id = self.goal_region_id if reached_goal and self.goal_region_id else self.start_region_id
        if self.task_family == "manipulation" and self.placed_in_target and self.goal_region_id:
            active_region_id = self.goal_region_id

        joint_positions: Dict[str, float] = {}
        if self.target_instance_id is not None:
            joint_positions[self.target_instance_id] = float(self.joint_position)

        return {
            "state_source": "deterministic_rollout_state_proxy",
            "step_idx": int(step_idx),
            "task_family": self.task_family,
            "active_region_id": active_region_id,
            "joint_position": float(self.joint_position),
            "joint_positions": joint_positions,
            "invalid_collision": bool(self.invalid_collision),
            "target_dropped": bool(self.target_dropped),
            "grasp_acquired": bool(self.grasp_acquired),
            "lifted_clear": bool(self.lifted_clear),
            "placed_in_target": bool(self.placed_in_target),
            "stable_after_place": bool(self.stable_after_place),
            "position_world": self.position_world.astype(float).tolist(),
            "ee_position_world": self.ee_position_world.astype(float).tolist(),
            "object_position_world": (
                None
                if self.object_position_world is None
                else self.object_position_world.astype(float).tolist()
            ),
            "target_instance_id": self.target_instance_id,
            "target_label": self.target_label,
        }


def _maybe_xyz(value: object) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
    except Exception:
        return None


def _clean_optional_text(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _infer_task_family(task_prompt: str) -> str:
    lowered = str(task_prompt or "").lower()
    if any(token in lowered for token in ("pick up", "grasp", "lift", "place", "stack", "regrasp")):
        return "manipulation"
    if any(token in lowered for token in ("open and close", "open ", "close ", "turn on", "turn off")):
        return "articulation"
    if any(token in lowered for token in ("navigate", "approach", "go to", "move toward")):
        return "navigation"
    return "other"
