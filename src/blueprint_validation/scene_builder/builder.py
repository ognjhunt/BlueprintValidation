"""Direct PLY-to-scene-package builder for teleop and PolaRiS."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..common import sanitize_filename_component, write_json, write_text_atomic
from ..config import ValidationConfig
from ..warmup import load_ply_means_numpy
from .manifest import ImportedAssetSpec, SceneAssetManifest, load_scene_asset_manifest


@dataclass(frozen=True)
class SceneBuildResult:
    scene_root: Path
    scene_manifest_path: Path
    usd_scene_path: Path
    task_config_path: Path
    isaac_lab_package_root: Path


def build_scene_package(config: ValidationConfig) -> SceneBuildResult:
    builder_cfg = config.scene_builder
    if not bool(builder_cfg.enabled):
        raise RuntimeError("scene_builder.enabled=false")
    if builder_cfg.source_ply_path is None:
        raise RuntimeError("scene_builder.source_ply_path is not set")
    if builder_cfg.asset_manifest_path is None:
        raise RuntimeError("scene_builder.asset_manifest_path is not set")

    manifest = load_scene_asset_manifest(builder_cfg.asset_manifest_path)
    scene_root = builder_cfg.output_scene_root.resolve()
    if scene_root.exists():
        shutil.rmtree(scene_root)
    (scene_root / "assets").mkdir(parents=True, exist_ok=True)
    (scene_root / "usd").mkdir(parents=True, exist_ok=True)
    if bool(builder_cfg.emit_polaris_metadata):
        (scene_root / "geniesim").mkdir(parents=True, exist_ok=True)
    if bool(builder_cfg.emit_isaac_lab):
        (scene_root / "isaac_lab").mkdir(parents=True, exist_ok=True)

    background = _write_static_background_asset(
        source_ply_path=builder_cfg.source_ply_path,
        asset_root=scene_root / "assets" / "static_scene",
        collision_mode=builder_cfg.static_collision_mode,
    )
    copied_assets = [
        _copy_imported_asset(scene_root=scene_root, spec=asset_spec) for asset_spec in manifest.assets
    ]

    scene_manifest_path = scene_root / "assets" / "scene_manifest.json"
    scene_manifest_payload = _build_scene_manifest_payload(
        manifest=manifest,
        background=background,
        copied_assets=copied_assets,
        source_ply_path=builder_cfg.source_ply_path,
    )
    write_json(scene_manifest_payload, scene_manifest_path)

    usd_scene_path = scene_root / "usd" / "scene.usda"
    _write_scene_usda(
        path=usd_scene_path,
        manifest=manifest,
        background=background,
        copied_assets=copied_assets,
    )

    task_config_path = scene_root / "geniesim" / "task_config.json"
    if bool(builder_cfg.emit_polaris_metadata):
        write_json(_build_task_config_payload(manifest), task_config_path)
    else:
        task_config_path = Path("")

    isaac_lab_root = scene_root / "isaac_lab"
    if bool(builder_cfg.emit_isaac_lab):
        _write_isaac_lab_package(
            root=isaac_lab_root,
            scene_id=manifest.scene_id,
            usd_scene_path=usd_scene_path,
            task=manifest.task,
            copied_assets=copied_assets,
            robot_type=builder_cfg.robot_type,
        )

    return SceneBuildResult(
        scene_root=scene_root,
        scene_manifest_path=scene_manifest_path,
        usd_scene_path=usd_scene_path,
        task_config_path=task_config_path,
        isaac_lab_package_root=isaac_lab_root,
    )


@dataclass(frozen=True)
class _BackgroundAsset:
    source_ply_path: Path
    copied_ply_path: Path
    background_usda_path: Path
    bbox_min: List[float]
    bbox_max: List[float]
    center: List[float]
    extents: List[float]
    collision_mode: str


@dataclass(frozen=True)
class _CopiedAsset:
    object_id: str
    label: str
    asset_type: str
    task_role: str
    copied_asset_path: Path
    relative_asset_path: str
    position: List[float]
    rotation_quaternion: List[float]
    scale: List[float]


def _write_static_background_asset(
    *,
    source_ply_path: Path,
    asset_root: Path,
    collision_mode: str,
) -> _BackgroundAsset:
    asset_root.mkdir(parents=True, exist_ok=True)
    means = load_ply_means_numpy(source_ply_path)
    if means.size == 0:
        raise RuntimeError(f"Source PLY contains no points: {source_ply_path}")
    bbox_min = means.min(axis=0).astype(float).tolist()
    bbox_max = means.max(axis=0).astype(float).tolist()
    center = [0.5 * (lo + hi) for lo, hi in zip(bbox_min, bbox_max)]
    extents = [max(0.05, hi - lo) for lo, hi in zip(bbox_min, bbox_max)]
    copied_ply_path = asset_root / "source_scene.ply"
    shutil.copy2(source_ply_path, copied_ply_path)
    background_usda_path = asset_root / "background_shell.usda"
    write_text_atomic(
        background_usda_path,
        _render_background_usda(
            center=center,
            extents=extents,
            collision_mode=collision_mode,
            source_ply_path=copied_ply_path.name,
        ),
    )
    return _BackgroundAsset(
        source_ply_path=source_ply_path.resolve(),
        copied_ply_path=copied_ply_path,
        background_usda_path=background_usda_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center=center,
        extents=extents,
        collision_mode=collision_mode,
    )


def _copy_imported_asset(scene_root: Path, spec: ImportedAssetSpec) -> _CopiedAsset:
    obj_dir = scene_root / "assets" / f"obj_{sanitize_filename_component(spec.object_id)}"
    obj_dir.mkdir(parents=True, exist_ok=True)
    copied_asset_path = obj_dir / spec.asset_path.name
    shutil.copy2(spec.asset_path, copied_asset_path)
    metadata = {
        "object_id": spec.object_id,
        "label": spec.label,
        "asset_type": spec.asset_type,
        "task_role": spec.task_role,
        "source_asset_path": str(spec.asset_path),
        "copied_asset_path": str(copied_asset_path),
        "pose": {
            "position": spec.position,
            "rotation_quaternion": spec.rotation_quaternion,
        },
        "scale": spec.scale,
        "articulation_hints": spec.articulation_hints,
        "collision_hints": spec.collision_hints,
    }
    write_json(metadata, obj_dir / "metadata.json")
    return _CopiedAsset(
        object_id=spec.object_id,
        label=spec.label,
        asset_type=spec.asset_type,
        task_role=spec.task_role,
        copied_asset_path=copied_asset_path,
        relative_asset_path=str(copied_asset_path.relative_to(scene_root)).replace("\\", "/"),
        position=list(spec.position),
        rotation_quaternion=list(spec.rotation_quaternion),
        scale=list(spec.scale),
    )


def _build_scene_manifest_payload(
    *,
    manifest: SceneAssetManifest,
    background: _BackgroundAsset,
    copied_assets: List[_CopiedAsset],
    source_ply_path: Path,
) -> Dict[str, Any]:
    return {
        "version": "v1",
        "scene_id": manifest.scene_id,
        "scene_family": "blueprint_validation_direct_builder",
        "environment_type": "manipulation_scene",
        "scene": {
            "source_ply_path": str(source_ply_path.resolve()),
            "background_asset": {
                "ply_path": str(background.copied_ply_path.relative_to(background.copied_ply_path.parents[2])).replace("\\", "/"),
                "usda_path": str(background.background_usda_path.relative_to(background.background_usda_path.parents[2])).replace("\\", "/"),
                "collision_mode": background.collision_mode,
                "bbox_min": background.bbox_min,
                "bbox_max": background.bbox_max,
            },
        },
        "objects": [
            {
                "id": asset.object_id,
                "name": asset.label,
                "category": asset.label,
                "asset": {
                    "format": asset.copied_asset_path.suffix.lstrip("."),
                    "path": asset.relative_asset_path,
                },
                "sim_role": "interactive" if asset.task_role != "goal_region" else "static",
                "task_role": asset.task_role,
                "transform": {
                    "position": {
                        "x": asset.position[0],
                        "y": asset.position[1],
                        "z": asset.position[2],
                    },
                    "rotation_quaternion": {
                        "w": asset.rotation_quaternion[0],
                        "x": asset.rotation_quaternion[1],
                        "y": asset.rotation_quaternion[2],
                        "z": asset.rotation_quaternion[3],
                    },
                    "scale": {
                        "x": asset.scale[0],
                        "y": asset.scale[1],
                        "z": asset.scale[2],
                    },
                },
            }
            for asset in copied_assets
        ],
        "metadata": {
            "builder": "blueprint_validation.scene_builder",
            "task_id": manifest.task.task_id,
            "task_text": manifest.task.task_text,
        },
    }


def _build_task_config_payload(manifest: SceneAssetManifest) -> Dict[str, Any]:
    return {
        "schema_version": "3.0",
        "scene_id": manifest.scene_id,
        "environment_type": "manipulation_scene",
        "suggested_tasks": [
            {
                "task_type": manifest.task.task_type,
                "target_object": manifest.task.target_object_id,
                "goal_region": manifest.task.goal_object_id,
                "difficulty": "medium",
                "priority": 1,
                "description_hint": manifest.task.task_text,
            }
        ],
        "robot_config": {
            "type": "franka",
            "base_position": [0.0, 0.0, 0.0],
            "workspace_bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]],
        },
        "metadata": {
            "task_template": "pick_place_v1",
            "tasks_total_after_reachability_filter": 1,
        },
    }


def _write_scene_usda(
    *,
    path: Path,
    manifest: SceneAssetManifest,
    background: _BackgroundAsset,
    copied_assets: List[_CopiedAsset],
) -> None:
    lines: List[str] = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "Scene"',
        "    {",
        '        def Xform "Background"',
        "        {",
        f'            prepend references = @../assets/static_scene/{background.background_usda_path.name}@',
        "        }",
    ]
    for asset in copied_assets:
        prim_name = sanitize_filename_component(asset.object_id, fallback="asset")
        lines.extend(
            [
                f'        def Xform "{prim_name}"',
                "        {",
                f'            prepend references = @../{asset.relative_asset_path}@',
                f"            double3 xformOp:translate = ({asset.position[0]}, {asset.position[1]}, {asset.position[2]})",
                "            uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\", \"xformOp:scale\"]",
                f"            quatd xformOp:orient = ({asset.rotation_quaternion[0]}, {asset.rotation_quaternion[1]}, {asset.rotation_quaternion[2]}, {asset.rotation_quaternion[3]})",
                f"            float3 xformOp:scale = ({asset.scale[0]}, {asset.scale[1]}, {asset.scale[2]})",
                "        }",
            ]
        )
    lines.extend(
        [
            "    }",
            "}",
            "",
        ]
    )
    write_text_atomic(path, "\n".join(lines))


def _write_isaac_lab_package(
    *,
    root: Path,
    scene_id: str,
    usd_scene_path: Path,
    task,
    copied_assets: List[_CopiedAsset],
    robot_type: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    package_name = "scene_task"
    package_root = root / package_name
    package_root.mkdir(parents=True, exist_ok=True)
    object_paths = {
        asset.object_id: f"/World/Scene/{sanitize_filename_component(asset.object_id, fallback='asset')}"
        for asset in copied_assets
    }
    target_path = object_paths.get(task.target_object_id, f"/World/Scene/{task.target_object_id}")
    goal_path = object_paths.get(task.goal_object_id, f"/World/Scene/{task.goal_object_id}")

    write_text_atomic(
        root / "__init__.py",
        (
            "from .scene_task import (\n"
            "    TeleopEnvCfg,\n"
            "    PickPlaceTask,\n"
            "    create_env,\n"
            "    get_reset_events,\n"
            "    get_interval_events,\n"
            ")\n"
            "__all__ = ['TeleopEnvCfg', 'PickPlaceTask', 'create_env', 'get_reset_events', 'get_interval_events']\n"
        ),
    )
    write_text_atomic(
        package_root / "__init__.py",
        (
            "from .env_cfg import TeleopEnvCfg\n"
            "from .blueprint_env import create_env\n"
            "from .task_pick_place import PickPlaceTask\n"
            "from .randomizations import get_reset_events, get_interval_events\n"
            "__all__ = ['TeleopEnvCfg', 'PickPlaceTask', 'create_env', 'get_reset_events', 'get_interval_events']\n"
        ),
    )
    write_text_atomic(
        package_root / "env_cfg.py",
        _render_env_cfg_py(
            scene_id=scene_id,
            usd_scene_path=usd_scene_path,
            target_object_id=task.target_object_id,
            target_object_path=target_path,
            goal_object_id=task.goal_object_id,
            goal_object_path=goal_path,
            robot_type=robot_type,
        ),
    )
    write_text_atomic(
        package_root / "task_pick_place.py",
        _render_task_py(task_id=task.task_id, task_text=task.task_text, target_object_id=task.target_object_id),
    )
    write_text_atomic(
        package_root / "randomizations.py",
        "def get_reset_events():\n    return []\n\n\ndef get_interval_events():\n    return []\n",
    )
    write_text_atomic(
        package_root / "reward_functions.py",
        (
            "def reward_pick_place(*args, **kwargs):\n"
            "    del args, kwargs\n"
            "    return 0.0\n"
        ),
    )
    write_text_atomic(
        package_root / "train_cfg.yaml",
        "seed: 0\nnum_envs: 1\nmax_iterations: 1\n",
    )
    write_text_atomic(
        package_root / "blueprint_env.py",
        _render_blueprint_env_py(
            scene_id=scene_id,
            task_id=task.task_id,
            task_text=task.task_text,
        ),
    )
    write_json(
        _build_blueprint_runtime_contract(
            task_package=package_name,
            scene_id=scene_id,
        ),
        package_root / "blueprint_runtime.json",
    )


def _render_background_usda(
    *,
    center: List[float],
    extents: List[float],
    collision_mode: str,
    source_ply_path: str,
) -> str:
    return "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "Background"',
            '    upAxis = "Z"',
            ")",
            "",
            'def Xform "Background"',
            "{",
            f'    string source_ply = "{source_ply_path}"',
            f'    token collision_mode = "{collision_mode}"',
            '    def Cube "CollisionBounds"',
            "    {",
            f"        double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})",
            f"        float3 xformOp:scale = ({0.5 * extents[0]}, {0.5 * extents[1]}, {0.5 * extents[2]})",
            "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]",
            "        double size = 2",
            "    }",
            "}",
            "",
        ]
    )


def _render_env_cfg_py(
    *,
    scene_id: str,
    usd_scene_path: Path,
    target_object_id: str,
    target_object_path: str,
    goal_object_id: str,
    goal_object_path: str,
    robot_type: str,
) -> str:
    usd_path = json.dumps(str(usd_scene_path.resolve()))
    target_id = json.dumps(target_object_id)
    target_path_json = json.dumps(target_object_path)
    goal_id = json.dumps(goal_object_id)
    goal_path_json = json.dumps(goal_object_path)
    return f'''"""Generated Isaac Lab task package for {scene_id}."""

from __future__ import annotations

import os
import types

SCENE_ID = {json.dumps(scene_id)}
USD_SCENE_PATH = {usd_path}
TARGET_OBJECT_ID = {target_id}
TARGET_OBJECT_PATH = {target_path_json}
GOAL_OBJECT_ID = {goal_id}
GOAL_OBJECT_PATH = {goal_path_json}
ROBOT_TYPE = {json.dumps(robot_type)}

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
except Exception:
    def configclass(cls):
        return cls

    class _Stub:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    sim_utils = types.SimpleNamespace(
        GroundPlaneCfg=lambda *args, **kwargs: None,
        DomeLightCfg=lambda *args, **kwargs: None,
        UsdFileCfg=lambda *args, **kwargs: None,
    )
    AssetBaseCfg = ArticulationCfg = RigidObjectCfg = ManagerBasedEnvCfg = InteractiveSceneCfg = _Stub
    RigidObjectCfg.InitialStateCfg = _Stub
    ArticulationCfg.InitialStateCfg = _Stub


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    scene = AssetBaseCfg(
        prim_path="/World/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_SCENE_PATH, scale=(1.0, 1.0, 1.0)),
    )
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get("BLUEPRINT_SCENE_BUILDER_ROBOT_USD", "robot/franka/franka.usd"),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={{}},
        ),
    )
    target = RigidObjectCfg(
        prim_path=TARGET_OBJECT_PATH,
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    goal = RigidObjectCfg(
        prim_path=GOAL_OBJECT_PATH,
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )


@configclass
class TeleopEnvCfg(ManagerBasedEnvCfg):
    def __init__(self):
        super().__init__()
        self.scene = getattr(self, "scene", SceneCfg())
'''


def _render_task_py(*, task_id: str, task_text: str, target_object_id: str) -> str:
    return f'''"""Generated pick/place task wrapper."""

from __future__ import annotations

TASK_ID = {json.dumps(task_id)}
TASK_TEXT = {json.dumps(task_text)}
TARGET_OBJECT_ID = {json.dumps(target_object_id)}


class PickPlaceTask:
    def __init__(self, env=None, cfg=None):
        self.env = env
        self.cfg = cfg

    def reset(self, env_ids=None):
        del env_ids
        return
'''


def _build_blueprint_runtime_contract(*, task_package: str, scene_id: str) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "runtime_kind": "blueprint_scene_env",
        "scene_id": scene_id,
        "task_package": task_package,
        "env_factory": f"{task_package}.create_env",
        "env_cfg_class": "TeleopEnvCfg",
        "action_dim": 7,
        "camera_keys": ["wrist_rgb", "front_rgb"],
        "state_keys": [
            "policy",
            "joint_positions",
            "joint_velocities",
            "end_effector_pose",
            "gripper_state",
        ],
    }


def _render_blueprint_env_py(*, scene_id: str, task_id: str, task_text: str) -> str:
    return f'''"""Runnable fallback env for generated Blueprint scene packages."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SCENE_ID = {json.dumps(scene_id)}
TASK_ID = {json.dumps(task_id)}
TASK_TEXT = {json.dumps(task_text)}
ACTION_DIM = 7
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 128


class _ActionSpec:
    def __init__(self, action_dim: int):
        self.shape = (int(action_dim),)


class _ActionManager:
    def __init__(self, action_dim: int):
        self.action_spec = _ActionSpec(action_dim)


class BlueprintSceneEnv:
    def __init__(self, *, headless: bool = False):
        self.device = "cpu"
        self.headless = bool(headless)
        self.action_manager = _ActionManager(ACTION_DIM)
        self.max_steps = 48
        self.task_success = np.asarray([False], dtype=bool)
        self._phase = 0.0
        self.reset()

    def reset(self):
        self._step_count = 0
        self._progress = 0.0
        self._pose = np.zeros((ACTION_DIM,), dtype=np.float32)
        self.task_success = np.asarray([False], dtype=bool)
        return self._observation()

    def step(self, action: Any):
        arr = _normalize_action(action)
        self._step_count += 1
        self._pose[:6] = np.clip(self._pose[:6] + arr[:6], -1.0, 1.0)
        self._pose[6] = np.clip(self._pose[6] + 0.1 * np.sign(arr[6]), -1.0, 1.0)
        self._phase += 0.25
        self._progress = min(1.0, max(self._progress, (self._step_count / 24.0) + (self._pose[6] + 1.0) * 0.1))
        success = bool(self._progress >= 0.95)
        done = bool(success or self._step_count >= self.max_steps)
        self.task_success = np.asarray([success], dtype=bool)
        reward = np.asarray([self._progress], dtype=np.float32)
        info = {{
            "rubric": {{
                "success": success,
                "progress": float(self._progress),
            }}
        }}
        return self._observation(), reward, np.asarray([done], dtype=bool), info

    def close(self):
        return None

    def _observation(self):
        ee_pose = np.asarray(
            [
                self._pose[0],
                self._pose[1],
                self._pose[2],
                self._pose[3],
                self._pose[4],
                self._pose[5],
            ],
            dtype=np.float32,
        )
        return {{
            "wrist_rgb": _render_camera(self._pose, camera="wrist", progress=float(self._progress), phase=self._phase),
            "front_rgb": _render_camera(self._pose, camera="front", progress=float(self._progress), phase=self._phase),
            "policy": np.asarray([self._progress, self._step_count / float(self.max_steps), self._pose[6]], dtype=np.float32),
            "joint_positions": self._pose.copy(),
            "joint_velocities": np.full((ACTION_DIM,), 0.01 * (self._step_count + 1), dtype=np.float32),
            "end_effector_pose": ee_pose,
            "gripper_state": np.asarray([self._pose[6]], dtype=np.float32),
        }}


def create_env(*, headless: bool = False):
    return BlueprintSceneEnv(headless=headless)


def _normalize_action(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim >= 2:
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.size < ACTION_DIM:
        padded = np.zeros((ACTION_DIM,), dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded
    return arr[:ACTION_DIM]


def _render_camera(pose: np.ndarray, *, camera: str, progress: float, phase: float) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, IMAGE_HEIGHT, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, IMAGE_WIDTH, dtype=np.float32)[None, :]
    base = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    tint = 35 if camera == "wrist" else 70
    base[..., 0] = np.clip((40 + tint + 120 * xx + 30 * progress), 0, 255).astype(np.uint8)
    base[..., 1] = np.clip((55 + 80 * yy + 25 * math.sin(phase)), 0, 255).astype(np.uint8)
    base[..., 2] = np.clip((80 + 90 * (1.0 - xx) + 20 * math.cos(phase)), 0, 255).astype(np.uint8)

    obj_x = int(np.clip((0.35 + 0.15 * pose[0]) * IMAGE_WIDTH, 10, IMAGE_WIDTH - 10))
    obj_y = int(np.clip((0.55 + 0.12 * pose[1]) * IMAGE_HEIGHT, 10, IMAGE_HEIGHT - 10))
    goal_x = int(np.clip((0.72 - 0.10 * pose[2]) * IMAGE_WIDTH, 10, IMAGE_WIDTH - 10))
    goal_y = int(np.clip((0.30 + 0.08 * pose[5]) * IMAGE_HEIGHT, 10, IMAGE_HEIGHT - 10))

    base[max(0, obj_y - 8): min(IMAGE_HEIGHT, obj_y + 8), max(0, obj_x - 8): min(IMAGE_WIDTH, obj_x + 8), :] = np.asarray([230, 90, 60], dtype=np.uint8)
    base[max(0, goal_y - 9): min(IMAGE_HEIGHT, goal_y + 9), max(0, goal_x - 9): min(IMAGE_WIDTH, goal_x + 9), :] = np.asarray([70, 220, 90], dtype=np.uint8)
    return base
'''
