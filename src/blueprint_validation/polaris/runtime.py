"""Runtime discovery and scene resolution helpers for PolaRiS integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import read_json
from ..config import FacilityConfig, ValidationConfig
from ..teleop.contracts import TeleopManifestError, load_and_validate_scene_package


@dataclass(frozen=True)
class PolarisRuntimeInfo:
    repo_path: Path
    hub_path: Path
    issues: List[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class PolarisSceneSpec:
    mode: str
    primary_eligible: bool
    environment_name: Optional[str]
    scene_root: Optional[Path]
    usd_path: Optional[Path]
    initial_conditions_path: Optional[Path]
    task_metadata_path: Optional[Path]
    instruction: str
    detail: str = ""


def polaris_primary_gate_enabled(config: ValidationConfig) -> bool:
    return bool(config.eval_polaris.enabled) and bool(config.eval_polaris.default_as_primary_gate)


def resolve_polaris_runtime(config: ValidationConfig) -> PolarisRuntimeInfo:
    repo_path = config.eval_polaris.repo_path.resolve()
    hub_path = config.eval_polaris.hub_path.resolve()
    issues: List[str] = []
    if not repo_path.exists():
        issues.append(f"PolaRiS repo not found: {repo_path}")
    else:
        expected = [
            repo_path / "scripts" / "eval.py",
            repo_path / "src" / "polaris" / "config.py",
            repo_path / "src" / "polaris" / "policy" / "abstract_client.py",
        ]
        for path in expected:
            if not path.exists():
                issues.append(f"Missing PolaRiS runtime file: {path}")
    if not hub_path.exists():
        issues.append(f"PolaRiS hub path not found: {hub_path}")
    return PolarisRuntimeInfo(repo_path=repo_path, hub_path=hub_path, issues=issues)


def resolve_scene_package_path(facility: FacilityConfig) -> Optional[Path]:
    if facility.scene_package_path is None:
        return None
    return facility.scene_package_path.resolve()


def resolve_polaris_scene_spec(
    config: ValidationConfig,
    facility: FacilityConfig,
) -> PolarisSceneSpec:
    mode = str(config.eval_polaris.environment_mode or "scene_package_bridge").strip().lower()
    if mode == "auto":
        if facility.scene_package_path is not None:
            mode = "scene_package_bridge"
        elif config.eval_polaris.environment_name:
            mode = "native_bundle"
        else:
            mode = "scan_only_bridge"

    if mode == "native_bundle":
        env_name = str(config.eval_polaris.environment_name or "").strip() or None
        env_root = config.eval_polaris.hub_path / str(env_name or "")
        usd_path = env_root / "scene.usda"
        ic_path = env_root / "initial_conditions.json"
        primary_eligible = usd_path.exists() and ic_path.exists()
        return PolarisSceneSpec(
            mode=mode,
            primary_eligible=primary_eligible,
            environment_name=env_name,
            scene_root=env_root if env_root.exists() else None,
            usd_path=usd_path if usd_path.exists() else None,
            initial_conditions_path=ic_path if ic_path.exists() else None,
            task_metadata_path=ic_path if ic_path.exists() else None,
            instruction="Evaluate the policy in the PolaRiS-native deployment scene.",
            detail="native_bundle",
        )

    if mode == "scan_only_bridge":
        return PolarisSceneSpec(
            mode=mode,
            primary_eligible=False,
            environment_name=None,
            scene_root=None,
            usd_path=None,
            initial_conditions_path=None,
            task_metadata_path=None,
            instruction="Research-only PolaRiS scan bridge evaluation.",
            detail="scan_only_bridge is never eligible for the primary gate",
        )

    scene_root = resolve_scene_package_path(facility)
    if scene_root is None:
        return PolarisSceneSpec(
            mode="scene_package_bridge",
            primary_eligible=False,
            environment_name=None,
            scene_root=None,
            usd_path=None,
            initial_conditions_path=None,
            task_metadata_path=None,
            instruction="Evaluate the policy in the deployment scene.",
            detail="scene_package_path is not configured",
        )

    detail = "scene_package_bridge"
    try:
        payload = load_and_validate_scene_package(scene_root)
    except TeleopManifestError as exc:
        return PolarisSceneSpec(
            mode="scene_package_bridge",
            primary_eligible=False,
            environment_name=None,
            scene_root=scene_root,
            usd_path=None,
            initial_conditions_path=None,
            task_metadata_path=None,
            instruction="Evaluate the policy in the deployment scene.",
            detail=str(exc),
        )

    task_metadata_path = _resolve_task_metadata_path(scene_root)
    instruction = _resolve_instruction_from_scene_package(task_metadata_path)
    primary_eligible = True
    if bool(config.eval_polaris.require_scene_package) and scene_root is None:
        primary_eligible = False
    if bool(config.eval_polaris.require_success_correlation_metadata) and task_metadata_path is None:
        primary_eligible = False
        detail = "Missing task metadata for strict scene_package_bridge"
    return PolarisSceneSpec(
        mode="scene_package_bridge",
        primary_eligible=primary_eligible,
        environment_name=payload.get("scene_manifest", {}).get("scene_id"),
        scene_root=scene_root,
        usd_path=Path(payload["usd_scene_path"]),
        initial_conditions_path=None,
        task_metadata_path=task_metadata_path,
        instruction=instruction,
        detail=detail,
    )


def _resolve_task_metadata_path(scene_root: Path) -> Optional[Path]:
    candidates = [
        scene_root / "geniesim" / "task_config.json",
        scene_root / "assets" / "scene_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_instruction_from_scene_package(task_metadata_path: Optional[Path]) -> str:
    if task_metadata_path is None or not task_metadata_path.exists():
        return "Evaluate the deployment policy in the packaged scene."
    try:
        payload = read_json(task_metadata_path)
    except Exception:
        return "Evaluate the deployment policy in the packaged scene."
    if isinstance(payload, dict):
        tasks = payload.get("suggested_tasks")
        if isinstance(tasks, list) and tasks:
            first = tasks[0]
            if isinstance(first, dict):
                hint = str(first.get("description_hint", "") or "").strip()
                if hint:
                    return hint
        scene_id = str(payload.get("scene_id", "") or "").strip()
        if scene_id:
            return f"Evaluate the policy in scene '{scene_id}'."
    return "Evaluate the deployment policy in the packaged scene."


def task_metadata_payload(task_metadata_path: Optional[Path]) -> Dict[str, Any]:
    if task_metadata_path is None or not task_metadata_path.exists():
        return {}
    payload = read_json(task_metadata_path)
    return payload if isinstance(payload, dict) else {}
