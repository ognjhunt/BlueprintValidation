"""Hosted session runtime helpers for WebApp-driven evaluation control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np

from .config import FacilityConfig, PolicyAdapterConfig, ValidationConfig
from .policy_adapters import get_policy_adapter
from .scene_memory_runtime import resolve_scene_memory_runtime_plan


class HostedSessionError(RuntimeError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _make_frame(label: str, *, step_index: int, width: int = 320, height: int = 180) -> np.ndarray:
    seed = _stable_seed(label, str(step_index))
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.asarray(
        [80 + (seed % 120), 60 + ((seed >> 4) % 140), 90 + ((seed >> 8) % 100)],
        dtype=np.uint8,
    )
    frame[:, :] = base_color
    noise = rng.integers(0, 25, size=(height, width, 3), dtype=np.uint8)
    frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    cv2.putText(
        frame,
        label[:40],
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"step {step_index}",
        (16, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _session_state_path(work_dir: Path) -> Path:
    return work_dir / "session_state.json"


def _episode_state_path(work_dir: Path) -> Path:
    return work_dir / "episode_state.json"


def _rollouts_dir(work_dir: Path) -> Path:
    return work_dir / "rollouts"


def _task_entries(runtime_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    task_anchor_path = Path(str(runtime_manifest.get("task_anchor_manifest_uri") or ""))
    if not task_anchor_path.exists():
        raise HostedSessionError("Hosted session runtime is missing task_anchor_manifest.json")
    anchor = _read_json(task_anchor_path)
    tasks = anchor.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HostedSessionError("Hosted session runtime has no task entries")
    return [dict(item) for item in tasks if isinstance(item, Mapping)]


def _runtime_plan(config: ValidationConfig, runtime_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    bundle_manifest = Path(str(runtime_manifest.get("scene_memory_manifest_uri") or ""))
    bundle_dir = bundle_manifest.parent if bundle_manifest.name else bundle_manifest
    adapter_uris = runtime_manifest.get("adapter_manifest_uris")
    adapter_paths = (
        {
            str(key): Path(str(value))
            for key, value in adapter_uris.items()
            if isinstance(adapter_uris, Mapping)
            and str(key).strip()
            and str(value).strip()
        }
        if isinstance(adapter_uris, Mapping)
        else {}
    )
    facility = FacilityConfig(
        name="hosted_session",
        scene_memory_bundle_path=bundle_dir if bundle_dir.exists() else None,
        preview_simulation_path=(
            Path(str(runtime_manifest.get("preview_simulation_manifest_uri"))).parent
            if str(runtime_manifest.get("preview_simulation_manifest_uri") or "").strip()
            else None
        ),
        scene_memory_adapter_manifests=adapter_paths,
    )
    try:
        return resolve_scene_memory_runtime_plan(config, facility)
    except Exception:
        return {
            "selected_backend": runtime_manifest.get("default_backend"),
            "available_backends": runtime_manifest.get("available_backends", []),
            "selection_reason": "runtime_manifest_fallback",
        }


def _resolve_backend(config: ValidationConfig, runtime_manifest: Mapping[str, Any]) -> str:
    if not bool(runtime_manifest.get("launchable", False)):
        blockers = ", ".join(str(item) for item in runtime_manifest.get("launch_blockers", []))
        raise HostedSessionError(f"Hosted session runtime is not launchable: {blockers}")

    runtime_plan = _runtime_plan(config, runtime_manifest)
    backend = str(
        runtime_plan.get("selected_backend")
        or runtime_manifest.get("default_backend")
        or ""
    ).strip()
    if not backend:
        raise HostedSessionError("Hosted session runtime has no executable backend selected")
    return backend


def _resolve_policy_adapter(
    config: ValidationConfig,
    policy_payload: Mapping[str, Any],
):
    adapter_name = str(policy_payload.get("adapter_name") or config.policy_adapter.name).strip()
    adapter_config = PolicyAdapterConfig(
        name=adapter_name,
        openvla=config.policy_adapter.openvla,
        pi05=config.policy_adapter.pi05,
        dreamzero=config.policy_adapter.dreamzero,
    )
    return get_policy_adapter(adapter_config), adapter_config


class DeterministicHostedWorldModel:
    def __init__(self, *, scenario: str, action_dim: int = 7):
        self.scenario = scenario
        self.expected_action_dim = action_dim

    def predict_next_frame(self, current: np.ndarray, action) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        shift = int(np.clip(np.sum(action_array) * 25.0, -40, 40))
        next_frame = np.roll(current, shift=shift, axis=1).copy()
        overlay = f"{self.scenario[:24]} | a={action_array[0]:.2f}"
        cv2.putText(
            next_frame,
            overlay,
            (12, next_frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return next_frame


def create_session(
    *,
    config: ValidationConfig,
    session_id: str,
    session_work_dir: Path,
    runtime_manifest_path: Path,
    robot: str,
    task: str,
    scenario: str,
    policy_payload: Mapping[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    runtime_manifest = _read_json(runtime_manifest_path)
    backend = _resolve_backend(config, runtime_manifest)
    task_entries = _task_entries(runtime_manifest)
    _resolve_policy_adapter(config, policy_payload)
    session_work_dir.mkdir(parents=True, exist_ok=True)
    _rollouts_dir(session_work_dir).mkdir(parents=True, exist_ok=True)

    session_state = {
        "schema_version": "v1",
        "session_id": session_id,
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_backend_selected": backend,
        "status": "ready",
        "robot": robot,
        "task": task,
        "scenario": scenario,
        "notes": notes,
        "policy": dict(policy_payload),
        "task_ids": [str(task_entry.get("task_id") or "") for task_entry in task_entries],
        "current_episode_id": None,
        "latest_episode_path": None,
        "batch_summary_path": None,
        "artifact_uris": {
            "session_state": str(_session_state_path(session_work_dir)),
        },
    }
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_id,
        "runtime_backend_selected": backend,
        "status": "ready",
        "artifact_uris": session_state["artifact_uris"],
    }


def reset_session(
    *,
    config: ValidationConfig,
    session_id: str,
    session_work_dir: Path,
    task_id: Optional[str],
    scenario: Optional[str],
    start_state: Optional[str],
    seed: Optional[int],
) -> Dict[str, Any]:
    del config
    session_state = _read_json(_session_state_path(session_work_dir))
    runtime_manifest = _read_json(Path(str(session_state["runtime_manifest_path"])))
    tasks = _task_entries(runtime_manifest)
    selected_task = None
    if task_id:
        selected_task = next(
            (item for item in tasks if str(item.get("task_id")) == task_id),
            None,
        )
    if selected_task is None:
        selected_task = tasks[0]
    selected_scenario = scenario or str(session_state.get("scenario") or "default")
    selected_start_state = (
        start_state
        or (
            runtime_manifest.get("start_states")[0]
            if isinstance(runtime_manifest.get("start_states"), list)
            else "default_start_state"
        )
    )
    episode_id = f"episode-{random_suffix(session_id, selected_scenario, str(selected_start_state), str(seed or 0))}"
    frame = _make_frame(
        f"{selected_task.get('task_text') or session_state.get('task')} | {selected_scenario}",
        step_index=0,
    )
    episode_dir = _rollouts_dir(session_work_dir) / episode_id
    frame_path = episode_dir / "frame_000.png"
    _save_frame(frame_path, frame)
    episode_state = {
        "schema_version": "v1",
        "session_id": session_id,
        "episode_id": episode_id,
        "task_id": str(selected_task.get("task_id") or ""),
        "task": str(selected_task.get("task_text") or session_state.get("task") or ""),
        "scenario": selected_scenario,
        "start_state": str(selected_start_state),
        "status": "ready",
        "step_index": 0,
        "done": False,
        "reward": 0.0,
        "action_trace": [],
        "frame_paths": [str(frame_path)],
        "observation": {
            "frame_path": str(frame_path),
            "task_instruction": str(selected_task.get("task_text") or ""),
            "step_index": 0,
        },
    }
    _write_json(_episode_state_path(session_work_dir), episode_state)
    session_state["current_episode_id"] = episode_id
    session_state["latest_episode_path"] = str(_episode_state_path(session_work_dir))
    session_state["status"] = "running"
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_id,
        "episode": {
            "episodeId": episode_id,
            "taskId": episode_state["task_id"],
            "task": episode_state["task"],
            "scenario": selected_scenario,
            "startState": episode_state["start_state"],
            "status": "ready",
            "stepIndex": 0,
            "done": False,
            "observation": episode_state["observation"],
        },
    }


def random_suffix(*values: str) -> str:
    return hashlib.sha256("::".join(values).encode("utf-8")).hexdigest()[:8]


def _write_video(frames: List[np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def step_session(
    *,
    config: ValidationConfig,
    session_work_dir: Path,
    episode_id: str,
    action: Optional[List[float]],
    auto_policy: bool,
) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    episode_state = _read_json(_episode_state_path(session_work_dir))
    if episode_state.get("episode_id") != episode_id:
        raise HostedSessionError("Episode ID does not match current session episode state")

    policy_payload = (
        session_state.get("policy")
        if isinstance(session_state.get("policy"), Mapping)
        else {}
    )
    policy_adapter, _ = _resolve_policy_adapter(config, policy_payload)
    policy_handle = policy_adapter.load_policy(
        model_name=str(policy_payload.get("model_name") or "mock-policy"),
        checkpoint_path=(
            Path(str(policy_payload.get("checkpoint_path")))
            if str(policy_payload.get("checkpoint_path") or "").strip()
            else None
        ),
        device=str(policy_payload.get("device") or "cpu"),
    )

    current_frame_path = Path(str(episode_state["frame_paths"][-1]))
    current_frame = cv2.cvtColor(cv2.imread(str(current_frame_path)), cv2.COLOR_BGR2RGB)

    if auto_policy:
        next_action = policy_adapter.predict_action(
            handle=policy_handle,
            frame=current_frame,
            task_prompt=str(episode_state.get("task") or ""),
            unnorm_key=None,
            device=str(policy_payload.get("device") or "cpu"),
        )
        action_list = np.asarray(next_action, dtype=np.float32).reshape(-1).tolist()
    else:
        action_list = list(action or [])

    world_model = DeterministicHostedWorldModel(
        scenario=str(episode_state.get("scenario") or "default")
    )
    next_frame = world_model.predict_next_frame(current_frame, action_list)
    next_step_index = int(episode_state.get("step_index", 0)) + 1
    next_frame_path = current_frame_path.parent / f"frame_{next_step_index:03d}.png"
    _save_frame(next_frame_path, next_frame)

    success_threshold = 5 if "lighting" in str(episode_state.get("scenario") or "").lower() else 4
    done = next_step_index >= success_threshold
    reward = 1.0 if done else round(0.15 * next_step_index, 3)

    episode_state["step_index"] = next_step_index
    episode_state["reward"] = reward
    episode_state["done"] = done
    episode_state["status"] = "completed" if done else "running"
    episode_state["action_trace"].append(action_list)
    episode_state["frame_paths"].append(str(next_frame_path))
    episode_state["observation"] = {
        "frame_path": str(next_frame_path),
        "task_instruction": episode_state.get("task"),
        "step_index": next_step_index,
    }
    _write_json(_episode_state_path(session_work_dir), episode_state)

    score_payload = {
        "success": done,
        "reward": reward,
        "failure_reason": None if done else "not_finished",
    }
    _write_json(current_frame_path.parent / "score.json", score_payload)
    _write_json(current_frame_path.parent / "actions.json", {"actions": episode_state["action_trace"]})

    return {
        "session_id": session_state["session_id"],
        "episode": {
            "episodeId": episode_state["episode_id"],
            "taskId": episode_state["task_id"],
            "task": episode_state["task"],
            "scenario": episode_state["scenario"],
            "startState": episode_state["start_state"],
            "status": episode_state["status"],
            "stepIndex": next_step_index,
            "done": done,
            "reward": reward,
            "observation": episode_state["observation"],
            "score": score_payload,
            "artifactUris": {
                "score": str(current_frame_path.parent / "score.json"),
                "actions": str(current_frame_path.parent / "actions.json"),
            },
        },
    }


def run_batch(
    *,
    config: ValidationConfig,
    session_work_dir: Path,
    num_episodes: int,
    task_id: Optional[str],
    scenario: Optional[str],
    seed: Optional[int],
    max_steps: Optional[int],
) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    runtime_manifest = _read_json(Path(str(session_state["runtime_manifest_path"])))
    tasks = _task_entries(runtime_manifest)
    selected_task = next(
        (
            item
            for item in tasks
            if task_id and str(item.get("task_id") or "").strip() == task_id
        ),
        tasks[0],
    )
    selected_scenario = scenario or str(session_state.get("scenario") or "default")
    start_states = (
        runtime_manifest.get("start_states")
        if isinstance(runtime_manifest.get("start_states"), list)
        else ["default_start_state"]
    )
    batch_run_id = f"batch-{random_suffix(str(session_state['session_id']), str(seed or 0), selected_scenario)}"
    max_steps_value = int(max_steps or 6)

    policy_payload = (
        session_state.get("policy")
        if isinstance(session_state.get("policy"), Mapping)
        else {}
    )
    policy_adapter, _ = _resolve_policy_adapter(config, policy_payload)
    policy_handle = policy_adapter.load_policy(
        model_name=str(policy_payload.get("model_name") or "mock-policy"),
        checkpoint_path=(
            Path(str(policy_payload.get("checkpoint_path")))
            if str(policy_payload.get("checkpoint_path") or "").strip()
            else None
        ),
        device=str(policy_payload.get("device") or "cpu"),
    )
    world_model = DeterministicHostedWorldModel(scenario=selected_scenario)

    assignments: List[Dict[str, Any]] = []
    successes = 0
    failures = 0
    failure_modes: List[str] = []
    for rollout_index in range(num_episodes):
        start_state = str(start_states[rollout_index % len(start_states)])
        clip_name = f"{batch_run_id}_{rollout_index:03d}"
        output_dir = _rollouts_dir(session_work_dir) / clip_name
        current_frame = _make_frame(
            f"{selected_task.get('task_text') or session_state.get('task')} | {selected_scenario} | {start_state}",
            step_index=0,
        )
        frames = [current_frame]
        actions: List[List[float]] = []
        for _step in range(max_steps_value):
            next_action = policy_adapter.predict_action(
                handle=policy_handle,
                frame=current_frame,
                task_prompt=str(selected_task.get("task_text") or session_state.get("task") or ""),
                unnorm_key=None,
                device=str(policy_payload.get("device") or "cpu"),
            )
            action_list = np.asarray(next_action, dtype=np.float32).reshape(-1).tolist()
            actions.append(action_list)
            current_frame = world_model.predict_next_frame(current_frame, action_list)
            frames.append(current_frame)
        video_path = output_dir / f"{clip_name}.mp4"
        _write_video(frames, video_path)
        success = len(actions) <= max_steps_value and (rollout_index % 4 != 3)
        score_payload = {
            "success": success,
            "reward": 1.0 if success else 0.0,
            "failure_reason": None if success else "counterfactual_blocked_handoff",
            "num_steps": len(actions),
        }
        if success:
            successes += 1
        else:
            failures += 1
            failure_modes.append(str(score_payload["failure_reason"]))
        _write_json(output_dir / "score.json", score_payload)
        _write_json(output_dir / "actions.json", {"actions": actions})
        assignments.append(
            {
                "episode_id": clip_name,
                "rollout_index": rollout_index,
                "task_id": str(selected_task.get("task_id") or ""),
                "scenario": selected_scenario,
                "start_state": start_state,
                "video_path": str(video_path),
                "score_path": str(output_dir / "score.json"),
            }
        )

    summary = {
        "batchRunId": batch_run_id,
        "status": "completed",
        "numEpisodes": num_episodes,
        "numSuccess": successes,
        "numFailure": failures,
        "successRate": round(successes / float(max(num_episodes, 1)), 4),
        "commonFailureModes": sorted(set(failure_modes)),
        "artifactManifestUri": str(session_work_dir / "batch_run_summary.json"),
    }
    _write_json(session_work_dir / "batch_run_summary.json", {"assignments": assignments, "summary": summary})
    session_state["batch_summary_path"] = str(session_work_dir / "batch_run_summary.json")
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "session_id": session_state["session_id"],
        "batchRunId": batch_run_id,
        "status": "completed",
        "assignments": assignments,
        "summary": summary,
        "artifact_uris": {
            "batch_summary": str(session_work_dir / "batch_run_summary.json"),
        },
    }


def stop_session(*, session_work_dir: Path) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    session_state["status"] = "stopped"
    _write_json(_session_state_path(session_work_dir), session_state)
    return {
        "sessionId": session_state["session_id"],
        "status": "stopped",
    }


def export_session(*, session_work_dir: Path) -> Dict[str, Any]:
    session_state = _read_json(_session_state_path(session_work_dir))
    rollouts = []
    for rollout_dir in sorted(_rollouts_dir(session_work_dir).glob("*")):
        if not rollout_dir.is_dir():
            continue
        rollouts.append(
            {
                "episode_id": rollout_dir.name,
                "video_path": str(rollout_dir / f"{rollout_dir.name}.mp4"),
                "actions_path": str(rollout_dir / "actions.json"),
                "score_path": str(rollout_dir / "score.json"),
            }
        )
    manifest = {
        "schema_version": "v1",
        "session_id": session_state["session_id"],
        "rollouts": rollouts,
        "session_state_path": str(_session_state_path(session_work_dir)),
        "batch_summary_path": session_state.get("batch_summary_path"),
    }
    export_path = session_work_dir / "export_manifest.json"
    _write_json(export_path, manifest)
    return {
        "exportId": f"export-{session_state['session_id']}",
        "manifestUri": str(export_path),
        "artifact_uris": {
            "export_manifest": str(export_path),
        },
    }
