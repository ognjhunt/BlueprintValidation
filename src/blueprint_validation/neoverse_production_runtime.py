"""Production NeoVerse runtime backend with authoritative session state."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence

import numpy as np

from blueprint_contracts.site_world_contract import (
    SiteWorldIntakeError,
    load_site_world_bundle,
    normalize_trajectory_payload,
)

from .neoverse_runtime_core import (
    _blocked_status,
    _coerce_camera_frame,
    _dedupe_strings,
    _env_truthy,
    _extract_first_frame,
    _load_frame,
    _preferred_local_path,
    _read_json,
    _save_frame,
    _stable_id,
    _utc_now_iso,
    _write_json,
)
from .presentation_bundle_renderer import PresentationBundleRenderer
from .pose_driven_preview import PoseDrivenPreviewRenderer
from .runtime_backend import RuntimeMetadata
from .runtime_layer_grounding import (
    composite_masked_refinement,
    composite_runtime_layer,
    load_runtime_layer_bundle,
    snapshot_runtime_layer_bundle,
    update_presentation_session_manifest,
    validate_runtime_layer_spec,
    verify_canonical_package_version,
)


_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _presentation_map(spec: Mapping[str, Any]) -> dict[str, Any]:
    return dict(spec.get("presentation") or {}) if isinstance(spec.get("presentation"), Mapping) else {}


def _canonical_world_model_map(spec: Mapping[str, Any]) -> dict[str, Any]:
    return dict(spec.get("canonical_world_model") or {}) if isinstance(spec.get("canonical_world_model"), Mapping) else {}


def _presentation_ref(presentation: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(presentation.get(key) or "").strip()
        if value:
            return value
    return ""


def _has_dedicated_presentation_bundle(spec: Mapping[str, Any]) -> bool:
    presentation = _presentation_map(spec)
    manifest_ref = _presentation_ref(
        presentation,
        "presentation_world_manifest_path",
        "presentation_world_manifest_uri",
    )
    asset_ref = _presentation_ref(
        presentation,
        "primary_asset_path",
        "primary_asset_uri",
    )
    return bool(manifest_ref and asset_ref)


def _has_canonical_world_model_bundle(spec: Mapping[str, Any]) -> bool:
    canonical_world_model = _canonical_world_model_map(spec)
    asset_ref = _presentation_ref(
        canonical_world_model,
        "primary_asset_path",
        "primary_asset_uri",
    )
    return bool(asset_ref)


def _spec_with_canonical_world_presentation(spec: Mapping[str, Any]) -> dict[str, Any]:
    canonical_world_model = _canonical_world_model_map(spec)
    remapped = dict(spec)
    if canonical_world_model:
        remapped["presentation"] = {
            **dict(canonical_world_model),
            "presentation_world_manifest_path": str(canonical_world_model.get("manifest_path") or ""),
            "presentation_world_manifest_uri": str(canonical_world_model.get("manifest_uri") or ""),
            "bundle_status": str(canonical_world_model.get("status") or canonical_world_model.get("bundle_status") or "missing"),
        }
    return remapped


def _quality_flag_baseline(spec: Mapping[str, Any]) -> dict[str, Any]:
    canonical_world_model = _canonical_world_model_map(spec)
    runtime_layer_policy = dict(spec.get("runtime_layer_policy") or {}) if isinstance(spec.get("runtime_layer_policy"), Mapping) else {}
    return {
        "primary_runtime_backend": str(spec.get("primary_runtime_backend") or "neoverse"),
        "world_model_backend": str(spec.get("world_model_backend") or canonical_world_model.get("world_model_backend") or "neoverse"),
        "scene_representation": str(spec.get("scene_representation") or canonical_world_model.get("scene_representation") or "gsplat_scene_v1"),
        "render_source": str(
            spec.get("runtime_render_source")
            or canonical_world_model.get("render_source")
            or ("canonical_world_model" if _has_canonical_world_model_bundle(spec) else "arkit_rgbd_last_resort")
        ),
        "fallback_mode": str(spec.get("fallback_mode") or canonical_world_model.get("fallback_mode") or "arkit_rgbd_last_resort"),
        "grounding_status": str(
            spec.get("grounding_status")
            or runtime_layer_policy.get("grounding_status")
            or "grounded"
        ),
        "canonical_package_version": str(spec.get("canonical_package_version") or ""),
    }


def _normalized_neoverse_thread_limit(raw_value: str | None) -> str:
    try:
        limit = int(str(raw_value or "").strip() or "1")
    except ValueError:
        limit = 1
    return str(max(1, limit))


def _thread_limited_runner_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    thread_limit = _normalized_neoverse_thread_limit(env.get("NEOVERSE_BLAS_NUM_THREADS"))
    for key in _THREAD_LIMIT_ENV_VARS:
        env.setdefault(key, thread_limit)
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    return env


@dataclass(frozen=True)
class NeoVerseRunnerConfig:
    model_root: str
    checkpoint_path: str
    cache_root: str
    runner_command: str
    device: str
    gpu_enabled: bool
    render_timeout_seconds: float


class NeoVerseRunnerTimeoutError(RuntimeError):
    """Raised when the production render subprocess exceeds its timeout."""


class NeoVerseRunner(Protocol):
    def readiness(self) -> Dict[str, Any]:
        ...

    def model_identity(self) -> Dict[str, Any]:
        ...

    def checkpoint_identity(self) -> Dict[str, Any]:
        ...

    def prepare_site_world(
        self,
        *,
        site_world_id: str,
        workspace_dir: Path,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ...

    def render_snapshot(
        self,
        *,
        site_world_id: str,
        session_id: str,
        workspace_dir: Path,
        snapshot_path: Path,
        output_dir: Path,
        cameras: Sequence[Mapping[str, Any]],
        base_frame_path: Path,
    ) -> Dict[str, Any]:
        ...


class LocalNeoVerseRunnerAdapter:
    """Adapter for a local NeoVerse model runner process."""

    def __init__(self, config: NeoVerseRunnerConfig) -> None:
        self.config = config

    def _checkpoint_path(self) -> Optional[Path]:
        explicit = Path(self.config.checkpoint_path).expanduser() if self.config.checkpoint_path else None
        if explicit and explicit.is_file():
            return explicit
        model_root = Path(self.config.model_root).expanduser() if self.config.model_root else None
        if model_root and model_root.exists():
            candidates = [
                item
                for item in model_root.rglob("*")
                if item.is_file() and item.suffix in {".ckpt", ".pt", ".pth", ".safetensors"}
            ]
            if candidates:
                candidates.sort(key=lambda item: ("reconstructor" not in item.name.lower(), len(str(item))))
                return candidates[0]
        return None

    def readiness(self) -> Dict[str, Any]:
        model_root = Path(self.config.model_root).expanduser() if self.config.model_root else None
        checkpoint_path = self._checkpoint_path()
        runner_ready = bool(self.config.runner_command.strip())
        return {
            "ready": bool(model_root and model_root.exists() and checkpoint_path and checkpoint_path.exists() and runner_ready),
            "model_ready": bool(model_root and model_root.exists()),
            "checkpoint_ready": bool(checkpoint_path and checkpoint_path.exists()),
            "runner_command_ready": runner_ready,
            "device": self.config.device,
            "gpu_enabled": self.config.gpu_enabled,
            "notes": [] if runner_ready else ["runner_command is not configured"],
        }

    def model_identity(self) -> Dict[str, Any]:
        return {
            "model_family": "neoverse",
            "model_id": Path(self.config.model_root).name if self.config.model_root else "unconfigured",
            "model_root": self.config.model_root,
            "device": self.config.device,
            "gpu_enabled": self.config.gpu_enabled,
            "model_ready": bool(self.readiness().get("model_ready")),
        }

    def checkpoint_identity(self) -> Dict[str, Any]:
        checkpoint_path = self._checkpoint_path()
        checkpoint_name = checkpoint_path.name if checkpoint_path is not None else "unconfigured"
        return {
            "checkpoint_id": checkpoint_name,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else self.config.checkpoint_path,
            "checkpoint_ready": bool(self.readiness().get("checkpoint_ready")),
        }

    def _require_ready(self) -> None:
        readiness = self.readiness()
        if not bool(readiness.get("ready")):
            raise RuntimeError(
                "NeoVerse production runner is not ready. "
                f"model_ready={readiness.get('model_ready')} "
                f"checkpoint_ready={readiness.get('checkpoint_ready')} "
                f"runner_command_ready={readiness.get('runner_command_ready')}"
            )

    def prepare_site_world(
        self,
        *,
        site_world_id: str,
        workspace_dir: Path,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self._require_ready()
        manifest_path = workspace_dir / "neoverse_workspace_manifest.json"
        payload = {
            "site_world_id": site_world_id,
            "prepared_at": _utc_now_iso(),
            "model_identity": self.model_identity(),
            "checkpoint_identity": self.checkpoint_identity(),
            "spec_path_hint": str(registration.get("spec_path") or ""),
            "health": dict(health),
            "registration": dict(registration),
            "spec": dict(spec),
        }
        _write_json(manifest_path, payload)
        return {
            "workspace_manifest_path": str(manifest_path),
        }

    def render_snapshot(
        self,
        *,
        site_world_id: str,
        session_id: str,
        workspace_dir: Path,
        snapshot_path: Path,
        output_dir: Path,
        cameras: Sequence[Mapping[str, Any]],
        base_frame_path: Path,
    ) -> Dict[str, Any]:
        self._require_ready()
        request_path = output_dir / "neoverse_render_request.json"
        response_path = output_dir / "neoverse_render_response.json"
        request_payload = {
            "site_world_id": site_world_id,
            "session_id": session_id,
            "workspace_manifest_path": str(workspace_dir / "neoverse_workspace_manifest.json"),
            "snapshot_path": str(snapshot_path),
            "base_frame_path": str(base_frame_path),
            "output_dir": str(output_dir),
            "cameras": [dict(camera) for camera in cameras],
            "model_identity": self.model_identity(),
            "checkpoint_identity": self.checkpoint_identity(),
        }
        _write_json(request_path, request_payload)
        command = shlex.split(self.config.runner_command.strip()) + [str(request_path), str(response_path)]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=_thread_limited_runner_env(),
        )
        try:
            stdout, stderr = process.communicate(timeout=max(1.0, float(self.config.render_timeout_seconds)))
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            details = (stderr or stdout or "").strip()
            raise NeoVerseRunnerTimeoutError(
                f"NeoVerse runner timed out after {self.config.render_timeout_seconds:.1f}s"
                + (f": {details[:300]}" if details else "")
            ) from exc
        if process.returncode != 0:
            stderr = (stderr or stdout or "").strip()
            raise RuntimeError(f"NeoVerse runner failed with exit code {process.returncode}: {stderr[:500]}")
        if not response_path.exists():
            raise RuntimeError("NeoVerse runner did not produce a render response manifest.")
        response = _read_json(response_path)
        if not isinstance(response.get("camera_frames"), list):
            raise RuntimeError("NeoVerse runner response is missing camera_frames.")
        return response


class NeoVerseProductionRuntimeStore:
    """Production-facing runtime store with durable session state and runner integration."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        base_url: str,
        ws_base_url: Optional[str] = None,
        runner: Optional[NeoVerseRunner] = None,
        enable_immediate_preview: Optional[bool] = None,
        enable_async_render_refinement: Optional[bool] = None,
        render_refinement_workers: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = (ws_base_url or self.base_url.replace("http://", "ws://").replace("https://", "wss://")).rstrip("/")
        self.site_worlds_dir = self.root_dir / "site_worlds"
        self.sessions_dir = self.root_dir / "sessions"
        self.site_worlds_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner or LocalNeoVerseRunnerAdapter(
            NeoVerseRunnerConfig(
                model_root=str(os.getenv("NEOVERSE_MODEL_ROOT", "")).strip(),
                checkpoint_path=str(os.getenv("NEOVERSE_CHECKPOINT_PATH", "")).strip(),
                cache_root=str(os.getenv("NEOVERSE_CACHE_ROOT", "")).strip(),
                runner_command=str(os.getenv("NEOVERSE_RUNNER_COMMAND", "")).strip(),
                device=str(os.getenv("NEOVERSE_DEVICE", "cuda:0")).strip() or "cuda:0",
                gpu_enabled=_env_truthy("NEOVERSE_GPU_ENABLED") if os.getenv("NEOVERSE_GPU_ENABLED") else True,
                render_timeout_seconds=float(str(os.getenv("NEOVERSE_RENDER_TIMEOUT_SECONDS", "300")).strip() or "300"),
            )
        )
        immediate_preview_from_env = _env_truthy("NEOVERSE_IMMEDIATE_PREVIEW") if os.getenv("NEOVERSE_IMMEDIATE_PREVIEW") else None
        low_vram_preview_default = _env_truthy("NEOVERSE_LOW_VRAM") if os.getenv("NEOVERSE_LOW_VRAM") else False
        self.enable_immediate_preview = (
            enable_immediate_preview
            if enable_immediate_preview is not None
            else immediate_preview_from_env
            if immediate_preview_from_env is not None
            else low_vram_preview_default
        )
        async_refinement_from_env = (
            _env_truthy("NEOVERSE_ASYNC_RENDER_REFINEMENT")
            if os.getenv("NEOVERSE_ASYNC_RENDER_REFINEMENT")
            else None
        )
        self.enable_async_render_refinement = (
            enable_async_render_refinement
            if enable_async_render_refinement is not None
            else async_refinement_from_env
            if async_refinement_from_env is not None
            else self.enable_immediate_preview
        )
        self.render_refinement_workers = max(
            1,
            int(
                render_refinement_workers
                or str(os.getenv("NEOVERSE_RENDER_REFINEMENT_WORKERS", "1")).strip()
                or "1"
            ),
        )
        self._render_refinement_lock = threading.Lock()
        self._render_refinement_targets: Dict[str, str] = {}
        self._render_refinement_futures: Dict[str, Future[None]] = {}
        self._snapshot_render_lock = threading.Lock()
        self._snapshot_render_futures: Dict[str, Future[Dict[str, Any]]] = {}
        self._render_refinement_executor = (
            ThreadPoolExecutor(
                max_workers=self.render_refinement_workers,
                thread_name_prefix="neoverse-render-refine",
            )
            if self.enable_async_render_refinement
            else None
        )
        self._presentation_bundle_renderer = PresentationBundleRenderer()
        self._pose_preview_renderer = PoseDrivenPreviewRenderer()

    def _log_runtime_stage(self, event: str, **fields: Any) -> None:
        payload = " ".join(
            f"{key}={json.dumps(value, sort_keys=True)}"
            for key, value in fields.items()
            if value is not None
        )
        message = f"[neoverse_runtime] {event}"
        if payload:
            message = f"{message} {payload}"
        print(message, flush=True)

    def _site_world_dir(self, site_world_id: str) -> Path:
        return self.site_worlds_dir / site_world_id

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _site_world_state_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_registration.json"

    def _site_world_health_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_health.json"

    def _site_world_spec_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_spec.json"

    def _session_state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session_state.json"

    def _snapshot_render_key(self, session_id: str, snapshot_id: str) -> str:
        return f"{session_id}:{snapshot_id}"

    def _get_inflight_snapshot_render(self, session_id: str, snapshot_id: str) -> Future[Dict[str, Any]] | None:
        key = self._snapshot_render_key(session_id, snapshot_id)
        with self._snapshot_render_lock:
            future = self._snapshot_render_futures.get(key)
            if future is None or future.done():
                return None
            return future

    def _run_snapshot_render_once(
        self,
        *,
        session_id: str,
        snapshot_id: str,
        render_fn: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        key = self._snapshot_render_key(session_id, snapshot_id)
        created = False
        with self._snapshot_render_lock:
            future = self._snapshot_render_futures.get(key)
            if future is None:
                future = Future()
                self._snapshot_render_futures[key] = future
                created = True
        if not created:
            self._log_runtime_stage(
                "render_snapshot.join_inflight",
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            return future.result()
        try:
            result = render_fn()
        except Exception as exc:
            future.set_exception(exc)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            with self._snapshot_render_lock:
                current = self._snapshot_render_futures.get(key)
                if current is future:
                    self._snapshot_render_futures.pop(key, None)

    def _workspace_dir(self, site_world_id: str, build_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "workspace" / build_id

    def _snapshot_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "world_snapshots"

    def _presentation_world_manifest_path(self, site_world_id: str, build_id: str) -> Path:
        return self._workspace_dir(site_world_id, build_id) / "presentation_world_manifest.json"

    def _runtime_demo_manifest_path(self, site_world_id: str, build_id: str) -> Path:
        return self._workspace_dir(site_world_id, build_id) / "runtime_demo_manifest.json"

    def _site_world_endpoint_url(self, site_world_id: str, suffix: str) -> str:
        quoted_id = urllib.parse.quote(site_world_id, safe="")
        return f"{self.base_url}/v1/site-worlds/{quoted_id}/{suffix}"

    def _build_synthetic_presentation_manifests(
        self,
        *,
        site_world_id: str,
        build_id: str,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        presentation = _presentation_map(spec)
        presentation_manifest_path = self._presentation_world_manifest_path(site_world_id, build_id)
        runtime_demo_manifest_path = self._runtime_demo_manifest_path(site_world_id, build_id)
        presentation_manifest_path.parent.mkdir(parents=True, exist_ok=True)

        presentation_manifest = {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "scene_id": registration.get("scene_id") or spec.get("scene_id"),
            "capture_id": registration.get("capture_id") or spec.get("capture_id"),
            "build_id": build_id,
            "status": "ready",
            "bundle_status": "ready",
            "derivation_mode": "canonical_pretty",
            "scene_kind": "canonical_object_geometry",
            "fallback_policy": "canonical_only",
            "canonical_package_uri": spec.get("canonical_package_uri"),
            "canonical_package_version": spec.get("canonical_package_version"),
            "ui_optional": True,
            "ui_base_url": None,
            "readiness": {
                "bundle_status": "ready",
                "launchable": bool(health.get("launchable", True)),
            },
            "orientation": (
                dict(presentation.get("orientation") or {})
                if isinstance(presentation.get("orientation"), Mapping)
                else {}
            ),
        }
        runtime_demo_manifest = {
            "schema_version": "v1",
            "site_world_id": site_world_id,
            "scene_id": registration.get("scene_id") or spec.get("scene_id"),
            "capture_id": registration.get("capture_id") or spec.get("capture_id"),
            "build_id": build_id,
            "status": "ready",
            "bundle_status": "ready",
            "derivation_mode": "canonical_pretty",
            "scene_kind": "canonical_object_geometry",
            "fallback_policy": "canonical_only",
            "canonical_package_uri": spec.get("canonical_package_uri"),
            "canonical_package_version": spec.get("canonical_package_version"),
            "ui_base_url": None,
            "ui_optional": True,
            "presentation_world_manifest_url": self._site_world_endpoint_url(site_world_id, "presentation-world-manifest"),
        }
        _write_json(presentation_manifest_path, presentation_manifest)
        _write_json(runtime_demo_manifest_path, runtime_demo_manifest)

        presentation_update = {
            **presentation,
            "presentation_world_manifest_path": str(presentation_manifest_path),
            "runtime_demo_manifest_path": str(runtime_demo_manifest_path),
            "presentation_world_manifest_uri": "",
            "runtime_demo_manifest_uri": "",
            "derivation_mode": "canonical_pretty",
            "scene_kind": "canonical_object_geometry",
            "fallback_policy": "canonical_only",
            "bundle_status": "ready",
            "status": "ready",
            "ui_optional": True,
            "ui_base_url": None,
        }
        for key in ("primary_asset_path", "primary_asset_uri", "bundle_type", "renderer_backend"):
            presentation_update.pop(key, None)
        return presentation_manifest, runtime_demo_manifest, presentation_update

    def validate_spec(
        self,
        spec: Mapping[str, Any],
        *,
        protected_regions_manifest: Mapping[str, Any] | None = None,
    ) -> tuple[bool, list[str], list[str]]:
        blockers: list[str] = []
        warnings: list[str] = []
        blockers.extend(validate_runtime_layer_spec(spec))
        runtime_layer_policy = (
            dict(spec.get("runtime_layer_policy") or {})
            if isinstance(spec.get("runtime_layer_policy"), Mapping)
            else {}
        )

        qualification_state = str(spec.get("qualification_state") or "").strip().lower()
        if qualification_state != "ready":
            warnings.append(f"qualification_state:{qualification_state or 'missing'}")
        if not bool(spec.get("downstream_evaluation_eligibility")):
            warnings.append("downstream_evaluation_eligibility:false")
        grounding_status = str(
            (protected_regions_manifest or {}).get("grounding_status")
            or runtime_layer_policy.get("grounding_status")
            or spec.get("grounding_status")
            or ""
        ).strip().lower()
        if grounding_status == "ungrounded":
            reason = str(
                (protected_regions_manifest or {}).get("ungrounded_reason")
                or runtime_layer_policy.get("ungrounded_reason")
                or spec.get("ungrounded_reason")
                or "ungrounded"
            ).strip()
            blockers.append(f"runtime_grounding:{reason or 'ungrounded'}")

        conditioning_map = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
        local_map = dict(conditioning_map.get("local_paths") or {}) if isinstance(conditioning_map.get("local_paths"), Mapping) else {}
        poses_path = _preferred_local_path(local_map.get("arkit_poses_path"), conditioning_map.get("arkit_poses_uri"))
        intrinsics_path = _preferred_local_path(local_map.get("arkit_intrinsics_path"), conditioning_map.get("arkit_intrinsics_uri"))
        if poses_path is None:
            blockers.append("missing_local_conditioning:arkit_poses")
        if intrinsics_path is None:
            blockers.append("missing_local_conditioning:arkit_intrinsics")

        conditioning_source = _preferred_local_path(
            local_map.get("keyframe_path"),
            local_map.get("raw_video_path"),
            conditioning_map.get("keyframe_uri"),
            conditioning_map.get("raw_video_uri"),
        )
        if conditioning_source is None:
            blockers.append("missing_local_conditioning:visual_source")

        geometry_map = dict(spec.get("geometry") or {}) if isinstance(spec.get("geometry"), Mapping) else {}
        if _preferred_local_path(local_map.get("occupancy_path"), geometry_map.get("occupancy_path")) is None:
            warnings.append("occupancy_path_missing")
        if _preferred_local_path(local_map.get("object_index_path"), geometry_map.get("object_index_path")) is None:
            warnings.append("object_index_path_missing")

        presentation = _presentation_map(spec)
        if presentation:
            presentation_manifest_path = _preferred_local_path(
                presentation.get("presentation_world_manifest_path"),
                presentation.get("presentation_world_manifest_uri"),
            )
            if presentation_manifest_path is None:
                warnings.append("presentation_manifest_missing")
            primary_asset_path = _preferred_local_path(
                presentation.get("primary_asset_path"),
                presentation.get("primary_asset_uri"),
            )
            bundle_status = str(presentation.get("bundle_status") or "").strip().lower()
            derivation_mode = str(presentation.get("derivation_mode") or "").strip().lower()
            requires_dedicated_asset = derivation_mode != "canonical_pretty"
            if primary_asset_path is None and requires_dedicated_asset:
                warnings.append("presentation_primary_asset_missing")
            elif bundle_status not in {"", "ready", "demo_ready"}:
                warnings.append(f"presentation_bundle_status:{bundle_status}")
            orientation = (
                dict(presentation.get("orientation") or {})
                if isinstance(presentation.get("orientation"), Mapping)
                else {}
            )
            try:
                encoded_width = int(orientation.get("encoded_width") or 0)
                encoded_height = int(orientation.get("encoded_height") or 0)
            except (TypeError, ValueError):
                encoded_width = 0
                encoded_height = 0
            if (encoded_width < 0) or (encoded_height < 0):
                warnings.append("presentation_orientation_invalid")

        return len(blockers) == 0, blockers, warnings

    def runtime_info(self, *, service_version: str) -> Dict[str, Any]:
        readiness = self.runner.readiness()
        return RuntimeMetadata(
            runtime_kind="neoverse_production",
            production_grade=True,
            service="neoverse-production-runtime",
            version=service_version,
            runtime_base_url=self.base_url,
            websocket_base_url=self.ws_base_url,
            engine_identity={
                "engine": "neoverse",
                "mode": "authoritative_session_runtime",
                "runner": self.runner.__class__.__name__,
            },
            model_identity=self.runner.model_identity(),
            checkpoint_identity=self.runner.checkpoint_identity(),
            state_guarantees={
                "authoritative_state": True,
                "restart_safe": True,
                "deterministic_replay": True,
                "render_source": "neoverse_runner",
            },
            capabilities={
                "site_world_package_registration": True,
                "site_world_registration": True,
                "session_reset": True,
                "session_step": True,
                "session_render": True,
                "session_state": True,
                "session_stream": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": True,
                "debug_render_outputs": True,
            },
            readiness=readiness,
        ).to_dict()

    def register_site_world_package(
        self,
        *,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Dict[str, Any]:
        scene_id = str(spec.get("scene_id") or registration.get("scene_id") or "").strip()
        capture_id = str(spec.get("capture_id") or registration.get("capture_id") or "").strip()
        if not scene_id or not capture_id:
            raise RuntimeError("site world registration requires scene_id and capture_id")
        site_world_id = str(registration.get("site_world_id") or _stable_id("siteworld", scene_id, capture_id)).strip()
        build_id = str(registration.get("build_id") or _stable_id("build", scene_id, capture_id, _utc_now_iso())).strip()
        workspace_dir = self._workspace_dir(site_world_id, build_id)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        conditioning_map = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
        local_map = dict(conditioning_map.get("local_paths") or {}) if isinstance(conditioning_map.get("local_paths"), Mapping) else {}
        conditioning_source = _preferred_local_path(
            local_map.get("keyframe_path"),
            local_map.get("raw_video_path"),
            conditioning_map.get("keyframe_uri"),
            conditioning_map.get("raw_video_uri"),
        )
        if conditioning_source is None:
            raise RuntimeError("site world build requires a local conditioning source")
        base_frame = (
            _extract_first_frame(conditioning_source)
            if conditioning_source.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi"}
            else _load_frame(conditioning_source)
        )
        base_frame_path = workspace_dir / "base_frame.png"
        _save_frame(base_frame_path, base_frame)

        runtime_layer_bundle = load_runtime_layer_bundle(spec)
        launchable, runtime_blockers, runtime_warnings = self.validate_spec(
            spec,
            protected_regions_manifest=runtime_layer_bundle["protected_regions_manifest"],
        )
        version_error = verify_canonical_package_version(
            spec=spec,
            protected_regions_manifest=runtime_layer_bundle["protected_regions_manifest"],
            canonical_render_policy=runtime_layer_bundle["canonical_render_policy"],
            presentation_variance_policy=runtime_layer_bundle["presentation_variance_policy"],
        )
        if version_error is not None:
            raise RuntimeError(version_error)
        runtime_layer_snapshots = snapshot_runtime_layer_bundle(runtime_layer_bundle, workspace_dir)
        runner_workspace = self.runner.prepare_site_world(
            site_world_id=site_world_id,
            workspace_dir=workspace_dir,
            spec=spec,
            registration=registration,
            health=health,
        )
        supported_cameras = list(registration.get("supported_cameras") or [])
        if not supported_cameras:
            for profile in spec.get("robot_profiles", []) or []:
                if not isinstance(profile, Mapping):
                    continue
                for camera in profile.get("observation_cameras", []) or []:
                    if isinstance(camera, Mapping):
                        camera_id = str(camera.get("id") or "").strip()
                        if camera_id and camera_id not in supported_cameras:
                            supported_cameras.append(camera_id)
        if not supported_cameras:
            supported_cameras = ["head_rgb"]

        persisted_spec = json.loads(json.dumps(dict(spec)))
        persisted_presentation = _presentation_map(persisted_spec)
        dedicated_presentation_bundle = _has_dedicated_presentation_bundle(spec)
        presentation_manifest_path = _preferred_local_path(
            persisted_presentation.get("presentation_world_manifest_path"),
            persisted_presentation.get("presentation_world_manifest_uri"),
        )
        runtime_demo_manifest_path = _preferred_local_path(
            persisted_presentation.get("runtime_demo_manifest_path"),
            persisted_presentation.get("runtime_demo_manifest_uri"),
        )
        presentation_derivation_mode = str(persisted_presentation.get("derivation_mode") or "").strip()
        presentation_ui_optional = bool(persisted_presentation.get("ui_optional"))

        if not dedicated_presentation_bundle:
            synthetic_presentation_manifest, synthetic_runtime_demo_manifest, presentation_update = (
                self._build_synthetic_presentation_manifests(
                    site_world_id=site_world_id,
                    build_id=build_id,
                    spec=spec,
                    registration=registration,
                    health=health,
                )
            )
            persisted_spec["presentation"] = presentation_update
            persisted_presentation = presentation_update
            presentation_manifest_path = Path(
                str(presentation_update["presentation_world_manifest_path"])
            ).resolve()
            runtime_demo_manifest_path = Path(
                str(presentation_update["runtime_demo_manifest_path"])
            ).resolve()
            presentation_derivation_mode = str(
                synthetic_presentation_manifest.get("derivation_mode") or "canonical_pretty"
            )
            presentation_ui_optional = bool(
                synthetic_runtime_demo_manifest.get("ui_optional", True)
            )
        elif presentation_manifest_path is None and str(
            persisted_presentation.get("presentation_world_manifest_path") or ""
        ).strip():
            presentation_manifest_path = Path(
                str(persisted_presentation.get("presentation_world_manifest_path") or "")
            ).expanduser().resolve()

        if dedicated_presentation_bundle and runtime_demo_manifest_path is None:
            synthetic_runtime_demo_manifest = {
                "schema_version": "v1",
                "site_world_id": site_world_id,
                "scene_id": registration.get("scene_id") or spec.get("scene_id"),
                "capture_id": registration.get("capture_id") or spec.get("capture_id"),
                "build_id": build_id,
                "status": "ready",
                "bundle_status": "ready",
                "derivation_mode": str(
                    persisted_presentation.get("derivation_mode") or "dedicated_presentation"
                ),
                "scene_kind": str(
                    persisted_presentation.get("scene_kind") or "derived_presentation"
                ),
                "fallback_policy": str(
                    persisted_presentation.get("fallback_policy") or "canonical_only"
                ),
                "canonical_package_uri": spec.get("canonical_package_uri"),
                "canonical_package_version": spec.get("canonical_package_version"),
                "ui_base_url": None,
                "ui_optional": True,
                "presentation_world_manifest_url": self._site_world_endpoint_url(
                    site_world_id,
                    "presentation-world-manifest",
                ),
            }
            runtime_demo_manifest_path = self._runtime_demo_manifest_path(site_world_id, build_id)
            _write_json(runtime_demo_manifest_path, synthetic_runtime_demo_manifest)
            persisted_presentation = {
                **persisted_presentation,
                "runtime_demo_manifest_path": str(runtime_demo_manifest_path),
                "runtime_demo_manifest_uri": "",
                "ui_optional": True,
                "ui_base_url": None,
            }
            persisted_spec["presentation"] = persisted_presentation
            presentation_ui_optional = True

        launchable = bool(health.get("launchable", True)) and launchable
        blockers = _dedupe_strings(list(registration.get("blockers") or []), list(health.get("blockers") or []), runtime_blockers)
        warnings = _dedupe_strings(list(registration.get("warnings") or []), list(health.get("warnings") or []), runtime_warnings)
        runtime_metadata = self.runtime_info(service_version="1.0.0")
        canonical_world_model = _canonical_world_model_map(persisted_spec)
        registration_payload = {
            **dict(registration),
            "schema_version": "v1",
            "runtime_kind": "neoverse_production",
            "production_grade": True,
            "site_world_id": site_world_id,
            "build_id": build_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "status": "ready" if launchable else _blocked_status(registration.get("status"), health.get("status")),
            "runtime_base_url": self.base_url,
            "websocket_base_url": self.ws_base_url,
            "vm_instance_id": os.getenv("VASTAI_INSTANCE_ID") or os.getenv("HOSTNAME") or "local-vm",
            "canonical_package_uri": spec.get("canonical_package_uri"),
            "canonical_package_version": spec.get("canonical_package_version"),
            "workspace_dir": str(workspace_dir),
            "workspace_manifest_path": str(runner_workspace.get("workspace_manifest_path") or ""),
            "conditioning_source_path": str(conditioning_source),
            "base_frame_path": str(base_frame_path),
            "runtime_layer_policy_snapshots": runtime_layer_snapshots,
            "supported_cameras": supported_cameras,
            "scenario_catalog": list(spec.get("scenario_catalog") or []),
            "start_state_catalog": list(spec.get("start_state_catalog") or []),
            "task_catalog": list(spec.get("task_catalog") or []),
            "robot_profiles": list(spec.get("robot_profiles") or []),
            "runtime_capabilities": {
                "supports_step_rollout": True,
                "supports_batch_rollout": True,
                "supports_camera_views": True,
                "supports_stream": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": True,
                "debug_render_outputs": True,
            },
            "runtime_engine_identity": dict(runtime_metadata.get("engine_identity") or {}),
            "runtime_model_identity": dict(runtime_metadata.get("model_identity") or {}),
            "runtime_checkpoint_identity": dict(runtime_metadata.get("checkpoint_identity") or {}),
            "registration_mode": str(registration.get("registration_mode") or "package_registration"),
            "intake_source": str(registration.get("intake_source") or "built_site_world_package"),
            "compatibility_notice": str(registration.get("compatibility_notice") or ""),
            "health_uri": f"{self.base_url}/v1/site-worlds/{site_world_id}/health",
            "generated_at": _utc_now_iso(),
            "blockers": blockers,
            "warnings": warnings,
            "grounding_status": spec.get("grounding_status") or (spec.get("runtime_layer_policy") or {}).get("grounding_status"),
            "ungrounded_reason": spec.get("ungrounded_reason") or (spec.get("runtime_layer_policy") or {}).get("ungrounded_reason"),
            "empty_index_cause": spec.get("empty_index_cause"),
            "primary_runtime_backend": str(persisted_spec.get("primary_runtime_backend") or "neoverse"),
            "canonical_world_model": canonical_world_model,
            "presentation_world_manifest_path": str(presentation_manifest_path or ""),
            "runtime_demo_manifest_path": str(runtime_demo_manifest_path or ""),
            "presentation_world_manifest_url": (
                self._site_world_endpoint_url(site_world_id, "presentation-world-manifest")
                if presentation_manifest_path and presentation_manifest_path.is_file()
                else None
            ),
            "runtime_demo_manifest_url": (
                self._site_world_endpoint_url(site_world_id, "runtime-demo-manifest")
                if runtime_demo_manifest_path and runtime_demo_manifest_path.is_file()
                else None
            ),
            "presentation_derivation_mode": presentation_derivation_mode or None,
            "presentation_ui_optional": presentation_ui_optional,
        }
        self._augment_runner_workspace_manifest(
            workspace_manifest_path=Path(str(runner_workspace.get("workspace_manifest_path") or "")).resolve(),
            spec=persisted_spec,
            registration_payload=registration_payload,
            conditioning_source=conditioning_source,
            base_frame_path=base_frame_path,
        )
        health_payload = {
            **dict(health),
            "schema_version": "v1",
            "runtime_kind": "neoverse_production",
            "production_grade": True,
            "site_world_id": site_world_id,
            "build_id": build_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "healthy": launchable,
            "launchable": launchable,
            "status": "healthy" if launchable else _blocked_status(health.get("status"), registration.get("status")),
            "blockers": blockers,
            "warnings": warnings,
            "canonical_package_version": spec.get("canonical_package_version"),
            "last_heartbeat_at": _utc_now_iso(),
            "runtime_base_url": self.base_url,
            "websocket_base_url": self.ws_base_url,
            "primary_runtime_backend": str(registration_payload.get("primary_runtime_backend") or "neoverse"),
            "canonical_world_model": canonical_world_model,
            "supported_cameras": supported_cameras,
            "runtime_capabilities": dict(registration_payload.get("runtime_capabilities") or {}),
            "runtime_engine_identity": dict(registration_payload.get("runtime_engine_identity") or {}),
            "runtime_model_identity": dict(registration_payload.get("runtime_model_identity") or {}),
            "runtime_checkpoint_identity": dict(registration_payload.get("runtime_checkpoint_identity") or {}),
            "readiness": self.runner.readiness(),
        }
        _write_json(self._site_world_spec_path(site_world_id), dict(persisted_spec))
        _write_json(self._site_world_state_path(site_world_id), registration_payload)
        _write_json(self._site_world_health_path(site_world_id), health_payload)
        self._presentation_bundle_renderer.invalidate(site_world_id)
        self._pose_preview_renderer.invalidate(site_world_id)
        return registration_payload

    def load_site_world(self, site_world_id: str) -> Dict[str, Any]:
        path = self._site_world_state_path(site_world_id)
        if not path.is_file():
            raise FileNotFoundError(site_world_id)
        return _read_json(path)

    def load_site_world_health(self, site_world_id: str) -> Dict[str, Any]:
        path = self._site_world_health_path(site_world_id)
        if not path.is_file():
            raise FileNotFoundError(site_world_id)
        payload = _read_json(path)
        payload["last_heartbeat_at"] = _utc_now_iso()
        _write_json(path, payload)
        return payload

    def _load_site_world_bundle(self, site_world_id: str) -> Dict[str, Any]:
        try:
            bundle = load_site_world_bundle(self._site_world_state_path(site_world_id), require_spec=True)
        except SiteWorldIntakeError as exc:
            raise RuntimeError(str(exc)) from exc
        return {
            "registration": bundle.registration,
            "health": bundle.health,
            "spec": bundle.spec,
            "resolved": bundle.resolved,
        }

    def _catalog_entry(self, entries: Sequence[Any], selected_id: str, *, label: str) -> Dict[str, Any]:
        for item in entries:
            if isinstance(item, Mapping) and (
                str(item.get("id") or "").strip() == selected_id
                or str(item.get("task_id") or "").strip() == selected_id
            ):
                return dict(item)
        raise RuntimeError(f"unknown {label}: {selected_id}")

    def _snapshot_path(self, session_id: str, snapshot_id: str) -> Path:
        return self._snapshot_dir(session_id) / f"{snapshot_id}.json"

    def _base_frame_path(self, site_world_id: str) -> Path:
        registration = self.load_site_world(site_world_id)
        path = Path(
            str(registration.get("base_frame_path") or registration.get("seed_frame_path") or "")
        ).resolve()
        if not path.is_file():
            raise RuntimeError(f"base frame missing for {site_world_id}")
        return path

    def _augment_runner_workspace_manifest(
        self,
        *,
        workspace_manifest_path: Path,
        spec: Mapping[str, Any],
        registration_payload: Mapping[str, Any],
        conditioning_source: Path,
        base_frame_path: Path,
    ) -> None:
        if not workspace_manifest_path.is_file():
            return
        conditioning = dict(spec.get("conditioning") or {}) if isinstance(spec.get("conditioning"), Mapping) else {}
        conditioning_local = (
            dict(conditioning.get("local_paths") or {})
            if isinstance(conditioning.get("local_paths"), Mapping)
            else {}
        )
        geometry = dict(spec.get("geometry") or {}) if isinstance(spec.get("geometry"), Mapping) else {}
        presentation = _presentation_map(spec)
        workspace_manifest = _read_json(workspace_manifest_path)
        manifest_registration = (
            dict(workspace_manifest.get("registration") or {})
            if isinstance(workspace_manifest.get("registration"), Mapping)
            else {}
        )
        manifest_registration.update(
            {
                "site_world_id": registration_payload.get("site_world_id"),
                "build_id": registration_payload.get("build_id"),
                "conditioning_source_path": str(conditioning_source),
                "base_frame_path": str(base_frame_path),
                "arkit_poses_path": str(
                    _preferred_local_path(
                        conditioning_local.get("arkit_poses_path"),
                        conditioning.get("arkit_poses_path"),
                    )
                    or ""
                ),
                "arkit_intrinsics_path": str(
                    _preferred_local_path(
                        conditioning_local.get("arkit_intrinsics_path"),
                        conditioning.get("arkit_intrinsics_path"),
                    )
                    or ""
                ),
                "arkit_depth_path": str(
                    _preferred_local_path(
                        conditioning_local.get("arkit_depth_path"),
                        conditioning.get("arkit_depth_path"),
                    )
                    or ""
                ),
                "scene_memory_manifest_path": str(
                    _preferred_local_path(
                        conditioning_local.get("scene_memory_manifest_path"),
                        conditioning.get("scene_memory_manifest_path"),
                    )
                    or ""
                ),
                "conditioning_bundle_path": str(
                    _preferred_local_path(
                        conditioning_local.get("conditioning_bundle_path"),
                        conditioning.get("conditioning_bundle_path"),
                    )
                    or ""
                ),
                "object_index_path": str(
                    _preferred_local_path(
                        conditioning_local.get("object_index_path"),
                        geometry.get("object_index_path"),
                    )
                    or ""
                ),
                "object_geometry_manifest_path": str(
                    _preferred_local_path(
                        conditioning_local.get("object_geometry_manifest_path"),
                        geometry.get("object_geometry_manifest_path"),
                    )
                    or ""
                ),
                "presentation_world_manifest_path": str(
                    _preferred_local_path(
                        presentation.get("presentation_world_manifest_path"),
                        presentation.get("presentation_world_manifest_uri"),
                    )
                    or ""
                ),
                "presentation_primary_asset_path": str(
                    _preferred_local_path(
                        presentation.get("primary_asset_path"),
                        presentation.get("primary_asset_uri"),
                    )
                    or ""
                ),
                "runtime_demo_manifest_path": str(
                    _preferred_local_path(
                        presentation.get("runtime_demo_manifest_path"),
                        presentation.get("runtime_demo_manifest_uri"),
                    )
                    or ""
                ),
            }
        )
        workspace_manifest["registration"] = manifest_registration
        workspace_manifest["resolved_spec"] = dict(spec)
        _write_json(workspace_manifest_path, workspace_manifest)

    def _runtime_layer_bundle_for_site_world(self, site_world_id: str) -> Dict[str, Any]:
        registration = self.load_site_world(site_world_id)
        snapshots = dict(registration.get("runtime_layer_policy_snapshots") or {})
        protected_path = Path(str(snapshots.get("protected_regions_manifest_path") or "")).resolve()
        render_policy_path = Path(str(snapshots.get("canonical_render_policy_path") or "")).resolve()
        variance_policy_path = Path(str(snapshots.get("presentation_variance_policy_path") or "")).resolve()
        return {
            "protected_regions_manifest": _read_json(protected_path),
            "canonical_render_policy": _read_json(render_policy_path),
            "presentation_variance_policy": _read_json(variance_policy_path),
        }

    def create_session(
        self,
        site_world_id: str,
        *,
        session_id: Optional[str] = None,
        requested_backend: Optional[str] = None,
        robot_profile_id: str,
        task_id: str,
        scenario_id: str,
        start_state_id: str,
        notes: str = "",
        canonical_package_uri: str | None = None,
        canonical_package_version: str | None = None,
        prompt: str | None = None,
        trajectory: Mapping[str, Any] | str | None = None,
        presentation_model: str | None = None,
        debug_mode: bool | None = None,
        unsafe_allow_blocked_site_world: bool = False,
    ) -> Dict[str, Any]:
        del requested_backend
        bundle = self._load_site_world_bundle(site_world_id)
        registration = bundle["registration"]
        site_world = bundle["resolved"]
        health = self.load_site_world_health(site_world_id)
        allow_blocked_site_world = bool(unsafe_allow_blocked_site_world) or _env_truthy("BLUEPRINT_UNSAFE_ALLOW_BLOCKED_SITE_WORLD")
        if not bool(health.get("launchable")) and not allow_blocked_site_world:
            raise RuntimeError(f"site world {site_world_id} is not launchable")
        if not bool(self.runner.readiness().get("ready")):
            raise RuntimeError("NeoVerse production runtime is not ready for session creation.")

        expected_package_uri = str(site_world.get("canonical_package_uri") or "").strip()
        expected_package_version = str(site_world.get("canonical_package_version") or "").strip()
        if canonical_package_uri and expected_package_uri and str(canonical_package_uri).strip() != expected_package_uri:
            raise RuntimeError("canonical_package_uri_mismatch")
        if canonical_package_version and expected_package_version and str(canonical_package_version).strip() != expected_package_version:
            raise RuntimeError("canonical_package_version_mismatch")

        robot_profile = self._catalog_entry(site_world.get("robot_profiles", []), robot_profile_id, label="robot profile")
        task_entry = self._catalog_entry(site_world.get("task_catalog", []), task_id, label="task")
        scenario_entry = self._catalog_entry(site_world.get("scenario_catalog", []), scenario_id, label="scenario")
        start_state_entry = self._catalog_entry(site_world.get("start_state_catalog", []), start_state_id, label="start state")
        trajectory_payload = normalize_trajectory_payload(trajectory)
        session_id = str(session_id or _stable_id("session", site_world_id, robot_profile_id, task_id, _utc_now_iso())).strip()
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        session_state = {
            "schema_version": "v1",
            "session_id": session_id,
            "runtime_kind": "neoverse_production",
            "production_grade": True,
            "site_world_id": site_world_id,
            "build_id": registration.get("build_id"),
            "scene_id": registration.get("scene_id"),
            "capture_id": registration.get("capture_id"),
            "runtime_base_url": registration.get("runtime_base_url"),
            "status": "ready",
            "canonical_package_uri": expected_package_uri,
            "canonical_package_version": expected_package_version,
            "robot_profile": robot_profile,
            "task": task_entry,
            "scenario": scenario_entry,
            "start_state": start_state_entry,
            "notes": notes,
            "presentation_config": {
                "prompt": prompt,
                "trajectory": trajectory_payload,
                "presentation_model": presentation_model or "runtime_default",
                "debug_mode": bool(debug_mode),
                "unsafe_allow_blocked_site_world": allow_blocked_site_world,
            },
            "unsafe_allow_blocked_site_world": allow_blocked_site_world,
            "runtime_engine_identity": dict(registration.get("runtime_engine_identity") or {}),
            "runtime_model_identity": dict(registration.get("runtime_model_identity") or self.runner.model_identity()),
            "runtime_checkpoint_identity": dict(registration.get("runtime_checkpoint_identity") or self.runner.checkpoint_identity()),
            "quality_flags": {
                **_quality_flag_baseline(site_world),
                "presentation_quality": "normal",
                "editable_ratio": 0.0,
                "locked_ratio": 0.0,
            },
            "protected_region_violations": [],
            "debug_artifacts": {},
            "step_index": 0,
            "done": False,
            "success": None,
            "failure_reason": None,
            "reward": 0.0,
            "action_trace": [],
            "latest_render_paths": {},
            "explorer_state": {
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0},
                "camera_id": "head_rgb",
                "latest_frame_paths": {},
                "grounded_source": None,
                "refine_status": "idle",
                "quality_flags": {},
                "debug_artifacts": {},
                "viewport": {},
                "snapshot_id": None,
            },
            "current_world_snapshot_id": None,
            "current_world_snapshot_path": None,
            "render_manifest_path": None,
            "created_at": _utc_now_iso(),
        }
        _write_json(self._session_state_path(session_id), session_state)
        update_presentation_session_manifest(
            session_dir=session_dir,
            payload={
                "schema_version": "v1",
                "session_id": session_id,
                "site_world_id": site_world_id,
                "canonical_package_uri": expected_package_uri,
                "canonical_package_version": expected_package_version,
                "prompt": prompt,
                "trajectory": trajectory_payload,
                "presentation_model": presentation_model or "runtime_default",
                "debug_mode": bool(debug_mode),
                "quality_flags": session_state["quality_flags"],
                "protected_region_violations": session_state["protected_region_violations"],
                "latest_debug_artifacts": {},
            },
        )
        return {
            "session_id": session_id,
            "site_world_id": site_world_id,
            "build_id": registration.get("build_id"),
            "status": "ready",
            "runtime_kind": "neoverse_production",
            "production_grade": True,
            "canonical_package_version": expected_package_version,
            "presentation_config": dict(session_state["presentation_config"]),
            "unsafe_allow_blocked_site_world": allow_blocked_site_world,
            "quality_flags": dict(session_state["quality_flags"]),
            "protected_region_violations": list(session_state["protected_region_violations"]),
            "debug_artifacts": dict(session_state["debug_artifacts"]),
            "runtime_engine_identity": dict(session_state["runtime_engine_identity"]),
            "runtime_model_identity": dict(session_state["runtime_model_identity"]),
            "runtime_checkpoint_identity": dict(session_state["runtime_checkpoint_identity"]),
            "runtime_capabilities": registration.get("runtime_capabilities", {}),
            "observation_cameras": list(robot_profile.get("observation_cameras") or []),
        }

    def load_session(self, session_id: str) -> Dict[str, Any]:
        path = self._session_state_path(session_id)
        if not path.is_file():
            raise FileNotFoundError(session_id)
        return _read_json(path)

    def _persist_session_state(self, session_state: Dict[str, Any]) -> None:
        _write_json(self._session_state_path(str(session_state["session_id"])), session_state)

    def _record_render_failure(
        self,
        session_state: Dict[str, Any],
        *,
        code: str,
        message: str,
    ) -> None:
        session_state["latest_render_error_code"] = code
        session_state["latest_render_error_message"] = message
        self._persist_session_state(session_state)

    def _clear_render_failure(self, session_state: Dict[str, Any]) -> None:
        session_state["latest_render_error_code"] = None
        session_state["latest_render_error_message"] = None

    def _default_explorer_pose(self) -> Dict[str, float]:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}

    def _normalized_explorer_pose(self, pose: Mapping[str, Any] | None) -> Dict[str, float]:
        source = dict(pose or {})
        return {
            "x": float(source.get("x") or 0.0),
            "y": float(source.get("y") or 0.0),
            "z": float(source.get("z") or 0.0),
            "yaw": float(source.get("yaw") or 0.0),
            "pitch": float(source.get("pitch") or 0.0),
        }

    def _explorer_state(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        existing = dict(session_state.get("explorer_state") or {})
        existing.setdefault("pose", self._default_explorer_pose())
        existing.setdefault("camera_id", "head_rgb")
        existing.setdefault("latest_frame_paths", {})
        existing.setdefault("grounded_source", None)
        existing.setdefault("refine_status", "idle")
        existing.setdefault("quality_flags", {})
        existing.setdefault("debug_artifacts", {})
        existing.setdefault("viewport", {})
        existing.setdefault("snapshot_id", None)
        session_state["explorer_state"] = existing
        return existing

    def _recover_render_state(self, session_state: Dict[str, Any]) -> bool:
        session_id = str(session_state.get("session_id") or "")
        snapshot_id = str(session_state.get("current_world_snapshot_id") or "").strip()
        if not session_id or not snapshot_id:
            return False
        render_manifest_path = self._session_dir(session_id) / "renders" / snapshot_id / "render_manifest.json"
        if not render_manifest_path.is_file():
            return False

        manifest = _read_json(render_manifest_path)
        latest_render_paths = {
            str(key): str(value)
            for key, value in dict(manifest.get("camera_frame_paths") or {}).items()
            if str(key).strip() and Path(str(value)).is_file()
        }
        if not latest_render_paths:
            return False

        session_state["latest_render_paths"] = latest_render_paths
        session_state["render_manifest_path"] = str(render_manifest_path)
        session_state["quality_flags"] = dict(manifest.get("quality_flags") or session_state.get("quality_flags") or {})
        session_state["protected_region_violations"] = list(
            manifest.get("protected_region_violations") or session_state.get("protected_region_violations") or []
        )
        session_state["debug_artifacts"] = dict(manifest.get("debug_artifacts") or session_state.get("debug_artifacts") or {})
        return True

    def _observation_from_current_state(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(session_state["session_id"])
        snapshot_path = Path(str(session_state.get("current_world_snapshot_path") or "")).resolve()
        snapshot = _read_json(snapshot_path) if snapshot_path.is_file() else self._snapshot_from_state(session_state, self._initial_world_state(session_state))
        state_payload = self.session_state(session_id)
        observation = dict(state_payload.get("observation") or {})
        if observation.get("worldSnapshot") is None:
            observation["worldSnapshot"] = {
                "snapshotId": session_state.get("current_world_snapshot_id"),
                "world_state": snapshot.get("world_state"),
                "status": session_state.get("status"),
            }
        return observation

    def _initial_world_state(self, session_state: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "step_index": 0,
            "task_progress": 0.0,
            "robot_state": {
                "base_pose": [0.0, 0.0, 0.0],
                "joint_targets": [0.0] * int(((session_state.get("robot_profile") or {}).get("action_space") or {}).get("dim") or 7),
            },
            "task_status": "running",
        }

    def _snapshot_from_state(self, session_state: Dict[str, Any], world_state: Dict[str, Any]) -> Dict[str, Any]:
        snapshot_id = _stable_id(
            "snapshot",
            str(session_state["session_id"]),
            str(world_state.get("step_index", 0)),
            json.dumps(world_state, sort_keys=True),
        )
        snapshot = {
            "snapshot_id": snapshot_id,
            "session_id": session_state["session_id"],
            "site_world_id": session_state["site_world_id"],
            "task_id": (session_state.get("task") or {}).get("id"),
            "scenario_id": (session_state.get("scenario") or {}).get("id"),
            "start_state_id": (session_state.get("start_state") or {}).get("id"),
            "world_state": world_state,
            "presentation_config": dict(session_state.get("presentation_config") or {}),
            "canonical_package_version": session_state.get("canonical_package_version"),
            "runtime_kind": "neoverse_production",
            "generated_at": _utc_now_iso(),
        }
        return snapshot

    def _save_snapshot(self, session_id: str, snapshot: Dict[str, Any]) -> Path:
        path = self._snapshot_path(session_id, str(snapshot["snapshot_id"]))
        _write_json(path, snapshot)
        return path

    def _render_camera_selection(
        self,
        robot_profile: Mapping[str, Any],
        *,
        requested_camera_ids: Optional[Sequence[str]] = None,
    ) -> list[Dict[str, Any]]:
        cameras = [dict(item) for item in robot_profile.get("observation_cameras", []) if isinstance(item, Mapping)]
        if not cameras:
            cameras = [{"id": "head_rgb", "role": "head", "required": True, "default_enabled": True}]
        if requested_camera_ids:
            requested = {str(item).strip() for item in requested_camera_ids if str(item).strip()}
            selected = [camera for camera in cameras if str(camera.get("id") or "").strip() in requested]
            return selected or cameras[:1]
        preferred = [camera for camera in cameras if bool(camera.get("required", False))]
        if not preferred:
            preferred = [camera for camera in cameras if bool(camera.get("default_enabled", False))]
        return preferred[:1] or cameras[:1]

    def _canonical_frame_from_runner_output(self, path: Path, camera_id: str) -> Any:
        if not path.is_file():
            raise RuntimeError(f"NeoVerse runner frame missing for camera {camera_id}")
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            frame = _extract_first_frame(path)
        else:
            frame = _load_frame(path)
        return _coerce_camera_frame(frame, camera_id)

    def _fallback_runner_payload(
        self,
        *,
        render_dir: Path,
        cameras: Sequence[Mapping[str, Any]],
        base_frame_path: Path,
        error: Exception,
    ) -> Dict[str, Any]:
        base_frame = _load_frame(base_frame_path)
        camera_frames: list[Dict[str, Any]] = []
        for camera in cameras:
            camera_id = str(camera.get("id") or camera.get("cameraId") or "head_rgb").strip() or "head_rgb"
            canonical_frame = _coerce_camera_frame(base_frame, camera_id)
            output_path = render_dir / f"{camera_id}-runner-fallback.png"
            _save_frame(output_path, canonical_frame)
            camera_frames.append(
                {
                    "cameraId": camera_id,
                    "path": str(output_path),
                    "fallback": True,
                }
            )
        return {
            "camera_frames": camera_frames,
            "quality_flags": {
                "presentation_quality": "degraded",
                "runner_fallback": "base_frame",
            },
            "protected_region_violations": [],
            "debug_artifacts": {
                "runner_fallback": "base_frame",
                "runner_error": str(error),
            },
        }

    def _composite_camera_frame(
        self,
        *,
        canonical_frame: Any,
        runtime_layer_bundle: Mapping[str, Any],
        session_state: Dict[str, Any],
        session_dir: Path,
        snapshot: Mapping[str, Any],
        camera_id: str,
    ) -> Dict[str, Any]:
        try:
            return composite_runtime_layer(
                canonical_frame=canonical_frame,
                protected_regions_manifest=runtime_layer_bundle["protected_regions_manifest"],
                canonical_render_policy=runtime_layer_bundle["canonical_render_policy"],
                presentation_config=dict(session_state.get("presentation_config") or {}),
                presentation_variance_policy=runtime_layer_bundle["presentation_variance_policy"],
                session_dir=session_dir,
                step_index=int((snapshot.get("world_state") or {}).get("step_index", 0)),
                camera_id=camera_id,
            )
        except Exception as exc:
            self._log_runtime_stage(
                "render_snapshot.composite_fallback",
                session_id=session_state.get("session_id"),
                camera_id=camera_id,
                error=str(exc),
            )
            return {
                "frame": canonical_frame,
                "quality_flags": {
                    "presentation_quality": "degraded",
                    "composite_fallback": "canonical_frame",
                },
                "protected_region_violations": [],
                "debug_artifacts": {
                    "composite_fallback": "canonical_frame",
                    "composite_error": str(exc),
                },
            }

    def _authoritative_runner_quality_flags(
        self,
        *,
        session_state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        site_world_id = str(session_state["site_world_id"])
        spec = self._load_site_world_bundle(site_world_id)["resolved"]
        return {
            **_quality_flag_baseline(spec),
            "world_model_backend": str(spec.get("world_model_backend") or "neoverse"),
            "scene_representation": str(spec.get("scene_representation") or "neoverse_video_world_model_v1"),
            "render_source": str(spec.get("runtime_render_source") or spec.get("render_source") or "neoverse_full_capture"),
            "presentation_quality": "neoverse_generated",
            "fallback_mode": "none",
        }

    def _base_frame_preview_runner_payload(
        self,
        *,
        render_dir: Path,
        cameras: Sequence[Mapping[str, Any]],
        base_frame_path: Path,
    ) -> Dict[str, Any]:
        base_frame = _load_frame(base_frame_path)
        camera_frames: list[Dict[str, Any]] = []
        for camera in cameras:
            camera_id = str(camera.get("id") or camera.get("cameraId") or "head_rgb").strip() or "head_rgb"
            preview_frame = _coerce_camera_frame(base_frame, camera_id)
            output_path = render_dir / f"{camera_id}-preview.png"
            _save_frame(output_path, preview_frame)
            camera_frames.append(
                {
                    "cameraId": camera_id,
                    "path": str(output_path),
                    "preview": True,
                }
            )
        refinement_status = "pending" if self.enable_async_render_refinement else "disabled"
        return {
            "camera_frames": camera_frames,
            "quality_flags": {
                "presentation_quality": "preview",
                "render_mode": "preview",
                "refinement_status": refinement_status,
            },
            "protected_region_violations": [],
            "debug_artifacts": {
                "preview_mode": "base_frame",
                "render_mode": "preview",
                "refinement_status": refinement_status,
            },
        }

    def _presentation_runner_payload(
        self,
        *,
        site_world_id: str,
        spec: Mapping[str, Any],
        world_state: Mapping[str, Any],
        render_dir: Path,
        cameras: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        scene = self._pose_preview_renderer.scene(site_world_id, spec)
        camera_frames: list[Dict[str, Any]] = []
        per_camera_debug: dict[str, Any] = {}
        for camera in cameras:
            camera_id = str(camera.get("id") or camera.get("cameraId") or "head_rgb").strip() or "head_rgb"
            preview_frame, debug = self._presentation_bundle_renderer.render_camera(
                site_world_id=site_world_id,
                spec=spec,
                view_config=scene.view_config(world_state, camera_id),
            )
            output_path = render_dir / f"{camera_id}-presentation.png"
            _save_frame(output_path, preview_frame)
            camera_frames.append(
                {
                    "cameraId": camera_id,
                    "path": str(output_path),
                    "preview": True,
                    "presentation_bundle": True,
                }
            )
            per_camera_debug[camera_id] = debug
        refinement_status = "pending" if self.enable_async_render_refinement else "disabled"
        primary_camera_id = str(camera_frames[0]["cameraId"]) if camera_frames else ""
        primary_debug = dict(per_camera_debug.get(primary_camera_id) or {})
        return {
            "camera_frames": camera_frames,
            "quality_flags": {
                **_quality_flag_baseline(spec),
                "presentation_quality": "preview",
                "render_mode": "preview",
                "refinement_status": refinement_status,
                "preview_mode": "presentation_bundle",
                "preview_source": "server_gsplat",
                "render_source": "presentation_bundle",
                "fallback_mode": "arkit_rgbd_last_resort",
                "renderer_backend": "gsplat",
                "presentation_bundle_status": str(primary_debug.get("presentation_bundle_status") or "ready"),
                "display_orientation": primary_debug.get("display_orientation"),
            },
            "protected_region_violations": [],
            "debug_artifacts": {
                "preview_mode": "presentation_bundle",
                "preview_source": "server_gsplat",
                "renderer_backend": "gsplat",
                "render_mode": "preview",
                "refinement_status": refinement_status,
                "presentation_bundle_preview": primary_debug,
                "presentation_bundle_preview_per_camera": per_camera_debug,
            },
        }

    def _canonical_world_runner_payload(
        self,
        *,
        site_world_id: str,
        spec: Mapping[str, Any],
        world_state: Mapping[str, Any],
        render_dir: Path,
        cameras: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        scene = self._pose_preview_renderer.scene(site_world_id, spec)
        camera_frames: list[Dict[str, Any]] = []
        per_camera_debug: dict[str, Any] = {}
        renderer_spec = _spec_with_canonical_world_presentation(spec)
        for camera in cameras:
            camera_id = str(camera.get("id") or camera.get("cameraId") or "head_rgb").strip() or "head_rgb"
            preview_frame, debug = self._presentation_bundle_renderer.render_camera(
                site_world_id=site_world_id,
                spec=renderer_spec,
                view_config=scene.view_config(world_state, camera_id),
            )
            output_path = render_dir / f"{camera_id}-canonical-world.png"
            _save_frame(output_path, preview_frame)
            camera_frames.append(
                {
                    "cameraId": camera_id,
                    "path": str(output_path),
                    "preview": True,
                    "canonical_world_model": True,
                }
            )
            per_camera_debug[camera_id] = debug
        refinement_status = "pending" if self.enable_async_render_refinement else "disabled"
        primary_camera_id = str(camera_frames[0]["cameraId"]) if camera_frames else ""
        primary_debug = dict(per_camera_debug.get(primary_camera_id) or {})
        return {
            "camera_frames": camera_frames,
            "quality_flags": {
                **_quality_flag_baseline(spec),
                "presentation_quality": "preview",
                "render_mode": "preview",
                "refinement_status": refinement_status,
                "preview_mode": "canonical_world_model",
                "preview_source": "canonical_world_model",
                "render_source": "canonical_world_model",
                "fallback_mode": "arkit_rgbd_last_resort",
                "world_model_backend": str(_canonical_world_model_map(spec).get("world_model_backend") or "neoverse"),
                "scene_representation": str(_canonical_world_model_map(spec).get("scene_representation") or "gsplat_scene_v1"),
                "renderer_backend": str(primary_debug.get("renderer_backend") or "gsplat"),
                "display_orientation": primary_debug.get("display_orientation"),
            },
            "protected_region_violations": [],
            "debug_artifacts": {
                "preview_mode": "canonical_world_model",
                "preview_source": "canonical_world_model",
                "world_model_backend": str(_canonical_world_model_map(spec).get("world_model_backend") or "neoverse"),
                "scene_representation": str(_canonical_world_model_map(spec).get("scene_representation") or "gsplat_scene_v1"),
                "renderer_backend": str(primary_debug.get("renderer_backend") or "gsplat"),
                "render_mode": "preview",
                "refinement_status": refinement_status,
                "canonical_world_preview": primary_debug,
                "canonical_world_preview_per_camera": per_camera_debug,
            },
        }

    def _preview_runner_payload(
        self,
        *,
        session_state: Dict[str, Any],
        snapshot: Mapping[str, Any],
        render_dir: Path,
        cameras: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        site_world_id = str(session_state["site_world_id"])
        bundle = self._load_site_world_bundle(site_world_id)
        spec = dict(bundle["resolved"])
        if isinstance(bundle.get("spec"), Mapping):
            original_presentation = _presentation_map(bundle["spec"])
            resolved_presentation = _presentation_map(spec)
            if original_presentation:
                spec["presentation"] = {
                    **original_presentation,
                    **resolved_presentation,
                }
        world_state = dict(snapshot.get("world_state") or {})
        if _has_canonical_world_model_bundle(spec):
            try:
                return self._canonical_world_runner_payload(
                    site_world_id=site_world_id,
                    spec=spec,
                    world_state=world_state,
                    render_dir=render_dir,
                    cameras=cameras,
                )
            except Exception as exc:
                self._log_runtime_stage(
                    "preview.canonical_world_fallback",
                    site_world_id=site_world_id,
                    session_id=session_state.get("session_id"),
                    snapshot_id=snapshot.get("snapshot_id"),
                    error=str(exc),
                )
        presentation_error: Optional[str] = None
        if _has_dedicated_presentation_bundle(spec):
            try:
                return self._presentation_runner_payload(
                    site_world_id=site_world_id,
                    spec=spec,
                    world_state=world_state,
                    render_dir=render_dir,
                    cameras=cameras,
                )
            except Exception as exc:
                presentation_error = str(exc)
                logger_message = {
                    "site_world_id": site_world_id,
                    "session_id": session_state.get("session_id"),
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "error": presentation_error,
                }
                self._log_runtime_stage("preview.presentation_fallback", **logger_message)
        camera_frames: list[Dict[str, Any]] = []
        per_camera_debug: dict[str, Any] = {}
        for camera in cameras:
            camera_id = str(camera.get("id") or camera.get("cameraId") or "head_rgb").strip() or "head_rgb"
            preview_frame, debug, _coverage_mask = self._pose_preview_renderer.render_camera(
                site_world_id=site_world_id,
                spec=spec,
                world_state=world_state,
                camera_id=camera_id,
            )
            output_path = render_dir / f"{camera_id}-preview.png"
            _save_frame(output_path, preview_frame)
            camera_frames.append(
                {
                    "cameraId": camera_id,
                    "path": str(output_path),
                    "preview": True,
                    "pose_driven": True,
                }
            )
            per_camera_debug[camera_id] = debug
        refinement_status = "pending" if self.enable_async_render_refinement else "disabled"
        primary_camera_id = str(camera_frames[0]["cameraId"]) if camera_frames else ""
        return {
            "camera_frames": camera_frames,
            "quality_flags": {
                **_quality_flag_baseline(spec),
                "presentation_quality": "preview",
                "render_mode": "preview",
                "refinement_status": refinement_status,
                "preview_mode": "pose_driven",
                "preview_source": "arkit_rgbd_last_resort",
                "render_source": "arkit_rgbd_last_resort",
                "fallback_mode": "arkit_rgbd_last_resort",
                "presentation_render_error": presentation_error,
            },
            "protected_region_violations": [],
            "debug_artifacts": {
                "preview_mode": "pose_driven",
                "preview_source": "arkit_rgbd",
                "fallback_mode": "canonical_only",
                "presentation_render_error": presentation_error,
                "render_mode": "preview",
                "refinement_status": refinement_status,
                "pose_driven_preview": dict(per_camera_debug.get(primary_camera_id) or {}),
                "pose_driven_preview_per_camera": per_camera_debug,
            },
        }

    def _materialize_render_snapshot(
        self,
        *,
        session_state: Dict[str, Any],
        snapshot: Dict[str, Any],
        runner_payload: Mapping[str, Any],
        cameras: Sequence[Mapping[str, Any]],
        render_dir: Path,
        persist_session_state: bool,
    ) -> Dict[str, Any]:
        session_id = str(session_state["session_id"])
        site_world_id = str(session_state["site_world_id"])
        session_dir = self._session_dir(session_id)
        runtime_layer_bundle = self._runtime_layer_bundle_for_site_world(site_world_id)
        frame_records = [dict(item) for item in runner_payload.get("camera_frames", []) if isinstance(item, Mapping)]
        if not frame_records:
            raise RuntimeError("NeoVerse runner did not return any camera frames.")
        latest_render_paths: Dict[str, str] = {}
        camera_summaries: list[Dict[str, Any]] = []
        primary_camera_id = ""
        quality_flags = {
            **dict(session_state.get("quality_flags") or {}),
            **dict(runner_payload.get("quality_flags") or {"presentation_quality": "normal"}),
        }
        protected_region_violations = [
            dict(item) for item in runner_payload.get("protected_region_violations", []) if isinstance(item, Mapping)
        ]
        debug_artifacts = dict(runner_payload.get("debug_artifacts") or {})
        authoritative_runner_output = (
            str(runner_payload.get("quality_flags", {}).get("presentation_quality") or "").strip() == "neoverse_generated"
            and not protected_region_violations
        )
        for camera in cameras:
            camera_id = str(camera.get("id") or "").strip()
            if not camera_id:
                continue
            frame_record = next((item for item in frame_records if str(item.get("cameraId") or "") == camera_id), None)
            if frame_record is None:
                continue
            raw_path = Path(str(frame_record.get("path") or "")).resolve()
            generated_frame = self._canonical_frame_from_runner_output(raw_path, camera_id)
            if authoritative_runner_output:
                composite = {
                    "frame": generated_frame,
                    "quality_flags": self._authoritative_runner_quality_flags(session_state=session_state),
                    "protected_region_violations": [],
                    "debug_artifacts": {"authoritative_render": str(raw_path)},
                }
            else:
                self._log_runtime_stage(
                    "render_snapshot.composite.start",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    camera_id=camera_id,
                )
                composite_started = time.monotonic()
                composite = self._composite_camera_frame(
                    canonical_frame=generated_frame,
                    runtime_layer_bundle=runtime_layer_bundle,
                    session_state=session_state,
                    session_dir=session_dir,
                    snapshot=snapshot,
                    camera_id=camera_id,
                )
                self._log_runtime_stage(
                    "render_snapshot.composite.done",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    camera_id=camera_id,
                    elapsed_ms=round((time.monotonic() - composite_started) * 1000),
                )
            output_path = render_dir / f"{camera_id}.png"
            _save_frame(output_path, composite["frame"])
            latest_render_paths[camera_id] = str(output_path)
            if not primary_camera_id:
                primary_camera_id = camera_id
                merged_quality_flags = dict(quality_flags)
                merged_quality_flags.update(
                    {
                        key: value
                        for key, value in dict(composite.get("quality_flags") or {}).items()
                        if value is not None
                    }
                )
                quality_flags = merged_quality_flags
                merged_debug_artifacts = dict(debug_artifacts)
                merged_debug_artifacts.update(dict(composite.get("debug_artifacts") or {}))
                debug_artifacts = merged_debug_artifacts
            protected_region_violations.extend(
                [
                    {**dict(item), "camera_id": camera_id}
                    for item in composite.get("protected_region_violations", [])
                    if isinstance(item, Mapping)
                ]
            )
            camera_summaries.append(
                {
                    "cameraId": camera_id,
                    "role": str(camera.get("role") or ""),
                    "required": bool(camera.get("required", False)),
                    "available": True,
                    "framePath": f"{self.base_url}/v1/sessions/{session_id}/render?camera_id={camera_id}",
                }
            )
        render_manifest = {
            "snapshot_id": snapshot["snapshot_id"],
            "camera_frame_paths": latest_render_paths,
            "primary_camera_id": primary_camera_id,
            "quality_flags": quality_flags,
            "protected_region_violations": protected_region_violations,
            "debug_artifacts": debug_artifacts,
        }
        _write_json(render_dir / "render_manifest.json", render_manifest)
        if persist_session_state:
            session_state["latest_render_paths"] = latest_render_paths
            session_state["render_manifest_path"] = str(render_dir / "render_manifest.json")
            session_state["quality_flags"] = quality_flags
            session_state["protected_region_violations"] = protected_region_violations
            session_state["debug_artifacts"] = debug_artifacts
        self._log_runtime_stage(
            "render_snapshot.persist_manifest",
            session_id=session_id,
            snapshot_id=snapshot.get("snapshot_id"),
            primary_camera_id=primary_camera_id,
            render_manifest_path=str(render_dir / "render_manifest.json"),
        )
        update_presentation_session_manifest(
            session_dir=session_dir,
            payload={
                "quality_flags": quality_flags,
                "protected_region_violations": protected_region_violations,
                "latest_debug_artifacts": debug_artifacts,
            },
        )
        return {
            "frame_path": f"{self.base_url}/v1/sessions/{session_id}/render?camera_id={primary_camera_id}",
            "primaryCameraId": primary_camera_id,
            "cameraFrames": camera_summaries,
            "runtimeMetadata": {
                "site_world_id": session_state.get("site_world_id"),
                "build_id": session_state.get("build_id"),
                "step_index": (snapshot.get("world_state") or {}).get("step_index"),
                "runtime_kind": "neoverse_production",
                "production_grade": True,
                "canonical_package_version": session_state.get("canonical_package_version"),
                "quality_flags": quality_flags,
                "protected_region_violations": protected_region_violations,
                "debug_artifacts": debug_artifacts,
            },
            "worldSnapshot": {
                "snapshotId": snapshot["snapshot_id"],
                "task_id": (session_state.get("task") or {}).get("id"),
                "scenario_id": (session_state.get("scenario") or {}).get("id"),
                "start_state_id": (session_state.get("start_state") or {}).get("id"),
                "status": session_state.get("status"),
                "world_state": snapshot.get("world_state"),
            },
        }

    def _schedule_render_refinement(self, session_id: str, snapshot_id: str) -> None:
        if not self.enable_async_render_refinement or self._render_refinement_executor is None:
            return
        with self._render_refinement_lock:
            self._render_refinement_targets[session_id] = snapshot_id
            current = self._render_refinement_futures.get(session_id)
            if current is not None and not current.done():
                self._log_runtime_stage(
                    "render_refinement.coalesced",
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                )
                return
            future = self._render_refinement_executor.submit(self._run_render_refinement_loop, session_id)
            self._render_refinement_futures[session_id] = future

    def _run_render_refinement_loop(self, session_id: str) -> None:
        try:
            while True:
                with self._render_refinement_lock:
                    snapshot_id = self._render_refinement_targets.get(session_id)
                if not snapshot_id:
                    return
                self._refine_snapshot(session_id, snapshot_id)
                with self._render_refinement_lock:
                    latest = self._render_refinement_targets.get(session_id)
                    if latest == snapshot_id:
                        self._render_refinement_targets.pop(session_id, None)
                        return
        finally:
            with self._render_refinement_lock:
                self._render_refinement_futures.pop(session_id, None)

    def _refine_snapshot(self, session_id: str, snapshot_id: str) -> None:
        session_state = self.load_session(session_id)
        current_snapshot_id = str(session_state.get("current_world_snapshot_id") or "")
        if current_snapshot_id != snapshot_id:
            self._log_runtime_stage(
                "render_refinement.skip_stale",
                session_id=session_id,
                snapshot_id=snapshot_id,
                current_snapshot_id=current_snapshot_id,
            )
            return
        snapshot_path = self._snapshot_path(session_id, snapshot_id)
        if not snapshot_path.is_file():
            self._log_runtime_stage(
                "render_refinement.skip_missing_snapshot",
                session_id=session_id,
                snapshot_id=snapshot_id,
            )
            return
        snapshot = _read_json(snapshot_path)
        site_world_id = str(session_state["site_world_id"])
        render_dir = self._session_dir(session_id) / "renders" / snapshot_id
        render_dir.mkdir(parents=True, exist_ok=True)
        base_frame_path = self._base_frame_path(site_world_id)
        robot_profile = dict(session_state.get("robot_profile") or {})
        cameras = self._render_camera_selection(robot_profile)
        workspace_dir = Path(str(self.load_site_world(site_world_id).get("workspace_dir") or "")).resolve()
        self._log_runtime_stage(
            "render_refinement.start",
            session_id=session_id,
            snapshot_id=snapshot_id,
            cameras=[str(camera.get("id") or "") for camera in cameras],
        )
        started = time.monotonic()
        try:
            runner_payload = self.runner.render_snapshot(
                site_world_id=site_world_id,
                session_id=session_id,
                workspace_dir=workspace_dir,
                snapshot_path=snapshot_path,
                output_dir=render_dir,
                cameras=cameras,
                base_frame_path=base_frame_path,
            )
        except Exception as exc:
            self._log_runtime_stage(
                "render_refinement.failed",
                session_id=session_id,
                snapshot_id=snapshot_id,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                error=str(exc),
            )
            latest_debug_artifacts = dict(session_state.get("debug_artifacts") or {})
            latest_debug_artifacts.update(
                {
                    "refinement_status": "failed",
                    "refinement_error": str(exc),
                }
            )
            session_state["debug_artifacts"] = latest_debug_artifacts
            session_state["quality_flags"] = {
                **dict(session_state.get("quality_flags") or {}),
                "refinement_status": "failed",
            }
            self._persist_session_state(session_state)
            return
        observation = self._materialize_render_snapshot(
            session_state=session_state,
            snapshot=snapshot,
            runner_payload={
                **dict(runner_payload),
                "quality_flags": {
                    **dict(runner_payload.get("quality_flags") or {}),
                    "render_mode": "refined",
                    "refinement_status": "complete",
                },
                "debug_artifacts": {
                    **dict(runner_payload.get("debug_artifacts") or {}),
                    "render_mode": "refined",
                    "refinement_status": "complete",
                },
            },
            cameras=cameras,
            render_dir=render_dir,
            persist_session_state=False,
        )
        reloaded = self.load_session(session_id)
        if str(reloaded.get("current_world_snapshot_id") or "") != snapshot_id:
            self._log_runtime_stage(
                "render_refinement.discard_stale",
                session_id=session_id,
                snapshot_id=snapshot_id,
                current_snapshot_id=reloaded.get("current_world_snapshot_id"),
            )
            return
        session_state["latest_render_paths"] = {
            str(key): str(value)
            for key, value in dict(_read_json(render_dir / "render_manifest.json").get("camera_frame_paths") or {}).items()
        }
        session_state["render_manifest_path"] = str(render_dir / "render_manifest.json")
        session_state["quality_flags"] = dict(observation.get("runtimeMetadata", {}).get("quality_flags") or {})
        session_state["protected_region_violations"] = list(
            observation.get("runtimeMetadata", {}).get("protected_region_violations") or []
        )
        session_state["debug_artifacts"] = dict(observation.get("runtimeMetadata", {}).get("debug_artifacts") or {})
        self._persist_session_state(session_state)
        self._log_runtime_stage(
            "render_refinement.done",
            session_id=session_id,
            snapshot_id=snapshot_id,
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )

    def _action_trace_payload(self, session_state: Dict[str, Any]) -> list[Dict[str, Any]]:
        action_space = dict((session_state.get("robot_profile") or {}).get("action_space") or {})
        labels = [str(label).strip() or f"action_{index}" for index, label in enumerate(action_space.get("labels") or [])]
        payload: list[Dict[str, Any]] = []
        for index, action in enumerate(list(session_state.get("action_trace", []))):
            if not isinstance(action, Sequence) or isinstance(action, (str, bytes, bytearray)):
                continue
            row: Dict[str, Any] = {"index": index}
            for action_index, value in enumerate(action):
                label = labels[action_index] if action_index < len(labels) else f"action_{action_index}"
                row[label] = float(value)
            payload.append(row)
        return payload

    def _render_snapshot(self, session_state: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(session_state["session_id"])
        site_world_id = str(session_state["site_world_id"])
        snapshot_id = str(snapshot["snapshot_id"])
        return self._run_snapshot_render_once(
            session_id=session_id,
            snapshot_id=snapshot_id,
            render_fn=lambda: self._render_snapshot_impl(
                session_state=session_state,
                snapshot=snapshot,
                site_world_id=site_world_id,
                snapshot_id=snapshot_id,
            ),
        )

    def _render_snapshot_impl(
        self,
        *,
        session_state: Dict[str, Any],
        snapshot: Dict[str, Any],
        site_world_id: str,
        snapshot_id: str,
    ) -> Dict[str, Any]:
        session_id = str(session_state["session_id"])
        render_dir = self._session_dir(session_id) / "renders" / snapshot_id
        render_dir.mkdir(parents=True, exist_ok=True)
        base_frame_path = self._base_frame_path(site_world_id)
        robot_profile = dict(session_state.get("robot_profile") or {})
        cameras = self._render_camera_selection(robot_profile)
        workspace_dir = Path(str(self.load_site_world(site_world_id).get("workspace_dir") or "")).resolve()
        snapshot_path = Path(str(session_state["current_world_snapshot_path"] or self._save_snapshot(session_id, snapshot)))
        self._log_runtime_stage(
            "render_snapshot.start",
            session_id=session_id,
            snapshot_id=snapshot_id,
            site_world_id=site_world_id,
            cameras=[str(camera.get("id") or "") for camera in cameras],
        )
        runner_started = time.monotonic()
        try:
            self._log_runtime_stage(
                "render_snapshot.runner.start",
                session_id=session_id,
                snapshot_id=snapshot_id,
                workspace_dir=str(workspace_dir),
                snapshot_path=str(snapshot_path),
            )
            runner_payload = self.runner.render_snapshot(
                site_world_id=site_world_id,
                session_id=session_id,
                workspace_dir=workspace_dir,
                snapshot_path=snapshot_path,
                output_dir=render_dir,
                cameras=cameras,
                base_frame_path=base_frame_path,
            )
            self._log_runtime_stage(
                "render_snapshot.runner.done",
                session_id=session_id,
                snapshot_id=snapshot_id,
                elapsed_ms=round((time.monotonic() - runner_started) * 1000),
                frame_count=len(runner_payload.get("camera_frames") or []),
            )
        except NeoVerseRunnerTimeoutError as exc:
            self._log_runtime_stage(
                "render_snapshot.runner.timeout",
                session_id=session_id,
                snapshot_id=snapshot_id,
                elapsed_ms=round((time.monotonic() - runner_started) * 1000),
                error=str(exc),
            )
            runner_payload = self._fallback_runner_payload(
                render_dir=render_dir,
                cameras=cameras,
                base_frame_path=base_frame_path,
                error=exc,
            )
        return self._materialize_render_snapshot(
            session_state=session_state,
            snapshot=snapshot,
            runner_payload=runner_payload,
            cameras=cameras,
            render_dir=render_dir,
            persist_session_state=True,
        )

    def _episode_payload(self, session_state: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "episodeId": str(session_state["session_id"]),
            "taskId": str((session_state.get("task") or {}).get("id") or ""),
            "task": str((session_state.get("task") or {}).get("task_text") or (session_state.get("task") or {}).get("name") or ""),
            "scenarioId": str((session_state.get("scenario") or {}).get("id") or ""),
            "scenario": str((session_state.get("scenario") or {}).get("name") or ""),
            "startStateId": str((session_state.get("start_state") or {}).get("id") or ""),
            "startState": str((session_state.get("start_state") or {}).get("name") or ""),
            "status": str(session_state.get("status") or "ready"),
            "stepIndex": int(session_state.get("step_index", 0)),
            "done": bool(session_state.get("done", False)),
            "reward": float(session_state.get("reward", 0.0) or 0.0),
            "success": session_state.get("success"),
            "failureReason": session_state.get("failure_reason"),
            "observation": observation,
            "observationCameras": observation.get("cameraFrames", []),
            "actionTrace": self._action_trace_payload(session_state),
            "artifactUris": {},
            "runtimeKind": "neoverse_production",
            "productionGrade": True,
            "canonicalPackageVersion": session_state.get("canonical_package_version"),
            "presentationConfig": dict(session_state.get("presentation_config") or {}),
            "qualityFlags": dict(session_state.get("quality_flags") or {}),
            "protectedRegionViolations": list(session_state.get("protected_region_violations") or []),
            "debugArtifacts": dict(session_state.get("debug_artifacts") or {}),
            "engineIdentity": dict(session_state.get("runtime_engine_identity") or {}),
            "modelIdentity": dict(session_state.get("runtime_model_identity") or {}),
            "checkpointIdentity": dict(session_state.get("runtime_checkpoint_identity") or {}),
        }

    def reset_session(
        self,
        session_id: str,
        *,
        task_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        start_state_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._log_runtime_stage(
            "reset_session.start",
            session_id=session_id,
            task_id=task_id,
            scenario_id=scenario_id,
            start_state_id=start_state_id,
        )
        session_state = self.load_session(session_id)
        bundle = self._load_site_world_bundle(str(session_state.get("site_world_id") or ""))
        site_world = bundle["resolved"]
        if task_id:
            session_state["task"] = self._catalog_entry(site_world.get("task_catalog", []), task_id, label="task")
        if scenario_id:
            session_state["scenario"] = self._catalog_entry(site_world.get("scenario_catalog", []), scenario_id, label="scenario")
        if start_state_id:
            session_state["start_state"] = self._catalog_entry(site_world.get("start_state_catalog", []), start_state_id, label="start state")
        session_state["step_index"] = 0
        session_state["done"] = False
        session_state["success"] = None
        session_state["failure_reason"] = None
        session_state["reward"] = 0.0
        session_state["action_trace"] = []
        session_state["status"] = "running"
        snapshot = self._snapshot_from_state(session_state, self._initial_world_state(session_state))
        snapshot_path = self._save_snapshot(session_id, snapshot)
        session_state["current_world_snapshot_id"] = snapshot["snapshot_id"]
        session_state["current_world_snapshot_path"] = str(snapshot_path)
        session_state["latest_render_paths"] = {}
        session_state["render_manifest_path"] = None
        self._clear_render_failure(session_state)
        self._persist_session_state(session_state)
        self._log_runtime_stage(
            "reset_session.persisted_initial_state",
            session_id=session_id,
            snapshot_id=snapshot.get("snapshot_id"),
            snapshot_path=str(snapshot_path),
        )
        if self.enable_immediate_preview:
            try:
                render_dir = self._session_dir(session_id) / "renders" / str(snapshot["snapshot_id"])
                render_dir.mkdir(parents=True, exist_ok=True)
                cameras = self._render_camera_selection(dict(session_state.get("robot_profile") or {}))
                observation = self._materialize_render_snapshot(
                    session_state=session_state,
                    snapshot=snapshot,
                    runner_payload=self._preview_runner_payload(
                        session_state=session_state,
                        snapshot=snapshot,
                        render_dir=render_dir,
                        cameras=cameras,
                    ),
                    cameras=cameras,
                    render_dir=render_dir,
                    persist_session_state=True,
                )
                self._clear_render_failure(session_state)
                self._persist_session_state(session_state)
                self._schedule_render_refinement(session_id, str(snapshot["snapshot_id"]))
                self._log_runtime_stage(
                    "reset_session.preview_ready",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    primary_camera_id=observation.get("primaryCameraId"),
                )
                return {
                    "session_id": session_id,
                    "episode": self._episode_payload(session_state, observation),
                }
            except Exception as exc:
                self._record_render_failure(
                    session_state,
                    code="pose_preview_failed",
                    message=str(exc),
                )
                raise
        try:
            self._log_runtime_stage(
                "reset_session.render_snapshot.start",
                session_id=session_id,
                snapshot_id=snapshot.get("snapshot_id"),
            )
            observation = self._render_snapshot(session_state, snapshot)
            self._log_runtime_stage(
                "reset_session.render_snapshot.done",
                session_id=session_id,
                snapshot_id=snapshot.get("snapshot_id"),
            )
        except Exception as exc:
            self._log_runtime_stage(
                "reset_session.render_snapshot.error",
                session_id=session_id,
                snapshot_id=snapshot.get("snapshot_id"),
                error=str(exc),
            )
            if self._recover_render_state(session_state):
                self._clear_render_failure(session_state)
                self._persist_session_state(session_state)
                self._log_runtime_stage(
                    "reset_session.recovered_render_state",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                )
                observation = self._observation_from_current_state(session_state)
                return {
                    "session_id": session_id,
                    "episode": self._episode_payload(session_state, observation),
                }
            self._record_render_failure(
                session_state,
                code="render_snapshot_failed",
                message=str(exc),
            )
            raise
        self._clear_render_failure(session_state)
        self._persist_session_state(session_state)
        self._log_runtime_stage(
            "reset_session.done",
            session_id=session_id,
            snapshot_id=snapshot.get("snapshot_id"),
            primary_camera_id=observation.get("primaryCameraId"),
        )
        return {
            "session_id": session_id,
            "episode": self._episode_payload(session_state, observation),
        }

    def _advance_world_state(self, session_state: Dict[str, Any], action: Sequence[float]) -> Dict[str, Any]:
        snapshot_path = Path(str(session_state.get("current_world_snapshot_path") or "")).resolve()
        current = _read_json(snapshot_path) if snapshot_path.is_file() else self._snapshot_from_state(session_state, self._initial_world_state(session_state))
        world_state = dict(current.get("world_state") or self._initial_world_state(session_state))
        expected_dim = int(((session_state.get("robot_profile") or {}).get("action_space") or {}).get("dim") or len(action) or 7)
        action_list = [float(value) for value in action]
        if len(action_list) != expected_dim:
            raise RuntimeError(f"Action-space mismatch: received {len(action_list)} values, expected {expected_dim}.")
        if any(abs(value) > 1.0 for value in action_list):
            raise RuntimeError("Action values must be in [-1.0, 1.0] for the production runtime.")
        joint_targets = list((world_state.get("robot_state") or {}).get("joint_targets") or [0.0] * expected_dim)
        if len(joint_targets) < expected_dim:
            joint_targets.extend([0.0] * (expected_dim - len(joint_targets)))
        joint_targets = [round(joint_targets[index] + action_list[index], 4) for index in range(expected_dim)]
        base_pose = list((world_state.get("robot_state") or {}).get("base_pose") or [0.0, 0.0, 0.0])
        while len(base_pose) < 3:
            base_pose.append(0.0)
        base_pose[0] = round(base_pose[0] + action_list[0] * 0.1, 4)
        base_pose[1] = round(base_pose[1] + action_list[1] * 0.1, 4)
        base_pose[2] = round(base_pose[2] + action_list[2] * 0.05, 4)
        next_step = int(world_state.get("step_index", 0)) + 1
        progress_delta = max(0.05, min(0.25, sum(abs(value) for value in action_list) / float(max(expected_dim, 1) * 2.0)))
        task_progress = round(min(float(world_state.get("task_progress", 0.0) or 0.0) + progress_delta, 1.0), 4)
        done = task_progress >= 1.0 or next_step >= 12
        status = "completed" if done else "running"
        return {
            "step_index": next_step,
            "task_progress": task_progress,
            "robot_state": {
                "base_pose": base_pose,
                "joint_targets": joint_targets,
            },
            "task_status": status,
        }

    def step_session(self, session_id: str, *, action: Sequence[float]) -> Dict[str, Any]:
        session_state = self.load_session(session_id)
        if bool(session_state.get("done")):
            raise RuntimeError(f"session {session_id} is already complete")
        world_state = self._advance_world_state(session_state, action)
        snapshot = self._snapshot_from_state(session_state, world_state)
        snapshot_path = self._save_snapshot(session_id, snapshot)
        session_state["current_world_snapshot_id"] = snapshot["snapshot_id"]
        session_state["current_world_snapshot_path"] = str(snapshot_path)
        session_state["latest_render_paths"] = {}
        session_state["render_manifest_path"] = None
        session_state["step_index"] = int(world_state.get("step_index", 0))
        session_state["action_trace"] = list(session_state.get("action_trace", [])) + [list(action)]
        session_state["reward"] = float(world_state.get("task_progress", 0.0) or 0.0)
        session_state["done"] = bool(world_state.get("task_status") == "completed")
        session_state["success"] = True if session_state["done"] and session_state["reward"] >= 1.0 else None
        session_state["failure_reason"] = None if session_state["success"] is not False else "task_incomplete"
        session_state["status"] = "completed" if session_state["done"] else "running"
        self._clear_render_failure(session_state)
        self._persist_session_state(session_state)
        if self.enable_immediate_preview:
            try:
                render_dir = self._session_dir(session_id) / "renders" / str(snapshot["snapshot_id"])
                render_dir.mkdir(parents=True, exist_ok=True)
                cameras = self._render_camera_selection(dict(session_state.get("robot_profile") or {}))
                observation = self._materialize_render_snapshot(
                    session_state=session_state,
                    snapshot=snapshot,
                    runner_payload=self._preview_runner_payload(
                        session_state=session_state,
                        snapshot=snapshot,
                        render_dir=render_dir,
                        cameras=cameras,
                    ),
                    cameras=cameras,
                    render_dir=render_dir,
                    persist_session_state=True,
                )
                self._clear_render_failure(session_state)
                self._persist_session_state(session_state)
                self._schedule_render_refinement(session_id, str(snapshot["snapshot_id"]))
                self._log_runtime_stage(
                    "step_session.preview_ready",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    step_index=session_state.get("step_index"),
                    primary_camera_id=observation.get("primaryCameraId"),
                )
                return {
                    "session_id": session_id,
                    "episode": self._episode_payload(session_state, observation),
                }
            except Exception as exc:
                self._record_render_failure(
                    session_state,
                    code="pose_preview_failed",
                    message=str(exc),
                )
                raise
        try:
            observation = self._render_snapshot(session_state, snapshot)
        except Exception as exc:
            if self._recover_render_state(session_state):
                self._clear_render_failure(session_state)
                self._persist_session_state(session_state)
                observation = self._observation_from_current_state(session_state)
                return {
                    "session_id": session_id,
                    "episode": self._episode_payload(session_state, observation),
                }
            self._record_render_failure(
                session_state,
                code="render_snapshot_failed",
                message=str(exc),
            )
            raise
        self._clear_render_failure(session_state)
        self._persist_session_state(session_state)
        return {
            "session_id": session_id,
            "episode": self._episode_payload(session_state, observation),
        }

    def session_state(self, session_id: str) -> Dict[str, Any]:
        session_state = self.load_session(session_id)
        explorer_state = self._explorer_state(session_state)
        snapshot_path = Path(str(session_state.get("current_world_snapshot_path") or "")).resolve()
        if not snapshot_path.is_file():
            snapshot = self._snapshot_from_state(session_state, self._initial_world_state(session_state))
            snapshot_path = self._save_snapshot(session_id, snapshot)
            session_state["current_world_snapshot_id"] = snapshot["snapshot_id"]
            session_state["current_world_snapshot_path"] = str(snapshot_path)
            self._persist_session_state(session_state)
        snapshot = _read_json(snapshot_path)
        if not dict(session_state.get("latest_render_paths") or {}):
            if self._recover_render_state(session_state):
                self._persist_session_state(session_state)
        latest_render_paths = {
            str(key): str(value)
            for key, value in dict(session_state.get("latest_render_paths") or {}).items()
            if Path(str(value)).is_file()
        }
        robot_profile = dict(session_state.get("robot_profile") or {})
        cameras = self._render_camera_selection(robot_profile)
        primary_camera_id = next(iter(latest_render_paths.keys()), "")
        if not primary_camera_id and cameras:
            primary_camera_id = str(cameras[0].get("id") or "head_rgb").strip() or "head_rgb"
        observation = {
            "frame_path": (
                f"{self.base_url}/v1/sessions/{session_id}/render?camera_id={primary_camera_id}"
                if primary_camera_id and latest_render_paths
                else None
            ),
            "primaryCameraId": primary_camera_id,
            "cameraFrames": [
                {
                    "cameraId": str(camera.get("id") or ""),
                    "role": str(camera.get("role") or ""),
                    "required": bool(camera.get("required", False)),
                    "available": str(camera.get("id") or "") in latest_render_paths,
                    "framePath": (
                        f"{self.base_url}/v1/sessions/{session_id}/render?camera_id={str(camera.get('id') or '')}"
                        if str(camera.get("id") or "") in latest_render_paths
                        else None
                    ),
                }
                for camera in cameras
            ],
            "runtimeMetadata": {
                "site_world_id": session_state.get("site_world_id"),
                "build_id": session_state.get("build_id"),
                "step_index": session_state.get("step_index"),
                "runtime_kind": "neoverse_production",
                "production_grade": True,
                "canonical_package_version": session_state.get("canonical_package_version"),
                "quality_flags": session_state.get("quality_flags"),
                "protected_region_violations": session_state.get("protected_region_violations"),
                "debug_artifacts": session_state.get("debug_artifacts"),
                "latest_render_error_code": session_state.get("latest_render_error_code"),
                "latest_render_error_message": session_state.get("latest_render_error_message"),
            },
            "worldSnapshot": {
                "snapshotId": session_state.get("current_world_snapshot_id"),
                "world_state": snapshot.get("world_state"),
                "status": session_state.get("status"),
            },
        }
        return {
            "session_id": session_id,
            "site_world_id": session_state.get("site_world_id"),
            "build_id": session_state.get("build_id"),
            "runtime_kind": "neoverse_production",
            "production_grade": True,
            "status": session_state.get("status"),
            "step_index": session_state.get("step_index"),
            "done": session_state.get("done"),
            "reward": session_state.get("reward"),
            "observation": observation,
            "action_trace": self._action_trace_payload(session_state),
            "canonical_package_version": session_state.get("canonical_package_version"),
            "presentation_config": dict(session_state.get("presentation_config") or {}),
            "quality_flags": dict(session_state.get("quality_flags") or {}),
            "protected_region_violations": list(session_state.get("protected_region_violations") or []),
            "debug_artifacts": dict(session_state.get("debug_artifacts") or {}),
            "runtime_engine_identity": dict(session_state.get("runtime_engine_identity") or {}),
            "runtime_model_identity": dict(session_state.get("runtime_model_identity") or {}),
            "runtime_checkpoint_identity": dict(session_state.get("runtime_checkpoint_identity") or {}),
            "explorer_state": {
                **explorer_state,
                "latest_frame_paths": {
                    str(key): str(value)
                    for key, value in dict(explorer_state.get("latest_frame_paths") or {}).items()
                    if Path(str(value)).is_file()
                },
            },
        }

    def render_bytes(self, session_id: str, camera_id: str) -> bytes:
        session_state = self.load_session(session_id)
        if not dict(session_state.get("latest_render_paths") or {}):
            if self._recover_render_state(session_state):
                self._clear_render_failure(session_state)
                self._persist_session_state(session_state)
        latest = dict(session_state.get("latest_render_paths") or {})
        render_path = Path(str(latest.get(camera_id) or "")).resolve()
        if not render_path.is_file():
            snapshot_path = Path(str(session_state.get("current_world_snapshot_path") or "")).resolve()
            if not snapshot_path.is_file():
                self._record_render_failure(
                    session_state,
                    code="render_bytes_failed",
                    message=f"missing world snapshot for {session_id}",
                )
                raise RuntimeError(f"missing world snapshot for {session_id}")
            snapshot = _read_json(snapshot_path)
            inflight_render = self._get_inflight_snapshot_render(session_id, str(snapshot.get("snapshot_id") or ""))
            if inflight_render is not None:
                self._log_runtime_stage(
                    "render_bytes.await_inflight",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    camera_id=camera_id,
                )
                inflight_render.result()
                session_state = self.load_session(session_id)
                latest = dict(session_state.get("latest_render_paths") or {})
                render_path = Path(str(latest.get(camera_id) or "")).resolve()
                if render_path.is_file():
                    payload = render_path.read_bytes()
                    if payload:
                        return payload
            try:
                session_id = str(session_state["session_id"])
                site_world_id = str(session_state["site_world_id"])
                session_dir = self._session_dir(session_id)
                render_dir = session_dir / "renders" / str(snapshot["snapshot_id"])
                render_dir.mkdir(parents=True, exist_ok=True)
                base_frame_path = self._base_frame_path(site_world_id)
                robot_profile = dict(session_state.get("robot_profile") or {})
                cameras = self._render_camera_selection(robot_profile, requested_camera_ids=[camera_id])
                if self.enable_immediate_preview:
                    try:
                        observation = self._materialize_render_snapshot(
                            session_state=session_state,
                            snapshot=snapshot,
                            runner_payload=self._preview_runner_payload(
                                session_state=session_state,
                                snapshot=snapshot,
                                render_dir=render_dir,
                                cameras=cameras,
                            ),
                            cameras=cameras,
                            render_dir=render_dir,
                            persist_session_state=True,
                        )
                        self._clear_render_failure(session_state)
                        self._persist_session_state(session_state)
                        self._schedule_render_refinement(session_id, str(snapshot["snapshot_id"]))
                        render_path = Path(
                            str(
                                dict(session_state.get("latest_render_paths") or {}).get(camera_id)
                                or observation.get("frame_path")
                                or ""
                            )
                        ).resolve()
                        if render_path.is_file():
                            payload = render_path.read_bytes()
                            if payload:
                                return payload
                    except Exception as exc:
                        self._record_render_failure(
                            session_state,
                            code="pose_preview_failed",
                            message=str(exc),
                        )
                        raise
                workspace_dir = Path(str(self.load_site_world(site_world_id).get("workspace_dir") or "")).resolve()
                render_snapshot_path = Path(str(session_state["current_world_snapshot_path"] or self._save_snapshot(session_id, snapshot)))
                self._log_runtime_stage(
                    "render_bytes.runner.start",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    camera_id=camera_id,
                )
                runner_started = time.monotonic()
                try:
                    runner_payload = self.runner.render_snapshot(
                        site_world_id=site_world_id,
                        session_id=session_id,
                        workspace_dir=workspace_dir,
                        snapshot_path=render_snapshot_path,
                        output_dir=render_dir,
                        cameras=cameras,
                        base_frame_path=base_frame_path,
                    )
                    self._log_runtime_stage(
                        "render_bytes.runner.done",
                        session_id=session_id,
                        snapshot_id=snapshot.get("snapshot_id"),
                        camera_id=camera_id,
                        elapsed_ms=round((time.monotonic() - runner_started) * 1000),
                    )
                except NeoVerseRunnerTimeoutError as exc:
                    self._log_runtime_stage(
                        "render_bytes.runner.timeout",
                        session_id=session_id,
                        snapshot_id=snapshot.get("snapshot_id"),
                        camera_id=camera_id,
                        elapsed_ms=round((time.monotonic() - runner_started) * 1000),
                        error=str(exc),
                    )
                    runner_payload = self._fallback_runner_payload(
                        render_dir=render_dir,
                        cameras=cameras,
                        base_frame_path=base_frame_path,
                        error=exc,
                    )
                runtime_layer_bundle = self._runtime_layer_bundle_for_site_world(site_world_id)
                frame_records = [dict(item) for item in runner_payload.get("camera_frames", []) if isinstance(item, Mapping)]
                frame_record = next((item for item in frame_records if str(item.get("cameraId") or "") == camera_id), None)
                if frame_record is None:
                    raise RuntimeError(f"NeoVerse runner frame missing for camera {camera_id}")
                raw_path = Path(str(frame_record.get("path") or "")).resolve()
                canonical_frame = self._canonical_frame_from_runner_output(raw_path, camera_id)
                self._log_runtime_stage(
                    "render_bytes.composite.start",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    camera_id=camera_id,
                )
                composite_started = time.monotonic()
                composite = self._composite_camera_frame(
                    canonical_frame=canonical_frame,
                    runtime_layer_bundle=runtime_layer_bundle,
                    session_state=session_state,
                    session_dir=session_dir,
                    snapshot=snapshot,
                    camera_id=camera_id,
                )
                self._log_runtime_stage(
                    "render_bytes.composite.done",
                    session_id=session_id,
                    snapshot_id=snapshot.get("snapshot_id"),
                    camera_id=camera_id,
                    elapsed_ms=round((time.monotonic() - composite_started) * 1000),
                )
                output_path = render_dir / f"{camera_id}.png"
                _save_frame(output_path, composite["frame"])
                latest[camera_id] = str(output_path)
                session_state["latest_render_paths"] = latest
                session_state["quality_flags"] = dict(composite.get("quality_flags") or session_state.get("quality_flags") or {})
                session_state["protected_region_violations"] = list(composite.get("protected_region_violations") or [])
                session_state["debug_artifacts"] = dict(runner_payload.get("debug_artifacts") or {})
                session_state["debug_artifacts"].update(dict(composite.get("debug_artifacts") or {}))
                self._clear_render_failure(session_state)
                self._persist_session_state(session_state)
                render_path = output_path
            except Exception as exc:
                self._record_render_failure(
                    session_state,
                    code="render_bytes_failed",
                    message=str(exc),
                )
                raise
        payload = render_path.read_bytes()
        if not payload:
            raise RuntimeError(f"missing render bytes for {session_id}:{camera_id}")
        return payload

    def explorer_render(
        self,
        session_id: str,
        *,
        camera_id: str,
        pose: Mapping[str, Any],
        viewport_width: int | None,
        viewport_height: int | None,
        refine_mode: str | None,
    ) -> Dict[str, Any]:
        session_state = self.load_session(session_id)
        explorer_state = self._explorer_state(session_state)
        site_world_id = str(session_state["site_world_id"])
        spec = self._load_site_world_bundle(site_world_id)["resolved"]
        snapshot_path = Path(str(session_state.get("current_world_snapshot_path") or "")).resolve()
        if not snapshot_path.is_file():
            snapshot = self._snapshot_from_state(session_state, self._initial_world_state(session_state))
            snapshot_path = self._save_snapshot(session_id, snapshot)
            session_state["current_world_snapshot_id"] = snapshot["snapshot_id"]
            session_state["current_world_snapshot_path"] = str(snapshot_path)
            self._persist_session_state(session_state)
        snapshot = _read_json(snapshot_path)
        world_state = dict(snapshot.get("world_state") or self._initial_world_state(session_state))
        normalized_pose = self._normalized_explorer_pose(pose or explorer_state.get("pose"))
        requested_width = int(viewport_width or 0) or None
        requested_height = int(viewport_height or 0) or None
        snapshot_id = str(snapshot.get("snapshot_id") or session_state.get("current_world_snapshot_id") or "snapshot")
        render_dir = self._session_dir(session_id) / "explorer_renders" / snapshot_id / _stable_id(
            "explorer-view",
            camera_id,
            json.dumps(normalized_pose, sort_keys=True),
            str(requested_width or ""),
            str(requested_height or ""),
            str(refine_mode or ""),
        )
        render_dir.mkdir(parents=True, exist_ok=True)
        output_path = render_dir / f"{camera_id}.png"
        debug_artifacts: Dict[str, Any] = {}
        grounded_source = "runtime_fallback"
        refine_status = "idle"

        try:
            grounded_frame, preview_debug, coverage_mask = self._pose_preview_renderer.render_camera(
                site_world_id=site_world_id,
                spec=spec,
                world_state=world_state,
                camera_id=camera_id,
                explorer_pose=normalized_pose,
                requested_width=requested_width,
                requested_height=requested_height,
            )
            grounded_source = "arkit_rgbd"
            debug_artifacts.update({"grounded_preview": preview_debug})
            quality_flags: Dict[str, Any] = {
                "presentation_quality": "preview",
                "grounded_source": grounded_source,
                "preview_mode": "pose_driven",
                "preview_source": "arkit_rgbd",
                "preview_size": list(preview_debug.get("preview_size") or []),
                "coverage_ratio": preview_debug.get("coverage_ratio"),
            }
        except Exception as exc:
            base_frame = _load_frame(self._base_frame_path(site_world_id))
            grounded_frame = _coerce_camera_frame(base_frame, camera_id)
            coverage_mask = np.ones(grounded_frame.shape[:2], dtype=bool)
            debug_artifacts.update({"grounded_preview_error": str(exc)})
            quality_flags = {
                "presentation_quality": "degraded",
                "grounded_source": grounded_source,
                "preview_mode": "base_frame",
                "preview_source": "runtime_fallback",
            }

        final_frame = grounded_frame
        if str(refine_mode or "").strip().lower() == "request" and grounded_source == "arkit_rgbd":
            refine_status = "running"
            try:
                grounded_input_path = render_dir / f"{camera_id}-grounded-input.png"
                _save_frame(grounded_input_path, grounded_frame)
                refinement_snapshot_path = render_dir / "explorer_snapshot.json"
                refinement_snapshot = {
                    **snapshot,
                    "presentation_config": {
                        **dict(session_state.get("presentation_config") or {}),
                        "trajectory": {"trajectory": "static"},
                    },
                }
                _write_json(refinement_snapshot_path, refinement_snapshot)
                robot_profile = dict(session_state.get("robot_profile") or {})
                camera_summary = next(
                    (
                        dict(camera)
                        for camera in self._render_camera_selection(robot_profile, requested_camera_ids=[camera_id])
                        if str(camera.get("id") or "") == camera_id
                    ),
                    {"id": camera_id, "role": "head", "required": True},
                )
                generated_dir = render_dir / "generated"
                generated_dir.mkdir(parents=True, exist_ok=True)
                runner_payload = self.runner.render_snapshot(
                    site_world_id=site_world_id,
                    session_id=f"{session_id}-explorer",
                    workspace_dir=render_dir / "explorer_workspace",
                    snapshot_path=refinement_snapshot_path,
                    output_dir=generated_dir,
                    cameras=[camera_summary],
                    base_frame_path=grounded_input_path,
                )
                frame_records = [
                    dict(item)
                    for item in runner_payload.get("camera_frames", [])
                    if isinstance(item, Mapping)
                ]
                frame_record = next(
                    (item for item in frame_records if str(item.get("cameraId") or "") == camera_id),
                    frame_records[0] if frame_records else None,
                )
                if frame_record is None:
                    raise RuntimeError(f"NeoVerse explorer refinement did not return a frame for {camera_id}")
                generated_frame = self._canonical_frame_from_runner_output(
                    Path(str(frame_record.get("path") or "")).resolve(),
                    camera_id,
                )
                runtime_layer_bundle = self._runtime_layer_bundle_for_site_world(site_world_id)
                refinement = composite_masked_refinement(
                    grounded_frame=grounded_frame,
                    generated_frame=generated_frame,
                    coverage_mask=coverage_mask,
                    protected_regions_manifest=runtime_layer_bundle["protected_regions_manifest"],
                    canonical_render_policy=runtime_layer_bundle["canonical_render_policy"],
                    presentation_config=dict(session_state.get("presentation_config") or {}),
                    presentation_variance_policy=runtime_layer_bundle["presentation_variance_policy"],
                    session_dir=self._session_dir(session_id),
                    step_index=int(world_state.get("step_index", 0) or 0),
                    camera_id=camera_id,
                )
                final_frame = refinement["frame"]
                quality_flags = {
                    **quality_flags,
                    **dict(refinement.get("quality_flags") or {}),
                }
                debug_artifacts.update(dict(runner_payload.get("debug_artifacts") or {}))
                debug_artifacts.update(dict(refinement.get("debug_artifacts") or {}))
                if refinement.get("protected_region_violations"):
                    quality_flags["protected_region_violations"] = list(refinement.get("protected_region_violations") or [])
                refine_status = "complete"
            except Exception as exc:
                debug_artifacts.update({"refinement_error": str(exc)})
                quality_flags = {
                    **quality_flags,
                    "refinement_applied": False,
                }
                refine_status = "failed"

        _save_frame(output_path, final_frame)
        explorer_state.update(
            {
                "pose": normalized_pose,
                "camera_id": camera_id,
                "latest_frame_paths": {camera_id: str(output_path)},
                "grounded_source": grounded_source,
                "refine_status": refine_status,
                "quality_flags": quality_flags,
                "debug_artifacts": debug_artifacts,
                "viewport": {
                    "requested_width": requested_width,
                    "requested_height": requested_height,
                    "output_width": int(final_frame.shape[1]),
                    "output_height": int(final_frame.shape[0]),
                },
                "snapshot_id": snapshot_id,
            }
        )
        session_state["explorer_state"] = explorer_state
        self._persist_session_state(session_state)
        return {
            "session_id": session_id,
            "camera_id": camera_id,
            "frame_path": f"{self.base_url}/v1/sessions/{session_id}/explorer-frame?camera_id={camera_id}",
            "pose": normalized_pose,
            "viewport": dict(explorer_state.get("viewport") or {}),
            "grounded_source": grounded_source,
            "refine_status": refine_status,
            "quality_flags": quality_flags,
            "debug_artifacts": debug_artifacts,
            "snapshot_id": snapshot_id,
        }

    def explorer_frame_bytes(self, session_id: str, camera_id: str) -> bytes:
        session_state = self.load_session(session_id)
        explorer_state = self._explorer_state(session_state)
        latest = dict(explorer_state.get("latest_frame_paths") or {})
        frame_path = Path(str(latest.get(camera_id) or "")).resolve()
        if not frame_path.is_file():
            payload = self.explorer_render(
                session_id,
                camera_id=camera_id,
                pose=dict(explorer_state.get("pose") or self._default_explorer_pose()),
                viewport_width=int((explorer_state.get("viewport") or {}).get("requested_width") or 0) or None,
                viewport_height=int((explorer_state.get("viewport") or {}).get("requested_height") or 0) or None,
                refine_mode=None,
            )
            frame_path = Path(str((self._explorer_state(self.load_session(session_id)).get("latest_frame_paths") or {}).get(camera_id) or "")).resolve()
            if not frame_path.is_file():
                raise RuntimeError(f"missing explorer render bytes for {session_id}:{camera_id}: {payload}")
        payload = frame_path.read_bytes()
        if not payload:
            raise RuntimeError(f"missing explorer render bytes for {session_id}:{camera_id}")
        return payload
