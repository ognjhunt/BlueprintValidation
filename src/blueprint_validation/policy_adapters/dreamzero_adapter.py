"""DreamZero policy adapter implementation (action-only integration mode)."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..common import get_logger
from ..config import PolicyAdapterConfig, PolicyEvalConfig, PolicyFinetuneConfig
from .base import PolicyAdapter, PolicyHandle, PolicyTrainingResult

logger = get_logger("policy_adapters.dreamzero")


class DreamZeroPolicyAdapter(PolicyAdapter):
    """DreamZero adapter using action-only calls for external world-model evaluation.

    This mode intentionally keeps DreamDojo as the rollout world model while using
    DreamZero as the policy-side action generator.
    """

    def __init__(self, adapter_config: PolicyAdapterConfig):
        super().__init__(adapter_config)
        self.backend = adapter_config.dreamzero

    @property
    def name(self) -> str:
        return "dreamzero"

    def base_model_ref(self, eval_config: PolicyEvalConfig) -> tuple[str, Optional[Path]]:
        del eval_config
        model_name = str(
            self.backend.base_model_name or self.backend.checkpoint_path or ""
        ).strip()
        return model_name, self.backend.checkpoint_path

    def load_policy(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
        device: str,
    ) -> PolicyHandle:
        model_ref = checkpoint_path or self.backend.checkpoint_path
        if (not model_ref or not model_ref.exists()) and model_name:
            model_ref = Path(model_name)

        runtime = self._instantiate_runtime(model_ref=model_ref, device=device)
        return PolicyHandle(
            model=runtime,
            processor=None,
            metadata={
                "model_name": model_name,
                "checkpoint_path": str(model_ref or ""),
                "frame_history": [],
                "frame_history_size": int(self.backend.frame_history),
            },
        )

    def predict_action(
        self,
        handle: PolicyHandle,
        frame,
        task_prompt: str,
        unnorm_key: Optional[str],
        device: str,
    ):
        del unnorm_key, device
        history = handle.metadata.setdefault("frame_history", [])
        history.append(frame)
        max_hist = max(1, int(handle.metadata.get("frame_history_size", self.backend.frame_history)))
        if len(history) > max_hist:
            del history[:-max_hist]

        action = self._predict_action_chunk(
            runtime=handle.model,
            history=history,
            task_prompt=task_prompt,
        )
        return self._normalize_action(action)

    def dataset_transform(
        self,
        source_dataset_dir: Path,
        output_root: Path,
        dataset_name: str,
    ) -> Path:
        if not source_dataset_dir.exists():
            raise RuntimeError(f"Source dataset directory does not exist: {source_dataset_dir}")

        target = output_root / dataset_name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        for item in source_dataset_dir.iterdir():
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return target

    def train_policy(
        self,
        base_model_name: str,
        base_checkpoint: Optional[Path],
        dataset_root: Path,
        dataset_name: str,
        output_dir: Path,
        finetune_config: PolicyFinetuneConfig,
    ) -> PolicyTrainingResult:
        if not bool(self.backend.allow_training):
            return PolicyTrainingResult(
                status="skipped",
                adapted_checkpoint_path=base_checkpoint,
                elapsed_seconds=0.0,
                detail=(
                    "DreamZero training is disabled by default "
                    "(policy_adapter.dreamzero.allow_training=false)."
                ),
                raw={"status": "skipped", "reason": "dreamzero_training_disabled"},
            )

        script_path = Path(self.backend.train_script)
        if not script_path.is_absolute():
            script_path = self.backend.repo_path / script_path
        if not script_path.exists():
            return PolicyTrainingResult(
                status="failed",
                adapted_checkpoint_path=None,
                elapsed_seconds=0.0,
                detail=f"DreamZero train script not found: {script_path}",
                raw={"status": "failed", "reason": "train_script_missing"},
            )

        model_ref = str(base_checkpoint) if base_checkpoint else str(base_model_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--base_model",
            model_ref,
            "--dataset_root",
            str(dataset_root),
            "--dataset_name",
            dataset_name,
            "--output_dir",
            str(output_dir),
        ]
        cmd.extend(list(self.backend.extra_train_args or []))
        cmd.extend(list(finetune_config.extra_args or []))

        start = time.monotonic()
        proc = subprocess.run(
            cmd,
            cwd=str(self.backend.repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.monotonic() - start

        if proc.returncode != 0:
            return PolicyTrainingResult(
                status="failed",
                adapted_checkpoint_path=None,
                elapsed_seconds=elapsed,
                detail=(proc.stderr or proc.stdout or "DreamZero training failed")[-1200:],
                raw={
                    "status": "failed",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )

        adapted = self.resolve_latest_checkpoint(output_dir)
        return PolicyTrainingResult(
            status="success" if adapted is not None else "failed",
            adapted_checkpoint_path=adapted,
            elapsed_seconds=elapsed,
            detail="" if adapted is not None else "No DreamZero checkpoint found after training.",
            raw={
                "status": "success" if adapted is not None else "failed",
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )

    def resolve_latest_checkpoint(self, run_root_dir: Path) -> Optional[Path]:
        if not run_root_dir.exists():
            return None
        candidates = []
        for pattern in ("checkpoint-*", "iter_*", "*.pt", "*.ckpt"):
            candidates.extend([p for p in run_root_dir.rglob(pattern) if p.exists()])
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _instantiate_runtime(self, *, model_ref: Path | None, device: str):
        module_name = str(self.backend.inference_module).strip()
        class_name = str(self.backend.inference_class).strip()
        if not module_name or not class_name:
            raise RuntimeError(
                "DreamZero adapter misconfigured: inference_module/inference_class must be set."
            )

        repo_path = self.backend.repo_path
        if repo_path.exists():
            repo_path_str = str(repo_path)
            if repo_path_str not in sys.path:
                sys.path.insert(0, repo_path_str)

        module = importlib.import_module(module_name)
        runtime_cls = getattr(module, class_name, None)
        if runtime_cls is None:
            raise RuntimeError(
                f"DreamZero inference class not found: {module_name}.{class_name}"
            )

        model_id = str(model_ref) if model_ref is not None else ""

        from_pretrained = getattr(runtime_cls, "from_pretrained", None)
        if callable(from_pretrained):
            try:
                return from_pretrained(model_id, device=device)
            except TypeError:
                try:
                    return from_pretrained(model_id)
                except TypeError:
                    pass

        for kwargs in (
            {"model_path": model_id, "device": device},
            {"checkpoint_path": model_id, "device": device},
            {"model_name": model_id, "device": device},
            {"model_path": model_id},
            {},
        ):
            try:
                return runtime_cls(**kwargs)
            except TypeError:
                continue

        raise RuntimeError(
            "Could not instantiate DreamZero inference runtime. "
            f"Tried class {module_name}.{class_name} with common constructor signatures."
        )

    def _predict_action_chunk(self, *, runtime: Any, history: list, task_prompt: str):
        attempts = []
        for method_name in ("predict_action", "infer_action", "infer", "generate"):
            method = getattr(runtime, method_name, None)
            if not callable(method):
                continue
            calls = [
                lambda m=method: m(frames=history, prompt=task_prompt),
                lambda m=method: m(history=history, prompt=task_prompt),
                lambda m=method: m(frames=history, task=task_prompt),
                lambda m=method: m(history, task_prompt),
                lambda m=method: m(history=history, instruction=task_prompt),
                lambda m=method: m(history=history, prompt=task_prompt, num_steps=1),
            ]
            for call in calls:
                try:
                    out = call()
                    return self._extract_action(out)
                except TypeError as exc:
                    attempts.append(f"{method_name}:{exc}")
                    continue

        if callable(runtime):
            for call in (
                lambda: runtime(frames=history, prompt=task_prompt),
                lambda: runtime(history=history, prompt=task_prompt),
                lambda: runtime(history, task_prompt),
            ):
                try:
                    out = call()
                    return self._extract_action(out)
                except TypeError as exc:
                    attempts.append(f"__call__:{exc}")
                    continue

        raise RuntimeError(
            "DreamZero adapter could not produce action from inference runtime. "
            f"Attempt failures: {attempts[:3]}"
        )

    def _extract_action(self, output):
        if isinstance(output, dict):
            if "action" in output:
                return output["action"]
            if "actions" in output:
                actions = output["actions"]
                arr = np.asarray(actions)
                if arr.ndim >= 2:
                    return arr[0]
                return actions
        if isinstance(output, (list, tuple)) and output:
            first = output[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                arr = np.asarray(output)
                if arr.ndim >= 2:
                    return arr[0]
            return first
        return output

    def _normalize_action(self, action) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("DreamZero action contract violation: non-finite action values detected.")
        target_dim = int(self.backend.policy_action_dim)
        if self.backend.strict_action_contract and arr.shape[0] != target_dim:
            raise RuntimeError(
                "DreamZero strict_action_contract violation: "
                f"received action_dim={arr.shape[0]}, expected={target_dim}."
            )
        if self.backend.strict_action_contract:
            lower = float(self.backend.strict_action_min)
            upper = float(self.backend.strict_action_max)
            if np.any(arr < lower) or np.any(arr > upper):
                raise RuntimeError(
                    "DreamZero strict_action_contract violation: "
                    f"action values out of bounds [{lower}, {upper}] "
                    f"(min={float(np.min(arr)):.4f}, max={float(np.max(arr)):.4f})."
                )
        if arr.shape[0] < target_dim:
            arr = np.pad(arr, (0, target_dim - arr.shape[0]))
        elif arr.shape[0] > target_dim:
            arr = arr[:target_dim]
        return arr.astype(np.float32)
