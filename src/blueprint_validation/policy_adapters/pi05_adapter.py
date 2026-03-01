"""pi0.5 policy adapter implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..common import get_logger
from ..config import PolicyAdapterConfig, PolicyEvalConfig, PolicyFinetuneConfig
from ..training.pi05_finetune import resolve_latest_pi05_checkpoint, run_pi05_finetune
from ..training.rlds_to_lerobot import convert_rlds_to_lerobot_dataset
from .base import PolicyAdapter, PolicyHandle, PolicyTrainingResult

logger = get_logger("policy_adapters.pi05")


class Pi05PolicyAdapter(PolicyAdapter):
    def __init__(self, adapter_config: PolicyAdapterConfig):
        super().__init__(adapter_config)
        self.backend = adapter_config.pi05

    @property
    def name(self) -> str:
        return "pi05"

    def base_model_ref(self, eval_config: PolicyEvalConfig) -> tuple[str, Optional[Path]]:
        return eval_config.model_name, eval_config.checkpoint_path

    def _ensure_openpi_importable(self) -> None:
        repo = self.backend.openpi_repo
        src_dir = repo / "src"
        for candidate in (repo, src_dir):
            candidate_text = str(candidate)
            if candidate.exists() and candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)

    def _load_openpi_policy(self, checkpoint_ref: str):
        self._ensure_openpi_importable()
        policy_config = importlib.import_module("openpi.policies.policy_config")
        create_fn = getattr(policy_config, "create_trained_policy", None)
        if not callable(create_fn):
            raise RuntimeError("openpi.policies.policy_config.create_trained_policy not found")

        kwargs_candidates = [
            {"config_name": self.backend.profile, "checkpoint_dir": checkpoint_ref},
            {"profile": self.backend.profile, "checkpoint_dir": checkpoint_ref},
            {"config_name": self.backend.profile, "checkpoint_path": checkpoint_ref},
            {"profile": self.backend.profile, "checkpoint_path": checkpoint_ref},
        ]
        errors = []
        for kwargs in kwargs_candidates:
            try:
                return create_fn(**kwargs)
            except TypeError as exc:
                errors.append(str(exc))
        raise RuntimeError(
            "Failed to initialize pi05 policy with openpi create_trained_policy. "
            f"Errors: {errors[:2]}"
        )

    def _looks_openvla_like_reference(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
    ) -> bool:
        if checkpoint_path and checkpoint_path.exists():
            return "openvla" in str(checkpoint_path).strip().lower()
        return "openvla" in (model_name or "").strip().lower()

    def load_policy(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
        device: str,
    ) -> PolicyHandle:
        del device  # in-process openpi policy manages runtime placement.
        if self.backend.runtime_mode != "inprocess":
            raise RuntimeError(
                f"Unsupported pi05 runtime_mode={self.backend.runtime_mode}. "
                "Only 'inprocess' is implemented."
            )
        if self._looks_openvla_like_reference(
            model_name=model_name, checkpoint_path=checkpoint_path
        ):
            raise RuntimeError(
                "pi05 adapter selected, but eval_policy reference appears OpenVLA-like. "
                "Set eval_policy.model_name/checkpoint_path to a pi05/OpenPI model reference."
            )
        checkpoint_ref = (
            str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else model_name
        )
        policy = self._load_openpi_policy(checkpoint_ref)
        return PolicyHandle(
            model=policy,
            processor=None,
            metadata={
                "model_name": model_name,
                "checkpoint_path": str(checkpoint_path or ""),
                "pending_actions": [],
                "profile": self.backend.profile,
                "policy_action_dim": self.backend.policy_action_dim,
                "policy_state_dim": self.backend.policy_state_dim,
            },
        )

    def _normalize_action(self, action) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        target_dim = int(self.backend.policy_action_dim)
        if self.backend.strict_action_contract and arr.shape[0] != target_dim:
            raise RuntimeError(
                "pi05 strict_action_contract violation: "
                f"received action_dim={arr.shape[0]}, expected={target_dim}. "
                "Disable strict_action_contract only for research-mode experiments."
            )
        if arr.shape[0] < target_dim:
            arr = np.pad(arr, (0, target_dim - arr.shape[0]))
        elif arr.shape[0] > target_dim:
            arr = arr[:target_dim]
        return arr.astype(np.float32)

    def _build_observation(self, frame, task_prompt: str) -> dict:
        state = np.zeros((int(self.backend.policy_state_dim),), dtype=np.float32)
        observation = {
            "image": frame,
            "wrist_image": frame,
            "state": state,
            "prompt": task_prompt,
        }
        if self.backend.profile == "pi05_droid":
            observation["right_wrist_image"] = frame
        return observation

    def _predict_chunk(self, model, observation: dict):
        method_names = ["infer", "predict_actions", "predict_action", "act", "sample_actions"]
        attempts = []
        for method_name in method_names:
            method = getattr(model, method_name, None)
            if not callable(method):
                continue
            for call in (
                lambda: method(observation=observation),
                lambda: method(obs=observation),
                lambda: method(observation),
                lambda: method(**observation),
            ):
                try:
                    return call()
                except TypeError as exc:
                    attempts.append(f"{method_name}:{exc}")
                    continue
        if callable(model):
            for call in (
                lambda: model(observation=observation),
                lambda: model(obs=observation),
                lambda: model(observation),
            ):
                try:
                    return call()
                except TypeError as exc:
                    attempts.append(f"__call__:{exc}")
                    continue
        raise RuntimeError(
            f"pi05 policy does not expose a usable inference method ({attempts[:2]})"
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
        queue = handle.metadata.setdefault("pending_actions", [])
        if queue:
            return self._normalize_action(queue.pop(0))

        observation = self._build_observation(frame=frame, task_prompt=task_prompt)
        chunk = self._predict_chunk(handle.model, observation)

        if isinstance(chunk, dict):
            if "actions" in chunk:
                chunk = chunk["actions"]
            elif "action" in chunk:
                chunk = chunk["action"]

        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim == 1:
            return self._normalize_action(arr)
        if arr.ndim >= 2:
            arr = arr.reshape(arr.shape[0], -1)
            for step in arr[1:]:
                queue.append(step.tolist())
            return self._normalize_action(arr[0])
        return self._normalize_action(arr.reshape(-1))

    def dataset_transform(
        self,
        source_dataset_dir: Path,
        output_root: Path,
        dataset_name: str,
    ) -> Path:
        return convert_rlds_to_lerobot_dataset(
            source_dataset_dir=source_dataset_dir,
            output_root=output_root,
            dataset_name=dataset_name,
            profile=self.backend.profile,
            policy_state_dim=self.backend.policy_state_dim,
            policy_action_dim=self.backend.policy_action_dim,
        )

    def train_policy(
        self,
        base_model_name: str,
        base_checkpoint: Optional[Path],
        dataset_root: Path,
        dataset_name: str,
        output_dir: Path,
        finetune_config: PolicyFinetuneConfig,
    ) -> PolicyTrainingResult:
        result = run_pi05_finetune(
            config=finetune_config,
            backend=self.backend,
            base_model_name=base_model_name,
            base_checkpoint=base_checkpoint,
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            facility_id=output_dir.name,
            output_dir=output_dir,
        )
        adapted = result.get("adapted_checkpoint_path")
        return PolicyTrainingResult(
            status=result.get("status", "failed"),
            adapted_checkpoint_path=Path(adapted) if adapted else None,
            elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
            detail=result.get("stderr", "") or result.get("detail", ""),
            raw=result,
        )

    def resolve_latest_checkpoint(self, run_root_dir: Path) -> Optional[Path]:
        return resolve_latest_pi05_checkpoint(run_root_dir)
