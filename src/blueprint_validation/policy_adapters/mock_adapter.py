"""Lightweight deterministic policy adapter for local hosted-session testing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import PolicyEvalConfig, PolicyFinetuneConfig
from .base import PolicyAdapter, PolicyHandle, PolicyTrainingResult


class MockPolicyAdapter(PolicyAdapter):
    @property
    def name(self) -> str:
        return "mock"

    def base_model_ref(self, eval_config: PolicyEvalConfig) -> tuple[str, Optional[Path]]:
        return str(eval_config.model_name or "mock-policy"), eval_config.checkpoint_path

    def load_policy(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
        device: str,
    ) -> PolicyHandle:
        return PolicyHandle(
            model={"model_name": model_name, "checkpoint_path": str(checkpoint_path or ""), "device": device},
            processor=None,
            metadata={"step_index": 0},
        )

    def predict_action(
        self,
        handle: PolicyHandle,
        frame,
        task_prompt: str,
        unnorm_key: Optional[str],
        device: str,
    ):
        del frame, unnorm_key, device
        step_index = int(handle.metadata.get("step_index", 0))
        base = (sum(ord(char) for char in task_prompt) + step_index) % 11
        handle.metadata["step_index"] = step_index + 1
        return np.asarray(
            [
                round(((base % 5) - 2) * 0.1, 3),
                round((((base + 1) % 5) - 2) * 0.1, 3),
                round((((base + 2) % 5) - 2) * 0.1, 3),
                0.05,
                -0.05,
                0.02,
                1.0 if step_index % 4 == 0 else -1.0,
            ],
            dtype=np.float32,
        )

    def dataset_transform(
        self,
        source_dataset_dir: Path,
        output_root: Path,
        dataset_name: str,
    ) -> Path:
        target = output_root / dataset_name
        target.mkdir(parents=True, exist_ok=True)
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
        del base_model_name, base_checkpoint, dataset_root, dataset_name, finetune_config
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "mock-policy.ckpt"
        checkpoint.write_text("mock", encoding="utf-8")
        return PolicyTrainingResult(
            status="success",
            adapted_checkpoint_path=checkpoint,
            elapsed_seconds=0.0,
            detail="mock adapter does not train; wrote placeholder checkpoint",
        )

    def resolve_latest_checkpoint(self, run_root_dir: Path) -> Optional[Path]:
        candidate = run_root_dir / "mock-policy.ckpt"
        return candidate if candidate.exists() else None
