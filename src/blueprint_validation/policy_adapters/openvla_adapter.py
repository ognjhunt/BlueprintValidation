"""OpenVLA policy adapter implementation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..common import get_logger
from ..config import PolicyFinetuneConfig
from ..evaluation.openvla_runner import load_openvla
from ..training.openvla_finetune import run_openvla_finetune
from .base import PolicyAdapter, PolicyHandle, PolicyTrainingResult

logger = get_logger("policy_adapters.openvla")


class OpenVLAPolicyAdapter(PolicyAdapter):
    @property
    def name(self) -> str:
        return "openvla"

    def load_policy(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
        device: str,
    ) -> PolicyHandle:
        model, processor = load_openvla(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        return PolicyHandle(
            model=model,
            processor=processor,
            metadata={"model_name": model_name, "checkpoint_path": str(checkpoint_path or "")},
        )

    def predict_action(
        self,
        handle: PolicyHandle,
        frame,
        task_prompt: str,
        unnorm_key: Optional[str],
        device: str,
    ):
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - exercised in lightweight test envs
            Image = None

        try:
            import torch
            torch_dtype = torch.bfloat16
        except ImportError:  # pragma: no cover - exercised in lightweight test envs
            torch_dtype = None

        image = Image.fromarray(frame) if Image else frame
        prompt = f"In: What action should the robot take to {task_prompt}?\nOut:"
        inputs = handle.processor(prompt, image, return_tensors="pt")

        if hasattr(inputs, "to"):
            if torch_dtype is not None:
                inputs = inputs.to(device, dtype=torch_dtype)
            else:
                inputs = inputs.to(device)
        else:
            inputs = {
                key: (
                    value.to(device, dtype=torch_dtype)
                    if hasattr(value, "to") and torch_dtype is not None
                    else value.to(device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }

        predict_kwargs = dict(inputs)
        predict_kwargs["do_sample"] = False
        if unnorm_key:
            predict_kwargs["unnorm_key"] = unnorm_key
        try:
            action = handle.model.predict_action(**predict_kwargs)
        except TypeError:
            predict_kwargs.pop("unnorm_key", None)
            action = handle.model.predict_action(**predict_kwargs)
        return np.asarray(action, dtype=np.float32)

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

        # Keep transform deterministic and inspectable: copy the RLDS-style export as-is.
        # OpenVLA users can either train with a compatible custom loader or transform this
        # directory into a registered OXE dataset.
        for item in source_dataset_dir.iterdir():
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        meta = {
            "adapter": self.name,
            "source_dataset_dir": str(source_dataset_dir),
            "dataset_name": dataset_name,
            "format": "rlds_style_jsonl",
        }
        (target / "adapter_dataset_meta.json").write_text(json.dumps(meta, indent=2))
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
        local_cfg = PolicyFinetuneConfig(**finetune_config.__dict__)
        local_cfg.data_root_dir = dataset_root
        local_cfg.dataset_name = dataset_name

        checkpoint_str = (
            str(base_checkpoint) if base_checkpoint and base_checkpoint.exists() else base_model_name
        )
        result = run_openvla_finetune(
            config=local_cfg,
            vla_path=checkpoint_str,
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
