"""OpenVLA-OFT policy adapter implementation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..common import get_logger
from ..config import PolicyFinetuneConfig
from ..training.openvla_finetune import run_openvla_finetune
from .base import PolicyAdapter, PolicyHandle, PolicyTrainingResult

logger = get_logger("policy_adapters.openvla_oft")


class OpenVLAOFTPolicyAdapter(PolicyAdapter):
    @property
    def name(self) -> str:
        return "openvla_oft"

    def load_policy(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
        device: str,
    ) -> PolicyHandle:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch

        model_id = str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else model_name
        logger.info("Loading OpenVLA-OFT policy from %s", model_id)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        return PolicyHandle(
            model=model,
            processor=processor,
            metadata={
                "model_name": model_name,
                "checkpoint_path": str(checkpoint_path or ""),
                "pending_actions": [],
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
        # Consume pending chunked actions first.
        queue = handle.metadata.setdefault("pending_actions", [])
        if queue:
            return np.asarray(queue.pop(0), dtype=np.float32)

        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            Image = None

        try:
            import torch
            torch_dtype = torch.bfloat16
        except ImportError:  # pragma: no cover
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

        action = self._predict_chunk(handle.model, predict_kwargs)
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > 0:
            for step in arr[1:]:
                queue.append(step.tolist())
            return arr[0]
        if arr.ndim == 1:
            return arr
        return np.asarray(arr.reshape(-1), dtype=np.float32)

    def _predict_chunk(self, model, predict_kwargs: dict):
        method_candidates = [
            "predict_action_chunk",
            "predict_actions",
            "predict_action_parallel",
            "predict_action",
        ]
        for method_name in method_candidates:
            method = getattr(model, method_name, None)
            if not callable(method):
                continue
            try:
                return method(**predict_kwargs)
            except TypeError:
                kwargs = dict(predict_kwargs)
                kwargs.pop("unnorm_key", None)
                return method(**kwargs)
        raise RuntimeError("Loaded OpenVLA-OFT model does not expose action prediction method.")

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

        meta = {
            "adapter": self.name,
            "source_dataset_dir": str(source_dataset_dir),
            "dataset_name": dataset_name,
            "format": "rlds_style_jsonl",
            "decoding": "chunked_parallel_when_available",
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
        local_cfg.recipe = "oft"

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
