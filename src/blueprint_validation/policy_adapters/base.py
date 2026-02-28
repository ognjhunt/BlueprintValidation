"""Policy adapter interface for multi-policy support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import PolicyAdapterConfig, PolicyEvalConfig, PolicyFinetuneConfig


@dataclass
class PolicyHandle:
    """Loaded policy handle used for rollout inference."""

    model: Any
    processor: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyTrainingResult:
    """Standardized result from adapter training."""

    status: str
    adapted_checkpoint_path: Optional[Path]
    elapsed_seconds: float
    detail: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class PolicyAdapter(ABC):
    """Abstract adapter contract for policy families."""

    def __init__(self, adapter_config: PolicyAdapterConfig):
        self.adapter_config = adapter_config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def base_model_ref(self, eval_config: PolicyEvalConfig) -> tuple[str, Optional[Path]]:
        """Resolve baseline model id and optional checkpoint path for this adapter."""
        ...

    @abstractmethod
    def load_policy(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
        device: str,
    ) -> PolicyHandle: ...

    @abstractmethod
    def predict_action(
        self,
        handle: PolicyHandle,
        frame,
        task_prompt: str,
        unnorm_key: Optional[str],
        device: str,
    ): ...

    @abstractmethod
    def dataset_transform(
        self,
        source_dataset_dir: Path,
        output_root: Path,
        dataset_name: str,
    ) -> Path: ...

    @abstractmethod
    def train_policy(
        self,
        base_model_name: str,
        base_checkpoint: Optional[Path],
        dataset_root: Path,
        dataset_name: str,
        output_dir: Path,
        finetune_config: PolicyFinetuneConfig,
    ) -> PolicyTrainingResult: ...

    @abstractmethod
    def resolve_latest_checkpoint(self, run_root_dir: Path) -> Optional[Path]:
        """Resolve the newest adapter checkpoint under a run root."""
        ...
