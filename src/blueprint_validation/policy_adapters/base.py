"""Policy adapter interface for multi-policy support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import PolicyFinetuneConfig


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

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load_policy(
        self,
        model_name: str,
        checkpoint_path: Optional[Path],
        device: str,
    ) -> PolicyHandle:
        ...

    @abstractmethod
    def predict_action(
        self,
        handle: PolicyHandle,
        frame,
        task_prompt: str,
        unnorm_key: Optional[str],
        device: str,
    ):
        ...

    @abstractmethod
    def dataset_transform(
        self,
        source_dataset_dir: Path,
        output_root: Path,
        dataset_name: str,
    ) -> Path:
        ...

    @abstractmethod
    def train_policy(
        self,
        base_model_name: str,
        base_checkpoint: Optional[Path],
        dataset_root: Path,
        dataset_name: str,
        output_dir: Path,
        finetune_config: PolicyFinetuneConfig,
    ) -> PolicyTrainingResult:
        ...
