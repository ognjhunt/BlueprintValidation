"""Manage DreamDojo model checkpoints (baseline vs site-adapted)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..common import get_logger

logger = get_logger("training.checkpoint_manager")


@dataclass
class CheckpointInfo:
    """Information about a DreamDojo checkpoint."""

    path: Path
    is_baseline: bool
    facility_id: Optional[str] = None
    lora_path: Optional[Path] = None

    @property
    def exists(self) -> bool:
        return self.path.exists()


class CheckpointManager:
    """Manage baseline and site-adapted DreamDojo checkpoints."""

    def __init__(self, base_checkpoint: Path, work_dir: Path):
        self.base_checkpoint = base_checkpoint
        self.work_dir = work_dir

    def get_baseline(self) -> CheckpointInfo:
        return CheckpointInfo(
            path=self.base_checkpoint,
            is_baseline=True,
        )

    def get_adapted(self, facility_id: str) -> CheckpointInfo:
        lora_dir = self.work_dir / facility_id / "finetune" / "lora_weights"
        return CheckpointInfo(
            path=self.base_checkpoint,
            is_baseline=False,
            facility_id=facility_id,
            lora_path=lora_dir if lora_dir.exists() else None,
        )

    def has_adapted(self, facility_id: str) -> bool:
        info = self.get_adapted(facility_id)
        return info.lora_path is not None and info.lora_path.exists()

    def list_adapted(self) -> list[str]:
        """List facility IDs that have adapted checkpoints."""
        adapted = []
        if self.work_dir.exists():
            for fdir in self.work_dir.iterdir():
                if fdir.is_dir() and self.has_adapted(fdir.name):
                    adapted.append(fdir.name)
        return adapted
