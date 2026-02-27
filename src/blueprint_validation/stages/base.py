"""Abstract base class for pipeline stages."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from ..common import PreflightCheck, StageResult, get_logger
from ..config import FacilityConfig, ValidationConfig

logger = get_logger("stages.base")


class PipelineStage(ABC):
    """Abstract base for all validation pipeline stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this stage (e.g., 's1_render')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this stage does."""
        ...

    @abstractmethod
    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        """Execute this stage. Returns a StageResult with outputs and metrics."""
        ...

    def preflight(self, config: ValidationConfig) -> List[PreflightCheck]:
        """Run preflight checks for this stage. Override in subclasses."""
        return []

    def execute(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        """Wrapper that handles timing, logging, and error capture."""
        logger.info("Starting stage: %s — %s", self.name, self.description)
        start = time.monotonic()
        try:
            result = self.run(config, facility, work_dir, previous_results)
            elapsed = time.monotonic() - start
            logger.info(
                "Stage %s completed in %.1fs with status: %s",
                self.name,
                elapsed,
                result.status,
            )
            return StageResult(
                stage_name=result.stage_name,
                status=result.status,
                elapsed_seconds=elapsed,
                outputs=result.outputs,
                metrics=result.metrics,
                detail=result.detail,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error("Stage %s failed after %.1fs: %s", self.name, elapsed, e)
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=elapsed,
                detail=str(e),
            )
