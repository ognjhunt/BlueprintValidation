"""Shared types, logging, and error definitions."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=level, stream=sys.stderr)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"blueprint_validation.{name}")


class PipelineError(Exception):
    """Base error for pipeline failures."""

    def __init__(self, stage: str, message: str, detail: str = ""):
        self.stage = stage
        self.detail = detail
        super().__init__(f"[{stage}] {message}")


class PreflightError(PipelineError):
    """Raised when preflight checks fail."""

    def __init__(self, message: str, checks: List[PreflightCheck] | None = None):
        self.checks = checks or []
        super().__init__("preflight", message)


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageResult:
    """Result of a pipeline stage execution."""

    stage_name: str
    status: str  # "success", "failed", "skipped"
    elapsed_seconds: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "outputs": self.outputs,
            "metrics": self.metrics,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
