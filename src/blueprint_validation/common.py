"""Shared types, logging, and error definitions."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
    payload = json.dumps(data, indent=2, default=str)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sanitize_filename_component(
    value: object,
    *,
    fallback: str = "item",
    max_length: int = 120,
) -> str:
    """Return a path-safe filename component for derived artifact names."""
    raw = str(value or "").strip()
    text = raw.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    if not text or text in {".", ".."}:
        text = str(fallback or "item").strip() or "item"
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("._-") or "item"
    if int(max_length) > 0 and len(text) > int(max_length):
        text = text[: int(max_length)].rstrip("._-")
        if not text:
            text = "item"
    return text


def sanitize_filename_component_with_hash(
    value: object,
    *,
    fallback: str = "item",
    max_length: int = 120,
    hash_len: int = 8,
) -> str:
    """Return a path-safe component with a stable short hash suffix.

    The hash prevents collisions when distinct raw values collapse to the same
    sanitized stem.
    """
    stem = sanitize_filename_component(value, fallback=fallback, max_length=max_length)
    raw = str(value or "")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[: max(4, int(hash_len))]
    with_hash = f"{stem}-{digest}"
    if int(max_length) > 0 and len(with_hash) > int(max_length):
        trim_budget = int(max_length) - (len(digest) + 1)
        if trim_budget > 0:
            trimmed_stem = stem[:trim_budget].rstrip("._-")
            if not trimmed_stem:
                trimmed_stem = str(fallback or "item")
            return f"{trimmed_stem}-{digest}"
    return with_hash
