"""Reproducibility/provenance stamps for pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_hash(payload: Any) -> str:
    canonical = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def command_version(cmd: str, *, args: Iterable[str] = ("-version",)) -> str | None:
    binary = shutil.which(cmd)
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary, *list(args)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    text = (result.stdout or result.stderr or "").strip().splitlines()
    return text[0].strip() if text else None


def get_git_status(repo_root: Path | None = None) -> Dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    git = shutil.which("git")
    if git is None:
        return {"repo_root": str(root), "git_available": False}

    def _run(args: list[str]) -> str:
        result = subprocess.run(
            [git, *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return (result.stdout or "").strip()

    commit = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_run(["status", "--porcelain"]))
    return {
        "repo_root": str(root),
        "git_available": True,
        "commit": commit or None,
        "branch": branch or None,
        "dirty": dirty,
    }


def runtime_versions() -> Dict[str, Any]:
    cv2_version = None
    try:
        import cv2  # type: ignore

        cv2_version = str(getattr(cv2, "__version__", "") or None)
    except Exception:
        cv2_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": command_version("ffmpeg"),
        "ffprobe": command_version("ffprobe"),
        "opencv": cv2_version,
    }


def build_provenance_stamp(
    *,
    stage: str,
    config_obj: Any | None = None,
    input_paths: Iterable[Path] | None = None,
    output_paths: Iterable[Path] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    in_paths = [Path(p) for p in (input_paths or [])]
    out_paths = [Path(p) for p in (output_paths or [])]
    input_hashes = {
        str(p): file_sha256(p) for p in in_paths if p.exists() and p.is_file()
    }
    output_hashes = {
        str(p): file_sha256(p) for p in out_paths if p.exists() and p.is_file()
    }
    config_hash = canonical_json_hash(config_obj) if config_obj is not None else None
    stamp = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "git": get_git_status(),
        "runtime": runtime_versions(),
    }
    if extra:
        stamp["extra"] = _to_jsonable(extra)
    stamp["stamp_hash"] = canonical_json_hash(stamp)
    return stamp
