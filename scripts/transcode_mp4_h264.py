#!/usr/bin/env python3
"""Batch-convert MP4 files to H.264 without re-running the pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    input_path: Path
    output_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively transcode MP4 files to H.264 (libx264). "
            "By default writes to a sibling '<input>_h264' tree."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Root directory to scan recursively for .mp4")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for converted files (ignored with --in-place)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace source files in place (writes temporary file then swaps).",
    )
    parser.add_argument(
        "--backup-ext",
        default="",
        help=(
            "When using --in-place, keep source as '<name><backup-ext>' before replacement "
            "(example: '.bak'). Default: no backup."
        ),
    )
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF (lower is higher quality)")
    parser.add_argument("--preset", default="medium", help="libx264 preset (ultrafast..veryslow)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion when output file already exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without running ffmpeg")
    return parser.parse_args()


def _find_mp4_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.mp4")
        if p.is_file() and "__tmp_h264__" not in p.name and not p.name.startswith(".")
    )


def _build_jobs(
    *,
    input_root: Path,
    files: list[Path],
    in_place: bool,
    output_root: Path | None,
) -> list[Job]:
    jobs: list[Job] = []
    if in_place:
        for src in files:
            jobs.append(Job(input_path=src, output_path=src))
        return jobs

    target_root = output_root if output_root is not None else input_root.parent / f"{input_root.name}_h264"
    for src in files:
        rel = src.relative_to(input_root)
        jobs.append(Job(input_path=src, output_path=target_root / rel))
    return jobs


def _ffmpeg_cmd(*, src: Path, dst: Path, crf: int, preset: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]


def _transcode_one(
    *,
    job: Job,
    in_place: bool,
    backup_ext: str,
    crf: int,
    preset: str,
    dry_run: bool,
) -> None:
    src = job.input_path
    final_dst = job.output_path

    if in_place:
        tmp_dst = src.with_name(f"{src.stem}.__tmp_h264__.mp4")
    else:
        tmp_dst = final_dst
        final_dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = _ffmpeg_cmd(src=src, dst=tmp_dst, crf=crf, preset=preset)
    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed for {src}: {err}")
    if not tmp_dst.exists() or tmp_dst.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced empty output for {src}: {tmp_dst}")

    if not in_place:
        return

    backup_path: Path | None = None
    if backup_ext:
        backup_path = src.with_name(f"{src.name}{backup_ext}")
        if backup_path.exists():
            tmp_dst.unlink(missing_ok=True)
            raise RuntimeError(f"Backup target already exists: {backup_path}")
        os.replace(src, backup_path)
        os.replace(tmp_dst, src)
    else:
        os.replace(tmp_dst, src)


def main() -> int:
    args = _parse_args()
    input_root = args.input_dir.expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        print(f"Input directory not found: {input_root}", file=sys.stderr)
        return 2
    if args.in_place and args.output_root is not None:
        print("--output-root cannot be used with --in-place", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH", file=sys.stderr)
        return 2

    files = _find_mp4_files(input_root)
    if not files:
        print(f"No MP4 files found under {input_root}")
        return 0

    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    jobs = _build_jobs(
        input_root=input_root,
        files=files,
        in_place=bool(args.in_place),
        output_root=output_root,
    )

    converted = 0
    skipped = 0
    failures = 0

    print(f"Found {len(jobs)} MP4 files under {input_root}")
    for idx, job in enumerate(jobs, start=1):
        src = job.input_path
        dst = job.output_path

        if args.skip_existing and not args.in_place and dst.exists():
            skipped += 1
            print(f"[{idx}/{len(jobs)}] SKIP exists: {dst}")
            continue

        print(f"[{idx}/{len(jobs)}] CONVERT {src} -> {dst}")
        try:
            _transcode_one(
                job=job,
                in_place=bool(args.in_place),
                backup_ext=str(args.backup_ext or ""),
                crf=int(args.crf),
                preset=str(args.preset),
                dry_run=bool(args.dry_run),
            )
            converted += 1
        except Exception as exc:
            failures += 1
            print(f"[{idx}/{len(jobs)}] ERROR {src}: {exc}", file=sys.stderr)

    print(
        f"Done. converted={converted} skipped={skipped} failures={failures} "
        f"mode={'in-place' if args.in_place else 'copy'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
