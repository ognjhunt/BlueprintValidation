#!/usr/bin/env python3
"""Run optional VLM QA over Stage-1 rendered clips.

This script reads Stage-1 render manifest clips and scores each clip with the VLM judge.
Per-clip prompt input uses `expected_focus_text` when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_validation.config import VLMJudgeConfig  # noqa: E402
from blueprint_validation.evaluation.vlm_judge import score_rollout  # noqa: E402

DEFAULT_SCORING_PROMPT = (
    "You are evaluating a Stage-1 camera clip.\n"
    'Expected focus for this clip: "{task}"\n\n'
    "Score on a 0-10 scale:\n"
    "1. task_score: How well does framing and temporal focus match the expected focus?\n"
    "2. visual_score: How sharp/clean is the clip (avoid blur/green-cast artifacts)?\n"
    "3. spatial_score: How useful is this camera view for downstream world-model/policy training?\n"
    'Return JSON: {"task_score": N, "visual_score": N, "spatial_score": N, "reasoning": "..."}'
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 VLM clip QA runner")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Facility run directory (contains renders/render_manifest.json).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit manifest path. Defaults to <run-dir>/renders/render_manifest.json.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <run-dir>/s1_vlm_clip_qa.json.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GOOGLE_GENAI_API_KEY",
        help="Environment variable that holds Gemini API key.",
    )
    parser.add_argument(
        "--video-metadata-fps",
        type=float,
        default=10.0,
        help=(
            "Explicit Gemini native-video metadata FPS. "
            "Set to 0 to disable explicit FPS metadata."
        ),
    )
    parser.add_argument(
        "--task-threshold",
        type=float,
        default=7.0,
        help="Minimum task_score for pass.",
    )
    parser.add_argument(
        "--visual-threshold",
        type=float,
        default=7.0,
        help="Minimum visual_score for pass.",
    )
    parser.add_argument(
        "--spatial-threshold",
        type=float,
        default=7.0,
        help="Minimum spatial_score for pass.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Optional clip cap. <=0 means all clips.",
    )
    parser.add_argument(
        "--max-judge-frames",
        type=int,
        default=0,
        help=(
            "Deprecated. Ignored because native video upload is always used for scoring."
        ),
    )
    parser.add_argument(
        "--facility-description",
        default="",
        help="Optional facility context passed to the VLM judge.",
    )
    parser.add_argument(
        "--require-expected-focus",
        action="store_true",
        help="Fail clips that do not already contain expected_focus_text.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call VLM API; emit prompt metadata only.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return non-zero when any clip fails QA or has errors.",
    )
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _infer_expected_focus_text(clip: dict[str, Any]) -> str:
    """Fallback for legacy manifests that do not include expected_focus_text."""
    path_context = clip.get("path_context")
    if not isinstance(path_context, dict):
        path_context = {}

    role = _clean_text(path_context.get("target_role")).lower()
    target_label = (
        _clean_text(path_context.get("target_label"))
        or _clean_text(path_context.get("target_instance_id"))
        or _clean_text(path_context.get("target_category"))
    )
    path_type = _clean_text(clip.get("path_type")).lower()

    if role == "targets":
        if target_label:
            return f"Primary target focus: keep {target_label} centered and clearly visible."
        return "Primary target focus: keep the task target centered and clearly visible."
    if role == "context":
        if target_label:
            return (
                f"Context focus: keep {target_label} visible with nearby objects and interaction area."
            )
        return "Context focus: keep the task region visible with nearby objects and interaction area."
    if role == "overview":
        return "Overview focus: capture broad scene layout and navigation anchors."
    if role == "fallback":
        return "Fallback focus: capture stable, useful scene coverage."

    if path_type == "manipulation":
        if target_label:
            return (
                f"Manipulation focus: keep {target_label} and the interaction zone in frame."
            )
        return "Manipulation focus: keep the task object and interaction zone in frame."
    if path_type == "orbit":
        return "Orbit focus: provide stable global scene coverage."
    if path_type == "sweep":
        return "Sweep focus: scan scene to expose spatial relationships and task regions."
    if path_type == "file":
        return "Path-file focus: follow predefined path with stable framing."
    return "General focus: produce clear, stable scene coverage useful for downstream use."


def _resolve_expected_focus_text(clip: dict[str, Any]) -> tuple[str, bool]:
    text = _clean_text(clip.get("expected_focus_text"))
    if text:
        return text, False
    return _infer_expected_focus_text(clip), True


def _clip_pass(
    *,
    task_score: float,
    visual_score: float,
    spatial_score: float,
    task_threshold: float,
    visual_threshold: float,
    spatial_threshold: float,
) -> bool:
    return (
        float(task_score) >= float(task_threshold)
        and float(visual_score) >= float(visual_threshold)
        and float(spatial_score) >= float(spatial_threshold)
    )


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if args.manifest_path is not None
        else (run_dir / "renders" / "render_manifest.json")
    )
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else (run_dir / "s1_vlm_clip_qa.json")
    )

    if not manifest_path.exists():
        print(f"Render manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to load manifest {manifest_path}: {exc}", file=sys.stderr)
        return 2

    clips = list(manifest.get("clips", []))
    if int(args.max_clips) > 0:
        clips = clips[: int(args.max_clips)]
    if int(args.max_judge_frames) > 0:
        print(
            "Warning: --max-judge-frames is deprecated and ignored "
            "(native Gemini video scoring is always used).",
            file=sys.stderr,
        )

    judge_cfg = VLMJudgeConfig(
        model=str(args.model),
        api_key_env=str(args.api_key_env),
        video_metadata_fps=float(args.video_metadata_fps),
        scoring_prompt=DEFAULT_SCORING_PROMPT,
    )

    rows: list[dict[str, Any]] = []
    num_missing_expected_focus = 0
    num_pass = 0
    num_fail = 0
    num_error = 0

    for clip in clips:
        clip_name = _clean_text(clip.get("clip_name")) or "unknown_clip"
        video_path_text = _clean_text(clip.get("video_path"))
        video_path = Path(video_path_text).expanduser() if video_path_text else None
        path_context = clip.get("path_context")
        if not isinstance(path_context, dict):
            path_context = {}

        expected_focus_text, inferred = _resolve_expected_focus_text(clip)
        if inferred:
            num_missing_expected_focus += 1

        row: dict[str, Any] = {
            "clip_name": clip_name,
            "video_path": str(video_path) if video_path is not None else "",
            "path_type": _clean_text(clip.get("path_type")),
            "target_role": _clean_text(path_context.get("target_role")),
            "expected_focus_text": expected_focus_text,
            "expected_focus_inferred": bool(inferred),
            "task_prompt_used": expected_focus_text,
        }

        if bool(args.require_expected_focus) and inferred:
            row["error"] = "missing expected_focus_text in manifest clip"
            row["pass"] = False
            num_error += 1
            rows.append(row)
            continue

        if bool(args.dry_run):
            row["dry_run"] = True
            row["video_exists"] = bool(video_path is not None and video_path.exists())
            row["pass"] = True
            num_pass += 1
            rows.append(row)
            continue

        if video_path is None or not video_path.exists():
            row["error"] = "video_path missing or file not found"
            row["pass"] = False
            num_error += 1
            rows.append(row)
            continue

        try:
            score = score_rollout(
                video_path=video_path,
                task_prompt=expected_focus_text,
                config=judge_cfg,
                facility_description=str(args.facility_description or ""),
            )
            row["task_score"] = float(score.task_score)
            row["visual_score"] = float(score.visual_score)
            row["spatial_score"] = float(score.spatial_score)
            row["reasoning"] = str(score.reasoning)
            passed = _clip_pass(
                task_score=float(score.task_score),
                visual_score=float(score.visual_score),
                spatial_score=float(score.spatial_score),
                task_threshold=float(args.task_threshold),
                visual_threshold=float(args.visual_threshold),
                spatial_threshold=float(args.spatial_threshold),
            )
            row["pass"] = bool(passed)
            if passed:
                num_pass += 1
            else:
                num_fail += 1
        except Exception as exc:
            row["error"] = str(exc)
            row["pass"] = False
            num_error += 1
        rows.append(row)

    summary = {
        "num_clips": len(clips),
        "num_pass": num_pass,
        "num_fail": num_fail,
        "num_error": num_error,
        "num_missing_expected_focus_text": num_missing_expected_focus,
        "manifest_path": str(manifest_path),
        "model": str(args.model),
        "video_metadata_fps": float(args.video_metadata_fps),
        "judge_input_mode": "native_video_upload",
        "dry_run": bool(args.dry_run),
        "require_expected_focus": bool(args.require_expected_focus),
    }
    payload = {"summary": summary, "clips": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved: {output_path}")

    if bool(args.strict_exit) and (num_fail > 0 or num_error > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
