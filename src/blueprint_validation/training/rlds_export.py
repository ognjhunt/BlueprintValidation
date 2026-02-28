"""Utilities for exporting rollout records to RLDS-style episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..common import get_logger

logger = get_logger("training.rlds_export")


@dataclass
class EpisodeExport:
    episode_id: str
    path: Path
    num_steps: int
    success: bool
    task: str
    rollout_index: int


def extract_video_frames(video_path: Path, output_dir: Path) -> List[Path]:
    """Extract RGB frames from MP4 video into JPEG files."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Path] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = output_dir / f"{frame_idx:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        frame_idx += 1
    cap.release()
    return frames


def rollout_success(entry: Dict, task_threshold: float = 7.0) -> bool:
    """Determine success using explicit manipulation checks when present."""
    if entry.get("is_manipulation_task"):
        grasp = entry.get("grasp_acquired")
        lifted = entry.get("lifted_clear")
        placed = entry.get("placed_in_target")
        if grasp is not None and lifted is not None and placed is not None:
            return bool(grasp) and bool(lifted) and bool(placed)
    return float(entry.get("task_score", 0.0)) >= task_threshold


def export_rollouts_to_rlds_jsonl(
    rollouts: List[Dict],
    output_dir: Path,
    condition: str,
    split: str,
    task_threshold: float,
    min_steps_per_rollout: int,
    include_failed_rollouts: bool,
) -> Dict:
    """Export rollout records into RLDS-style JSONL episode files.

    Output structure:
      output_dir/
        episodes.jsonl
        episodes_meta.json
        frames/<episode_id>/<step>.jpg
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = output_dir / "frames"
    episodes_jsonl = output_dir / "episodes.jsonl"
    exported: List[EpisodeExport] = []

    with episodes_jsonl.open("w") as f:
        for entry in rollouts:
            actions = entry.get("action_sequence") or []
            if len(actions) < min_steps_per_rollout:
                continue

            success = rollout_success(entry, task_threshold=task_threshold)
            if not include_failed_rollouts and not success:
                continue

            video_path = Path(entry["video_path"])
            if not video_path.exists():
                logger.warning("Skipping missing rollout video: %s", video_path)
                continue

            episode_id = (
                f"{condition}_{split}_{int(entry['rollout_index']):04d}_"
                f"{entry['task'][:32].replace(' ', '_')}"
            )
            frame_paths = extract_video_frames(video_path, frames_root / episode_id)
            if len(frame_paths) < 2:
                continue

            num_steps = min(len(actions), len(frame_paths) - 1)
            if num_steps < min_steps_per_rollout:
                continue

            steps = []
            task_prompt = entry.get("task", "")
            for step_idx in range(num_steps):
                is_last = step_idx == num_steps - 1
                steps.append(
                    {
                        "observation": {
                            "image_path": str(frame_paths[step_idx]),
                        },
                        "action": list(actions[step_idx]),
                        "language_instruction": task_prompt,
                        "reward": 1.0 if (is_last and success) else 0.0,
                        "is_first": step_idx == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                    }
                )

            payload = {
                "episode_id": episode_id,
                "condition": condition,
                "split": split,
                "task": task_prompt,
                "rollout_index": int(entry["rollout_index"]),
                "task_score": float(entry.get("task_score", 0.0)),
                "success": success,
                "steps": steps,
                "manipulation": {
                    "is_manipulation_task": bool(entry.get("is_manipulation_task", False)),
                    "grasp_acquired": entry.get("grasp_acquired"),
                    "lifted_clear": entry.get("lifted_clear"),
                    "placed_in_target": entry.get("placed_in_target"),
                    "stable_after_place": entry.get("stable_after_place"),
                },
            }
            f.write(json.dumps(payload) + "\n")

            exported.append(
                EpisodeExport(
                    episode_id=episode_id,
                    path=episodes_jsonl,
                    num_steps=num_steps,
                    success=success,
                    task=task_prompt,
                    rollout_index=int(entry["rollout_index"]),
                )
            )

    meta = {
        "format": "rlds_style_jsonl",
        "condition": condition,
        "split": split,
        "num_episodes": len(exported),
        "num_successes": sum(1 for e in exported if e.success),
        "num_failures": sum(1 for e in exported if not e.success),
        "mean_steps": (
            round(sum(e.num_steps for e in exported) / len(exported), 2) if exported else 0
        ),
        "episodes_jsonl": str(episodes_jsonl),
    }
    (output_dir / "episodes_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# TFRecord conversion for OpenVLA-OFT fine-tuning
# ---------------------------------------------------------------------------


def convert_jsonl_to_tfrecord(
    train_jsonl_path: Path,
    eval_jsonl_path: Path | None,
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """Convert JSONL episodes into TFRecord RLDS format consumed by OpenVLA-OFT.

    Creates the directory layout expected by ``tensorflow_datasets``::

        output_dir/
            dataset_name/
                1.0.0/
                    dataset_info.json
                    features.json
                    train/
                        train-00000-of-00001.tfrecord
                    eval/  (if eval_jsonl_path provided)
                        eval-00000-of-00001.tfrecord

    Each step record contains ``observation/image``, ``action``,
    ``language_instruction``, ``reward``, ``is_first``, ``is_last``,
    ``is_terminal``.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.warning(
            "tensorflow not installed — writing JSONL-only dataset at %s. "
            "Install tensorflow to produce TFRecords for OpenVLA-OFT.",
            output_dir,
        )
        dataset_root = output_dir / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)
        return dataset_root

    dataset_root = output_dir / dataset_name
    ds_root = dataset_root / "1.0.0"

    for split_name, jsonl_path in [("train", train_jsonl_path), ("eval", eval_jsonl_path)]:
        if jsonl_path is None or not jsonl_path.exists():
            continue
        split_dir = ds_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        tfrecord_path = split_dir / f"{split_name}-00000-of-00001.tfrecord"

        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for line in jsonl_path.read_text().strip().splitlines():
                episode = json.loads(line)
                for step in episode.get("steps", []):
                    image_path = step["observation"]["image_path"]
                    if not Path(image_path).exists():
                        continue

                    image_bytes = Path(image_path).read_bytes()
                    action = step["action"]
                    # Pad or truncate to 7-D for OpenVLA-OFT action interface.
                    action = (action + [0.0] * 7)[:7]

                    feature = {
                        "observation/image": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image_bytes])
                        ),
                        "action": tf.train.Feature(
                            float_list=tf.train.FloatList(value=action)
                        ),
                        "language_instruction": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[step["language_instruction"].encode("utf-8")]
                            )
                        ),
                        "reward": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[step.get("reward", 0.0)])
                        ),
                        "is_first": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(step.get("is_first", False))]
                            )
                        ),
                        "is_last": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(step.get("is_last", False))]
                            )
                        ),
                        "is_terminal": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(step.get("is_terminal", False))]
                            )
                        ),
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(example.SerializeToString())

        logger.info("Wrote TFRecord split '%s' to %s", split_name, tfrecord_path)

    # Write minimal dataset_info.json
    info = {
        "name": dataset_name,
        "version": "1.0.0",
        "description": f"RLDS dataset generated by BlueprintValidation pipeline from {dataset_name}",
        "splits": {},
    }
    for split_name in ("train", "eval"):
        split_dir = ds_root / split_name
        if split_dir.exists():
            info["splits"][split_name] = {"name": split_name}
    (ds_root / "dataset_info.json").write_text(json.dumps(info, indent=2))

    logger.info("RLDS TFRecord dataset written to %s", ds_root)
    return dataset_root
