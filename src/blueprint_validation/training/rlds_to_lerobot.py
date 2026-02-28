"""Convert Stage-4 RLDS-style JSONL episodes into a LeRobot-like dataset layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

from ..common import get_logger, write_json

logger = get_logger("training.rlds_to_lerobot")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _normalize_action(action, target_dim: int) -> list[float]:
    values = [float(v) for v in (action or [])]
    if len(values) < target_dim:
        values.extend([0.0] * (target_dim - len(values)))
    return values[:target_dim]


def _discover_splits(source_dataset_dir: Path) -> Dict[str, Path]:
    direct = source_dataset_dir / "episodes.jsonl"
    if direct.exists():
        return {"train": direct}

    discovered: Dict[str, Path] = {}
    for split in ("train", "eval", "heldout"):
        candidate = source_dataset_dir / split / "episodes.jsonl"
        if candidate.exists():
            discovered[split] = candidate
    return discovered


def _map_image_slots(image_path: str, profile: str) -> Dict[str, str]:
    mapped = {
        "observation/image": image_path,
        "observation/wrist_image": image_path,
    }
    if profile == "pi05_droid":
        mapped["observation/right_wrist_image"] = image_path
    return mapped


def convert_rlds_to_lerobot_dataset(
    source_dataset_dir: Path,
    output_root: Path,
    dataset_name: str,
    profile: str,
    policy_state_dim: int,
    policy_action_dim: int,
) -> Path:
    """Convert RLDS-style JSONL episodes to a LeRobot-like JSONL schema.

    The converter intentionally keeps image references as paths to avoid copying
    large frame directories. It writes converted JSONL files under:
      output_root/dataset_name/<split>/episodes.jsonl
    """

    if not source_dataset_dir.exists():
        raise RuntimeError(f"Source dataset directory does not exist: {source_dataset_dir}")

    splits = _discover_splits(source_dataset_dir)
    if not splits:
        raise RuntimeError(
            "No episodes.jsonl found in source dataset directory. "
            f"Expected {source_dataset_dir}/episodes.jsonl or split subdirectories."
        )

    target_root = output_root / dataset_name
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_steps = 0
    split_stats: Dict[str, Dict[str, int]] = {}

    for split_name, split_jsonl in splits.items():
        out_dir = target_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_dir / "episodes.jsonl"

        episodes_written = 0
        steps_written = 0
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for payload in _iter_jsonl(split_jsonl):
                raw_steps = payload.get("steps") or []
                if not raw_steps:
                    continue

                converted_steps = []
                task_text = str(payload.get("task") or "")
                for step in raw_steps:
                    obs = step.get("observation") or {}
                    image_path = str(obs.get("image_path") or "")
                    if not image_path:
                        raise RuntimeError(
                            f"Missing observation.image_path in episode {payload.get('episode_id')}"
                        )
                    if not Path(image_path).exists():
                        raise RuntimeError(
                            f"Missing frame image referenced by dataset: {image_path}"
                        )

                    action = _normalize_action(step.get("action"), policy_action_dim)
                    state = [0.0] * policy_state_dim
                    prompt = str(step.get("language_instruction") or task_text)

                    converted_steps.append(
                        {
                            **_map_image_slots(image_path, profile),
                            "observation/state": state,
                            "actions": action,
                            "prompt": prompt,
                            "is_first": bool(step.get("is_first", False)),
                            "is_last": bool(step.get("is_last", False)),
                            "is_terminal": bool(step.get("is_terminal", False)),
                            "reward": float(step.get("reward", 0.0)),
                        }
                    )

                if not converted_steps:
                    continue

                converted_episode = {
                    "episode_id": payload.get("episode_id"),
                    "task": task_text,
                    "steps": converted_steps,
                    "source_format": "rlds_style_jsonl",
                }
                handle.write(json.dumps(converted_episode) + "\n")
                episodes_written += 1
                steps_written += len(converted_steps)

        split_stats[split_name] = {
            "episodes": episodes_written,
            "steps": steps_written,
        }
        total_episodes += episodes_written
        total_steps += steps_written

    if total_episodes == 0:
        raise RuntimeError(
            f"Conversion produced zero episodes from source dataset at {source_dataset_dir}"
        )

    summary = {
        "source_dataset_dir": str(source_dataset_dir),
        "output_dataset_dir": str(target_root),
        "dataset_name": dataset_name,
        "profile": profile,
        "policy_state_dim": policy_state_dim,
        "policy_action_dim": policy_action_dim,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "splits": split_stats,
        "schema": {
            "step_keys": [
                "observation/image",
                "observation/wrist_image",
                "observation/state",
                "actions",
                "prompt",
            ],
            "profile_specific_keys": (
                ["observation/right_wrist_image"] if profile == "pi05_droid" else []
            ),
        },
    }
    write_json(summary, target_root / "conversion_summary.json")
    logger.info(
        "Converted RLDS dataset %s -> %s (%d episodes)",
        source_dataset_dir,
        target_root,
        total_episodes,
    )
    return target_root


def normalize_action_dim(action, target_dim: int) -> list[float]:
    """Public helper for tests."""
    return _normalize_action(action, target_dim)


def infer_split_layout(source_dataset_dir: Path) -> Tuple[str, ...]:
    """Public helper for tests."""
    return tuple(sorted(_discover_splits(source_dataset_dir).keys()))
