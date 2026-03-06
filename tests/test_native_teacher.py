from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write_tiny_video(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(6):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


def test_generate_teacher_rollouts_uses_train_split_assignments(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.common import write_json
    from blueprint_validation.training.native_teacher import generate_teacher_rollouts

    video_path = tmp_path / "renders" / "clip_000.mp4"
    _write_tiny_video(video_path)

    benchmark_path = tmp_path / "claim_benchmark.json"
    benchmark_payload = {
        "version": 1,
        "task_specs": [
            {
                "task_spec_id": "task_spec_pick",
                "task_prompt": "Pick up bowl_101 and place it in the target zone",
                "task_family": "manipulation",
                "target_instance_id": "101",
                "target_label": "bowl",
                "success_predicate": {
                    "type": "manipulation_pick_place_stable",
                    "target_instance_id": "101",
                    "goal_region_id": "target_zone",
                    "target_center_xyz": [0.35, 0.0, 0.65],
                    "goal_point_xyz": [0.70, 0.0, 0.80],
                },
            }
        ],
        "assignments": [
            {
                "rollout_index": 0,
                "task_spec_id": "task_spec_pick",
                "clip_name": "clip_000",
                "clip_index": 0,
                "start_clip_id": "clip_000",
                "start_region_id": "manipulation:101",
                "target_instance_id": "101",
                "target_label": "bowl",
            }
        ],
    }
    benchmark_path.write_text(json.dumps(benchmark_payload))
    sample_config.facilities["test_facility"].claim_benchmark_path = benchmark_path

    work_dir = tmp_path / "outputs" / "test_facility"
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "clips": [
                {
                    "clip_name": "clip_000",
                    "clip_index": 0,
                    "path_type": "manipulation",
                    "video_path": str(video_path),
                    "initial_camera": {
                        "position": [0.0, 0.0, 0.8],
                        "forward": [1.0, 0.0, 0.0],
                        "right": [0.0, 1.0, 0.0],
                        "up": [0.0, 0.0, 1.0],
                    },
                    "path_context": {"approach_point": [0.35, 0.0, 0.65]},
                }
            ]
        },
        render_dir / "render_manifest.json",
    )
    policy_eval_dir = work_dir / "policy_eval"
    policy_eval_dir.mkdir(parents=True, exist_ok=True)
    write_json({"claim_protocol": "fixed_same_facility_uplift"}, policy_eval_dir / "claim_manifest.json")
    write_json(
        {
            "world_snapshot_hash": "world_hash",
            "train_eval_cell_ids": ["eval_cell_train_000"],
            "eval_eval_cell_ids": [],
            "cells": [
                {
                    "eval_cell_id": "eval_cell_train_000",
                    "task_spec_id": "task_spec_pick",
                    "start_clip_id": "clip_000",
                    "start_region_id": "manipulation:101",
                    "rollout_index": 0,
                }
            ],
        },
        policy_eval_dir / "claim_split_manifest.json",
    )

    monkeypatch.setattr(
        "blueprint_validation.training.native_teacher.load_dreamdojo_world_model",
        lambda **_kwargs: object(),
    )

    def _fake_rollout(**kwargs):
        video = kwargs["output_dir"] / f"{kwargs['clip_name']}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"x")
        return SimpleNamespace(
            video_path=video,
            num_steps=len(kwargs["action_sequence"]),
            action_sequence=list(kwargs["action_sequence"]),
            state_trace=[
                {"grasp_acquired": True, "step_idx": 1},
                {"lifted_clear": True, "step_idx": 2},
                {"placed_in_target": True, "step_idx": 3},
                {"stable_after_place": True, "step_idx": 4},
            ],
        )

    monkeypatch.setattr(
        "blueprint_validation.training.native_teacher.run_scripted_rollout",
        _fake_rollout,
    )

    rows, summary = generate_teacher_rollouts(
        config=sample_config,
        facility=sample_config.facilities["test_facility"],
        work_dir=work_dir,
        output_dir=work_dir / "native_teacher_test",
        condition="adapted",
        mode="site",
        max_steps=12,
    )

    assert summary["num_assignments"] == 1
    assert rows
    assert all(row["source_type"] == "planner_demo" for row in rows)
    assert all(row["success_source"] == "sim_ground_truth" for row in rows)
    assert all(row["eval_cell_id"] == "eval_cell_train_000" for row in rows)
