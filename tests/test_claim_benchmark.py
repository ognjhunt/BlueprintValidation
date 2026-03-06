import json

from blueprint_validation.evaluation.claim_benchmark import (
    claim_benchmark_alignment_failures,
    claim_benchmark_strictness_failures,
    load_pinned_claim_benchmark,
)
from blueprint_validation.evaluation.claim_protocol import build_claim_split_payload


def test_load_pinned_claim_benchmark_hydrates_assignments(tmp_path):
    benchmark_path = tmp_path / "claim_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task_specs": [
                    {
                        "task_spec_id": "task_spec_nav_sink",
                        "task_prompt": "Navigate to the sink",
                        "task_family": "navigation",
                        "success_predicate": {
                            "type": "navigation_reach_and_hold",
                            "goal_region_id": "region::sink",
                            "hold_steps": 2,
                            "collision_key": "invalid_collision",
                        },
                    }
                ],
                "assignments": [
                    {
                        "rollout_index": 3,
                        "task_spec_id": "task_spec_nav_sink",
                        "clip_name": "clip_001",
                        "start_clip_id": "frozen_start_001",
                        "start_region_id": "nav_start::sink_001",
                    }
                ],
            }
        )
    )
    render_manifest = {
        "clips": [
            {
                "clip_index": 1,
                "clip_name": "clip_001",
                "path_type": "navigation",
                "video_path": str(tmp_path / "clip_001.mp4"),
                "initial_camera": {"position": [0.0, 0.0, 1.0]},
                "path_context": {"approach_point": [0.0, 0.0, 0.5]},
            }
        ]
    }

    benchmark = load_pinned_claim_benchmark(
        benchmark_path=benchmark_path,
        render_manifest=render_manifest,
        video_orientation_fix="rotate180",
    )

    assert benchmark.tasks == ["Navigate to the sink"]
    assert benchmark.assignments[0]["task"] == "Navigate to the sink"
    assert benchmark.assignments[0]["clip_index"] == 1
    assert benchmark.assignments[0]["start_clip_id"] == "frozen_start_001"
    assert benchmark.assignments[0]["start_region_id"] == "nav_start::sink_001"
    assert benchmark.assignments[0]["video_orientation_fix"] == "rotate180"


def test_build_claim_split_payload_uses_pinned_start_ids():
    task_specs = [
        {
            "task_spec_id": "task_spec_a",
            "task_prompt": "Navigate to region A",
            "task_family": "navigation",
        },
        {
            "task_spec_id": "task_spec_b",
            "task_prompt": "Navigate to region B",
            "task_family": "navigation",
        },
    ]
    assignments = [
        {
            "rollout_index": 0,
            "task": "Navigate to region A",
            "clip_index": 10,
            "clip_name": "render_clip_010",
            "start_clip_id": "benchmark_start_a",
            "start_region_id": "region_start::a",
            "path_type": "navigation",
        },
        {
            "rollout_index": 1,
            "task": "Navigate to region B",
            "clip_index": 11,
            "clip_name": "render_clip_011",
            "start_clip_id": "benchmark_start_b",
            "start_region_id": "region_start::b",
            "path_type": "navigation",
        },
        {
            "rollout_index": 2,
            "task": "Navigate to region A",
            "clip_index": 12,
            "clip_name": "render_clip_012",
            "start_clip_id": "benchmark_start_c",
            "start_region_id": "region_start::c",
            "path_type": "navigation",
        },
        {
            "rollout_index": 3,
            "task": "Navigate to region B",
            "clip_index": 13,
            "clip_name": "render_clip_013",
            "start_clip_id": "benchmark_start_d",
            "start_region_id": "region_start::d",
            "path_type": "navigation",
        },
    ]

    split = build_claim_split_payload(
        task_specs=task_specs,
        assignments=assignments,
        world_snapshot_hash="world_hash",
        train_split=0.5,
        split_strategy="disjoint_tasks_and_starts",
    )

    start_ids = {cell["start_clip_id"] for cell in split["cells"]}
    region_ids = {cell["start_region_id"] for cell in split["cells"]}
    assert "benchmark_start_a" in start_ids
    assert "benchmark_start_d" in start_ids
    assert "region_start::a" in region_ids
    assert "region_start::d" in region_ids


def test_claim_benchmark_alignment_failures_report_missing_clip_names(tmp_path):
    benchmark_path = tmp_path / "claim_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task_specs": [
                    {
                        "task_spec_id": "task_spec_a",
                        "task_prompt": "Navigate to region A",
                        "task_family": "navigation",
                    }
                ],
                "assignments": [
                    {
                        "rollout_index": 0,
                        "task_spec_id": "task_spec_a",
                        "clip_name": "clip_999_missing",
                        "start_clip_id": "clip_999_missing",
                        "start_region_id": "region_start::a",
                    }
                ],
            }
        )
    )

    failures = claim_benchmark_alignment_failures(
        benchmark_path=benchmark_path,
        render_manifest={"clips": [{"clip_index": 0, "clip_name": "clip_000"}]},
    )

    assert failures
    assert "clip_999_missing" in failures[0]


def test_claim_benchmark_strictness_failures_report_undersized_benchmark(tmp_path):
    benchmark_path = tmp_path / "claim_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "version": 1,
                "task_specs": [
                    {
                        "task_spec_id": "task_spec_a",
                        "task_prompt": "Navigate to region A",
                        "task_family": "navigation",
                    }
                ],
                "assignments": [
                    {
                        "rollout_index": 0,
                        "task_spec_id": "task_spec_a",
                        "clip_name": "clip_000",
                        "start_clip_id": "clip_000",
                        "start_region_id": "region_start::a",
                    }
                ],
            }
        )
    )
    benchmark = load_pinned_claim_benchmark(
        benchmark_path=benchmark_path,
        render_manifest={"clips": [{"clip_index": 0, "clip_name": "clip_000"}]},
    )

    failures = claim_benchmark_strictness_failures(
        benchmark=benchmark,
        min_eval_task_specs=2,
        min_eval_start_clips=2,
        min_common_eval_cells=4,
    )

    assert len(failures) == 3
    assert "unique_task_specs=1 < min_eval_task_specs=2" in failures[0]
