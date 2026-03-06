from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _write_frame(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _prepare_claim_eval_workspace(
    sample_config,
    tmp_path: Path,
    *,
    training_seeds: list[int] | None = None,
    num_eval_cells: int = 1,
) -> tuple[Path, Path]:
    from blueprint_validation.common import write_json
    from blueprint_validation.evaluation.claim_protocol import checkpoint_content_hash
    from blueprint_validation.stages.s4b_rollout_dataset import _json_manifest_hash

    training_seeds = training_seeds or [0, 1, 2, 3, 4, 5]
    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.eval_policy.primary_endpoint = "task_success"
    sample_config.eval_policy.freeze_world_snapshot = True
    sample_config.eval_policy.min_practical_success_lift_pp = 0.0
    sample_config.eval_policy.split_strategy = "disjoint_tasks_and_starts"
    sample_config.eval_policy.claim_replication.training_seeds = training_seeds
    sample_config.eval_policy.claim_strictness.min_eval_task_specs = 1
    sample_config.eval_policy.claim_strictness.min_eval_start_clips = 1
    sample_config.eval_policy.claim_strictness.min_common_eval_cells = 1
    sample_config.eval_policy.claim_strictness.min_positive_training_seeds = 1
    sample_config.eval_policy.reliability.min_replay_pass_rate = 0.0
    sample_config.eval_policy.reliability.min_controllability_pass_rate = 0.0
    sample_config.eval_policy.reliability.max_scoring_failure_rate = 1.0
    sample_config.policy_compare.enabled = True
    sample_config.policy_compare.control_arms = [
        "frozen_baseline",
        "site_trained",
        "generic_control",
    ]

    work_dir = tmp_path / "outputs" / "test_facility"
    dataset_root = sample_config.rollout_dataset.export_dir / work_dir.name / "adapted" / "heldout"
    dataset_root.mkdir(parents=True, exist_ok=True)
    frame_path = _write_frame(dataset_root / "frames" / "episode_0000" / "0000.jpg")

    adapted_ckpt = work_dir / "finetune" / "adapted_checkpoint"
    adapted_ckpt.mkdir(parents=True, exist_ok=True)
    (adapted_ckpt / "weights.bin").write_bytes(b"world")
    world_hash = checkpoint_content_hash(adapted_ckpt)

    policy_eval_dir = work_dir / "policy_eval"
    policy_eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        [
            {
                "task_spec_id": "task_spec_pick",
                "task_prompt": "Pick up bowl_101 and place it in the target zone",
                "task_family": "manipulation",
                "success_predicate": {
                    "type": "manipulation_pick_place_stable",
                    "target_instance_id": "101",
                    "goal_region_id": "target_zone",
                    "require_stable_after_place": True,
                },
            }
        ],
        policy_eval_dir / "task_specs.json",
    )
    eval_cells = [
        {
            "eval_cell_id": f"eval_cell_{idx + 1:03d}",
            "task_spec_id": "task_spec_pick",
            "task_prompt": "Pick up bowl_101 and place it in the target zone",
            "start_clip_id": f"clip_{idx:03d}",
            "start_region_id": f"manipulation:{100 + idx}",
            "start_frame_hash": f"frame_hash_{idx:03d}",
            "world_snapshot_hash": world_hash,
        }
        for idx in range(num_eval_cells)
    ]
    write_json(
        {
            "eval_eval_cell_ids": [cell["eval_cell_id"] for cell in eval_cells],
            "train_eval_cell_ids": [],
            "cells": eval_cells,
        },
        policy_eval_dir / "claim_split_manifest.json",
    )
    write_json(
        {
            "claim_protocol": "fixed_same_facility_uplift",
            "primary_endpoint": "task_success",
            "world_snapshot_hash": world_hash,
            "task_specs_path": str(policy_eval_dir / "task_specs.json"),
            "split_manifest_path": str(policy_eval_dir / "claim_split_manifest.json"),
        },
        policy_eval_dir / "claim_manifest.json",
    )

    episode_payloads = [
        {
            "episode_id": f"episode_{idx:04d}",
            "task": "Pick up bowl_101 and place it in the target zone",
            "eval_cell_id": cell["eval_cell_id"],
            "task_spec_id": "task_spec_pick",
            "start_clip_id": cell["start_clip_id"],
            "start_region_id": cell["start_region_id"],
            "start_frame_hash": cell["start_frame_hash"],
            "world_snapshot_hash": world_hash,
            "steps": [
                {
                    "observation": {"image_path": str(frame_path)},
                    "action": [0.0] * 7,
                    "language_instruction": "Pick up bowl_101 and place it in the target zone",
                    "reward": 0.0,
                    "is_first": True,
                    "is_last": True,
                    "is_terminal": True,
                }
            ],
        }
        for idx, cell in enumerate(eval_cells)
    ]
    (dataset_root / "episodes.jsonl").write_text(
        "\n".join(json.dumps(payload) for payload in episode_payloads) + "\n"
    )

    pair_train_dir = work_dir / "policy_pair_train"
    pair_train_dir.mkdir(parents=True, exist_ok=True)
    generic_runs = []
    site_runs = []
    for seed in training_seeds:
        generic_ckpt = pair_train_dir / f"generic_seed_{seed}"
        site_ckpt = pair_train_dir / f"site_seed_{seed}"
        generic_ckpt.mkdir(parents=True, exist_ok=True)
        site_ckpt.mkdir(parents=True, exist_ok=True)
        generic_runs.append(
            {
                "seed": seed,
                "status": "success",
                "adapted_checkpoint_path": str(generic_ckpt),
                "elapsed_seconds": 1.0,
                "detail": "",
            }
        )
        site_runs.append(
            {
                "seed": seed,
                "status": "success",
                "adapted_checkpoint_path": str(site_ckpt),
                "elapsed_seconds": 1.0,
                "detail": "",
            }
        )
    write_json(
        {
            "policy_base": generic_runs[0],
            "policy_site": site_runs[0],
            "dataset_lineage": {
                "world_snapshot_hash": world_hash,
                "claim_manifest_hash": _json_manifest_hash(policy_eval_dir / "claim_manifest.json"),
                "claim_split_manifest_hash": _json_manifest_hash(
                    policy_eval_dir / "claim_split_manifest.json"
                ),
                "train_eval_cell_ids_hash": "",
                "heldout_eval_cell_ids_hash": "",
            },
            "replicates": {
                "generic_control": generic_runs,
                "site_trained": site_runs,
            },
        },
        pair_train_dir / "policy_pair_train_summary.json",
    )
    return work_dir, adapted_ckpt


def test_s4d_claim_protocol_fails_on_world_hash_drift(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(sample_config, tmp_path)
    write_json(
        {
            "claim_protocol": "fixed_same_facility_uplift",
            "world_snapshot_hash": "mismatch",
            "task_specs_path": str(work_dir / "policy_eval" / "task_specs.json"),
            "split_manifest_path": str(work_dir / "policy_eval" / "claim_split_manifest.json"),
        },
        work_dir / "policy_eval" / "claim_manifest.json",
    )
    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert "world snapshot hash" in result.detail.lower()


def test_s4d_claim_protocol_emits_claim_report(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(sample_config, tmp_path)

    class _FakeAdapter:
        def base_model_ref(self, _eval_policy):
            return "fake_model", None

        def load_policy(self, *, model_name, checkpoint_path, device):
            return {"checkpoint": str(checkpoint_path or ""), "model_name": model_name}

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.get_policy_adapter",
        lambda *_args, **_kwargs: _FakeAdapter(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.load_dreamdojo_world_model",
        lambda **_kwargs: object(),
    )

    def _fake_run_rollout_with_adapter(**kwargs):
        clip_name = kwargs["clip_name"]
        video_path = kwargs["output_dir"] / f"{clip_name}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x")
        if "policy_site" in clip_name:
            state_trace = [
                {"grasp_acquired": True},
                {"lifted_clear": True},
                {"placed_in_target": True},
                {"stable_after_place": True},
            ]
        else:
            state_trace = [
                {"grasp_acquired": False},
                {"lifted_clear": False},
                {"placed_in_target": False},
                {"stable_after_place": False},
            ]
        return SimpleNamespace(
            video_path=video_path,
            action_sequence=[[0.0] * 7 for _ in range(2)],
            num_steps=2,
            state_trace=state_trace,
            action_contract={
                "policy_dim": 7,
                "world_dim": 7,
                "dataset_dim": 7,
                "compliant": True,
                "reason": "",
            },
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.run_rollout_with_adapter",
        _fake_run_rollout_with_adapter,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval._read_rgb_image",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.score_rollout_manipulation",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
            reasoning="ok",
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.score_rollout",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            reasoning="ok",
        ),
    )

    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "success"
    assert result.metrics["claim_protocol"] == "fixed_same_facility_uplift"
    assert result.metrics["claim_outcome"] == "PASS"
    assert result.metrics["claim_passed"] is True
    assert Path(result.outputs["claim_report_path"]).exists()


def test_s4d_claim_protocol_fails_when_heldout_eval_cells_are_missing(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(
        sample_config,
        tmp_path,
        num_eval_cells=2,
    )
    heldout_path = (
        sample_config.rollout_dataset.export_dir
        / work_dir.name
        / "adapted"
        / "heldout"
        / "episodes.jsonl"
    )
    lines = heldout_path.read_text().splitlines()
    heldout_path.write_text(lines[0] + "\n")

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.get_policy_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_model_ref=lambda _eval_policy: ("fake_model", None),
            load_policy=lambda **_kwargs: {"checkpoint": "ok"},
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.load_dreamdojo_world_model",
        lambda **_kwargs: object(),
    )

    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert "missing registered eval cells" in result.detail


def test_s4d_claim_protocol_fails_when_generic_control_seed_is_missing(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.common import read_json, write_json
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(sample_config, tmp_path)
    summary_path = work_dir / "policy_pair_train" / "policy_pair_train_summary.json"
    payload = read_json(summary_path)
    payload["replicates"]["generic_control"] = payload["replicates"]["generic_control"][:-1]
    write_json(payload, summary_path)
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.get_policy_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_model_ref=lambda _eval_policy: ("fake_model", None),
            load_policy=lambda **_kwargs: {"checkpoint": "ok"},
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.load_dreamdojo_world_model",
        lambda **_kwargs: object(),
    )

    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert "generic_control" in result.detail
    assert "seeds" in result.detail


def test_s4d_claim_protocol_marks_low_sample_claim_inconclusive(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(
        sample_config,
        tmp_path,
        num_eval_cells=2,
    )
    sample_config.eval_policy.claim_strictness.min_eval_task_specs = 3
    sample_config.eval_policy.claim_strictness.min_eval_start_clips = 3
    sample_config.eval_policy.claim_strictness.min_common_eval_cells = 30

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.get_policy_adapter",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_model_ref=lambda _eval_policy: ("fake_model", None),
            load_policy=lambda **_kwargs: {"checkpoint": "ok"},
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.load_dreamdojo_world_model",
        lambda **_kwargs: object(),
    )

    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert result.metrics["claim_outcome"] == "INCONCLUSIVE"
    assert result.metrics["claim_passed"] is False


def test_s4d_claim_protocol_marks_site_vs_generic_tie_inconclusive(
    sample_config, tmp_path, monkeypatch
):
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(sample_config, tmp_path)

    class _FakeAdapter:
        def base_model_ref(self, _eval_policy):
            return "fake_model", None

        def load_policy(self, *, model_name, checkpoint_path, device):
            return {"checkpoint": str(checkpoint_path or ""), "model_name": model_name}

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.get_policy_adapter",
        lambda *_args, **_kwargs: _FakeAdapter(),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.load_dreamdojo_world_model",
        lambda **_kwargs: object(),
    )

    def _fake_run_rollout_with_adapter(**kwargs):
        clip_name = kwargs["clip_name"]
        video_path = kwargs["output_dir"] / f"{clip_name}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"x")
        if "frozen_baseline" in clip_name:
            state_trace = [
                {"grasp_acquired": False},
                {"lifted_clear": False},
                {"placed_in_target": False},
                {"stable_after_place": False},
            ]
        else:
            state_trace = [
                {"grasp_acquired": True},
                {"lifted_clear": True},
                {"placed_in_target": True},
                {"stable_after_place": True},
            ]
        return SimpleNamespace(
            video_path=video_path,
            action_sequence=[[0.0] * 7 for _ in range(2)],
            num_steps=2,
            state_trace=state_trace,
            action_contract={
                "policy_dim": 7,
                "world_dim": 7,
                "dataset_dim": 7,
                "compliant": True,
                "reason": "",
            },
        )

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.run_rollout_with_adapter",
        _fake_run_rollout_with_adapter,
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval._read_rgb_image",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.score_rollout_manipulation",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
            reasoning="ok",
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.score_rollout",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            reasoning="ok",
        ),
    )

    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "success"
    assert result.metrics["claim_outcome"] == "INCONCLUSIVE"
    assert result.metrics["claim_passed"] is False
