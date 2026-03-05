from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _write_frame(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _prepare_claim_eval_workspace(sample_config, tmp_path: Path) -> tuple[Path, Path]:
    from blueprint_validation.common import write_json
    from blueprint_validation.evaluation.claim_protocol import checkpoint_content_hash

    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.eval_policy.primary_endpoint = "task_success"
    sample_config.eval_policy.freeze_world_snapshot = True
    sample_config.eval_policy.min_practical_success_lift_pp = 0.0
    sample_config.eval_policy.claim_replication.training_seeds = [0]
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
    write_json(
        {
            "eval_eval_cell_ids": ["eval_cell_001"],
            "train_eval_cell_ids": [],
            "cells": [
                {
                    "eval_cell_id": "eval_cell_001",
                    "task_spec_id": "task_spec_pick",
                    "task_prompt": "Pick up bowl_101 and place it in the target zone",
                    "start_region_id": "manipulation:101",
                    "world_snapshot_hash": world_hash,
                }
            ],
        },
        policy_eval_dir / "claim_split_manifest.json",
    )
    write_json(
        {
            "claim_protocol": "fixed_same_facility_uplift",
            "world_snapshot_hash": world_hash,
            "task_specs_path": str(policy_eval_dir / "task_specs.json"),
            "split_manifest_path": str(policy_eval_dir / "claim_split_manifest.json"),
        },
        policy_eval_dir / "claim_manifest.json",
    )

    episode_payload = {
        "episode_id": "episode_0000",
        "task": "Pick up bowl_101 and place it in the target zone",
        "eval_cell_id": "eval_cell_001",
        "task_spec_id": "task_spec_pick",
        "start_region_id": "manipulation:101",
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
    (dataset_root / "episodes.jsonl").write_text(json.dumps(episode_payload) + "\n")

    pair_train_dir = work_dir / "policy_pair_train"
    pair_train_dir.mkdir(parents=True, exist_ok=True)
    generic_ckpt = pair_train_dir / "generic_seed_0"
    site_ckpt = pair_train_dir / "site_seed_0"
    generic_ckpt.mkdir(parents=True, exist_ok=True)
    site_ckpt.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "policy_base": {
                "status": "success",
                "adapted_checkpoint_path": str(generic_ckpt),
                "elapsed_seconds": 1.0,
                "detail": "",
            },
            "policy_site": {
                "status": "success",
                "adapted_checkpoint_path": str(site_ckpt),
                "elapsed_seconds": 1.0,
                "detail": "",
            },
            "replicates": {
                "generic_control": [
                    {
                        "seed": 0,
                        "status": "success",
                        "adapted_checkpoint_path": str(generic_ckpt),
                        "elapsed_seconds": 1.0,
                        "detail": "",
                    }
                ],
                "site_trained": [
                    {
                        "seed": 0,
                        "status": "success",
                        "adapted_checkpoint_path": str(site_ckpt),
                        "elapsed_seconds": 1.0,
                        "detail": "",
                    }
                ],
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
    assert "World snapshot hash drift" in result.detail


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
    assert result.metrics["claim_passed"] is True
    assert Path(result.outputs["claim_report_path"]).exists()
