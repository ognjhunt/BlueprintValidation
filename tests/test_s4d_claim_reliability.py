from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_s4d_claim_protocol_fails_without_native_task_state(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage
    from tests.test_s4d_claim_protocol import _prepare_claim_eval_workspace

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(sample_config, tmp_path)

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
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.run_rollout_with_adapter",
        lambda **kwargs: SimpleNamespace(
            video_path=kwargs["output_dir"] / f"{kwargs['clip_name']}.mp4",
            action_sequence=[[0.0] * 7 for _ in range(2)],
            num_steps=2,
            state_trace=[],
            action_contract={
                "policy_dim": 7,
                "world_dim": 7,
                "dataset_dim": 7,
                "compliant": True,
                "reason": "",
            },
        ),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval._read_rgb_image",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval._score_rollout_video",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=8.0,
            spatial_score=8.0,
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
        ),
    )

    def _write_video_path(**kwargs):
        path = kwargs["output_dir"] / f"{kwargs['clip_name']}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(
        "blueprint_validation.stages.s4d_policy_pair_eval.run_rollout_with_adapter",
        lambda **kwargs: SimpleNamespace(
            video_path=_write_video_path(**kwargs),
            action_sequence=[[0.0] * 7 for _ in range(2)],
            num_steps=2,
            state_trace=[],
            action_contract={
                "policy_dim": 7,
                "world_dim": 7,
                "dataset_dim": 7,
                "compliant": True,
                "reason": "",
            },
        ),
    )

    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert result.metrics["claim_outcome"] == "INCONCLUSIVE"
    assert result.metrics["num_state_failures"] > 0
    assert result.metrics["headline_eligible"] is False


def test_s4d_claim_protocol_fails_on_reliability_gate(sample_config, tmp_path, monkeypatch):
    from blueprint_validation.stages.s4d_policy_pair_eval import PolicyPairEvalStage
    from tests.test_s4d_claim_protocol import _prepare_claim_eval_workspace

    work_dir, _adapted_ckpt = _prepare_claim_eval_workspace(sample_config, tmp_path)
    sample_config.eval_policy.reliability.min_replay_pass_rate = 1.0
    sample_config.eval_policy.reliability.min_controllability_pass_rate = 1.0

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

    def _fake_run_rollout_with_adapter(**kwargs):
        path = kwargs["output_dir"] / f"{kwargs['clip_name']}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return SimpleNamespace(
            video_path=path,
            action_sequence=[[0.0] * 7 for _ in range(2)],
            num_steps=2,
            state_trace=[
                {"grasp_acquired": True},
                {"lifted_clear": True},
                {"placed_in_target": True},
                {"stable_after_place": True},
            ],
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
        "blueprint_validation.stages.s4d_policy_pair_eval._score_rollout_video",
        lambda **_kwargs: SimpleNamespace(
            task_score=8.0,
            visual_score=0.0,
            spatial_score=0.0,
            grasp_acquired=True,
            lifted_clear=True,
            placed_in_target=True,
            stable_after_place=True,
        ),
    )

    fac = sample_config.facilities["test_facility"]
    result = PolicyPairEvalStage().run(sample_config, fac, work_dir, {})
    assert result.status == "failed"
    assert result.metrics["claim_outcome"] == "INCONCLUSIVE"
    assert result.metrics["reliability_gate"]["passed"] is False
    assert result.metrics["headline_eligible"] is False
