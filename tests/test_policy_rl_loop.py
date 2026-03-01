"""Tests for policy RL loop helper logic."""

from __future__ import annotations

import pytest


def test_select_rollouts_returns_top_and_near_miss():
    from blueprint_validation.training.policy_rl_loop import _select_rollouts

    rows = []
    for i, reward in enumerate([0.9, 0.8, 0.6, 0.5, 0.3, 0.1]):
        rows.append({"task": "pick", "rl_reward": reward, "rollout_index": i})

    selected, near_miss, hard_negative = _select_rollouts(
        rollout_rows=rows,
        group_size=2,
        top_quantile=0.30,
        near_miss_min_quantile=0.30,
        near_miss_max_quantile=0.60,
    )
    assert selected
    assert near_miss
    assert hard_negative
    assert all(r["task"] == "pick" for r in selected)
    assert all("advantage" in r for r in selected + near_miss + hard_negative)


def test_build_policy_refine_mix_respects_fractions():
    from blueprint_validation.training.policy_rl_loop import _build_policy_refine_mix

    selected = [{"rollout_index": i, "task": "pick"} for i in range(10)]
    near = [{"rollout_index": 100 + i, "task": "pick"} for i in range(10)]
    hard = [{"rollout_index": 200 + i, "task": "pick"} for i in range(10)]

    mixed, metrics = _build_policy_refine_mix(
        selected_success=selected,
        near_miss=near,
        hard_negative=hard,
        near_miss_fraction=0.3,
        hard_negative_fraction=0.1,
        seed=17,
    )
    assert len(mixed) == 14
    assert metrics["selected_success"] == 10
    assert metrics["selected_near_miss"] == 3
    assert metrics["selected_hard_negative"] == 1


def test_score_rollout_reward_heuristic_only(sample_config, tmp_path):
    from blueprint_validation.training.policy_rl_loop import _score_rollout_reward

    sample_config.policy_rl_loop.reward_mode = "heuristic_only"
    sample_config.policy_rl_loop.vlm_reward_fraction = 0.0
    fac = list(sample_config.facilities.values())[0]
    video_path = tmp_path / "dummy.mp4"
    video_path.write_bytes(b"x")

    result = _score_rollout_reward(
        config=sample_config,
        facility=fac,
        task="Pick up the tote",
        rollout_index=0,
        video_path=video_path,
        action_sequence=[[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
    )
    assert result["vlm_reward"] == 0.0
    assert result["heuristic_reward"] > 0.0
    assert result["rl_reward"] > 0.0


def test_policy_rl_stage_skips_when_disabled(sample_config, tmp_path):
    from blueprint_validation.stages.s3c_policy_rl_loop import PolicyRLLoopStage

    sample_config.policy_rl_loop.enabled = False
    fac = list(sample_config.facilities.values())[0]
    stage = PolicyRLLoopStage()
    result = stage.run(sample_config, fac, tmp_path, {})
    assert result.status == "skipped"


def test_refine_policy_returns_failure_when_training_does_not_produce_checkpoint(
    sample_config, tmp_path
):
    from blueprint_validation.policy_adapters.base import PolicyTrainingResult
    from blueprint_validation.training.policy_rl_loop import _refine_policy

    class _DummyAdapter:
        def dataset_transform(self, source_dataset_dir, output_root, dataset_name):
            del source_dataset_dir
            out = output_root / dataset_name
            out.mkdir(parents=True, exist_ok=True)
            return out

        def train_policy(
            self,
            base_model_name,
            base_checkpoint,
            dataset_root,
            dataset_name,
            output_dir,
            finetune_config,
        ):
            del (
                base_model_name,
                base_checkpoint,
                dataset_root,
                dataset_name,
                output_dir,
                finetune_config,
            )
            return PolicyTrainingResult(
                status="failed",
                adapted_checkpoint_path=None,
                elapsed_seconds=0.0,
                detail="intentional failure",
            )

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    current_ckpt = tmp_path / "policy_ckpt"
    current_ckpt.mkdir(parents=True, exist_ok=True)

    ckpt, ok, detail = _refine_policy(
        config=sample_config,
        policy_adapter=_DummyAdapter(),
        base_model_name="dummy/model",
        base_checkpoint=None,
        source_dataset_dir=src_dir,
        dataset_name="dummy_dataset",
        iteration=0,
        output_dir=tmp_path / "out",
        current_policy_checkpoint=current_ckpt,
    )
    assert ckpt == current_ckpt
    assert ok is False
    assert "intentional failure" in detail


def test_run_policy_rl_iterations_fails_fast_when_policy_refine_fails_in_full_mode(
    sample_config, tmp_path, monkeypatch
):
    import numpy as np

    from blueprint_validation.training import policy_rl_loop as rl_mod

    class _DummyAdapter:
        def base_model_ref(self, eval_config):
            del eval_config
            return "dummy/model", None

        def load_policy(self, model_name, checkpoint_path, device):
            del model_name, checkpoint_path, device
            return object()

    sample_config.action_boost.enabled = True
    sample_config.action_boost.require_full_pipeline = True
    sample_config.policy_rl_loop.enabled = True
    sample_config.policy_rl_loop.iterations = 1
    sample_config.policy_rl_loop.rollouts_per_task = 1
    sample_config.eval_policy.tasks = ["Navigate to the counter"]
    fac = list(sample_config.facilities.values())[0]

    render_manifest = tmp_path / "renders" / "render_manifest.json"
    render_manifest.parent.mkdir(parents=True, exist_ok=True)
    render_manifest.write_text('{"clips":[]}')

    monkeypatch.setattr(rl_mod, "_resolve_render_manifest", lambda work_dir: render_manifest)
    monkeypatch.setattr(
        rl_mod,
        "_extract_initial_frames",
        lambda manifest: [np.zeros((16, 16, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(rl_mod, "_build_task_list", lambda config: ["Navigate to the counter"])
    monkeypatch.setattr(rl_mod, "_has_cuda", lambda: False)
    monkeypatch.setattr(rl_mod, "get_policy_adapter", lambda cfg: _DummyAdapter())
    monkeypatch.setattr(rl_mod, "load_dreamdojo_world_model", lambda **kwargs: object())
    monkeypatch.setattr(
        rl_mod,
        "_collect_rollouts",
        lambda **kwargs: [
            {
                "condition": "adapted",
                "task": "Navigate to the counter",
                "rollout_index": 0,
                "video_path": str(tmp_path / "rollout.mp4"),
                "num_steps": 10,
                "action_sequence": [[0.0] * 7 for _ in range(10)],
                "is_manipulation_task": False,
                "task_score": 5.0,
                "rl_reward": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        rl_mod,
        "_export_selected_rollouts",
        lambda **kwargs: (tmp_path / "dataset", tmp_path / "manifest.json"),
    )
    monkeypatch.setattr(
        rl_mod,
        "_refine_policy",
        lambda **kwargs: (None, False, "simulated refine failure"),
    )

    with pytest.raises(RuntimeError, match="Policy refine failed in RL loop iteration"):
        rl_mod.run_policy_rl_iterations(
            config=sample_config,
            facility=fac,
            work_dir=tmp_path,
            output_dir=tmp_path / "rl",
            initial_policy_checkpoint=None,
            adapted_world_checkpoint=tmp_path,
        )
