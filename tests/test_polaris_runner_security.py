"""Security tests for the PolaRiS scene package bridge."""

from __future__ import annotations

from pathlib import Path

import pytest


class _NoopPolicyClient:
    def infer(self, _request):
        return {"action": []}


def test_scene_package_bridge_requires_unsafe_opt_in(sample_config, tmp_path: Path, monkeypatch) -> None:
    from blueprint_validation.polaris.runner import _evaluate_scene_package_bridge
    from blueprint_validation.polaris.runtime import PolarisSceneSpec

    monkeypatch.delenv("BLUEPRINT_UNSAFE_ALLOW_SCENE_PACKAGE_IMPORT", raising=False)
    scene_root = tmp_path / "scene_pkg"
    scene_root.mkdir(parents=True, exist_ok=True)

    scene_spec = PolarisSceneSpec(
        mode="scene_package_bridge",
        primary_eligible=True,
        environment_name=None,
        scene_root=scene_root,
        usd_path=None,
        initial_conditions_path=None,
        task_metadata_path=None,
        instruction="test",
        detail="scene_package_bridge",
    )

    with pytest.raises(RuntimeError, match="disabled by default"):
        _evaluate_scene_package_bridge(
            config=sample_config,
            scene_spec=scene_spec,
            candidate_label="adapted_openvla",
            output_dir=tmp_path / "out",
            client=_NoopPolicyClient(),
        )
