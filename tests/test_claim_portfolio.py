from __future__ import annotations

import json
from pathlib import Path


def _write_claim_artifacts(root: Path, facility_id: str, *, benchmark_path: Path, benchmark_hash: str, world_hash: str, site_vs_frozen: float, site_vs_generic: float) -> None:
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s4b_rollout_dataset import _json_manifest_hash

    fac_dir = root / facility_id
    policy_eval_dir = fac_dir / "policy_eval"
    policy_eval_dir.mkdir(parents=True, exist_ok=True)
    claim_manifest_path = policy_eval_dir / "claim_manifest.json"
    claim_split_path = policy_eval_dir / "claim_split_manifest.json"
    write_json({"world_snapshot_hash": world_hash}, claim_manifest_path)
    write_json({"world_snapshot_hash": world_hash, "train_eval_cell_ids": ["train"], "eval_eval_cell_ids": ["eval"]}, claim_split_path)
    write_json(
        {
            "stage_name": "s4_policy_eval",
            "status": "success",
            "metrics": {
                "claim_benchmark_path": str(benchmark_path),
                "claim_benchmark_manifest_hash": benchmark_hash,
            },
        },
        fac_dir / "s4_policy_eval_result.json",
    )
    write_json(
        {
            "stage_name": "s4d_policy_pair_eval",
            "status": "success",
            "metrics": {
                "claim_protocol": "fixed_same_facility_uplift",
                "headline_eligible": True,
                "investor_grade_generic_control": True,
                "generic_control_mode": "leave_one_facility_out",
                "world_snapshot_hash": world_hash,
                "bootstrap_site_vs_frozen": {"mean_lift_pp": site_vs_frozen},
                "site_vs_generic_attribution": {"mean_lift_delta_pp": site_vs_generic},
                "task_family_summary": {
                    "manipulation": {
                        "frozen_baseline": 0.20,
                        "generic_control": 0.24,
                        "site_trained": 0.35,
                    }
                },
            },
        },
        fac_dir / "s4d_policy_pair_eval_result.json",
    )
    dataset_root = root.parent / "policy_datasets" / facility_id
    dataset_root.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "dataset_lineage": {
                "claim_manifest_hash": _json_manifest_hash(claim_manifest_path),
                "claim_split_manifest_hash": _json_manifest_hash(claim_split_path),
            }
        },
        dataset_root / "dataset_export_summary.json",
    )


def test_claim_portfolio_artifact_written(sample_config, tmp_path):
    from blueprint_validation.config import FacilityConfig
    from blueprint_validation.evaluation.claim_benchmark import claim_benchmark_manifest_hash
    from blueprint_validation.evaluation.claim_portfolio import build_claim_portfolio_artifact

    sample_config.eval_policy.claim_protocol = "fixed_same_facility_uplift"
    sample_config.rollout_dataset.export_dir = tmp_path / "policy_datasets"
    for facility_id in ("facility_b", "facility_c"):
        benchmark_path = tmp_path / f"{facility_id}_benchmark.json"
        benchmark_path.write_text('{"version": 1, "task_specs": [], "assignments": []}')
        sample_config.facilities[facility_id] = FacilityConfig(
            name=facility_id,
            ply_path=tmp_path / f"{facility_id}.ply",
            claim_benchmark_path=benchmark_path,
        )
        sample_config.facilities[facility_id].ply_path.write_text("")
    sample_config.facilities["test_facility"].claim_benchmark_path = tmp_path / "test_facility_benchmark.json"
    sample_config.facilities["test_facility"].claim_benchmark_path.write_text('{"version": 1, "task_specs": [], "assignments": []}')

    work_dir = tmp_path / "outputs"
    for idx, facility_id in enumerate(sample_config.facilities.keys(), start=1):
        benchmark_hash = claim_benchmark_manifest_hash(Path(sample_config.facilities[facility_id].claim_benchmark_path))
        _write_claim_artifacts(
            work_dir,
            facility_id,
            benchmark_path=Path(sample_config.facilities[facility_id].claim_benchmark_path),
            benchmark_hash=benchmark_hash,
            world_hash=f"world_{idx}",
            site_vs_frozen=9.0 + idx,
            site_vs_generic=3.0,
        )

    output_path = build_claim_portfolio_artifact(sample_config, work_dir)
    assert output_path is not None
    payload = json.loads(output_path.read_text())
    assert payload["eligible_facility_count"] == 3
    assert payload["go_to_robot_gate"]["passed"] is True
