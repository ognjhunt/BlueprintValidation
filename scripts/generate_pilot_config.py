#!/usr/bin/env python3
"""Generate a fast pilot validation config from BlueprintCapturePipeline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _discover_run_candidates(runs_root: Path) -> List[Tuple[Path, Path]]:
    by_run: Dict[Path, Tuple[int, float, int, Path]] = {}
    for name in ("export_last_refined.ply", "export_last.ply"):
        refined_bonus = 1 if name == "export_last_refined.ply" else 0
        for ply in runs_root.rglob(name):
            if not ply.is_file():
                continue
            run_dir = ply.parent.parent if ply.parent.name == "nurec" else ply.parent
            mtime = ply.stat().st_mtime
            hints_bonus = 1 if (run_dir / "task_targets.json").exists() else 0
            current = by_run.get(run_dir)
            score = (hints_bonus, mtime, refined_bonus)
            if current is None or score > (current[0], current[1], current[2]):
                by_run[run_dir] = (hints_bonus, mtime, refined_bonus, ply)
    ranked = sorted(
        by_run.items(),
        key=lambda item: (item[1][0], item[1][1], item[1][2]),
        reverse=True,
    )
    return [(run_dir, payload[3]) for run_dir, payload in ranked]


def _latest_task_hints(runs_root: Path) -> Path | None:
    candidates = sorted(
        runs_root.rglob("task_targets.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _facility_from_run(
    run_dir: Path,
    ply_path: Path,
    idx: int,
    fallback_task_hints: Path | None,
) -> tuple[str, dict]:
    fid = "facility_a" if idx == 0 else "facility_b"
    task_hints_path = run_dir / "task_targets.json"
    facility = {
        "name": f"Auto Facility {idx + 1} ({run_dir.name})",
        "ply_path": str(ply_path),
        "description": f"Auto-imported from BlueprintCapturePipeline run '{run_dir.name}'.",
        "landmarks": [],
        "floor_height_m": 0.0,
        "ceiling_height_m": 5.0,
        "manipulation_zones": [
            {
                "name": "default_pick_zone",
                "approach_point": [0.0, 0.0, 0.8],
                "target_point": [0.6, 0.0, 0.8],
                "camera_height_m": 0.65,
                "camera_look_down_deg": 45.0,
                "arc_radius_m": 0.4,
            }
        ],
    }
    if task_hints_path.exists():
        facility["task_hints_path"] = str(task_hints_path)
    elif fallback_task_hints is not None:
        facility["task_hints_path"] = str(fallback_task_hints)
    return fid, facility


def _build_config(facilities: dict, include_cross_site: bool) -> dict:
    return {
        "schema_version": "v1",
        "project_name": "Blueprint Validation Pilot (Auto)",
        "facilities": facilities,
        "render": {
            "resolution": [480, 640],
            "fps": 10,
            "num_frames": 25,
            "camera_height_m": 1.2,
            "camera_look_down_deg": 15,
            "camera_paths": [
                {"type": "orbit", "radius_m": 2.5, "num_orbits": 1},
                {"type": "sweep", "length_m": 6.0},
                {"type": "manipulation", "height_override_m": 0.65, "look_down_override_deg": 45},
            ],
            "num_clips_per_path": 1,
        },
        "robot_composite": {
            "enabled": False,
            "urdf_path": "./configs/robots/sample_6dof_arm.urdf",
            "base_xyz": [0.0, 0.0, 0.0],
            "base_rpy": [0.0, 0.0, 0.0],
            "start_joint_positions": [0.0, -1.1, 1.2, -1.0, -1.2, 0.0],
            "end_joint_positions": [0.4, -0.8, 1.0, -1.2, -1.0, 0.3],
            "min_visible_joint_ratio": 0.6,
            "min_consistency_score": 0.6,
        },
        "gemini_polish": {
            "enabled": False,
            "model": "gemini-3.1-flash-image-preview",
            "api_key_env": "GOOGLE_GENAI_API_KEY",
            "sample_every_n_frames": 3,
        },
        "enrich": {
            "cosmos_model": "nvidia/Cosmos-Transfer2.5-2B",
            "cosmos_checkpoint": "../data/checkpoints/cosmos-transfer-2.5-2b/",
            "cosmos_repo": "/opt/cosmos-transfer",
            "controlnet_inputs": ["rgb", "depth"],
            "num_variants_per_render": 2,
            "guidance": 7.0,
            "variants": [
                {"name": "daylight", "prompt": "Bright clean industrial environment"},
                {"name": "busy_shift", "prompt": "Busy shift with pallets and workers"},
            ],
        },
        "finetune": {
            "dreamdojo_repo": "/opt/DreamDojo",
            "dreamdojo_checkpoint": "../data/checkpoints/DreamDojo/2B_pretrain/",
            "model_size": "2B",
            "use_lora": True,
            "lora_rank": 32,
            "lora_alpha": 32,
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 20,
            "save_every_n_epochs": 1,
            "max_training_hours": 4,
        },
        "eval_policy": {
            "openvla_model": "openvla/openvla-7b",
            "openvla_checkpoint": "../data/checkpoints/openvla-7b/",
            "unnorm_key": "bridge_orig",
            "num_rollouts": 8,
            "max_steps_per_rollout": 60,
            "conditions": ["baseline", "adapted"],
            "tasks": [
                "Navigate forward through the corridor",
                "Turn left at the intersection",
            ],
            "manipulation_tasks": [
                "Pick up the tote from the shelf",
                "Place the tote onto the target shelf",
                "Pick from left bin and place into right bin",
            ],
            "vlm_judge": {
                "model": "gemini-3-flash-preview",
                "api_key_env": "GOOGLE_GENAI_API_KEY",
                "enable_agentic_vision": True,
            },
        },
        "policy_finetune": {
            "enabled": True,
            "openvla_repo": "/opt/openvla",
            "finetune_script": "vla-scripts/finetune.py",
            "data_root_dir": "../data/openvla_datasets",
            "dataset_name": "bridge_orig",
            "lora_rank": 32,
            "batch_size": 4,
            "grad_accumulation_steps": 2,
            "learning_rate": 5e-4,
            "save_steps": 250,
            "max_steps": 400,
            "image_aug": True,
            "nproc_per_node": 1,
        },
        "policy_adapter": {"name": "openvla"},
        "rollout_dataset": {
            "enabled": True,
            "seed": 17,
            "train_split": 0.75,
            "min_steps_per_rollout": 4,
            "task_score_threshold": 7.0,
            "include_failed_rollouts": False,
            "max_action_delta_norm": 5.0,
            "require_consistent_action_dim": True,
            "baseline_dataset_name": "blueprint_baseline_generated",
            "adapted_dataset_name": "blueprint_site_generated",
            "export_dir": "../data/outputs/policy_datasets",
        },
        "policy_compare": {
            "enabled": True,
            "heldout_num_rollouts": 6,
            "heldout_seed": 123,
            "eval_world_model": "adapted",
            "heldout_tasks": [
                "Pick up the tote from the shelf",
                "Place the tote onto the target shelf",
            ],
            "task_score_success_threshold": 7.0,
            "manipulation_task_keywords": [
                "pick",
                "grasp",
                "lift",
                "place",
                "stack",
                "regrasp",
                "tote",
                "bin",
            ],
            "require_grasp_for_manipulation": True,
            "require_lift_for_manipulation": True,
            "require_place_for_manipulation": True,
        },
        "eval_visual": {"metrics": ["psnr", "ssim", "lpips"], "lpips_backbone": "alex"},
        "eval_spatial": {"num_sample_frames": 12, "vlm_model": "gemini-3-flash-preview"},
        "eval_crosssite": {
            "num_clips_per_model": 20 if include_cross_site else 0,
            "vlm_model": "gemini-3-flash-preview",
        },
        "cloud": {
            "provider": "runpod",
            "gpu_type": "H100",
            "num_gpus": 1,
            "max_cost_usd": 250,
            "auto_shutdown": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-pipeline-root",
        type=Path,
        default=Path("/Users/nijelhunt_1/workspace/BlueprintCapturePipeline"),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/pilot_validation.auto.yaml"),
    )
    args = parser.parse_args()

    runs_root = args.capture_pipeline_root / "runs"
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    candidates = _discover_run_candidates(runs_root)
    if not candidates:
        raise SystemExit(f"No PLY outputs found under: {runs_root}")

    fallback_task_hints = _latest_task_hints(runs_root)
    facilities = {}
    for idx, (run_dir, ply_path) in enumerate(candidates[:2]):
        fid, facility = _facility_from_run(
            run_dir,
            ply_path,
            idx,
            fallback_task_hints=fallback_task_hints,
        )
        facilities[fid] = facility

    include_cross_site = len(facilities) >= 2
    config = _build_config(facilities, include_cross_site=include_cross_site)

    out = args.output_config
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n")

    print(f"Wrote pilot config: {out}")
    for fid, fcfg in facilities.items():
        print(f"  {fid}: {fcfg['ply_path']}")
        if "task_hints_path" in fcfg:
            print(f"    task_hints_path: {fcfg['task_hints_path']}")
    if not include_cross_site:
        print("Cross-site stage will be skipped until a second facility scan is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
