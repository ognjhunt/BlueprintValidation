#!/usr/bin/env python3
"""Generate a fast pilot validation config from BlueprintCapturePipeline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


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


def _discover_handoff_candidates(capture_root: Path) -> List[Path]:
    ranked: list[tuple[int, int, float, Path]] = []
    for handoff_path in capture_root.rglob("opportunity_handoff.json"):
        if not handoff_path.is_file():
            continue
        pipeline_dir = handoff_path.parent
        bundle_root = pipeline_dir / "advanced_geometry"
        has_bundle = int(bundle_root.is_dir())
        has_ply = int((bundle_root / "3dgs_compressed.ply").is_file())
        ranked.append((has_bundle, has_ply, handoff_path.stat().st_mtime, handoff_path))
    ranked.sort(reverse=True)
    return [path for *_score, path in ranked]


def _latest_task_hints(runs_root: Path) -> Path | None:
    candidates = sorted(
        runs_root.rglob("task_targets.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _zones_from_task_targets(task_targets_path: Path) -> List[dict]:
    """Extract manipulation zones from a task_targets.json file.

    Reads OBB centers from manipulation_candidates and uses them as
    approach/target points instead of hardcoded defaults.
    """
    try:
        data = json.loads(task_targets_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    zones: List[dict] = []
    candidates = data.get("manipulation_candidates", [])
    for cand in candidates:
        bbox = cand.get("boundingBox") or {}
        center = bbox.get("center")
        if not center or len(center) != 3:
            continue
        label = cand.get("label", "object")
        extents = bbox.get("extents", [0.3, 0.3, 0.3])
        max_xy = max(extents[0], extents[1]) if len(extents) >= 2 else 0.3
        standoff = min(max(0.6, max_xy * 1.5), 3.0)
        cam_height = max(0.4, center[2] + 0.3)

        zones.append(
            {
                "name": f"{label}_{len(zones)}",
                "approach_point": [round(c, 3) for c in center],
                "target_point": [
                    round(center[0] + standoff * 0.5, 3),
                    round(center[1], 3),
                    round(center[2], 3),
                ],
                "camera_height_m": round(cam_height, 2),
                "camera_look_down_deg": 45.0,
                "arc_radius_m": round(standoff, 2),
            }
        )
    return zones


def _candidate_hints_path(
    *,
    pipeline_dir: Optional[Path],
    bundle_root: Optional[Path],
    legacy_run_dir: Optional[Path],
    fallback_task_hints: Path | None,
) -> Optional[Path]:
    candidates: list[Path] = []
    if bundle_root is not None:
        candidates.append(bundle_root / "task_targets.synthetic.json")
    if pipeline_dir is not None:
        candidates.append(pipeline_dir / "task_targets.json")
    if legacy_run_dir is not None:
        candidates.append(legacy_run_dir / "task_targets.json")
    if fallback_task_hints is not None:
        candidates.append(fallback_task_hints)
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _facility_from_artifacts(
    *,
    pipeline_dir: Optional[Path],
    handoff_path: Optional[Path],
    bundle_root: Optional[Path],
    legacy_run_dir: Optional[Path],
    legacy_ply_path: Optional[Path],
    idx: int,
    fallback_task_hints: Path | None,
) -> tuple[str, dict]:
    fid = "facility_a" if idx == 0 else "facility_b"
    effective_hints = _candidate_hints_path(
        pipeline_dir=pipeline_dir,
        bundle_root=bundle_root,
        legacy_run_dir=legacy_run_dir,
        fallback_task_hints=fallback_task_hints,
    )

    # Try to extract real manipulation zones from task_targets.json
    zones: List[dict] = []
    if effective_hints is not None and effective_hints.exists():
        zones = _zones_from_task_targets(effective_hints)

    if not zones:
        zones = [
            {
                "name": "default_pick_zone",
                "approach_point": [0.0, 0.0, 0.8],
                "target_point": [0.6, 0.0, 0.8],
                "camera_height_m": 0.65,
                "camera_look_down_deg": 45.0,
                "arc_radius_m": 0.4,
            }
        ]

    source_name = ""
    if handoff_path is not None:
        source_name = handoff_path.parent.name
    elif legacy_run_dir is not None:
        source_name = legacy_run_dir.name
    else:
        source_name = f"target_{idx + 1}"

    facility = {
        "name": f"Auto Facility {idx + 1} ({source_name})",
        "description": f"Auto-imported from BlueprintCapturePipeline run '{source_name}'.",
        "landmarks": [],
        "floor_height_m": 0.0,
        "ceiling_height_m": 5.0,
        "manipulation_zones": zones,
    }
    if handoff_path is not None:
        facility["opportunity_handoff_path"] = str(handoff_path)
        if bundle_root is not None and bundle_root.exists():
            facility["geometry_bundle_path"] = str(bundle_root)
    else:
        if legacy_ply_path is not None:
            facility["ply_path"] = str(legacy_ply_path)
        if effective_hints is not None:
            facility["task_hints_path"] = str(effective_hints)
    return fid, facility


def _pick_repo_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # Default to last (usually local vendor path) when nothing exists yet.
    return str(candidates[-1])


def _build_config(
    qualified_opportunities: dict,
    include_cross_site: bool,
    policy_finetune_enabled: bool,
    dreamdojo_repo: str,
    cosmos_repo: str,
    openvla_repo: str,
    openpi_repo: str,
    policy_adapter_name: str,
    eval_model_name: str,
    eval_checkpoint_path: str,
) -> dict:
    return {
        "schema_version": "v1",
        "project_name": "Blueprint Validation Pilot (Auto)",
        "qualified_opportunities": qualified_opportunities,
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
            "scene_aware": True,
            "collision_check": True,
            "voxel_size_m": 0.1,
            "density_threshold": 3,
            "min_clearance_m": 0.15,
            "vlm_fallback": True,
            "vlm_fallback_model": "gemini-3-flash-preview",
            "vlm_fallback_num_views": 4,
        },
        "robot_composite": {
            "enabled": True,
            "urdf_path": "./configs/robots/sample_6dof_arm.urdf",
            "base_xyz": [0.0, 0.0, 0.0],
            "base_rpy": [0.0, 0.0, 0.0],
            "start_joint_positions": [0.0, -1.1, 1.2, -1.0, -1.2, 0.0],
            "end_joint_positions": [0.4, -0.8, 1.0, -1.2, -1.0, 0.3],
            "min_visible_joint_ratio": 0.6,
            "min_consistency_score": 0.6,
        },
        "gemini_polish": {
            "enabled": True,
            "model": "gemini-3.1-flash-image-preview",
            "api_key_env": "GOOGLE_GENAI_API_KEY",
            "sample_every_n_frames": 3,
        },
        "robosplat": {
            "enabled": True,
            "backend": "auto",
            "parity_mode": "hybrid",
            "runtime_preset": "balanced",
            "variants_per_input": 4,
            "object_source_priority": ["task_hints_obb", "vlm_detect", "cluster"],
            "demo_source": "synthetic",
            "bootstrap_if_missing_demo": True,
            "bootstrap_num_rollouts": 6,
            "bootstrap_horizon_steps": 24,
            "bootstrap_tasks_limit": 4,
            "quality_gate_enabled": True,
            "min_variants_required_per_clip": 1,
            "fallback_to_legacy_scan": True,
            "fallback_on_backend_error": True,
            "persist_scene_variants": False,
            "vendor_repo_path": "./vendor/robosplat",
            "vendor_ref": "",
        },
        "robosplat_scan": {  # legacy compatibility
            "enabled": True,
            "num_augmented_clips_per_input": 2,
            "yaw_jitter_deg": 6.0,
            "pitch_jitter_deg": 4.0,
            "camera_height_jitter_m": 0.12,
            "relight_gain_min": 0.85,
            "relight_gain_max": 1.20,
            "color_temp_shift": True,
            "temporal_speed_factors": [0.9, 1.1],
        },
        "enrich": {
            "cosmos_model": "nvidia/Cosmos-Transfer2.5-2B",
            "cosmos_checkpoint": "../data/checkpoints/cosmos-transfer-2.5-2b/",
            "cosmos_repo": cosmos_repo,
            "controlnet_inputs": ["rgb", "depth"],
            "num_variants_per_render": 2,
            "guidance": 7.0,
            "dynamic_variants": True,
            "dynamic_variants_model": "gemini-3-flash-preview",
        },
        "finetune": {
            "dreamdojo_repo": dreamdojo_repo,
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
            "model_name": eval_model_name,
            "checkpoint_path": eval_checkpoint_path,
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
            "enabled": policy_finetune_enabled,
            "openvla_repo": openvla_repo,
            "finetune_script": "vla-scripts/finetune.py",
            "data_root_dir": "../data/openvla_datasets",
            "dataset_name": "bridge_orig",
            "recipe": "oft",
            "use_l1_regression": True,
            "lora_rank": 32,
            "batch_size": 4,
            "grad_accumulation_steps": 2,
            "learning_rate": 5e-4,
            "save_steps": 250,
            "max_steps": 400,
            "image_aug": True,
            "nproc_per_node": 1,
        },
        "policy_rl_loop": {
            "enabled": False,
            "iterations": 2,
            "horizon_steps": 24,
            "rollouts_per_task": 8,
            "group_size": 4,
            "reward_mode": "hybrid",
            "vlm_reward_fraction": 0.25,
            "top_quantile": 0.30,
            "near_miss_min_quantile": 0.30,
            "near_miss_max_quantile": 0.60,
            "policy_refine_steps_per_iter": 1000,
            "world_model_refresh_enabled": True,
            "world_model_refresh_epochs": 3,
            "world_model_refresh_learning_rate": 5e-5,
        },
        "wm_refresh_loop": {
            "enabled": True,
            "iterations": 1,
            "source_condition": "adapted",
        },
        "policy_adapter": {
            "name": policy_adapter_name,
            "openvla": {
                "openvla_repo": openvla_repo,
                "finetune_script": "vla-scripts/finetune.py",
                "extra_train_args": [],
            },
            "pi05": {
                "openpi_repo": openpi_repo,
                "profile": "pi05_libero",
                "runtime_mode": "inprocess",
                "train_backend": "pytorch",
                "train_script": "scripts/train_pytorch.py",
                "norm_stats_script": "scripts/compute_norm_stats.py",
                "policy_action_dim": 7,
                "policy_state_dim": 7,
                "extra_train_args": [],
            },
        },
        "rollout_dataset": {
            "enabled": policy_finetune_enabled,
            "seed": 17,
            "train_split": 0.75,
            "min_steps_per_rollout": 4,
            "task_score_threshold": 7.0,
            "include_failed_rollouts": False,
            "max_action_delta_norm": 5.0,
            "require_consistent_action_dim": True,
            "baseline_dataset_name": "bridge_dataset",
            "adapted_dataset_name": "bridge_orig",
            "export_dir": "../data/outputs/policy_datasets",
        },
        "policy_compare": {
            "enabled": False,
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
    parser.add_argument(
        "--policy-finetune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable/disable policy finetune + rollout dataset export in generated config "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--policy-adapter",
        choices=["openvla_oft", "pi05"],
        default="openvla_oft",
        help="Select policy adapter defaults in generated config (default: openvla_oft).",
    )
    parser.add_argument(
        "--pi05-model-ref",
        default="",
        help=(
            "Explicit pi05 base model reference for eval_policy.model_name. "
            "Required when --policy-adapter=pi05 unless --pi05-checkpoint-path is provided."
        ),
    )
    parser.add_argument(
        "--pi05-checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Explicit pi05 checkpoint path for eval_policy.checkpoint_path. "
            "Required when --policy-adapter=pi05 unless --pi05-model-ref is provided."
        ),
    )
    args = parser.parse_args()

    runs_root = args.capture_pipeline_root / "runs"
    handoff_candidates = _discover_handoff_candidates(args.capture_pipeline_root)
    ply_candidates = _discover_run_candidates(runs_root) if runs_root.exists() else []
    if not handoff_candidates and not ply_candidates:
        raise SystemExit(
            "No qualified handoffs or legacy PLY outputs found under: "
            f"{args.capture_pipeline_root}"
        )

    fallback_task_hints = _latest_task_hints(runs_root) if runs_root.exists() else None
    qualified_opportunities = {}

    for idx, handoff_path in enumerate(handoff_candidates[:2]):
        pipeline_dir = handoff_path.parent
        bundle_root = pipeline_dir / "advanced_geometry"
        fid, facility = _facility_from_artifacts(
            pipeline_dir=pipeline_dir,
            handoff_path=handoff_path,
            bundle_root=bundle_root if bundle_root.exists() else None,
            legacy_run_dir=None,
            legacy_ply_path=None,
            idx=idx,
            fallback_task_hints=fallback_task_hints,
        )
        qualified_opportunities[fid] = facility

    if not qualified_opportunities:
        for idx, (run_dir, ply_path) in enumerate(ply_candidates[:2]):
            fid, facility = _facility_from_artifacts(
                pipeline_dir=None,
                handoff_path=None,
                bundle_root=None,
                legacy_run_dir=run_dir,
                legacy_ply_path=ply_path,
                idx=idx,
                fallback_task_hints=fallback_task_hints,
            )
            qualified_opportunities[fid] = facility

    policy_finetune_enabled = bool(args.policy_finetune)
    dreamdojo_repo = _pick_repo_path(
        Path("/opt/DreamDojo"),
        args.capture_pipeline_root / "vendor" / "DreamDojo",
        Path.cwd() / "data" / "vendor" / "DreamDojo",
    )
    cosmos_repo = _pick_repo_path(
        Path("/opt/cosmos-transfer"),
        args.capture_pipeline_root / "vendor" / "cosmos-transfer",
        Path.cwd() / "data" / "vendor" / "cosmos-transfer",
    )
    openvla_repo = _pick_repo_path(
        Path("/opt/openvla-oft"),
        args.capture_pipeline_root / "vendor" / "openvla-oft",
        Path.cwd() / "data" / "vendor" / "openvla-oft",
    )
    openpi_repo = _pick_repo_path(
        Path("/opt/openpi"),
        args.capture_pipeline_root / "vendor" / "openpi",
        Path.cwd() / "data" / "vendor" / "openpi",
    )

    include_cross_site = len(qualified_opportunities) >= 2
    policy_adapter_name = str(args.policy_adapter)
    if policy_adapter_name == "pi05" and not (
        str(args.pi05_model_ref).strip() or args.pi05_checkpoint_path
    ):
        raise SystemExit(
            "When --policy-adapter=pi05, provide --pi05-model-ref and/or --pi05-checkpoint-path."
        )

    if policy_adapter_name == "pi05":
        eval_model_name = str(args.pi05_model_ref).strip() or "openpi/pi05"
        if args.pi05_checkpoint_path is not None:
            eval_checkpoint_path = str(args.pi05_checkpoint_path)
        else:
            eval_checkpoint_path = "../data/checkpoints/pi05/"
    else:
        eval_model_name = "openvla/openvla-7b"
        eval_checkpoint_path = "../data/checkpoints/openvla-7b/"

    config = _build_config(
        qualified_opportunities,
        include_cross_site=include_cross_site,
        policy_finetune_enabled=policy_finetune_enabled,
        dreamdojo_repo=dreamdojo_repo,
        cosmos_repo=cosmos_repo,
        openvla_repo=openvla_repo,
        openpi_repo=openpi_repo,
        policy_adapter_name=policy_adapter_name,
        eval_model_name=eval_model_name,
        eval_checkpoint_path=eval_checkpoint_path,
    )

    out = args.output_config
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"Wrote pilot config: {out}")
    for fid, fcfg in qualified_opportunities.items():
        if "opportunity_handoff_path" in fcfg:
            print(f"  {fid}: handoff={fcfg['opportunity_handoff_path']}")
            if "geometry_bundle_path" in fcfg:
                print(f"    geometry_bundle_path: {fcfg['geometry_bundle_path']}")
        else:
            print(f"  {fid}: ply_path={fcfg['ply_path']}")
            if "task_hints_path" in fcfg:
                print(f"    task_hints_path: {fcfg['task_hints_path']}")
    print(f"  policy_adapter: {policy_adapter_name}")
    print(f"  eval_policy.model_name: {eval_model_name}")
    print(f"  eval_policy.checkpoint_path: {eval_checkpoint_path}")
    if not include_cross_site:
        print("Cross-site stage will be skipped until a second facility scan is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
