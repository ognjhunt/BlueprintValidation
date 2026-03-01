"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import yaml


@dataclass
class ManipulationZoneConfig:
    """A region in the facility where manipulation tasks occur."""

    name: str
    approach_point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    target_point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    camera_height_m: float = 0.6
    camera_look_down_deg: float = 45.0
    arc_radius_m: float = 0.4


@dataclass
class FacilityConfig:
    name: str
    ply_path: Path
    task_hints_path: Optional[Path] = None
    description: str = ""
    landmarks: List[str] = field(default_factory=list)
    floor_height_m: float = 0.0
    ceiling_height_m: float = 5.0
    manipulation_zones: List[ManipulationZoneConfig] = field(default_factory=list)
    # Scene orientation correction — the pipeline assumes Z-up.
    # "auto" detects the up axis from point cloud extents during warmup.
    # Override with "z", "y", "-y", "-z", "x", or "-x" if auto-detection is wrong.
    up_axis: str = "auto"
    scene_rotation_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class CameraPathSpec:
    type: str  # "orbit", "sweep", "file", "manipulation"
    # orbit params
    radius_m: float = 3.0
    num_orbits: int = 2
    # sweep params
    length_m: float = 10.0
    # file params
    path: Optional[str] = None
    # per-path overrides
    height_override_m: Optional[float] = None
    look_down_override_deg: Optional[float] = None
    # manipulation arc params
    approach_point: Optional[List[float]] = None
    arc_radius_m: float = 0.4
    # internal metadata: where this spec came from
    source_tag: Optional[str] = None


@dataclass
class RenderConfig:
    resolution: tuple[int, int] = (480, 640)  # H x W
    fps: int = 10
    num_frames: int = 49
    camera_height_m: float = 1.2
    camera_look_down_deg: float = 15.0
    camera_paths: List[CameraPathSpec] = field(default_factory=list)
    num_clips_per_path: int = 3
    # Scene-aware camera placement
    scene_aware: bool = True
    collision_check: bool = True
    voxel_size_m: float = 0.1
    density_threshold: int = 3
    min_clearance_m: float = 0.15
    vlm_fallback: bool = True
    vlm_fallback_model: str = "gemini-3-flash-preview"
    vlm_fallback_num_views: int = 4
    # Task-scoped scene-aware camera generation (budget mode)
    task_scoped_scene_aware: bool = False
    task_scoped_max_specs: int = 40
    task_scoped_context_per_target: int = 2
    task_scoped_overview_specs: int = 6
    task_scoped_fallback_specs: int = 16
    task_scoped_profile: str = "dreamdojo"


@dataclass
class RobotCompositeConfig:
    enabled: bool = True
    urdf_path: Optional[Path] = None
    end_effector_link: Optional[str] = None
    base_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    start_joint_positions: List[float] = field(default_factory=list)
    end_joint_positions: List[float] = field(default_factory=list)
    min_visible_joint_ratio: float = 0.6
    min_consistency_score: float = 0.6
    line_color_bgr: List[int] = field(default_factory=lambda: [50, 180, 255])
    line_thickness: int = 3


@dataclass
class GeminiPolishConfig:
    enabled: bool = True
    model: str = "gemini-3.1-flash-image-preview"
    api_key_env: str = "GOOGLE_GENAI_API_KEY"
    prompt: str = (
        "Preserve robot arm pose and scene geometry exactly. Improve photorealism, lighting, "
        "material coherence, and blending quality."
    )
    sample_every_n_frames: int = 2


@dataclass
class VariantSpec:
    name: str
    prompt: str


@dataclass
class EnrichConfig:
    cosmos_model: str = "nvidia/Cosmos-Transfer2.5-2B"
    cosmos_checkpoint: Path = Path("./data/checkpoints/cosmos-transfer-2.5-2b/")
    cosmos_repo: Path = Path("/opt/cosmos-transfer")
    disable_guardrails: bool = True
    controlnet_inputs: List[str] = field(default_factory=lambda: ["rgb", "depth"])
    num_variants_per_render: int = 5
    variants: List[VariantSpec] = field(default_factory=list)
    guidance: float = 7.0
    # Dynamic variant generation: use Gemini to produce scene-appropriate prompts
    dynamic_variants: bool = True
    dynamic_variants_model: str = "gemini-3-flash-preview"
    # If false, Stage 2 fails when dynamic variant generation is unavailable.
    allow_dynamic_variant_fallback: bool = True
    # Optional explicit context frame index used for Cosmos image context anchoring.
    # If unset, Stage 2 uses a deterministic quarter-way frame for each clip.
    context_frame_index: Optional[int] = None
    # Acceptance gate for control-faithful enrichment. Disabled when <= 0.
    min_frame0_ssim: float = 0.0
    # Delete generated output files rejected by the frame-0 SSIM gate.
    delete_rejected_outputs: bool = False


@dataclass
class FinetuneConfig:
    dreamdojo_repo: Path = Path("/opt/DreamDojo")
    dreamdojo_checkpoint: Path = Path("./data/checkpoints/DreamDojo/2B_pretrain/")
    # Optional isolated Python runtime for DreamDojo Stage 3 (for dependency pinning).
    python_executable: Optional[Path] = None
    experiment_config: Optional[str] = None  # DreamDojo experiment config name
    model_size: str = "2B"
    # video backend for Stage 3 dataloader: "opencv" uses Blueprint dataset class (no torchcodec),
    # "vendor" keeps DreamDojo's MultiVideoActionDataset path.
    video_dataset_backend: str = "opencv"
    # Run a dataloader sample probe (dataset[0]) before launching training.
    probe_dataloader_sample: bool = True
    # LoRA via Cosmos config system (model.config.use_lora / train_architecture=lora)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    learning_rate: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_every_n_epochs: int = 10
    max_training_hours: float = 72.0


@dataclass
class VLMJudgeConfig:
    model: str = "gemini-3-flash-preview"
    api_key_env: str = "GOOGLE_GENAI_API_KEY"
    enable_agentic_vision: bool = True
    scoring_prompt: str = (
        "You are evaluating a robot policy rollout video.\n"
        "Score the following on a 1-10 scale:\n"
        '1. Task completion (did the robot accomplish "{task}"?)\n'
        "2. Visual plausibility (does the environment look realistic?)\n"
        "3. Spatial coherence (are objects in consistent positions across frames?)\n"
        'Return JSON: {{"task_score": N, "visual_score": N, "spatial_score": N, "reasoning": "..."}}'
    )


@dataclass
class PolicyEvalConfig:
    model_name: str = "openvla/openvla-7b"
    checkpoint_path: Path = Path("./data/checkpoints/openvla-7b/")
    unnorm_key: str = "bridge_orig"
    num_rollouts: int = 50
    max_steps_per_rollout: int = 100
    tasks: List[str] = field(default_factory=list)
    manipulation_tasks: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=lambda: ["baseline", "adapted"])
    min_absolute_difference: float = 0.5  # minimum raw score difference for PASS
    vlm_judge: VLMJudgeConfig = field(default_factory=VLMJudgeConfig)

    @property
    def openvla_model(self) -> str:
        """Legacy alias for model_name."""
        return self.model_name

    @openvla_model.setter
    def openvla_model(self, value: str) -> None:
        self.model_name = value

    @property
    def openvla_checkpoint(self) -> Path:
        """Legacy alias for checkpoint_path."""
        return self.checkpoint_path

    @openvla_checkpoint.setter
    def openvla_checkpoint(self, value: Path) -> None:
        self.checkpoint_path = value


@dataclass
class PolicyFinetuneConfig:
    enabled: bool = True
    openvla_repo: Path = Path("/opt/openvla-oft")
    finetune_script: str = "vla-scripts/finetune.py"
    data_root_dir: Optional[Path] = None
    dataset_name: str = "bridge_orig"
    run_root_dir: Path = Path("./data/outputs/policy_finetune/runs")
    adapter_tmp_dir: Path = Path("./data/outputs/policy_finetune/adapters")
    lora_rank: int = 32
    batch_size: int = 8
    grad_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    save_steps: int = 1000
    max_steps: int = 5000
    image_aug: bool = True
    nproc_per_node: int = 1
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    recipe: str = "oft"
    action_chunk_size: int = 8  # Legacy OFT option (ignored by vendored openvla-oft finetune.py).
    use_continuous_actions: bool = True  # Legacy OFT option (ignored by vendored script).
    use_l1_regression: bool = True
    parallel_decoding: bool = True  # Legacy OFT option (ignored by vendored script).
    extra_args: List[str] = field(default_factory=list)


@dataclass
class OpenVLAAdapterBackendConfig:
    openvla_repo: Path = Path("/opt/openvla-oft")
    finetune_script: str = "vla-scripts/finetune.py"
    extra_train_args: List[str] = field(default_factory=list)


@dataclass
class Pi05AdapterBackendConfig:
    openpi_repo: Path = Path("/opt/openpi")
    profile: str = "pi05_libero"  # pi05_libero | pi05_droid
    runtime_mode: str = "inprocess"  # inprocess (only supported mode)
    train_backend: str = "pytorch"  # pytorch (only supported backend)
    train_script: str = "scripts/train_pytorch.py"
    norm_stats_script: str = "scripts/compute_norm_stats.py"
    policy_action_dim: int = 7
    policy_state_dim: int = 7
    extra_train_args: List[str] = field(default_factory=list)


@dataclass
class PolicyAdapterConfig:
    name: str = "openvla_oft"
    openvla: OpenVLAAdapterBackendConfig = field(default_factory=OpenVLAAdapterBackendConfig)
    pi05: Pi05AdapterBackendConfig = field(default_factory=Pi05AdapterBackendConfig)


@dataclass
class RoboSplatScanConfig:
    enabled: bool = True
    num_augmented_clips_per_input: int = 2
    yaw_jitter_deg: float = 6.0
    pitch_jitter_deg: float = 4.0
    camera_height_jitter_m: float = 0.12
    relight_gain_min: float = 0.85
    relight_gain_max: float = 1.20
    color_temp_shift: bool = True
    temporal_speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RoboSplatConfig:
    enabled: bool = True
    backend: str = "auto"  # auto|vendor|native|legacy_scan
    parity_mode: str = "hybrid"  # hybrid|strict|scan_only
    runtime_preset: str = "balanced"  # balanced|high_quality|fast
    variants_per_input: int = 4
    object_source_priority: List[str] = field(
        default_factory=lambda: ["task_hints_obb", "vlm_detect", "cluster"]
    )
    demo_source: str = "synthetic"  # synthetic|real|required_real
    demo_manifest_path: Optional[Path] = None
    min_successful_demos: int = 4
    demo_success_task_score_threshold: float = 7.0
    require_manipulation_success_flags: bool = True
    world_model_bootstrap_enabled: bool = False
    bootstrap_if_missing_demo: bool = True
    bootstrap_num_rollouts: int = 6
    bootstrap_horizon_steps: int = 24
    bootstrap_tasks_limit: int = 4
    quality_gate_enabled: bool = True
    min_variants_required_per_clip: int = 1
    fallback_to_legacy_scan: bool = True
    fallback_on_backend_error: bool = True
    persist_scene_variants: bool = False
    vendor_repo_path: Path = Path("./vendor/robosplat")
    vendor_ref: str = ""


@dataclass
class SplatSimConfig:
    enabled: bool = False
    mode: str = "hybrid"  # hybrid|strict
    per_zone_rollouts: int = 2
    horizon_steps: int = 30
    min_successful_rollouts_per_zone: int = 1
    fallback_to_prior_manifest: bool = True


@dataclass
class PolicyRLLoopConfig:
    enabled: bool = False
    iterations: int = 2
    horizon_steps: int = 24
    rollouts_per_task: int = 8
    group_size: int = 4
    reward_mode: str = "hybrid"  # "hybrid", "vlm_only", "heuristic_only"
    vlm_reward_fraction: float = 0.25
    top_quantile: float = 0.30
    near_miss_min_quantile: float = 0.30
    near_miss_max_quantile: float = 0.60
    policy_refine_steps_per_iter: int = 1000
    world_model_refresh_enabled: bool = True
    world_model_refresh_epochs: int = 3
    world_model_refresh_learning_rate: float = 5.0e-5


@dataclass
class RolloutDatasetConfig:
    enabled: bool = True
    seed: int = 17
    train_split: float = 0.8
    min_steps_per_rollout: int = 4
    task_score_threshold: float = 7.0
    include_failed_rollouts: bool = False
    max_action_delta_norm: float = 5.0
    require_consistent_action_dim: bool = True
    baseline_dataset_name: str = "bridge_dataset"  # Must exist in vendored OpenVLA OXE registry.
    adapted_dataset_name: str = "bridge_orig"  # Must exist in vendored OpenVLA OXE registry.
    export_dir: Path = Path("./data/outputs/policy_datasets")


@dataclass
class PolicyCompareConfig:
    enabled: bool = False
    heldout_num_rollouts: int = 20
    heldout_seed: int = 123
    eval_world_model: str = "adapted"
    heldout_tasks: List[str] = field(default_factory=list)
    task_score_success_threshold: float = 7.0
    manipulation_task_keywords: List[str] = field(
        default_factory=lambda: [
            "pick",
            "grasp",
            "lift",
            "place",
            "stack",
            "regrasp",
            "tote",
            "bin",
        ]
    )
    require_grasp_for_manipulation: bool = True
    require_lift_for_manipulation: bool = True
    require_place_for_manipulation: bool = True


@dataclass
class VisualFidelityConfig:
    metrics: List[str] = field(default_factory=lambda: ["psnr", "ssim", "lpips"])
    lpips_backbone: str = "alex"


@dataclass
class SpatialAccuracyConfig:
    num_sample_frames: int = 20
    vlm_model: str = "gemini-3-flash-preview"


@dataclass
class CrossSiteConfig:
    num_clips_per_model: int = 30
    vlm_model: str = "gemini-3-flash-preview"


@dataclass
class CloudConfig:
    provider: str = "runpod"
    gpu_type: str = "H100"
    num_gpus: int = 1
    max_cost_usd: float = 500.0
    auto_shutdown: bool = True


@dataclass
class ValidationConfig:
    schema_version: str = "v1"
    project_name: str = ""
    facilities: Dict[str, FacilityConfig] = field(default_factory=dict)
    render: RenderConfig = field(default_factory=RenderConfig)
    robot_composite: RobotCompositeConfig = field(default_factory=RobotCompositeConfig)
    gemini_polish: GeminiPolishConfig = field(default_factory=GeminiPolishConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval_policy: PolicyEvalConfig = field(default_factory=PolicyEvalConfig)
    policy_finetune: PolicyFinetuneConfig = field(default_factory=PolicyFinetuneConfig)
    policy_adapter: PolicyAdapterConfig = field(default_factory=PolicyAdapterConfig)
    robosplat: RoboSplatConfig = field(default_factory=RoboSplatConfig)
    robosplat_scan: RoboSplatScanConfig = field(default_factory=RoboSplatScanConfig)
    splatsim: SplatSimConfig = field(default_factory=SplatSimConfig)
    policy_rl_loop: PolicyRLLoopConfig = field(default_factory=PolicyRLLoopConfig)
    rollout_dataset: RolloutDatasetConfig = field(default_factory=RolloutDatasetConfig)
    policy_compare: PolicyCompareConfig = field(default_factory=PolicyCompareConfig)
    eval_visual: VisualFidelityConfig = field(default_factory=VisualFidelityConfig)
    eval_spatial: SpatialAccuracyConfig = field(default_factory=SpatialAccuracyConfig)
    eval_crosssite: CrossSiteConfig = field(default_factory=CrossSiteConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_manipulation_zones(raw_list: List[Dict[str, Any]]) -> List[ManipulationZoneConfig]:
    zones = []
    for raw in raw_list:
        zones.append(
            ManipulationZoneConfig(
                name=raw["name"],
                approach_point=raw.get("approach_point", [0.0, 0.0, 0.0]),
                target_point=raw.get("target_point", [0.0, 0.0, 0.0]),
                camera_height_m=float(raw.get("camera_height_m", 0.6)),
                camera_look_down_deg=float(raw.get("camera_look_down_deg", 45.0)),
                arc_radius_m=float(raw.get("arc_radius_m", 0.4)),
            )
        )
    return zones


def _parse_scene_rotation_deg(raw_value: Any) -> List[float]:
    if raw_value is None:
        return [0.0, 0.0, 0.0]
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
        raise ValueError("facility.scene_rotation_deg must be a list of three numbers [rx, ry, rz]")
    return [float(v) for v in raw_value]


def _parse_facility(raw: Dict[str, Any], base_dir: Path) -> FacilityConfig:
    task_hints_value = raw.get("task_hints_path")
    return FacilityConfig(
        name=raw["name"],
        ply_path=_resolve_path(raw["ply_path"], base_dir),
        task_hints_path=_resolve_path(task_hints_value, base_dir) if task_hints_value else None,
        description=raw.get("description", ""),
        landmarks=raw.get("landmarks", []),
        floor_height_m=raw.get("floor_height_m", 0.0),
        ceiling_height_m=raw.get("ceiling_height_m", 5.0),
        manipulation_zones=_parse_manipulation_zones(raw.get("manipulation_zones", [])),
        up_axis=str(raw.get("up_axis", "auto")).strip(),
        scene_rotation_deg=_parse_scene_rotation_deg(raw.get("scene_rotation_deg")),
    )


def _parse_camera_paths(raw_list: List[Dict[str, Any]], base_dir: Path) -> List[CameraPathSpec]:
    paths = []
    for raw in raw_list:
        path_value = raw.get("path")
        paths.append(
            CameraPathSpec(
                type=raw["type"],
                radius_m=raw.get("radius_m", 3.0),
                num_orbits=raw.get("num_orbits", 2),
                length_m=raw.get("length_m", 10.0),
                path=str(_resolve_path(path_value, base_dir)) if path_value else None,
                height_override_m=raw.get("height_override_m"),
                look_down_override_deg=raw.get("look_down_override_deg"),
                approach_point=raw.get("approach_point"),
                arc_radius_m=raw.get("arc_radius_m", 0.4),
                source_tag=raw.get("source_tag"),
            )
        )
    return paths


def _parse_variants(raw_list: List[Dict[str, str]]) -> List[VariantSpec]:
    return [VariantSpec(name=v["name"], prompt=v["prompt"]) for v in raw_list]


def _resolve_eval_policy_model_fields(
    eval_policy_raw: Dict[str, Any],
) -> tuple[str, str | Path]:
    model_name = eval_policy_raw.get("model_name")
    checkpoint_path = eval_policy_raw.get("checkpoint_path")

    if model_name is None and eval_policy_raw.get("openvla_model") is not None:
        model_name = eval_policy_raw.get("openvla_model")
        warnings.warn(
            "eval_policy.openvla_model is deprecated; use eval_policy.model_name.",
            DeprecationWarning,
            stacklevel=3,
        )
    if checkpoint_path is None and eval_policy_raw.get("openvla_checkpoint") is not None:
        checkpoint_path = eval_policy_raw.get("openvla_checkpoint")
        warnings.warn(
            "eval_policy.openvla_checkpoint is deprecated; use eval_policy.checkpoint_path.",
            DeprecationWarning,
            stacklevel=3,
        )

    return (
        str(model_name or "openvla/openvla-7b"),
        checkpoint_path or "./data/checkpoints/openvla-7b/",
    )


def load_config(path: Path) -> ValidationConfig:
    """Load and parse a YAML config file into a ValidationConfig."""
    config_path = path.resolve()
    base_dir = config_path.parent

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping")

    config = ValidationConfig(
        schema_version=raw.get("schema_version", "v1"),
        project_name=raw.get("project_name", ""),
    )

    # Facilities
    for fid, fdata in raw.get("facilities", {}).items():
        config.facilities[fid] = _parse_facility(fdata, base_dir)

    # Render
    if "render" in raw:
        r = raw["render"]
        config.render = RenderConfig(
            resolution=tuple(r.get("resolution", [480, 640])),
            fps=r.get("fps", 10),
            num_frames=r.get("num_frames", 49),
            camera_height_m=r.get("camera_height_m", 1.2),
            camera_look_down_deg=r.get("camera_look_down_deg", 15.0),
            camera_paths=_parse_camera_paths(r.get("camera_paths", []), base_dir),
            num_clips_per_path=r.get("num_clips_per_path", 3),
            scene_aware=r.get("scene_aware", True),
            collision_check=r.get("collision_check", True),
            voxel_size_m=float(r.get("voxel_size_m", 0.1)),
            density_threshold=int(r.get("density_threshold", 3)),
            min_clearance_m=float(r.get("min_clearance_m", 0.15)),
            vlm_fallback=r.get("vlm_fallback", True),
            vlm_fallback_model=r.get("vlm_fallback_model", "gemini-3-flash-preview"),
            vlm_fallback_num_views=int(r.get("vlm_fallback_num_views", 4)),
            task_scoped_scene_aware=bool(r.get("task_scoped_scene_aware", False)),
            task_scoped_max_specs=int(r.get("task_scoped_max_specs", 40)),
            task_scoped_context_per_target=int(r.get("task_scoped_context_per_target", 2)),
            task_scoped_overview_specs=int(r.get("task_scoped_overview_specs", 6)),
            task_scoped_fallback_specs=int(r.get("task_scoped_fallback_specs", 16)),
            task_scoped_profile=str(r.get("task_scoped_profile", "dreamdojo")),
        )

    # Enrich
    if "enrich" in raw:
        e = raw["enrich"]
        config.enrich = EnrichConfig(
            cosmos_model=e.get("cosmos_model", "nvidia/Cosmos-Transfer2.5-2B"),
            cosmos_checkpoint=_resolve_path(
                e.get("cosmos_checkpoint", "./data/checkpoints/cosmos-transfer-2.5-2b/"),
                base_dir,
            ),
            cosmos_repo=_resolve_path(
                e.get("cosmos_repo", "/opt/cosmos-transfer"),
                base_dir,
            ),
            disable_guardrails=bool(e.get("disable_guardrails", True)),
            controlnet_inputs=e.get("controlnet_inputs", ["rgb", "depth"]),
            num_variants_per_render=e.get("num_variants_per_render", 5),
            variants=_parse_variants(e.get("variants", [])),
            guidance=e.get("guidance", 7.0),
            dynamic_variants=e.get("dynamic_variants", True),
            dynamic_variants_model=e.get("dynamic_variants_model", "gemini-3-flash-preview"),
            allow_dynamic_variant_fallback=bool(e.get("allow_dynamic_variant_fallback", True)),
            context_frame_index=(
                int(e["context_frame_index"]) if e.get("context_frame_index") is not None else None
            ),
            min_frame0_ssim=float(e.get("min_frame0_ssim", 0.0)),
            delete_rejected_outputs=bool(e.get("delete_rejected_outputs", False)),
        )

    if "robot_composite" in raw:
        rc = raw["robot_composite"]
        urdf_value = rc.get("urdf_path")
        config.robot_composite = RobotCompositeConfig(
            enabled=rc.get("enabled", True),
            urdf_path=_resolve_path(urdf_value, base_dir) if urdf_value else None,
            end_effector_link=rc.get("end_effector_link"),
            base_xyz=rc.get("base_xyz", [0.0, 0.0, 0.0]),
            base_rpy=rc.get("base_rpy", [0.0, 0.0, 0.0]),
            start_joint_positions=rc.get("start_joint_positions", []),
            end_joint_positions=rc.get("end_joint_positions", []),
            min_visible_joint_ratio=float(rc.get("min_visible_joint_ratio", 0.6)),
            min_consistency_score=float(rc.get("min_consistency_score", 0.6)),
            line_color_bgr=rc.get("line_color_bgr", [50, 180, 255]),
            line_thickness=rc.get("line_thickness", 3),
        )

    if "gemini_polish" in raw:
        gp = raw["gemini_polish"]
        config.gemini_polish = GeminiPolishConfig(
            enabled=gp.get("enabled", True),
            model=gp.get("model", "gemini-3.1-flash-image-preview"),
            api_key_env=gp.get("api_key_env", "GOOGLE_GENAI_API_KEY"),
            prompt=gp.get("prompt", GeminiPolishConfig().prompt),
            sample_every_n_frames=gp.get("sample_every_n_frames", 2),
        )

    # Finetune
    if "finetune" in raw:
        ft = raw["finetune"]
        experiment_config = ft.get("experiment_config")
        if experiment_config and (
            "/" in experiment_config
            or experiment_config.endswith(".sh")
            or experiment_config.startswith(".")
        ):
            experiment_config = str(_resolve_path(experiment_config, base_dir))
        config.finetune = FinetuneConfig(
            dreamdojo_repo=_resolve_path(ft.get("dreamdojo_repo", "/opt/DreamDojo"), base_dir),
            dreamdojo_checkpoint=_resolve_path(
                ft.get("dreamdojo_checkpoint", "./data/checkpoints/DreamDojo/2B_pretrain/"),
                base_dir,
            ),
            python_executable=(
                _resolve_path(ft.get("python_executable"), base_dir)
                if ft.get("python_executable")
                else None
            ),
            experiment_config=experiment_config,
            model_size=ft.get("model_size", "2B"),
            video_dataset_backend=str(ft.get("video_dataset_backend", "opencv")),
            probe_dataloader_sample=bool(ft.get("probe_dataloader_sample", True)),
            use_lora=ft.get("use_lora", True),
            lora_rank=ft.get("lora_rank", 32),
            lora_alpha=ft.get("lora_alpha", 32),
            lora_target_modules=ft.get(
                "lora_target_modules",
                "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            ),
            learning_rate=ft.get("learning_rate", 1e-4),
            num_epochs=ft.get("num_epochs", 50),
            batch_size=ft.get("batch_size", 1),
            gradient_accumulation_steps=ft.get("gradient_accumulation_steps", 4),
            warmup_steps=ft.get("warmup_steps", 100),
            save_every_n_epochs=ft.get("save_every_n_epochs", 10),
            max_training_hours=ft.get("max_training_hours", 72.0),
        )

    # Policy eval
    if "eval_policy" in raw:
        ep = raw["eval_policy"]
        vlm_raw = ep.get("vlm_judge", {})
        model_name, checkpoint_path = _resolve_eval_policy_model_fields(ep)
        config.eval_policy = PolicyEvalConfig(
            model_name=model_name,
            checkpoint_path=_resolve_path(checkpoint_path, base_dir),
            unnorm_key=ep.get("unnorm_key", "bridge_orig"),
            num_rollouts=ep.get("num_rollouts", 50),
            max_steps_per_rollout=ep.get("max_steps_per_rollout", 100),
            tasks=ep.get("tasks", []),
            manipulation_tasks=ep.get("manipulation_tasks", []),
            conditions=ep.get("conditions", ["baseline", "adapted"]),
            min_absolute_difference=float(ep.get("min_absolute_difference", 0.5)),
            vlm_judge=VLMJudgeConfig(
                model=vlm_raw.get("model", "gemini-3-flash-preview"),
                api_key_env=vlm_raw.get("api_key_env", "GOOGLE_GENAI_API_KEY"),
                enable_agentic_vision=vlm_raw.get("enable_agentic_vision", True),
                scoring_prompt=vlm_raw.get("scoring_prompt", VLMJudgeConfig().scoring_prompt),
            ),
        )

    # Policy fine-tuning (OpenVLA-OFT adapter stage)
    if "policy_finetune" in raw:
        pf = raw["policy_finetune"]
        data_root_dir = pf.get("data_root_dir")
        config.policy_finetune = PolicyFinetuneConfig(
            enabled=pf.get("enabled", True),
            openvla_repo=_resolve_path(pf.get("openvla_repo", "/opt/openvla-oft"), base_dir),
            finetune_script=pf.get("finetune_script", "vla-scripts/finetune.py"),
            data_root_dir=(_resolve_path(data_root_dir, base_dir) if data_root_dir else None),
            dataset_name=pf.get("dataset_name", "bridge_orig"),
            run_root_dir=_resolve_path(
                pf.get("run_root_dir", "./data/outputs/policy_finetune/runs"),
                base_dir,
            ),
            adapter_tmp_dir=_resolve_path(
                pf.get("adapter_tmp_dir", "./data/outputs/policy_finetune/adapters"),
                base_dir,
            ),
            lora_rank=pf.get("lora_rank", 32),
            batch_size=pf.get("batch_size", 8),
            grad_accumulation_steps=pf.get("grad_accumulation_steps", 2),
            learning_rate=pf.get("learning_rate", 5e-4),
            save_steps=pf.get("save_steps", 1000),
            max_steps=pf.get("max_steps", 5000),
            image_aug=pf.get("image_aug", True),
            nproc_per_node=pf.get("nproc_per_node", 1),
            wandb_project=pf.get("wandb_project"),
            wandb_entity=pf.get("wandb_entity"),
            recipe=pf.get("recipe", "oft"),
            action_chunk_size=int(pf.get("action_chunk_size", 8)),
            use_continuous_actions=pf.get("use_continuous_actions", True),
            use_l1_regression=pf.get("use_l1_regression", True),
            parallel_decoding=pf.get("parallel_decoding", True),
            extra_args=[str(v) for v in pf.get("extra_args", [])],
        )

    if "policy_adapter" in raw:
        pa = raw["policy_adapter"]
        openvla_raw = pa.get("openvla", {}) if isinstance(pa, dict) else {}
        pi05_raw = pa.get("pi05", {}) if isinstance(pa, dict) else {}
        config.policy_adapter = PolicyAdapterConfig(
            name=pa.get("name", "openvla_oft"),
            openvla=OpenVLAAdapterBackendConfig(
                openvla_repo=_resolve_path(
                    openvla_raw.get("openvla_repo", config.policy_finetune.openvla_repo),
                    base_dir,
                ),
                finetune_script=str(
                    openvla_raw.get("finetune_script", config.policy_finetune.finetune_script)
                ),
                extra_train_args=[str(v) for v in openvla_raw.get("extra_train_args", [])],
            ),
            pi05=Pi05AdapterBackendConfig(
                openpi_repo=_resolve_path(
                    pi05_raw.get("openpi_repo", "/opt/openpi"),
                    base_dir,
                ),
                profile=str(pi05_raw.get("profile", "pi05_libero")),
                runtime_mode=str(pi05_raw.get("runtime_mode", "inprocess")),
                train_backend=str(pi05_raw.get("train_backend", "pytorch")),
                train_script=str(pi05_raw.get("train_script", "scripts/train_pytorch.py")),
                norm_stats_script=str(
                    pi05_raw.get("norm_stats_script", "scripts/compute_norm_stats.py")
                ),
                policy_action_dim=int(pi05_raw.get("policy_action_dim", 7)),
                policy_state_dim=int(pi05_raw.get("policy_state_dim", 7)),
                extra_train_args=[str(v) for v in pi05_raw.get("extra_train_args", [])],
            ),
        )
    else:
        # Keep adapter backend defaults synchronized with policy_finetune defaults
        # for legacy OpenVLA-only configs.
        config.policy_adapter = PolicyAdapterConfig(
            name=config.policy_adapter.name,
            openvla=OpenVLAAdapterBackendConfig(
                openvla_repo=config.policy_finetune.openvla_repo,
                finetune_script=config.policy_finetune.finetune_script,
                extra_train_args=[],
            ),
            pi05=config.policy_adapter.pi05,
        )

    if "robosplat" in raw:
        rs_full = raw["robosplat"]
        config.robosplat = RoboSplatConfig(
            enabled=rs_full.get("enabled", True),
            backend=str(rs_full.get("backend", "auto")),
            parity_mode=str(rs_full.get("parity_mode", "hybrid")),
            runtime_preset=str(rs_full.get("runtime_preset", "balanced")),
            variants_per_input=int(rs_full.get("variants_per_input", 4)),
            object_source_priority=[
                str(v)
                for v in rs_full.get(
                    "object_source_priority",
                    ["task_hints_obb", "vlm_detect", "cluster"],
                )
            ],
            demo_source=str(rs_full.get("demo_source", "synthetic")),
            demo_manifest_path=(
                _resolve_path(rs_full.get("demo_manifest_path"), base_dir)
                if rs_full.get("demo_manifest_path")
                else None
            ),
            min_successful_demos=int(rs_full.get("min_successful_demos", 4)),
            demo_success_task_score_threshold=float(
                rs_full.get("demo_success_task_score_threshold", 7.0)
            ),
            require_manipulation_success_flags=rs_full.get(
                "require_manipulation_success_flags", True
            ),
            world_model_bootstrap_enabled=rs_full.get("world_model_bootstrap_enabled", False),
            bootstrap_if_missing_demo=rs_full.get("bootstrap_if_missing_demo", True),
            bootstrap_num_rollouts=int(rs_full.get("bootstrap_num_rollouts", 6)),
            bootstrap_horizon_steps=int(rs_full.get("bootstrap_horizon_steps", 24)),
            bootstrap_tasks_limit=int(rs_full.get("bootstrap_tasks_limit", 4)),
            quality_gate_enabled=rs_full.get("quality_gate_enabled", True),
            min_variants_required_per_clip=int(rs_full.get("min_variants_required_per_clip", 1)),
            fallback_to_legacy_scan=rs_full.get("fallback_to_legacy_scan", True),
            fallback_on_backend_error=rs_full.get("fallback_on_backend_error", True),
            persist_scene_variants=rs_full.get("persist_scene_variants", False),
            vendor_repo_path=_resolve_path(
                rs_full.get("vendor_repo_path", "./vendor/robosplat"),
                base_dir,
            ),
            vendor_ref=str(rs_full.get("vendor_ref", "")),
        )

    if "robosplat_scan" in raw:
        rs = raw["robosplat_scan"]
        config.robosplat_scan = RoboSplatScanConfig(
            enabled=rs.get("enabled", True),
            num_augmented_clips_per_input=int(rs.get("num_augmented_clips_per_input", 2)),
            yaw_jitter_deg=float(rs.get("yaw_jitter_deg", 6.0)),
            pitch_jitter_deg=float(rs.get("pitch_jitter_deg", 4.0)),
            camera_height_jitter_m=float(rs.get("camera_height_jitter_m", 0.12)),
            relight_gain_min=float(rs.get("relight_gain_min", 0.85)),
            relight_gain_max=float(rs.get("relight_gain_max", 1.20)),
            color_temp_shift=rs.get("color_temp_shift", True),
            temporal_speed_factors=[float(v) for v in rs.get("temporal_speed_factors", [0.9, 1.1])],
        )
        if "robosplat" not in raw:
            # Legacy compatibility mapping for one release cycle.
            warnings.warn(
                (
                    "Config uses legacy `robosplat_scan`. Mapping to `robosplat` "
                    "with backend=legacy_scan for compatibility; migrate to "
                    "`robosplat` block."
                ),
                UserWarning,
                stacklevel=2,
            )
            config.robosplat = RoboSplatConfig(
                enabled=config.robosplat_scan.enabled,
                backend="legacy_scan",
                parity_mode="scan_only",
                runtime_preset="fast",
                variants_per_input=max(1, int(config.robosplat_scan.num_augmented_clips_per_input)),
                object_source_priority=["cluster"],
                demo_source="synthetic",
                demo_manifest_path=None,
                min_successful_demos=max(
                    1, int(config.robosplat_scan.num_augmented_clips_per_input)
                ),
                demo_success_task_score_threshold=7.0,
                require_manipulation_success_flags=True,
                world_model_bootstrap_enabled=False,
                bootstrap_if_missing_demo=False,
                bootstrap_num_rollouts=0,
                bootstrap_horizon_steps=0,
                bootstrap_tasks_limit=0,
                quality_gate_enabled=True,
                min_variants_required_per_clip=1,
                fallback_to_legacy_scan=True,
                fallback_on_backend_error=True,
                persist_scene_variants=False,
                vendor_repo_path=_resolve_path("./vendor/robosplat", base_dir),
                vendor_ref="",
            )

    if "splatsim" in raw:
        ss = raw["splatsim"]
        config.splatsim = SplatSimConfig(
            enabled=ss.get("enabled", False),
            mode=str(ss.get("mode", "hybrid")),
            per_zone_rollouts=int(ss.get("per_zone_rollouts", 2)),
            horizon_steps=int(ss.get("horizon_steps", 30)),
            min_successful_rollouts_per_zone=int(ss.get("min_successful_rollouts_per_zone", 1)),
            fallback_to_prior_manifest=ss.get("fallback_to_prior_manifest", True),
        )

    if "policy_rl_loop" in raw:
        pr = raw["policy_rl_loop"]
        config.policy_rl_loop = PolicyRLLoopConfig(
            enabled=pr.get("enabled", False),
            iterations=int(pr.get("iterations", 2)),
            horizon_steps=int(pr.get("horizon_steps", 24)),
            rollouts_per_task=int(pr.get("rollouts_per_task", 8)),
            group_size=int(pr.get("group_size", 4)),
            reward_mode=pr.get("reward_mode", "hybrid"),
            vlm_reward_fraction=float(pr.get("vlm_reward_fraction", 0.25)),
            top_quantile=float(pr.get("top_quantile", 0.30)),
            near_miss_min_quantile=float(pr.get("near_miss_min_quantile", 0.30)),
            near_miss_max_quantile=float(pr.get("near_miss_max_quantile", 0.60)),
            policy_refine_steps_per_iter=int(pr.get("policy_refine_steps_per_iter", 1000)),
            world_model_refresh_enabled=pr.get("world_model_refresh_enabled", True),
            world_model_refresh_epochs=int(pr.get("world_model_refresh_epochs", 3)),
            world_model_refresh_learning_rate=float(
                pr.get("world_model_refresh_learning_rate", 5.0e-5)
            ),
        )

    if "rollout_dataset" in raw:
        rd = raw["rollout_dataset"]
        config.rollout_dataset = RolloutDatasetConfig(
            enabled=rd.get("enabled", True),
            seed=rd.get("seed", 17),
            train_split=float(rd.get("train_split", 0.8)),
            min_steps_per_rollout=rd.get("min_steps_per_rollout", 4),
            task_score_threshold=float(rd.get("task_score_threshold", 7.0)),
            include_failed_rollouts=rd.get("include_failed_rollouts", False),
            max_action_delta_norm=float(rd.get("max_action_delta_norm", 5.0)),
            require_consistent_action_dim=rd.get("require_consistent_action_dim", True),
            baseline_dataset_name=rd.get(
                "baseline_dataset_name",
                "bridge_dataset",
            ),
            adapted_dataset_name=rd.get(
                "adapted_dataset_name",
                "bridge_orig",
            ),
            export_dir=_resolve_path(
                rd.get("export_dir", "./data/outputs/policy_datasets"),
                base_dir,
            ),
        )

    if "policy_compare" in raw:
        pc = raw["policy_compare"]
        config.policy_compare = PolicyCompareConfig(
            enabled=pc.get("enabled", False),
            heldout_num_rollouts=pc.get("heldout_num_rollouts", 20),
            heldout_seed=pc.get("heldout_seed", 123),
            eval_world_model=pc.get("eval_world_model", "adapted"),
            heldout_tasks=pc.get("heldout_tasks", []),
            task_score_success_threshold=float(pc.get("task_score_success_threshold", 7.0)),
            manipulation_task_keywords=pc.get(
                "manipulation_task_keywords",
                PolicyCompareConfig().manipulation_task_keywords,
            ),
            require_grasp_for_manipulation=pc.get(
                "require_grasp_for_manipulation",
                True,
            ),
            require_lift_for_manipulation=pc.get(
                "require_lift_for_manipulation",
                True,
            ),
            require_place_for_manipulation=pc.get(
                "require_place_for_manipulation",
                True,
            ),
        )

    # Visual fidelity
    if "eval_visual" in raw:
        ev = raw["eval_visual"]
        config.eval_visual = VisualFidelityConfig(
            metrics=ev.get("metrics", ["psnr", "ssim", "lpips"]),
            lpips_backbone=ev.get("lpips_backbone", "alex"),
        )

    # Spatial accuracy
    if "eval_spatial" in raw:
        es = raw["eval_spatial"]
        config.eval_spatial = SpatialAccuracyConfig(
            num_sample_frames=es.get("num_sample_frames", 20),
            vlm_model=es.get("vlm_model", "gemini-3-flash-preview"),
        )

    # Cross-site
    if "eval_crosssite" in raw:
        ec = raw["eval_crosssite"]
        config.eval_crosssite = CrossSiteConfig(
            num_clips_per_model=ec.get("num_clips_per_model", 30),
            vlm_model=ec.get("vlm_model", "gemini-3-flash-preview"),
        )

    # Cloud
    if "cloud" in raw:
        cl = raw["cloud"]
        config.cloud = CloudConfig(
            provider=cl.get("provider", "runpod"),
            gpu_type=cl.get("gpu_type", "H100"),
            num_gpus=cl.get("num_gpus", 1),
            max_cost_usd=cl.get("max_cost_usd", 500.0),
            auto_shutdown=cl.get("auto_shutdown", True),
        )

    return config
