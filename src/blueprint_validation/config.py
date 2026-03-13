"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import yaml

from blueprint_contracts.handoff_contract import load_and_validate_qualified_opportunity_handoff

from .config_validation import validate_config_keys


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
    ply_path: Optional[Path] = None
    evaluation_prep_path: Optional[Path] = None
    scene_memory_bundle_path: Optional[Path] = None
    preview_simulation_path: Optional[Path] = None
    opportunity_handoff_path: Optional[Path] = None
    geometry_bundle_path: Optional[Path] = None
    scene_package_path: Optional[Path] = None
    scene_memory_adapter_manifests: Dict[str, Path] = field(default_factory=dict)
    task_hints_path: Optional[Path] = None
    holi_spatial_grounding_path: Optional[Path] = None
    labels_path: Optional[Path] = None
    structure_path: Optional[Path] = None
    task_anchor_manifest_path: Optional[Path] = None
    object_geometry_manifest_path: Optional[Path] = None
    review_queue_path: Optional[Path] = None
    claim_benchmark_path: Optional[Path] = None
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
    # Stage-2 correction for RGB/depth control orientation.
    # Allowed: none|rotate180|hflip|vflip|hvflip
    video_orientation_fix: str = "none"
    intake_mode: str = "legacy_direct"
    qualification_state: Optional[str] = None
    downstream_evaluation_eligibility: Optional[bool] = None
    opportunity_id: str = ""
    site_submission_id: str = ""
    opportunity_handoff: Optional[Dict[str, Any]] = None

    @property
    def uses_qualified_handoff(self) -> bool:
        return self.opportunity_handoff_path is not None or self.evaluation_prep_path is not None

    @property
    def has_scene_memory_bundle(self) -> bool:
        return self.scene_memory_bundle_path is not None


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
    arc_span_deg: float = 150.0
    arc_phase_offset_deg: float = 0.0
    # internal metadata: where this spec came from
    source_tag: Optional[str] = None
    # optional target identity metadata
    target_instance_id: Optional[str] = None
    target_label: Optional[str] = None
    target_category: Optional[str] = None
    target_role: Optional[str] = None
    # Optional target OBB extents in meters (XYZ full widths).
    target_extents_m: Optional[List[float]] = None
    # Optional absolute world-space locked camera pose (used by scene-locked mode).
    locked_eye_point: Optional[List[float]] = None
    locked_look_at_point: Optional[List[float]] = None
    locked_probe_motion_radius_m: Optional[float] = None


@dataclass
class RenderConfig:
    backend: str = "auto"  # auto|gsplat|isaac_scene
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
    vlm_fallback: bool = False
    vlm_fallback_model: str = "gemini-3-flash-preview"
    vlm_fallback_num_views: int = 4
    # Task-scoped scene-aware camera generation (budget mode)
    task_scoped_scene_aware: bool = False
    task_scoped_max_specs: int = 40
    task_scoped_context_per_target: int = 2
    task_scoped_overview_specs: int = 6
    task_scoped_fallback_specs: int = 16
    task_scoped_profile: str = "dreamdojo"
    # Keep requested clip duration after occupancy/collision filtering drops poses.
    preserve_num_frames_after_collision_filter: bool = True
    # Stage-1 coverage density knobs for task-scoped camera generation.
    task_scoped_num_clips_per_path: int = 1
    task_scoped_num_frames_override: int = 0
    # Stage-1 coverage gate applied before Stage 2 enrichment.
    stage1_coverage_gate_enabled: bool = True
    stage1_coverage_min_visible_frame_ratio: float = 0.35
    stage1_coverage_min_approach_angle_bins: int = 2
    stage1_coverage_angle_bin_deg: float = 45.0
    stage1_coverage_blur_laplacian_min: float = 20.0
    stage1_coverage_blur_sample_every_n_frames: int = 5
    stage1_coverage_blur_max_samples_per_clip: int = 12
    stage1_coverage_min_center_band_ratio: float = 0.4
    stage1_coverage_center_band_x: List[float] = field(default_factory=lambda: [0.2, 0.8])
    stage1_coverage_center_band_y: List[float] = field(default_factory=lambda: [0.2, 0.8])
    # Stage-1 intrinsic camera-quality planning and retry controls.
    stage1_quality_planner_enabled: bool = True
    stage1_quality_candidate_budget: str = "medium"  # low|medium|high
    stage1_quality_autoretry_enabled: bool = True
    stage1_quality_max_regen_attempts: int = 2
    stage1_quality_min_clip_score: float = 0.55
    stage1_strict_require_task_hints: bool = False
    # Stage-1 active-perception loop (native-video VLM critic + deterministic corrections).
    stage1_active_perception_enabled: bool = True
    stage1_active_perception_scope: str = "all"  # all|targeted|manipulation
    stage1_active_perception_max_loops: int = 2
    stage1_active_perception_fail_closed: bool = True
    stage1_probe_frames_override: int = 0
    stage1_probe_resolution_scale: float = 0.0
    stage1_probe_min_viable_pose_ratio: float = 0.55
    stage1_probe_min_unique_positions: int = 8
    stage1_probe_dedupe_enabled: bool = True
    stage1_probe_dedupe_max_regen_attempts: int = 2
    stage1_probe_dedupe_center_dist_m: float = 0.08
    stage1_probe_consensus_votes: int = 3
    stage1_probe_consensus_high_variance_delta: float = 3.0
    stage1_probe_tiebreak_extra_votes: int = 2
    stage1_probe_tiebreak_spread_threshold: float = 3.0
    stage1_probe_primary_model_only: bool = True
    stage1_vlm_min_task_score: float = 7.0
    stage1_vlm_min_visual_score: float = 7.0
    stage1_vlm_min_spatial_score: float = 6.0
    stage1_keep_probe_videos: bool = False
    stage1_repeat_dedupe_enabled: bool = True
    stage1_repeat_dedupe_max_regen_attempts: int = 2
    stage1_repeat_min_xy_jitter_m: float = 0.06
    stage1_repeat_similarity_ssim_threshold: float = 0.995
    scene_locked_profile: str = "auto"  # auto|none|kitchen_0787|facility_a
    orientation_autocorrect_enabled: bool = True
    orientation_autocorrect_mode: str = "auto"  # auto|fail_fast|warn_only
    manipulation_random_xy_offset_m: float = 0.0
    non_manipulation_random_xy_offset_m: float = 1.0
    manipulation_target_z_bias_m: float = 0.0


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
    enabled: bool = False
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
    dynamic_variants: bool = False
    dynamic_variants_model: str = "gemini-3-flash-preview"
    # If false, Stage 2 fails when dynamic variant generation is unavailable.
    allow_dynamic_variant_fallback: bool = True
    # Optional explicit context frame index used for Cosmos image context anchoring.
    # If unset, Stage 2 uses a deterministic quarter-way frame for each clip.
    context_frame_index: Optional[int] = None
    # Context frame selection policy: target_centered (manipulation-aware) or fixed.
    context_frame_mode: str = "target_centered"
    # Optional upper bound on frames passed into Cosmos per clip.
    # Disabled when <= 0. When enabled and inputs are longer, Stage 2 trims a window
    # centered around the resolved context frame.
    max_input_frames: int = 0
    # Optional source clip selection budget. <= 0 keeps all clips.
    max_source_clips: int = 0
    # Strict minimum selected source clips after selection/filtering.
    min_source_clips: int = 8
    # Strict minimum accepted enriched outputs after quality gating.
    min_valid_outputs: int = 8
    # Maximum tolerated fraction of blur-gate rejects.
    max_blur_reject_rate: float = 0.30
    # Maximum tolerated fraction of green-dominant pixels in outputs.
    green_frame_ratio_max: float = 0.10
    # Enable per-output visual-collapse heuristics (blur/green cast/low motion).
    enable_visual_collapse_gate: bool = True
    # Native-video VLM quality gate for enriched outputs (Stage 2).
    # Disabled by default to require explicit opt-in before external upload.
    vlm_quality_gate_enabled: bool = False
    # Fail Stage 2 if VLM gate cannot be evaluated or clip never passes within retries.
    vlm_quality_fail_closed: bool = True
    # Enable bounded per-variant auto-regeneration for failed VLM quality checks.
    vlm_quality_autoretry_enabled: bool = True
    # Number of regeneration attempts after initial generation (per variant clip).
    vlm_quality_max_regen_attempts: int = 2
    # Minimum acceptable VLM scores for enriched outputs.
    vlm_quality_min_task_score: float = 7.0
    vlm_quality_min_visual_score: float = 7.0
    vlm_quality_min_spatial_score: float = 6.0
    # Require VLM reasoning to be consistent with scalar scores.
    vlm_quality_require_reasoning_consistency: bool = True
    # Context-frame shift (in frames) used in deterministic retry ladder.
    vlm_quality_retry_context_frame_stride: int = 6
    # On final retry, optionally disable depth control to recover from depth artifacts.
    vlm_quality_disable_depth_on_final_retry: bool = True
    # all|task_targeted|explicit
    source_clip_selection_mode: str = "all"
    # If true, task-targeted source selection fails the stage instead of silently falling back.
    source_clip_selection_fail_closed: bool = True
    source_clip_task: Optional[str] = None
    source_clip_name: Optional[str] = None
    # Optional multi-view image context construction for Cosmos.
    multi_view_context_enabled: bool = False
    multi_view_context_offsets: List[int] = field(default_factory=lambda: [-12, 0, 12])
    # Lightweight retrieval hook for scene-context frame suggestions.
    scene_index_enabled: bool = False
    scene_index_k: int = 3
    scene_index_sample_every_n_frames: int = 8
    # Output MP4 encoding quality passed to Cosmos inference runtime.
    cosmos_output_quality: int = 5
    # Acceptance gate for control-faithful enrichment. Disabled when <= 0.
    min_frame0_ssim: float = 0.0
    # Delete generated output files rejected by the frame-0 SSIM gate.
    delete_rejected_outputs: bool = False


@dataclass
class SceneMemoryBackendRuntimeConfig:
    enabled: bool = True
    allow_runtime_execution: bool = False
    repo_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    inference_script: Optional[str] = None
    checkpoint_path: Optional[Path] = None


@dataclass
class NeoVerseServiceConfig:
    enabled: bool = False
    service_url: Optional[str] = None
    api_key_env: str = "NEOVERSE_RUNTIME_SERVICE_API_KEY"
    timeout_seconds: int = 120
    websocket_base_url: Optional[str] = None


@dataclass
class SceneMemoryRuntimeConfig:
    enabled: bool = True
    preferred_backends: List[str] = field(
        default_factory=lambda: ["neoverse", "gen3c", "cosmos_transfer"]
    )
    watchlist_backends: List[str] = field(default_factory=lambda: ["3dsceneprompt"])
    allow_backend_fallback: bool = True
    neoverse_service: NeoVerseServiceConfig = field(default_factory=NeoVerseServiceConfig)
    neoverse: SceneMemoryBackendRuntimeConfig = field(
        default_factory=lambda: SceneMemoryBackendRuntimeConfig(inference_script="inference.py")
    )
    gen3c: SceneMemoryBackendRuntimeConfig = field(
        default_factory=lambda: SceneMemoryBackendRuntimeConfig(inference_script="inference.py")
    )


@dataclass
class DatasetPromptLintConfig:
    enabled: bool = True
    min_chars: int = 8
    min_tokens: int = 2
    min_unique_token_ratio: float = 0.35
    allow_generic_substrings: bool = False


@dataclass
class DatasetTemporalGateConfig:
    enabled: bool = True
    min_frames_for_check: int = 8
    max_frames_to_sample: int = 96
    min_mean_interframe_delta: float = 1.5
    max_freeze_ratio: float = 0.70
    max_abrupt_cut_ratio: float = 0.35
    max_blockiness_score: float = 0.45


@dataclass
class DatasetDistributionConfig:
    enabled: bool = True
    min_total_clips_for_caps: int = 16
    min_unique_variants: int = 2
    min_unique_source_clips: int = 4
    max_single_variant_fraction: float = 0.85
    max_single_source_clip_fraction: float = 0.60
    max_prompt_dominance_fraction: float = 0.70


@dataclass
class DatasetQualityConfig:
    strict_manifest_validation: bool = True
    quarantine_rejections: bool = True
    fail_on_rejections: bool = True
    max_reject_fraction: float = 0.50
    enable_duplicate_detection: bool = True
    enable_leakage_detection: bool = True
    prompt_lint: DatasetPromptLintConfig = field(default_factory=DatasetPromptLintConfig)
    temporal_gates: DatasetTemporalGateConfig = field(default_factory=DatasetTemporalGateConfig)
    distribution: DatasetDistributionConfig = field(default_factory=DatasetDistributionConfig)


@dataclass
class FinetuneConfig:
    dreamdojo_repo: Path = Path("/opt/DreamDojo")
    dreamdojo_checkpoint: Path = Path("./data/checkpoints/DreamDojo/2B_pretrain/")
    # Optional isolated Python runtime for DreamDojo Stage 3 (for dependency pinning).
    python_executable: Optional[Path] = None
    experiment_config: Optional[str] = None  # DreamDojo experiment config name
    # Optional explicit cosmos action-conditioned experiment used for Stage 4 world-model eval.
    eval_world_experiment: Optional[str] = None
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
    dataset_quality: DatasetQualityConfig = field(default_factory=DatasetQualityConfig)


@dataclass
class VLMJudgeConfig:
    model: str = "gemini-3-flash-preview"
    fallback_models: List[str] = field(
        default_factory=lambda: ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash"]
    )
    api_key_env: str = "GOOGLE_GENAI_API_KEY"
    enable_agentic_vision: bool = True
    # Explicit Gemini File API video metadata FPS for scoring calls (0 disables explicit fps).
    video_metadata_fps: float = 10.0
    scoring_prompt: str = (
        "You are evaluating a robot policy rollout video.\n"
        "Score the following on a 1-10 scale:\n"
        '1. Task completion (did the robot accomplish "{task}"?)\n'
        "2. Visual plausibility (does the environment look realistic?)\n"
        "3. Spatial coherence (are objects in consistent positions across frames?)\n"
        'Return JSON: {{"task_score": N, "visual_score": N, "spatial_score": N, "reasoning": "..."}}'
    )


@dataclass
class PolicyEvalReliabilityConfig:
    max_horizon_steps: int = 12
    keyframe_reanchor_every: int = 4
    min_replay_pass_rate: float = 0.70
    min_controllability_pass_rate: float = 0.70
    enforce_stage_success: bool = False
    max_scoring_failure_rate: float = 0.02
    fail_on_short_rollout: bool = False
    min_rollout_frames: int = 13
    min_rollout_steps: int = 12


@dataclass
class ClaimReplicationConfig:
    training_seeds: List[int] = field(default_factory=lambda: list(range(8)))


@dataclass
class ClaimStrictnessConfig:
    min_eval_task_specs: int = 3
    min_eval_start_clips: int = 3
    min_common_eval_cells: int = 30
    min_positive_training_seeds: int = 4
    p_value_threshold: float = 0.05
    require_site_specific_advantage: bool = True
    site_vs_generic_min_lift_pp: float = 0.0


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
    headline_scope: str = "wm_only"  # wm_only|wm_uplift|dual
    rollout_driver: str = "scripted"  # scripted|stress|both
    scripted_rollouts_per_task: int = 12
    mode: str = "claim"  # claim|research
    required_action_dim: int = 7
    manip_eval_mode: str = "overlay_marker"  # overlay_marker|raw
    min_assignment_quality_score: float = 0.0
    require_object_grounded_manip_tasks: bool = True
    min_absolute_difference: float = 1.0  # minimum raw score difference for PASS
    min_manip_success_delta_pp: float = 15.0
    require_native_action_compat: bool = True
    claim_protocol: str = "none"  # none|fixed_same_facility_uplift
    primary_endpoint: str = "vlm_task_score"  # vlm_task_score|task_success
    freeze_world_snapshot: bool = False
    split_strategy: str = "legacy"  # legacy|disjoint_tasks_and_starts
    min_practical_success_lift_pp: float = 5.0
    claim_replication: ClaimReplicationConfig = field(default_factory=ClaimReplicationConfig)
    claim_strictness: ClaimStrictnessConfig = field(default_factory=ClaimStrictnessConfig)
    reliability: PolicyEvalReliabilityConfig = field(default_factory=PolicyEvalReliabilityConfig)
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
    seed: int = 0
    extra_args: List[str] = field(default_factory=list)


@dataclass
class OpenVLAAdapterBackendConfig:
    openvla_repo: Path = Path("/opt/openvla-oft")
    finetune_script: str = "vla-scripts/finetune.py"
    base_model_name: Optional[str] = None
    base_checkpoint_path: Optional[Path] = None
    policy_action_dim: int = 7
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
    strict_action_contract: bool = True
    allow_synthetic_state_for_eval: bool = False
    extra_train_args: List[str] = field(default_factory=list)


@dataclass
class DreamZeroAdapterBackendConfig:
    repo_path: Path = Path("/opt/dreamzero")
    base_model_name: Optional[str] = None
    checkpoint_path: Path = Path("./data/checkpoints/dreamzero/")
    # Import target used by the adapter to instantiate inference runtime.
    inference_module: str = "dreamzero.inference"
    inference_class: str = "DreamZeroInference"
    policy_action_dim: int = 7
    frame_history: int = 4
    strict_action_contract: bool = True
    strict_action_min: float = -1.0
    strict_action_max: float = 1.0
    train_script: str = "scripts/train.py"
    extra_train_args: List[str] = field(default_factory=list)
    # Keep policy training disabled by default in pragmatic single-GPU mode.
    allow_training: bool = False


@dataclass
class PolicyAdapterConfig:
    name: str = "openvla_oft"
    openvla: OpenVLAAdapterBackendConfig = field(default_factory=OpenVLAAdapterBackendConfig)
    pi05: Pi05AdapterBackendConfig = field(default_factory=Pi05AdapterBackendConfig)
    dreamzero: DreamZeroAdapterBackendConfig = field(default_factory=DreamZeroAdapterBackendConfig)


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
class ExternalInteractionConfig:
    enabled: bool = True
    manifest_path: Optional[Path] = None
    source_name: str = "external"


@dataclass
class ExternalRolloutsConfig:
    enabled: bool = False
    manifest_path: Optional[Path] = None
    source_name: str = "teleop"
    # Current pipeline wiring ingests these sessions only into policy-training datasets.
    # wm_only|wm_and_policy are accepted for forward compatibility, but WM ingestion is advisory-only.
    mode: str = "wm_and_policy"  # policy_only|wm_only|wm_and_policy


@dataclass
class NativeTeacherConfig:
    enabled: bool = False
    include_generic_control: bool = True
    generate_corrections: bool = True
    planner_horizon_steps: int = 16


@dataclass
class ClaimPortfolioConfig:
    min_facilities: int = 3
    min_mean_site_vs_frozen_lift_pp: float = 8.0
    min_mean_site_vs_generic_lift_pp: float = 2.0
    max_negative_task_family_delta_pp: float = -5.0
    require_manipulation_nonzero: bool = True


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
    policy_refine_near_miss_fraction: float = 0.30
    policy_refine_hard_negative_fraction: float = 0.10
    world_model_refresh_enabled: bool = True
    world_model_refresh_mix_with_stage2: bool = True
    world_model_refresh_require_stage2_vlm_pass: bool = True
    world_model_refresh_stage2_fraction: float = 0.60
    world_model_refresh_success_fraction: float = 0.25
    world_model_refresh_near_miss_fraction: float = 0.15
    world_model_refresh_min_total_clips: int = 128
    world_model_refresh_max_total_clips: int = 512
    world_model_refresh_seed: int = 17
    world_model_refresh_epochs: int = 3
    world_model_refresh_learning_rate: float = 5.0e-5


@dataclass
class WorldModelRefreshLoopConfig:
    enabled: bool = True
    iterations: int = 1
    source_condition: str = "adapted"  # baseline|adapted
    fail_if_refresh_fails: bool = True
    fail_on_degenerate_mix: bool = True
    min_non_hard_rollouts: int = 8
    max_hard_negative_fraction: float = 0.75
    require_valid_video_decode: bool = True
    enforce_vlm_quality_floor: bool = True
    min_refresh_task_score: float = 7.0
    min_refresh_visual_score: float = 7.0
    min_refresh_spatial_score: float = 6.0
    fail_on_reasoning_conflict: bool = True
    backfill_from_stage2_vlm_passed: bool = True
    quantile_fallback_enabled: bool = True
    quantile_success_threshold: float = 0.85
    quantile_near_miss_threshold: float = 0.50


@dataclass
class RolloutDatasetConfig:
    enabled: bool = True
    seed: int = 17
    train_split: float = 0.8
    min_steps_per_rollout: int = 4
    task_score_threshold: float = 7.0
    include_failed_rollouts: bool = False
    selection_mode: str = "success_near_miss"  # success_only|success_near_miss|success_near_miss_hard
    near_miss_min_task_score: float = 5.0
    near_miss_max_task_score: float = 6.99
    near_miss_target_fraction: float = 0.30
    hard_negative_target_fraction: float = 0.00
    per_task_max_episodes: int = 0
    max_action_delta_norm: float = 5.0
    require_consistent_action_dim: bool = True
    baseline_dataset_name: str = "bridge_dataset"  # Must exist in vendored OpenVLA OXE registry.
    adapted_dataset_name: str = "bridge_orig"  # Must exist in vendored OpenVLA OXE registry.
    export_dir: Path = Path("./data/outputs/policy_datasets")


@dataclass
class ActionBoostConfig:
    enabled: bool = True
    require_full_pipeline: bool = True
    # Legacy key name retained for compatibility; runtime auto-switch now targets wm_uplift.
    auto_switch_headline_scope_to_dual: bool = True
    auto_enable_rollout_dataset: bool = True
    auto_enable_policy_finetune: bool = True
    auto_enable_policy_rl_loop: bool = True
    compute_profile: str = "standard"  # lean|standard|aggressive
    strict_disjoint_eval: bool = True


@dataclass
class PolicyCompareConfig:
    enabled: bool = False
    heldout_num_rollouts: int = 20
    heldout_seed: int = 123
    eval_world_model: str = "adapted"
    heldout_tasks: List[str] = field(default_factory=list)
    control_arms: List[str] = field(
        default_factory=lambda: ["frozen_baseline", "site_trained", "generic_control"]
    )
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
    min_valid_samples: int = 3
    fail_on_reasoning_conflict: bool = True


@dataclass
class PolarisEvalConfig:
    enabled: bool = False
    repo_path: Path = Path("/opt/PolaRiS")
    hub_path: Path = Path("./PolaRiS-Hub")
    environment_mode: str = "scene_package_bridge"
    environment_name: Optional[str] = None
    default_as_primary_gate: bool = True
    use_for_claim_gate: bool = True
    num_rollouts: int = 16
    device: str = "cuda"
    policy_client: str = "OpenVLA"
    observation_mode: str = "external_only"
    action_mode: str = "native"
    export_dir: Path = Path("./data/outputs/polaris")
    require_scene_package: bool = True
    require_success_correlation_metadata: bool = True


@dataclass
class SceneBuilderConfig:
    enabled: bool = False
    source_ply_path: Optional[Path] = None
    output_scene_root: Path = Path("./data/scene_package")
    static_collision_mode: str = "simple"
    asset_manifest_path: Optional[Path] = None
    scene_edit_manifest_path: Optional[Path] = None
    task_hints_path: Optional[Path] = None
    robot_type: str = "franka"
    task_template: str = "pick_place_v1"
    emit_isaac_lab: bool = True
    emit_polaris_metadata: bool = True
    fail_on_physics_qc: bool = False


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
    scene_memory_runtime: SceneMemoryRuntimeConfig = field(
        default_factory=SceneMemoryRuntimeConfig
    )
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval_policy: PolicyEvalConfig = field(default_factory=PolicyEvalConfig)
    eval_polaris: PolarisEvalConfig = field(default_factory=PolarisEvalConfig)
    scene_builder: SceneBuilderConfig = field(default_factory=SceneBuilderConfig)
    policy_finetune: PolicyFinetuneConfig = field(default_factory=PolicyFinetuneConfig)
    policy_adapter: PolicyAdapterConfig = field(default_factory=PolicyAdapterConfig)
    robosplat: RoboSplatConfig = field(default_factory=RoboSplatConfig)
    robosplat_scan: RoboSplatScanConfig = field(default_factory=RoboSplatScanConfig)
    external_interaction: ExternalInteractionConfig = field(default_factory=ExternalInteractionConfig)
    external_rollouts: ExternalRolloutsConfig = field(default_factory=ExternalRolloutsConfig)
    native_teacher: NativeTeacherConfig = field(default_factory=NativeTeacherConfig)
    claim_portfolio: ClaimPortfolioConfig = field(default_factory=ClaimPortfolioConfig)
    action_boost: ActionBoostConfig = field(default_factory=ActionBoostConfig)
    policy_rl_loop: PolicyRLLoopConfig = field(default_factory=PolicyRLLoopConfig)
    wm_refresh_loop: WorldModelRefreshLoopConfig = field(default_factory=WorldModelRefreshLoopConfig)
    rollout_dataset: RolloutDatasetConfig = field(default_factory=RolloutDatasetConfig)
    policy_compare: PolicyCompareConfig = field(default_factory=PolicyCompareConfig)
    eval_visual: VisualFidelityConfig = field(default_factory=VisualFidelityConfig)
    eval_spatial: SpatialAccuracyConfig = field(default_factory=SpatialAccuracyConfig)
    eval_crosssite: CrossSiteConfig = field(default_factory=CrossSiteConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)

    @property
    def qualified_opportunities(self) -> Dict[str, FacilityConfig]:
        """Preferred public alias for downstream evaluation targets."""
        return self.facilities


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_optional_path(path_value: Any, base_dir: Path) -> Optional[Path]:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    return _resolve_path(raw, base_dir)


def _mapping_text(payload: Dict[str, Any], key: str) -> str:
    return str(payload.get(key, "") or "").strip()


def _resolve_handoff_nested_path(
    payload: Optional[Dict[str, Any]],
    base_dir: Path,
    *keys: str,
) -> Optional[Path]:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        resolved = _resolve_optional_path(value, base_dir)
        if resolved is not None:
            return resolved
    return None


def _resolve_existing_bundle_member(bundle_root: Optional[Path], filename: str) -> Optional[Path]:
    if bundle_root is None:
        return None
    candidate = bundle_root / filename
    if candidate.exists():
        return candidate
    return None


def _load_evaluation_prep_manifest(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("schema_version") or "").strip() != "v1":
        return None
    return payload


def _resolve_evaluation_prep_artifact(
    payload: Optional[Dict[str, Any]],
    base_dir: Path,
    key: str,
) -> Optional[Path]:
    if not isinstance(payload, dict):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    return _resolve_optional_path(artifacts.get(key), base_dir)


def _infer_capture_pipeline_geometry_bundle_path(
    handoff_path: Optional[Path],
    handoff_payload: Optional[Dict[str, Any]],
) -> Optional[Path]:
    if handoff_path is None or not isinstance(handoff_payload, dict):
        return None
    if "scene_id" not in handoff_payload or "capture_id" not in handoff_payload:
        return None
    candidate = handoff_path.parent / "advanced_geometry"
    if candidate.exists():
        return candidate.resolve()
    return None


def _infer_capture_pipeline_evaluation_prep_path(
    handoff_path: Optional[Path],
) -> Optional[Path]:
    if handoff_path is None:
        return None
    candidate = handoff_path.parent / "evaluation_prep" / "evaluation_prep_manifest.json"
    if candidate.exists():
        return candidate.resolve()
    return None


def _infer_capture_pipeline_scene_memory_bundle_path(
    handoff_path: Optional[Path],
) -> Optional[Path]:
    if handoff_path is None:
        return None
    candidate = handoff_path.parent / "scene_memory"
    manifest = candidate / "scene_memory_manifest.json"
    if manifest.exists():
        return candidate.resolve()
    return None


def _infer_capture_pipeline_preview_simulation_path(
    handoff_path: Optional[Path],
) -> Optional[Path]:
    if handoff_path is None:
        return None
    candidate = handoff_path.parent / "preview_simulation"
    manifest = candidate / "preview_simulation_manifest.json"
    if manifest.exists():
        return candidate.resolve()
    return None


def _resolve_scene_memory_adapter_manifests(
    raw: Dict[str, Any],
    *,
    base_dir: Path,
    scene_memory_bundle_path: Optional[Path],
) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    raw_mapping = (
        dict(raw.get("scene_memory_adapter_manifests", {}))
        if isinstance(raw.get("scene_memory_adapter_manifests"), dict)
        else {}
    )
    for adapter_id, raw_path in raw_mapping.items():
        path = _resolve_optional_path(raw_path, base_dir)
        if path is not None:
            resolved[str(adapter_id)] = path
    if scene_memory_bundle_path is not None:
        adapter_dir = scene_memory_bundle_path / "adapter_manifests"
        for adapter_id in ("gen3c", "neoverse", "cosmos_transfer"):
            path = adapter_dir / f"{adapter_id}.json"
            if path.exists():
                resolved.setdefault(adapter_id, path.resolve())
    return resolved


def _resolve_facility_paths(
    raw: Dict[str, Any],
    *,
    base_dir: Path,
    evaluation_prep_path: Optional[Path],
    evaluation_prep_payload: Optional[Dict[str, Any]],
    handoff_path: Optional[Path],
    handoff_payload: Optional[Dict[str, Any]],
) -> Dict[str, Optional[Path]]:
    prep_base_dir = evaluation_prep_path.parent if evaluation_prep_path is not None else base_dir
    handoff_base_dir = handoff_path.parent if handoff_path is not None else prep_base_dir
    handoff_geometry = (
        dict(handoff_payload.get("geometry_package", {}))
        if isinstance(handoff_payload, dict) and isinstance(handoff_payload.get("geometry_package"), dict)
        else {}
    )
    handoff_scene = (
        dict(handoff_payload.get("scene_package", {}))
        if isinstance(handoff_payload, dict) and isinstance(handoff_payload.get("scene_package"), dict)
        else {}
    )
    handoff_scene_memory = (
        dict(handoff_payload.get("scene_memory_package", {}))
        if isinstance(handoff_payload, dict) and isinstance(handoff_payload.get("scene_memory_package"), dict)
        else {}
    )
    evaluation_geometry_manifest_path = _resolve_evaluation_prep_artifact(
        evaluation_prep_payload,
        prep_base_dir,
        "geometry_bundle_manifest",
    )
    evaluation_geometry = (
        _load_evaluation_prep_manifest(evaluation_geometry_manifest_path)
        if evaluation_geometry_manifest_path is not None
        else None
    )

    geometry_bundle_path = _resolve_optional_path(raw.get("geometry_bundle_path"), base_dir)
    if geometry_bundle_path is None:
        geometry_bundle_path = _resolve_optional_path(
            (evaluation_geometry or {}).get("bundle_path"),
            prep_base_dir,
        )
    if geometry_bundle_path is None:
        geometry_bundle_path = _resolve_handoff_nested_path(
            handoff_geometry,
            handoff_base_dir,
            "bundle_path",
            "root_path",
        )
    if geometry_bundle_path is None:
        geometry_bundle_path = _infer_capture_pipeline_geometry_bundle_path(handoff_path, handoff_payload)

    scene_memory_bundle_path = _resolve_optional_path(raw.get("scene_memory_bundle_path"), base_dir)
    if scene_memory_bundle_path is None:
        scene_memory_bundle_path = _resolve_handoff_nested_path(
            handoff_scene_memory,
            handoff_base_dir,
            "bundle_path",
            "root_path",
        )
    if scene_memory_bundle_path is None:
        scene_memory_bundle_path = _infer_capture_pipeline_scene_memory_bundle_path(handoff_path)

    preview_simulation_path = _resolve_optional_path(raw.get("preview_simulation_path"), base_dir)
    if preview_simulation_path is None:
        preview_simulation_path = _infer_capture_pipeline_preview_simulation_path(handoff_path)

    ply_path = _resolve_optional_path(raw.get("ply_path"), base_dir)
    if ply_path is None:
        ply_path = _resolve_optional_path((evaluation_geometry or {}).get("ply_path"), prep_base_dir)
    if ply_path is None:
        ply_path = _resolve_handoff_nested_path(
            handoff_geometry,
            handoff_base_dir,
            "ply_path",
        )
    if ply_path is None and geometry_bundle_path is not None:
        ply_path = geometry_bundle_path / "3dgs_compressed.ply"

    scene_package_path = _resolve_optional_path(raw.get("scene_package_path"), base_dir)
    if scene_package_path is None:
        scene_package_path = _resolve_handoff_nested_path(
            handoff_scene,
            handoff_base_dir,
            "scene_package_path",
            "root_path",
            "bundle_path",
        )

    task_hints_path = _resolve_optional_path(raw.get("task_hints_path"), base_dir)
    if task_hints_path is None:
        task_hints_path = _resolve_optional_path((evaluation_geometry or {}).get("task_hints_path"), prep_base_dir)
    if task_hints_path is None:
        task_hints_path = _resolve_handoff_nested_path(
            handoff_geometry,
            handoff_base_dir,
            "task_hints_path",
        )
    if task_hints_path is None:
        task_hints_path = _resolve_existing_bundle_member(
            geometry_bundle_path,
            "task_targets.synthetic.json",
        )

    holi_spatial_grounding_path = _resolve_optional_path(raw.get("holi_spatial_grounding_path"), base_dir)
    if holi_spatial_grounding_path is None:
        holi_spatial_grounding_path = _resolve_optional_path(
            (evaluation_geometry or {}).get("holi_spatial_grounding_path"),
            prep_base_dir,
        )
    if holi_spatial_grounding_path is None:
        holi_spatial_grounding_path = _resolve_handoff_nested_path(
            handoff_geometry,
            handoff_base_dir,
            "holi_spatial_grounding_path",
        )
    if holi_spatial_grounding_path is None:
        holi_spatial_grounding_path = _resolve_existing_bundle_member(
            geometry_bundle_path,
            "holi_spatial_grounding.json",
        )

    labels_path = _resolve_optional_path((evaluation_geometry or {}).get("labels_path"), prep_base_dir)
    if labels_path is None:
        labels_path = _resolve_handoff_nested_path(
            handoff_geometry,
            handoff_base_dir,
            "labels_path",
        )
    if labels_path is None:
        labels_path = _resolve_existing_bundle_member(geometry_bundle_path, "labels.json")

    structure_path = _resolve_optional_path((evaluation_geometry or {}).get("structure_path"), prep_base_dir)
    if structure_path is None:
        structure_path = _resolve_handoff_nested_path(
            handoff_geometry,
            handoff_base_dir,
            "structure_path",
        )
    if structure_path is None:
        structure_path = _resolve_existing_bundle_member(geometry_bundle_path, "structure.json")

    task_anchor_manifest_path = _resolve_evaluation_prep_artifact(
        evaluation_prep_payload,
        prep_base_dir,
        "task_anchor_manifest",
    )
    object_geometry_manifest_path = _resolve_evaluation_prep_artifact(
        evaluation_prep_payload,
        prep_base_dir,
        "object_geometry_manifest",
    )
    review_queue_path = _resolve_evaluation_prep_artifact(
        evaluation_prep_payload,
        prep_base_dir,
        "review_queue",
    )

    return {
        "evaluation_prep_path": evaluation_prep_path,
        "scene_memory_bundle_path": scene_memory_bundle_path,
        "preview_simulation_path": preview_simulation_path,
        "geometry_bundle_path": geometry_bundle_path,
        "ply_path": ply_path,
        "scene_package_path": scene_package_path,
        "task_hints_path": task_hints_path,
        "holi_spatial_grounding_path": holi_spatial_grounding_path,
        "labels_path": labels_path,
        "structure_path": structure_path,
        "task_anchor_manifest_path": task_anchor_manifest_path,
        "object_geometry_manifest_path": object_geometry_manifest_path,
        "review_queue_path": review_queue_path,
    }


def _merge_target_sections(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, str] = {}
    for section_name in ("qualified_opportunities", "facilities"):
        section = raw.get(section_name, {}) or {}
        if not isinstance(section, dict):
            continue
        for target_id, target_raw in section.items():
            target_mapping = dict(target_raw)
            if target_id in merged:
                if merged[target_id] != target_mapping:
                    raise ValueError(
                        "Config target "
                        f"'{target_id}' is defined differently in both "
                        f"{sources[target_id]} and {section_name}"
                    )
                continue
            merged[target_id] = target_mapping
            sources[target_id] = section_name
    return merged


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


_ALLOWED_VIDEO_ORIENTATION_FIXES = {"none", "rotate180", "hflip", "vflip", "hvflip"}


def _parse_video_orientation_fix(raw_value: Any) -> str:
    value = str(raw_value or "none").strip().lower()
    if value not in _ALLOWED_VIDEO_ORIENTATION_FIXES:
        allowed = ", ".join(sorted(_ALLOWED_VIDEO_ORIENTATION_FIXES))
        raise ValueError(f"facility.video_orientation_fix must be one of: {allowed}")
    return value


_ALLOWED_SOURCE_CLIP_SELECTION_MODES = {"all", "task_targeted", "explicit"}
_ALLOWED_ENRICH_CONTEXT_FRAME_MODES = {"target_centered", "fixed"}


def _parse_source_clip_selection_mode(raw_value: Any) -> str:
    value = str(raw_value or "all").strip().lower()
    if value not in _ALLOWED_SOURCE_CLIP_SELECTION_MODES:
        allowed = ", ".join(sorted(_ALLOWED_SOURCE_CLIP_SELECTION_MODES))
        raise ValueError(f"enrich.source_clip_selection_mode must be one of: {allowed}")
    return value


def _parse_enrich_context_frame_mode(raw_value: Any) -> str:
    value = str(raw_value or "target_centered").strip().lower()
    if value not in _ALLOWED_ENRICH_CONTEXT_FRAME_MODES:
        allowed = ", ".join(sorted(_ALLOWED_ENRICH_CONTEXT_FRAME_MODES))
        raise ValueError(f"enrich.context_frame_mode must be one of: {allowed}")
    return value


_ALLOWED_ACTION_BOOST_COMPUTE_PROFILES = {"lean", "standard", "aggressive"}


def _parse_action_boost_compute_profile(raw_value: Any) -> str:
    value = str(raw_value or "standard").strip().lower()
    if value not in _ALLOWED_ACTION_BOOST_COMPUTE_PROFILES:
        allowed = ", ".join(sorted(_ALLOWED_ACTION_BOOST_COMPUTE_PROFILES))
        raise ValueError(f"action_boost.compute_profile must be one of: {allowed}")
    return value


_ALLOWED_ROLLOUT_SELECTION_MODES = {"success_only", "success_near_miss", "success_near_miss_hard"}


def _parse_rollout_selection_mode(raw_value: Any) -> str:
    value = str(raw_value or "success_near_miss").strip().lower()
    if value not in _ALLOWED_ROLLOUT_SELECTION_MODES:
        allowed = ", ".join(sorted(_ALLOWED_ROLLOUT_SELECTION_MODES))
        raise ValueError(f"rollout_dataset.selection_mode must be one of: {allowed}")
    return value


_ALLOWED_WM_REFRESH_SOURCE_CONDITIONS = {"baseline", "adapted"}


_ALLOWED_MANIP_EVAL_MODES = {"overlay_marker", "raw"}
_ALLOWED_STAGE1_QUALITY_CANDIDATE_BUDGETS = {"low", "medium", "high"}
_ALLOWED_STAGE1_ACTIVE_PERCEPTION_SCOPES = {"all", "targeted", "manipulation"}
_ALLOWED_SCENE_LOCKED_PROFILES = {"auto", "none", "kitchen_0787", "facility_a"}
_ALLOWED_CLAIM_PROTOCOLS = {"none", "fixed_same_facility_uplift"}
_ALLOWED_PRIMARY_ENDPOINTS = {"vlm_task_score", "task_success"}
_ALLOWED_CLAIM_SPLIT_STRATEGIES = {"legacy", "disjoint_tasks_and_starts"}


def _parse_manip_eval_mode(raw_value: Any) -> str:
    value = str(raw_value or "overlay_marker").strip().lower()
    if value not in _ALLOWED_MANIP_EVAL_MODES:
        allowed = ", ".join(sorted(_ALLOWED_MANIP_EVAL_MODES))
        raise ValueError(f"eval_policy.manip_eval_mode must be one of: {allowed}")
    return value


def _parse_stage1_quality_candidate_budget(raw_value: Any) -> str:
    value = str(raw_value or "medium").strip().lower()
    if value not in _ALLOWED_STAGE1_QUALITY_CANDIDATE_BUDGETS:
        allowed = ", ".join(sorted(_ALLOWED_STAGE1_QUALITY_CANDIDATE_BUDGETS))
        raise ValueError(f"render.stage1_quality_candidate_budget must be one of: {allowed}")
    return value


def _parse_stage1_active_perception_scope(raw_value: Any) -> str:
    value = str(raw_value or "all").strip().lower()
    if value not in _ALLOWED_STAGE1_ACTIVE_PERCEPTION_SCOPES:
        allowed = ", ".join(sorted(_ALLOWED_STAGE1_ACTIVE_PERCEPTION_SCOPES))
        raise ValueError(f"render.stage1_active_perception_scope must be one of: {allowed}")
    return value


def _parse_scene_locked_profile(raw_value: Any) -> str:
    value = str(raw_value or "auto").strip().lower()
    if value in {"off", "false", "disabled"}:
        value = "none"
    if value not in _ALLOWED_SCENE_LOCKED_PROFILES:
        allowed = ", ".join(sorted(_ALLOWED_SCENE_LOCKED_PROFILES))
        raise ValueError(f"render.scene_locked_profile must be one of: {allowed}")
    return value


def _parse_claim_protocol(raw_value: Any) -> str:
    value = str(raw_value or "none").strip().lower()
    if value not in _ALLOWED_CLAIM_PROTOCOLS:
        allowed = ", ".join(sorted(_ALLOWED_CLAIM_PROTOCOLS))
        raise ValueError(f"eval_policy.claim_protocol must be one of: {allowed}")
    return value


def _parse_primary_endpoint(raw_value: Any) -> str:
    value = str(raw_value or "vlm_task_score").strip().lower()
    if value not in _ALLOWED_PRIMARY_ENDPOINTS:
        allowed = ", ".join(sorted(_ALLOWED_PRIMARY_ENDPOINTS))
        raise ValueError(f"eval_policy.primary_endpoint must be one of: {allowed}")
    return value


def _parse_claim_split_strategy(raw_value: Any) -> str:
    value = str(raw_value or "legacy").strip().lower()
    if value not in _ALLOWED_CLAIM_SPLIT_STRATEGIES:
        allowed = ", ".join(sorted(_ALLOWED_CLAIM_SPLIT_STRATEGIES))
        raise ValueError(f"eval_policy.split_strategy must be one of: {allowed}")
    return value


def _parse_wm_refresh_source_condition(raw_value: Any) -> str:
    value = str(raw_value or "adapted").strip().lower()
    if value not in _ALLOWED_WM_REFRESH_SOURCE_CONDITIONS:
        allowed = ", ".join(sorted(_ALLOWED_WM_REFRESH_SOURCE_CONDITIONS))
        raise ValueError(f"wm_refresh_loop.source_condition must be one of: {allowed}")
    return value


def _min_sign_flip_seed_count(p_value_threshold: float) -> int:
    """Return the minimum seed count needed for an exact two-sided sign-flip test."""
    threshold = float(p_value_threshold)
    for seed_count in range(1, 65):
        min_p = 3.0 / ((2**seed_count) + 1.0)
        if min_p < threshold:
            return seed_count
    return 65


def _parse_facility(facility_id: str, raw: Dict[str, Any], base_dir: Path) -> FacilityConfig:
    evaluation_prep_path = _resolve_optional_path(raw.get("evaluation_prep_path"), base_dir)
    handoff_path = _resolve_optional_path(raw.get("opportunity_handoff_path"), base_dir)
    if evaluation_prep_path is None:
        evaluation_prep_path = _infer_capture_pipeline_evaluation_prep_path(handoff_path)
    evaluation_prep_payload = _load_evaluation_prep_manifest(evaluation_prep_path)
    if handoff_path is None:
        handoff_path = _resolve_evaluation_prep_artifact(
            evaluation_prep_payload,
            evaluation_prep_path.parent if evaluation_prep_path is not None else base_dir,
            "qualified_opportunity_handoff",
        )
    handoff_payload = (
        load_and_validate_qualified_opportunity_handoff(handoff_path)
        if handoff_path is not None
        else None
    )
    resolved_paths = _resolve_facility_paths(
        raw,
        base_dir=base_dir,
        evaluation_prep_path=evaluation_prep_path,
        evaluation_prep_payload=evaluation_prep_payload,
        handoff_path=handoff_path,
        handoff_payload=handoff_payload,
    )

    claim_benchmark_value = raw.get("claim_benchmark_path")
    intake_mode = "qualified_opportunity" if handoff_path is not None else "legacy_direct"
    name = _mapping_text(raw, "name")
    if not name and handoff_payload is not None:
        name = _mapping_text(handoff_payload, "opportunity_id")
    if not name:
        name = facility_id

    description = str(raw.get("description", "") or "").strip()
    if not description and handoff_payload is not None:
        description = _mapping_text(handoff_payload, "operator_approved_summary")

    scene_memory_adapter_manifests = _resolve_scene_memory_adapter_manifests(
        raw,
        base_dir=base_dir,
        scene_memory_bundle_path=resolved_paths["scene_memory_bundle_path"],
    )

    return FacilityConfig(
        name=name,
        ply_path=resolved_paths["ply_path"],
        evaluation_prep_path=resolved_paths["evaluation_prep_path"],
        scene_memory_bundle_path=resolved_paths["scene_memory_bundle_path"],
        preview_simulation_path=resolved_paths["preview_simulation_path"],
        opportunity_handoff_path=handoff_path,
        geometry_bundle_path=resolved_paths["geometry_bundle_path"],
        scene_package_path=resolved_paths["scene_package_path"],
        scene_memory_adapter_manifests=scene_memory_adapter_manifests,
        task_hints_path=resolved_paths["task_hints_path"],
        holi_spatial_grounding_path=resolved_paths["holi_spatial_grounding_path"],
        labels_path=resolved_paths["labels_path"],
        structure_path=resolved_paths["structure_path"],
        task_anchor_manifest_path=resolved_paths["task_anchor_manifest_path"],
        object_geometry_manifest_path=resolved_paths["object_geometry_manifest_path"],
        review_queue_path=resolved_paths["review_queue_path"],
        claim_benchmark_path=(
            _resolve_path(claim_benchmark_value, base_dir) if claim_benchmark_value else None
        ),
        description=description,
        landmarks=raw.get("landmarks", []),
        floor_height_m=raw.get("floor_height_m", 0.0),
        ceiling_height_m=raw.get("ceiling_height_m", 5.0),
        manipulation_zones=_parse_manipulation_zones(raw.get("manipulation_zones", [])),
        up_axis=str(raw.get("up_axis", "auto")).strip(),
        scene_rotation_deg=_parse_scene_rotation_deg(raw.get("scene_rotation_deg")),
        video_orientation_fix=_parse_video_orientation_fix(raw.get("video_orientation_fix", "none")),
        intake_mode=intake_mode,
        qualification_state=(
            str(handoff_payload.get("qualification_state")).strip()
            if isinstance(handoff_payload, dict) and handoff_payload.get("qualification_state") is not None
            else None
        ),
        downstream_evaluation_eligibility=(
            bool(handoff_payload.get("downstream_evaluation_eligibility"))
            if isinstance(handoff_payload, dict)
            and handoff_payload.get("downstream_evaluation_eligibility") is not None
            else None
        ),
        opportunity_id=(
            _mapping_text(handoff_payload, "opportunity_id")
            if isinstance(handoff_payload, dict)
            else ""
        ),
        site_submission_id=(
            _mapping_text(handoff_payload, "site_submission_id")
            if isinstance(handoff_payload, dict)
            else ""
        ),
        opportunity_handoff=handoff_payload,
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
                arc_span_deg=raw.get("arc_span_deg", 150.0),
                arc_phase_offset_deg=raw.get("arc_phase_offset_deg", 0.0),
                source_tag=raw.get("source_tag"),
                target_instance_id=raw.get("target_instance_id"),
                target_label=raw.get("target_label"),
                target_category=raw.get("target_category"),
                target_role=raw.get("target_role"),
                target_extents_m=raw.get("target_extents_m"),
                locked_eye_point=raw.get("locked_eye_point"),
                locked_look_at_point=raw.get("locked_look_at_point"),
                locked_probe_motion_radius_m=raw.get("locked_probe_motion_radius_m"),
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


def _parse_render_config(raw: Dict[str, Any], base_dir: Path) -> RenderConfig:
    return RenderConfig(
        backend=str(raw.get("backend", "auto")).strip().lower(),
        resolution=tuple(raw.get("resolution", [480, 640])),
        fps=raw.get("fps", 10),
        num_frames=raw.get("num_frames", 49),
        camera_height_m=raw.get("camera_height_m", 1.2),
        camera_look_down_deg=raw.get("camera_look_down_deg", 15.0),
        camera_paths=_parse_camera_paths(raw.get("camera_paths", []), base_dir),
        num_clips_per_path=raw.get("num_clips_per_path", 3),
        scene_aware=raw.get("scene_aware", True),
        collision_check=raw.get("collision_check", True),
        voxel_size_m=float(raw.get("voxel_size_m", 0.1)),
        density_threshold=int(raw.get("density_threshold", 3)),
        min_clearance_m=float(raw.get("min_clearance_m", 0.15)),
        vlm_fallback=raw.get("vlm_fallback", False),
        vlm_fallback_model=raw.get("vlm_fallback_model", "gemini-3-flash-preview"),
        vlm_fallback_num_views=int(raw.get("vlm_fallback_num_views", 4)),
        task_scoped_scene_aware=bool(raw.get("task_scoped_scene_aware", False)),
        task_scoped_max_specs=int(raw.get("task_scoped_max_specs", 40)),
        task_scoped_context_per_target=int(raw.get("task_scoped_context_per_target", 2)),
        task_scoped_overview_specs=int(raw.get("task_scoped_overview_specs", 6)),
        task_scoped_fallback_specs=int(raw.get("task_scoped_fallback_specs", 16)),
        task_scoped_profile=str(raw.get("task_scoped_profile", "dreamdojo")),
        preserve_num_frames_after_collision_filter=bool(
            raw.get("preserve_num_frames_after_collision_filter", True)
        ),
        task_scoped_num_clips_per_path=int(raw.get("task_scoped_num_clips_per_path", 1)),
        task_scoped_num_frames_override=int(raw.get("task_scoped_num_frames_override", 0)),
        stage1_coverage_gate_enabled=bool(raw.get("stage1_coverage_gate_enabled", True)),
        stage1_coverage_min_visible_frame_ratio=float(
            raw.get("stage1_coverage_min_visible_frame_ratio", 0.35)
        ),
        stage1_coverage_min_approach_angle_bins=int(
            raw.get("stage1_coverage_min_approach_angle_bins", 2)
        ),
        stage1_coverage_angle_bin_deg=float(raw.get("stage1_coverage_angle_bin_deg", 45.0)),
        stage1_coverage_blur_laplacian_min=float(
            raw.get("stage1_coverage_blur_laplacian_min", 20.0)
        ),
        stage1_coverage_blur_sample_every_n_frames=int(
            raw.get("stage1_coverage_blur_sample_every_n_frames", 5)
        ),
        stage1_coverage_blur_max_samples_per_clip=int(
            raw.get("stage1_coverage_blur_max_samples_per_clip", 12)
        ),
        stage1_coverage_min_center_band_ratio=float(
            raw.get("stage1_coverage_min_center_band_ratio", 0.4)
        ),
        stage1_coverage_center_band_x=list(raw.get("stage1_coverage_center_band_x", [0.2, 0.8])),
        stage1_coverage_center_band_y=list(raw.get("stage1_coverage_center_band_y", [0.2, 0.8])),
        stage1_quality_planner_enabled=bool(raw.get("stage1_quality_planner_enabled", True)),
        stage1_quality_candidate_budget=_parse_stage1_quality_candidate_budget(
            raw.get("stage1_quality_candidate_budget", "medium")
        ),
        stage1_quality_autoretry_enabled=bool(raw.get("stage1_quality_autoretry_enabled", True)),
        stage1_quality_max_regen_attempts=int(raw.get("stage1_quality_max_regen_attempts", 2)),
        stage1_quality_min_clip_score=float(raw.get("stage1_quality_min_clip_score", 0.55)),
        stage1_strict_require_task_hints=bool(raw.get("stage1_strict_require_task_hints", False)),
        stage1_active_perception_enabled=bool(raw.get("stage1_active_perception_enabled", True)),
        stage1_active_perception_scope=_parse_stage1_active_perception_scope(
            raw.get("stage1_active_perception_scope", "all")
        ),
        stage1_active_perception_max_loops=int(raw.get("stage1_active_perception_max_loops", 2)),
        stage1_active_perception_fail_closed=bool(
            raw.get("stage1_active_perception_fail_closed", True)
        ),
        stage1_probe_frames_override=int(raw.get("stage1_probe_frames_override", 0)),
        stage1_probe_resolution_scale=float(raw.get("stage1_probe_resolution_scale", 0.0)),
        stage1_probe_min_viable_pose_ratio=float(
            raw.get("stage1_probe_min_viable_pose_ratio", 0.55)
        ),
        stage1_probe_min_unique_positions=int(raw.get("stage1_probe_min_unique_positions", 8)),
        stage1_probe_dedupe_enabled=bool(raw.get("stage1_probe_dedupe_enabled", True)),
        stage1_probe_dedupe_max_regen_attempts=int(
            raw.get("stage1_probe_dedupe_max_regen_attempts", 2)
        ),
        stage1_probe_dedupe_center_dist_m=float(raw.get("stage1_probe_dedupe_center_dist_m", 0.08)),
        stage1_probe_consensus_votes=int(raw.get("stage1_probe_consensus_votes", 3)),
        stage1_probe_consensus_high_variance_delta=float(
            raw.get("stage1_probe_consensus_high_variance_delta", 3.0)
        ),
        stage1_probe_tiebreak_extra_votes=int(raw.get("stage1_probe_tiebreak_extra_votes", 2)),
        stage1_probe_tiebreak_spread_threshold=float(
            raw.get("stage1_probe_tiebreak_spread_threshold", 3.0)
        ),
        stage1_probe_primary_model_only=bool(raw.get("stage1_probe_primary_model_only", True)),
        stage1_vlm_min_task_score=float(raw.get("stage1_vlm_min_task_score", 7.0)),
        stage1_vlm_min_visual_score=float(raw.get("stage1_vlm_min_visual_score", 7.0)),
        stage1_vlm_min_spatial_score=float(raw.get("stage1_vlm_min_spatial_score", 6.0)),
        stage1_keep_probe_videos=bool(raw.get("stage1_keep_probe_videos", False)),
        stage1_repeat_dedupe_enabled=bool(raw.get("stage1_repeat_dedupe_enabled", True)),
        stage1_repeat_dedupe_max_regen_attempts=int(
            raw.get("stage1_repeat_dedupe_max_regen_attempts", 2)
        ),
        stage1_repeat_min_xy_jitter_m=float(raw.get("stage1_repeat_min_xy_jitter_m", 0.06)),
        stage1_repeat_similarity_ssim_threshold=float(
            raw.get("stage1_repeat_similarity_ssim_threshold", 0.995)
        ),
        scene_locked_profile=_parse_scene_locked_profile(
            raw.get("scene_locked_profile", "auto")
        ),
        orientation_autocorrect_enabled=bool(raw.get("orientation_autocorrect_enabled", True)),
        orientation_autocorrect_mode=str(raw.get("orientation_autocorrect_mode", "auto")),
        manipulation_random_xy_offset_m=float(raw.get("manipulation_random_xy_offset_m", 0.0)),
        non_manipulation_random_xy_offset_m=float(
            raw.get("non_manipulation_random_xy_offset_m", 1.0)
        ),
        manipulation_target_z_bias_m=float(raw.get("manipulation_target_z_bias_m", 0.0)),
    )


def _parse_enrich_config(raw: Dict[str, Any], base_dir: Path) -> EnrichConfig:
    return EnrichConfig(
        cosmos_model=raw.get("cosmos_model", "nvidia/Cosmos-Transfer2.5-2B"),
        cosmos_checkpoint=_resolve_path(
            raw.get("cosmos_checkpoint", "./data/checkpoints/cosmos-transfer-2.5-2b/"),
            base_dir,
        ),
        cosmos_repo=_resolve_path(raw.get("cosmos_repo", "/opt/cosmos-transfer"), base_dir),
        disable_guardrails=bool(raw.get("disable_guardrails", True)),
        controlnet_inputs=raw.get("controlnet_inputs", ["rgb", "depth"]),
        num_variants_per_render=raw.get("num_variants_per_render", 5),
        variants=_parse_variants(raw.get("variants", [])),
        guidance=raw.get("guidance", 7.0),
        dynamic_variants=raw.get("dynamic_variants", False),
        dynamic_variants_model=raw.get("dynamic_variants_model", "gemini-3-flash-preview"),
        allow_dynamic_variant_fallback=bool(raw.get("allow_dynamic_variant_fallback", True)),
        context_frame_index=(
            int(raw["context_frame_index"]) if raw.get("context_frame_index") is not None else None
        ),
        context_frame_mode=_parse_enrich_context_frame_mode(
            raw.get("context_frame_mode", "target_centered")
        ),
        max_input_frames=int(raw.get("max_input_frames", 0)),
        max_source_clips=int(raw.get("max_source_clips", 0)),
        min_source_clips=int(raw.get("min_source_clips", 8)),
        min_valid_outputs=int(raw.get("min_valid_outputs", 8)),
        max_blur_reject_rate=float(raw.get("max_blur_reject_rate", 0.30)),
        green_frame_ratio_max=float(raw.get("green_frame_ratio_max", 0.10)),
        enable_visual_collapse_gate=bool(raw.get("enable_visual_collapse_gate", True)),
        vlm_quality_gate_enabled=bool(raw.get("vlm_quality_gate_enabled", False)),
        vlm_quality_fail_closed=bool(raw.get("vlm_quality_fail_closed", True)),
        vlm_quality_autoretry_enabled=bool(raw.get("vlm_quality_autoretry_enabled", True)),
        vlm_quality_max_regen_attempts=int(raw.get("vlm_quality_max_regen_attempts", 2)),
        vlm_quality_min_task_score=float(raw.get("vlm_quality_min_task_score", 7.0)),
        vlm_quality_min_visual_score=float(raw.get("vlm_quality_min_visual_score", 7.0)),
        vlm_quality_min_spatial_score=float(raw.get("vlm_quality_min_spatial_score", 6.0)),
        vlm_quality_require_reasoning_consistency=bool(
            raw.get("vlm_quality_require_reasoning_consistency", True)
        ),
        vlm_quality_retry_context_frame_stride=int(
            raw.get("vlm_quality_retry_context_frame_stride", 6)
        ),
        vlm_quality_disable_depth_on_final_retry=bool(
            raw.get("vlm_quality_disable_depth_on_final_retry", True)
        ),
        source_clip_selection_mode=_parse_source_clip_selection_mode(
            raw.get("source_clip_selection_mode", "all")
        ),
        source_clip_selection_fail_closed=bool(
            raw.get("source_clip_selection_fail_closed", True)
        ),
        source_clip_task=(str(raw.get("source_clip_task")).strip() if raw.get("source_clip_task") else None),
        source_clip_name=(str(raw.get("source_clip_name")).strip() if raw.get("source_clip_name") else None),
        multi_view_context_enabled=bool(raw.get("multi_view_context_enabled", False)),
        multi_view_context_offsets=[int(v) for v in raw.get("multi_view_context_offsets", [-12, 0, 12])],
        scene_index_enabled=bool(raw.get("scene_index_enabled", False)),
        scene_index_k=int(raw.get("scene_index_k", 3)),
        scene_index_sample_every_n_frames=int(raw.get("scene_index_sample_every_n_frames", 8)),
        cosmos_output_quality=int(raw.get("cosmos_output_quality", 5)),
        min_frame0_ssim=float(raw.get("min_frame0_ssim", 0.0)),
        delete_rejected_outputs=bool(raw.get("delete_rejected_outputs", False)),
    )


def _parse_scene_memory_backend_runtime_config(
    raw: Dict[str, Any],
    *,
    base_dir: Path,
    default_inference_script: Optional[str] = None,
) -> SceneMemoryBackendRuntimeConfig:
    repo_path_raw = raw.get("repo_path")
    python_raw = raw.get("python_executable")
    checkpoint_raw = raw.get("checkpoint_path")
    inference_script_raw = raw.get("inference_script", default_inference_script)
    inference_script = None
    if inference_script_raw is not None:
        text = str(inference_script_raw).strip()
        inference_script = text or None
    return SceneMemoryBackendRuntimeConfig(
        enabled=bool(raw.get("enabled", True)),
        allow_runtime_execution=bool(raw.get("allow_runtime_execution", False)),
        repo_path=_resolve_optional_path(repo_path_raw, base_dir),
        python_executable=_resolve_optional_path(python_raw, base_dir),
        inference_script=inference_script,
        checkpoint_path=_resolve_optional_path(checkpoint_raw, base_dir),
    )


def _parse_neoverse_service_config(raw: Dict[str, Any]) -> NeoVerseServiceConfig:
    service_url_raw = raw.get("service_url")
    websocket_base_url_raw = raw.get("websocket_base_url")
    service_url = str(service_url_raw or "").strip() or None
    websocket_base_url = str(websocket_base_url_raw or "").strip() or None
    return NeoVerseServiceConfig(
        enabled=bool(raw.get("enabled", bool(service_url))),
        service_url=service_url,
        api_key_env=str(raw.get("api_key_env", "NEOVERSE_RUNTIME_SERVICE_API_KEY") or "NEOVERSE_RUNTIME_SERVICE_API_KEY").strip(),
        timeout_seconds=max(1, int(raw.get("timeout_seconds", 120) or 120)),
        websocket_base_url=websocket_base_url,
    )


def _parse_scene_memory_runtime_config(
    raw: Dict[str, Any],
    base_dir: Path,
) -> SceneMemoryRuntimeConfig:
    return SceneMemoryRuntimeConfig(
        enabled=bool(raw.get("enabled", True)),
        preferred_backends=[str(v).strip().lower() for v in raw.get(
            "preferred_backends",
            SceneMemoryRuntimeConfig().preferred_backends,
        ) if str(v).strip()],
        watchlist_backends=[str(v).strip().lower() for v in raw.get(
            "watchlist_backends",
            SceneMemoryRuntimeConfig().watchlist_backends,
        ) if str(v).strip()],
        allow_backend_fallback=bool(raw.get("allow_backend_fallback", True)),
        neoverse_service=_parse_neoverse_service_config(
            dict(raw.get("neoverse_service", {}) or {})
        ),
        neoverse=_parse_scene_memory_backend_runtime_config(
            dict(raw.get("neoverse", {}) or {}),
            base_dir=base_dir,
            default_inference_script="inference.py",
        ),
        gen3c=_parse_scene_memory_backend_runtime_config(
            dict(raw.get("gen3c", {}) or {}),
            base_dir=base_dir,
            default_inference_script="inference.py",
        ),
    )


def _env_text(name: str) -> Optional[str]:
    value = str(os.environ.get(name, "") or "").strip()
    return value or None


def _apply_scene_memory_runtime_env_overrides(
    config: SceneMemoryRuntimeConfig,
    *,
    base_dir: Path,
) -> None:
    service_cfg = config.neoverse_service
    service_url = _env_text("NEOVERSE_RUNTIME_SERVICE_URL")
    service_api_key_env = _env_text("NEOVERSE_RUNTIME_SERVICE_API_KEY_ENV")
    timeout_seconds = _env_text("NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS")
    websocket_base_url = _env_text("NEOVERSE_RUNTIME_PUBLIC_WS_BASE_URL")
    if service_url is not None:
        service_cfg.service_url = service_url.rstrip("/")
        service_cfg.enabled = True
    if service_api_key_env is not None:
        service_cfg.api_key_env = service_api_key_env
    if timeout_seconds is not None:
        service_cfg.timeout_seconds = max(1, int(timeout_seconds))
    if websocket_base_url is not None:
        service_cfg.websocket_base_url = websocket_base_url.rstrip("/")

    runtime = config.neoverse
    repo_path = _env_text("NEOVERSE_REPO_PATH")
    python_executable = _env_text("NEOVERSE_PYTHON_EXECUTABLE")
    checkpoint_path = _env_text("NEOVERSE_CHECKPOINT_PATH")

    override_present = any(
        value is not None
        for value in (
            repo_path,
            python_executable,
            checkpoint_path,
        )
    )
    if repo_path is not None:
        runtime.repo_path = _resolve_path(repo_path, base_dir)
    if python_executable is not None:
        runtime.python_executable = _resolve_path(python_executable, base_dir)
    if checkpoint_path is not None:
        runtime.checkpoint_path = _resolve_path(checkpoint_path, base_dir)
    if override_present:
        runtime.allow_runtime_execution = True


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
    validate_config_keys(raw, config_path=config_path)

    config = ValidationConfig(
        schema_version=raw.get("schema_version", "v1"),
        project_name=raw.get("project_name", ""),
    )

    # Evaluation targets (preferred key: qualified_opportunities; legacy alias: facilities)
    for fid, fdata in _merge_target_sections(raw).items():
        config.facilities[fid] = _parse_facility(fid, fdata, base_dir)

    # Render
    if "render" in raw:
        config.render = _parse_render_config(raw["render"], base_dir)

    # Enrich
    if "enrich" in raw:
        config.enrich = _parse_enrich_config(raw["enrich"], base_dir)

    if "scene_memory_runtime" in raw:
        config.scene_memory_runtime = _parse_scene_memory_runtime_config(
            raw["scene_memory_runtime"],
            base_dir,
        )
    _apply_scene_memory_runtime_env_overrides(
        config.scene_memory_runtime,
        base_dir=base_dir,
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
            enabled=gp.get("enabled", False),
            model=gp.get("model", "gemini-3.1-flash-image-preview"),
            api_key_env=gp.get("api_key_env", "GOOGLE_GENAI_API_KEY"),
            prompt=gp.get("prompt", GeminiPolishConfig().prompt),
            sample_every_n_frames=gp.get("sample_every_n_frames", 2),
        )

    # Finetune
    if "finetune" in raw:
        ft = raw["finetune"]
        dataset_quality_raw = ft.get("dataset_quality", {}) or {}
        prompt_lint_raw = dataset_quality_raw.get("prompt_lint", {}) or {}
        temporal_raw = dataset_quality_raw.get("temporal_gates", {}) or {}
        distribution_raw = dataset_quality_raw.get("distribution", {}) or {}
        experiment_config = ft.get("experiment_config")
        if experiment_config and (
            "/" in experiment_config
            or experiment_config.endswith((".sh", ".yaml"))
            or experiment_config.startswith(".")
        ):
            experiment_config = str(_resolve_path(experiment_config, base_dir))
        eval_world_experiment = ft.get("eval_world_experiment")
        if eval_world_experiment:
            eval_world_experiment = str(eval_world_experiment).strip()
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
            eval_world_experiment=eval_world_experiment,
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
            dataset_quality=DatasetQualityConfig(
                strict_manifest_validation=bool(
                    dataset_quality_raw.get("strict_manifest_validation", True)
                ),
                quarantine_rejections=bool(dataset_quality_raw.get("quarantine_rejections", True)),
                fail_on_rejections=bool(dataset_quality_raw.get("fail_on_rejections", True)),
                max_reject_fraction=float(dataset_quality_raw.get("max_reject_fraction", 0.50)),
                enable_duplicate_detection=bool(
                    dataset_quality_raw.get("enable_duplicate_detection", True)
                ),
                enable_leakage_detection=bool(
                    dataset_quality_raw.get("enable_leakage_detection", True)
                ),
                prompt_lint=DatasetPromptLintConfig(
                    enabled=bool(prompt_lint_raw.get("enabled", True)),
                    min_chars=int(prompt_lint_raw.get("min_chars", 8)),
                    min_tokens=int(prompt_lint_raw.get("min_tokens", 2)),
                    min_unique_token_ratio=float(
                        prompt_lint_raw.get("min_unique_token_ratio", 0.35)
                    ),
                    allow_generic_substrings=bool(
                        prompt_lint_raw.get("allow_generic_substrings", False)
                    ),
                ),
                temporal_gates=DatasetTemporalGateConfig(
                    enabled=bool(temporal_raw.get("enabled", True)),
                    min_frames_for_check=int(temporal_raw.get("min_frames_for_check", 8)),
                    max_frames_to_sample=int(temporal_raw.get("max_frames_to_sample", 96)),
                    min_mean_interframe_delta=float(
                        temporal_raw.get("min_mean_interframe_delta", 1.5)
                    ),
                    max_freeze_ratio=float(temporal_raw.get("max_freeze_ratio", 0.70)),
                    max_abrupt_cut_ratio=float(temporal_raw.get("max_abrupt_cut_ratio", 0.35)),
                    max_blockiness_score=float(temporal_raw.get("max_blockiness_score", 0.45)),
                ),
                distribution=DatasetDistributionConfig(
                    enabled=bool(distribution_raw.get("enabled", True)),
                    min_total_clips_for_caps=int(
                        distribution_raw.get("min_total_clips_for_caps", 16)
                    ),
                    min_unique_variants=int(distribution_raw.get("min_unique_variants", 2)),
                    min_unique_source_clips=int(
                        distribution_raw.get("min_unique_source_clips", 4)
                    ),
                    max_single_variant_fraction=float(
                        distribution_raw.get("max_single_variant_fraction", 0.85)
                    ),
                    max_single_source_clip_fraction=float(
                        distribution_raw.get("max_single_source_clip_fraction", 0.60)
                    ),
                    max_prompt_dominance_fraction=float(
                        distribution_raw.get("max_prompt_dominance_fraction", 0.70)
                    ),
                ),
            ),
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
            headline_scope=str(ep.get("headline_scope", "wm_only")),
            rollout_driver=str(ep.get("rollout_driver", "scripted")),
            scripted_rollouts_per_task=int(ep.get("scripted_rollouts_per_task", 12)),
            mode=str(ep.get("mode", "claim")),
            required_action_dim=int(ep.get("required_action_dim", 7)),
            manip_eval_mode=_parse_manip_eval_mode(ep.get("manip_eval_mode", "overlay_marker")),
            min_assignment_quality_score=float(ep.get("min_assignment_quality_score", 0.0)),
            require_object_grounded_manip_tasks=bool(
                ep.get("require_object_grounded_manip_tasks", True)
            ),
            min_absolute_difference=float(ep.get("min_absolute_difference", 1.0)),
            min_manip_success_delta_pp=float(ep.get("min_manip_success_delta_pp", 15.0)),
            require_native_action_compat=bool(ep.get("require_native_action_compat", True)),
            claim_protocol=_parse_claim_protocol(ep.get("claim_protocol", "none")),
            primary_endpoint=_parse_primary_endpoint(
                ep.get("primary_endpoint", "vlm_task_score")
            ),
            freeze_world_snapshot=bool(ep.get("freeze_world_snapshot", False)),
            split_strategy=_parse_claim_split_strategy(ep.get("split_strategy", "legacy")),
            min_practical_success_lift_pp=float(ep.get("min_practical_success_lift_pp", 5.0)),
            claim_replication=ClaimReplicationConfig(
                training_seeds=[
                    int(v)
                    for v in (ep.get("replication", {}) or {}).get(
                        "training_seeds", list(range(8))
                    )
                ]
            ),
            claim_strictness=ClaimStrictnessConfig(
                min_eval_task_specs=int(
                    (ep.get("claim_strictness", {}) or {}).get("min_eval_task_specs", 3)
                ),
                min_eval_start_clips=int(
                    (ep.get("claim_strictness", {}) or {}).get("min_eval_start_clips", 3)
                ),
                min_common_eval_cells=int(
                    (ep.get("claim_strictness", {}) or {}).get("min_common_eval_cells", 30)
                ),
                min_positive_training_seeds=int(
                    (ep.get("claim_strictness", {}) or {}).get("min_positive_training_seeds", 4)
                ),
                p_value_threshold=float(
                    (ep.get("claim_strictness", {}) or {}).get("p_value_threshold", 0.05)
                ),
                require_site_specific_advantage=bool(
                    (ep.get("claim_strictness", {}) or {}).get(
                        "require_site_specific_advantage", True
                    )
                ),
                site_vs_generic_min_lift_pp=float(
                    (ep.get("claim_strictness", {}) or {}).get(
                        "site_vs_generic_min_lift_pp", 0.0
                    )
                ),
            ),
            reliability=PolicyEvalReliabilityConfig(
                max_horizon_steps=int(
                    (ep.get("reliability", {}) or {}).get("max_horizon_steps", 12)
                ),
                keyframe_reanchor_every=int(
                    (ep.get("reliability", {}) or {}).get("keyframe_reanchor_every", 4)
                ),
                min_replay_pass_rate=float(
                    (ep.get("reliability", {}) or {}).get("min_replay_pass_rate", 0.70)
                ),
                min_controllability_pass_rate=float(
                    (ep.get("reliability", {}) or {}).get(
                        "min_controllability_pass_rate", 0.70
                    )
                ),
                enforce_stage_success=bool(
                    (ep.get("reliability", {}) or {}).get("enforce_stage_success", False)
                ),
                max_scoring_failure_rate=float(
                    (ep.get("reliability", {}) or {}).get("max_scoring_failure_rate", 0.02)
                ),
                fail_on_short_rollout=bool(
                    (ep.get("reliability", {}) or {}).get("fail_on_short_rollout", False)
                ),
                min_rollout_frames=int(
                    (ep.get("reliability", {}) or {}).get("min_rollout_frames", 13)
                ),
                min_rollout_steps=int(
                    (ep.get("reliability", {}) or {}).get("min_rollout_steps", 12)
                ),
            ),
            vlm_judge=VLMJudgeConfig(
                model=vlm_raw.get("model", "gemini-3-flash-preview"),
                fallback_models=[
                    str(m)
                    for m in vlm_raw.get("fallback_models", VLMJudgeConfig().fallback_models)
                    if str(m).strip()
                ],
                api_key_env=vlm_raw.get("api_key_env", "GOOGLE_GENAI_API_KEY"),
                enable_agentic_vision=vlm_raw.get("enable_agentic_vision", True),
                video_metadata_fps=float(vlm_raw.get("video_metadata_fps", 10.0)),
                scoring_prompt=vlm_raw.get("scoring_prompt", VLMJudgeConfig().scoring_prompt),
            ),
        )

    if "eval_polaris" in raw:
        epol = raw["eval_polaris"]
        environment_name = epol.get("environment_name")
        config.eval_polaris = PolarisEvalConfig(
            enabled=bool(epol.get("enabled", False)),
            repo_path=_resolve_path(epol.get("repo_path", "/opt/PolaRiS"), base_dir),
            hub_path=_resolve_path(epol.get("hub_path", "./PolaRiS-Hub"), base_dir),
            environment_mode=str(epol.get("environment_mode", "scene_package_bridge")).strip(),
            environment_name=(str(environment_name).strip() if environment_name else None),
            default_as_primary_gate=bool(epol.get("default_as_primary_gate", True)),
            use_for_claim_gate=bool(epol.get("use_for_claim_gate", True)),
            num_rollouts=int(epol.get("num_rollouts", 16)),
            device=str(epol.get("device", "cuda")).strip(),
            policy_client=str(epol.get("policy_client", "OpenVLA")).strip(),
            observation_mode=str(epol.get("observation_mode", "external_only")).strip(),
            action_mode=str(epol.get("action_mode", "native")).strip(),
            export_dir=_resolve_path(
                epol.get("export_dir", "./data/outputs/polaris"),
                base_dir,
            ),
            require_scene_package=bool(epol.get("require_scene_package", True)),
            require_success_correlation_metadata=bool(
                epol.get("require_success_correlation_metadata", True)
            ),
        )

    if "scene_builder" in raw:
        sb = raw["scene_builder"]
        source_ply_path = sb.get("source_ply_path")
        asset_manifest_path = sb.get("asset_manifest_path")
        scene_edit_manifest_path = sb.get("scene_edit_manifest_path")
        task_hints_path = sb.get("task_hints_path")
        config.scene_builder = SceneBuilderConfig(
            enabled=bool(sb.get("enabled", False)),
            source_ply_path=(
                _resolve_path(source_ply_path, base_dir)
                if source_ply_path is not None and str(source_ply_path).strip()
                else None
            ),
            output_scene_root=_resolve_path(
                sb.get("output_scene_root", "./data/scene_package"),
                base_dir,
            ),
            static_collision_mode=str(sb.get("static_collision_mode", "simple")).strip(),
            asset_manifest_path=(
                _resolve_path(asset_manifest_path, base_dir)
                if asset_manifest_path is not None and str(asset_manifest_path).strip()
                else None
            ),
            scene_edit_manifest_path=(
                _resolve_path(scene_edit_manifest_path, base_dir)
                if scene_edit_manifest_path is not None and str(scene_edit_manifest_path).strip()
                else None
            ),
            task_hints_path=(
                _resolve_path(task_hints_path, base_dir)
                if task_hints_path is not None and str(task_hints_path).strip()
                else None
            ),
            robot_type=str(sb.get("robot_type", "franka")).strip(),
            task_template=str(sb.get("task_template", "pick_place_v1")).strip(),
            emit_isaac_lab=bool(sb.get("emit_isaac_lab", True)),
            emit_polaris_metadata=bool(sb.get("emit_polaris_metadata", True)),
            fail_on_physics_qc=bool(sb.get("fail_on_physics_qc", False)),
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
            seed=int(pf.get("seed", 0)),
            extra_args=[str(v) for v in pf.get("extra_args", [])],
        )

    if "policy_adapter" in raw:
        pa = raw["policy_adapter"]
        openvla_raw = pa.get("openvla", {}) if isinstance(pa, dict) else {}
        pi05_raw = pa.get("pi05", {}) if isinstance(pa, dict) else {}
        dreamzero_raw = pa.get("dreamzero", {}) if isinstance(pa, dict) else {}
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
                base_model_name=(
                    str(openvla_raw.get("base_model_name", "")).strip() or None
                ),
                base_checkpoint_path=(
                    _resolve_path(openvla_raw["base_checkpoint_path"], base_dir)
                    if openvla_raw.get("base_checkpoint_path") is not None
                    else None
                ),
                policy_action_dim=int(openvla_raw.get("policy_action_dim", 7)),
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
                strict_action_contract=bool(pi05_raw.get("strict_action_contract", True)),
                allow_synthetic_state_for_eval=bool(
                    pi05_raw.get("allow_synthetic_state_for_eval", False)
                ),
                extra_train_args=[str(v) for v in pi05_raw.get("extra_train_args", [])],
            ),
            dreamzero=DreamZeroAdapterBackendConfig(
                repo_path=_resolve_path(
                    dreamzero_raw.get("repo_path", "/opt/dreamzero"),
                    base_dir,
                ),
                base_model_name=(
                    str(dreamzero_raw.get("base_model_name", "")).strip() or None
                ),
                checkpoint_path=_resolve_path(
                    dreamzero_raw.get("checkpoint_path", "./data/checkpoints/dreamzero/"),
                    base_dir,
                ),
                inference_module=str(
                    dreamzero_raw.get("inference_module", "dreamzero.inference")
                ),
                inference_class=str(
                    dreamzero_raw.get("inference_class", "DreamZeroInference")
                ),
                policy_action_dim=int(dreamzero_raw.get("policy_action_dim", 7)),
                frame_history=int(dreamzero_raw.get("frame_history", 4)),
                strict_action_contract=bool(dreamzero_raw.get("strict_action_contract", True)),
                strict_action_min=float(dreamzero_raw.get("strict_action_min", -1.0)),
                strict_action_max=float(dreamzero_raw.get("strict_action_max", 1.0)),
                train_script=str(dreamzero_raw.get("train_script", "scripts/train.py")),
                extra_train_args=[str(v) for v in dreamzero_raw.get("extra_train_args", [])],
                allow_training=bool(dreamzero_raw.get("allow_training", False)),
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
            dreamzero=config.policy_adapter.dreamzero,
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

    if "external_interaction" in raw:
        ei = raw["external_interaction"]
        manifest_raw = ei.get("manifest_path")
        config.external_interaction = ExternalInteractionConfig(
            enabled=bool(ei.get("enabled", False)),
            manifest_path=(
                _resolve_path(manifest_raw, base_dir)
                if manifest_raw is not None and str(manifest_raw).strip()
                else None
            ),
            source_name=str(ei.get("source_name", "external")),
        )

    if "external_rollouts" in raw:
        er = raw["external_rollouts"]
        manifest_raw = er.get("manifest_path")
        config.external_rollouts = ExternalRolloutsConfig(
            enabled=bool(er.get("enabled", False)),
            manifest_path=(
                _resolve_path(manifest_raw, base_dir)
                if manifest_raw is not None and str(manifest_raw).strip()
                else None
            ),
            source_name=str(er.get("source_name", "teleop")),
            mode=str(er.get("mode", "wm_and_policy")),
        )

    if "native_teacher" in raw:
        nt = raw["native_teacher"]
        config.native_teacher = NativeTeacherConfig(
            enabled=bool(nt.get("enabled", False)),
            include_generic_control=bool(nt.get("include_generic_control", True)),
            generate_corrections=bool(nt.get("generate_corrections", True)),
            planner_horizon_steps=int(nt.get("planner_horizon_steps", 16)),
        )

    if "claim_portfolio" in raw:
        cp = raw["claim_portfolio"]
        config.claim_portfolio = ClaimPortfolioConfig(
            min_facilities=int(cp.get("min_facilities", 3)),
            min_mean_site_vs_frozen_lift_pp=float(
                cp.get("min_mean_site_vs_frozen_lift_pp", 8.0)
            ),
            min_mean_site_vs_generic_lift_pp=float(
                cp.get("min_mean_site_vs_generic_lift_pp", 2.0)
            ),
            max_negative_task_family_delta_pp=float(
                cp.get("max_negative_task_family_delta_pp", -5.0)
            ),
            require_manipulation_nonzero=bool(cp.get("require_manipulation_nonzero", True)),
        )

    if "action_boost" in raw:
        ab = raw["action_boost"]
        config.action_boost = ActionBoostConfig(
            enabled=bool(ab.get("enabled", True)),
            require_full_pipeline=bool(ab.get("require_full_pipeline", True)),
            auto_switch_headline_scope_to_dual=bool(
                ab.get("auto_switch_headline_scope_to_dual", True)
            ),
            auto_enable_rollout_dataset=bool(ab.get("auto_enable_rollout_dataset", True)),
            auto_enable_policy_finetune=bool(ab.get("auto_enable_policy_finetune", True)),
            auto_enable_policy_rl_loop=bool(ab.get("auto_enable_policy_rl_loop", True)),
            compute_profile=_parse_action_boost_compute_profile(
                ab.get("compute_profile", "standard")
            ),
            strict_disjoint_eval=bool(ab.get("strict_disjoint_eval", True)),
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
            policy_refine_near_miss_fraction=float(
                pr.get("policy_refine_near_miss_fraction", 0.30)
            ),
            policy_refine_hard_negative_fraction=float(
                pr.get("policy_refine_hard_negative_fraction", 0.10)
            ),
            world_model_refresh_enabled=pr.get("world_model_refresh_enabled", True),
            world_model_refresh_mix_with_stage2=bool(
                pr.get("world_model_refresh_mix_with_stage2", True)
            ),
            world_model_refresh_require_stage2_vlm_pass=bool(
                pr.get("world_model_refresh_require_stage2_vlm_pass", True)
            ),
            world_model_refresh_stage2_fraction=float(
                pr.get("world_model_refresh_stage2_fraction", 0.60)
            ),
            world_model_refresh_success_fraction=float(
                pr.get("world_model_refresh_success_fraction", 0.25)
            ),
            world_model_refresh_near_miss_fraction=float(
                pr.get("world_model_refresh_near_miss_fraction", 0.15)
            ),
            world_model_refresh_min_total_clips=int(
                pr.get("world_model_refresh_min_total_clips", 128)
            ),
            world_model_refresh_max_total_clips=int(
                pr.get("world_model_refresh_max_total_clips", 512)
            ),
            world_model_refresh_seed=int(pr.get("world_model_refresh_seed", 17)),
            world_model_refresh_epochs=int(pr.get("world_model_refresh_epochs", 3)),
            world_model_refresh_learning_rate=float(
                pr.get("world_model_refresh_learning_rate", 5.0e-5)
            ),
        )

    if "wm_refresh_loop" in raw:
        wr = raw["wm_refresh_loop"]
        config.wm_refresh_loop = WorldModelRefreshLoopConfig(
            enabled=bool(wr.get("enabled", False)),
            iterations=int(wr.get("iterations", 1)),
            source_condition=_parse_wm_refresh_source_condition(
                wr.get("source_condition", "adapted")
            ),
            fail_if_refresh_fails=bool(wr.get("fail_if_refresh_fails", True)),
            fail_on_degenerate_mix=bool(wr.get("fail_on_degenerate_mix", True)),
            min_non_hard_rollouts=int(wr.get("min_non_hard_rollouts", 8)),
            max_hard_negative_fraction=float(wr.get("max_hard_negative_fraction", 0.75)),
            require_valid_video_decode=bool(wr.get("require_valid_video_decode", True)),
            enforce_vlm_quality_floor=bool(wr.get("enforce_vlm_quality_floor", True)),
            min_refresh_task_score=float(wr.get("min_refresh_task_score", 7.0)),
            min_refresh_visual_score=float(wr.get("min_refresh_visual_score", 7.0)),
            min_refresh_spatial_score=float(wr.get("min_refresh_spatial_score", 6.0)),
            fail_on_reasoning_conflict=bool(wr.get("fail_on_reasoning_conflict", True)),
            backfill_from_stage2_vlm_passed=bool(wr.get("backfill_from_stage2_vlm_passed", True)),
            quantile_fallback_enabled=bool(wr.get("quantile_fallback_enabled", True)),
            quantile_success_threshold=float(wr.get("quantile_success_threshold", 0.85)),
            quantile_near_miss_threshold=float(wr.get("quantile_near_miss_threshold", 0.50)),
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
            selection_mode=_parse_rollout_selection_mode(
                rd.get("selection_mode", "success_near_miss")
            ),
            near_miss_min_task_score=float(rd.get("near_miss_min_task_score", 5.0)),
            near_miss_max_task_score=float(rd.get("near_miss_max_task_score", 6.99)),
            near_miss_target_fraction=float(rd.get("near_miss_target_fraction", 0.30)),
            hard_negative_target_fraction=float(rd.get("hard_negative_target_fraction", 0.00)),
            per_task_max_episodes=int(rd.get("per_task_max_episodes", 0)),
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
            control_arms=[
                str(v)
                for v in pc.get(
                    "control_arms",
                    ["frozen_baseline", "site_trained", "generic_control"],
                )
                if str(v).strip()
            ],
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

    _apply_fixed_world_claim_defaults(config)

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
            min_valid_samples=int(es.get("min_valid_samples", 3)),
            fail_on_reasoning_conflict=bool(es.get("fail_on_reasoning_conflict", True)),
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

    # Cross-field validation for action-boost curriculum knobs.
    if not (0.0 < float(config.rollout_dataset.train_split) < 1.0):
        raise ValueError("rollout_dataset.train_split must be in (0, 1)")
    if not (0.0 <= float(config.rollout_dataset.near_miss_target_fraction) <= 1.0):
        raise ValueError("rollout_dataset.near_miss_target_fraction must be in [0, 1]")
    if not (0.0 <= float(config.rollout_dataset.hard_negative_target_fraction) <= 1.0):
        raise ValueError("rollout_dataset.hard_negative_target_fraction must be in [0, 1]")
    if (
        float(config.rollout_dataset.near_miss_target_fraction)
        + float(config.rollout_dataset.hard_negative_target_fraction)
        > 1.0
    ):
        raise ValueError(
            "rollout_dataset.near_miss_target_fraction + "
            "rollout_dataset.hard_negative_target_fraction must be <= 1"
        )
    if float(config.rollout_dataset.near_miss_min_task_score) > float(
        config.rollout_dataset.near_miss_max_task_score
    ):
        raise ValueError(
            "rollout_dataset.near_miss_min_task_score must be <= "
            "rollout_dataset.near_miss_max_task_score"
        )
    if not (0.0 <= float(config.policy_rl_loop.policy_refine_near_miss_fraction) <= 1.0):
        raise ValueError("policy_rl_loop.policy_refine_near_miss_fraction must be in [0, 1]")
    if not (0.0 <= float(config.policy_rl_loop.policy_refine_hard_negative_fraction) <= 1.0):
        raise ValueError("policy_rl_loop.policy_refine_hard_negative_fraction must be in [0, 1]")
    if float(config.policy_rl_loop.world_model_refresh_stage2_fraction) < 0.0:
        raise ValueError("policy_rl_loop.world_model_refresh_stage2_fraction must be >= 0")
    if float(config.policy_rl_loop.world_model_refresh_success_fraction) < 0.0:
        raise ValueError("policy_rl_loop.world_model_refresh_success_fraction must be >= 0")
    if float(config.policy_rl_loop.world_model_refresh_near_miss_fraction) < 0.0:
        raise ValueError("policy_rl_loop.world_model_refresh_near_miss_fraction must be >= 0")
    if (
        float(config.policy_rl_loop.world_model_refresh_stage2_fraction)
        + float(config.policy_rl_loop.world_model_refresh_success_fraction)
        + float(config.policy_rl_loop.world_model_refresh_near_miss_fraction)
        > 1.0
    ):
        raise ValueError(
            "policy_rl_loop world_model_refresh_* fractions must sum to <= 1"
        )
    if int(config.policy_rl_loop.world_model_refresh_min_total_clips) < 1:
        raise ValueError("policy_rl_loop.world_model_refresh_min_total_clips must be >= 1")
    if int(config.policy_rl_loop.world_model_refresh_max_total_clips) < int(
        config.policy_rl_loop.world_model_refresh_min_total_clips
    ):
        raise ValueError(
            "policy_rl_loop.world_model_refresh_max_total_clips must be >= "
            "policy_rl_loop.world_model_refresh_min_total_clips"
        )
    if int(config.wm_refresh_loop.iterations) < 1:
        raise ValueError("wm_refresh_loop.iterations must be >= 1")
    if int(config.wm_refresh_loop.min_non_hard_rollouts) < 0:
        raise ValueError("wm_refresh_loop.min_non_hard_rollouts must be >= 0")
    if not (0.0 <= float(config.wm_refresh_loop.max_hard_negative_fraction) <= 1.0):
        raise ValueError("wm_refresh_loop.max_hard_negative_fraction must be in [0, 1]")
    if not (0.0 <= float(config.wm_refresh_loop.quantile_near_miss_threshold) <= 1.0):
        raise ValueError("wm_refresh_loop.quantile_near_miss_threshold must be in [0, 1]")
    if not (0.0 <= float(config.wm_refresh_loop.quantile_success_threshold) <= 1.0):
        raise ValueError("wm_refresh_loop.quantile_success_threshold must be in [0, 1]")
    if not (0.0 <= float(config.wm_refresh_loop.min_refresh_task_score) <= 10.0):
        raise ValueError("wm_refresh_loop.min_refresh_task_score must be in [0, 10]")
    if not (0.0 <= float(config.wm_refresh_loop.min_refresh_visual_score) <= 10.0):
        raise ValueError("wm_refresh_loop.min_refresh_visual_score must be in [0, 10]")
    if not (0.0 <= float(config.wm_refresh_loop.min_refresh_spatial_score) <= 10.0):
        raise ValueError("wm_refresh_loop.min_refresh_spatial_score must be in [0, 10]")
    if float(config.wm_refresh_loop.quantile_success_threshold) < float(
        config.wm_refresh_loop.quantile_near_miss_threshold
    ):
        raise ValueError(
            "wm_refresh_loop.quantile_success_threshold must be >= "
            "wm_refresh_loop.quantile_near_miss_threshold"
        )
    if not (0.0 <= float(config.eval_policy.reliability.max_scoring_failure_rate) <= 1.0):
        raise ValueError("eval_policy.reliability.max_scoring_failure_rate must be in [0, 1]")
    if int(config.eval_policy.reliability.min_rollout_frames) < 2:
        raise ValueError("eval_policy.reliability.min_rollout_frames must be >= 2")
    if int(config.eval_policy.reliability.min_rollout_steps) < 1:
        raise ValueError("eval_policy.reliability.min_rollout_steps must be >= 1")
    if (
        str(config.eval_policy.claim_protocol).strip().lower() == "fixed_same_facility_uplift"
        and str(config.eval_policy.primary_endpoint).strip().lower() != "task_success"
    ):
        raise ValueError(
            "eval_policy.primary_endpoint must be 'task_success' when "
            "claim_protocol=fixed_same_facility_uplift"
        )
    if (
        str(config.eval_policy.claim_protocol).strip().lower() == "fixed_same_facility_uplift"
        and not bool(config.eval_policy.freeze_world_snapshot)
    ):
        raise ValueError(
            "eval_policy.freeze_world_snapshot must be true when "
            "claim_protocol=fixed_same_facility_uplift"
        )
    training_seeds = list(config.eval_policy.claim_replication.training_seeds or [])
    if len(training_seeds) < 1:
        raise ValueError("eval_policy.replication.training_seeds must contain at least one seed")
    if len({int(v) for v in training_seeds}) != len(training_seeds):
        raise ValueError("eval_policy.replication.training_seeds must be unique")
    if (
        str(config.eval_policy.claim_protocol).strip().lower() == "fixed_same_facility_uplift"
        and str(config.eval_policy.split_strategy).strip().lower() != "disjoint_tasks_and_starts"
    ):
        raise ValueError(
            "eval_policy.split_strategy must be 'disjoint_tasks_and_starts' when "
            "claim_protocol=fixed_same_facility_uplift"
        )
    if str(config.eval_policy.claim_protocol).strip().lower() == "fixed_same_facility_uplift":
        missing_benchmarks = sorted(
            facility_id
            for facility_id, facility in config.facilities.items()
            if getattr(facility, "claim_benchmark_path", None) is None
        )
        if missing_benchmarks:
            raise ValueError(
                "facility.claim_benchmark_path must be set for every facility when "
                "claim_protocol=fixed_same_facility_uplift "
                f"(missing: {', '.join(missing_benchmarks)})"
            )
        min_claim_seeds = max(
            6,
            _min_sign_flip_seed_count(float(config.eval_policy.claim_strictness.p_value_threshold)),
        )
        if len(training_seeds) < min_claim_seeds:
            raise ValueError(
                "eval_policy.replication.training_seeds must contain at least "
                f"{min_claim_seeds} seeds when claim_protocol=fixed_same_facility_uplift "
                "(required for the configured paired sign-flip p-value threshold)."
            )
    if float(config.eval_policy.min_practical_success_lift_pp) < 0.0:
        raise ValueError("eval_policy.min_practical_success_lift_pp must be >= 0")
    strictness = config.eval_policy.claim_strictness
    if int(strictness.min_eval_task_specs) < 1:
        raise ValueError("eval_policy.claim_strictness.min_eval_task_specs must be >= 1")
    if int(strictness.min_eval_start_clips) < 1:
        raise ValueError("eval_policy.claim_strictness.min_eval_start_clips must be >= 1")
    if int(strictness.min_common_eval_cells) < 1:
        raise ValueError("eval_policy.claim_strictness.min_common_eval_cells must be >= 1")
    if int(strictness.min_positive_training_seeds) < 1:
        raise ValueError("eval_policy.claim_strictness.min_positive_training_seeds must be >= 1")
    if float(strictness.p_value_threshold) <= 0.0 or float(strictness.p_value_threshold) >= 1.0:
        raise ValueError("eval_policy.claim_strictness.p_value_threshold must be in (0, 1)")
    if not list(config.policy_compare.control_arms or []):
        raise ValueError("policy_compare.control_arms must contain at least one arm")
    if str(config.eval_policy.claim_protocol).strip().lower() == "fixed_same_facility_uplift":
        if int(strictness.min_positive_training_seeds) > len(training_seeds):
            raise ValueError(
                "eval_policy.claim_strictness.min_positive_training_seeds cannot exceed the "
                "configured number of training seeds"
            )
        configured_arms = {
            str(v).strip()
            for v in list(config.policy_compare.control_arms or [])
            if str(v).strip()
        }
        required_arms = {"frozen_baseline", "site_trained", "generic_control"}
        missing_arms = sorted(required_arms - configured_arms)
        if missing_arms:
            raise ValueError(
                "policy_compare.control_arms must include frozen_baseline, site_trained, and "
                "generic_control when claim_protocol=fixed_same_facility_uplift "
                f"(missing: {', '.join(missing_arms)})"
            )
    if int(config.policy_adapter.dreamzero.policy_action_dim) < 1:
        raise ValueError("policy_adapter.dreamzero.policy_action_dim must be >= 1")
    if int(config.policy_adapter.dreamzero.frame_history) < 1:
        raise ValueError("policy_adapter.dreamzero.frame_history must be >= 1")
    if float(config.policy_adapter.dreamzero.strict_action_min) >= float(
        config.policy_adapter.dreamzero.strict_action_max
    ):
        raise ValueError(
            "policy_adapter.dreamzero.strict_action_min must be < "
            "policy_adapter.dreamzero.strict_action_max"
        )
    if not str(config.policy_adapter.dreamzero.inference_module).strip():
        raise ValueError("policy_adapter.dreamzero.inference_module must be non-empty")
    if not str(config.policy_adapter.dreamzero.inference_class).strip():
        raise ValueError("policy_adapter.dreamzero.inference_class must be non-empty")
    if not str(config.policy_adapter.dreamzero.train_script).strip():
        raise ValueError("policy_adapter.dreamzero.train_script must be non-empty")
    if not str(config.external_interaction.source_name).strip():
        raise ValueError("external_interaction.source_name must be non-empty")
    if not str(config.external_rollouts.source_name).strip():
        raise ValueError("external_rollouts.source_name must be non-empty")
    external_rollout_mode = str(config.external_rollouts.mode or "wm_and_policy").strip().lower()
    if external_rollout_mode not in {"policy_only", "wm_only", "wm_and_policy"}:
        raise ValueError(
            "external_rollouts.mode must be one of: policy_only, wm_only, wm_and_policy"
        )
    config.external_rollouts.mode = external_rollout_mode
    if int(config.native_teacher.planner_horizon_steps) < 1:
        raise ValueError("native_teacher.planner_horizon_steps must be >= 1")
    if int(config.claim_portfolio.min_facilities) < 1:
        raise ValueError("claim_portfolio.min_facilities must be >= 1")
    if float(config.claim_portfolio.min_mean_site_vs_frozen_lift_pp) < 0.0:
        raise ValueError("claim_portfolio.min_mean_site_vs_frozen_lift_pp must be >= 0")
    if float(config.claim_portfolio.min_mean_site_vs_generic_lift_pp) < 0.0:
        raise ValueError("claim_portfolio.min_mean_site_vs_generic_lift_pp must be >= 0")
    if int(config.enrich.min_source_clips) < 0:
        raise ValueError("enrich.min_source_clips must be >= 0")
    if int(config.enrich.min_valid_outputs) < 0:
        raise ValueError("enrich.min_valid_outputs must be >= 0")
    if int(config.enrich.max_input_frames) < 0:
        raise ValueError("enrich.max_input_frames must be >= 0")
    if (
        str(config.enrich.context_frame_mode).strip().lower() == "fixed"
        and config.enrich.context_frame_index is None
    ):
        raise ValueError(
            "enrich.context_frame_index must be set when enrich.context_frame_mode='fixed'"
        )
    if len(list(config.enrich.multi_view_context_offsets or [])) < 1:
        raise ValueError("enrich.multi_view_context_offsets must contain at least one offset")
    if bool(config.enrich.multi_view_context_enabled) and 0 not in {
        int(v) for v in list(config.enrich.multi_view_context_offsets or [])
    }:
        raise ValueError(
            "enrich.multi_view_context_offsets must include 0 when multi_view_context_enabled=true"
        )
    if not (0.0 <= float(config.enrich.max_blur_reject_rate) <= 1.0):
        raise ValueError("enrich.max_blur_reject_rate must be in [0, 1]")
    if not (0.0 <= float(config.enrich.green_frame_ratio_max) <= 1.0):
        raise ValueError("enrich.green_frame_ratio_max must be in [0, 1]")
    if not (0.0 <= float(config.enrich.min_frame0_ssim) <= 1.0):
        raise ValueError("enrich.min_frame0_ssim must be in [0, 1]")
    if int(config.enrich.vlm_quality_max_regen_attempts) < 0:
        raise ValueError("enrich.vlm_quality_max_regen_attempts must be >= 0")
    if int(config.enrich.vlm_quality_retry_context_frame_stride) < 1:
        raise ValueError("enrich.vlm_quality_retry_context_frame_stride must be >= 1")
    if not (0.0 <= float(config.enrich.vlm_quality_min_task_score) <= 10.0):
        raise ValueError("enrich.vlm_quality_min_task_score must be in [0, 10]")
    if not (0.0 <= float(config.enrich.vlm_quality_min_visual_score) <= 10.0):
        raise ValueError("enrich.vlm_quality_min_visual_score must be in [0, 10]")
    if not (0.0 <= float(config.enrich.vlm_quality_min_spatial_score) <= 10.0):
        raise ValueError("enrich.vlm_quality_min_spatial_score must be in [0, 10]")
    if int(config.enrich.scene_index_k) < 0:
        raise ValueError("enrich.scene_index_k must be >= 0")
    if int(config.enrich.scene_index_sample_every_n_frames) < 1:
        raise ValueError("enrich.scene_index_sample_every_n_frames must be >= 1")
    if not (0.0 <= float(config.finetune.dataset_quality.max_reject_fraction) <= 1.0):
        raise ValueError("finetune.dataset_quality.max_reject_fraction must be in [0, 1]")
    if int(config.finetune.dataset_quality.prompt_lint.min_chars) < 0:
        raise ValueError("finetune.dataset_quality.prompt_lint.min_chars must be >= 0")
    if int(config.finetune.dataset_quality.prompt_lint.min_tokens) < 0:
        raise ValueError("finetune.dataset_quality.prompt_lint.min_tokens must be >= 0")
    if not (
        0.0 <= float(config.finetune.dataset_quality.prompt_lint.min_unique_token_ratio) <= 1.0
    ):
        raise ValueError(
            "finetune.dataset_quality.prompt_lint.min_unique_token_ratio must be in [0, 1]"
        )
    if int(config.finetune.dataset_quality.temporal_gates.min_frames_for_check) < 1:
        raise ValueError("finetune.dataset_quality.temporal_gates.min_frames_for_check must be >= 1")
    if int(config.finetune.dataset_quality.temporal_gates.max_frames_to_sample) < 1:
        raise ValueError("finetune.dataset_quality.temporal_gates.max_frames_to_sample must be >= 1")
    if float(config.finetune.dataset_quality.temporal_gates.min_mean_interframe_delta) < 0.0:
        raise ValueError(
            "finetune.dataset_quality.temporal_gates.min_mean_interframe_delta must be >= 0"
        )
    if not (0.0 <= float(config.finetune.dataset_quality.temporal_gates.max_freeze_ratio) <= 1.0):
        raise ValueError("finetune.dataset_quality.temporal_gates.max_freeze_ratio must be in [0, 1]")
    if not (
        0.0 <= float(config.finetune.dataset_quality.temporal_gates.max_abrupt_cut_ratio) <= 1.0
    ):
        raise ValueError(
            "finetune.dataset_quality.temporal_gates.max_abrupt_cut_ratio must be in [0, 1]"
        )
    if float(config.finetune.dataset_quality.temporal_gates.max_blockiness_score) < 0.0:
        raise ValueError("finetune.dataset_quality.temporal_gates.max_blockiness_score must be >= 0")
    if int(config.finetune.dataset_quality.distribution.min_total_clips_for_caps) < 1:
        raise ValueError("finetune.dataset_quality.distribution.min_total_clips_for_caps must be >= 1")
    if int(config.finetune.dataset_quality.distribution.min_unique_variants) < 1:
        raise ValueError("finetune.dataset_quality.distribution.min_unique_variants must be >= 1")
    if int(config.finetune.dataset_quality.distribution.min_unique_source_clips) < 1:
        raise ValueError(
            "finetune.dataset_quality.distribution.min_unique_source_clips must be >= 1"
        )
    if not (
        0.0 <= float(config.finetune.dataset_quality.distribution.max_single_variant_fraction) <= 1.0
    ):
        raise ValueError(
            "finetune.dataset_quality.distribution.max_single_variant_fraction must be in [0, 1]"
        )
    if not (
        0.0
        <= float(config.finetune.dataset_quality.distribution.max_single_source_clip_fraction)
        <= 1.0
    ):
        raise ValueError(
            "finetune.dataset_quality.distribution.max_single_source_clip_fraction must be in [0, 1]"
        )
    if not (
        0.0 <= float(config.finetune.dataset_quality.distribution.max_prompt_dominance_fraction) <= 1.0
    ):
        raise ValueError(
            "finetune.dataset_quality.distribution.max_prompt_dominance_fraction must be in [0, 1]"
        )
    if int(config.render.stage1_quality_max_regen_attempts) < 0:
        raise ValueError("render.stage1_quality_max_regen_attempts must be >= 0")
    if not (0.0 <= float(config.render.stage1_quality_min_clip_score) <= 1.0):
        raise ValueError("render.stage1_quality_min_clip_score must be in [0, 1]")
    if str(config.render.stage1_quality_candidate_budget).strip().lower() not in (
        _ALLOWED_STAGE1_QUALITY_CANDIDATE_BUDGETS
    ):
        allowed = ", ".join(sorted(_ALLOWED_STAGE1_QUALITY_CANDIDATE_BUDGETS))
        raise ValueError(f"render.stage1_quality_candidate_budget must be one of: {allowed}")
    if str(config.render.stage1_active_perception_scope).strip().lower() not in (
        _ALLOWED_STAGE1_ACTIVE_PERCEPTION_SCOPES
    ):
        allowed = ", ".join(sorted(_ALLOWED_STAGE1_ACTIVE_PERCEPTION_SCOPES))
        raise ValueError(f"render.stage1_active_perception_scope must be one of: {allowed}")
    if str(config.render.scene_locked_profile).strip().lower() not in _ALLOWED_SCENE_LOCKED_PROFILES:
        allowed = ", ".join(sorted(_ALLOWED_SCENE_LOCKED_PROFILES))
        raise ValueError(f"render.scene_locked_profile must be one of: {allowed}")
    if int(config.render.stage1_active_perception_max_loops) < 0:
        raise ValueError("render.stage1_active_perception_max_loops must be >= 0")
    if int(config.render.stage1_probe_frames_override) < 0:
        raise ValueError("render.stage1_probe_frames_override must be >= 0")
    if not (0.0 <= float(config.render.stage1_probe_resolution_scale) <= 1.0):
        raise ValueError("render.stage1_probe_resolution_scale must be in [0, 1]")
    if not (0.0 <= float(config.render.stage1_probe_min_viable_pose_ratio) <= 1.0):
        raise ValueError("render.stage1_probe_min_viable_pose_ratio must be in [0, 1]")
    if int(config.render.stage1_probe_min_unique_positions) < 1:
        raise ValueError("render.stage1_probe_min_unique_positions must be >= 1")
    if int(config.render.stage1_probe_dedupe_max_regen_attempts) < 0:
        raise ValueError("render.stage1_probe_dedupe_max_regen_attempts must be >= 0")
    if float(config.render.stage1_probe_dedupe_center_dist_m) < 0.0:
        raise ValueError("render.stage1_probe_dedupe_center_dist_m must be >= 0")
    if int(config.render.stage1_probe_consensus_votes) < 1:
        raise ValueError("render.stage1_probe_consensus_votes must be >= 1")
    if not (0.0 <= float(config.render.stage1_probe_consensus_high_variance_delta) <= 30.0):
        raise ValueError("render.stage1_probe_consensus_high_variance_delta must be in [0, 30]")
    if int(config.render.stage1_probe_tiebreak_extra_votes) < 0:
        raise ValueError("render.stage1_probe_tiebreak_extra_votes must be >= 0")
    if not (0.0 <= float(config.render.stage1_probe_tiebreak_spread_threshold) <= 30.0):
        raise ValueError("render.stage1_probe_tiebreak_spread_threshold must be in [0, 30]")
    if not (0.0 <= float(config.render.stage1_vlm_min_task_score) <= 10.0):
        raise ValueError("render.stage1_vlm_min_task_score must be in [0, 10]")
    if not (0.0 <= float(config.render.stage1_vlm_min_visual_score) <= 10.0):
        raise ValueError("render.stage1_vlm_min_visual_score must be in [0, 10]")
    if not (0.0 <= float(config.render.stage1_vlm_min_spatial_score) <= 10.0):
        raise ValueError("render.stage1_vlm_min_spatial_score must be in [0, 10]")
    if int(config.render.stage1_repeat_dedupe_max_regen_attempts) < 0:
        raise ValueError("render.stage1_repeat_dedupe_max_regen_attempts must be >= 0")
    if float(config.render.stage1_repeat_min_xy_jitter_m) < 0.0:
        raise ValueError("render.stage1_repeat_min_xy_jitter_m must be >= 0")
    if not (0.0 <= float(config.render.stage1_repeat_similarity_ssim_threshold) <= 1.0):
        raise ValueError("render.stage1_repeat_similarity_ssim_threshold must be in [0, 1]")
    allowed_render_backends = {"auto", "gsplat", "isaac_scene"}
    if str(config.render.backend) not in allowed_render_backends:
        allowed = ", ".join(sorted(allowed_render_backends))
        raise ValueError(f"render.backend must be one of: {allowed}")
    if (
        str(config.render.backend) == "isaac_scene"
        and not bool(config.scene_builder.enabled)
        and not any(facility.scene_package_path is not None for facility in config.facilities.values())
    ):
        raise ValueError(
            "render.backend=isaac_scene requires facility.scene_package_path or scene_builder.enabled=true"
        )
    if not (0.0 <= float(config.eval_policy.vlm_judge.video_metadata_fps) <= 24.0):
        raise ValueError("eval_policy.vlm_judge.video_metadata_fps must be in [0, 24]")
    allowed_scene_memory_runtime_backends = {
        "neoverse",
        "gen3c",
        "cosmos_transfer",
        "3dsceneprompt",
    }
    if not config.scene_memory_runtime.preferred_backends:
        raise ValueError("scene_memory_runtime.preferred_backends must not be empty")
    invalid_preferred_runtime_backends = [
        backend
        for backend in config.scene_memory_runtime.preferred_backends
        if backend not in allowed_scene_memory_runtime_backends
    ]
    if invalid_preferred_runtime_backends:
        allowed = ", ".join(sorted(allowed_scene_memory_runtime_backends))
        raise ValueError(
            "scene_memory_runtime.preferred_backends contains unsupported backends: "
            f"{', '.join(invalid_preferred_runtime_backends)}. Allowed: {allowed}"
        )
    invalid_watchlist_runtime_backends = [
        backend
        for backend in config.scene_memory_runtime.watchlist_backends
        if backend not in allowed_scene_memory_runtime_backends
    ]
    if invalid_watchlist_runtime_backends:
        allowed = ", ".join(sorted(allowed_scene_memory_runtime_backends))
        raise ValueError(
            "scene_memory_runtime.watchlist_backends contains unsupported backends: "
            f"{', '.join(invalid_watchlist_runtime_backends)}. Allowed: {allowed}"
        )
    if bool(config.eval_policy.reliability.fail_on_short_rollout):
        eval_mode = (config.eval_policy.mode or "claim").strip().lower()
        effective_max_steps = int(config.eval_policy.max_steps_per_rollout)
        if eval_mode == "claim":
            effective_max_steps = min(
                effective_max_steps,
                int(config.eval_policy.reliability.max_horizon_steps),
            )
        required_frames = int(config.eval_policy.reliability.min_rollout_frames)
        if effective_max_steps + 1 < required_frames:
            raise ValueError(
                "eval_policy.max_steps_per_rollout (after claim-mode horizon clamp) is too low "
                f"for reliability.min_rollout_frames={required_frames}"
            )
        required_steps = int(config.eval_policy.reliability.min_rollout_steps)
        if effective_max_steps < required_steps:
            raise ValueError(
                "eval_policy.max_steps_per_rollout (after claim-mode horizon clamp) is too low "
                f"for reliability.min_rollout_steps={required_steps}"
            )
    if int(config.eval_spatial.min_valid_samples) < 1:
        raise ValueError("eval_spatial.min_valid_samples must be >= 1")
    allowed_scene_builder_collision_modes = {"mesh", "convex_decomp", "simple"}
    if config.scene_builder.static_collision_mode not in allowed_scene_builder_collision_modes:
        allowed = ", ".join(sorted(allowed_scene_builder_collision_modes))
        raise ValueError(f"scene_builder.static_collision_mode must be one of: {allowed}")
    allowed_scene_builder_robot_types = {"franka"}
    if config.scene_builder.robot_type not in allowed_scene_builder_robot_types:
        allowed = ", ".join(sorted(allowed_scene_builder_robot_types))
        raise ValueError(f"scene_builder.robot_type must be one of: {allowed}")
    allowed_scene_builder_task_templates = {"pick_place_v1"}
    if config.scene_builder.task_template not in allowed_scene_builder_task_templates:
        allowed = ", ".join(sorted(allowed_scene_builder_task_templates))
        raise ValueError(f"scene_builder.task_template must be one of: {allowed}")
    if bool(config.scene_builder.enabled):
        if config.scene_builder.source_ply_path is None:
            raise ValueError("scene_builder.source_ply_path must be set when scene_builder.enabled=true")
        if config.scene_builder.asset_manifest_path is None:
            raise ValueError(
                "scene_builder.asset_manifest_path must be set when scene_builder.enabled=true"
            )
    allowed_polaris_modes = {"auto", "scene_package_bridge", "native_bundle", "scan_only_bridge"}
    if config.eval_polaris.environment_mode not in allowed_polaris_modes:
        allowed = ", ".join(sorted(allowed_polaris_modes))
        raise ValueError(f"eval_polaris.environment_mode must be one of: {allowed}")
    allowed_polaris_observation_modes = {"external_only", "external_wrist_stitched"}
    if config.eval_polaris.observation_mode not in allowed_polaris_observation_modes:
        allowed = ", ".join(sorted(allowed_polaris_observation_modes))
        raise ValueError(f"eval_polaris.observation_mode must be one of: {allowed}")
    allowed_polaris_action_modes = {"auto", "native", "joint_position_bridge"}
    if config.eval_polaris.action_mode not in allowed_polaris_action_modes:
        allowed = ", ".join(sorted(allowed_polaris_action_modes))
        raise ValueError(f"eval_polaris.action_mode must be one of: {allowed}")
    if int(config.eval_polaris.num_rollouts) < 1:
        raise ValueError("eval_polaris.num_rollouts must be >= 1")
    if not str(config.eval_polaris.policy_client).strip():
        raise ValueError("eval_polaris.policy_client must be non-empty")

    return config


def _apply_fixed_world_claim_defaults(config: ValidationConfig) -> None:
    """Auto-enable claim-path dependencies for the fixed-world fallback gate."""
    if str(config.eval_policy.claim_protocol).strip().lower() != "fixed_same_facility_uplift":
        return

    if (str(config.eval_policy.headline_scope or "wm_only").strip().lower()) == "wm_only":
        config.eval_policy.headline_scope = "wm_uplift"
    config.policy_compare.enabled = True
    config.rollout_dataset.enabled = True
    config.policy_finetune.enabled = True
