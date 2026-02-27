"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FacilityConfig:
    name: str
    ply_path: Path
    description: str = ""
    landmarks: List[str] = field(default_factory=list)
    floor_height_m: float = 0.0
    ceiling_height_m: float = 5.0


@dataclass
class CameraPathSpec:
    type: str  # "orbit", "sweep", "file"
    # orbit params
    radius_m: float = 3.0
    num_orbits: int = 2
    # sweep params
    length_m: float = 10.0
    # file params
    path: Optional[str] = None


@dataclass
class RenderConfig:
    resolution: tuple[int, int] = (480, 640)  # H x W
    fps: int = 10
    num_frames: int = 49
    camera_height_m: float = 1.2
    camera_look_down_deg: float = 15.0
    camera_paths: List[CameraPathSpec] = field(default_factory=list)
    num_clips_per_path: int = 3


@dataclass
class VariantSpec:
    name: str
    prompt: str


@dataclass
class EnrichConfig:
    cosmos_model: str = "nvidia/Cosmos-Transfer2.5-2B"
    cosmos_checkpoint: Path = Path("./data/checkpoints/cosmos-transfer-2.5-2b/")
    controlnet_inputs: List[str] = field(default_factory=lambda: ["rgb", "depth"])
    num_variants_per_render: int = 5
    variants: List[VariantSpec] = field(default_factory=list)
    guidance: float = 7.0


@dataclass
class FinetuneConfig:
    dreamdojo_repo: Path = Path("/opt/DreamDojo")
    dreamdojo_checkpoint: Path = Path("./data/checkpoints/DreamDojo/2B_pretrain/")
    model_size: str = "2B"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_every_n_epochs: int = 10
    max_training_hours: float = 72.0


@dataclass
class VLMJudgeConfig:
    model: str = "gemini-3-flash"
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
    openvla_model: str = "openvla/openvla-7b"
    openvla_checkpoint: Path = Path("./data/checkpoints/openvla-7b/")
    num_rollouts: int = 50
    max_steps_per_rollout: int = 100
    tasks: List[str] = field(default_factory=list)
    vlm_judge: VLMJudgeConfig = field(default_factory=VLMJudgeConfig)


@dataclass
class VisualFidelityConfig:
    metrics: List[str] = field(default_factory=lambda: ["psnr", "ssim", "lpips"])
    lpips_backbone: str = "alex"


@dataclass
class SpatialAccuracyConfig:
    num_sample_frames: int = 20
    vlm_model: str = "gemini-3-flash"


@dataclass
class CrossSiteConfig:
    num_clips_per_model: int = 30
    vlm_model: str = "gemini-3-flash"


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
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval_policy: PolicyEvalConfig = field(default_factory=PolicyEvalConfig)
    eval_visual: VisualFidelityConfig = field(default_factory=VisualFidelityConfig)
    eval_spatial: SpatialAccuracyConfig = field(default_factory=SpatialAccuracyConfig)
    eval_crosssite: CrossSiteConfig = field(default_factory=CrossSiteConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)


def _parse_facility(raw: Dict[str, Any]) -> FacilityConfig:
    return FacilityConfig(
        name=raw["name"],
        ply_path=Path(raw["ply_path"]),
        description=raw.get("description", ""),
        landmarks=raw.get("landmarks", []),
        floor_height_m=raw.get("floor_height_m", 0.0),
        ceiling_height_m=raw.get("ceiling_height_m", 5.0),
    )


def _parse_camera_paths(raw_list: List[Dict[str, Any]]) -> List[CameraPathSpec]:
    paths = []
    for raw in raw_list:
        paths.append(
            CameraPathSpec(
                type=raw["type"],
                radius_m=raw.get("radius_m", 3.0),
                num_orbits=raw.get("num_orbits", 2),
                length_m=raw.get("length_m", 10.0),
                path=raw.get("path"),
            )
        )
    return paths


def _parse_variants(raw_list: List[Dict[str, str]]) -> List[VariantSpec]:
    return [VariantSpec(name=v["name"], prompt=v["prompt"]) for v in raw_list]


def load_config(path: Path) -> ValidationConfig:
    """Load and parse a YAML config file into a ValidationConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = ValidationConfig(
        schema_version=raw.get("schema_version", "v1"),
        project_name=raw.get("project_name", ""),
    )

    # Facilities
    for fid, fdata in raw.get("facilities", {}).items():
        config.facilities[fid] = _parse_facility(fdata)

    # Render
    if "render" in raw:
        r = raw["render"]
        config.render = RenderConfig(
            resolution=tuple(r.get("resolution", [480, 640])),
            fps=r.get("fps", 10),
            num_frames=r.get("num_frames", 49),
            camera_height_m=r.get("camera_height_m", 1.2),
            camera_look_down_deg=r.get("camera_look_down_deg", 15.0),
            camera_paths=_parse_camera_paths(r.get("camera_paths", [])),
            num_clips_per_path=r.get("num_clips_per_path", 3),
        )

    # Enrich
    if "enrich" in raw:
        e = raw["enrich"]
        config.enrich = EnrichConfig(
            cosmos_model=e.get("cosmos_model", "nvidia/Cosmos-Transfer2.5-2B"),
            cosmos_checkpoint=Path(
                e.get("cosmos_checkpoint", "./data/checkpoints/cosmos-transfer-2.5-2b/")
            ),
            controlnet_inputs=e.get("controlnet_inputs", ["rgb", "depth"]),
            num_variants_per_render=e.get("num_variants_per_render", 5),
            variants=_parse_variants(e.get("variants", [])),
            guidance=e.get("guidance", 7.0),
        )

    # Finetune
    if "finetune" in raw:
        ft = raw["finetune"]
        config.finetune = FinetuneConfig(
            dreamdojo_repo=Path(ft.get("dreamdojo_repo", "/opt/DreamDojo")),
            dreamdojo_checkpoint=Path(
                ft.get("dreamdojo_checkpoint", "./data/checkpoints/DreamDojo/2B_pretrain/")
            ),
            model_size=ft.get("model_size", "2B"),
            lora_rank=ft.get("lora_rank", 16),
            lora_alpha=ft.get("lora_alpha", 32),
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
        config.eval_policy = PolicyEvalConfig(
            openvla_model=ep.get("openvla_model", "openvla/openvla-7b"),
            openvla_checkpoint=Path(
                ep.get("openvla_checkpoint", "./data/checkpoints/openvla-7b/")
            ),
            num_rollouts=ep.get("num_rollouts", 50),
            max_steps_per_rollout=ep.get("max_steps_per_rollout", 100),
            tasks=ep.get("tasks", []),
            vlm_judge=VLMJudgeConfig(
                model=vlm_raw.get("model", "gemini-3-flash"),
                api_key_env=vlm_raw.get("api_key_env", "GOOGLE_GENAI_API_KEY"),
                enable_agentic_vision=vlm_raw.get("enable_agentic_vision", True),
                scoring_prompt=vlm_raw.get("scoring_prompt", VLMJudgeConfig().scoring_prompt),
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
            vlm_model=es.get("vlm_model", "gemini-3-flash"),
        )

    # Cross-site
    if "eval_crosssite" in raw:
        ec = raw["eval_crosssite"]
        config.eval_crosssite = CrossSiteConfig(
            num_clips_per_model=ec.get("num_clips_per_model", 30),
            vlm_model=ec.get("vlm_model", "gemini-3-flash"),
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
