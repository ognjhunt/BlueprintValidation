"""OpenVLA-OFT inference loop for policy evaluation inside DreamDojo world model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re
import sys
from importlib import import_module

import numpy as np

from ..common import get_logger

logger = get_logger("evaluation.openvla_runner")

_ACTION_CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
_EXPERIMENT_BY_ACTION_EMBED_WIDTH = {
    1536: "cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame",
    28: "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
}
_ACTION_EMBED_WIDTH_BY_EXPERIMENT = {
    experiment: width for width, experiment in _EXPERIMENT_BY_ACTION_EMBED_WIDTH.items()
}
_EXPERIMENT_BY_ACTION_DIM = {
    384: "cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame",
    7: "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
}
_ACTION_DIM_BY_EXPERIMENT = {experiment: dim for dim, experiment in _EXPERIMENT_BY_ACTION_DIM.items()}


def _normalize_action_chunk(
    action,
    *,
    expected_action_dim: Optional[int],
    actions_per_latent_frame: Optional[int],
) -> np.ndarray:
    """Normalize action input into a [T, D] chunk compatible with DreamDojo."""
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.ndim == 1:
        action_array = action_array.reshape(1, -1)
    elif action_array.ndim > 2:
        action_array = action_array.reshape(-1, action_array.shape[-1])

    per_step_dim = int(action_array.shape[-1])
    if expected_action_dim is not None and per_step_dim != int(expected_action_dim):
        raise RuntimeError(
            "Action-space mismatch between policy and world model: "
            f"policy action_dim={per_step_dim}, world_model action_dim={expected_action_dim}. "
            "Use a policy and DreamDojo checkpoint with matching action dimensions."
        )

    ratio = max(int(actions_per_latent_frame or 1), 1)
    num_steps = int(action_array.shape[0])
    if num_steps < ratio:
        pad = np.repeat(action_array[-1:, :], ratio - num_steps, axis=0)
        action_array = np.concatenate([action_array, pad], axis=0)
    elif num_steps % ratio != 0:
        target_steps = ((num_steps + ratio - 1) // ratio) * ratio
        pad = np.repeat(action_array[-1:, :], target_steps - num_steps, axis=0)
        action_array = np.concatenate([action_array, pad], axis=0)

    return action_array


@dataclass
class RolloutResult:
    task_prompt: str
    condition: str  # "baseline" or "adapted"
    video_path: Optional[Path]
    action_sequence: List[List[float]]
    num_steps: int
    success: bool


def load_openvla(model_name: str, checkpoint_path: Optional[Path] = None, device: str = "cuda"):
    """Load OpenVLA-OFT model for inference."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import torch

    model_id = str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else model_name

    logger.info("Loading OpenVLA-OFT from %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    if not hasattr(model, "predict_action"):
        raise RuntimeError(
            f"Loaded OpenVLA-OFT model from {model_id} does not expose predict_action(). "
            "Ensure you are using an OpenVLA-OFT-compatible checkpoint."
        )

    return model, processor


def _resolve_world_model_checkpoint_path(checkpoint_path: Path) -> Path:
    """Resolve a usable DreamDojo checkpoint directory from common root paths."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        return checkpoint_path

    candidates: list[Path] = []

    def _maybe_add_latest(root: Path) -> None:
        latest_txt = root / "latest_checkpoint.txt"
        if not latest_txt.exists():
            return
        try:
            token = latest_txt.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not token:
            return
        candidate = root / token
        if candidate.exists():
            candidates.append(candidate)

    def _maybe_add_iters(root: Path) -> None:
        if not root.exists():
            return
        iter_dirs = sorted(
            [p for p in root.glob("iter_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(iter_dirs)

    _maybe_add_latest(checkpoint_path)
    _maybe_add_latest(checkpoint_path / "2B_pretrain")
    _maybe_add_iters(checkpoint_path)
    _maybe_add_iters(checkpoint_path / "2B_pretrain")
    _maybe_add_iters(checkpoint_path / "checkpoints")

    return candidates[0] if candidates else checkpoint_path


def _resolve_checkpoint_model_dir(checkpoint_path: Path) -> Optional[Path]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        return None
    if (checkpoint_path / ".metadata").exists():
        return checkpoint_path
    if (checkpoint_path / "model" / ".metadata").exists():
        return checkpoint_path / "model"
    return None


def _read_action_embed_width(checkpoint_path: Path) -> Optional[int]:
    model_dir = _resolve_checkpoint_model_dir(checkpoint_path)
    if model_dir is None:
        return None
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except Exception as e:
        raise RuntimeError(
            "torch.distributed.checkpoint is required to validate DreamDojo checkpoint compatibility."
        ) from e

    reader = FileSystemReader(str(model_dir))
    metadata = reader.read_metadata()
    state_meta = getattr(metadata, "state_dict_metadata", {}) or {}

    widths = set()
    for key, tensor_meta in state_meta.items():
        key_str = str(key)
        if "action_embedder" not in key_str or not key_str.endswith(".weight"):
            continue
        size = getattr(tensor_meta, "size", None)
        if size is None or len(size) < 2:
            continue
        widths.add(int(size[1]))

    if not widths:
        return None
    resolved = min(widths)
    if len(widths) > 1:
        logger.info(
            "Detected multiple action_embedder widths in checkpoint %s: %s; using action-input width=%d",
            model_dir,
            sorted(widths),
            resolved,
        )
    return resolved


def _resolve_dreamdojo_config_yaml(config_token: str, dreamdojo_repo: Optional[Path]) -> Optional[Path]:
    token = config_token.strip()
    if not token:
        return None
    if token.lower().startswith("cosmos_predict2"):
        return None
    if dreamdojo_repo is None:
        raise RuntimeError(
            "DreamDojo config hint was provided, but dreamdojo_repo is not set. "
            "Set finetune.dreamdojo_repo or set finetune.eval_world_experiment to a cosmos experiment name."
        )

    maybe_path = Path(token)
    if maybe_path.is_absolute() or "/" in token or token.endswith(".yaml"):
        candidate = maybe_path if maybe_path.is_absolute() else (dreamdojo_repo / "configs" / maybe_path)
        if candidate.suffix != ".yaml":
            candidate_yaml = candidate.with_suffix(".yaml")
            if candidate_yaml.exists():
                candidate = candidate_yaml
        return candidate

    stem = token.lower()
    if stem.startswith("dreamdojo_"):
        stem = stem[len("dreamdojo_") :]
    return dreamdojo_repo / "configs" / f"{Path(stem).stem}.yaml"


def _resolve_action_conditioned_experiment(
    checkpoint_path: Path,
    configured_experiment: Optional[str],
    dreamdojo_repo: Optional[Path],
) -> str:
    checkpoint_width = _read_action_embed_width(checkpoint_path)
    requested = (configured_experiment or "").strip()

    if requested:
        if requested.lower().startswith("cosmos_predict2"):
            experiment_name = requested
        else:
            config_path = _resolve_dreamdojo_config_yaml(requested, dreamdojo_repo)
            if config_path is None or not config_path.exists():
                raise RuntimeError(
                    f"DreamDojo experiment config not found for '{requested}'. "
                    "Set finetune.eval_world_experiment explicitly to the matching cosmos experiment."
                )
            text = config_path.read_text(encoding="utf-8")
            match = re.search(r"^\s*action_dim\s*:\s*(\d+)\s*$", text, flags=re.MULTILINE)
            if not match:
                raise RuntimeError(
                    f"Could not read action_dim from DreamDojo config: {config_path}. "
                    "Set finetune.eval_world_experiment explicitly."
                )
            action_dim = int(match.group(1))
            experiment_name = _EXPERIMENT_BY_ACTION_DIM.get(action_dim)
            if not experiment_name:
                raise RuntimeError(
                    f"Unsupported DreamDojo action_dim={action_dim} from {config_path}. "
                    "Set finetune.eval_world_experiment explicitly to a compatible cosmos experiment."
                )
            logger.info(
                "Resolved eval world experiment from DreamDojo config %s (action_dim=%d): %s",
                config_path,
                action_dim,
                experiment_name,
            )
    else:
        if checkpoint_width is None:
            raise RuntimeError(
                "Could not infer action embedding width from checkpoint metadata. "
                "Set finetune.eval_world_experiment explicitly to avoid mismatched world-model config."
            )
        experiment_name = _EXPERIMENT_BY_ACTION_EMBED_WIDTH.get(checkpoint_width)
        if not experiment_name:
            raise RuntimeError(
                f"Unsupported action embed width {checkpoint_width} in checkpoint {checkpoint_path}. "
                "Set finetune.eval_world_experiment explicitly."
            )
        logger.info(
            "Auto-resolved eval world experiment from checkpoint action embed width=%d: %s",
            checkpoint_width,
            experiment_name,
        )

    expected_width = _ACTION_EMBED_WIDTH_BY_EXPERIMENT.get(experiment_name)
    if checkpoint_width is not None and expected_width is not None and checkpoint_width != expected_width:
        raise RuntimeError(
            "DreamDojo checkpoint is incompatible with selected action-conditioned experiment: "
            f"checkpoint action embed width={checkpoint_width}, experiment expects {expected_width} "
            f"({experiment_name})."
        )

    return experiment_name


def _build_inference_experiment_opts(experiment_name: str) -> list[str]:
    # Some vendored DreamDojo checkouts miss GR00T dataloader registry entries.
    # These overrides only affect dataset bindings (unused in inference), while
    # preserving model architecture and checkpoint compatibility.
    if "gr00t_gr1_customized_13frame" in experiment_name:
        return [
            "data_train=bridge_13frame_480_640_train",
            "data_val=bridge_13frame_480_640_val",
        ]
    return []


class _Video2WorldStepModel:
    """Adapter that exposes predict_next_frame() via DreamDojo's Video2WorldInference."""

    def __init__(
        self,
        pipe,
        guidance: float,
        negative_prompt: str,
        num_steps: int = 20,
        expected_action_dim: Optional[int] = None,
        expected_actions_per_latent_frame: Optional[int] = None,
    ):
        self._pipe = pipe
        self._guidance = float(guidance)
        self._negative_prompt = str(negative_prompt)
        self._num_steps = int(num_steps)
        self._expected_action_dim = expected_action_dim
        self._expected_actions_per_latent_frame = expected_actions_per_latent_frame
        self.expected_action_dim = expected_action_dim

        self._patch_action_ratio_if_needed()

    def _patch_action_ratio_if_needed(self) -> None:
        expected = self._expected_actions_per_latent_frame
        if expected is None or expected <= 0:
            return

        model = getattr(self._pipe, "model", None)
        if model is None:
            return

        patched = 0
        visited_ids: set[int] = set()

        def _patch_obj(obj) -> None:
            nonlocal patched
            if obj is None:
                return
            oid = id(obj)
            if oid in visited_ids:
                return
            visited_ids.add(oid)
            if not hasattr(obj, "_num_action_per_latent_frame"):
                return
            current_ratio = int(getattr(obj, "_num_action_per_latent_frame", 0) or 0)
            if current_ratio <= 0:
                setattr(obj, "_num_action_per_latent_frame", int(expected))
                patched += 1
                logger.warning(
                    "Patched invalid _num_action_per_latent_frame=%d to %d from checkpoint-compatible metadata.",
                    current_ratio,
                    expected,
                )

        _patch_obj(model)
        _patch_obj(getattr(model, "net", None))
        modules_fn = getattr(model, "modules", None)
        if callable(modules_fn):
            try:
                for module in modules_fn():
                    _patch_obj(module)
            except Exception:
                pass

        if patched:
            logger.info("Patched _num_action_per_latent_frame on %d module(s).", patched)

    def predict_next_frame(self, current_frame: np.ndarray, action) -> np.ndarray:
        import torch
        import torchvision.transforms.functional as TVF

        self._patch_action_ratio_if_needed()

        frame = np.asarray(current_frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[2] > 3:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        h, w = frame.shape[:2]

        model_required_frames = int(
            self._pipe.model.tokenizer.get_pixel_num_frames(self._pipe.model.config.state_t)
        )
        img_tensor = TVF.to_tensor(frame).unsqueeze(0) * 255.0  # (1, C, H, W)
        if model_required_frames <= 1:
            video_frames = img_tensor
        else:
            padding = torch.zeros(
                (model_required_frames - 1, img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]),
                dtype=img_tensor.dtype,
            )
            video_frames = torch.cat([img_tensor, padding], dim=0)
        vid_input = video_frames.to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        action_array = _normalize_action_chunk(
            action,
            expected_action_dim=self._expected_action_dim,
            actions_per_latent_frame=self._expected_actions_per_latent_frame,
        )
        action_tensor = torch.from_numpy(action_array).float()

        video = self._pipe.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=action_tensor,
            guidance=self._guidance,
            num_video_frames=model_required_frames,
            num_latent_conditional_frames=1,
            resolution=f"{h},{w}",
            seed=1,
            negative_prompt=self._negative_prompt,
            num_steps=self._num_steps,
            lam_video=None,
        )

        video_normalized = (video + 1.0) / 2.0
        video_uint8 = (
            (torch.clamp(video_normalized[0], 0, 1) * 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .detach()
            .cpu()
            .numpy()
        )
        return video_uint8[-1]


def _try_import_action_conditioned_modules():
    """Try modern DreamDojo action-conditioned imports."""
    ActionConditionedInferenceArguments = None
    Video2WorldInference = None
    try:
        from cosmos_predict2.action_conditioned_config import (  # type: ignore
            ActionConditionedInferenceArguments as _Args,
        )
        from cosmos_predict2._src.predict2.inference.video2world import (  # type: ignore
            Video2WorldInference as _Pipe,
        )

        ActionConditionedInferenceArguments = _Args
        Video2WorldInference = _Pipe
    except Exception:
        ActionConditionedInferenceArguments = None
        Video2WorldInference = None
    return ActionConditionedInferenceArguments, Video2WorldInference


def _load_legacy_action_conditioned_model(effective_checkpoint: Path, device: str):
    """Fallback loader for legacy tests/vendors exposing setup() under action_conditioned.*."""
    legacy_candidates = (
        "cosmos_predict2.action_conditioned.inference",
        "cosmos_predict2.action_conditioned",
    )
    for module_name in legacy_candidates:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        setup_fn = getattr(module, "setup", None)
        args_cls = getattr(module, "ActionConditionedInferenceArguments", None)
        if not callable(setup_fn) or args_cls is None:
            continue
        try:
            args = args_cls(checkpoint_dir=str(effective_checkpoint))
        except TypeError:
            args = args_cls()
            if hasattr(args, "checkpoint_dir"):
                setattr(args, "checkpoint_dir", str(effective_checkpoint))
        model = setup_fn(args)
        if hasattr(model, "to"):
            try:
                model = model.to(device)
            except Exception:
                pass
        if hasattr(model, "predict_next_frame"):
            return model
    return None


def load_dreamdojo_world_model(
    checkpoint_path: Path,
    adapted_checkpoint: Optional[Path] = None,
    configured_experiment: Optional[str] = None,
    dreamdojo_repo: Optional[Path] = None,
    device: str = "cuda",
):
    """Load DreamDojo world model for action-conditioned video prediction.

    DreamDojo is built on cosmos_predict2 and uses the action-conditioned
    inference API. If adapted_checkpoint is provided, loads the fine-tuned
    model instead of the baseline.

    Returns an object with a predict_next_frame(frame, action) method.
    """
    effective_checkpoint = Path(adapted_checkpoint) if adapted_checkpoint else Path(checkpoint_path)
    effective_checkpoint = _resolve_world_model_checkpoint_path(effective_checkpoint)
    checkpoint_for_loader = _resolve_checkpoint_model_dir(effective_checkpoint) or effective_checkpoint

    if not checkpoint_for_loader.exists():
        raise RuntimeError(f"DreamDojo checkpoint path does not exist: {checkpoint_for_loader}")

    logger.info("Loading DreamDojo from %s", checkpoint_for_loader)
    if adapted_checkpoint:
        logger.info("Using adapted (fine-tuned) checkpoint")
    else:
        logger.info("Using baseline (pretrained) checkpoint")

    # Prefer already-installed cosmos_predict2 first, then explicit repo fallback.
    ActionConditionedInferenceArguments, Video2WorldInference = _try_import_action_conditioned_modules()
    if ActionConditionedInferenceArguments is None or Video2WorldInference is None:
        if dreamdojo_repo is not None:
            if not dreamdojo_repo.exists():
                raise RuntimeError(
                    "DreamDojo/cosmos_predict2 is not importable and finetune.dreamdojo_repo "
                    f"does not exist: {dreamdojo_repo}"
                )
            dreamdojo_repo_str = str(dreamdojo_repo)
            if dreamdojo_repo_str not in sys.path:
                sys.path.insert(0, dreamdojo_repo_str)
            ActionConditionedInferenceArguments, Video2WorldInference = (
                _try_import_action_conditioned_modules()
            )

    if ActionConditionedInferenceArguments is None or Video2WorldInference is None:
        legacy_model = _load_legacy_action_conditioned_model(
            effective_checkpoint=effective_checkpoint,
            device=device,
        )
        if legacy_model is not None:
            return legacy_model
        raise RuntimeError(
            "DreamDojo/cosmos_predict2 is not importable. Install DreamDojo in the current "
            "environment or pass finetune.dreamdojo_repo to enable repo-path fallback."
        )

    experiment_name = _resolve_action_conditioned_experiment(
        checkpoint_path=checkpoint_for_loader,
        configured_experiment=configured_experiment,
        dreamdojo_repo=dreamdojo_repo,
    )
    checkpoint_width = _read_action_embed_width(checkpoint_for_loader)
    expected_action_dim = _ACTION_DIM_BY_EXPERIMENT.get(experiment_name)
    expected_actions_per_latent_frame: Optional[int] = None
    if checkpoint_width is not None and expected_action_dim is not None and expected_action_dim > 0:
        expected_actions_per_latent_frame = checkpoint_width // expected_action_dim
        if expected_actions_per_latent_frame <= 0:
            expected_actions_per_latent_frame = None
    logger.info("Using action-conditioned experiment: %s", experiment_name)

    inference_args = ActionConditionedInferenceArguments()

    pipe = Video2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=str(checkpoint_for_loader),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=_ACTION_CONFIG_FILE,
        experiment_opts=_build_inference_experiment_opts(experiment_name),
        offload_diffusion_model=False,
        offload_text_encoder=False,
        offload_tokenizer=False,
    )
    return _Video2WorldStepModel(
        pipe=pipe,
        guidance=float(inference_args.guidance),
        negative_prompt=str(inference_args.negative_prompt),
        num_steps=20,
        expected_action_dim=expected_action_dim,
        expected_actions_per_latent_frame=expected_actions_per_latent_frame,
    )


def run_rollout(
    world_model,
    openvla_model,
    openvla_processor,
    initial_frame: np.ndarray,
    task_prompt: str,
    max_steps: int = 100,
    unnorm_key: Optional[str] = "bridge_orig",
    output_dir: Optional[Path] = None,
    clip_name: str = "rollout",
    device: str = "cuda",
) -> RolloutResult:
    """Run a single policy rollout: OpenVLA-OFT predicts actions, DreamDojo generates frames."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - exercised in lightweight test envs
        Image = None

    try:
        import torch

        torch_dtype = torch.bfloat16
    except ImportError:  # pragma: no cover - exercised in lightweight test envs
        torch = None
        torch_dtype = None

    frames = [initial_frame.copy()]
    actions = []
    current_frame = initial_frame

    logger.info("Running rollout for task: %s (max %d steps)", task_prompt, max_steps)

    for step in range(max_steps):
        # OpenVLA-OFT predicts action from current observation
        image = Image.fromarray(current_frame) if Image else current_frame
        prompt = f"In: What action should the robot take to {task_prompt}?\nOut:"

        inputs = openvla_processor(prompt, image, return_tensors="pt")
        if hasattr(inputs, "to"):
            if torch_dtype is not None:
                inputs = inputs.to(device, dtype=torch_dtype)
            else:
                inputs = inputs.to(device)
        else:
            inputs = {
                key: (
                    value.to(device, dtype=torch_dtype)
                    if hasattr(value, "to") and torch_dtype is not None
                    else value.to(device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }

        predict_kwargs = dict(inputs)
        predict_kwargs["do_sample"] = False
        if unnorm_key:
            predict_kwargs["unnorm_key"] = unnorm_key

        try:
            action = openvla_model.predict_action(**predict_kwargs)
        except TypeError:
            predict_kwargs.pop("unnorm_key", None)
            action = openvla_model.predict_action(**predict_kwargs)

        actions.append(action.tolist() if hasattr(action, "tolist") else list(action))

        # DreamDojo generates next frame from action
        if not hasattr(world_model, "predict_next_frame"):
            raise RuntimeError("World model missing predict_next_frame()")
        next_frame = world_model.predict_next_frame(current_frame, action)

        frames.append(next_frame)
        current_frame = next_frame

    # Save rollout video
    video_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{clip_name}.mp4"

        import cv2

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    return RolloutResult(
        task_prompt=task_prompt,
        condition="",
        video_path=video_path,
        action_sequence=actions,
        num_steps=len(actions),
        success=True,
    )
