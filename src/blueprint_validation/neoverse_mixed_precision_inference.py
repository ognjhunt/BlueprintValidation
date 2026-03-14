"""NeoVerse bridge with explicit mixed precision controls for runtime rendering."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _runtime_frame_budgets(*, static_scene: bool, requested: int) -> list[int]:
    budget = max(1, int(requested))
    if static_scene:
        return [min(2, budget)]
    budgets = [budget]
    if budget > 4:
        budgets.append(max(4, budget // 2))
    if 2 not in budgets:
        budgets.append(2)
    return budgets


def _runtime_num_frames(static_scene: bool) -> int:
    raw = str(os.environ.get("NEOVERSE_RUNTIME_NUM_FRAMES") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 2 if static_scene else 8


def _runtime_inference_steps() -> int:
    raw = str(os.environ.get("NEOVERSE_RUNTIME_INFERENCE_STEPS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 8


def _runtime_cfg_scale() -> float:
    raw = str(os.environ.get("NEOVERSE_RUNTIME_CFG_SCALE") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 1.0


def _disable_lora_by_default() -> bool:
    raw = str(os.environ.get("NEOVERSE_RUNTIME_ENABLE_LORA") or "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def _requested_reconstructor_device(device: str) -> list[str]:
    forced = str(os.environ.get("NEOVERSE_RUNTIME_RECONSTRUCTOR_DEVICE") or "").strip()
    if forced:
        return [forced]
    return [device, "cpu"] if device != "cpu" else ["cpu"]


def _normalize_device(raw: str) -> str:
    value = str(raw or "cuda:0").strip() or "cuda:0"
    if value == "cuda":
        return "cuda:0"
    return value


def _configure_imports() -> Path:
    repo_root = Path(str(os.environ.get("NEOVERSE_REPO_ROOT") or "")).expanduser().resolve()
    if not repo_root.is_dir():
        raise RuntimeError("NEOVERSE_REPO_ROOT is not configured or does not exist.")
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    return repo_root


def _lora_path(model_root: Path) -> str | None:
    candidate = model_root / "NeoVerse" / "loras" / "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    return str(candidate) if candidate.is_file() else None


def _load_pipeline(*, model_root: Path, reconstructor_path: Path, device: str, use_lora: bool, low_vram: bool):
    from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline

    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=str(model_root),
        reconstructor_path=str(reconstructor_path),
        lora_path=_lora_path(model_root) if use_lora else None,
        lora_alpha=1.0,
        device=device,
        torch_dtype=torch.bfloat16,
        enable_vram_management=False,
    )
    if low_vram:
        # Let the pipeline offload large modules between stages instead of pinning everything on GPU.
        pipe.enable_vram_management(vram_buffer=1.0)
    pipe.reconstructor = pipe.reconstructor.to(dtype=torch.float32, device="cpu")
    pipe.vram_management_enabled = True
    pipe.device = device
    return pipe


def _render_once(
    *,
    pipe: Any,
    input_path: Path,
    output_path: Path,
    prompt: str,
    trajectory_name: str,
    static_scene: bool,
    num_frames: int,
    device: str,
    inference_steps: int,
    cfg_scale: float,
    reconstructor_device: str,
) -> None:
    from diffsynth import save_video
    from diffsynth.utils.auxiliary import CameraTrajectory, homo_matrix_inverse, load_video
    from torchvision.transforms import functional as tvf

    print(
        f"[neoverse_mixed_precision] loading views input={input_path} frames={num_frames} static_scene={static_scene}",
        flush=True,
    )
    images = load_video(
        str(input_path),
        num_frames,
        resolution=(560, 336),
        resize_mode="center_crop",
        static_scene=static_scene,
    )
    camera_trajectory = CameraTrajectory.from_predefined(trajectory_name, num_frames=num_frames, mode="relative")
    height, width = images[0].size[1], images[0].size[0]
    view_device = device if reconstructor_device.startswith("cuda") else reconstructor_device
    print(
        f"[neoverse_mixed_precision] prepared views size={width}x{height} reconstructor_device={reconstructor_device} target_device={device}",
        flush=True,
    )
    timestamps = torch.zeros((1, len(images)), dtype=torch.int64, device=view_device)
    if not static_scene:
        timestamps = torch.arange(0, len(images), dtype=torch.int64, device=view_device).unsqueeze(0)
    views = {
        "img": torch.stack([tvf.to_tensor(image)[None] for image in images], dim=1).to(view_device, dtype=torch.float32),
        "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=view_device),
        "is_static": torch.ones((1, len(images)), dtype=torch.bool, device=view_device)
        if static_scene
        else torch.zeros((1, len(images)), dtype=torch.bool, device=view_device),
        "timestamp": timestamps,
    }
    if getattr(pipe, "vram_management_enabled", False):
        pipe.load_models_to_device([])
    print("[neoverse_mixed_precision] running reconstructor", flush=True)
    pipe.reconstructor.to(reconstructor_device)
    with torch.autocast("cuda", enabled=False):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)
    pipe.reconstructor.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not reconstructor_device.startswith("cuda"):
        predictions["rendered_intrinsics"] = predictions["rendered_intrinsics"].to(device)
        predictions["rendered_extrinsics"] = predictions["rendered_extrinsics"].to(device)
        predictions["rendered_timestamps"] = predictions["rendered_timestamps"].to(device)
        for batch in predictions["splats"]:
            for splat in batch:
                for name, value in splat.__dict__.items():
                    if isinstance(value, torch.Tensor):
                        setattr(splat, name, value.to(device))
        views = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in views.items()
        }

    print("[neoverse_mixed_precision] running gaussian rasterizer", flush=True)
    gaussians = predictions["splats"]
    intrinsics = predictions["rendered_intrinsics"][0]
    extrinsics = predictions["rendered_extrinsics"][0]
    rendered_timestamps = predictions["rendered_timestamps"][0]
    intrinsics = intrinsics[:1].repeat(len(camera_trajectory), 1, 1)
    rendered_timestamps = rendered_timestamps[:1].repeat(len(camera_trajectory))
    target_cam2world = camera_trajectory.c2w.to(device=device, dtype=extrinsics.dtype)
    if camera_trajectory.mode == "relative" and not static_scene:
        target_cam2world = extrinsics @ target_cam2world
    target_world2cam = homo_matrix_inverse(target_cam2world)
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[target_world2cam],
        render_Ks=[intrinsics],
        render_timestamps=[rendered_timestamps],
        sh_degree=0,
        width=width,
        height=height,
    )
    target_mask = (target_alpha > 1.0).float()
    target_rgb[0, 0] = views["img"][0, 0].permute(1, 2, 0)
    target_mask[0, 0] = 1.0
    print(
        f"[neoverse_mixed_precision] running diffusion frames={len(camera_trajectory)} steps={inference_steps} cfg={cfg_scale}",
        flush=True,
    )
    frames = pipe(
        prompt=prompt,
        negative_prompt="",
        seed=42,
        rand_device=device,
        height=height,
        width=width,
        num_frames=len(camera_trajectory),
        cfg_scale=cfg_scale,
        num_inference_steps=inference_steps,
        tiled=False,
        source_views=views,
        target_rgb=target_rgb,
        target_depth=target_depth,
        target_mask=target_mask,
        target_poses=target_cam2world.unsqueeze(0),
        target_intrs=intrinsics.unsqueeze(0),
    )
    print(f"[neoverse_mixed_precision] saving video to {output_path}", flush=True)
    save_video(frames, str(output_path), fps=16)


def run_inference(args: argparse.Namespace) -> Path:
    _configure_imports()
    device = _normalize_device(args.device)
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    model_root = Path(args.model_path).expanduser().resolve()
    reconstructor_path = Path(args.reconstructor_path).expanduser().resolve()
    static_scene = bool(args.static_scene or input_path.suffix.lower() in _IMAGE_SUFFIXES)
    use_lora = not (args.disable_lora or _disable_lora_by_default())
    low_vram = bool(args.low_vram)
    inference_steps = _runtime_inference_steps()
    cfg_scale = _runtime_cfg_scale()
    requested_frames = args.num_frames or _runtime_num_frames(static_scene)
    attempts: list[str] = []

    for num_frames in _runtime_frame_budgets(static_scene=static_scene, requested=requested_frames):
        for reconstructor_device in _requested_reconstructor_device(device):
            attempt_label = (
                f"frames={num_frames} lora={'on' if use_lora else 'off'} "
                f"low_vram={'on' if low_vram else 'off'} reconstructor={reconstructor_device}"
            )
            start = time.time()
            try:
                print(f"[neoverse_mixed_precision] start {attempt_label}", flush=True)
                pipe = _load_pipeline(
                    model_root=model_root,
                    reconstructor_path=reconstructor_path,
                    device=device,
                    use_lora=use_lora,
                    low_vram=low_vram,
                )
                try:
                    _render_once(
                        pipe=pipe,
                        input_path=input_path,
                        output_path=output_path,
                        prompt=args.prompt,
                        trajectory_name=args.trajectory,
                        static_scene=static_scene,
                        num_frames=num_frames,
                        device=device,
                        inference_steps=inference_steps,
                        cfg_scale=cfg_scale,
                        reconstructor_device=reconstructor_device,
                    )
                finally:
                    del pipe
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(f"[neoverse_mixed_precision] success {attempt_label} elapsed={time.time() - start:.2f}s", flush=True)
                return output_path
            except Exception as exc:  # pragma: no cover - exercised in host debugging
                attempts.append(f"{attempt_label}: {exc}")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                print(f"[neoverse_mixed_precision] failed {attempt_label}: {exc}", file=sys.stderr, flush=True)
                if reconstructor_device == "cpu":
                    break

    raise RuntimeError("NeoVerse mixed-precision attempts failed: " + " | ".join(attempts))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NeoVerse with runtime-safe mixed precision.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reconstructor_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=0)
    parser.add_argument("--static_scene", action="store_true")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--disable_lora", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_inference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
