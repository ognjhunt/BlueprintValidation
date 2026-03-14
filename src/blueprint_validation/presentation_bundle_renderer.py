from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import threading
from typing import Any, Mapping, Optional

import numpy as np

from .optional_dependencies import require_optional_dependency


logger = logging.getLogger(__name__)
_SUPERSPLAT_CHUNK_SIZE = 256
_SH_C0 = np.float32(0.28209479177387814)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_local_path(*values: Any) -> Optional[Path]:
    for raw in values:
        value = str(raw or "").strip()
        if not value or value.startswith(("gs://", "http://", "https://")):
            continue
        path = Path(value).expanduser().resolve()
        if path.exists():
            return path
    return None


@dataclass(frozen=True)
class GsplatCameraPose:
    c2w: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def viewmat(self) -> Any:
        torch = require_optional_dependency(
            "torch",
            extra="vision",
            purpose="presentation bundle rendering",
        )
        return torch.from_numpy(np.linalg.inv(self.c2w).astype(np.float32))

    def K(self) -> Any:
        torch = require_optional_dependency(
            "torch",
            extra="vision",
            purpose="presentation bundle rendering",
        )
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )


@dataclass
class GaussianSplatData:
    means: Any
    scales: Any
    quats: Any
    opacities: Any
    sh_coeffs: Any
    num_points: int

    def to(self, device: str | Any) -> "GaussianSplatData":
        return GaussianSplatData(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            opacities=self.opacities.to(device),
            sh_coeffs=self.sh_coeffs.to(device),
            num_points=self.num_points,
        )


def _sorted_sh_names(properties: Any, prefix: str = "f_rest_") -> list[str]:
    names = [prop.name for prop in properties if prop.name.startswith(prefix)]
    return sorted(
        names,
        key=lambda name: int(name[len(prefix) :]) if name[len(prefix) :].isdigit() else name,
    )


def _is_supersplat_compressed_ply(plydata: Any) -> bool:
    if "chunk" not in plydata or "vertex" not in plydata:
        return False
    vertex_names = {prop.name for prop in plydata["vertex"].properties}
    required = {"packed_position", "packed_rotation", "packed_scale", "packed_color"}
    return required.issubset(vertex_names)


def _decode_supersplat_compressed(plydata: Any) -> tuple[np.ndarray, ...]:
    vertex = plydata["vertex"]
    chunk = plydata["chunk"]
    n = len(vertex.data)
    if n == 0:
        raise ValueError("compressed splat asset has zero vertices")

    chunk_indices = np.arange(n, dtype=np.int64) // _SUPERSPLAT_CHUNK_SIZE
    if len(chunk.data) <= int(chunk_indices[-1]):
        raise ValueError("compressed splat chunk table is truncated")

    def chunk_field(name: str) -> np.ndarray:
        return np.asarray(chunk[name], dtype=np.float32)[chunk_indices]

    packed_position = np.asarray(vertex["packed_position"], dtype=np.uint32)
    packed_rotation = np.asarray(vertex["packed_rotation"], dtype=np.uint32)
    packed_scale = np.asarray(vertex["packed_scale"], dtype=np.uint32)
    packed_color = np.asarray(vertex["packed_color"], dtype=np.uint32)

    min_x = chunk_field("min_x")
    max_x = chunk_field("max_x")
    min_y = chunk_field("min_y")
    max_y = chunk_field("max_y")
    min_z = chunk_field("min_z")
    max_z = chunk_field("max_z")

    pos_x = ((packed_position >> 21) & 0x7FF).astype(np.float32) / 2047.0
    pos_y = ((packed_position >> 11) & 0x3FF).astype(np.float32) / 1023.0
    pos_z = (packed_position & 0x7FF).astype(np.float32) / 2047.0
    means = np.stack(
        [
            min_x + pos_x * (max_x - min_x),
            min_y + pos_y * (max_y - min_y),
            min_z + pos_z * (max_z - min_z),
        ],
        axis=-1,
    ).astype(np.float32)

    min_sx = chunk_field("min_scale_x")
    max_sx = chunk_field("max_scale_x")
    min_sy = chunk_field("min_scale_y")
    max_sy = chunk_field("max_scale_y")
    min_sz = chunk_field("min_scale_z")
    max_sz = chunk_field("max_scale_z")

    scale_x = ((packed_scale >> 21) & 0x7FF).astype(np.float32) / 2047.0
    scale_y = ((packed_scale >> 11) & 0x3FF).astype(np.float32) / 1023.0
    scale_z = (packed_scale & 0x7FF).astype(np.float32) / 2047.0
    scales = np.stack(
        [
            min_sx + scale_x * (max_sx - min_sx),
            min_sy + scale_y * (max_sy - min_sy),
            min_sz + scale_z * (max_sz - min_sz),
        ],
        axis=-1,
    ).astype(np.float32)

    largest = (packed_rotation >> 30).astype(np.int8)
    c0 = ((packed_rotation >> 20) & 0x3FF).astype(np.float32)
    c1 = ((packed_rotation >> 10) & 0x3FF).astype(np.float32)
    c2 = (packed_rotation & 0x3FF).astype(np.float32)
    decoded = (np.stack([c0, c1, c2], axis=-1) / 1023.0 - 0.5) / (np.sqrt(2.0) * 0.5)

    quats = np.zeros((n, 4), dtype=np.float32)
    component_orders = {
        0: (1, 2, 3),
        1: (0, 2, 3),
        2: (0, 1, 3),
        3: (0, 1, 2),
    }
    for largest_idx, (a, b, c) in component_orders.items():
        mask = largest == largest_idx
        if not np.any(mask):
            continue
        quats[mask, a] = decoded[mask, 0]
        quats[mask, b] = decoded[mask, 1]
        quats[mask, c] = decoded[mask, 2]
        sq = np.sum(quats[mask] * quats[mask], axis=1)
        quats[mask, largest_idx] = np.sqrt(np.clip(1.0 - sq, 0.0, 1.0))
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)

    min_r = chunk_field("min_r")
    max_r = chunk_field("max_r")
    min_g = chunk_field("min_g")
    max_g = chunk_field("max_g")
    min_b = chunk_field("min_b")
    max_b = chunk_field("max_b")

    color_r = ((packed_color >> 24) & 0xFF).astype(np.float32) / 255.0
    color_g = ((packed_color >> 16) & 0xFF).astype(np.float32) / 255.0
    color_b = ((packed_color >> 8) & 0xFF).astype(np.float32) / 255.0
    alpha = (packed_color & 0xFF).astype(np.float32) / 255.0

    dc_r = min_r + color_r * (max_r - min_r)
    dc_g = min_g + color_g * (max_g - min_g)
    dc_b = min_b + color_b * (max_b - min_b)
    sh_dc = np.stack(
        [
            (dc_r - 0.5) / _SH_C0,
            (dc_g - 0.5) / _SH_C0,
            (dc_b - 0.5) / _SH_C0,
        ],
        axis=-1,
    ).astype(np.float32)

    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    opacities = np.log(alpha / (1.0 - alpha)).astype(np.float32)
    sh_coeffs = sh_dc[:, None, :]

    if "sh" in plydata:
        sh = plydata["sh"]
        sh_rest_names = _sorted_sh_names(sh.properties)
        if sh_rest_names:
            sh_rest_raw = np.stack(
                [np.asarray(sh[name], dtype=np.uint8) for name in sh_rest_names],
                axis=-1,
            ).astype(np.float32)
            sh_rest_raw = ((sh_rest_raw + 0.5) / 256.0 - 0.5) * 8.0
            usable = (sh_rest_raw.shape[1] // 3) * 3
            if usable > 0:
                sh_rest = sh_rest_raw[:, :usable].reshape(n, usable // 3, 3)
                sh_coeffs = np.concatenate([sh_dc[:, None, :], sh_rest], axis=1).astype(np.float32)

    return means, scales, quats, opacities, sh_coeffs


def _load_splat(ply_path: Path, *, device: str) -> GaussianSplatData:
    torch = require_optional_dependency(
        "torch",
        extra="vision",
        purpose="presentation bundle rendering",
    )
    plyfile = require_optional_dependency(
        "plyfile",
        extra="vision",
        purpose="presentation bundle rendering",
    )
    plydata = plyfile.PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    vertex_names = {prop.name for prop in vertex.properties}
    n = len(vertex.data)
    if {"x", "y", "z"}.issubset(vertex_names):
        means_np = np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float32),
                np.asarray(vertex["y"], dtype=np.float32),
                np.asarray(vertex["z"], dtype=np.float32),
            ],
            axis=-1,
        )
        scales_np = np.stack(
            [
                np.asarray(vertex["scale_0"], dtype=np.float32),
                np.asarray(vertex["scale_1"], dtype=np.float32),
                np.asarray(vertex["scale_2"], dtype=np.float32),
            ],
            axis=-1,
        )
        quats_np = np.stack(
            [
                np.asarray(vertex["rot_0"], dtype=np.float32),
                np.asarray(vertex["rot_1"], dtype=np.float32),
                np.asarray(vertex["rot_2"], dtype=np.float32),
                np.asarray(vertex["rot_3"], dtype=np.float32),
            ],
            axis=-1,
        )
        opacities_np = np.asarray(vertex["opacity"], dtype=np.float32)
        sh_dc = np.stack(
            [
                np.asarray(vertex["f_dc_0"], dtype=np.float32),
                np.asarray(vertex["f_dc_1"], dtype=np.float32),
                np.asarray(vertex["f_dc_2"], dtype=np.float32),
            ],
            axis=-1,
        )[:, None, :]
        sh_rest_names = _sorted_sh_names(vertex.properties)
        if sh_rest_names:
            sh_rest_raw = np.stack(
                [np.asarray(vertex[name], dtype=np.float32) for name in sh_rest_names],
                axis=-1,
            )
            sh_rest = sh_rest_raw.reshape(n, len(sh_rest_names) // 3, 3)
            sh_coeffs_np = np.concatenate([sh_dc, sh_rest], axis=1).astype(np.float32)
        else:
            sh_coeffs_np = sh_dc.astype(np.float32)
    elif _is_supersplat_compressed_ply(plydata):
        means_np, scales_np, quats_np, opacities_np, sh_coeffs_np = _decode_supersplat_compressed(plydata)
    else:
        raise ValueError(f"unsupported splat schema in {ply_path.name}")

    data = GaussianSplatData(
        means=torch.from_numpy(means_np),
        scales=torch.from_numpy(scales_np),
        quats=torch.from_numpy(quats_np),
        opacities=torch.from_numpy(opacities_np),
        sh_coeffs=torch.from_numpy(sh_coeffs_np),
        num_points=n,
    )
    return data.to(device) if device != "cpu" else data


def _render_frame(splat: GaussianSplatData, pose: GsplatCameraPose) -> tuple[np.ndarray, np.ndarray]:
    torch = require_optional_dependency(
        "torch",
        extra="vision",
        purpose="presentation bundle rendering",
    )
    gsplat = require_optional_dependency(
        "gsplat",
        extra="vision",
        purpose="presentation bundle rendering",
    )

    viewmat = pose.viewmat().unsqueeze(0).to(splat.means.device)
    K = pose.K().unsqueeze(0).to(splat.means.device)
    background = torch.ones(3, device=splat.means.device)
    renders, _alphas, _info = gsplat.rasterization(
        means=splat.means,
        quats=splat.quats,
        scales=torch.exp(splat.scales),
        opacities=torch.sigmoid(splat.opacities),
        colors=splat.sh_coeffs,
        viewmats=viewmat,
        Ks=K,
        width=pose.width,
        height=pose.height,
        sh_degree=int(np.sqrt(splat.sh_coeffs.shape[1]) - 1),
        backgrounds=background.unsqueeze(0),
        render_mode="RGB+ED",
        packed=False,
    )
    rgb = renders[0, :, :, :3].clamp(0, 1).cpu().numpy()
    depth = renders[0, :, :, 3].cpu().numpy()
    return (rgb * 255).astype(np.uint8), depth


@dataclass
class PresentationBundleScene:
    site_world_id: str
    manifest_path: Optional[Path]
    bundle_status: str
    renderer_backend: str
    bundle_type: str
    primary_asset_path: Path
    orientation: dict[str, Any]
    splat: GaussianSplatData


class PresentationBundleRenderer:
    def __init__(self) -> None:
        self._scene_lock = threading.Lock()
        self._scenes: dict[str, PresentationBundleScene] = {}

    def invalidate(self, site_world_id: str) -> None:
        with self._scene_lock:
            self._scenes.pop(site_world_id, None)

    def _device(self) -> str:
        try:
            torch = require_optional_dependency(
                "torch",
                extra="vision",
                purpose="presentation bundle rendering",
            )
        except RuntimeError:
            return "cpu"
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"

    def _build_scene(self, site_world_id: str, spec: Mapping[str, Any]) -> PresentationBundleScene:
        presentation = dict(spec.get("presentation") or {}) if isinstance(spec.get("presentation"), Mapping) else {}
        manifest_path = _resolve_local_path(
            presentation.get("presentation_world_manifest_path"),
            presentation.get("presentation_world_manifest_uri"),
        )
        manifest = _read_json(manifest_path) if manifest_path is not None and manifest_path.is_file() else {}
        bundle_status = str(
            presentation.get("bundle_status")
            or manifest.get("bundle_status")
            or ((manifest.get("readiness") or {}).get("bundle_status") if isinstance(manifest.get("readiness"), Mapping) else "")
            or manifest.get("status")
            or "missing"
        ).strip().lower()
        renderer_backend = str(
            presentation.get("renderer_backend")
            or manifest.get("renderer_backend")
            or "gsplat"
        ).strip().lower()
        bundle_type = str(
            presentation.get("bundle_type")
            or manifest.get("bundle_type")
            or "gsplat_scene_v1"
        ).strip()
        primary_asset_path = _resolve_local_path(
            presentation.get("primary_asset_path"),
            manifest.get("primary_asset_path"),
            manifest.get("primary_asset_uri"),
        )
        if renderer_backend != "gsplat":
            raise RuntimeError(f"unsupported presentation renderer backend: {renderer_backend}")
        if bundle_type != "gsplat_scene_v1":
            raise RuntimeError(f"unsupported presentation bundle type: {bundle_type}")
        if primary_asset_path is None:
            raise RuntimeError(f"site world {site_world_id} is missing a presentation primary asset")
        splat = _load_splat(primary_asset_path, device=self._device())
        orientation = (
            dict(presentation.get("orientation") or {})
            if isinstance(presentation.get("orientation"), Mapping)
            else dict(manifest.get("orientation") or {})
            if isinstance(manifest.get("orientation"), Mapping)
            else {}
        )
        return PresentationBundleScene(
            site_world_id=site_world_id,
            manifest_path=manifest_path,
            bundle_status=bundle_status,
            renderer_backend=renderer_backend,
            bundle_type=bundle_type,
            primary_asset_path=primary_asset_path,
            orientation=orientation,
            splat=splat,
        )

    def scene(self, site_world_id: str, spec: Mapping[str, Any]) -> PresentationBundleScene:
        with self._scene_lock:
            cached = self._scenes.get(site_world_id)
            if cached is not None:
                return cached
        built = self._build_scene(site_world_id, spec)
        with self._scene_lock:
            self._scenes[site_world_id] = built
        return built

    def render_camera(
        self,
        *,
        site_world_id: str,
        spec: Mapping[str, Any],
        view_config: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cv2 = require_optional_dependency(
            "cv2",
            extra="vision",
            purpose="presentation bundle rendering",
        )
        scene = self.scene(site_world_id, spec)
        pose = GsplatCameraPose(
            c2w=np.asarray(view_config.get("world_from_camera"), dtype=np.float32).reshape(4, 4),
            fx=float(view_config.get("fx") or 0.0),
            fy=float(view_config.get("fy") or 0.0),
            cx=float(view_config.get("cx") or 0.0),
            cy=float(view_config.get("cy") or 0.0),
            width=int(view_config.get("raw_width") or 0),
            height=int(view_config.get("raw_height") or 0),
        )
        if pose.width <= 0 or pose.height <= 0:
            raise RuntimeError("presentation view config is missing image dimensions")
        rgb, depth = _render_frame(scene.splat, pose)
        rotation = int(view_config.get("display_rotation_degrees") or scene.orientation.get("display_rotation_degrees") or 0)
        if rotation == 90:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        elif rotation == 270:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        preview_width = int(view_config.get("preview_width") or pose.width)
        preview_height = int(view_config.get("preview_height") or pose.height)
        preview_rgb = cv2.resize(rgb, (preview_width, preview_height), interpolation=cv2.INTER_CUBIC)
        return preview_rgb, {
            "preview_mode": "presentation_bundle",
            "preview_source": "server_gsplat",
            "renderer_backend": scene.renderer_backend,
            "bundle_type": scene.bundle_type,
            "presentation_bundle_status": scene.bundle_status,
            "display_orientation": str(
                view_config.get("display_orientation")
                or scene.orientation.get("display_orientation")
                or "landscape"
            ),
            "display_rotation_degrees": rotation,
            "primary_asset_path": str(scene.primary_asset_path),
            "depth_range": [
                float(np.min(depth)) if depth.size else 0.0,
                float(np.max(depth)) if depth.size else 0.0,
            ],
        }
