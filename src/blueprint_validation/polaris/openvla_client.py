"""OpenVLA policy client and observation helpers for PolaRiS integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .websocket_policy import WebsocketPolicyClient

try:  # pragma: no cover - depends on upstream PolaRiS checkout
    from polaris.config import PolicyArgs
    from polaris.policy.abstract_client import InferenceClient
except Exception:  # pragma: no cover - import-safe fallback for local tests
    PolicyArgs = Any
    InferenceClient = object


def stitch_external_wrist_images(external: np.ndarray, wrist: Optional[np.ndarray]) -> np.ndarray:
    if wrist is None:
        return external
    external_rgb = np.asarray(external, dtype=np.uint8)
    wrist_rgb = np.asarray(wrist, dtype=np.uint8)
    if external_rgb.shape[0] != wrist_rgb.shape[0]:
        target_h = min(external_rgb.shape[0], wrist_rgb.shape[0])
        external_rgb = external_rgb[:target_h]
        wrist_rgb = wrist_rgb[:target_h]
    return np.concatenate([external_rgb, wrist_rgb], axis=1)


def extract_policy_observation(obs: dict[str, Any], observation_mode: str) -> dict[str, Any]:
    splat_obs = dict(obs.get("splat", {}) or {})
    external = np.asarray(splat_obs.get("external_cam"), dtype=np.uint8)
    wrist = splat_obs.get("wrist_cam")
    wrist_array = np.asarray(wrist, dtype=np.uint8) if wrist is not None else None
    policy_obs = dict(obs.get("policy", {}) or {})
    request: dict[str, Any] = {
        "instruction": "",
        "external_image": external,
        "wrist_image": wrist_array,
        "joint_position": _to_numpy_vector(policy_obs.get("arm_joint_pos")),
        "gripper_position": _to_numpy_vector(policy_obs.get("gripper_pos")),
    }
    mode = str(observation_mode or "external_only").strip().lower()
    if mode == "external_wrist_stitched":
        request["image"] = stitch_external_wrist_images(external, wrist_array)
    else:
        request["image"] = external
    return request


def normalize_openvla_action(action: Any, *, expected_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("Policy action contains non-finite values.")
    if expected_dim is not None and arr.size != int(expected_dim):
        raise ValueError(f"Expected action_dim={int(expected_dim)}, got {arr.size}.")
    return arr


def _to_numpy_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[-1])
    return arr


@dataclass
class _ClientState:
    websocket: WebsocketPolicyClient
    observation_mode: str
    expected_action_dim: int | None = None


if hasattr(InferenceClient, "register"):  # pragma: no branch

    @InferenceClient.register(client_name="OpenVLA")
    class OpenVLAInferenceClient(InferenceClient):  # type: ignore[misc]
        def __init__(self, args: PolicyArgs) -> None:
            self.args = args
            self.state = _ClientState(
                websocket=WebsocketPolicyClient(host=args.host, port=args.port),
                observation_mode=getattr(args, "observation_mode", "external_only"),
            )

        @property
        def rerender(self) -> bool:
            return True

        def infer(
            self, obs: dict[str, Any], instruction: str, return_viz: bool = False
        ) -> tuple[np.ndarray, np.ndarray | None]:
            request = extract_policy_observation(obs, self.state.observation_mode)
            request["instruction"] = instruction
            response = self.state.websocket.infer(request)
            action = normalize_openvla_action(
                response.get("action"),
                expected_dim=self.state.expected_action_dim,
            )
            viz = request.get("image") if return_viz else None
            return action, viz

        def reset(self):
            return
