"""Policy adapter registry and base interfaces."""

from .base import PolicyAdapter, PolicyHandle, PolicyTrainingResult
from .registry import get_policy_adapter

__all__ = [
    "PolicyAdapter",
    "PolicyHandle",
    "PolicyTrainingResult",
    "get_policy_adapter",
]
