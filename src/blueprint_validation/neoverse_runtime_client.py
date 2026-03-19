"""Compatibility wrapper around the generic site-world runtime client."""

from __future__ import annotations

from .runtime_service_client import RuntimeServiceClient, RuntimeServiceClientConfig


NeoVerseRuntimeClientConfig = RuntimeServiceClientConfig
NeoVerseRuntimeClient = RuntimeServiceClient

__all__ = ["NeoVerseRuntimeClient", "NeoVerseRuntimeClientConfig"]
