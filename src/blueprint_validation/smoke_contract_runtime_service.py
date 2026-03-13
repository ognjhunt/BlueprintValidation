"""Local smoke-contract NeoVerse runtime service entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .neoverse_runtime_core import SmokeContractRuntimeStore
from .runtime_service_app import create_runtime_app


def _smoke_runtime_store() -> SmokeContractRuntimeStore:
    host = os.getenv("NEOVERSE_RUNTIME_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("NEOVERSE_RUNTIME_SERVICE_PORT", "8788"))
    base_url = os.getenv("NEOVERSE_RUNTIME_PUBLIC_BASE_URL", f"http://{host}:{port}")
    ws_base_url = os.getenv("NEOVERSE_RUNTIME_PUBLIC_WS_BASE_URL")
    root_dir = Path(os.getenv("NEOVERSE_RUNTIME_ROOT", "./data/neoverse-smoke-runtime"))
    return SmokeContractRuntimeStore(root_dir=root_dir, base_url=base_url, ws_base_url=ws_base_url)


STORE = _smoke_runtime_store()
app = create_runtime_app(backend=STORE, title="NeoVerse Smoke Contract Runtime")


def main() -> int:
    host = os.getenv("NEOVERSE_RUNTIME_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("NEOVERSE_RUNTIME_SERVICE_PORT", "8788"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
