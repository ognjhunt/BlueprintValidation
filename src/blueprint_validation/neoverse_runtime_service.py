"""Production NeoVerse runtime service entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from blueprint_contracts.site_world_contract import load_site_world_bundle

from .neoverse_production_runtime import NeoVerseProductionRuntimeStore
from .runtime_service_app import create_runtime_app


def _production_runtime_store() -> NeoVerseProductionRuntimeStore:
    host = os.getenv("NEOVERSE_RUNTIME_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("NEOVERSE_RUNTIME_SERVICE_PORT", "8787"))
    base_url = os.getenv("NEOVERSE_RUNTIME_PUBLIC_BASE_URL", f"http://{host}:{port}")
    ws_base_url = os.getenv("NEOVERSE_RUNTIME_PUBLIC_WS_BASE_URL")
    root_dir = Path(os.getenv("NEOVERSE_RUNTIME_ROOT", "./data/neoverse-runtime"))
    return NeoVerseProductionRuntimeStore(root_dir=root_dir, base_url=base_url, ws_base_url=ws_base_url)


def _bootstrap_registration_paths() -> list[Path]:
    raw_values = [
        os.getenv("NEOVERSE_RUNTIME_BOOTSTRAP_REGISTRATION_PATH", ""),
        os.getenv("NEOVERSE_RUNTIME_BOOTSTRAP_REGISTRATION_PATHS", ""),
    ]
    entries: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_values:
        for chunk in raw.replace(",", "\n").replace(os.pathsep, "\n").splitlines():
            value = chunk.strip()
            if not value:
                continue
            path = Path(value).expanduser().resolve()
            if path in seen:
                continue
            seen.add(path)
            entries.append(path)
    return entries


def _bootstrap_site_worlds(store: NeoVerseProductionRuntimeStore) -> None:
    for registration_path in _bootstrap_registration_paths():
        bundle = load_site_world_bundle(registration_path, require_spec=True)
        payload = store.register_site_world_package(
            spec=dict(bundle.spec or {}),
            registration=dict(bundle.registration or {}),
            health=dict(bundle.health or {}),
        )
        print(
            "[neoverse_runtime_service] bootstrapped site world "
            f"site_world_id={payload.get('site_world_id')} registration_path={registration_path}",
            flush=True,
        )


STORE = _production_runtime_store()
_bootstrap_site_worlds(STORE)
app = create_runtime_app(backend=STORE, title="NeoVerse Production Runtime")


def main() -> int:
    host = os.getenv("NEOVERSE_RUNTIME_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("NEOVERSE_RUNTIME_SERVICE_PORT", "8787"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
