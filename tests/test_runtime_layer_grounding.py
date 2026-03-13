from __future__ import annotations

import json
from pathlib import Path

from blueprint_validation.runtime_layer_grounding import (
    _canonical_hash_payload,
    compute_canonical_package_version,
    verify_canonical_package_version,
)


def test_verify_canonical_package_version_ignores_volatile_manifest_fields(tmp_path: Path) -> None:
    scene_memory_manifest = {
        "generated_at": "2026-03-13T00:00:00Z",
        "objects": [{"id": "obj-1"}],
    }
    conditioning_bundle = {
        "runtime_registration_status": "skipped",
        "frames": ["frame-1"],
    }
    object_geometry_manifest = {
        "generated_at": "2026-03-13T00:00:00Z",
        "objects": [{"id": "obj-1", "bbox": [0, 0, 1, 1]}],
    }
    task_anchor_manifest = {
        "generated_at": "2026-03-13T00:00:00Z",
        "anchors": [{"task_id": "task-1"}],
    }
    protected_regions_manifest = {
        "generated_at": "2026-03-13T00:00:00Z",
        "regions": [],
    }
    canonical_render_policy = {
        "generated_at": "2026-03-13T00:00:00Z",
        "mode": "canonical",
    }
    presentation_variance_policy = {
        "generated_at": "2026-03-13T00:00:00Z",
        "variance": "limited",
    }

    base = tmp_path
    (base / "scene_memory_manifest.json").write_text(json.dumps(scene_memory_manifest), encoding="utf-8")
    (base / "conditioning_bundle.json").write_text(json.dumps(conditioning_bundle), encoding="utf-8")
    (base / "object_geometry_manifest.json").write_text(json.dumps(object_geometry_manifest), encoding="utf-8")
    (base / "task_anchor_manifest.json").write_text(json.dumps(task_anchor_manifest), encoding="utf-8")

    spec = {
        "site_world_id": "siteworld-1",
        "conditioning": {
            "scene_memory_manifest_path": str(base / "scene_memory_manifest.json"),
            "conditioning_bundle_path": str(base / "conditioning_bundle.json"),
            "local_paths": {
                "scene_memory_manifest_path": str(base / "scene_memory_manifest.json"),
                "conditioning_bundle_path": str(base / "conditioning_bundle.json"),
            },
        },
        "geometry": {
            "object_geometry_manifest_path": str(base / "object_geometry_manifest.json"),
        },
        "task_anchor_manifest_path": str(base / "task_anchor_manifest.json"),
    }
    spec["canonical_package_version"] = compute_canonical_package_version(
        scene_memory_manifest=_canonical_hash_payload(scene_memory_manifest),
        conditioning_bundle=_canonical_hash_payload(conditioning_bundle),
        object_geometry_manifest=_canonical_hash_payload(object_geometry_manifest),
        task_anchor_manifest=_canonical_hash_payload(task_anchor_manifest),
        site_world_spec=_canonical_hash_payload(spec),
        protected_regions_manifest=_canonical_hash_payload(protected_regions_manifest),
        canonical_render_policy=_canonical_hash_payload(canonical_render_policy),
        presentation_variance_policy=_canonical_hash_payload(presentation_variance_policy),
    )

    assert (
        verify_canonical_package_version(
            spec=spec,
            protected_regions_manifest=protected_regions_manifest,
            canonical_render_policy=canonical_render_policy,
            presentation_variance_policy=presentation_variance_policy,
        )
        is None
    )
