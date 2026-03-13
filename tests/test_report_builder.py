from __future__ import annotations

import json
from pathlib import Path

from blueprint_validation.config import ValidationConfig
from blueprint_validation.reporting.report_builder import build_report


def test_report_builder_uses_session_and_export_artifacts(tmp_path: Path) -> None:
    (tmp_path / "session_state.json").write_text(
        json.dumps(
            {
                "session_id": "session-1",
                "site_world_id": "site-world-1",
                "runtime_backend_public_name": "NeoVerse runtime service",
                "runtime_backend_selected": "neoverse_service",
                "runtime_kind": "neoverse_production",
                "production_grade": True,
                "runtime_model_identity": {"model_id": "test-model"},
                "runtime_checkpoint_identity": {"checkpoint_id": "test-ckpt"},
                "status": "ready",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime_probe.json").write_text(
        json.dumps({"runtime_kind": "neoverse_production", "production_grade": True}),
        encoding="utf-8",
    )
    (tmp_path / "runtime_batch_manifest.json").write_text(
        json.dumps({"summary": {"numEpisodes": 2, "numSuccess": 1, "numFailure": 1, "successRate": 0.5}}),
        encoding="utf-8",
    )
    (tmp_path / "export_manifest.json").write_text(
        json.dumps({"artifact_uris": {"export_manifest": str(tmp_path / "export_manifest.json")}}),
        encoding="utf-8",
    )
    path = build_report(ValidationConfig(), tmp_path, fmt="markdown", output_path=tmp_path / "report.md")
    assert path.exists()
    assert (tmp_path / "standardized_eval_report.json").exists()
    assert "NeoVerse Session Report" in path.read_text(encoding="utf-8")
