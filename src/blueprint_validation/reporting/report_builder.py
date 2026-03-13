"""Build minimal session/export reports for the NeoVerse-only path."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..common import read_json, write_json, write_text_atomic
from ..config import ValidationConfig
from ..public_contract import build_standardized_eval_report


def _load_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return dict(payload) if isinstance(payload, dict) else None


def _collect_results(config: ValidationConfig, work_dir: Path) -> Dict[str, Any]:
    session = _load_optional_json(work_dir / "session_state.json")
    runtime_smoke = _load_optional_json(work_dir / "runtime_smoke.json")
    batch = _load_optional_json(work_dir / "runtime_batch_manifest.json")
    export_data = _load_optional_json(work_dir / "export_manifest.json")
    return {
        "project_name": config.project_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session": session or {},
        "runtime_smoke": runtime_smoke or {},
        "batch": batch or {},
        "export": export_data or {},
    }


def _render_markdown(report_data: Dict[str, Any]) -> str:
    session = report_data.get("session", {})
    batch = report_data.get("batch", {})
    export_data = report_data.get("export", {})
    lines = [
        "# NeoVerse Session Report",
        "",
        f"- Generated at: {report_data.get('generated_at', 'N/A')}",
        f"- Session ID: {session.get('session_id', 'N/A')}",
        f"- Site world ID: {session.get('site_world_id', 'N/A')}",
        f"- Runtime: {session.get('runtime_backend_public_name', 'N/A')}",
        f"- Session status: {session.get('status', 'N/A')}",
        "",
        "## Batch Summary",
        "",
    ]
    summary = batch.get("summary", {}) if isinstance(batch.get("summary"), dict) else {}
    if summary:
        lines.extend(
            [
                f"- Episodes: {summary.get('numEpisodes', 0)}",
                f"- Successes: {summary.get('numSuccess', 0)}",
                f"- Failures: {summary.get('numFailure', 0)}",
                f"- Success rate: {summary.get('successRate', 'N/A')}",
            ]
        )
    else:
        lines.append("- No batch manifest found.")
    lines.extend(["", "## Export Artifacts", ""])
    artifacts = export_data.get("artifact_uris", {}) if isinstance(export_data.get("artifact_uris"), dict) else {}
    if artifacts:
        for key, value in artifacts.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No export manifest found.")
    lines.append("")
    return "\n".join(lines)


def build_report(
    config: ValidationConfig,
    work_dir: Path,
    fmt: str = "markdown",
    output_path: Path = Path("validation_report.md"),
) -> Path:
    report_data = _collect_results(config, work_dir)
    standardized_report = build_standardized_eval_report(report_data)
    standardized_output_path = output_path.with_name("standardized_eval_report.json")
    write_json(standardized_report, standardized_output_path)

    if fmt == "json":
        output_path = output_path.with_suffix(".json")
        write_json(report_data, output_path)
    else:
        output_path = output_path.with_suffix(".md")
        write_text_atomic(output_path, _render_markdown(report_data))
    return output_path
