"""Direct tests for Stage-1 post audit behavior."""

from __future__ import annotations


def test_post_s1_audit_fails_cleanly_on_invalid_manifest(sample_config, tmp_path):
    from blueprint_validation.common import write_json
    from blueprint_validation.stages.s1_render import RenderStage

    work_dir = tmp_path
    render_dir = work_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    write_json({"facility": "Test", "clips": "not_a_list"}, render_dir / "render_manifest.json")

    stage = RenderStage()
    facility = list(sample_config.facilities.values())[0]
    summary = stage.run_post_s1_audit(
        config=sample_config,
        facility=facility,
        work_dir=work_dir,
        vlm_rescore_first=0,
    )

    assert summary["status"] == "failed"
    assert "Invalid render manifest for post-S1 audit" in str(summary.get("detail", ""))
    assert str(summary.get("summary_path", "")).endswith("post_s1_audit_summary.json")
