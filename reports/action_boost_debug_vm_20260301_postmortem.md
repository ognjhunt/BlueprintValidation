# Deep Postmortem: `action_boost_debug_vm_20260301` (kitchen_0787-focused)

## Executive Summary
This debug sequence validated pipeline plumbing and resume behavior, but it did **not** validate adapted-policy gains.

Final authoritative state (`pipeline_summary.json`) is `overall_status: success` with `6 success / 13 skipped / 0 failed` stages, driven by WM-only mode and reuse of prior results.

Bottom-line outcome metrics from final Stage 4 result:
- `baseline_mean_task_score = 0.487`
- `adapted_mean_task_score = 0.475`
- `absolute_point_differential = -0.012`
- `improvement_pct = -2.46`
- `win_rate = 0.013`
- `p_value = 0.783465`
- `manipulation_success_delta_pp = 0.0`
- `reliability_gate.passed = false` with `replay_pass_rate = 0.01875`

Interpretation: no statistically meaningful uplift and small negative point delta for adapted vs baseline in this run lineage.

## Chronology
Timestamp note: log timestamps align with UTC (matching JSON `+00:00` stage timestamps). ET shown as `America/New_York`.

### Attempt-by-attempt timeline

| Attempt | UTC Window | ET Window | Wallclock | Stage Counts (S/F/K) | Failed Stages |
|---|---|---|---:|---:|---|
| `run_all.log` | 2026-03-01 21:08:42.845 -> 22:18:45.654 | 2026-03-01 16:08:42.845 -> 17:18:45.654 | 4202.809s | 4 / 8 / 6 | `s3_finetune`, `s4_policy_eval`, `s4a`, `s3b`, `s3c`, `s4e`, `s4b`, `s4c` |
| `run_all_stage4_resume.log` | 2026-03-01 22:27:08.730 -> 22:28:25.734 | 2026-03-01 17:27:08.730 -> 17:28:25.734 | 77.004s | 4 / 7 / 7 | `s4_policy_eval`, `s4a`, `s3b`, `s3c`, `s4e`, `s4b`, `s4c` |
| `run_wm_only_refresh_resume.log` | 2026-03-01 23:12:07.039 -> 2026-03-02 00:03:36.855 | 2026-03-01 18:12:07.039 -> 19:03:36.855 | 3089.816s | 5 / 1 / 13 | `s3d_wm_refresh_loop` |
| `run_wm_only_refresh_resume_2.log` | 2026-03-02 00:05:48.148 -> 00:05:53.613 | 2026-03-01 19:05:48.148 -> 19:05:53.613 | 5.465s | no summary block | incomplete/aborted start of `s3d` |
| `run_wm_only_refresh_fast.log` | 2026-03-02 00:27:19.336 -> 00:39:13.433 | 2026-03-01 19:27:19.336 -> 19:39:13.433 | 714.097s | 6 / 0 / 13 | none |

### Stage status vectors (canonical order)
Order:
`s0 s1 s1b s1c s1d s1e s2 s3 s4 s3d s4a s3b s3c s4e s4b s4c s4d s5 s6`

Legend: `S=success`, `F=failed`, `K=skipped`, `-=not present in that attempt summary`

- `run_all`: `K S K K K K S F F - F F F F F F K S S`
- `run_all_stage4_resume`: `K S K K K K S K F - F F F F F F K S S`
- `run_wm_only_refresh_resume`: `K S K K K K S K S F K K K K K K K S S`
- `run_wm_only_refresh_resume_2`: `- - - - - - - - - - - - - - - - - - -`
- `run_wm_only_refresh_fast`: `K S K K K K S K S S K K K K K K K S S`

## Quant Metrics

### 1) Final pipeline state (`pipeline_summary.json`)
- `overall_status: success`
- `aborted_early: false`
- `fail_fast: false`
- `resume_from_results: true`
- `failed_stage_keys: []`
- Stage distribution: `6 success / 0 failed / 13 skipped`
- Reported total elapsed (sum of stage elapsed fields): `4324.375s` (72m 4.4s)

Successful stages:
- `s1_render`, `s2_enrich`, `s4_policy_eval`, `s3d_wm_refresh_loop`, `s5_visual_fidelity`, `s6_spatial_accuracy`

Skipped stages include policy/OpenVLA branches in WM-only mode plus `s3_finetune` debug override.

### 2) Per-attempt severity counts

| Attempt | ERROR | WARNING | CRITICAL | `FAILED/failed` keyword |
|---|---:|---:|---:|---:|
| `run_all.log` | 11 | 11 | 0 | 23 |
| `run_all_stage4_resume.log` | 9 | 10 | 10 | 20 |
| `run_wm_only_refresh_resume.log` | 2 | 9 | 13 | 10 |
| `run_wm_only_refresh_resume_2.log` | 0 | 0 | 0 | 0 |
| `run_wm_only_refresh_fast.log` | 0 | 0 | 0 | 0 |

### 3) Policy evaluation (`s4_policy_eval_result.json`, `policy_eval_report.json`, `vlm_scores.json`, `judge_audit.csv`)

Core metrics:
- `headline_scope: wm_only`
- `rollout_driver: scripted`
- `num_rollouts_baseline = 80`
- `num_rollouts_adapted = 80`
- `num_unique_task_templates = 16`
- `baseline_mean_task_score = 0.487`
- `adapted_mean_task_score = 0.475`
- `absolute_point_differential = -0.012`
- `improvement_pct = -2.46`
- `win_rate = 0.013`
- `p_value = 0.783465`
- `manipulation_success_rate` baseline/adapted: `0.0 / 0.0`

Reliability gate:
- `replay_pass_rate = 0.01875`
- `controllability_pass_rate = 1.0`
- `passed = false`

Cross-check with `vlm_scores.json`:
- Recomputed means: baseline `0.4875`, adapted `0.475`
- Reported baseline `0.487` equals recomputed value at 3-decimal precision (rounded/truncated representation difference of 0.0005).

`task_score` distribution by condition (from `vlm_scores.json`):
- baseline: `{0.0:45, 1.0:34, 5.0:1}`
- adapted: `{0.0:45, 1.0:33, 2.0:1, 3.0:1}`

### 4) WM refresh metrics (`s3d_*`, `wm_refresh_iteration_summaries.json`, `refresh_manifest.json`, `finetune_log.json`)
- `iterations_requested = 1`
- `iterations_completed = 1`
- `source_condition = adapted`
- `num_source_rollouts = 80`
- `num_success_rollouts = 0`
- `num_near_miss_rollouts = 0`
- `num_hard_negative_rollouts = 80`
- Iteration 0 train return code: `0`
- Iteration 0 train elapsed: `708.6016s`

Mix composition (`mix_metrics`):
- `available_total = 81`
- `selected_total = 81`
- `selected_stage2 = 1`
- `selected_success = 0`
- `selected_near_miss = 0`
- `selected_hard_negative = 80`
- `target_stage2 = 49`, `target_success = 20`, `target_near_miss = 12`, `target_hard_negative = 0`

### 5) Visual + spatial metrics
Visual fidelity (`visual_fidelity.json`):
- `num_comparisons = 1`
- `overall_mean_psnr = 7.9203`
- `overall_mean_ssim = 0.361`
- `overall_mean_lpips = 0.8327`

Spatial accuracy (`spatial_accuracy.json`):
- `mean_spatial_score = 1.0`
- `mean_visual_score = 1.0`
- `mean_landmark_score = 1.0`
- `num_evaluated = 1`

### 6) Artifact inventory (`kitchen_0787`)
Top-level subdirectory inventory:

| Subdir | Files | Bytes |
|---|---:|---:|
| `wm_refresh_loop` | 183 | 13,390,041,004 |
| `renders` | 64 | 17,550,494 |
| `policy_eval` | 165 | 8,644,813 |
| `wm_refresh_loop_prev_fast_20260302_002719` | 173 | 4,567,429 |
| `finetune` | 12 | 2,649,809 |
| `enriched` | 15 | 2,132,771 |
| `spatial_accuracy` | 1 | 707 |
| `visual_fidelity` | 1 | 340 |
| `policy_finetune` | 0 | 0 |
| `policy_rl_loop` | 0 | 0 |

Attached vs unattached classification (rule from plan):
- Attached files: `460` files, `13,421,032,738` bytes
- Unattached/carryover files: `174` files, `4,577,673` bytes
- Unattached share by files: `27.44%`
- Unattached share by bytes: `0.0341%`

### 7) Integrity metrics (`file_index.tsv` vs `sha256_manifest.txt`)
- `file_index.tsv` tracked entries: `637`
- `sha256_manifest.txt` tracked entries: `637`
- Set diff: `0` both directions (exact match of tracked paths)

Actual files on disk under run directory: `644`
- Extra on-disk files not in tracked index set (`7`):
  - `file_index.tsv` and `sha256_manifest.txt` themselves
  - `.DS_Store` files (`5` occurrences)

This is consistent with expected indexing behavior and OS metadata noise.

## Root Cause Chain

### Primary failures by attempt
1. `run_all.log`
- `s3_finetune` timeout after 3600s.
- `s4_policy_eval` fails on TIMM version incompatibility.
- `s3c_policy_rl_loop` fails on same TIMM incompatibility.

2. `run_all_stage4_resume.log`
- `s4_policy_eval` fails on action-space mismatch (`policy action_dim=7`, `world_model action_dim=384`).
- `s3c_policy_rl_loop` fails with CUDA OOM.

3. `run_wm_only_refresh_resume.log`
- `s4_policy_eval` succeeds.
- `s3d_wm_refresh_loop` fails with dataset decode mismatch (`Decoded 3 frames, expected 13` on `adapted_clip_007_manipulation_001.mp4`).

4. `run_wm_only_refresh_fast.log`
- `s3d_wm_refresh_loop` succeeds (`max_iter=100` run), final summary becomes successful.

### Cascading failures from missing prerequisites
In `run_all` and `run_all_stage4_resume`, downstream stages fail due upstream failure dependencies:
- `s4a_rlds_export` (needs successful `s4`)
- `s3b_policy_finetune` (requires rollout dataset from `s4a` or configured data root)
- `s4e_trained_eval` (requires trained checkpoint from `s3b/s3c`)
- `s4b_rollout_dataset` (needs policy eval scores)
- `s4c_policy_pair_train` (needs `s4b` dataset summary)

Observed cascade failure events: `10` total across those two attempts.

## Unattached / Carryover Artifacts

### Classified as unattached/carryover (inside `kitchen_0787`)
- `wm_refresh_loop_prev_fast_20260302_002719` (173 files, 4,567,429 bytes)
- Root `kitchen_0787/.DS_Store` (10,244 bytes)
- Empty dirs: `policy_finetune`, `policy_rl_loop`

### Provenance of `wm_refresh_loop_prev_fast_20260302_002719`
Evidence that this is a prior failed/interrupted refresh snapshot:
- Contains `iter_00/finetune_log.json` with `status: failed`, `returncode: 1`, and decode-mismatch traceback.
- Contains partial training logs (`console.log`, `stdout.log`, `debug.log`) but no massive finalized checkpoint artifact.
- `refresh_manifest.json` is byte-identical to current loop manifest (`sha256` equal).

Current vs carryover delta:
- Current `wm_refresh_loop`: 183 files, 13,390,041,004 bytes
- Carryover `wm_refresh_loop_prev_fast_*`: 173 files, 4,567,429 bytes
- Delta: `+10 files`, `+13,385,473,575 bytes`
- Dominant delta source: final checkpoint blob `__0_0.distcp` (~13.19 GB) in current loop.

## Reliability and Data-Quality Risks
1. Reliability gate failed despite stage success.
- `replay_pass_rate` is extremely low (`0.01875`) and gate status is `passed=false`.

2. No empirical gain signal in this run lineage.
- Adapted mean is lower by `0.012` points; `p=0.783465` indicates no significant difference.

3. Refresh mix is highly skewed.
- Selected refresh dataset is `80/81` hard negatives, with `0` success and `0` near-miss source rollouts.

4. Spatial metric consistency issue.
- `spatial_accuracy.json` scalar scores are all `1.0`, while reasoning text says the images are too distorted to evaluate landmarks/layout. This mismatch weakens trust in scalar quality metrics.

5. Resume provenance mixes timestamps across attempts.
- Final `pipeline_summary.json` includes stage timestamps from different runs (`s5/s6` earlier than final `s3d` timestamp), so chronology must be read as a merged resume state, not a single fresh pass.

## Validation Checks (Requested)
1. Policy means cross-check vs `vlm_scores.json`: **pass with reported precision nuance**.
- Baseline recompute `0.4875` vs reported `0.487`.
- Adapted recompute `0.475` vs reported `0.475`.

2. Failed-stage lists in log tails vs parsed status tables: **pass** for all attempts that contain a summary block.

3. `sha256_manifest.txt` vs `file_index.tsv` tracked sets: **pass** (exact path-set match).

4. Unattached classification determinism + inclusion of `wm_refresh_loop_prev_fast_*`: **pass**.

5. Timeline monotonicity per attempt + cross-attempt timestamp reuse callout: **pass**.

## Conclusions
- This sequence demonstrates that the pipeline, resume logic, and WM-refresh checkpoint production can complete under a debug/WM-only path.
- It does **not** demonstrate policy improvement.
- Your partial conclusion (“plumbing validated, gains not shown”) is directionally correct.
- Strongest caveat: final success is a resumed composite state with skipped branches and mixed-provenance stage timestamps, not a single fully fresh end-to-end run.

## Key Source Files
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/pipeline_summary.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/run_all.log`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/run_all_stage4_resume.log`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/run_wm_only_refresh_resume.log`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/run_wm_only_refresh_resume_2.log`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/run_wm_only_refresh_fast.log`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/file_index.tsv`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/sha256_manifest.txt`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/s4_policy_eval_result.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/policy_eval/policy_eval_report.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/policy_eval/vlm_scores.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/policy_eval/judge_audit.csv`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/s3d_wm_refresh_loop_result.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/wm_refresh_loop/wm_refresh_iteration_summaries.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/wm_refresh_loop/iter_00/refresh_manifest.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/wm_refresh_loop/iter_00/finetune_log.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/wm_refresh_loop_prev_fast_20260302_002719/iter_00/finetune_log.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/visual_fidelity/visual_fidelity.json`
- `/Users/nijelhunt_1/BlueprintValidationBackups/vast/32211174/latest/work_dir/action_boost_debug_vm_20260301/kitchen_0787/spatial_accuracy/spatial_accuracy.json`
