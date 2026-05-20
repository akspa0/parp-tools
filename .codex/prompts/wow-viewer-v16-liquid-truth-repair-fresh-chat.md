# wow-viewer V16 Liquid Truth Repair (Fresh Chat)

## Goal

Fix the remaining V16 liquid-supervision truth failures for:

- `0_7_0_3694`
- `3_0_1_8303`

without disturbing the already-good cases:

- `0_5_3_3368`
- `0_5_5_3494`
- `3_3_5_12340`
- `4_0_0_11927`

This is a dataset-truth and validation lane, not a training lane.

## Mandatory Context

1. Read `AGENTS.md` at `I:/parp/parp-tools/AGENTS.md`.
2. Read `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md`.
3. Read:
   - `wow-viewer/specs/002-v16-liquid-supervision-truth-repair/spec.md`
   - `wow-viewer/specs/002-v16-liquid-supervision-truth-repair/plan.md`
   - `wow-viewer/specs/002-v16-liquid-supervision-truth-repair/tasks.md`
4. Read current holdout evidence:
   - `wow-viewer/output/datasets/v16/inspection/0_7_0_3694.validation_audit_overview.png`
   - `wow-viewer/output/datasets/v16/inspection/3_0_1_8303.validation_audit_overview.png`
   - `wow-viewer/output/datasets/v16/inspection/3_3_5_12340.validation_audit_overview.png`
   - `wow-viewer/output/datasets/v16/harvest_signal_inspection/3_3_5_12340/3_3_5_12340.overview.png`
   - `wow-viewer/output/datasets/v16/harvest_signal_inspection/3_0_1_8303/3_0_1_8303.samples.json`

## Current Proven State

- `0_5_3_3368` and `0_5_5_3494` now repair to dominant `mcnk`
- `3_3_5_12340` and `4_0_0_11927` now repair to explicit `mh2o`
- `0_7_0_3694` remains `unified`-only
- `3_0_1_8303` remains `unified`-only
- raw harvest sampling for `3_0_1_8303 --kinds mh2o mcnk_liquid` wrote `0 samples`
- raw harvest sampling for `3_3_5_12340 --kinds mh2o mcnk_liquid` wrote `8 samples`

## Required Workflow

1. Make the raw sample path fail loud when requested kinds produce zero samples.
2. Add deterministic one-tile trace mode for a known-bad wet tile.
3. Prove where the explicit source disappears for `3_0_1_8303` and `0_7_0_3694`.
4. Fix only that seam.
5. Rerun `patch-liquids` only for the affected build(s).
6. Regenerate human validation images.

## Commands Already Proven Useful

```powershell
cd I:\parp\parp-tools\wow-viewer\data-harvester
$env:UV_CACHE_DIR='i:\parp\parp-tools\.uv-cache'

uv run python scripts/inspect_v16_dataset.py `
  --builds 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --sample-count 16 `
  --sample-seed 42 `
  --sample-mode liquid_focus `
  --write-overview `
  --overview-columns 2

uv run python scripts/inspect_v16_harvest_samples.py `
  --build 3_0_1_8303 `
  --maps Azeroth Kalimdor Northrend Expansion01 `
  --kinds mh2o mcnk_liquid `
  --sample-count 8 `
  --sample-seed 1234

uv run python scripts/inspect_v16_harvest_samples.py `
  --build 3_3_5_12340 `
  --maps Azeroth Kalimdor Northrend Expansion01 `
  --kinds mh2o mcnk_liquid `
  --sample-count 8 `
  --sample-seed 1234
```

## Deliverables

1. Code fix for the holdout liquid-truth seam.
2. Repaired `signal_validation.json` / `liquid_patch_report.json` for the affected builds.
3. Fresh before/after human validation PNGs.
4. Short final report that states plainly whether `0_7_0_3694` and `3_0_1_8303` are fixed, still broken, or proven source-limited.
