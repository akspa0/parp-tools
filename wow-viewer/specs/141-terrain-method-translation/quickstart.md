# Quickstart: Terrain Method Translation

All commands are PowerShell-ready and run from `wow-viewer/data-harvester`. They are dry-run or inspection commands until a later task explicitly adds a user confirmation gate.

## Inspect the method ledger

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run --no-cache python scripts/v60_audit_terrain_methods.py --dry-run
```

Expected output includes six initial method records, their modality classifications, source URLs, and translation statuses. DSM2DTM, ResDepth, SMRF, and CSF must remain offline/reference or diagnostic entries; they are not RGB-only candidates by analogy.

Observed proof on 2026-08-10:

- Six methods classified: four `reference`, two `diagnostic`.
- Four contracts classified: `rgb_only`, `height_prior`, `point_cloud`, and `combined`.
- The RGB+`height_257` sample was rejected with a forbidden-read report.
- Focused Spec 141 tests: `17 passed`.
- Full `tests/v60` regression: `106 passed`.
- `ruff` and `py_compile`: passed.

To write the same report instead of printing a dry-run artifact:

```powershell
uv run --no-cache python scripts/v60_audit_terrain_methods.py --write --output "../output/datasets/v60/v7-clean-signal-method-audit-v1/method_translation_report.json"
```

## Validate the RGB-only benchmark plan

```powershell
uv run --no-cache python scripts/v60_build_rgb_method_benchmark.py --dry-run --source authored
```

The plan must include no-mask, predicted-mask, and withheld-mask conditions, preserve map-held-out identity, and list identity and tile-mean baselines. It must not read `height_257`, `terrain_shadow_256`, `shadow_mask`, WDL, or source-side object masks as inference inputs.

## User-run gate, after implementation and dry proof

The eventual training command will be written here only after the benchmark builder and forbidden-read tests pass. Codex does not launch that training.

## Optional future height-prior diagnostic

The eventual DSM/point-cloud command must require an explicit configured source and must write `runtime_claim=offline_diagnostic`. No command may present that result as RGB-minimap deployment evidence.

## Evidence handoff

Every completed slice updates:

- `specs/141-terrain-method-translation/` design artifacts;
- `memory-bank/activeContext.md`;
- `memory-bank/progress.md`;
- `memory-bank/workstream-terrain-ml.md`.

The handoff records the exact method status, proof gate, open failure, and next bounded action.
