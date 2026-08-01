# Quickstart: Canonical Dataset Curation and Signal-Mismatch Bucketing

**Feature**: 122-dataset-curation | **Date**: 2026-07-30

All commands below are illustrative of the contract in [contracts/cli-contract.md](contracts/cli-contract.md);
exact flags may shift slightly during implementation (tasks.md tracks the real build). Every step
is dry-run-first; nothing writes until an explicit `--write`/`--confirm-run` flag is passed. The C#
`curate` command runs in seconds to low-single-digit minutes per map (Technical Context, plan.md)
and needs no GPU. Nothing in this feature requires the user to run training or any billed/heavy
operation — the heaviest step is a CPU pass over an already-harvested store.

## 1. Build (once, from the repo root of `wow-viewer`)

```powershell
dotnet build I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug
```

Confirms the new `WowViewer.Core.Curation` library and the `curate` subcommand compile cleanly
alongside everything else.

## 2. Dry-run curation against an existing v50 store

```powershell
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\harvest\WowViewer.Tool.Harvest -- curate `
  --clients-root H:\CLIENTS `
  --build 0_5_3_3368 `
  --store I:\parp\parp-tools\wow-viewer\output\datasets\v50\v50.1\curriculum-0_5_3_3368-dual_v1.zarr
```

Expect a printed plan: total tile count, which checks will run (difficulty/coverage/lighting
buckets, height-normal mismatch, non-finite check, has-flag check, synthetic-fidelity — with any
check missing its backing signal explicitly named as skipped, not silently absent), and the output
paths under `<store>/curation/<curation_run_id>/`. Nothing is written yet.

## 3. Write the curation manifest for real

```powershell
dotnet run --project I:\parp\parp-tools\wow-viewer\tools\harvest\WowViewer.Tool.Harvest -- curate `
  --clients-root H:\CLIENTS `
  --build 0_5_3_3368 `
  --store I:\parp\parp-tools\wow-viewer\output\datasets\v50\v50.1\curriculum-0_5_3_3368-dual_v1.zarr `
  --write
```

Verify (SC-006): the printed `tile_count` in the completion message matches the store's own row
count exactly. `curation_manifest.parquet`, `curation_findings.parquet`, and `curation_run.json`
now exist under `<store>/curation/<curation_run_id>/`, and `<store>/curation/latest` points at that
run.

## 4. Query a non-clean bucket (proves US2 — full access, not a filter)

```powershell
cd I:\parp\parp-tools\wow-viewer\data-harvester
uv run python -c "from harvester.curation_store import load_curation_manifest, load_curation_findings; import sys; m = load_curation_manifest(sys.argv[1]); f = load_curation_findings(sys.argv[1]); print('pathological tiles:', (m.coverage_bucket == 'blank').sum()); print('height-normal mismatches:', (f.category == 'height_normal_mismatch').sum())" `
  I:\parp\parp-tools\wow-viewer\output\datasets\v50\v50.1\curriculum-0_5_3_3368-dual_v1.zarr
```

Confirm both counts are non-error, non-empty-by-default results — querying the "bad" bucket takes
exactly the same code path as querying "clean" would.

## 5. Legacy comparison (SC-003 gate, one-time, before any script retires)

```powershell
cd I:\parp\parp-tools\wow-viewer\data-harvester
uv run python scripts/spec122_compare_legacy_mismatch.py `
  --store I:\parp\parp-tools\wow-viewer\output\datasets\v50\v50.1\curriculum-0_5_3_3368-dual_v1.zarr `
  --report I:\parp\parp-tools\wow-viewer\output\datasets\v50\v50.1\curation\spec122_legacy_comparison.json
```

Read the report: the new C# `height_normal_mismatch` finding set should match (or, with a stated
reason, improve on) what `mismatch_detector.py` flags on the same tiles today. This gate must pass
before `mismatch_detector.py`/`v16_curation.py`/`spec111/lighting_buckets.py` are converted to thin
readers (D-04).

## 6. Tests

```powershell
dotnet test I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug --filter WowViewer.Core.Curation.Tests
```

```powershell
cd I:\parp\parp-tools\wow-viewer\data-harvester
uv run python -m pytest tests/test_curation_store.py -q
```

## Estimated time/footprint

- C# build: under a minute incremental.
- `curate --write` over one map (hundreds of tiles): low-single-digit minutes, CPU-only, no
  additional disk beyond two Parquet files (expected low tens of MB per map) and a small JSON
  record.
- No GPU, no client-library download beyond what harvesting a store already required, no billed
  cloud step anywhere in this feature.
