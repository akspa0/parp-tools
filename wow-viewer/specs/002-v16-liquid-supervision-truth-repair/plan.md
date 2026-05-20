# Implementation Plan: V16 Liquid Supervision Truth Repair

**Branch**: `002-v16-liquid-supervision-truth-repair` | **Date**: 2026-05-20 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `wow-viewer/specs/002-v16-liquid-supervision-truth-repair/spec.md`

## Summary

Fix the remaining V16 liquid-truth failures by working in this order:

1. make validation surfaces fail loud
2. prove raw-harvest truth on known bad builds
3. repair only the specific era/source seams that are still collapsing to `unified`
4. rerun in-place store repair and regenerate human validation images

This is a dataset-truth lane, not a model-training lane.

## Technical Context

**Language/Version**: Python 3.11 for repair/inspection tooling; C# (.NET 10) for harvest-stream and tensor-pack extraction

**Primary Dependencies**: `uv`, `numpy`, `zarr`, `pyarrow`, `Pillow`; `WowViewer.Tool.Harvest`; `WowViewer.Core.IO.Maps`

**Storage**: Zarr v3 stores, Parquet indexes, JSON reports, PNG overviews, NPZ raw samples

**Testing**: `uv run python -m py_compile ...`, raw harvest sample runs, finalized-store overview generation, focused `dotnet build` for harvest tool

**Target Platform**: Windows, staged clients under `I:\parp\parp-tools\output\tmp\wowarchive-clients`

**Constraints**:

- avoid full re-harvest unless the raw streamed source truly lacks the needed truth
- keep the repair path patch-in-place where possible
- preserve already-good builds while fixing the holdouts

## Constitution Check

- Repo-local, bounded, evidence-first: pass
- No rewrite of client readers from scratch: pass
- Human-verifiable outputs required before training: pass
- One concern per slice: pass

## Project Structure

### Documentation

```text
wow-viewer/specs/002-v16-liquid-supervision-truth-repair/
  spec.md
  plan.md
  tasks.md
```

### Source Code

```text
wow-viewer/
  data-harvester/scripts/
    build_v16_dataset.py
    inspect_v16_dataset.py
    inspect_v16_harvest_samples.py
  tools/harvest/WowViewer.Tool.Harvest/
    Program.cs
  src/core/WowViewer.Core.IO/Maps/
    AdtTensorPackBuilder.cs
    AlphaWdtReader.cs
    NpzTileSerializer.cs
```

## Current Known State

From current repaired stores:

- `0_5_3_3368`: `mcnk`-backed after repair
- `0_5_5_3494`: `mcnk`-backed after repair
- `3_3_5_12340`: `mh2o`-backed after repair
- `4_0_0_11927`: `mh2o`-backed after repair
- `0_7_0_3694`: still `unified`-only
- `3_0_1_8303`: still `unified`-only

Additional evidence:

- raw harvest sample run for `3_0_1_8303` with `--kinds mh2o mcnk_liquid` wrote `0 samples`
- raw harvest sample run for `3_3_5_12340` with the same kinds wrote `8 samples`

## Implementation Phases

### Phase 1: Fail-Loud Truth Surfaces

**Goal**: Stop silent success on missing raw liquid categories.

**Approach**:

- update `inspect_v16_harvest_samples.py` so requested kinds that produce zero samples are treated as explicit missing truth
- write a machine-readable summary that states which kinds succeeded and which failed
- keep the existing overview PNG path for successful categories

**Exit Criteria**:

- `3_0_1_8303` raw sample run no longer ends with a misleading normal-success outcome when no `mh2o` / `mcnk_liquid` samples were found

### Phase 2: One-Tile Trace for Holdouts

**Goal**: Make `0_7_0_3694` and `3_0_1_8303` debuggable tile-by-tile.

**Approach**:

- add a deterministic tile-trace mode to raw sampling or a companion script
- target one or more known-wet tiles from the current validation overviews
- dump raw arrays, metadata, derived source flags, and visual panels for just that tile

**Exit Criteria**:

- one exact `3_0_1_8303` tile and one exact `0_7_0_3694` tile can be traced without broad random sampling

### Phase 3: Repair the Holdout Harvest Seam

**Goal**: Recover explicit source truth for `0_7_0_3694` and `3_0_1_8303` if it exists.

**Approach**:

- audit the streamed NPZ contents for holdout tiles first
- then audit the extraction seam in `AdtTensorPackBuilder.cs`, `Program.cs`, `AlphaWdtReader.cs`, or the source-specific readers only where the evidence points
- do not widen the fix to already-good builds

**Exit Criteria**:

- holdout raw tiles either produce explicit source arrays or emit precise diagnostics proving the source is absent upstream

### Phase 4: Repatch Stores and Regenerate Human Validation

**Goal**: Turn a proven raw-harvest fix into repaired final stores and fresh images.

**Approach**:

- rerun `patch-liquids` for the affected builds only
- rerun finalized-store overviews
- compare raw-sample overviews against final-store overviews

**Exit Criteria**:

- `0_7_0_3694` / `3_0_1_8303` are no longer ambiguous in both JSON reports and visual validation images

## Complexity Tracking

No constitution violations. The main risk is over-fixing orientation/provenance logic globally when the remaining issue may be isolated to one build seam.

## Quick Start Validation

```powershell
# 1. Rebuild the harvest tool when C# extraction logic changes
dotnet build I:\parp\parp-tools\wow-viewer\tools\harvest\WowViewer.Tool.Harvest\WowViewer.Tool.Harvest.csproj -c Debug

# 2. Raw holdout sample inspection
cd I:\parp\parp-tools\wow-viewer\data-harvester
$env:UV_CACHE_DIR='i:\parp\parp-tools\.uv-cache'

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

# 3. Final-store visual validation
uv run python scripts/inspect_v16_dataset.py `
  --builds 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --sample-count 16 `
  --sample-seed 42 `
  --sample-mode liquid_focus `
  --write-overview `
  --overview-columns 2

# 4. In-place liquid repair after the seam is fixed
uv run python scripts/build_v16_dataset.py patch-liquids `
  --builds 0_7_0_3694 3_0_1_8303 `
  --map-workers 4 `
  --batch-size 128
```
