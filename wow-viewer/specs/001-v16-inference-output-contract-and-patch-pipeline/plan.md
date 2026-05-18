# Implementation Plan: V16 Inference Output Contract & Patch Pipeline

**Branch**: `001-v16-inference-output-contract-and-patch-pipeline` | **Date**: 2026-05-18 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `wow-viewer/specs/001-v16-inference-output-contract-and-patch-pipeline/spec.md`

## Summary

Close three gaps in the V16 inference→patch pipeline: (1) eliminate the per-tile summary staging step by reading `.pred.zarr` directly, (2) add liquid chunk patching to `terrain-patch-adt`, (3) provide a one-shot pipeline command and validation tooling. This completes the terrain-AI training→inference→ADT-patching loop.

## Technical Context

**Language/Version**: C# (.NET 10) for converter tooling, Python 3.11+ for inference/validation scripts

**Primary Dependencies**: `Zarr` (Python Zarr v3), `System.Text.Json`, `WowViewer.Core.IO.Maps` (ADT readers/writers), `WowViewer.Core.Runtime.World.Terrain` (terrain patching)

**Storage**: Zarr v3 LocalStore (prediction stores), Parquet (index files), JSON (reports)

**Testing**: `dotnet test` for C# converter commands, `uv run python` for Python validation scripts

**Target Platform**: Windows (primary), cross-platform .NET 10

**Project Type**: Library + CLI tooling

**Constraints**: Must not break existing `terrain-patch-adt` staging-based flow; must produce identical output for the same inputs

## Constitution Check

- Library-first: terrain patching logic lives in shared library, CLI is thin wrapper
- Test-first: each slice includes validation against real data or existing stores
- No new external dependencies beyond what already exists in the solution

## Project Structure

### Documentation (this feature)

```text
specs/001-v16-inference-output-contract-and-patch-pipeline/
  spec.md              # Feature specification
  plan.md              # This file
  tasks.md             # Task breakdown
```

### Source Code (repository root)

```text
wow-viewer/
  tools/converter/WowViewer.Tool.Converter/
    TerrainPatchAdtCommand.cs          # Modify: add --pred-zarr, liquid patching, patch report
    InferAndPatchCommand.cs            # New: one-shot pipeline command
    ValidateInferencePairCommand.cs    # New: input/output store alignment validator
    Program.cs                         # Modify: wire new commands
  data-harvester/scripts/
    infer_v16.py                       # Reference only (already produces .pred.zarr)
    validate_v16_inference_pair.py     # New: Python-side pair validation (optional mirror)
  tests/
    WowViewer.Core.Tests/              # Add: inference pair validation tests
```

## Implementation Phases

### Phase 1: Direct Zarr Consumption (US1)

**Goal**: `terrain-patch-adt` reads predictions from `.pred.zarr` without staging.

**Approach**: Add `--pred-zarr` option to `TerrainPatchAdtCommand`. When set, read `index.parquet` from the Zarr store, resolve tile coordinates, and stream arrays from Zarr instead of reading `.npy` files from staging directories.

**Risk**: Zarr v3 read performance under many small array slices. Mitigate by reading full first-dimension slices when possible.

### Phase 2: Liquid Chunk Patching (US2)

**Goal**: Patched ADTs carry predicted liquid data.

**Approach**: After height/normal patching, read `liquid_pred_mask_256` and `liquid_height` from the prediction source. For LK ADTs, write `MH2O` chunk into `_obj0.adt`. For Alpha ADTs, write `MCLQ` into the embedded tile. Use existing `Mcal.cs` / `McalAlpha` infrastructure for MCLQ encoding; implement minimal MH2O writer for the LK path.

**Risk**: MH2O format is complex (multiple sub-types). Start with type 1 (simple height-map liquid) which covers most cases.

### Phase 3: One-Shot Pipeline & Reports (US3, US4)

**Goal**: Single command for infer+patch+convert; patch reports for every run.

**Approach**: New `InferAndPatchCommand` that shells out to `infer_v16.py` (or calls the model directly), then invokes `TerrainPatchAdtCommand`, then optionally `convert-lk-to-alpha`. Always writes `patch_report.json`.

### Phase 4: Validation Tooling (US5)

**Goal**: Prove input/output stores are properly paired.

**Approach**: New `ValidateInferencePairCommand` (C#) and optional `validate_v16_inference_pair.py` (Python) that check row count, tile_id alignment, and order.

## Complexity Tracking

No constitution violations. All work fits within existing project structure.

## Quick Start Validation

```powershell
# 1. Build
dotnet build wow-viewer/WowViewer.slnx -c Debug

# 2. Direct Zarr patch (replaces staging-based flow)
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- \
  terrain-patch-adt \
  --pred-zarr "wow-viewer/output/datasets/v16_inference/<run>/<build>.pred.zarr" \
  --input-adt-dir "<staged-client-map-root>" \
  --output-dir "<patched-output>"

# 3. Validate inference pair
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- \
  validate-inference-pair \
  --input-zarr "wow-viewer/output/datasets/v16/<build>.zarr" \
  --output-zarr "wow-viewer/output/datasets/v16_inference/<run>/<build>.pred.zarr"

# 4. One-shot pipeline
dotnet run --project wow-viewer/tools/converter/WowViewer.Tool.Converter -- \
  infer-and-patch \
  --build 3_3_5_12340 \
  --checkpoint "wow-viewer/models/v16/runs/<run>/checkpoints/v16_best.pt" \
  --client-root "<staged-client>" \
  --map Azeroth \
  --output-dir "<output>"
```
