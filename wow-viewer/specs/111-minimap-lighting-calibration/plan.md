# Implementation Plan: Minimap Lighting Calibration and Lighting-Aware Terrain Reconstruction

**Branch**: `111-minimap-lighting-calibration` | **Date**: 2026-07-17 | **Spec**: [spec.md](spec.md)

## Summary

Use the just-corrected, ground-truth-validated `TerrainSolarDirection`/`TerrainMinimapCompositor`
production path to determine, for every 0.5.3.3368 dataset tile that has both an authored minimap and
decoded ground-truth terrain, which time-of-day best explains the authored minimap's *shading
pattern* -- a geometric signal the existing tint-based `MinimapLightingProvenance` cannot see. Bucket
the whole 0.5.3.3368 corpus by that inference, use the real bucket distribution to reweight the
existing (and currently drifted) synthetic-lighting-variant generator so training lighting matches
reality, and only then retrain the existing image-to-terrain reconstruction model and compare it
against the current deployed checkpoint under an explicit, separately-authorized go/no-go gate.

## Technical Context

**Language/Version**: C# / .NET 10 (shading-match inference, reusing existing terrain/minimap
libraries); Python 3.11+ / uv (dataset iteration, rebalancing, training, evaluation)

**Primary Dependencies**: Existing `WowViewer.Core` / `WowViewer.Core.IO` terrain and minimap
libraries (`TerrainMinimapCompositor`, `TerrainSolarDirection`, `MinimapLightingProvenance`); existing
C#-to-Python length-prefixed streaming protocol; PyTorch, NumPy, Zarr, PyArrow (existing
`data-harvester` stack)

**Storage**: Existing per-build Zarr stores (additive fields only); Parquet index/report derived from
those fields. No new on-disk artifact format; no NPZ (constitution principle V)

**Testing**: C# focused xUnit tests (`WowViewer.Core.Tests`) for the shading-match scorer; Python
`pytest` via `uv run python -m pytest` for bucketing/rebalancing/training-config changes

**Target Platform**: CPU for shading-match inference and bucketing (bulk dataset iteration); local
CUDA or user-authorized cloud GPU for the Phase 3 training run only

**Project Type**: Existing data-harvester library + CLI tools, extending an existing C# terrain/IO
library

**Performance Goals**: Bulk-iterable over the full 0.5.3.3368 corpus in a single dataset pass (same
order of magnitude as the existing `synthetic-minimap` whole-map export); no new per-frame/runtime
performance constraint

**Constraints**: Shading-match inference MUST reuse the existing production lighting code path, not a
second reimplementation (see research.md); no ground-truth lighting/time may reach the deployed
model's input; no DepthAnything-family/multi-head/shared-weight architecture; no GPU training run or
cloud pod launches without a separate, explicit user go-ahead

**Scale/Scope**: One new C# shading-match component, one dataset-wide bucketing pass over the
0.5.3.3368 corpus, one rebalancing adjustment to an existing Python training-data generator, one
retrain-and-evaluate pass against one existing model lineage

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo Independence**: pass -- every new file stays under `wow-viewer/`; no external repo paths.
- **Library-First**: pass -- the shading-match scorer is added to `WowViewer.Core`/`WowViewer.Core.IO`
  next to `MinimapLightingProvenance`, not as tool-only or script-only logic; the Harvest CLI stays a
  thin wrapper.
- **Real-Data Validation**: pass -- bucketing runs against the real, configured 0.5.3.3368 client
  corpus; results record build identity/fingerprint per the existing `MinimapLightingProvenance`
  discipline.
- **Residual Model Chain**: pass -- Phase 3 retrains the existing single reconstruction
  stage/checkpoint through the **v50 lane's canonical entry** (`scripts/v50_train_wdl_prior.py`,
  which wraps the spec103-named implementation and enforces `require_store_release`; Spec 102's
  chain stays BLOCKED and is not a target); it does not add a lighting-conditioned head, does not
  share weights across stages, and does not become multi-task.
- **Streaming-First Dataset Pipeline**: pass -- new shading-match results are written as additive
  Zarr/Parquet fields via the existing C#-to-Python streaming protocol; no intermediate NPZ.
- **No Game Client Path Assumptions**: pass -- the calibration pass reads from the existing configured
  client root; no path is hardcoded.
- **Format Reader/Writer Ownership**: pass -- reuses `TerrainMinimapCompositor` rather than writing a
  second minimap-rendering path; explicitly retires a duplicated lighting-direction reimplementation
  found in `data-harvester/src/harvester/spec103/terrain_lighting.py` rather than adding a third one.
- **One Phase at a Time / Bite-Sized Plans**: pass -- phases below are gated (Phase 2 depends on Phase
  1's real output; Phase 3 depends on Phase 2's rebalanced data and is separately execution-gated).

No violations requiring Complexity Tracking justification.

## Project Structure

### Documentation (this feature)

```text
specs/111-minimap-lighting-calibration/
├── plan.md                          # This file
├── research.md                      # Phase 0 output
├── data-model.md                    # Phase 1 output
├── quickstart.md                    # Phase 1 output
├── contracts/
│   └── minimap-lighting-calibration-contract.md
└── tasks.md                         # Phase 2 output (speckit-tasks, not this command)
```

### Source Code (repository root: `wow-viewer/`)

```text
src/core/WowViewer.Core/Maps/
└── MinimapLightingProvenance.cs          # extended: shading-match fields (additive)

src/core/WowViewer.Core.IO/Maps/
├── TerrainMinimapCompositor.cs           # unchanged; reused as-is for candidate rendering
└── MinimapShadingMatch.cs                # new: candidate sweep + directional-structure scoring
                                           # (lives in Core.IO, not Core: Core.IO -> Core is the only
                                           # allowed reference direction, and this type must call
                                           # TerrainMinimapCompositor.Compose)

tests/WowViewer.Core.Tests/
├── MinimapLightingProvenanceTests.cs     # extended coverage for new fields
└── MinimapShadingMatchTests.cs           # new: scorer unit coverage (synthetic fixtures)

tools/harvest/WowViewer.Tool.Harvest/
└── Program.cs                            # extended: AnalyzeAuthoredMinimapLighting chains
                                           # MinimapShadingMatch.Evaluate onto the existing
                                           # tint-based Infer() call for Full/V22 exports (reuses
                                           # the existing tile-iteration/streaming pathway rather
                                           # than a new parallel command; the build-fingerprint
                                           # gate makes this a no-op for non-0.5.3.3368 tiles)

data-harvester/src/harvester/
├── spec103/terrain_lighting.py           # drifted direction reimplementation retired/re-scoped;
│                                          # non-direction responsibilities (color/fog/MCSH bake)
│                                          # kept but re-labeled per research.md
├── spec111/                              # new: bucket ingestion, distribution report,
│   ├── lighting_buckets.py               #      rebalancing-weight computation
│   └── rebalance_lighting_variants.py
└── scripts/
    ├── report_lighting_buckets.py        # new: read Zarr fields -> distribution report
    └── train_spec111_reconstruction.py   # new: retrain existing model on rebalanced data
                                           #      (User-run only, per Phase 3 gate)

data-harvester/tests/spec111/
└── test_lighting_bucket_rebalancing.py   # new: rebalancing-weight and leak-safety coverage
```

**Structure Decision**: Extend the existing terrain/minimap C# library and the existing
`data-harvester` Python package rather than introducing a new project or service. The only new
top-level grouping is `data-harvester/src/harvester/spec111/`, mirroring the existing `spec103/`
convention for feature-scoped Python modules.

## Phases

1. **User Story 1 -- shading-match inference and bucketing (Implementation).** Add
   `MinimapShadingMatch` in C#, rendering candidates through the existing
   `TerrainMinimapCompositor`/`TerrainSolarDirection` path and correlating luma-value patterns
   independent of tint (research.md -- gradient-direction cosine similarity was tried first and
   found unable to discriminate hours at all once azimuth is fixed; value correlation is both
   tint-invariant for a single material and genuinely elevation-discriminative). Extend
   `MinimapLightingProvenance` with the new fields. Chain `MinimapShadingMatch.Evaluate` onto the
   existing `AnalyzeAuthoredMinimapLighting` tint-based `Infer()` call in
   `WowViewer.Tool.Harvest/Program.cs`, reusing the harvester's existing full-texture-decode
   tile-iteration and C#->Python streaming pathway (a transport detail, not a dataset-lane choice
   -- the destination lane is v50, and the dataset-wide store pass depends on Spec 109's clean-room
   builder carrying `minimap_lighting` as a DatasetSignal, per research.md) rather than adding a
   new parallel command; the build-fingerprint gate makes this a zero-cost no-op for
   non-0.5.3.3368 tiles. Add `report_lighting_buckets.py` to produce the per-map/overall
   distribution report from the streamed metadata. Focused C#/Python tests plus one real-client
   bounded run to sanity-check a handful of known tiles by eye.
2. **User Story 2 -- rebalance synthetic lighting variants (Implementation).** Retire the drifted
   direction math in `terrain_lighting.py` per research.md; add `spec111/lighting_buckets.py` and
   `rebalance_lighting_variants.py` to turn Phase 1's distribution report into resampling weights, and
   wire those weights into the existing synthetic-lighting-variant generation path. Verify the
   existing source/variant leak-safety tagging still holds under reweighted sampling, and verify the
   rebalanced training data's input contract still excludes ground-truth lighting/time. Focused Python
   tests.
3. **User Story 3 -- retrain and evaluate (Implementation up to the gate, then User-run).** Prepare
   `train_spec111_reconstruction.py` delegating to the canonical v50 trainer entry
   (`v50_train_wdl_prior.py`, `--release v50.1`, `require_store_release` enforced) and the
   existing group-held-out split contract (research.md); prepare the checkpoint-comparison
   evaluation. **Stop here and confirm with the user before executing.** Only after explicit
   authorization: run the training pass, evaluate the resulting checkpoint against the current
   deployed one on the held-out set, and record the improved/regressed/inconclusive outcome. A
   regression keeps the current checkpoint deployed.

## Complexity Tracking

*No constitution violations identified; table intentionally omitted.*
