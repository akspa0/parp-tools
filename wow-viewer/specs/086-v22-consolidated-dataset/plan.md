# Implementation Plan: V22 Consolidated Dataset

**Branch**: `086-v22-consolidated-dataset` | **Date**: 2026-06-30 | **Spec**: `specs/086-v22-consolidated-dataset/spec.md`

## Summary

V22 replaces the current V18 + patch-script sprawl with one dataset contract that is good enough for real object-learning work:

1. one-pass tile signals with no post-build patches,
2. native placement arrays in the store,
3. native per-build model library with the full parsed M2/WMO data needed for masking and identity learning,
4. native per-build tileset library with decoded BLP RGB,
5. C# consumer contracts that expose all of that without side-path IO,
6. validation and ablation gates aimed specifically at preventing another "dataset got richer but the model still plateaus early" failure.

The plan is not just data plumbing. It is a proof plan for whether the richer store actually improves supervision quality, observability, and model learnability.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**:
- `wow-viewer/tools/harvest/WowViewer.Tool.Harvest`
- `WowViewer.Core.IO.M2` (`M2GeometryReader`, `M2SkinReader`)
- `WowViewer.Core.IO.Wmo` (`WmoRenderDocumentReader`)
- existing BLP decode path already used for `mcly_texture_pixels_*`

**Storage**: Canonical Zarr dataset cache via the Python Zarr package. Phase 2 uses the existing C# raw array stream with a V22 profile as a builder seam; later phases write/read canonical Zarr through the Python package, not a C# Zarr implementation.

**Testing**: xUnit unit tests, bounded real-data build proofs, dataset contract checks, signal parity checks, and downstream C# read smoke proofs.

**Target Platform**: staged `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927` clients under `output/tmp/wowarchive-clients/`. `4_0_0_11927` is included only because development-map assets require it and existing object decode/render support covers that era. Other staged clients are out of scope for V22 unless this spec is explicitly reopened.

**Project Type**: dataset pipeline + consumer-library consolidation

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | PASS | All work remains inside `wow-viewer/` |
| Library-First | PASS | New logic lives in C# shared readers/serializers and `WowViewer.Tool.Harvest` |
| Real-Data Validation | REQUIRED | Two-build bounded proof before broad rebuild |
| No `H:\CLIENTS` | PASS | Only staged clients are allowed |
| One Phase at a Time | REQUIRED | Do not broaden beyond the three scoped builds before bounded proof and learnability gates pass |

## V22 Surface Inventory

V22 must explicitly cover every surface we currently spread across build stream, patch scripts, sidecars, and live archive parsing.

### A. Per-Tile Signal Arrays

Root-level arrays per tile:

- all 20 V18 base arrays,
- absorbed patch signals: `mcnr_mask_257`, `liquid_type_256`, `ground_intent_height_257`,
- integrated renderer-truth arrays: `object_visibility_mask`, `no_object_minimap`,
- placement reference arrays: counts, offsets, ids.

### B. Placement Arrays

Root-level flat arrays with per-tile offsets:

- `mddf_placement_data`
- `modf_placement_data`
- `mddf_count`, `modf_count`
- `mddf_unique_ids`, `modf_unique_ids`
- `mddf_model_ids`, `modf_model_ids`

### C. Model Library

Per-build `models/` group containing the full parsed asset needed for masking and identity work:

- M2 geometry + skin-derived triangles + render flags + blend modes + texture references + bone data,
- WMO merged geometry + materials + portals + doodad-set metadata,
- `load_error` and coverage bookkeeping.

### D. Tileset Library

Per-build `tilesets/` group containing:

- decoded BLP RGB,
- original size,
- path table,
- `load_error`,
- `mcly_tileset_ids` so chunk-layer ids map directly into the per-build tileset library.

### E. Metadata Tables

If emitted later, keep these as convenience audit views, not the source of truth:

- tile index report
- placement audit report
- decoded metadata report
- asset inventory report summarizing model/tileset coverage

## Failure Modes We Are Explicitly Designing Against

This plan should stop us from shipping a richer-looking dataset that still trains poorly.

### 1. Hidden fallback supervision

Risk: some tiles still fall back from precise masks to coarse masks and silently poison supervision.

Plan response:
- store `object_precise_mask`, `object_filtered_mask`, `object_mask`, and per-tile source diagnostics,
- store explicit `load_error` counts for models/tilesets,
- validate precise-mask coverage vs placement counts before training.

### 2. Side-path IO drift

Risk: one consumer reads tile records, another reads sidecars, another reparses assets from MPQ; contracts drift.

Plan response:
- the C# V22 dataset contract is canonical,
- sidecar reports stay for audits only,
- all training-facing data must be reachable from the store alone.

### 3. Signal-rich but semantics-poor model inputs

Risk: we add more channels, but they do not actually help the model disambiguate terrain from objects.

Plan response:
- preserve ablation-friendly separation between tile signals, placement references, model library, and tileset library,
- add bounded training proofs that test whether the new signals improve exact problem cases rather than just global loss.

### 4. Asset library incompleteness

Risk: placements reference models or textures missing from the library, so supervision remains partial.

Plan response:
- asset inventory validation gate before training,
- fail bounded proof if referenced-asset coverage drops below threshold,
- store `load_error` so missing assets are measurable, not silent.

### 5. Contract too ragged for downstream training

Risk: variable-length arrays and cached libraries make the dataset harder to use than the current one.

Plan response:
- `__getitem__` returns stable tile tensors plus placement refs,
- heavy model/tileset blobs loaded once and cached by id,
- collate helpers defined as part of the plan, not left implicit.

## Project Structure

```text
wow-viewer/src/core/WowViewer.Core.IO/Maps/
|-- RawArraySerializer.cs        # V22 raw tile profile and C# preprocessing aliases
|-- future stream payload types   # C# preprocessing and decoded payload contracts only

wow-viewer/tools/harvest/WowViewer.Tool.Harvest/
`-- Program.cs                   # stream/store contract expanded with V22 profile and asset library blobs

wow-viewer/tests/WowViewer.Core.Tests/
`-- RawArraySerializerTests.cs   # V22 raw stream contract tests

wow-viewer/data-harvester/
|-- scripts/build_v22_dataset.py # Python Zarr writer, fed decoded C# payloads
`-- src/harvester/v22_dataset.py # downstream Zarr reader/contract
```

## Implementation Phases

### Phase 1: Schema Freeze And Inventory Proof

Goal: lock the exact V22 store surface before touching the build pipeline.

Deliverables:
- final root-array list,
- final `models/` layout,
- final `tilesets/` layout,
- explicit field list for placement arrays,
- explicit C# read contract for V22 records.

Validation:
- write `v22-dataset-signals` architecture doc before implementation,
- confirm every currently used downstream signal has a place in V22,
- confirm every side-path artifact we care about is either promoted into the C# dataset contract or explicitly left audit-only.

Exit criteria:
- no unresolved "maybe store X elsewhere" questions,
- store schema stable enough for tests to pin.

### Phase 2: Stream Contract Expansion In C#

Goal: make the harvester emit everything V22 needs in one pass.

Work:
- keep existing tile signal emission,
- emit integrated versions of patched signals directly from C#,
- emit placement arrays and per-placement model ids,
- emit unique-model payloads once per build session,
- emit unique-tileset payloads once per build session,
- add load-error markers instead of throwing on unreadable assets.

Important design choice:
- tile blobs stay cheap and regular,
- model/tileset library blobs are separate message types in the stream, not duplicated per tile.

Validation:
- bounded stream dump for one tile with at least one M2, one WMO, and several terrain textures,
- confirm repeated placements do not duplicate model payloads.

### Phase 3: Python Zarr Writer And Store Layout

Goal: write the expanded C# decoded stream into a stable Zarr dataset layout using the Python package.

Work:
- add Python Zarr asset-library accumulators for models and tilesets,
- add placement offset writers,
- consume the decoded C# stream without reparsing the client,
- write canonical Zarr arrays/groups/metadata keys,
- keep sidecar reports as audit mirrors only,
- add resumability rules for partial library writes.

Validation:
- synthetic Zarr build with known tiny payloads,
- restart/resume proof,
- parity check between stored placements and source placement records.

### Phase 4: Consumer Contract

Goal: define the one Zarr-backed dataset API every downstream consumer uses.

Work:
- implement the V22 Zarr reader contract,
- expose cached `models` and `tilesets` libraries,
- expose placement refs and ids without forcing every batch to inline giant geometry blobs,
- provide collate helpers for tasks that do need batched placement/model alignment.

Validation:
- fixed-key contract tests,
- empty-tile tests,
- model/tileset cache-hit tests,
- shape/dtype tests against a synthetic store.

### Phase 5: Bounded Real-Data Proof Build

Goal: prove the store on real data before broad rebuild.

Scope:
- `3_3_5_12340` Azeroth bounded tile set,
- `0_5_3_3368` Azeroth bounded tile set,
- `4_0_0_11927` development-map bounded tile set for Cata-only development-map assets,
- include object-rich tiles and low-object tiles.

Validation matrix:
- precise-mask visual review,
- placement-array parity,
- model-library completeness (% of referenced model ids resolvable),
- tileset-library completeness (% of referenced tileset ids resolvable),
- WMO mask parity vs V18,
- signal coverage JSON with hard thresholds.

Exit criteria:
- no unresolved silent fallbacks,
- asset coverage high enough to justify full rebuild,
- downstream consumer can read the store end-to-end.

### Phase 6: Learnability Gates

Goal: explicitly test whether V22 fixes the kinds of supervision failures that led to bad plateaus.

This phase is not "train the final model." It is a bounded proof that the new dataset is more learnable.

Required proofs:
- **Tiny-overfit proof**: can a small model overfit 8-32 tiles using V22 signals faster and lower than the same route on V18?
- **Mask-consistency proof**: can geometry-driven reconstructed masks from stored model library match `object_precise_mask` on held tiles?
- **Asset-reference proof**: can a simple retrieval baseline use `mddf_model_ids` + stored geometry to outperform coarse footprint-only object matching?
- **Tileset proof**: can texture-aware supervision built from stored tilesets reproduce synthetic minimap/albedo without reading BLPs externally?

Exit criteria:
- at least one bounded model route shows improved fit or lower error on the previously object-confused cases,
- if not, stop and diagnose before broad rebuild migration.

### Phase 7: Three-Build Rebuild And Consumer Migration

Goal: rebuild the three scoped V22 stores and migrate consumers off V18 for those builds only.

Work:
- rebuild `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927`,
- switch selected downstream consumers from legacy dataset contracts to the C# V22 contract,
- deprecate patch scripts and old promotion flow,
- publish migration notes in architecture docs.

Validation:
- bounded train smoke on migrated consumers,
- asset-library coverage report for every build,
- no remaining consumer that requires MPQ reparse or sidecar-only path for core dataset semantics.

## Validation And Diagnostics Matrix

The dataset is not done when it builds. It is done when these checks pass.

### Store-Level

- signal coverage
- placement coverage
- model-library completeness
- tileset-library completeness
- load-error counts
- resumability / partial-build recovery

### Geometry-Level

- M2 triangle counts sane
- WMO merged mesh counts sane
- bounds finite
- render-flag arrays aligned with geometry batches
- texture path references resolvable

### Consumer-Level

- C# V22 reader fixed keys
- collate helper handles empty placements
- models/tilesets cache stable under multi-worker DataLoader use

### Learnability-Level

- tiny-overfit
- exact-mask reconstruction parity
- object-retrieval baseline improvement
- texture-composition baseline improvement

## Anti-Plateau Angle Coverage

Why this plan should help more than just "more channels":

- object masks stop being just footprints; they are now linked to the exact source model geometry,
- tilesets stop being ids only; texture RGB becomes directly available,
- placements stop being a sidecar table; the training contract can correlate footprint, pose, source model, and source texture in one batch,
- we add learnability gates before full migration so a bad dataset design gets caught early.

If the model still stalls after this, we will at least know whether the failure is:

1. mask supervision quality,
2. missing asset coverage,
3. consumer contract shape,
4. training architecture,
5. or a genuinely insufficient signal problem.

That is the point of V22: not just more data, but fewer unknowns.
