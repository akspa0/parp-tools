# wow-viewer Full Format Ownership Plan

This document records the Mar 28, 2026 program reset for `wow-viewer`.

The target is no longer a collection of narrow summary seams plus selective consumer hooks. The target is full first-party ownership in `wow-viewer` for every active format family currently handled by `gillijimproject_refactor/src/MdxViewer`, with `MdxViewer` reduced to a compatibility consumer until `WowViewer.App` replaces it.

## Non-Negotiable Goal

`wow-viewer` must fully own every chunk, bitfield, payload, decode rule, write rule, runtime-facing contract, and tool surface required for the active viewer's supported formats.

That means:

- `wow-viewer/src/core/WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.Runtime`, and `WowViewer.Core.PM4` become the canonical implementation owners.
- `WowViewer.Tool.Inspect`, `WowViewer.Tool.Converter`, and the future `WowViewer.App` consume those owned services instead of re-parsing files locally.
- `gillijimproject_refactor`, `Warcraft.NET`, `SereniaBLPLib`, `Pm4Research`, `WoWMapConverter`, `PM4Tool`, and other old roots remain extraction or comparison input until their behavior is re-owned.
- Current summary seams are still useful, but they are only stepping stones, not the end state.

## In-Scope Format Families

The full-ownership target covers every active viewer family, not just the families already partially migrated:

1. `ADT` root files
2. split `ADT` companions: `_tex0.adt`, `_obj0.adt`, `_lod.adt`
3. `WDT`
4. `WDL`
5. `WMO` root files
6. `WMO` group files and embedded-group payloads
7. `MDX`
8. `M2`
9. `BLP`
10. `PM4`
11. `DBC` and `DB2` families actually consumed by the active viewer or converter path
12. minimap translation and related auxiliary owned file seams used by the active viewer runtime

## Ownership Standard

A format family is not considered migrated just because `wow-viewer` can classify it or summarize a few chunks.

For each family, the owned target is:

- first-party chunk and payload contracts
- first-party readers for all active versions the current viewer handles
- first-party writers when the current toolchain writes that family today
- first-party runtime-facing service seams used by viewer, inspect, converter, and batch jobs
- first-party regression coverage against real fixed datasets where available
- no permanent dependency on old viewer-local parsing or opaque third-party readers for core correctness

## Current Gap Snapshot

The repo already has real narrow ownership for parts of `WDT`, `ADT`, `WMO`, `BLP`, `MDX`, `DBC`, and `PM4`, but the current state is still heavily summary-oriented.

The largest known full-ownership gaps are:

- `ADT` alpha decode and terrain texture semantics: `MCAL`, `MCLY`, neighbor stitching, 4.x residual synthesis, split-file payload routing
- `ADT` split-file full ownership: `_tex0`, `_obj0`, `_lod` chunk payloads, not just top-level summaries
- `WDL` first-party parsing
- `WMO` deep root/group ownership: texture tables, material semantics, liquid detail, visibility/topology usage, standalone group parity
- `MDX` deep ownership beyond top-level summary: animation tracks, geosets, bones, emitters, material semantics, render-time contracts
- `M2` first-party parsing and runtime contracts instead of Warcraft.NET consumer-only behavior
- `BLP` first-party pixel decode and write path instead of SereniaBLPLib-only ownership
- `PM4` full semantic ownership for grouping, transforms, linkage, correlation, and restore-facing geometry behavior
- `DBC` and `DB2` breadth beyond the current narrow lookup helpers

## Program Rules

1. Do not stop at summary ownership when the active viewer already depends on deeper behavior.
2. Do not leave permanent format truth in `MdxViewer` once the same family has a library home in `wow-viewer`.
3. Do not treat third-party libraries as the final owner of `M2`, `BLP`, or other core families if the active viewer depends on them for correctness.
4. Do not duplicate parsing in inspect, converter, or app surfaces once a shared library seam exists.
5. Keep version-specific behavior explicit instead of flattening Alpha, LK, Cataclysm, and later layouts into one vague reader.
6. Keep proof language precise: classification, summary, deep parse, runtime-service ownership, and write-path ownership are different milestones.

## Program Workstreams

### 1. Terrain And Map Ownership

Goal:
- fully re-own `ADT`, split `ADT`, and `WDT` read/write/runtime contracts

Priority gaps:
- `MCAL` decode unification
- `MCLY` semantics
- split-file routing and payload ownership
- placement/liquid/shadow payload parity
- build-aware profile routing in shared libraries instead of viewer-local terrain logic

Primary reference inputs:
- `src/MdxViewer/Terrain/*`
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/*`

### 2. Model Ownership

Goal:
- fully re-own `MDX` and `M2` parsing/runtime contracts

Priority gaps:
- `MDX` animation and geoset semantics
- `MDX` material/runtime ownership beyond `TEXS` and `MTLS`
- first-party `M2` chunk parsing and model-profile routing
- runtime texture slot and replaceable-texture rules

Primary reference inputs:
- `src/MdxViewer/Formats/Mdx/*`
- `src/MdxViewer/Rendering/*`
- `Warcraft.NET` as reference only

### 3. Texture Ownership

Goal:
- fully re-own `BLP` decode and write behavior

Priority gaps:
- palette, JPEG, and DXT pixel decode
- mip decode/write ownership
- replaceable and runtime texture resolution rules

Primary reference inputs:
- `SereniaBLPLib`
- `src/MdxViewer` texture/render/export paths

### 4. WMO Ownership

Goal:
- fully re-own root/group/embedded-group `WMO` behavior used by viewer and converter paths

Priority gaps:
- material semantics
- texture table ownership
- liquid detail decode
- standalone group parity
- deeper visibility and topology behavior used by runtime or converter paths

Primary reference inputs:
- current shared `wow-viewer` `WMO` seams
- `WmoRenderer`
- `WmoV14ToV17Converter`

### 5. PM4 Ownership

Goal:
- finish `Core.PM4` as the canonical owner for the active PM4 behavior surface

Priority gaps:
- remaining semantic fields and grouping behavior
- restore-facing geometry and conversion services
- remaining viewer-local PM4 behavior still living in `WorldScene`

Primary reference inputs:
- `WowViewer.Core.PM4`
- `src/MdxViewer/Terrain/WorldScene.cs`
- `Pm4Research`

### 6. Runtime And Tooling Cutover

Goal:
- ensure the same owned services drive inspect, converter, and the future app

Priority gaps:
- tool-local fallback logic
- viewer-local runtime ingestion and profile ownership
- DBC/DB2 breadth for active runtime consumers
- minimap translation/runtime auxiliary seams

## Recommended Execution Order

1. Finish terrain/map ownership first, especially `MCAL` and split `ADT`, because that is the highest regression risk and currently split across multiple codepaths.
2. Continue `WMO` and `MDX` from summary into deep payload ownership because those families already have working footholds in `wow-viewer`.
3. Start first-party `M2` ownership rather than leaving Warcraft.NET as the effective long-term parser.
4. Replace `BLP` black-box ownership with first-party decode/write seams.
5. Keep finishing `PM4` in parallel where it already has strong direct ownership momentum in `Core.PM4`.
6. Route converter and app behavior onto those owned services as each family becomes real, instead of saving all cutover work for the end.

## Immediate Program-Level Deliverables

The next planning and implementation cycles should produce all of the following:

1. a tracked family-by-family parity matrix against active `MdxViewer` behavior
2. a terrain ownership plan centered on `MCAL` and split `ADT`
3. a model ownership plan centered on full `MDX` and first-party `M2`
4. a texture ownership plan centered on full `BLP` decode/write parity
5. a tool cutover sequence that retires viewer-local parsing as each family becomes library-owned

## Validation Standard

For full-ownership claims, require the strongest applicable proof:

- `wow-viewer` build and tests
- real inspect or converter runs on fixed datasets
- `MdxViewer` compatibility checks only when a slice intentionally touches the current consumer path
- real runtime viewer validation when the claim reaches render/runtime parity

Do not describe summary-reader passes or compatibility probes as full format closure.

## Bottom Line

`wow-viewer` is now explicitly on the hook to become the first-party owner of every active `MdxViewer` format family.

The migration is no longer finished when the repo can classify files and print summaries. It is finished only when `wow-viewer` owns the formats deeply enough that the current viewer and tools are just consumers of that library surface.