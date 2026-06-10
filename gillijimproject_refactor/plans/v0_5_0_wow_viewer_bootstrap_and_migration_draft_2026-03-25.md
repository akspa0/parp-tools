# v0.5.0 wow-viewer Bootstrap And Migration Draft

This document is the first concrete architecture draft for moving the production viewer/tool stack out of `parp-tools` and into `https://github.com/akspa0/wow-viewer`.

It is intentionally narrower than a full implementation plan: the goal is to lock the repo shape, first migration slices, and source-of-truth boundaries before the actual code move starts.

## Scope

- `parp-tools` remains the R&D, archaeology, and reference repo.
- `wow-viewer` becomes the production repo for the shipping viewer and first-party shared library.
- first-party parsing, reading, writing, conversion, and runtime contracts are re-owned in the new repo.
- upstream externals stay external under `libs/`.
- performance work beyond first triage should land against the new library/runtime boundaries, not as endless local surgery in `parp-tools`.

## Proposed Top-Level Repo Tree

```text
wow-viewer/
  .github/
  docs/
  eng/
  scripts/
  libs/
    wowdev/WoWDBDefs/
    wowdev/wow-listfile/
    ModernWoWTools/Warcraft.NET/
    ModernWoWTools/ADTMeta/
    wowdev/DBCD/
    alpha-core/
    Marlamin/WoWTools.Minimaps/
    WoW-Tools/SereniaBLPLib/
  src/
    viewer/
      WowViewer.App/
      WowViewer.App.Tests/
    core/
      WowViewer.Core/
      WowViewer.Core.Tests/
      WowViewer.Core.IO/
      WowViewer.Core.IO.Tests/
      WowViewer.Core.Runtime/
      WowViewer.Core.Runtime.Tests/
    tools-shared/
      WowViewer.Tools.Shared/
  tools/
    converter/
      WowViewer.Tool.Converter/
    inspect/
      WowViewer.Tool.Inspect/
    catalog/
      WowViewer.Tool.Catalog/
  research/
    pm4/
    rollback/
  testdata/
```

## Repo Shape Rules

- `src/viewer/` is the obvious shipping-app home. A new engineer should find the app immediately.
- `src/core/` contains first-party owned library code only.
- `tools/` contains standalone executables that consume the same first-party library contracts as the viewer.
- `libs/` contains cloned upstream repos and support data that keep their own identity and update cadence.
- `research/` is optional in the new repo and should stay small. If a seam is mostly archaeology, it should remain in `parp-tools` instead.
- `testdata/` should only contain stable fixtures or pointers/notes for real validation paths, not giant ad-hoc dumps checked in casually.

## Bootstrap Policy

### Auto-cloned during bootstrap

- `wowdev/wow-listfile`
- `wowdev/WoWDBDefs`
- `wowdev/DBCD`
- `ModernWoWTools/Warcraft.NET`
- `Marlamin/WoWTools.Minimaps`
- `WoW-Tools/SereniaBLPLib`

### Optional/manual evaluation dependencies

- `ModernWoWTools/ADTMeta`
- `Marlamin/wow.tools.local`
- `Kruithne/wow.export`
- `MapUpconverter`
- any future `Noggit` / `noggit-red` alpha-era outreach work

The distinction matters: the first list is baseline repo/bootstrap material, while the second list is reference or later integration work.

## First-Party Library Boundaries

### `WowViewer.Core`

- FourCC and chunk primitives
- binary read/write helpers
- build/version catalogs and profile contracts
- shared terrain/map/model/WMO domain types
- shared placement and liquid domain contracts

### `WowViewer.Core.IO`

- owned readers and writers for Alpha and standard terrain/map formats
- owned model/WMO format readers and writers where the repo intends to own correctness
- conversion pipeline contracts
- import/export codecs currently split across viewer and converter code

### `WowViewer.Core.Runtime`

- data-source abstractions
- world/runtime scene contracts that are not renderer-specific
- SQL/DBC/listfile ingestion contracts
- placement/runtime normalization
- viewer/tool-facing service layer that keeps the app from owning format truth

### `WowViewer.App`

- UI shell
- renderer integration
- interaction tools
- view models/state orchestration
- no format-specific parsing logic beyond thin adapter calls into owned library services

### `WowViewer.Tools.Shared`

- common CLI option parsing helpers
- shared reporting/export plumbing
- reusable headless execution harnesses for converter/inspect/catalog tools

## Migration Inventory By Source Root

### `src/gillijimproject-csharp`

Absorb first:

- `WowFiles/Chunk*`, `WowChunkedFormat`, shared chunk primitives
- Alpha and LK ADT/WDT chunk models that still express useful file structure
- low-level utilities that remain valid after cleanup

Absorb carefully after review:

- placement/liquid helpers whose semantics may have drifted
- any writer code that bakes old assumptions now contradicted by active viewer behavior

Do not preserve as architecture:

- the project identity itself
- any assumption that this assembly remains a long-term dependency layer in `wow-viewer`

### `src/WoWMapConverter/WoWMapConverter.Core`

Absorb first:

- `Formats/Shared` chunk/domain models that are still sound
- the best current read/write/conversion seams from `Formats/`, `Converters/`, and `Builders/`
- DBC access helpers that can become neutral library services

Split and evaluate:

- `VLM/` services, which are useful but should not contaminate the baseline shared runtime library if they remain specialized export pipelines
- diagnostics/helpers that are valuable for tools but not necessarily viewer-runtime dependencies

Keep tool-facing rather than viewer-facing:

- dataset baking/export workflows
- specialized training/export services that are not required for the shipping viewer runtime

### `src/MdxViewer`

Absorb first:

- practical read/runtime behavior now missing from older libraries
- data-source abstractions and listfile/DBC access seams
- terrain/profile/build-version routing logic
- shared export/import seams that are not inherently UI-bound

Extract later behind cleaner contracts:

- scene/runtime services currently mixed with renderer assumptions
- SQL population services once their contracts are separated from UI state
- non-renderer PM4 inspection logic if it proves useful outside the app shell

Keep app-local:

- ImGui UI shell and panels
- render-loop orchestration
- direct input handling
- app-specific view state and debugging windows

## Migration Order

### Phase 0 - Bootstrap And Naming Lock

- create the `wow-viewer` solution skeleton
- establish the top-level folder layout above
- script baseline bootstrap for `libs/` and support repos
- lock package/project names before code movement starts

Exit condition:

- the repo can be cloned and bootstrapped repeatably without tribal setup steps

### Phase 1 - Core Chunk And Format Foundation

- build `WowViewer.Core` and `WowViewer.Core.IO`
- move/re-author chunk primitives, FourCC handling, and binary helpers
- consolidate build/version/profile catalogs
- land the first owned ADT/WDT/WMO/M2 domain contracts

Primary source roots:

- `gillijimproject-csharp/WowFiles`
- `WoWMapConverter.Core/Formats/Shared`
- selected viewer runtime/profile helpers

Exit condition:

- the new library can open and describe target files without the viewer hosting the core format truth

### Phase 2 - Terrain And Placement Canonicalization

- re-own Alpha and standard terrain readers/writers with explicit separation
- move placement and liquid contracts into the owned library
- define what remains partial versus what is considered canonical

Primary source roots:

- `MdxViewer/Terrain`
- `WoWMapConverter.Core/Formats`
- `WoWMapConverter.Core/Builders`
- `gillijimproject-csharp/WowFiles/Alpha`
- `gillijimproject-csharp/WowFiles/LichKing`

Exit condition:

- one owned library path exists for terrain/map/placement I/O instead of three drifting implementations

### Phase 3 - Runtime Ingestion Split

- move data-source, DBC, listfile, and SQL ingestion contracts into `WowViewer.Core.Runtime`
- keep the viewer as a consumer of runtime services instead of the owner of file/data ingestion policy
- establish SQLite/cache seams here rather than as UI features

Primary source roots:

- `MdxViewer/DataSources`
- `MdxViewer/Population`
- selected `WoWMapConverter.Core/Dbc`

Exit condition:

- headless tools and the viewer can share the same ingestion/runtime contracts

### Phase 4 - Viewer Thinning

- move the viewer onto the new runtime/library packages
- keep renderer and UI code app-local
- delete duplicate format/runtime logic from the app as owned-library replacements land

Exit condition:

- the viewer app can no longer be described as the place where format truth lives

### Phase 5 - Tool Cutover

- rebase converter, inspect, and catalog tools on the same core/runtime contracts
- move tool-specific execution into `tools/`
- leave experimental or archaeology-only tools behind in `parp-tools`

Exit condition:

- converter and inspection executables share library contracts with the viewer instead of duplicating code

### Phase 6 - Deeper Performance And Fidelity Work

- start the larger performance overhaul only after the runtime/library split is real
- then evaluate enhanced terrain, shader-family work, richer SQL actor fidelity, and later-client conversion extensions

Exit condition:

- performance work is improving the intended architecture, not cementing the old one

## First Vertical Slice

The first shipping proof should be narrow:

- bootstrap `wow-viewer`
- land `WowViewer.Core` + `WowViewer.Core.IO`
- open one real map/tile set through the new owned library
- have a minimal viewer shell in `src/viewer/` consume that library for loading rather than using app-local parsing

That slice is enough to prove the split is real without pretending the entire renderer/runtime stack has already migrated.

## What Stays In `parp-tools`

- broad archaeology and reverse-engineering notes
- unfinished PM4 research threads
- dead-end experiments
- one-off recovery branches and historical baselines
- any niche tool that does not earn its way into the production repo

## Validation Standard

- bootstrap scripts must work on a clean clone
- owned readers/writers must be validated against real data, not only fixtures
- terrain, placement, SQL-scene, and lighting claims require runtime real-data validation before being called complete
- build-only validation is acceptable for scaffolding, not for correctness claims