# wow-viewer Shared I/O Library Plan

This document locks the current non-PM4 shared-format direction for `wow-viewer` after the first shared map-summary and cross-family detection slices landed on Mar 26, 2026.

## Source-Of-Truth Rule

- `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` are the intended home for first-party non-PM4 file-format ownership.
- `gillijimproject_refactor` remains the current runtime and compatibility reference while deeper format behavior is still being ported.
- `WowViewer.Tool.Inspect` and `WowViewer.Tool.Converter` should consume shared detection or reader seams instead of owning parallel heuristics.

## Goal

Create one first-party shared format stack in `wow-viewer` that:

- owns reusable FourCC, chunk, detection, and summary contracts
- grows into canonical readers or writers for map, object, model, texture, and table families
- feeds inspect and converter surfaces without duplicate parsing logic
- keeps top-level detection, summary, deep parsing, and writing boundaries explicit instead of pretending they are already the same level of proof

## Current Landed Shared Boundary

### Core contracts

- `WowViewer.Core/Chunks`
  - `FourCC`
  - `ChunkHeader`
- `WowViewer.Core/Maps`
  - `MapChunkIds`
  - `MapFileKind`
  - `MapChunkLocation`
  - `MapFileSummary`
  - `AdtSummary`
  - `AdtChunkIds`
  - `AdtMcnkSummary`
  - `WdtSummary`
- `WowViewer.Core/Wmo`
  - `WmoChunkIds`
  - `WmoSummary`
  - `WmoGroupSummary`
  - `WmoLiquidBasicType`
  - `WmoGroupLiquidSummary`
  - `WmoGroupBatchSummary`
  - `WmoGroupFaceMaterialSummary`
  - `WmoGroupUvSummary`
  - `WmoGroupVertexColorSummary`
  - `WmoGroupDoodadRefSummary`
  - `WmoGroupIndexSummary`
- `WowViewer.Core/Files`
  - `WowFileKind`
  - `WowFileDetection`

### Core.IO seams

- `WowViewer.Core.IO/Chunked`
  - `ChunkHeaderReader`
  - `ChunkedFileReader`
- `WowViewer.Core.IO/Maps`
  - `MapFileSummaryReader`
  - `AdtSummaryReader`
  - `AdtMcnkSummaryReader`
  - `WdtSummaryReader`
- `WowViewer.Core.IO/Wmo`
  - `WmoSummaryReader`
  - `WmoGroupSummaryReader`
  - `WmoGroupReaderCommon`
  - `WmoGroupLiquidSummaryReader`
  - `WmoGroupBatchSummaryReader`
  - `WmoGroupFaceMaterialSummaryReader`
  - `WmoGroupUvSummaryReader`
  - `WmoGroupVertexColorSummaryReader`
  - `WmoGroupDoodadRefSummaryReader`
  - `WmoGroupIndexSummaryReader`
- `WowViewer.Core.IO/Files`
  - `WowFileDetector`
  - `Md5TranslateIndex`
  - `Md5TranslateResolver`
  - `MinimapService`
  - `IArchiveReader`
  - `IArchiveCatalog`
  - `IArchiveCatalogFactory`
  - `DbClientFileReader`
  - `ArchiveCatalogBootstrapper`
  - `ArchiveCatalogBootstrapResult`
  - `AlphaArchiveReader`
  - `PkwareExplode`
  - `MpqArchiveCatalog`
  - `MpqArchiveCatalogFactory`
- `WowViewer.Core.IO/Dbc`
  - `DbcReader`
  - `DbcHeader`
  - `MapDirectoryLookup`
  - `GroundEffectLookup`
  - `AreaIdMapper`
  - `AreaIdMapper` now also prefers DBCD + WoWDBDefs for known `AreaTable` and `Map` builds when extracted tables and definitions are available, using the same vendored DBCD project and WoWDBDefs definition layout the active viewer already consumes, with raw `DbcReader` kept as a narrow fallback

### Tool consumers

- `WowViewer.Tool.Inspect`
  - `map inspect --input <file.wdt|file.adt>`
  - `wmo inspect --input <file.wmo>`
- `WowViewer.Tool.Converter`
  - `detect --input <file>`

### Regression floor

- `wow-viewer/tests/WowViewer.Core.Tests`
  - FourCC and chunk-header boundary tests
  - synthetic WDT and ADT summary tests
  - real-data `development.wdt` and `development_0_0.adt` summary tests
  - synthetic WDT detection tests
  - real-data WDT, ADT, split ADT, and PM4 detection tests

## Current Fixed-Dataset Proof

- `development.wdt` -> `Wdt`, version `18`
- `development_0_0.adt` -> `Adt`, version `18`
- `development_0_0_tex0.adt` -> `AdtTex`, version `18`
- `development_0_0_obj0.adt` -> `AdtObj`, version `18`
- `development_00_00.pm4` -> `Pm4`, version `12304`

## Current Boundaries

- shared detection is real
- top-level WDT or ADT chunk summary is real
- shared ADT semantic summary for terrain-chunk counts, string-table counts, placement counts, and selected top-level MFBO or MH2O or MAMP or MTXF presence is now real
- shared ADT `MCNK` semantic summary for root-header coverage, selected flags, split-file subchunk presence, and per-chunk layer-count signals is now real
- shared WDT semantic summary for MPHD flags, MAIN occupancy, string-table counts, and top-level placement counts is now real
- shared WMO root semantic summary for `MOHD`-reported counts, selected top-level entry counts, string-table counts, flags, and bounds is now real
- shared WMO group semantic summary for `MOGP` headers and top-level geometry or metadata subchunk counts is now real
- shared WMO `MLIQ` semantic summary for dimensions, height range, tile visibility, and basic family inference is now real
- shared WMO `MOBA` batch semantic summary for entry counts, index coverage, material-id coverage, and flagged-batch counts is now real
- shared WMO `MOPY` face-material semantic summary for face counts, material-id coverage, hidden-face counts, and flagged-face counts is now real
- shared WMO `MOTV` UV semantic summary for primary UV ranges and extra-set counts is now real
- shared WMO `MOCV` vertex-color semantic summary for channel ranges, average alpha, and extra-set counts is now real
- shared WMO `MODR` doodad-ref semantic summary for ref counts, distinct refs, duplicate refs, and min or max ref ranges is now real
- shared WMO `MOVI` or `MOIN` index semantic summary for index counts, triangle counts, ranges, and degenerate-triangle counts is now real
- shared MD5 minimap translation and minimap tile path resolution are now real
- shared standard-archive read and DBC or DB2 table probing boundaries are now real
- shared archive bootstrap or external listfile parsing and Alpha per-asset MPQ wrapper reading are now real
- the concrete shared standard MPQ implementation used by the active `MdxViewer` path is now real
- shared DBC-backed map-directory and ground-effect lookup helpers are now real
- shared area-ID mapping plus embedded area-crosswalk ownership are now real
- shared area-ID mapping now also has a real DBCD + WoWDBDefs-backed load path for the active `AreaTable` or `Map` seam when extracted `gillijimproject_refactor/test_data/*/tree/DBFilesClient` inputs exist, with legacy workspace-root `test_data/*/tree/DBFilesClient` kept only as a fallback probe
- shared Alpha per-asset MPQ ownership now also covers the active `WoWMapConverter.Core` converter and VLM callers
- the dead duplicate old-repo helper layer behind those active paths has now been deleted from `WoWMapConverter.Core`
- this does not yet prove:
  - deep WDT semantic parsing
  - deep ADT root or split payload parsing
  - WMO, M2, or BLP payload parsing
  - general-purpose DBC or DB2 schema ownership beyond the narrow shared lookup helpers that now exist
  - any write path or round-trip support
  - runtime cutover inside the active viewer

## Immediate Next Slices

1. deepen shared ADT root and split-file top-level summaries
2. deepen shared ADT root and split-file summaries beyond the new semantic-summary layer into narrower chunk-internal signals only when a deep payload proof target is clear
3. deepen shared WMO ownership from root, group, `MLIQ`, `MOBA`, `MOPY`, `MOTV`, `MOCV`, `MODR`, and `MOVI` or `MOIN` semantic summary into the next narrow deep-payload seam only when a fixed validation target is available
4. keep inspect and converter as thin consumers of shared seams instead of adding direct parsing in tool entrypoints
5. continue shrinking `MdxViewer` imports of `WoWMapConverter.Core` by moving other narrow non-MPQ helpers onto `Core` or `Core.IO`
6. add first shared WMO or model-family top-level summary seam where converter or viewer compatibility still depends on old-repo ownership
7. decide whether the next non-PM4 slice is higher-level CASC or MPQ unification, a new shared format family, or a deeper summary or reader slice for existing families

## Failure Modes To Avoid

- letting inspect and converter rebuild their own file-family heuristics instead of consuming `WowFileDetector`
- describing shared detection as if it were canonical payload parsing
- drifting back into PM4-only workflow assets when the active task is non-PM4 shared I/O
- keeping multiple first-party parser roots alive once a shared `wow-viewer` seam exists

## Copilot Continuity Surface

- future sessions should treat these `.github` assets as the canonical shared workflow for the active non-PM4 shared-I/O slice:
  - `.github/skills/wow-viewer-shared-io-library/SKILL.md`
  - `.github/skills/wow-viewer-migration-continuation/SKILL.md`
  - `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`
  - `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
  - `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`
- use the shared-I/O implementation prompt when the ask is an actual `Core` or `Core.IO` slice, inspect verb, converter command, or regression update
- use the broader shared-I/O plan prompt when the ask is format ownership, source-root consolidation, read/write authority, or migration ordering
- whenever a new `wow-viewer` skill or implementation prompt is added, update `.github/copilot-instructions.md`, `wow-viewer/README.md`, this plan, and the memory-bank continuity files in the same slice so the new route is discoverable without manual recovery

## Validation Status

- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `64` tests after the shared detector, MD5 minimap translation, archive-reader, archive bootstrap, Alpha wrapper, concrete MPQ catalog, shared DBC lookup, and shared AreaIdMapper slices landed
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `66` tests after wiring the shared `AreaIdMapper` seam to DBCD + WoWDBDefs and adding explicit missing-tree diagnostics
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `71` tests after adding the shared WDT semantic summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `77` tests after adding the shared ADT semantic summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `84` tests after adding the shared ADT `MCNK` semantic summary slice and the first shared WMO root semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `87` tests after adding the shared WMO group semantic-summary slice and `MOGP`-first detection coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `88` tests after adding the shared WMO `MLIQ` semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `90` tests after adding the shared WMO `MOBA` batch semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `92` tests after adding the shared WMO `MOPY` face-material semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `93` tests after adding the shared WMO `MOTV` UV semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `94` tests after adding the shared WMO `MOCV` vertex-color semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `95` tests after adding the shared WMO `MODR` doodad-ref semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `97` tests after adding the shared WMO `MOVI` or `MOIN` index semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `37` tests after adding archive-backed `AreaIdMapper` coverage and shorthand-build normalization for archive-fed DBCD loads
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `46` tests after adding shared ADT semantic-summary coverage for root, `_tex0.adt`, and `_obj0.adt`
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `53` tests after adding shared ADT `MCNK` semantic-summary coverage plus a synthetic WMO root summary test
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `56` tests after adding synthetic WMO group summary coverage plus `MOGP`-first detector coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `57` tests after adding synthetic WMO `MLIQ` semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `59` tests after adding synthetic WMO `MOBA` batch semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `61` tests after adding synthetic WMO `MOPY` face-material semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `62` tests after adding synthetic WMO `MOTV` UV semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `63` tests after adding synthetic WMO `MOCV` vertex-color semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `64` tests after adding synthetic WMO `MODR` doodad-ref semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `66` tests after adding synthetic WMO `MOVI` and `MOIN` index semantic-summary coverage
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt` passed on Mar 27, 2026 and now reports the shared WDT semantic summary `wmoBased=False tiles=1496/4096 mainCellBytes=8 doodadNames=0 wmoNames=0 doodadPlacements=0 wmoPlacements=0`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now reports the shared ADT semantic summary `kind=AdtTex terrainChunks=256 textures=5 doodadNames=0 wmoNames=0 doodadPlacements=0 wmoPlacements=0 hasMfbo=False hasMh2o=False hasMamp=True hasMtxf=False`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 27, 2026 and now reports the shared ADT semantic summary `kind=AdtObj terrainChunks=256 textures=0 doodadNames=6 wmoNames=12 doodadPlacements=10 wmoPlacements=15 hasMfbo=False hasMh2o=False hasMamp=False hasMtxf=False`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now also reports the shared ADT `MCNK` semantic summary `mcnk=256 zero=0 headerLike=0 distinctIndex=0 duplicateIndex=0 areaIds=0 holes=0 liquidFlags=0 mccvFlags=0 mcly=256 mcal=203 mcsh=174 mccv=0 mclq=0 mcrd=0 mcrw=0 totalLayers=775 maxLayers=4 multiLayerChunks=203`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-summary-test.wmo` passed on Mar 27, 2026 and now reports the first shared WMO root semantic summary `materials=2/2 groups=4/4 portals=1 lights=3 textures=2 doodadNames=5/5 doodadPlacements=6/6 doodadSets=2/2 flags=0x00001234`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-summary-test.wmo` passed on Mar 27, 2026 and now reports the first shared WMO group semantic summary `Header: bytes=68 ... Geometry: faces=3 vertices=2 indices=3 ... hasLiquid=False`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-liquid-test.wmo` passed on Mar 27, 2026 and now reports the shared `MLIQ` semantic summary `payloadBytes=63 verts=2x2 tiles=1x1 ... visibleTiles=1/1 ... liquidType=Ocean`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-batch-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOBA` semantic summary `payloadBytes=48 entries=2 hasMaterialIds=True distinctMaterials=2 highestMaterialId=7 totalIndexCount=15 firstIndexRange=10-20 maxIndexEnd=29 flaggedBatches=1`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-face-v17-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOPY` semantic summary `payloadBytes=8 entryBytes=2 faces=4 distinctMaterials=2 highestMaterialId=7 hiddenFaces=1 flaggedFaces=2`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-uv-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOTV` semantic summary `payloadBytes=24 primaryUv=3 rangeU=[-0.200, 0.800] rangeV=[0.200, 0.900] extraUvSets=1 totalExtraUv=2 maxExtraUv=2`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-color-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOCV` semantic summary `payloadBytes=8 primaryColors=2 rangeR=[30, 70] rangeG=[20, 60] rangeB=[10, 50] rangeA=[40, 80] avgA=60 extraColorSets=1 totalExtraColors=3 maxExtraColors=3`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-doodadref-test.wmo` passed on Mar 27, 2026 and now reports the shared `MODR` semantic summary `payloadBytes=8 refs=4 distinctRefs=3 refRange=3-9 duplicateRefs=1`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-index-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOVI` semantic summary `payloadBytes=12 indices=6 triangles=2 distinctIndices=4 indexRange=0-3 degenerateTriangles=1`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert i:/parp/parp-tools/gillijimproject_refactor/test_data/0.5.3/alphawdt/World/Maps/PVPZone01/PVPZone01.wdt -o i:/parp/parp-tools/output/pvpzone01-alpha-to-lk-smoke-dbcd-check3 -v` passed on Mar 27, 2026 and confirmed the new explicit warning now names the preferred `gillijimproject_refactor/test_data/0.5.3/tree` and `gillijimproject_refactor/test_data/3.3.5/tree` `AreaTable` and `Map` roots first when extracted files are missing
- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 27, 2026 after switching the converter to lazy archive-backed `AreaIdMapper` initialization and adding `--alpha-client` plus `--lk-client`
