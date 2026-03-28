# wow-viewer Shared I/O Library Plan

This document locks the current non-PM4 shared-format direction for `wow-viewer` after the first shared map-summary and cross-family detection slices landed on Mar 26, 2026.

## Mar 28, 2026 - Full Ownership Reset

- User direction is now explicit: `wow-viewer` must fully re-own every active format family currently handled by `MdxViewer`, not just classify files and accumulate narrow summary seams.
- This shared-I/O plan remains valid as the non-PM4 implementation track, but current summary readers should be treated as stepping stones toward full first-party parse, write, and runtime-service ownership.
- In-scope non-PM4 families now explicitly include full `ADT` root and split ownership, `WDT`, `WDL`, `WMO` root/group ownership, `MDX`, `M2`, `BLP`, and the active `DBC`/`DB2` families consumed by viewer and converter paths.
- The dedicated program document for this reset is `gillijimproject_refactor/plans/wow_viewer_full_format_ownership_plan_2026-03-28.md`.

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
  - `WmoGroupVertexSummary`
  - `WmoGroupNormalSummary`
  - `WmoGroupInfoSummary`
  - `WmoMaterialSummary`
  - `WmoTextureTableSummary`
  - `WmoDoodadNameTableSummary`
  - `WmoDoodadSetSummary`
  - `WmoDoodadPlacementSummary`
  - `WmoGroupNameTableSummary`
  - `WmoSkyboxSummary`
  - `WmoPortalVertexSummary`
  - `WmoPortalInfoSummary`
  - `WmoPortalRefSummary`
  - `WmoLightSummary`
  - `WmoLightDetail`
  - `WmoFogSummary`
  - `WmoOpaqueChunkSummary`
  - `WmoDoodadNameReferenceSummary`
  - `WmoGroupNameReferenceSummary`
  - `WmoDoodadSetRangeSummary`
  - `WmoVisibleVertexSummary`
  - `WmoVisibleBlockSummary`
  - `WmoVisibleBlockReferenceSummary`
  - `WmoPortalVertexRangeSummary`
  - `WmoPortalRefRangeSummary`
  - `WmoPortalGroupRangeSummary`
  - `WmoGroupLightRefSummary`
  - `WmoGroupBspNodeSummary`
  - `WmoGroupBspFaceSummary`
  - `WmoGroupBspFaceRangeSummary`
  - `WmoEmbeddedGroupSummary`
  - `WmoEmbeddedGroupLinkageSummary`
  - `WmoEmbeddedGroupDetail`
- `WowViewer.Core/Mdx`
  - `MdxChunkIds`
  - `MdxChunkSummary`
  - `MdxTextureSummary`
  - `MdxMaterialLayerSummary`
  - `MdxMaterialSummary`
  - `MdxSummary`
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
  - `WmoGroupVertexSummaryReader`
  - `WmoGroupNormalSummaryReader`
  - `WmoGroupInfoSummaryReader`
  - `WmoMaterialSummaryReader`
  - `WmoTextureTableSummaryReader`
  - `WmoDoodadNameTableSummaryReader`
  - `WmoDoodadSetSummaryReader`
  - `WmoDoodadPlacementSummaryReader`
  - `WmoGroupNameTableSummaryReader`
  - `WmoSkyboxSummaryReader`
  - `WmoPortalVertexSummaryReader`
  - `WmoPortalInfoSummaryReader`
  - `WmoPortalRefSummaryReader`
  - `WmoLightSummaryReader`
  - `WmoLightDetailReader`
  - `WmoFogSummaryReader`
  - `WmoOpaqueChunkSummaryReader`
  - `WmoRootReaderCommon`
  - `WmoDoodadNameReferenceSummaryReader`
  - `WmoGroupNameReferenceSummaryReader`
  - `WmoDoodadSetRangeSummaryReader`
  - `WmoVisibleVertexSummaryReader`
  - `WmoVisibleBlockSummaryReader`
  - `WmoVisibleBlockReferenceSummaryReader`
  - `WmoPortalVertexRangeSummaryReader`
  - `WmoPortalRefRangeSummaryReader`
  - `WmoPortalGroupRangeSummaryReader`
  - `WmoGroupLightRefSummaryReader`
  - `WmoGroupBspNodeSummaryReader`
  - `WmoGroupBspFaceSummaryReader`
  - `WmoGroupBspFaceRangeSummaryReader`
  - `WmoEmbeddedGroupSummaryReader`
  - `WmoEmbeddedGroupLinkageSummaryReader`
  - `WmoEmbeddedGroupDetailReader`
- `WowViewer.Core.IO/Mdx`
  - `MdxSummaryReader`
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
  - `mdx inspect --input <file.mdx>`
  - `mdx inspect --archive-root <dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]`
  - `map inspect --input <file.wdt|file.adt>`
  - `wmo inspect --input <file.wmo> [--dump-lights]`
  - `wmo inspect --archive-root <game|data dir> --virtual-path <world/...wmo> [--listfile <listfile.txt>] [--dump-lights]`
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
- `world/generic/activedoodads/chest01/chest01.mdx` -> `Mdx`, version `1300`

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
- shared WMO `MOLR` light-ref semantic summary for ref counts, distinct refs, duplicate refs, and ref ranges is now real
- shared WMO `MOBN` BSP-node semantic summary for node counts, leaf or branch coverage, child-reference coverage, face-count ranges, and plane-distance ranges is now real
- shared WMO `MOBR` BSP-face-ref semantic summary for ref counts, distinct refs, duplicate refs, and ref ranges is now real
- shared WMO `MOBN -> MOBR` BSP-face range coverage summary is now real
- shared WMO `MOVI` or `MOIN` index semantic summary for index counts, triangle counts, ranges, and degenerate-triangle counts is now real
- shared WMO `MOVT` vertex semantic summary for vertex counts and computed bounds is now real
- shared WMO `MONR` normal semantic summary for component ranges, length ranges, and near-unit counts is now real
- shared WMO `MOGI` root group-info semantic summary for entry counts, flag coverage, name-offset ranges, and union bounds is now real
- shared WMO `MOMT` root material semantic summary for entry counts, shader or blend-mode coverage, and selected texture offsets is now real
- shared WMO `MOTX` root texture-table semantic summary for string counts, longest-entry length, max offsets, extension coverage, and `.blp` counts is now real
- shared WMO `MODN` root doodad-name table semantic summary for string counts, longest-entry length, max offsets, extension coverage, and `.mdx` or `.m2` counts is now real
- shared WMO `MODS` root doodad-set semantic summary for set counts, non-empty sets, and doodad-ref range signals is now real
- shared WMO `MODD` root doodad-placement semantic summary for placement counts, distinct name indices, scale range, alpha range, and placement bounds is now real
- shared WMO `MOGN` root group-name table semantic summary for string counts, longest-entry length, and max offsets is now real
- shared WMO `MOSB` root skybox semantic summary for payload size and resolved skybox name is now real
- shared WMO `MOPV`, `MOPT`, and `MOPR` root portal semantic summaries for vertex counts, portal-entry counts, ref counts, and related ranges are now real
- shared WMO `MOLT` root light semantic summary for light counts, attenuation usage, intensity range, and bounds is now real
- shared WMO `MOLT` per-light detail ownership is now also real across Alpha `32`-byte and standard `48`-byte root-light layouts, including raw `headerFlagsWord`, quaternion rotation, and attenuation fields when the later layout is present
- shared WMO `MFOG` root fog semantic summary for fog counts, flag coverage, radius ranges, and bounds is now real
- shared opaque root-chunk byte reporting for `MCVP` is now real
- shared root-WMO linkage summaries for `MODD -> MODN`, `MOGI -> MOGN`, and `MODS -> MODD` are now real
- shared root-WMO visibility summaries for `MOVV`, `MOVB`, and `MOVB -> MOVV` are now real
- shared root-WMO portal-linkage summaries for `MOPT -> MOPV`, `MOPR -> MOPT`, and `MOPR -> MOGI` are now real
- shared Alpha root-WMO `MOMO` wrapper handling is now real for the shared root-summary stack
- shared Alpha monolithic root embedded-`MOGP` aggregate summary ownership is now real
- shared Alpha monolithic root embedded-`MOGP` aggregate ownership now also covers optional `lightRefs`, `bspNodes`, and `bspFaceRefs` totals, with real `castle01.wmo.MPQ` proof of `0`, `583`, and `6716`
- shared Alpha `MOGI -> MOGP(root)` linkage-summary ownership is now real
- shared Alpha monolithic root per-embedded-group inspect routing is now real for `MOGP`, `MOBN`, `MOBR`, and `MOBN -> MOBR` on `castle01.wmo.MPQ`
- shared Alpha monolithic root per-embedded-group inspect routing now also reuses the existing shared `MONR`, `MOVT`, `MOVI` or `MOIN`, `MODR`, `MOCV`, `MOTV`, `MOPY`, and `MOBA` readers directly on embedded `MOGP` payloads, with real positive `castle01.wmo.MPQ` proof for those lines
- shared Alpha monolithic root per-embedded-group inspect routing now also has positive real proof for `MOLR(root)[n]` and `MLIQ(root)[n]` on `ironforge.wmo.MPQ`
- `WowViewer.Tool.Inspect wmo inspect` still treats invalid optional `MOLT` root-summary reads as non-fatal, but that guard is now a malformed-payload fallback rather than an Ironforge compatibility workaround
- shared Alpha root `MOLT` semantic-summary ownership is now also proven directly on real `ironforge.wmo.MPQ`, with `6976` payload bytes, `218` lights, and a positive `attenStartRange` reported through the shared reader
- shared standard `v16` root `MOLT` semantic-summary ownership is now also proven on `world/wmo/khazmodan/cities/ironforge/ironforge.wmo` loaded from the `0.6.0` MPQ set through `MpqArchiveCatalog` + `wow-listfile`, fixing the real 48-byte layout to a non-zero `headerFlagsWord` of `0x0101` at bytes `2..3`, quaternion rotation at offsets `24..39`, and attenuation at offsets `40` and `44`
- `WowViewer.Tool.Inspect wmo inspect` now also consumes that same shared standard-archive seam directly through `--archive-root` plus `--virtual-path`, with default vendored-listfile discovery when the repo-local `wow-listfile` is present
- shared MD5 minimap translation and minimap tile path resolution are now real
- shared standard-archive read and DBC or DB2 table probing boundaries are now real
- shared `BLP` header-summary ownership is now real for `BLP1` and `BLP2`, including compression fields, pixel format, image dimensions, palette or JPEG-header presence, and per-mip offset or size coverage
- shared `MDX` top-level summary ownership is now real for `MDLX` signature validation, top-level chunk order or count, known-vs-unknown chunk coverage, `VERS`, narrow `MODL` name or bounds or blend-time fields, shared `TEXS` texture-table summary coverage for replaceable ids, paths, and flags, and narrow `MTLS` material-layer summary coverage for priority plane, blend mode, flags, texture id, transform id, coord id, and static alpha
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
  - deep WMO, `MDX`, `M2`, or `BLP` payload parsing beyond the new `BLP` and `MDX` top-level-plus-`TEXS` summary seams
  - general-purpose DBC or DB2 schema ownership beyond the narrow shared lookup helpers that now exist
  - any write path or round-trip support
  - runtime cutover inside the active viewer

## Immediate Next Slices

1. deepen shared ADT root and split-file top-level summaries
2. deepen shared ADT root and split-file summaries beyond the new semantic-summary layer into narrower chunk-internal signals only when a deep payload proof target is clear
3. deepen shared WMO ownership from root, group, `MOGI`, `MOGN`, `MOSB`, `MOMT`, `MOTX`, `MODN`, `MODS`, `MODD`, `MOPV`, `MOPT`, `MOPR`, `MOLT`, `MFOG`, `MCVP`, plus the first linkage summaries for `MODD -> MODN`, `MOGI -> MOGN`, and `MODS -> MODD`, into the next narrow deep-payload seam only when a fixed validation target is available
4. start the first shared `M2` seam or a deeper but still narrow `MDX` chunk-internal seam such as `MTLS` only when a fixed real-data target is clear
  the current clean WMO continuation after the landed `MOLT` detail seam is wider real-data proof for standard `headerFlagsWord` variability across additional roots
4. keep Alpha root-WMO `MOMO` support aligned with real 0.5.3 per-asset archive validation as additional root summaries land
5. keep inspect and converter as thin consumers of shared seams instead of adding direct parsing in tool entrypoints
6. continue shrinking `MdxViewer` imports of `WoWMapConverter.Core` by moving other narrow non-MPQ helpers onto `Core` or `Core.IO`
7. follow the landed first shared model or texture-family seam with the next highest-value missing family, most likely `M2` or `MDX` header-summary ownership
8. decide whether the next non-PM4 slice after that is higher-level CASC or MPQ unification, another new shared format family, or a deeper summary or reader slice for existing families

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
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `98` tests after adding the shared WMO `MOVT` vertex semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `99` tests after adding the shared WMO `MONR` normal semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `101` tests after adding the shared WMO `MOGI` root group-info semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `103` tests after adding the shared WMO `MOMT` root material semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `104` tests after adding the shared WMO `MOTX` root texture-table semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `105` tests after adding the shared WMO `MODN` root doodad-name-table semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `106` tests after adding the shared WMO `MODS` root doodad-set semantic-summary slice
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `109` tests after batching the shared root-WMO `MODD`, `MOGN`, and `MOSB` semantic-summary slices
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `112` tests after batching the shared root-WMO `MOPV`, `MOPT`, and `MOPR` semantic-summary slices
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `115` tests after batching the shared root-WMO `MOLT`, `MFOG`, and `MCVP` semantic-summary slices
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `118` tests after batching the first root-WMO linkage summary slices for `MODD -> MODN`, `MOGI -> MOGN`, and `MODS -> MODD`
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `121` tests after batching the shared root-WMO `MOVV`, `MOVB`, and `MOVB -> MOVV` visibility-summary slices
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `125` tests after batching the shared root-WMO portal-linkage summaries for `MOPT -> MOPV`, `MOPR -> MOPT`, and `MOPR -> MOGI`
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `128` tests after adding Alpha `MOMO` root-WMO support and real-data `castle01.wmo.MPQ` coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `129` tests after adding Alpha monolithic embedded-group aggregate coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `130` tests after adding Alpha `MOGI -> MOGP(root)` linkage coverage
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
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `67` tests after adding synthetic WMO `MOVT` vertex semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `68` tests after adding synthetic WMO `MONR` normal semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `70` tests after adding synthetic standard and legacy `MOGI` root group-info semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `72` tests after adding synthetic standard and legacy `MOMT` material semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `73` tests after adding synthetic `MOTX` texture-table semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `74` tests after adding synthetic `MODN` doodad-name-table semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `75` tests after adding synthetic `MODS` doodad-set semantic-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `78` tests after batching synthetic `MODD`, `MOGN`, and `MOSB` root-WMO coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `81` tests after batching synthetic `MOPV`, `MOPT`, and `MOPR` root-WMO coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `84` tests after batching synthetic `MOLT`, `MFOG`, and `MCVP` root-WMO coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `87` tests after batching synthetic `MODD -> MODN`, `MOGI -> MOGN`, and `MODS -> MODD` linkage-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `90` tests after batching synthetic `MOVV`, `MOVB`, and `MOVB -> MOVV` visibility-summary coverage
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `94` tests after batching synthetic `MOPT -> MOPV`, `MOPR -> MOPT`, and `MOPR -> MOGI` portal-linkage coverage plus the missing-`MOVV` regression
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
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-vertex-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOVT` semantic summary `payloadBytes=36 vertices=3 boundsMin=(-4.00, -8.00, -6.00) boundsMax=(7.00, 5.00, 9.00)`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-normal-test.wmo` passed on Mar 27, 2026 and now reports the shared `MONR` semantic summary `payloadBytes=36 normals=3 rangeX=[0.000, 1.000] rangeY=[-1.000, 0.500] rangeZ=[0.000, 0.500] lengthRange=[0.866, 1.000] avgLength=0.955 nearUnit=2`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mogi-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOGI` semantic summary `payloadBytes=64 entryBytes=32 entries=2 distinctFlags=2 nonZeroFlags=1 nameOffsetRange=12-40 boundsMin=(-7.00, -2.00, -3.00) boundsMax=(4.00, 8.00, 9.00)`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-momt-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOMT` semantic summary `payloadBytes=128 entryBytes=64 entries=2 distinctShaders=2 distinctBlendModes=2 nonZeroFlags=1 maxTex1Ofs=24 maxTex2Ofs=20 maxTex3Ofs=88`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-motx-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOTX` semantic summary `payloadBytes=33 textures=3 longestEntry=16 maxOffset=16 extensions=2 blpEntries=2`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-modn-test.wmo` passed on Mar 27, 2026 and now reports the shared `MODN` semantic summary `payloadBytes=31 names=3 longestEntry=15 maxOffset=15 extensions=2 mdxEntries=2 m2Entries=1`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mods-test.wmo` passed on Mar 27, 2026 and now reports the shared `MODS` semantic summary `payloadBytes=96 entries=3 nonEmptySets=2 longestName=7 totalDoodadRefs=10 maxStartIndex=12 maxRangeEnd=18`
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-batch-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOSB`, `MOGN`, and `MODD` semantic summaries in one batched root-WMO smoke case
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portals-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOPV`, `MOPT`, and `MOPR` semantic summaries in one batched root-WMO smoke case
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-meta-batch-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOLT`, `MFOG`, and `MCVP` semantic summaries in one batched root-WMO smoke case
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-linkage-batch-test.wmo` passed on Mar 27, 2026 and now reports the shared `MODD -> MODN`, `MOGI -> MOGN`, and `MODS -> MODD` linkage summaries in one batched root-WMO smoke case
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-visibility-batch-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOVV`, `MOVB`, and `MOVB -> MOVV` visibility summaries in one batched root-WMO smoke case
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portal-linkage-batch-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOPT -> MOPV`, `MOPR -> MOPT`, and `MOPR -> MOGI` portal-linkage summaries in one batched root-WMO smoke case
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports real Alpha-era root-WMO semantic and linkage lines directly from the per-asset MPQ
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` now also reports the Alpha monolithic embedded-group aggregate `MOGP(root)` line
- the same real-data inspect path now also reports the Alpha `MOGI -> MOGP(root)` linkage line with paired flag/bounds comparison metrics
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portals-test.wmo` passed on Mar 27, 2026 and now reports the shared `MOPV`, `MOPT`, and `MOPR` semantic summaries in one batched root-WMO smoke case
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- detect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 26, 2026
- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert i:/parp/parp-tools/gillijimproject_refactor/test_data/0.5.3/alphawdt/World/Maps/PVPZone01/PVPZone01.wdt -o i:/parp/parp-tools/output/pvpzone01-alpha-to-lk-smoke-dbcd-check3 -v` passed on Mar 27, 2026 and confirmed the new explicit warning now names the preferred `gillijimproject_refactor/test_data/0.5.3/tree` and `gillijimproject_refactor/test_data/3.3.5/tree` `AreaTable` and `Map` roots first when extracted files are missing
- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 27, 2026 after switching the converter to lazy archive-backed `AreaIdMapper` initialization and adding `--alpha-client` plus `--lk-client`
