# wow-viewer

Initial repository skeleton for the planned production split from parp-tools.

Current first-pass project layout:

- `src/viewer/WowViewer.App`
- `src/core/WowViewer.Core`
- `src/core/WowViewer.Core.IO`
- `src/core/WowViewer.Core.Runtime`
- `src/core/WowViewer.Core.PM4`
- `src/tools-shared/WowViewer.Tools.Shared`
- `tools/converter/WowViewer.Tool.Converter`
- `tools/inspect/WowViewer.Tool.Inspect`

This scaffold is intentionally minimal. It exists to lock the repo shape, project identities, and reference graph before the real code-port work starts.

Current plan-adherence reality:

- `Core.PM4` is the only core library area with substantial implementation today.
- `WowViewer.Core`, `WowViewer.Core.IO`, and `WowViewer.Core.Runtime` are still early and should not be described as complete library boundaries yet.
- `WowViewer.Core.Runtime` now owns a first narrow world-render seam: shared render-frame telemetry contracts and optimization-hint logic extracted from `MdxViewer.WorldScene`.
- staged `WorldScene` to `wow-viewer` runtime decomposition is now documented in `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md` and `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`, with repeated `.skin` miss suppression called out as slice 01 before deeper pass extraction.
- The repo is now starting to correct that with a real bootstrap script and a first non-PM4 chunk or FourCC foundation slice, but the broader shared I/O and runtime cutover is still missing.

Current implementation policy:

- `WowViewer.Core.PM4`, `WowViewer.Core`, and `WowViewer.Core.IO` are the canonical implementation targets for new `wow-viewer` work.
- M2 runtime, skin-profile ownership, model lighting, shader or effect routing, and model-render performance work are canonical `wow-viewer` work; the first dedicated M2 library/runtime foundation now lives under `src/core/WowViewer.Core/M2`, `src/core/WowViewer.Core.IO/M2`, and `src/core/WowViewer.Core.Runtime/M2`.
- `gillijimproject_refactor`, including `MdxViewer` and `WoWMapConverter`, is now a reference or compatibility input for `wow-viewer` work, not the default owner of the design.
- Default validation for `wow-viewer` work is `dotnet build .\WowViewer.slnx -c Debug`, `dotnet test .\WowViewer.slnx -c Debug`, and the relevant inspect or converter command against the fixed development dataset.
- Build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln` only when a slice explicitly changes consumer compatibility or the user asks for that check.
- The explicit long-range target is full first-party ownership of every active format family currently handled by `MdxViewer`; current detector and summary seams are stepping stones, not the final boundary.

Current M2-native continuity note:

- The canonical implementation-facing M2 documentation set now lives under `docs/architecture/m2/`.
	- start with `docs/architecture/m2/README.md`
	- then use `implementation-contract.md`, `native-build-matrix.md`, and `consumer-cutover.md`
- Raw native evidence still lives in `docs/architecture/m2-native-client-research-2026-03-31.md`.
- Treat the consolidated `docs/architecture/m2/` folder as the first handoff for future `wow-viewer` M2 parser, runtime, lighting, shader, and performance work.
- Do not keep growing `MdxViewer` as the design owner for those M2 seams unless the task is explicitly compatibility-only.
- The staged workflow surface for this work now lives in `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md` and `.github/prompts/wow-viewer-m2-runtime/`.
- Cross-build recovery from Wrath through 6.x now routes through `.github/prompts/m2-cross-build-native-investigation.prompt.md`.
- The matching continuity plan lives in `../gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`.

Current PM4 terminology policy:

- Treat wowdev `PM4` and `PD4` docs as the source of truth for raw chunk and field names when they actually name a field.
- When the docs only expose placeholders such as `MSUR._0x02` or `MSUR._0x1c`, use the raw offset-style name first and local aliases second.
- Current local PM4 aliases such as `MSUR.AttributeMask`, `MSUR.GroupKey`, `MSUR.MdosIndex`, `MSUR.PackedParams`, derived `CK24`, and `MSLK.GroupObjectId` are research names, not original format terminology.
- Current stronger-than-doc corrections still worth preserving are:
	- `MSUR` bytes `0x04..0x0f` behave like real surface normals
	- the current `MSUR.Height` property name is misleading because float `0x10` behaves like a signed plane-distance term
	- `MSLK.RefIndex` is not closed as a universal `MSUR` index across the current corpus

Current PM4 research notes:

- On the fixed development-map workflow, derived `CK24` low-16 object values continue to separate many `WMO`-like PM4 meshes into narrower object families; treat that as strong research evidence, not final format truth.
- The recurrent `CK24=0x000000` bucket still behaves like an unresolved umbrella or root family rather than a clean single asset class. Current viewer evidence suggests many `M2`-like placements still live there.
- The expanded shared `pm4 unknowns` report now also exposes dominant `MSLK` and `MSUR` families across the fixed corpus. Current evidence says the biggest `group=3` `MSUR` families are overwhelmingly zero-`CK24`, while several large `group=18` families carry broad non-zero `CK24` and `MDOS` fanout and are better candidates for object-facing attribution work.
- The same report shows dominant `MSLK` families clustering in a small repeated set of `TypeFlags` or `Subtype` combinations with sentinel-tile `LinkId` dominance. That is useful prioritization evidence, not a final semantic decode of those fields.
- Current `MdxViewer` graph and hover-match surfaces are evidence-gathering tools only. They help rank likely asset candidates, but they do not prove that current `CK24`, `MSLK.GroupObjectId`, or subobject ownership semantics are closed.
- If later PM4 correlation work needs broader corpus support, a feature-indexed asset database or ML-assisted clustering pass may help rank candidates, but it should follow parser or linkage improvements instead of replacing them.

Bootstrap dependencies:

- `scripts/bootstrap.ps1` now clones the baseline upstream repos described in the migration draft into `libs/`:
	- `wowdev/wow-listfile`
	- `wowdev/WoWDBDefs`
	- `wowdev/DBCD`
	- `ModernWoWTools/Warcraft.NET`
	- `Marlamin/WoWTools.Minimaps`
	- `WoW-Tools/SereniaBLPLib`
- Optional evaluation repos can also be pulled with:
	- PowerShell: `./scripts/bootstrap.ps1 -IncludeOptional`
	- Bash: `./scripts/bootstrap.sh --include-optional`
- The active shared `AreaTable` and `Map` seam now uses the same vendored DBCD project the viewer already consumes from `gillijimproject_refactor/lib/wow.tools.local/DBCD`, and bundles `gillijimproject_refactor/lib/WoWDBDefs/definitions` into the `WowViewer.Core.IO` output.
- The shared `AreaIdMapper` seam now also supports archive-backed `AreaTable` and `Map` loads through `IArchiveReader` plus `DbClientFileReader`, so consumers do not need fake extracted DBC trees just to get DBCD + WoWDBDefs-backed mapping.
- Extracted fixed-data tables under `gillijimproject_refactor/test_data/0.5.3/tree/DBFilesClient` or `gillijimproject_refactor/test_data/3.3.5/tree/DBFilesClient` remain a narrow fallback or test path; when neither archive-backed nor explicit table inputs are provided, the current `AreaIdMapper` path warns explicitly and falls back to crosswalk-only behavior instead of silently pretending schema-backed loading happened.
- `WowViewer.Core.IO` no longer requires a bundled `Resources/area_crosswalk.csv` payload to compile or ship. The intended production path is archive-backed or explicit DBC-driven mapping from user-supplied data, with user-provided CSV crosswalks staying optional rather than release payloads.

Current shared-core foundation slice:

- `src/core/WowViewer.Core` now contains the first non-PM4 chunk primitives:
	- `FourCC`
	- `ChunkHeader`
- `src/core/WowViewer.Core` now also contains the first non-PM4 map-format constants and summary contracts:
  - `MapChunkIds`
  - `MapFileKind`
  - `MapChunkLocation`
  - `MapFileSummary`
  - `AdtSummary`
  - `AdtChunkIds`
  - `AdtMcnkSummary`
	- `AdtTileFamily`
	- `AdtTextureLayerDescriptor`
	- `AdtMcalDecodeProfile`
	- `AdtMcalAlphaEncoding`
	- `AdtMcalDecodedLayer`
	- `AdtMcalSummary`
	- `AdtTextureChunkLayer`
	- `AdtTextureChunk`
	- `AdtTextureFile`
  - `WdtSummary`
- `src/core/WowViewer.Core/Wmo` now contains the first shared WMO root-summary contracts:
  - `WmoChunkIds`
	- `WmoGroupFlags`
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
   - Alpha root `MOMO` wrapper handling is now also part of the shared root-summary ownership boundary
	- `WmoGroupFlags` currently provides the first typed interpretation layer for evidence-backed `MOGP` flag bits that gate BSP or exterior or vertex-color or exterior-lighting or light-ref or doodad-ref or liquid or extra-UV families; residual bits such as `0x00000002` remain intentionally raw until the corpus says more
	- `WmoSummary` now also exposes root `MOSB` presence as `HasSkybox`, so root summary consumers can tell whether the file advertises an explicit skybox without re-running the dedicated skybox reader first
- `src/core/WowViewer.Core/Mdx` now contains the first shared `MDX` top-level summary contracts:
	- `MdxChunkIds`
	- `MdxChunkSummary`
	- `MdxGeometryFile`
	- `MdxGeosetGeometry`
	- `MdxGlobalSequenceSummary`
	- `MdxSequenceSummary`
	- `MdxGeosetSummary`
	- `MdxGeosetAnimationSummary`
	- `MdxGeosetAnimationTrackSummary`
	- `MdxBoneSummary`
	- `MdxLightType`
	- `MdxLightSummary`
	- `MdxHelperSummary`
	- `MdxAttachmentSummary`
	- `MdxParticleEmitter2Summary`
	- `MdxRibbonEmitterSummary`
	- `MdxNodeTrackSummary`
	- `MdxTrackSummary`
	- `MdxVisibilityTrackSummary`
	- `MdxPivotPointSummary`
	- `MdxTextureSummary`
	- `MdxMaterialLayerSummary`
	- `MdxMaterialSummary`
	- `MdxSummary`
- `src/core/WowViewer.Core.IO` now contains the first non-PM4 I/O seam:
	- `ChunkHeaderReader`
- `src/core/WowViewer.Core.IO` now also contains the first shared WDT or ADT top-level reader slice:
  - `ChunkedFileReader`
  - `MapFileSummaryReader`
  - `AdtSummaryReader`
  - `AdtMcnkSummaryReader`
	- `AdtTileFamilyResolver`
	- `AdtMcalDecoder`
	- `AdtMcalSummaryReader`
	- `AdtTextureReader`
  - `WdtSummaryReader`
- `src/core/WowViewer.Core.IO/Wmo` now contains the first shared WMO root-summary reader:
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

Current WMO inspect workflow note:

- `wowviewer-inspect wmo inspect` now also supports `--flag-correlation` for root WMOs.
- This is a per-file evidence surface that correlates the `MOGP` bits seen in the current file against actual group signals such as BSP payloads, doodad refs, light refs, liquid, vertex colors, and extra UV sets.
- Treat it as an audit/ranking surface, not final runtime semantics for every unknown `MOGP` bit.
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
   - `WmoRootReaderCommon` now also flattens Alpha `MOMO` root subchunks for shared root-summary readers
- `src/core/WowViewer.Core.IO/Mdx` now contains the first shared `MDX` top-level summary reader:
	- `MdxGeometryReader`
	- `MdxSummaryReader`
- `src/core/WowViewer.Core.IO` now also contains the first shared minimap translation or path helpers:
	- `Md5TranslateIndex`
	- `Md5TranslateResolver`
	- `MinimapService`
- `src/core/WowViewer.Core.IO` now also contains the first shared archive-access contracts and DBC path helper:
	- `IArchiveReader`
	- `IArchiveCatalog`
	- `IArchiveCatalogFactory`
	- `DbClientFileReader`
- `src/core/WowViewer.Core.IO` now also contains the first shared archive bootstrap and Alpha wrapper helpers:
	- `ArchiveCatalogBootstrapper`
	- `ArchiveCatalogBootstrapResult`
	- `ArchiveListfileCache`
	- `ArchiveListfileCacheManifest`
	- `AlphaArchiveReader`
	- `PkwareExplode`
- `src/core/WowViewer.Core.IO` now also contains the concrete shared standard MPQ implementation used by the active consumer path:
	- `MpqArchiveCatalog`
	- `MpqArchiveCatalogFactory`
- `src/core/WowViewer.Core.IO` now also contains the first narrow shared DBC-backed lookup slice:
	- `DbcReader`
	- `DbcHeader`
	- `MapDirectoryLookup`
	- `GroundEffectLookup`
	- `AreaIdMapper`
	- `AreaIdMapper` now also prefers DBCD + WoWDBDefs for known `0.5.3` and `3.3.5` `AreaTable` and `Map` inputs from either shared archive readers or explicit file paths, keeping the raw `DbcReader` only as a narrow fallback
- `src/core/WowViewer.Core` now also contains the first cross-family file detection contracts:
	- `WowFileKind`
	- `WowFileDetection`
- `src/core/WowViewer.Core.IO` now also contains the first shared cross-family detector:
	- `WowFileDetector`
- `tests/WowViewer.Core.Tests` now locks the current FourCC and chunk-header boundary behavior.
	- it now also locks synthetic `BLP1` and `BLP2` header-summary behavior plus archive-backed real `0.6.0` standard-MPQ `BLP` coverage through the shared archive catalog
	- it now also locks synthetic `GLBS`, `GEOA`, `BONE`, `LITE`, `HELP`, `ATCH`, `PRE2`, `RIBB`, `CAMS`, `EVTS`, `HTST`, and `CLID` summary behavior plus fixed real `MDX` coverage on files such as `testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` and archive-backed `0.6.0` `world/generic/dwarf/passive doodads/braziers/dwarvenbrazier01.mdx`; the focused reader suite also smoke-parses the current unpacked `0.5.3` alpha corpus (`229` MDX files) to verify the new `LITE` path does not regress alpha-era parsing even though that bundled sample set currently contains no `LITE` chunks
	- it now also locks synthetic classic `GEOS` payload behavior plus real shared geometry coverage on the local standard-era archive dataset and the existing on-disk alpha-era creature corpus through `MdxGeometryReaderTests`
	- it now also locks synthetic and real-data WDT or ADT summary behavior against `development.wdt` and `development_0_0.adt`
	- it now also locks synthetic and real-data ADT semantic-summary behavior for `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
	- it now also locks shared ADT `MCNK` semantic-summary behavior for synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
	- it now also locks shared root-plus-`_tex0` texture-layer and decoded-alpha behavior for synthetic root and synthetic `_tex0` fixtures plus real-data `development_0_0.adt` and `development_0_0_tex0.adt`
	- it now also locks a synthetic WMO root semantic-summary case
	- it now also locks synthetic WMO group semantic-summary behavior for both `MVER + MOGP` and `MOGP`-first files, plus `MOGP`-first detector coverage
	- it now also locks synthetic WMO `MLIQ` semantic-summary behavior including height-range and ocean-inference coverage
	- it now also locks synthetic WMO `MOBA` batch semantic-summary behavior including v17-style material ids and v16-style material-less batches
	- it now also locks synthetic WMO `MOPY` face-material semantic-summary behavior including v17 two-byte and v16 four-byte entry layouts
	- it now also locks synthetic WMO `MOTV` UV semantic-summary behavior including one primary and one extra UV set
	- it now also locks synthetic WMO `MOCV` vertex-color semantic-summary behavior including one primary and one extra color set
	- it now also locks synthetic WMO `MODR` doodad-ref semantic-summary behavior including duplicate-ref coverage
	- it now also locks synthetic WMO `MOVI` and `MOIN` index semantic-summary behavior including a degenerate-triangle case
	- it now also locks synthetic WMO `MOVT` vertex semantic-summary behavior including computed bounds
	- it now also locks synthetic WMO `MONR` normal semantic-summary behavior including component-range and near-unit coverage
	- it now also locks synthetic root WMO `MOGI` semantic-summary behavior including standard and legacy entry layouts
	- it now also locks synthetic root WMO `MOMT` semantic-summary behavior including standard and legacy entry layouts
	- it now also locks synthetic root WMO `MOTX` texture-table semantic-summary behavior including mixed extensions and nested paths
	- it now also locks synthetic root WMO `MODN` doodad-name-table semantic-summary behavior including mixed `.mdx` or `.m2` names and nested paths
	- it now also locks synthetic root WMO `MODS` doodad-set semantic-summary behavior including empty and non-empty sets
	- it now also locks batched synthetic root-WMO cases for `MODD`, `MOGN`, `MOSB`, `MOPV`, `MOPT`, and `MOPR`
	- it now also locks a batched synthetic root-WMO case for `MOLT`, `MFOG`, and `MCVP`
	- it now also locks synthetic `MOLT` per-light detail behavior for Alpha and standard root layouts plus real Ironforge Alpha and `0.6.0` standard root-light detail coverage
	- it now also locks a batched synthetic root-WMO linkage case for `MODD -> MODN`, `MOGI -> MOGN`, and `MODS -> MODD`
	- it now also locks a batched synthetic root-WMO visibility case for `MOVV`, `MOVB`, and `MOVB -> MOVV`
	- it now also locks a batched synthetic root-WMO portal-linkage case for `MOPT -> MOPV`, `MOPR -> MOPT`, and `MOPR -> MOGI`, plus a missing-`MOVV` regression
	- it now also locks synthetic WMO group optional-chunk cases for `MOLR`, `MOBN`, `MOBR`, and `MOBN -> MOBR`
	- it now also locks real 0.5.3 Alpha per-asset WMO validation via `testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ`
	- it now also locks a synthetic Alpha monolithic embedded-`MOGP` aggregate case plus real aggregate checks on `castle01.wmo.MPQ`
	- it now also locks a synthetic Alpha `MOGI -> MOGP(root)` linkage case plus real linkage checks on `castle01.wmo.MPQ`
	- it now also locks real `castle01.wmo.MPQ` embedded-group optional-chunk totals and replays the real embedded groups through the shared `MOBN`, `MOBR`, and `MOBN -> MOBR` readers
	- it now also locks synthetic Alpha and standard WDT semantic-summary behavior plus real-data `development.wdt` occupancy and MPHD signals
	- it now also locks shared file detection for `development.wdt`, `development_0_0.adt`, `development_0_0_tex0.adt`, `development_0_0_obj0.adt`, and `development_00_00.pm4`

Current non-PM4 inspect slice:

- `tools/inspect/WowViewer.Tool.Inspect` now also supports:
	- `archive build-listfile-cache --archive-root <game|data dir> --cache-key <client.version.build> [--listfile <listfile.txt>] [--cache-dir <directory>]`
	- `blp inspect --input <file.blp>`
	- `blp inspect --archive-root <game|data dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]`
	- `m2 inspect --input <file.m2|file.mdx|file.mdl> [--profile-index <n>]`
	- `m2 inspect --archive-root <game|data dir> --virtual-path <path/to/file.m2|file.mdx|file.mdl> [--listfile <listfile.txt>] [--profile-index <n>]`
	- `mdx inspect --input <file.mdx>`
	- `mdx inspect --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]`
	- `mdx export-json --input <file.mdx> [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]`
	- `mdx export-json --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>] [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]`
	- `mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --input <file|directory> [--path-filter <text>] [--limit <n>]`
	- `mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --archive-root <game|data dir> [--listfile <listfile.txt>] [--path-filter <text>] [--limit <n>]`
	- `map inspect --input <file.wdt|file.adt> [--dump-tex-chunks]`
	- `map uniqueid-report --input <file.wdt|file.adt|directory> [--build <label>] [--output <report.json>]`
	- `wmo inspect --input <file.wmo> [--dump-lights]`
	- `wmo inspect --archive-root <game|data dir> --virtual-path <world/...wmo> [--listfile <listfile.txt>] [--dump-lights]`
- This is intentionally narrow for now:
	- it now also reports a first shared M2 foundation summary for canonicalized model identity, strict `MD20` root acceptance, typed bounds/name metadata, exact numbered `%02d.skin` profile selection, staged choose/load/initialize runtime state, and strict external `SKIN` table counts when a model and matching sidecar are available
	- it now also reports a first shared `BLP` header summary for format signature, version, compression fields, pixel format, image size, palette or JPEG-header presence, and per-mip offset or size coverage when a texture file is inspected
	- it now also reports a first shared `MDX` top-level summary for `MDLX` signature, chunk order, known-vs-unknown chunk coverage, `VERS`, narrow `MODL` name or bounds or blend-time signals, shared `GLBS` global-sequence signals, shared `SEQS` sequence signals, shared classic `GEOS` geoset signals, shared classic `GEOA` geoset-animation signals, shared classic `BONE` skeleton signals, shared classic `HELP` node signals, shared classic `ATCH` attachment signals, shared classic `PRE2` particle-emitter signals, shared classic `RIBB` ribbon-emitter signals, shared classic `CAMS` camera signals, shared classic `EVTS` event-node signals, shared classic `HTST` hit-test-shape signals, shared classic `CLID` collision-mesh signals, shared `PIVT` pivot-point signals, shared `TEXS` texture-table paths or flags, and narrow `MTLS` material-layer signals when a model file is inspected
	- `mdx export-json` now remains a thin shared-reader export surface and can include the shared `GEOS` payload seam through `--include-geometry`, the shared `CLID` payload seam through `--include-collision`, the shared `HTST` payload seam through `--include-hit-test`, and the shared `TXAN` payload seam through `--include-texture-animations`
	- it reads top-level chunk order, counts, version, and file-kind classification for WDT and ADT-family files
	- it now also reports a shared ADT semantic summary for terrain-chunk counts, string-table counts, placement counts, and selected MFBO or MH2O or MAMP or MTXF presence across root, `_tex0.adt`, and `_obj0.adt`
	- it now also reports a shared ADT `MCNK` semantic summary for root-header coverage, selected flags, split-file subchunk presence, and per-chunk layer-count signals across root, `_tex0.adt`, and `_obj0.adt`
	- it now also reports shared root-plus-`_tex0` per-chunk layer and decoded alpha detail on demand through `--dump-tex-chunks`
	- it now also supports `map uniqueid-report` as the first per-build placement manifest workflow for `MDDF` and `MODF` `UniqueId` collection, emitting JSON reports that can be diffed later into added/removed-object timelines
	- it now also reports a shared WDT semantic summary for MPHD WMO-based flags, MAIN tile occupancy, string-table counts, and top-level MDDF or MODF placement counts
	- it now also reports a first shared WMO root semantic summary for `MOHD`-reported counts, top-level entry counts, string-table counts, flags, and bounds
	- it now also reports a first shared WMO group semantic summary for `MOGP` header fields, geometry subchunk counts, optional extra UV-set count, doodad-ref count, and liquid presence
	- it now also reports a shared WMO `MLIQ` semantic summary for liquid dimensions, height range, visible-tile counts, and basic family inference when a group file contains liquid
	- it now also reports a shared WMO `MOBA` batch semantic summary for batch-entry counts, material-id coverage, index coverage, and flagged-batch counts when a group file contains batches
	- it now also reports a shared WMO `MOPY` face-material semantic summary for face counts, hidden-face counts, flag coverage, and material-id coverage when a group file contains face-material entries
	- it now also reports a shared WMO `MOTV` UV semantic summary for primary UV ranges and extra-set counts when a group file contains UV data
	- it now also reports a shared WMO `MOCV` vertex-color semantic summary for BGRA channel ranges, average alpha, and extra-set counts when a group file contains vertex colors
	- it now also reports a shared WMO `MODR` doodad-ref semantic summary for ref counts, distinct refs, duplicate refs, and ref range when a group file contains doodad refs
	- it now also reports shared WMO group `MOLR`, `MOBN`, `MOBR`, and `MOBN -> MOBR` summaries when those optional chunks are present
	- it now also reports a shared WMO `MOVI` or `MOIN` index semantic summary for index counts, triangle counts, ranges, and degenerate-triangle counts when a group file contains indices
	- it now also reports a shared WMO `MOVT` vertex semantic summary for vertex counts and computed bounds when a group file contains vertex payloads
	- it now also reports a shared WMO `MONR` normal semantic summary for component ranges, length ranges, and near-unit counts when a group file contains normal payloads
	- it now also reports a shared root WMO `MOGI` semantic summary for entry counts, flag coverage, name-offset ranges, and union bounds when group info is present
	- it now also reports a shared root WMO `MOMT` semantic summary for entry counts, shader or blend coverage, and selected texture offsets when material entries are present
	- it now also reports a shared root WMO `MOTX` semantic summary for string counts, longest-entry length, max offsets, extension coverage, and `.blp` counts when a texture table is present
	- it now also reports a shared root WMO `MODN` semantic summary for string counts, longest-entry length, max offsets, extension coverage, and `.mdx` or `.m2` counts when a doodad-name table is present
	- it now also reports a shared root WMO `MODS` semantic summary for set counts, non-empty sets, and doodad-ref range signals when doodad sets are present
	- it now also reports shared root-WMO `MODD`, `MOGN`, and `MOSB` semantic summaries for doodad placements, group-name tables, and skybox name ownership when those chunks are present
	- it now also reports shared root-WMO `MOPV`, `MOPT`, and `MOPR` semantic summaries for portal vertices, portal entries, and portal refs when those chunks are present
	- it now also reports shared root-WMO `MOLT`, `MFOG`, and `MCVP` semantic summaries for lights, fog, and opaque trailing root chunks when those chunks are present
	- it now also prints opt-in per-entry `MOLT[n]` lines when `--dump-lights` is provided, reusing the shared `WmoLightDetailReader` seam for Alpha `32`-byte and later `48`-byte root-light layouts
	- it now also reports shared root-WMO linkage summaries for `MODD -> MODN`, `MOGI -> MOGN`, and `MODS -> MODD` when those related chunks are present
	- it now also reports shared root-WMO visibility summaries for `MOVV`, `MOVB`, and `MOVB -> MOVV` when those chunks are present
	- it now also reports shared root-WMO portal-linkage summaries for `MOPT -> MOPV`, `MOPR -> MOPT`, and `MOPR -> MOGI` when those related chunks are present
	- it now also accepts Alpha per-asset `.wmo.MPQ` inputs for shared root-WMO inspect by routing them through the shared archive fallback and the Alpha `MOMO`-aware root reader path
	- it now also accepts standard shared-MPQ archive roots plus virtual WMO paths, defaulting to the vendored `wow-listfile` when that file is available under the repo root
	- it now also supports `archive build-listfile-cache` as the first shared per-client known-file cache workflow for MPQ-era roots, with internal MPQ listfiles treated as the trusted primary source and any supplied external/community listfile persisted as supplemental gap-fill input
	- archive-backed `mdx chunk-carriers` now enumerates the shared bootstrap `AllFiles` universe instead of only the catalog's pre-bootstrap known-file set, so trusted internal entries and cached supplemental entries actually affect carrier discovery
	- it now also reports an Alpha monolithic embedded-group aggregate `MOGP(root)` line when a root WMO contains top-level embedded `MOGP` blocks
	- that same `MOGP(root)` aggregate now also reports embedded `lightRefs`, `bspNodes`, and `bspFaceRefs`, which real `castle01.wmo.MPQ` currently proves as `0`, `583`, and `6716`
	- it now also reports an Alpha `MOGI -> MOGP(root)` linkage line showing paired count/flag/bounds metrics across root group-info and embedded-group surfaces
	- it now also reports real per-embedded-group `MOGP(root)[n]`, `MONR(root)[n]`, `MOVT(root)[n]`, `MOVI(root)[n]` or `MOIN(root)[n]`, `MODR(root)[n]`, `MOCV(root)[n]`, `MOTV(root)[n]`, `MOPY(root)[n]`, `MOBA(root)[n]`, `MOBN(root)[n]`, `MOBR(root)[n]`, and `MOBN->MOBR(root)[n]` lines for Alpha monolithic roots such as `castle01.wmo.MPQ`, with `MOLR(root)[n]` or `MLIQ(root)[n]` emitted when present
	- real `ironforge.wmo.MPQ` now provides positive per-group proof for both `MOLR(root)[n]` and `MLIQ(root)[n]`
	- shared root `MOLT` semantic summary now also reads real Alpha `ironforge.wmo.MPQ`, reporting `payloadBytes=6976`, `entries=218`, and `attenStartRange=[1.306, 8.333]`
	- a real standard `0.6.0` Ironforge root loaded through shared `MpqArchiveCatalog` + `wow-listfile` now also proves that 48-byte `MOLT` entries store quaternion rotation at offsets `24..39` and attenuation at offsets `40` and `44`
	- that same real standard root also proves bytes `2..3` are a non-zero `headerFlagsWord`, not padding; Ironforge currently reports `headerFlagsWordRange=[0x0101, 0x0101]`, `headerFlagsWordDistinct=1`, and `headerFlagsWordNonZero=218`
	- later-layout `MOLT` inspect output now reports `rotationEntries`, `nonIdentityRotations`, and `rotationLenRange`; real standard Ironforge currently proves `218`, `218`, and `[1.118, 1.118]`
	- optional root `MOLT` summary failures still remain non-fatal in inspect, but Ironforge now exercises the actual shared `MOLT` summary path instead of only the fallback path
	- it now gets file-kind classification from shared `WowFileDetector` instead of its own private heuristics
	- it is a shared `Core` + `Core.IO` consumer, not a tool-local parser
- Smoke-test commands that should now work on the fixed development dataset:
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- blp inspect --archive-root .\testdata\0.6.0\World of Warcraft\Data --virtual-path interface/minimap/minimaparrow.blp`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- mdx export-json --input .\testdata\0.5.3\tree\Creature\Wisp\Wisp.mdx --output .\output\mdx-wisp-summary.json`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- mdx export-json --archive-root .\testdata\0.6.0\World of Warcraft\Data --listfile .\libs\wowdev\wow-listfile\listfile.txt --virtual-path world/generic/activedoodads/chest01/chest01.mdx --include-geometry --output .\output\mdx-chest-geometry.json`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- mdx export-json --archive-root .\testdata\0.6.0\World of Warcraft\Data --listfile .\libs\wowdev\wow-listfile\listfile.txt --virtual-path character/dwarf/female/dwarffemale.mdx --include-collision --output .\output\mdx-dwarffemale-collision.json`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- mdx export-json --archive-root .\testdata\0.6.0\World of Warcraft\Data --listfile .\libs\wowdev\wow-listfile\listfile.txt --virtual-path creature/anubisath/anubisath.mdx --include-hit-test --output .\output\mdx-anubisath-hit-test.json`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- mdx export-json --archive-root .\testdata\0.6.0\World of Warcraft\Data --listfile .\libs\wowdev\wow-listfile\listfile.txt --virtual-path creature/airelemental/airelemental.mdx --include-texture-animations --output .\output\mdx-airelemental-texture-animations.json`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- mdx chunk-carriers --chunks LITE --archive-root .\testdata\0.6.0\World of Warcraft\Data --listfile .\libs\wowdev\wow-listfile\listfile.txt --path-filter braziers --limit 100`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- mdx chunk-carriers --chunks TXAN,PREM,CORN --input .\testdata\0.5.3\tree --limit 500`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- archive build-listfile-cache --archive-root .\testdata\0.6.0\World of Warcraft\Data --cache-key 0.6.0.3592 --listfile .\libs\wowdev\wow-listfile\listfile.txt`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- map inspect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development.wdt`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- map uniqueid-report --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development.wdt --build development --output .\output\reports\map-uniqueids\development.json`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root .\testdata\0.6.0\World of Warcraft\Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- map inspect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0.adt`

Current non-PM4 converter slice:

- `tools/converter/WowViewer.Tool.Converter` now also supports:
	- `detect --input <file>`
	- `export-tex-json --input <file.adt|file_tex0.adt> [--output <report.json>]`
- This is intentionally narrow for now:
	- it reports shared file-family classification and version using `WowFileDetector`
	- it now also exports shared root-plus-`_tex0` per-chunk layer and decoded-alpha data as JSON through `AdtTextureReader`
	- it is still not yet a broader terrain conversion workflow
- Smoke-test commands that should now work on the fixed development dataset:
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- detect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- export-tex-json --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0.adt`
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- detect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0_tex0.adt`
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- detect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0_obj0.adt`
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- export-tex-json --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0_tex0.adt`

Current first real code-port slice:

- `src/core/WowViewer.Core.PM4` now contains a research-seeded PM4 model and reader layer.
- Ported first slice from `Pm4Research.Core`:
	- typed chunk models for the currently trusted PM4 chunk set
	- research document container
	- binary PM4 reader
	- exploration snapshot builder
- This started as a raw research-facing PM4 layer. Remaining PM4 gaps should be closed directly in `Core.PM4` rather than deferred to `MdxViewer` as the default owner.

Current PM4 inspect slice:

- `src/core/WowViewer.Core.PM4` now also contains the first single-file PM4 analyzer and report layer.
- `src/core/WowViewer.Core.PM4` now also contains a research-only CK24 forensic report layer for targeted object-graph export, raw MSLK rows, raw linked MPRL rows, footprint counts, and placement-vs-heading comparison.
- `src/core/WowViewer.Core.PM4` now also contains a research-only hierarchy analyzer that ports the old object-hypothesis family splits and enriches them with shared placement evidence plus dominant `MSLK.GroupObjectId` ownership.
- `pm4 match` currently uses shared PM4 footprint overlap scoring first, then linked MPRL anchor proximity when present, and prefers MDX collision bounds over generic model bounds when the asset exposes collision data.
- `pm4 match --object-output-dir <directory>` writes one JSON file per PM4 object plus a tile manifest, including the selected MSUR surface slice and top placement candidates so later rebuild tooling can consume the artifacts directly.
- `tools/inspect/WowViewer.Tool.Inspect` now supports:
	- `pm4 inspect --input <file.pm4>`
	- `pm4 match --input <file.pm4> --archive-root <game|data dir> [--placements <tile_obj0.adt>] [--listfile <listfile.txt>] [--max-matches <n>] [--search-range <units>] [--output <report.json>] [--object-output-dir <directory>]`
	- `pm4 hierarchy --input <file.pm4> [--output <report.json>]`
	- `pm4 linkage --input <directory> [--output <report.json>]`
	- `pm4 mscn --input <directory> [--output <report.json>]`
	- `pm4 unknowns --input <directory> [--output <report.json>]`
	- `pm4 audit --input <file.pm4>`
	- `pm4 audit-directory --input <directory>`
	- `pm4 export-json --input <file.pm4> [--output <report.json>] [--ck24 <decimal|0xHEX>]`
- Smoke-test command that passed on Mar 25, 2026:
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 inspect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 match --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4 --placements ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0_obj0.adt --archive-root <game data dir> --max-matches 8 --search-range 128 --output .\output\pm4-match-development-00-00.json`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 match --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4 --placements ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0_obj0.adt --archive-root <game data dir> --max-matches 3 --search-range 128 --object-output-dir .\output\pm4-match-development-00-00-objects`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 hierarchy --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 linkage --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 mscn --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 unknowns --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 audit --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 export-json --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4 --ck24 0x412CDC --output .\output\pm4_ck24_412CDC_forensics.json`

Current PM4 runtime-contract slice:

- `src/core/WowViewer.Core.PM4` now contains the first library-owned PM4 runtime placement contract slice.
- Landed pieces:
	- public `Pm4AxisConvention`, `Pm4CoordinateMode`, and `Pm4PlanarTransform` contracts
	- shared `Pm4CoordinateService`
	- shared planar candidate contract in `Pm4PlacementContract`
	- shared placement math helpers in `Pm4PlacementMath`
	- first normal-based axis scoring and detection helpers ported from `WorldScene`
	- first planar-transform resolver ported from `WorldScene`, including MPRL centroid, footprint, and yaw scoring against candidate planar bases
	- first world-yaw correction solver ported from `WorldScene`, including signed basis fallback against MPRL heading evidence
	- first world-space surface centroid helper ported from `WorldScene`, keeping surface-derived pivot computation in shared PM4 math instead of viewer-owned placement code
	- first world-space yaw-application helper layer ported from `WorldScene`, keeping pivot rotation and corrected world-position conversion in shared PM4 math without pulling renderer-space mapping into the library
	- first reusable `Pm4PlacementSolution` contract and resolver entry point, so future consumers can ask `Core.PM4` for one typed placement result instead of stitching together transform, pivot, and yaw pieces manually
	- first typed coordinate-mode resolver, so future consumers can ask `Core.PM4` to score tile-local versus world-space interpretation with an explicit fallback contract instead of keeping that decision loop in `WorldScene`
	- first reusable linked-position-ref summary contract and summary helper, so future consumers can summarize linked MPRL floor-range, heading-range, and circular-mean evidence in `Core.PM4` instead of leaving that aggregation in `WorldScene`
	- first reusable `Pm4ConnectorKey` contract and connector-key builder, so future grouping or correlation work can derive quantized world-space connector keys from exterior vertices through typed placement solutions instead of leaving that derivation marooned in `WorldScene`
	- first reusable connector-group merge contracts and merge-map builder, so future grouping work can resolve canonical merged object groups from connector overlap, bounds padding, and center distance in `Core.PM4` instead of leaving those heuristics marooned in `WorldScene`
	- first reusable correlation-score contracts and correlation math helpers, so future placement or report work can score planar gaps, overlap ratios, footprint distance, and candidate ordering in `Core.PM4` instead of leaving those heuristics marooned in `WorldScene`
	- first reusable correlation object-state contracts and object-state builder, so future placement or report work can summarize PM4 correlation objects, sampled footprint hulls, and empty-geometry fallback state in `Core.PM4` instead of leaving that ownership in `WorldScene`
	- first reusable PM4 geometry-input contracts and geometry-input object-state builder, so future placement or report work can transform PM4 line or triangle geometry into shared correlation states in `Core.PM4` instead of leaving that world-point assembly in `WorldScene`
- The current single-file inspect output also records the working research note that CK24 low-16 object values may be plausible `UniqueID` candidates, but this remains unverified until correlated against real placement data.

Current PM4 shared-consumer slice:

- `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.PM4`.
- `WorldScene` now delegates these narrow PM4 solver seams to shared `Core.PM4` through explicit adapters:
	- shared `ResolvePlacementSolution(...)` consumption in the CK24 overlay path
	- `ResolvePlanarTransform(...)`
	- `TryComputeWorldYawCorrectionRadians(...)`
	- `ComputeSurfaceWorldCentroid(...)`
	- `SummarizeLinkedPositionRefs(...)`
	- `ResolveCk24CoordinateMode(...)`
	- `BuildCk24ConnectorKeys()`
	- `RebuildPm4MergedObjectGroups()`
	- `BuildPm4CorrelationObjectStates()`
	- shared correlation object-state, footprint-hull, and metric helpers in the PM4 or WMO correlation reporting path
- Important boundary:
	- PM4-owned geometry, transforms, and shared object-state construction now belong in `Core.PM4`
	- WMO-facing correlation report payloads stay in WMO or consumer space rather than being re-homed into PM4
	- renderer-space centroid and the broader PM4 placement or render path still remain in `WorldScene`
	- this is a secondary compatibility seam only, not the source of truth for new PM4 implementation work
	- this is build-validated solver sharing, not viewer runtime PM4 signoff

Current shared-I/O consumer slice:

- `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` now also references `wow-viewer/src/core/WowViewer.Core.IO`.
- `MdxViewer` now consumes these shared non-PM4 helpers from `Core.IO` instead of `WoWMapConverter.Core.Services`:
	- `Md5TranslateIndex`
	- `Md5TranslateResolver`
	- `MinimapService`
- `MdxViewer` now also consumes these shared archive contracts from `Core.IO` instead of exposing `NativeMpqService` directly in its active standard-MPQ consumer path:
	- `IArchiveReader`
	- `IArchiveCatalog`
	- `IArchiveCatalogFactory`
	- `DbClientFileReader`
- `MdxViewer` now also consumes these shared archive helpers from `Core.IO` instead of owning them locally or calling the old Alpha wrapper reader directly:
	- `ArchiveCatalogBootstrapper`
	- `ArchiveCatalogBootstrapResult`
	- `AlphaArchiveReader`
	- `PkwareExplode`
- `MdxViewer` now also instantiates the concrete standard MPQ implementation from `Core.IO` instead of using an adapter over `WoWMapConverter.Core.Services.NativeMpqService`:
	- `MpqArchiveCatalog`
	- `MpqArchiveCatalogFactory`
- `ViewerApp` now loads the optional MD5 minimap translation index through shared `Core.IO`.
- `MinimapRenderer` and `MapGlbExporter` now use shared minimap path and translation helpers.
- `MpqDataSource` and `MpqDBCProvider` now use shared archive-reader contracts for standard MPQ access.
- `MpqDataSource` now also uses shared archive bootstrap and Alpha wrapper helpers.
- the active bridge file `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs` is gone.
- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj` now also references `wow-viewer/src/core/WowViewer.Core.IO`.
- `WoWMapConverter.Core.VLM.VlmDatasetExporter` now uses shared archive interfaces plus shared DBC-backed lookup helpers instead of `NativeMpqService`, `MapDbcService`, and `GroundEffectService` in its active path.
- `WoWMapConverter.Core.VLM.VlmDatasetExporter`, `WmoV14ToV17Converter`, and `WmoV14ToV17ExtendedConverter` now also use shared `AlphaArchiveReader` for Alpha per-asset MPQ reads.
- `WoWMapConverter.Core.Converters.AlphaToLkConverter` now also uses shared `AreaIdMapper` for Alpha-to-LK area-ID mapping, and its active CLI path now accepts explicit `--alpha-client` and `--lk-client` archive roots instead of constructor-time extracted-tree probing.
- the dead duplicate old-repo helper layer behind that active VLM path has now been deleted from `WoWMapConverter.Core`, including the old local `DbcReader`, `NativeMpqService`, `Md5TranslateResolver`, `MapDbcService`, and `GroundEffectService` files.
- the duplicate old-repo `WoWMapConverter.Core.Services.AlphaMpqReader` file is now also gone.
- the old-repo `WoWMapConverter.Core.Dbc.AreaIdMapper`, dead `Services.AreaIdCrosswalk`, and the old embedded `area_crosswalk.csv` copy are now gone.
- Important boundary:
	- the active `MdxViewer` standard MPQ implementation now lives in `Core.IO`
	- the active old-repo Alpha per-asset MPQ caller seam now also runs on shared `Core.IO` through `VlmDatasetExporter` and the WMO converters
	- the active old-repo area-ID mapping seam now also runs on shared `Core.IO` through `AlphaToLkConverter`
	- `MdxViewer` and `WoWMapConverter.Core` still depend on old-repo code for broader terrain, converter, VLM, and other non-migrated services
	- the old archive or lookup helper layer used by the active VLM and area-mapping paths no longer exists in `WoWMapConverter.Core`
	- the new shared DBC slice proves narrow lookup ownership, not full general DBC or DB2 format ownership
	- this is build or test validation, not runtime viewer signoff

Current PM4 linkage slice:

- `src/core/WowViewer.Core.PM4` now also contains the first linkage-report family ported from `Pm4Research.Core`.
- `pm4 linkage --input <directory>` now reports:
	- corpus-wide `MSLK.RefIndex` mismatch counts
	- `MSUR.MdosIndex -> MSCN` miss counts
	- CK24 low-16 object-id reuse across full CK24 values and CK24 types
	- top mismatch-family buckets keyed by decoded tile-link or raw link id
- Current validated development-corpus highlights:
	- `616` PM4 files scanned
	- `150` files with ref-index mismatches
	- `58` files with bad `MDOS` refs
	- `4553` total ref-index mismatches
	- only `2` file-local low16 object-id groups reused across multiple full CK24 values in this corpus slice
- Current interpretation boundary:
	- low16 CK24 object values can still sit in plausible `UniqueID` ranges while not behaving like globally unique object identifiers by themselves
	- range alignment is therefore suggestive, not confirming

Current PM4 MSCN slice:

- `src/core/WowViewer.Core.PM4` now also contains the first MSCN relationship report family ported from `Pm4Research.Core`.
- `pm4 mscn --input <directory> [--output <report.json>]` now reports:
	- corpus-wide `MSUR.MdosIndex -> MSCN` fit or miss counts
	- CK24 coverage and mesh-plus-MSCN overlap counts
	- raw-vs-swapped MSCN bounds overlap against mesh-backed CK24 groups
	- per-file coordinate-space dominance and invalid-`MDOS` cluster examples
- Current validated development-corpus highlights:
	- `616` PM4 files scanned
	- `309` files with MSCN
	- `1,342,410` total MSCN points
	- raw MSCN-to-mesh bounds overlap: `1162` fits, `724` misses
	- swapped-XY MSCN-to-mesh bounds overlap: only `10` fits, `1876` misses
	- current corpus therefore supports raw MSCN bounds overlap more strongly than swapped XY in this slice
- Interpretation boundary:
	- this still does not make MSCN authoritative for final viewer reconstruction
	- it does materially weaken the earlier idea that simple XY-swapping is the main MSCN companion-space explanation for this corpus

Current PM4 unknowns slice:

- `src/core/WowViewer.Core.PM4` now also contains the first unknowns-report family ported from `Pm4Research.Core`.
- `pm4 unknowns --input <directory> [--output <report.json>]` now reports:
	- corpus-wide relationship-fit summaries for `MSUR`, `MSVI`, `MSPI`, `MSLK`, `MDSF`, `MDOS`, and `MPRR`
	- `MSLK.LinkId` sentinel-tile patterns
	- `MSLK.MspiIndexCount` interpretation buckets
	- field distributions for `MSHD`, `MSLK`, `MSUR`, `MPRL`, and `MPRR`
	- explicit open-question findings and next-step notes
- Current validated development-corpus highlights:
	- `616` PM4 files scanned
	- `309` non-empty geometry or link files
	- `1,273,335` `MSLK.LinkId` values, all currently fitting the sentinel-tile pattern in this corpus
	- `598,882` active `MSLK` path windows: `399,183` indices-only fits and `199,699` dual-fit windows
	- `MSLK.RefIndex -> MSUR` remains partial with `1,268,782` fits and `4,553` misses
	- `MPRR.Value1` remains mixed-domain with both `MPRL` and `MSVT` partial fits
- Interpretation boundary:
	- this report strengthens the relationship evidence base, but it still does not close the final semantics of `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR`, or PM4 coordinate ownership by itself

Current validation:

- `dotnet build .\WowViewer.slnx -c Debug` passed on Mar 25, 2026 in this workspace scaffold.
- `dotnet test .\WowViewer.slnx -c Debug` currently also covers the first shared WDT or ADT top-level summary slice in addition to PM4 and FourCC tests.
- `dotnet test .\WowViewer.slnx -c Debug` also now covers the first shared cross-family detector slice, the shared archive-reader, concrete MPQ catalog, and shared DBC lookup slices, plus the PM4 connector-key, connector-group merge, correlation-math, correlation object-state, correlation geometry-input, and linked-position-ref summary regression slices, with `59` passing tests on Mar 26, 2026.

Current Copilot continuity surface:

- Use `.github/skills/wow-viewer-pm4-library/SKILL.md` and `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md` when the ask is a real `Core.PM4` slice such as `pm4 inspect`, `pm4 audit`, `pm4 audit-directory`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, or `pm4 export-json`, a PM4 regression change, or a narrow PM4 solver extraction.
- Use `.github/skills/wow-viewer-shared-io-library/SKILL.md` and `.github/prompts/wow-viewer-shared-io-implementation.prompt.md` when the ask is a real non-PM4 `Core` or `Core.IO` slice such as ADT root or split-ADT (`_tex0.adt`, `_obj0.adt`, `_lod.adt`) work, WDT or WMO summary work, BLP or DBC or DB2 detection work, a file detector, a chunk reader, a map-summary seam, an inspect verb, a converter command, or a shared-format regression change.
- Use `.github/skills/wow-viewer-migration-continuation/SKILL.md` and `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` when the ask is broader migration routing, ownership planning, bootstrap shape, tool inventory, or sequencing.
- Use `.github/prompts/wow-viewer-editor-plan-set.prompt.md` when the ask is the broader viewer-to-editor transition, including PM4 `MPRL`-assisted terrain conform, saved object choices, moved-object persistence, map save ownership, or viewer-vs-editor workspace routing.
- Use `.github/prompts/wow-viewer-map-editing-foundation-plan.prompt.md` when the ask is planning the first true terrain or object editing or dirty-map or save pipeline slice.
- Use `.github/prompts/wow-viewer-editor-ui-surface-plan.prompt.md` when the ask is planning viewer and editor workspace presets, editor task clustering, and panel reorganization.
- editor continuity is now also recorded in `../gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`.
- Shared non-PM4 continuity is now also recorded in `../gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`.
- Forward-maintenance rule:
	- if a new `wow-viewer` skill or implementation prompt is added, update `.github/copilot-instructions.md`, this README section, the relevant `gillijimproject_refactor/plans/wow_viewer_*` continuity plan, and the memory bank in the same slice so future chats discover it automatically.

Current PM4 test slice:

- `tests/WowViewer.Core.PM4.Tests` now exists as the first real-data test project in `wow-viewer`.
- It currently locks:
	- the reader counts for `development_00_00.pm4`
	- the current single-file PM4 analysis summary and `UniqueID` research note
	- the single-file decode-audit findings for `development_00_00.pm4`
	- the current corpus-audit shape for the fixed development PM4 directory
	- the current linkage-report mismatch or reuse signals for the fixed development PM4 directory
	- the current range-based plus normal-based axis heuristics and tile-local placement heuristics on `development_00_00.pm4`
	- the current planar-transform resolver behavior on `development_00_00.pm4` plus a synthetic world-space quarter-turn case
	- the current coordinate-mode resolver behavior on `development_00_00.pm4`, a synthetic world-space case, and the missing-evidence fallback path
	- the current world-yaw correction behavior in a synthetic solver case
	- the current world-space surface centroid behavior in a synthetic tile-local centroid case
	- the current world-space pivot rotation and corrected world-position conversion behavior in synthetic solver cases
	- the current end-to-end placement-solution contract behavior in synthetic world-space resolver cases
	- the current linked-position-ref summary behavior in synthetic mixed normal-or-terminator and terminator-only cases
	- the current correlation-math metric and candidate-ranking behavior in synthetic overlap or same-tile precedence cases
	- the current correlation object-state bounds, empty-geometry fallback, and transformed footprint-hull behavior in synthetic cases
	- the current PM4 geometry-input object-state behavior in a synthetic transform case without viewer-specific world-point assembly
	- the current MSCN relationship counts and raw-vs-swapped overlap signals for the fixed development PM4 directory
	- the current unknowns-report signals for the fixed development PM4 directory
- Validation command that passed on Mar 25, 2026:
	- `dotnet test .\WowViewer.slnx -c Debug`
		- current result as of Mar 26, 2026: `31` passing PM4 tests
