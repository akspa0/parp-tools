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
- The repo is now starting to correct that with a real bootstrap script and a first non-PM4 chunk or FourCC foundation slice, but the broader shared I/O and runtime cutover is still missing.

Current implementation policy:

- `WowViewer.Core.PM4`, `WowViewer.Core`, and `WowViewer.Core.IO` are the canonical implementation targets for new `wow-viewer` work.
- `gillijimproject_refactor`, including `MdxViewer` and `WoWMapConverter`, is now a reference or compatibility input for `wow-viewer` work, not the default owner of the design.
- Default validation for `wow-viewer` work is `dotnet build .\WowViewer.slnx -c Debug`, `dotnet test .\WowViewer.slnx -c Debug`, and the relevant inspect or converter command against the fixed development dataset.
- Build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln` only when a slice explicitly changes consumer compatibility or the user asks for that check.

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
  - `WdtSummary`
- `src/core/WowViewer.Core/Wmo` now contains the first shared WMO root-summary contracts:
  - `WmoChunkIds`
  - `WmoSummary`
  - `WmoGroupSummary`
  - `WmoLiquidBasicType`
  - `WmoGroupLiquidSummary`
  - `WmoGroupBatchSummary`
- `src/core/WowViewer.Core.IO` now contains the first non-PM4 I/O seam:
	- `ChunkHeaderReader`
- `src/core/WowViewer.Core.IO` now also contains the first shared WDT or ADT top-level reader slice:
  - `ChunkedFileReader`
  - `MapFileSummaryReader`
  - `AdtSummaryReader`
  - `AdtMcnkSummaryReader`
  - `WdtSummaryReader`
- `src/core/WowViewer.Core.IO/Wmo` now contains the first shared WMO root-summary reader:
  - `WmoSummaryReader`
  - `WmoGroupSummaryReader`
  - `WmoGroupReaderCommon`
  - `WmoGroupLiquidSummaryReader`
  - `WmoGroupBatchSummaryReader`
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
	- it now also locks synthetic and real-data WDT or ADT summary behavior against `development.wdt` and `development_0_0.adt`
	- it now also locks synthetic and real-data ADT semantic-summary behavior for `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
	- it now also locks shared ADT `MCNK` semantic-summary behavior for synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
	- it now also locks a synthetic WMO root semantic-summary case
	- it now also locks synthetic WMO group semantic-summary behavior for both `MVER + MOGP` and `MOGP`-first files, plus `MOGP`-first detector coverage
	- it now also locks synthetic WMO `MLIQ` semantic-summary behavior including height-range and ocean-inference coverage
	- it now also locks synthetic WMO `MOBA` batch semantic-summary behavior including v17-style material ids and v16-style material-less batches
	- it now also locks synthetic Alpha and standard WDT semantic-summary behavior plus real-data `development.wdt` occupancy and MPHD signals
	- it now also locks shared file detection for `development.wdt`, `development_0_0.adt`, `development_0_0_tex0.adt`, `development_0_0_obj0.adt`, and `development_00_00.pm4`

Current non-PM4 inspect slice:

- `tools/inspect/WowViewer.Tool.Inspect` now also supports:
	- `map inspect --input <file.wdt|file.adt>`
	- `wmo inspect --input <file.wmo>`
- This is intentionally narrow for now:
	- it reads top-level chunk order, counts, version, and file-kind classification for WDT and ADT-family files
	- it now also reports a shared ADT semantic summary for terrain-chunk counts, string-table counts, placement counts, and selected MFBO or MH2O or MAMP or MTXF presence across root, `_tex0.adt`, and `_obj0.adt`
	- it now also reports a shared ADT `MCNK` semantic summary for root-header coverage, selected flags, split-file subchunk presence, and per-chunk layer-count signals across root, `_tex0.adt`, and `_obj0.adt`
	- it now also reports a shared WDT semantic summary for MPHD WMO-based flags, MAIN tile occupancy, string-table counts, and top-level MDDF or MODF placement counts
	- it now also reports a first shared WMO root semantic summary for `MOHD`-reported counts, top-level entry counts, string-table counts, flags, and bounds
	- it now also reports a first shared WMO group semantic summary for `MOGP` header fields, geometry subchunk counts, optional extra UV-set count, doodad-ref count, and liquid presence
	- it now also reports a shared WMO `MLIQ` semantic summary for liquid dimensions, height range, visible-tile counts, and basic family inference when a group file contains liquid
	- it now also reports a shared WMO `MOBA` batch semantic summary for batch-entry counts, material-id coverage, index coverage, and flagged-batch counts when a group file contains batches
	- it now gets file-kind classification from shared `WowFileDetector` instead of its own private heuristics
	- it is a shared `Core` + `Core.IO` consumer, not a tool-local parser
- Smoke-test commands that should now work on the fixed development dataset:
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- map inspect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development.wdt`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- map inspect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0.adt`

Current non-PM4 converter slice:

- `tools/converter/WowViewer.Tool.Converter` now also supports:
	- `detect --input <file>`
- This is intentionally narrow for now:
	- it reports shared file-family classification and version using `WowFileDetector`
	- it is the first non-placeholder converter command, but it is not yet a conversion workflow
- Smoke-test commands that should now work on the fixed development dataset:
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- detect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- detect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0_tex0.adt`
	- `dotnet run --project .\tools\converter\WowViewer.Tool.Converter\WowViewer.Tool.Converter.csproj -- detect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_0_0_obj0.adt`

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
- `tools/inspect/WowViewer.Tool.Inspect` now supports:
	- `pm4 inspect --input <file.pm4>`
	- `pm4 linkage --input <directory> [--output <report.json>]`
	- `pm4 mscn --input <directory> [--output <report.json>]`
	- `pm4 unknowns --input <directory> [--output <report.json>]`
	- `pm4 audit --input <file.pm4>`
	- `pm4 audit-directory --input <directory>`
	- `pm4 export-json --input <file.pm4> [--output <report.json>]`
- Smoke-test command that passed on Mar 25, 2026:
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 inspect --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 linkage --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 mscn --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 unknowns --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 audit --input ..\gillijimproject_refactor\test_data\development\World\Maps\development\development_00_00.pm4`
	- `dotnet run --project .\tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input ..\gillijimproject_refactor\test_data\development\World\Maps\development`

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
