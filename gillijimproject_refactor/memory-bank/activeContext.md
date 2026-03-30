# Active Context

## Mar 29, 2026 - v0.4.6 Release Target And Renderer-Layer Pivot

- user runtime feedback after the latest PM4 fixes is materially different from earlier sessions:
	- PM4 objects are now described as almost `100%` correct on the active development-map workflow
	- that makes the latest PM4 runtime changes a freeze-worthy release seam rather than another speculative experiment
- `v0.4.6` should now be treated as the active viewer release target in this tree
- the PM4 release-facing wins that need to stay called out together are:
	- ADT-scale PM4 camera-window indexing instead of the earlier wrong WDL-scale indexing
	- transposed PM4 filename tile remap into terrain tile space
	- correct handling for zero-byte PM4 carriers and empty-known PM4 windows
	- removal of terrain-AOI PM4 slicing for already loaded PM4 content
	- linked-group placement resolution for non-zero `CK24` seed groups instead of one shared seed transform
- next rendering/performance priority is no longer another narrow PM4 correctness pass by default
- current renderer direction requested by user:
	- move toward real render layers and explicit submission buckets instead of the current hard-coded `WorldScene` pass order with renderer-local immediate draw behavior
	- reduce draw-call and state churn so debugging overlays and exploration surfaces stop fighting the main scene path
- most likely first architecture seam:
	- keep world visibility/culling collection in `WorldScene`, but build per-frame render-layer submission lists for terrain opaque, WMO opaque, MDX opaque, liquids, transparent world geometry, PM4 overlay solids, PM4 overlay lines, and debug/editor overlays
	- route compatible items through a shared queue or batching surface instead of mixing cull, sort, GL-state changes, and direct draw calls inside one monolithic frame function
- important boundary:
	- `v0.4.6` still does not imply final renderer performance closure
	- the current performance work is a first reduction in waste, while the render-layer/submission redesign is still ahead
	- release packaging currently depends on a workflow-side publish mitigation because `WoWMapConverter.Core` still references `WoWRollback.PM4Module` as an `Exe` project, which causes duplicate dependency publish artifacts during viewer publish unless duplicate publish-output errors are relaxed

## Mar 29, 2026 - Second Viewer Performance Slice Defers WMO Doodads And Disables Object Fog By Default

- follow-up to the first `WorldScene` MDX classification pass after the user reported the viewer was still hitching hard during tile or data loads and that world objects were appearing inside unwanted fog
- strongest newly confirmed hitch source in the active viewer path:
	- `src/MdxViewer/Rendering/WmoRenderer.cs` was eagerly calling `LoadActiveDoodadSet()` in the constructor
	- that constructor path could recursively build many doodad `MdxRenderer`s on the render thread as soon as a WMO shell became visible
- landed behavior in this slice:
	- `WmoRenderer` now supports deferred initial doodad loading for world-scene WMO usage and incrementally loads queued doodad models during render under a small per-frame budget instead of eagerly expanding the whole doodad set in the constructor
	- `src/MdxViewer/Terrain/WorldAssetManager.cs` now opts world-scene WMO loads into that deferred doodad path
	- `src/MdxViewer/Terrain/WorldScene.cs` now lowers render-thread deferred asset processing from `24 loads / 20 ms` to `6 loads / 4 ms` per frame
	- `WorldScene` now disables object fog by default through a dedicated `ObjectFogEnabled` policy and still keeps WMO culling tied to real terrain fog distance instead of the disabled object-fog range
	- `src/MdxViewer/ViewerApp.cs` now exposes a `Fog Objects` checkbox in the world-objects panel so the old behavior can still be re-enabled for comparison
- validation completed:
	- editor error checks were clean for `WorldScene.cs`, `ViewerApp.cs`, `WmoRenderer.cs`, and `WorldAssetManager.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing solution warnings only
- important boundary:
	- this is still compile validation only in this session
	- no live viewer frame-time capture or real-data runtime signoff was completed yet for the deferred WMO doodad path or the new default no-object-fog policy

## Mar 29, 2026 - Viewer Performance Pivot Started With WorldScene MDX Classification Pass

- user direction has shifted away from more PM4-first work and toward real viewer rendering performance or lighting or shader quality, because current map loads still feel unusable at roughly `1-5 FPS`
- first chosen slice is deliberately CPU-side and narrow:
	- reduce per-frame duplicate object work in `src/MdxViewer/Terrain/WorldScene.cs`
	- do not start with shader-parity or sky or lighting refactors before the main scene loop is cheaper
- landed optimization:
	- `WorldScene` now classifies visible loaded `MDX` or taxi-actor instances once per frame into a reusable scratch list
	- the opaque and transparent doodad passes now reuse that one visibility result instead of redoing AABB distance checks or frustum tests or `TryGetQueuedMdx(...)` lookups in separate passes
	- shared per-instance fade values are also precomputed once and reused across both passes
- validation completed:
	- editor error check on `WorldScene.cs` passed
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing environment warnings only
- important boundary:
	- this is a first hot-path reduction only, not full FPS recovery
	- no live viewer frame-time capture or runtime signoff was completed in this pass yet

## Mar 29, 2026 - PM4 Terminology Reconciliation Locked For `wow-viewer`

- the current PM4 reader or analyzer stack is no longer allowed to blur wowdev field names with local research aliases
- current locked rule:
	- use wowdev PM4 or PD4 names when the docs actually name a field
	- use raw offset-style names first when the docs only expose placeholders
	- mention local names second as explicit aliases with confidence level when semantics are still open
- important current reconciliations:
	- `MSUR.AttributeMask`, `MSUR.GroupKey`, `MSUR.MdosIndex`, `MSUR.PackedParams`, `CK24`, `Ck24Type`, `Ck24ObjectId`, and `MSLK.GroupObjectId` are local research aliases, not original wowdev terminology
	- `CK24` remains a useful derived identity slice from `MSUR._0x1c`, but it should not be described as an official PM4 field name
	- `MSUR.Height` is now known to be a bad name for the final float; current geometry evidence says it behaves like a signed plane-distance term
	- `MSLK.RefIndex` should no longer be spoken about as if the wiki label `msur_index` were fully closed truth across the corpus
- continuity updates landed in:
	- `gillijimproject_refactor/src/Pm4Research.Core/README.md`
	- `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `wow-viewer/README.md`
- practical implication for future chats:
	- PM4 work should now default to terminology like `MSUR._0x1c (local alias: PackedParams; derived alias: CK24)` instead of presenting `PackedParams` or `CK24` as if they came from the original documentation

## Mar 29, 2026 - Shared CK24 PM4 Forensics Landed In `wow-viewer`

- `wow-viewer` now has a research-only shared CK24 forensic export path in `Core.PM4` instead of leaving richer PM4 graph evidence trapped in `MdxViewer` JSON only.
- Landed pieces in this slice:
	- `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ForensicsModels.cs` now carries shared CK24 forensic report contracts for per-component link groups, raw MSLK rows, raw linked MPRL rows, footprint counts, and placement comparison.
	- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4Ck24ForensicsAnalyzer.cs` now builds component-level CK24 reports using the same current MSLK surface-link semantics as the viewer-side PM4 graph export, while keeping the report labeled as research-only.
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `pm4 export-json --ck24 <decimal|0xHEX>` so the shared inspect surface can emit either the coarse single-file PM4 report or a targeted CK24 forensic JSON without adding another tool-local parser path.
	- PM4 export JSON in inspect now enables field serialization so `System.Numerics` vectors serialize as real coordinates instead of empty objects.
- Validation completed on Mar 29, 2026:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed after the new analyzer and inspect wiring landed.
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter Pm4ResearchIntegrationTests` passed with new real-data CK24 forensic assertions.
	- `pm4 export-json --input .../development_00_00.pm4 --ck24 0x412CDC --output .../pm4_ck24_412CDC_forensics.json` wrote real shared CK24 forensic JSON against the fixed development dataset.
- Important boundary:
	- this is still research or export evidence only; it does not prove final PM4 object semantics or runtime viewer closure.

## Mar 28, 2026 - Default Post-MDX Continuation Target

- With classic `MDX` expansion paused, the default next `wow-viewer` implementation track should move to `Core.PM4` library completion rather than another `MDX` seam.
- Reason:
	- the PM4 continuity plan already names direct library completion as the clean next slice
	- `Core.PM4` is the most substantial real library area in `wow-viewer` today
	- this avoids drifting back into speculative `MDX` ownership or vague non-PM4 planning without a concrete target
- Secondary fallback only if PM4 is not the task:
	- continue non-`MDX` shared-I/O work on ADT/WDT/WMO only when the slice is narrow, tool-thin, and backed by a concrete validation target

## Mar 28, 2026 - MDX Audit: Separate Real Legacy Parity From New Shared Readers

- Audit result: recent `wow-viewer` classic `MDX` work is mixed.
- Grounded parity path:
	- `GEOS` shared summary/payload work is aligned with real legacy `MdxFile` parsing and current `MdxViewer` metadata/probe consumption.
- Not direct classic-parser parity:
	- `TXAN` payload reader in `wow-viewer` is not a direct port of active classic `MdxViewer` parsing; legacy `MdxFile` carries `TransformId` and the renderer can consume texture animations, but the classic `MdxFile` parser does not currently read `TXAN` into `TextureAnimations`
	- `HTST` payload reader in `wow-viewer` currently has no matching active classic `MdxViewer` parser/runtime implementation and should be treated as a new shared-reader seam, not viewer parity
	- `CLID` payload reader in `wow-viewer` also exceeds active classic `MdxViewer` behavior; legacy `MdxFile` skips `CLID`, while active `MdxViewer` only consumes shared collision summary metadata for model-info/probe surfaces
- If `MDX` work is ever resumed, the hotter missed legacy seam is not another cold chunk family. The real parity gap is the already-used classic `ATSQ`/geoset-animation and material-animation behavior that the active renderer consumes but `wow-viewer` still exposes mainly as summary-level metadata.

## Mar 28, 2026 - MDX Chunk Expansion Paused By User Direction

- Do not continue speculative `wow-viewer` `MDX` chunk-summary or payload implementation work by default.
- The user explicitly does not want further `MDX` chunk chasing just because chunks exist in archive data, especially when those seams were not implemented in `MdxViewer` already.
- Treat current `MDX` work as paused unless a future task explicitly asks for:
	- a specific `MdxViewer` compatibility need
	- a concrete consumer requirement already proven necessary in the active viewer/tool path
	- or a narrowly named `MDX` seam the user directly requests
- Default continuation should move back to non-`MDX` priorities instead of using `PREM` or `CORN` or any other remaining `MDX` family as the next automatic slice.

## Mar 29, 2026 - Shared Classic `MDX` `TXAN` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past unresolved classic `TXAN` chunk discovery into first shared texture-animation payload ownership for actual `KTAT` or `KTAR` or `KTAS` transform keyframes.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now also contains typed `MdxTextureAnimationFile` and `MdxTextureAnimation` payload contracts for classic indexed texture-animation entries
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxTextureAnimationReader.cs` now reads classic `TXAN` payloads for `v1300` and `v1400`, including counted sections and actual translation or rotation or scaling keyframe payloads
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxTrackReader.cs` now centralizes shared vector3 and compressed-quaternion keyframe parsing so `HTST` node-track and `TXAN` texture-track readers use one track interpretation
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json --include-texture-animations`, making the new shared `TXAN` payload seam exportable without adding another tool-local parser
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxTextureAnimationReaderTests.cs` now covers a synthetic tracked `TXAN` fixture, a real Alpha negative carrier, and a fixed real standard-era positive carrier on `creature/airelemental/airelemental.mdx`
- Validation completed on Mar 29, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxTextureAnimationReaderTests` passed after the new payload seam landed
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the new export option landed
	- `mdx export-json --archive-root ... --virtual-path creature/airelemental/airelemental.mdx --include-texture-animations --output .../mdx-airelemental-texture-animations.json` wrote real standard-era shared texture-animation payload JSON through the archive path
- Important boundary:
	- this is still payload ownership and export only; it does not add runtime texture-transform evaluation, material playback in the renderer, or `MdxViewer` cutover
	- this is also not the recommended continuation path anymore; further `MDX` chunk expansion is paused unless the user explicitly reopens it

## Mar 29, 2026 - Shared Classic `MDX` `HTST` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `HTST` summary ownership into first shared hit-test payload ownership for fixed shape fields plus actual `KGTR` or `KGRT` or `KGSC` node-track keyframes.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now also contains typed `MdxHitTestFile` and `MdxHitTestShape` payload contracts plus reusable node-track payload contracts for vector3 or compressed-quaternion keyframes and interpolation metadata
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxHitTestReader.cs` now reads classic `HTST` payloads for `v1300` and `v1400`, including fixed box or cylinder or sphere or plane payloads plus actual transform keyframe payloads instead of summary-only counts or time ranges
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json --include-hit-test`, making the new shared `HTST` payload seam exportable without adding another tool-local parser
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxHitTestReaderTests.cs` now covers a synthetic tracked `HTST` fixture plus fixed real Alpha and standard-era hit-test carriers
- Validation completed on Mar 29, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxHitTestReaderTests` passed after the new payload seam landed
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the new export option landed
	- `mdx export-json --archive-root ... --virtual-path creature/anubisath/anubisath.mdx --include-hit-test --output .../mdx-anubisath-hit-test.json` wrote real standard-era shared hit-test payload JSON through the archive path
- Important boundary:
	- this is still payload ownership and export only; it does not add runtime hit detection, animated transform evaluation in the renderer, or `MdxViewer` cutover

## Mar 28, 2026 - Shared Classic `MDX` `CLID` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `CLID` summary ownership into first shared collision-mesh payload ownership for ordered `VRTX` or `TRI ` or `NRMS` geometry.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxCollisionFile.cs` and `MdxCollisionMesh.cs` now carry the shared top-level classic `MDX` collision payload contract and typed mesh payload ownership
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxCollisionReader.cs` now reads classic `CLID` payloads for `v1300` and `v1400`, while `MdxSummaryReader` now reuses the same shared `MdxCollisionChunkReader` helper instead of maintaining a second independent `CLID` interpretation
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json --include-collision`, making the new shared collision payload seam exportable without adding another tool-local parser
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxCollisionReaderTests.cs` now covers a synthetic classic `CLID` payload fixture plus fixed real Alpha and standard-era collision carriers
- Validation completed on Mar 28, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxCollisionReaderTests|MdxSummaryReaderTests"` passed after the new payload seam landed
	- `mdx export-json --archive-root ... --virtual-path character/dwarf/female/dwarffemale.mdx --include-collision --output .../mdx-dwarffemale-collision.json` wrote real standard-era shared collision payload JSON through the archive path
- Important boundary:
	- this is still payload ownership and export only; it does not add collision queries, runtime physics, or `MdxViewer` collision rendering cutover

## Mar 28, 2026 - `WowViewer.Tool.Inspect` `mdx export-json` Slice Landed In `wow-viewer`

- `wow-viewer` now has a first reusable JSON export surface for the shared classic `MDX` summary seam, with optional inclusion of the current shared `GEOS` payload seam.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json` for filesystem or archive-backed inputs, with optional `--output <report.json>`
	- `mdx export-json --include-geometry` now also includes the current shared `MdxGeometryReader` output, so the first shared `GEOS` payload seam is exportable without adding a second tool-local parser
	- the command stays a thin consumer of `WowViewer.Core.IO.Mdx.MdxSummaryReader` and `MdxGeometryReader`; it does not move ownership out of the shared readers
- Validation completed on Mar 28, 2026:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the command landed
	- `mdx export-json --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx --output .../mdx-wisp-summary.json` wrote real Alpha summary JSON
	- `mdx export-json --archive-root ... --virtual-path world/generic/activedoodads/chest01/chest01.mdx --include-geometry --output .../mdx-chest-geometry.json` wrote real standard-era summary-plus-geometry JSON through the shared archive path
- Important boundary:
	- this is export of the current shared summary and `GEOS` payload seams only; it does not add new `MDX` chunk-family ownership or runtime render behavior
	- unresolved chunk families like `PREM` and `CORN` remain out of scope for now

## Mar 28, 2026 - `WowViewer.Tool.Inspect` `mdx chunk-carriers` Workflow Landed In `wow-viewer`

- `wow-viewer` now has a repeatable carrier-discovery workflow for classic `MDX` chunk continuation instead of relying on ad hoc archive probing or filename guesses.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx chunk-carriers --chunks <FOURCC[,FOURCC...]>` against either a filesystem file or directory or an archive-backed standard dataset with `--archive-root` and optional `--listfile`
	- the command stays a thin consumer of shared `WowViewer.Core.IO.Mdx.MdxSummaryReader`; it does not add tool-local `MDX` parsing or alternate chunk heuristics
	- the command also supports `--path-filter <text>` and `--limit <n>` so archive-backed scans can stay narrow and data-backed instead of brute-forcing the whole listfile blindly
- Validation completed on Mar 28, 2026:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the new command landed
	- `mdx chunk-carriers --chunks LITE --archive-root ... --path-filter braziers --limit 100` found `4` real standard-era `LITE` carriers, including the fixed `dwarvenbrazier01.mdx` validation surface
	- `mdx chunk-carriers --chunks TXAN,PREM,CORN --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree --limit 500` scanned the current unpacked alpha corpus (`229` MDX files) and found no carriers for those chunk ids
- Important boundary:
	- this slice adds a workflow and discovery surface, not new shared `MDX` chunk ownership by itself
	- current next-seam status remains: the bundled alpha corpus still has no fixed `TXAN` or `PREM` or `CORN` carrier, so the next classic `MDX` reader slice should still start from a real carrier search rather than from assumed file names

## Mar 28, 2026 - Viewer UI Resize And Hit-Testing Regression Fixed In `MdxViewer`

- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now explicitly resyncs the Silk `ImGuiController` logical window size so the active viewer shell no longer drifts into broken panel sizing and unusable button hit-testing after resize or maximize.
- Verification completed on Mar 28, 2026:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the fix
	- a short viewer startup smoke launched cleanly after the patch
	- the user manually retested the UI and reported that it now seems to be working
- Important boundary:
	- this is still manual runtime signoff only for the shell regression; no automated UI regression coverage exists yet
	- the current implementation reflects a private Silk `ImGuiController.WindowResized(Vector2D<int>)` method, so future package upgrades should treat that integration point as fragile and re-check resize behavior first

## Mar 28, 2026 - Shared Classic `MDX` `GEOS` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GEOS` summary ownership into first shared classic geoset payload ownership for render-facing vertex or normal or UV or index or skin-table data.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeometryFile.cs` now carries the shared top-level classic `MDX` geometry-file contract for payload-level `GEOS` reads
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeosetGeometry.cs` now carries shared per-geoset payload ownership for vertices, normals, UV sets, primitive types, face groups, indices, vertex groups, matrix tables, bone tables, and footer metadata
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxGeometryReader.cs` now reads classic counted `GEOS` payloads for `v1300` and `v1400`, including direct `UVAS` sets used in Alpha-era files plus optional explicit `UVBS` data when present
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxGeometryReaderTests.cs` now covers a synthetic classic `GEOS` payload fixture, a fixed real standard-era archive-backed positive carrier, and a real on-disk alpha-era positive carrier from the existing `0.5.3` corpus
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` geometry and summary reader coverage passed after the new synthetic and real `GEOS` payload tests landed
	- the real standard-era path now uses a fixed archive-backed positive `GEOS` carrier, preferring `Creature/AncientOfWar/AncientofWar.mdx` when present and falling back to the existing `chest01.mdx` validation surface otherwise
	- the real alpha-era path now uses the existing unpacked `0.5.3` creature corpus as the positive payload surface; the user-provided `AncientofWar.mdx` attachment remains a good future fixed carrier once it is committed into `wow-viewer/testdata/0.5.3/tree`
- Scope guardrail:
	- this is still classic `GEOS` payload ownership only; it does not yet build runtime render buffers, bind skeleton state, evaluate geoset animation visibility, or replace `MdxViewer` model loading

## Mar 28, 2026 - Shared Classic `MDX` `LITE` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GLBS` summary ownership into first shared classic `LITE` light-summary ownership for counted `MDLGENOBJECT`-derived light metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxLightType.cs` now carries the shared classic light kind enum for `Omni`, `Direct`, and `Ambient`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxLightSummary.cs` now carries shared per-light identity, hierarchy, static attenuation or color or intensity metadata, and summary-only `KLAS`, `KLAE`, `KLAC`, `KLAI`, `KLBC`, `KLBI`, and `KVIS` track metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `Lights` and `LightCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `LITE` entries for `v1300` and `v1400`, including inherited node metadata plus fixed light payload fields and optional summary-only light-track metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `lights=` in the header and prints `LITE[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `LITE` fixture plus a fixed real archive-backed `0.6.0` `dwarvenbrazier01.mdx` light regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real archive-backed `LITE` coverage landed
	- the focused MDX reader suite now also includes a real unpacked `0.5.3` alpha-corpus smoke over `229` MDX files under `wow-viewer/testdata/0.5.3/tree`, proving the new `LITE` summary path does not break current alpha-era parsing and that the bundled `0.5.3` sample set contains no `LITE` chunks today
	- `WowViewer.Tool.Inspect mdx inspect` on `0.6.0` `world/generic/dwarf/passive doodads/braziers/dwarvenbrazier01.mdx` now reports `lights=1`, `CHUNK[7]: id=LITE`, and stable `LITE[0]` light metadata including `Omni02`, static attenuation `0.8333333 -> 0.9722222`, and a `KLAI(keys=26 ... time=[0, 3333])` intensity track
- Scope guardrail:
	- this is still classic `LITE` summary ownership only; it does not evaluate runtime lighting, animation-driven intensity/color playback, or viewer render-light parity

## Mar 28, 2026 - Shared Classic `MDX` `GLBS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `CLID` summary ownership into first shared classic `GLBS` global-sequence summary ownership for strict counted `uint32` duration tables.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGlobalSequenceSummary.cs` now carries shared per-index global-sequence duration metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `GlobalSequences` and `GlobalSequenceCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads `GLBS` as a strict `uint32` table and rejects payload sizes that are not divisible by `4`
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `globalSequences=` in the header and prints `GLBS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic `GLBS` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` global-sequence regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `GLBS` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `globalSequences=11`, `CHUNK[3]: id=GLBS`, and stable `GLBS[0..10]` durations `267,133,533,0,567,900,1167,667,467,933,300`
- Scope guardrail:
	- this is still classic `GLBS` summary ownership only; it does not evaluate track playback, resolve `globalSeqId` references into runtime animation state, or claim full animation-system ownership

## Mar 28, 2026 - Shared Classic `MDX` `CLID` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `HTST` summary ownership into first shared classic `CLID` collision-summary ownership for ordered `VRTX` or `TRI ` or `NRMS` collision-mesh metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxCollisionSummary.cs` now carries shared collision counts, max-index coverage, and derived collision bounds
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes nullable `Collision` and `HasCollision`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic `CLID` chunks for `v1300` and `v1400`, including ordered `VRTX` or `TRI ` or `NRMS` subchunks, derived collision bounds, and index coverage
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `collisionVertices=` and `collisionTriangles=` in the header and prints a `CLID:` line during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `CLID` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` collision regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `CLID` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `collisionVertices=8`, `collisionTriangles=12`, `CHUNK[17]: id=CLID`, and stable `CLID: vertices=8 triIndices=36 triangles=12 facetNormals=12 maxIndex=7 ...`
- Scope guardrail:
	- this is still classic `CLID` summary ownership only; it does not expose full collision geometry payloads, collision queries, export surfaces, or runtime physics behavior

## Mar 28, 2026 - Shared Classic `MDX` `HTST` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `EVTS` summary ownership into first shared classic `HTST` hit-test-shape summary ownership for counted `MDLGENOBJECT` hit-test nodes and fixed box or cylinder or sphere or plane payload metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now carries shared `MdxGeometryShapeType` and `MdxHitTestShapeSummary` contracts for classic hit-test shapes beside the earlier event, camera, ribbon, particle, attachment, helper, and bone seams
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `HitTestShapes` and `HitTestShapeCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `HTST` entries for `v1300` and `v1400`, including inherited node metadata plus fixed `SHAPE_BOX` or `SHAPE_CYLINDER` or `SHAPE_SPHERE` or `SHAPE_PLANE` payload fields
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `hitTestShapes=` in the header and prints `HTST[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `HTST` fixture across box or cylinder or sphere or plane shapes plus a fixed real Alpha `0.5.3` `Wisp.mdx` sphere regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `HTST` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `hitTestShapes=1`, `CHUNK[16]: id=HTST`, and stable `HTST[0]: name=HIT01 ... shapeType=Sphere(2) shape=center=(0.366, 0.009, 1.890) radius=0.833333`
- Scope guardrail:
	- this is still classic `HTST` summary ownership only; it does not evaluate runtime collision or hit detection, animation-driven shape transforms, or viewer physics behavior

## Mar 28, 2026 - Shared Classic `MDX` `EVTS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `CAMS` summary ownership into first shared classic `EVTS` event-summary ownership for counted `MDLGENOBJECT` event nodes and optional summary-only `KEVT` time-track metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now carries shared `MdxEventSummary` and `MdxEventTrackSummary` contracts so classic event nodes live beside the existing summary-only bone, helper, attachment, ribbon, camera, and particle seams
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `Events` and `EventCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `EVTS` entries for `v1300` and `v1400`, including per-section sizing, inherited node metadata, and optional `KEVT` key-time metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `events=` in the header and prints `EVTS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `EVTS` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` event regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `EVTS` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `events=3`, `CHUNK[15]: id=EVTS`, and stable `EVTS[0..2]` node metadata with only the final `$DTH` event carrying `KEVT(keys=1 globalSeqId=-1 time=[1667, 1667])`
- Scope guardrail:
	- this is still classic `EVTS` summary ownership only; it does not evaluate event playback semantics, event lookup tables, particle or sound dispatch, or runtime trigger behavior

## Mar 28, 2026 - Shared Classic `MDX` `CAMS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `RIBB` summary ownership into first shared classic `CAMS` camera-summary ownership for fixed camera metadata and summary-only camera-track metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxCameraSummary.cs` now owns shared per-camera identity, pivot or target-pivot data, fixed clip values, and optional summary-only `KCTR` or `KCRL` or `KVIS` or `KTTR` metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Cameras` and `CameraCount` alongside the earlier `Ribbons` seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `CAMS` entries for `v1300` and `v1400`, including per-camera section sizing, fixed camera payload fields, and optional summary-only `KCTR` or `KCRL` or `KVIS` or `KTTR` metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `CAMS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `CAMS` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` camera regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `24` passing tests after the new synthetic and real Alpha `CAMS` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `cameras=1`, `CHUNK[14]: id=CAMS`, and stable `CAMS[0]: name=Portrait ... positionTrack=none ... targetPositionTrack=none`
- Important boundary:
	- this is still classic `CAMS` summary ownership only; it does not evaluate camera playback, target interpolation, render-camera selection, or runtime portrait behavior
	- it does not replace `MdxViewer` camera handling or claim Alpha runtime camera parity

## Mar 28, 2026 - Shared Classic `MDX` `PRE2` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `RIBB` summary ownership into first shared classic `PRE2` particle-emitter summary ownership for `MDLGENOBJECT`-derived effect metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxParticleEmitter2Summary.cs` now owns shared per-emitter identity, hierarchy, flags, classic scalar particle fields, color or alpha or scale signals, optional model-path presence, spline-count metadata, and summary-only track metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxTrackSummary.cs` now owns shared summary-only metadata for classic non-node track families across both `PRE2` and `RIBB`, including `KP2S`, `KP2E`, `KP2L`, `KPLN`, `KP2G`, `KLIF`, `KP2W`, `KP2N`, `KP2Z`, `KRHA`, `KRHB`, `KRAL`, `KRCO`, and `KRTX`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `ParticleEmitters2` and `ParticleEmitter2Count` alongside the earlier `Ribbons` seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `PRE2` entries for `v1300` and `v1400`, including outer emitter sizing, inner `MDLGENOBJECT` node sizing, classic scalar payload fields, spline-block sizing, and summary-only `KVIS` or `KP2V` plus `KP2S`, `KP2R`, `KP2L`, `KPLN`, `KP2G`, `KLIF`, `KP2E`, `KP2W`, `KP2N`, and `KP2Z` metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `PRE2[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `PRE2` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` particle-emitter regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `22` passing tests after the new synthetic and real Alpha `PRE2` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `particleEmitters2=11`, `CHUNK[12]: id=PRE2`, stable `PRE2[0]: name=BlizParticle01 ... visibilityTrack=KVIS(...)`, and `PRE2[5]: name=BlizParticleBlackDeath ... speedTrack=KP2S(...) emissionRateTrack=KP2E(...)`
- Important boundary:
	- this is still classic `PRE2` summary ownership only; it does not evaluate particle spawn/update behavior, billboarding, UV animation playback, spline motion, or runtime render parity
	- it does not replace `MdxViewer` particle handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `ATCH` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `HELP` summary ownership into first shared attachment summary ownership for classic `MDLGENOBJECT`-derived attachment metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxAttachmentSummary.cs` now owns shared per-attachment identity, hierarchy, flags, attachment-id, optional path, and transform-track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxVisibilityTrackSummary.cs` now owns the shared classic attachment-visibility track metadata contract for `KVIS` or `KATV`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Attachments` and `AttachmentCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `ATCH` entries for `v1300` and `v1400`, including outer attachment-section sizing, inner `MDLGENOBJECT` node sizing, summary-only `KGTR` or `KGRT` or `KGSC` transform metadata, attachment-id/path fields, and optional `KVIS` or `KATV` visibility metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `ATCH[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `ATCH` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` attachment regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `17` passing tests after the new synthetic and real Alpha `ATCH` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `attachments=9`, `CHUNK[10]: id=ATCH`, and real `ATCH[0]` through `ATCH[8]` lines
- Important boundary:
	- this is still classic `ATCH` summary ownership only; it does not resolve attachment paths into assets, evaluate visibility values, or claim attachment-driven runtime render parity
	- it does not replace `MdxViewer` attachment handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `HELP` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `BONE` summary ownership into first shared helper-node summary ownership for classic `MDLGENOBJECT` metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxHelperSummary.cs` now owns shared per-helper identity, hierarchy, flag, and transform-track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxNodeTrackSummary.cs` now owns the generalized shared node-track contract reused by classic `BONE` and `HELP`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Helpers` and `HelperCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `HELP` entries for `v1300` and `v1400`, including `MDLGENOBJECT` name or object-id or parent-id or flag fields plus summary-only `KGTR` or `KGRT` or `KGSC` key-count, interpolation, global-sequence, and time-range metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `HELP[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `HELP` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` helper regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `16` passing tests after the new synthetic and real Alpha `HELP` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `helpers=9`, `CHUNK[9]: id=HELP`, and real `HELP[0]` through `HELP[8]` lines
- Important boundary:
	- this is still classic `HELP` summary ownership only; it does not evaluate node transforms, helper-driven billboards, attachment behavior, or runtime animation playback parity
	- it does not replace `MdxViewer` helper-node handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `BONE` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GEOA` summary ownership into first classic bone/node summary ownership for render-facing skeleton metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxBoneSummary.cs` and `MdxNodeTrackSummary.cs` now own shared per-bone identity, hierarchy, flag, geoset-link, and transform-track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Bones` and `BoneCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `BONE` entries for `v1300` and `v1400`, including `MDLGENOBJECT` name or object-id or parent-id or flag fields plus summary-only `KGTR` or `KGRT` or `KGSC` key-count, interpolation, global-sequence, and time-range metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `BONE[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `BONE` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` bone regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `14` passing tests after the new synthetic and real Alpha `BONE` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `bones=16`, `CHUNK[8]: id=BONE`, and real `BONE[0]` through `BONE[15]` lines
- Important boundary:
	- this is still classic `BONE` summary ownership only; it does not evaluate node transforms, bind pivots to runtime skeleton state, or claim animation playback parity
	- it does not replace `MdxViewer` model skeleton handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `GEOA` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GEOS` structure ownership into first classic geoset-animation summary ownership for render-facing animation metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeosetAnimationSummary.cs` and `MdxGeosetAnimationTrackSummary.cs` now own shared per-entry static color or alpha fields plus `KGAO` or `KGAC` track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `GeosetAnimations` and `GeosetAnimationCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `GEOA` entries for `v1300` and `v1400`, including static header fields and summary-only `KGAO` or `KGAC` key-count, interpolation, global-sequence, and time-range metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `GEOA[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `GEOA` fixture plus an optional real archive-backed `GEOA` probe path across fixed `0.6.0` effect assets
- Current verified validation for this landing:
	- synthetic and real-data `MdxSummaryReaderTests` coverage now includes `GEOA`
	- real Alpha `0.5.3` `MDX` files under `wow-viewer/testdata/0.5.3/tree`, such as `Creature/Wisp/Wisp.mdx`, do carry positive `GEOA` data and are the correct fixed validation surface for this seam
	- the fixed `0.6.0` archive probe set was widened across smoke or torch or brazier or vent assets, but no guaranteed positive `GEOA` carrier was found there
- Important boundary:
	- this is still classic `GEOA` summary ownership only; it does not evaluate animated values, build runtime geoset-visibility state, or claim viewer playback parity
	- it does not replace `MdxViewer` model animation handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `GEOS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past `SEQS` plus `PIVT` internal `MDX` summary ownership: the shared `MDX` seam now also exposes first classic geoset coverage for render-facing mesh structure.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeosetSummary.cs` now owns shared per-geoset summary contracts for core render-facing counts, material linkage, selection or flag fields, optional bounds, and animation-extent count
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Geosets` and `GeosetCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted tagged `GEOS` entries for `v1300` and `v1400`, including vertex, normal, UV, index, matrix, and bone-table counts plus material or bounds summary fields
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `GEOS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic geoset fixture and a real archive-backed `chest01.mdx` geoset case with exact fixed-asset signals
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `8` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt" --virtual-path "world/generic/activedoodads/chest01/chest01.mdx"` passed on Mar 28, 2026 and reported `geosets=2`, `CHUNK[5]: id=GEOS`, and real `GEOS[0]` plus `GEOS[1]` lines with stable counts
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `174` passing tests
- Important boundary:
	- this is still classic `GEOS` summary ownership only; it does not yet decode full mesh payloads, skinning semantics, geoset animations, or runtime render buffers
	- it does not replace `MdxViewer` model loading or claim runtime viewer mesh or skeleton parity

## Mar 28, 2026 - Shared `MDX` `PIVT` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past `SEQS`-only deeper `MDX` summary ownership: the shared `MDX` seam now also exposes first pivot-table coverage.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxPivotPointSummary.cs` now owns shared per-pivot summary contracts for pivot index and pivot position
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `PivotPoints` and `PivotPointCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads `PIVT` as strict `12`-byte `Vector3` entries and preserves the legacy hard-fail behavior for invalid `PIVT` payload sizes
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `PIVT[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers synthetic pivot tables, keeps the real `chest01.mdx` archive-backed summary case, and adds an optional real pivot-positive probe path
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `6` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt" --virtual-path "world/generic/activedoodads/chest01/chest01.mdx"` passed on Mar 28, 2026 and reported `pivotPoints=6`, `CHUNK[8]: id=PIVT`, and real `PIVT[0]` through `PIVT[5]` lines
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `172` passing tests
- Important boundary:
	- this is still `MDX` pivot-table summary ownership only; it does not yet bind pivots onto bones, helpers, emitters, or runtime node transforms
	- it does not replace `MdxViewer` model loading or claim runtime viewer animation or skeleton parity

## Mar 28, 2026 - Shared `MDX` `SEQS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past `TEXS`-plus-`MTLS`-only `MDX` summary ownership: the shared `MDX` seam now also exposes first sequence/animation-summary coverage.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSequenceSummary.cs` now owns shared per-sequence summary contracts for sequence name, time range, move speed, flags, frequency, replay range, optional blend time, and optional bounds
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Sequences` and `SequenceCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads observed `SEQS` summary variants, including counted legacy named `128/132/136/140`-byte records, counted named `0x8C` records, and the numeric-heavy `0x8C` `0.9.0` path as summary-only sequence metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `SEQS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers synthetic legacy-sequence and counted-named-`0x8C` sequence cases, keeps the real `chest01.mdx` archive-backed summary case, and adds an optional real animated-asset probe path
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `4` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --virtual-path world/generic/passivedoodads/particleemitters/greengroundfog.mdx --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and reported `model=GreenGroundFog`, `sequences=1`, `CHUNK[2]: id=SEQS`, and a real `SEQS[0]: name=Stand ... blendTime=150 ...` line
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `170` passing tests
- Important boundary:
	- this is still `MDX` sequence summary ownership only; it does not yet parse animation tracks, bones, pivot tables, geosets, or runtime playback semantics
	- it does not replace `MdxViewer` model loading or claim runtime viewer animation parity

## Mar 28, 2026 - Shared Root-ADT Plus `_tex0` Texture Reader And Broadened JSON Export Landed In `wow-viewer`

- The terrain texture-detail seam is no longer `_tex0`-only. `wow-viewer` now has one shared ADT texture reader for root `ADT` and `_tex0.adt` files.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/` now owns shared `AdtTextureChunkLayer`, `AdtTextureChunk`, and `AdtTextureFile` instead of the earlier `_tex0`-only contract names
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTextureReader.cs` now reads both root `ADT` and `_tex0.adt` files, carrying file kind, decode profile, chunk coordinates, `DoNotFixAlphaMap`, per-layer table data, and decoded overlay alpha payloads through one shared seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalSummaryReader.cs` now aggregates both root and `_tex0` `MCAL` signals through `AdtTextureReader` instead of keeping separate root parsing logic
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` still uses `export-tex-json`, but it now accepts `--input <file.adt|file_tex0.adt>` and emits readable enum strings in JSON output
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now also routes `map inspect --dump-tex-chunks` through the generalized shared reader for root `ADT` and `_tex0.adt`
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtTextureReaderTests.cs` now covers synthetic root and synthetic `_tex0` layer reads plus real root and real `_tex0` development-dataset reads
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTextureReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `37` targeted passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `168` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0.adt | Select-Object -First 20` passed on Mar 28, 2026 and reported root JSON beginning with `Kind: Adt`, `DecodeProfile: LichKingStrict`, empty `TextureNames`, and `Chunks`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt | Select-Object -First 20` passed on Mar 28, 2026 and reported `_tex0` JSON beginning with `Kind: AdtTex`, `DecodeProfile: Cataclysm400`, populated `TextureNames`, and `Chunks`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0.adt --output $env:TEMP\wowviewer-development_0_0-root.json` passed on Mar 28, 2026 and wrote the expected root JSON file
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0.adt --dump-tex-chunks | Select-Object -First 12` passed on Mar 28, 2026 and reported `ADT texture detail: kind=Adt profile=LichKingStrict textures=0 chunks=256`
- Important boundary:
	- this is a shared root-plus-`_tex0` texture-detail read and export seam only
	- the fixed development root dataset currently proves the root command path and chunk metadata path, but not positive real root-layer payload decode because its texture layers live in `_tex0.adt`
	- it still does not port Cataclysm residual-alpha synthesis or neighbor-edge stitching as first-class shared terrain services

## Mar 28, 2026 - Thin `_tex0` JSON Export Surface Landed In `WowViewer.Tool.Converter`

- The new shared `_tex0` reader is no longer inspect-only; `wow-viewer` now has its first thin converter/export consumer for that seam.
- Landed pieces:
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` now accepts `export-tex-json --input <file_tex0.adt> [--output <report.json>]`
	- the converter validates file kind through shared `WowFileDetector` and serializes shared `AdtTexReader` output directly instead of owning a second `_tex0` parser or formatter
	- stdout export and file-write export both now run on the fixed development `_tex0` dataset
- Current verified validation for this landing:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt | Select-Object -First 40` passed on Mar 28, 2026 and printed shared JSON rooted at `SourcePath`, `TextureNames`, and `Chunks`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt --output $env:TEMP\wowviewer-development_0_0_tex0.json` passed on Mar 28, 2026 and wrote the expected JSON file
- Important boundary:
	- this is a thin export surface over the existing shared `_tex0` read seam
	- it does not yet convert terrain into another runtime format, merge root plus split ADT families, or provide a write path back to WoW terrain files

## Mar 28, 2026 - Shared `_tex0` Per-Chunk Layer And Decoded Alpha Reader Landed In `wow-viewer`

- The next terrain ownership slice after split-family routing plus aggregate `MCAL` summary is now landed in `wow-viewer` core/core.io.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/` now also owns shared `AdtTexChunkLayer`, `AdtTexChunk`, and `AdtTexFile` contracts for `_tex0.adt` texture-name tables, per-`MCNK` layer tables, and decoded per-layer alpha payload exposure
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTexReader.cs` now reads `_tex0.adt` files into typed per-chunk layer data and reuses `AdtMcalDecoder` for decoded overlay alpha ownership instead of leaving that detail trapped in inspect-only output or aggregate counters
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/MapSummaryReaderCommon.cs` now exposes reusable string-table extraction through `ReadStringEntries(...)`
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalSummaryReader.cs` now aggregates `_tex0` `MCAL` signals through the new shared `AdtTexReader` instead of re-parsing `_tex0` chunk payloads locally
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `map inspect --input <file.adt> --dump-tex-chunks` and prints typed per-chunk `MCNK(tex)[n]` / `LAYER[n]` detail lines sourced from the shared reader
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtTexReaderTests.cs` now locks both a synthetic `_tex0` fixture and the real `development_0_0_tex0.adt` dataset against the new shared reader seam
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTexReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` targeted passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt --dump-tex-chunks | Select-Object -First 20` passed on Mar 28, 2026 and reported:
		- `ADT TEX detail: textures=5 chunks=256`
		- `MCNK(tex)[0]: xy=(0,0) layers=1 alphaBytes=0 decodedLayers=0`
		- real decoded per-layer `Compressed` and `BigAlpha` outputs later in the dump while preserving the earlier aggregate `ADT MCAL semantics: ... decodedLayers=519 ... compressed=515 bigAlpha=4 ...`
- Important boundary:
	- this is deeper shared `_tex0` read ownership for typed per-chunk layer and direct alpha payload exposure
	- it still does not port the full Cataclysm `TerrainBlend` runtime behavior from `StandardTerrainAdapter`, especially residual-alpha synthesis and neighbor-edge stitching semantics as first-class shared-core services
	- it does not replace the active `MdxViewer` terrain runtime or claim full terrain visual parity by itself

## Mar 28, 2026 - Shared `ADT` Split-Family Routing And Direct `MCAL` Decode Summary Seams Landed In `wow-viewer`

- The first terrain-focused shared-I/O tranche under the full-format-ownership reset is now landed in `wow-viewer` core/core.io.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/` now also owns shared `AdtTileFamily`, `AdtTextureLayerDescriptor`, `AdtMcalDecodeProfile`, `AdtMcalAlphaEncoding`, `AdtMcalDecodedLayer`, and `AdtMcalSummary`
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTileFamilyResolver.cs` now resolves root / `_tex0` / `_obj0` / `_lod` companion paths from any local ADT-family input and exposes preferred texture and placement owners
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalDecoder.cs` now owns a first shared direct-layer `MCAL` decode seam for LK strict and Cataclysm 4.0-style direct payload reads, including compressed alpha, packed 4-bit alpha, direct big-alpha, and the current fixed `63x63 -> 64x64` big-alpha expansion path
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalSummaryReader.cs` now aggregates per-file `MCAL` decode signals across root `ADT` and `_tex0.adt` `MCNK` payloads
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `ADT family:` routing plus `ADT MCAL semantics:` lines during `map inspect`
	- `MapFileKind` and `MapFileSummaryReader` now carry `_lod.adt` through as `AdtLod` instead of dropping it back to `Unknown`
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtMcalDecoderTests|AdtMcalSummaryReaderTests|AdtTileFamilyResolverTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` targeted passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `164` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 28, 2026 and reported:
		- `ADT family: root=present tex0=present obj0=present lod=missing textureSource=AdtTex placementSource=AdtObj`
		- `ADT MCAL semantics: profile=Cataclysm400 ... overlayLayers=519 decodedLayers=519 missingPayloadLayers=0 decodeFailures=0 compressed=515 bigAlpha=4 bigAlphaFixed=0 packed4=0`
- Important boundary:
	- this is direct split-family routing plus direct `MCAL` payload decode ownership only
	- it does not yet port the full Cataclysm `TerrainBlend` runtime behavior from `StandardTerrainAdapter`, especially residual-alpha synthesis and neighbor-chunk stitching semantics as first-class shared-core services
	- it does not replace the active `MdxViewer` terrain runtime or claim full terrain visual parity by itself

## Mar 28, 2026 - wow-viewer Full Format Ownership Reset

- User direction is now explicit: `wow-viewer` is expected to become the first-party owner of every active `MdxViewer` format family, fully, not just through detector or summary seams.
- The migration target is no longer "enough shared readers to inspect files". The target is full parse, decode, write, runtime-service, and tool ownership for the formats the active viewer currently handles.
- Current summary seams in `wow-viewer` remain valid, but they are now only stepping stones toward that larger ownership target.
- A dedicated program document now exists at `gillijimproject_refactor/plans/wow_viewer_full_format_ownership_plan_2026-03-28.md`.
- A family-by-family backlog now also exists at `gillijimproject_refactor/plans/wow_viewer_format_parity_matrix_2026-03-28.md`.
- Immediate high-risk ownership gaps called out by the reset:
	- `ADT` alpha decode and split-file routing parity
	- deep `WMO` ownership beyond current summaries
	- deep `MDX` ownership beyond top-level summary
	- first-party `M2` ownership instead of Warcraft.NET-only behavior
	- first-party `BLP` decode/write ownership instead of SereniaBLPLib-only behavior
	- continued `PM4` extraction until `WorldScene` is no longer the hidden owner of active semantics

## Mar 28, 2026 - Shared `MDX` Top-Level Plus `TEXS` And `MTLS` Summary Seams And `MdxViewer` Consumer Validation Landed

- `wow-viewer` now owns its first narrow `MDX` model-family seam instead of stopping model validation at cross-family detection.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/` now owns shared `MdxChunkIds`, `MdxChunkSummary`, `MdxTextureSummary`, `MdxMaterialLayerSummary`, `MdxMaterialSummary`, and `MdxSummary` contracts for top-level `MDX` header-summary work
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads `MDLX` files through an `MDX`-specific top-level chunk path, including `VERS`, `MODL`, `TEXS`, `MTLS`, chunk order, known-vs-unknown chunk counts, model name, bounds, blend time, texture count, replaceable-texture count, material count, material-layer count, per-texture path/flag summary, and narrow per-material layer summary fields
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `mdx inspect --input <file.mdx>` and `mdx inspect --archive-root <dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers both a synthetic `MDX` fixture and a real standard-archive `world/generic/activedoodads/chest01/chest01.mdx` read
	- `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` now also prints shared `MDX` summary output for probed model bytes, including `textures`, `replaceableTextures`, `materials`, `materialLayers`, the first shared `TEXS` paths, and compact first-layer `MTLS` signals alongside the earlier shared `BLP` texture-summary output
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --virtual-path world/generic/activedoodads/chest01/chest01.mdx --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and reported `version=1300`, `model=Chest01`, `textures=2`, `materials=2`, and real `TEXS[...]` plus `MTLS[0].LAYER[0]` / `MTLS[1].LAYER[0]` lines for the chest asset
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and now reports `SharedMDX: ... textures=2 replaceableTextures=0 materials=2 materialLayers=2 ... firstTextures=... firstMaterials=tex0/blend0/alpha1.000,tex1/blend0/alpha1.000`
- Important boundary:
	- this is shared `MDX` top-level plus narrow `TEXS` and `MTLS` summary ownership only; it does not replace `MdxFile.Load(...)`, animation-track parsing, `M2` handling, or any live viewer render-path model loading
	- real `MDX` chunk ids are stored as direct ASCII on disk, so this seam intentionally uses an `MDX`-specific header decode path instead of the generic reversed-FourCC chunk reader used by ADT/WDT/WMO files
	- this is still build plus inspect/probe validation, not runtime viewer signoff

## Mar 27, 2026 - `MdxViewer` Consumer Validation Now Exercises The Shared `BLP` Seam

- The first shared `BLP` seam in `wow-viewer` is no longer validated only inside `wow-viewer` tools and tests; the active viewer now consumes it through the existing non-UI probe path.
- Landed pieces:
	- `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` now runs shared `WowFileDetector` on the probed model bytes and shared `BlpSummaryReader` on resolved texture bytes classified as `Blp`
	- probe output now shows both shared format signals and the legacy decode-based alpha summary, which keeps the latest library seam visible from the compatibility consumer without forcing a render-path cutover
- Current verified validation for this compatibility step:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 27, 2026 and reported real per-texture `SharedBLP` lines for `CHEST1SIDE.BLP` and `CHEST1FRONT.BLP`
- Important boundary:
	- this is compile plus non-UI consumer validation only
	- it does not prove live viewer rendering parity or a full migration away from `SereniaBLPLib` texture decode

## Mar 27, 2026 - Shared `BLP` Header Summary Seam And Inspect Surface Landed

- The broader shared-I/O gap against `MdxViewer` moved past another WMO-only step: `wow-viewer` now owns a first real `BLP` seam instead of stopping at detector-level classification.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Blp/` now owns the shared `BLP` format, compression, pixel-format, mip-entry, and summary contracts
	- `wow-viewer/src/core/WowViewer.Core.IO/Blp/BlpSummaryReader.cs` now exposes the first shared `BLP` reader seam for `BLP1` and `BLP2` header summary coverage, including compression fields, alpha depth, pixel format, image size, palette or JPEG-header presence, and per-mip offset or size bounds checks
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `blp inspect --input <file.blp>` and `blp inspect --archive-root <dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]`, printing the shared summary plus per-mip lines
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/BlpSummaryReaderTests.cs`, including both `BLP1` and `BLP2` synthetic headers plus a real standard-archive `BLP` read through `MpqArchiveCatalog`
	- `wow-viewer/tests/WowViewer.Core.Tests/WowFileDetectorTests.cs` now also locks direct synthetic `BLP2` detector coverage
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "BlpSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- blp inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path interface/minimap/minimaparrow.blp` passed on Mar 27, 2026 and reported a real `BLP2` summary with `size=32x32`, `pixelFormat=Dxt3`, and `6` in-bounds mip levels
- Important boundary:
	- this is now real first-party `BLP` header-summary ownership plus a thin inspect surface, which materially reduces the gap between `wow-viewer` and the active `MdxViewer` read surface
	- it still does not prove full `BLP` pixel decode ownership, write support, or any model-family (`M2` or `MDX`) seam yet
	- if the next chat says to keep broadening shared-library parity after this landing, the clean next family is `M2` or `MDX`, not another narrow WMO-only refinement

## Mar 27, 2026 - Shared `MOLT` Per-Light Detail Seam And Opt-In Inspect Dump Landed

- Followed the settled root-light summary seam with the next narrow shared-I/O step instead of reopening layout offsets again: `wow-viewer` now owns reusable per-entry `MOLT` detail reads for both legacy Alpha and standard later roots.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLightDetail.cs` now owns the shared per-light contract for `MOLT` entries, including payload offset, entry size, type, attenuation flag, raw BGRA color, position, intensity, attenuation range, and optional standard-layout `headerFlagsWord` plus quaternion rotation fields
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightReaderCommon.cs` now centralizes shared `MOLT` entry-size inference and per-entry field decoding so summary and detail reads stay aligned across Alpha `32`-byte and later `48`-byte layouts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightDetailReader.cs` now exposes the reusable shared per-light detail seam instead of forcing the inspect tool to parse `MOLT` payloads itself
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightSummaryReader.cs` now aggregates through that shared detail decode path instead of duplicating the per-entry layout logic
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `wmo inspect ... --dump-lights` and prints opt-in `MOLT[n]` lines for each root-light entry while keeping the default report summary-only
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoLightDetailReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now proves both Alpha `ironforge.wmo.MPQ` legacy entry details and standard `0.6.0` `world/wmo/khazmodan/cities/ironforge/ironforge.wmo` later-layout detail fields
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoLightDetailReaderTests|Read_IronforgeAlphaPerAssetMpq_ProducesExpectedRootLightSummary|Read_IronforgeAlphaPerAssetMpq_RootLightDetails_UseLegacyLayout|Read_IronforgeStandard060_RootLightSummary_UsesStandardTailAttenuationOffsets|Read_IronforgeStandard060_RootLightDetails_ExposeRawStandardLayoutFields"` passed on Mar 27, 2026 with `8` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo --dump-lights | Select-String '^(MOLT:|MOLT\[0\]:|MOLT\[1\]:)'` passed on Mar 27, 2026 and now reports real standard per-light lines including `MOLT[0]: ... headerFlagsWord=0x0101 ... rotation=(-0.000, 0.000, -1.000, -0.500) ...`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ --dump-lights | Select-String '^(MOLT:|MOLT\[0\]:|MOLT\[1\]:)'` passed on Mar 27, 2026 and now reports real Alpha per-light lines including `MOLT[0]: ... entryBytes=32 ... headerFlagsWord=n/a ... rotation=n/a`
- Important boundary:
	- this proves shared per-entry `MOLT` ownership and an inspect surface that exposes the settled raw fields directly on real Alpha and standard roots
	- it still does not prove the semantic meaning of the later-layout `headerFlagsWord` bits across multiple standard assets or any deeper light rendering behavior
	- if the next chat says to continue the current shared-I/O WMO path without a narrower target, resume from the next standard-root `MOLT` seam: prove whether `headerFlagsWord` varies across additional real `v16` roots now that the raw per-entry dump is available

## Mar 27, 2026 - WMO Group Optional `MOLR`, `MOBN`, `MOBR`, And `MOBN->MOBR` Summary Slice Landed

- The next narrow shared-I/O follow-up stayed inside the existing WMO group summary seam and added ownership for the remaining low-risk optional group chunks instead of jumping into broader group-routing work.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupLightRefSummary.cs`, `WmoGroupBspNodeSummary.cs`, `WmoGroupBspFaceSummary.cs`, and `WmoGroupBspFaceRangeSummary.cs` now own shared group-level summary contracts for `MOLR`, `MOBN`, `MOBR`, and `MOBN -> MOBR`
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupLightRefSummaryReader.cs`, `WmoGroupBspNodeSummaryReader.cs`, `WmoGroupBspFaceSummaryReader.cs`, and `WmoGroupBspFaceRangeSummaryReader.cs` now read those optional group chunks through the existing shared `MOGP` boundary
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now expose group-level `lightRefs`, `bspNodes`, and `bspFaceRefs` counts so inspect and embedded-group aggregate paths do not need tool-local chunk scans
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupSummaryReader.cs` now also aggregate embedded-group `lightRefs`, `bspNodes`, and `bspFaceRefs` totals for Alpha monolithic roots
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated group lines for `MOLR`, `MOBN`, `MOBR`, and `MOBN->MOBR`, and the Alpha `MOGP(root)` aggregate line now also includes `lightRefs`, `bspNodes`, and `bspFaceRefs`
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupLightRefSummaryReaderTests.cs`, `WmoGroupBspNodeSummaryReaderTests.cs`, `WmoGroupBspFaceSummaryReaderTests.cs`, and `WmoGroupBspFaceRangeSummaryReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now validates the optional embedded-group chunk totals on `castle01.wmo.MPQ` and replays the real embedded `MOGP` payloads through the new BSP readers
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupSummaryReaderTests|WmoGroupLightRefSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `9` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports:
		- `MOGP(root): groups=2 ... doodadRefs=24 lightRefs=0 bspNodes=583 bspFaceRefs=6716 ...`
- Important boundary:
	- this proves shared summary ownership for optional group `MOLR`, `MOBN`, `MOBR`, and narrow `MOBN -> MOBR` range coverage
	- it does not yet expose per-embedded-group inspect routing on root files or deeper BSP topology semantics beyond count and range signals

## Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Landed For `MOBN`, `MOBR`, And `MOBN->MOBR`

- Followed the aggregate-only Alpha root work by adding a shared per-embedded-group detail seam instead of leaving root inspect stuck at totals.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupDetail.cs` now owns the per-embedded-group contract for root-embedded `MOGP` details
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupDetailReader.cs` now enumerates root-embedded `MOGP` payloads through shared `WmoGroupSummaryReader`, `WmoGroupLightRefSummaryReader`, `WmoGroupBspNodeSummaryReader`, `WmoGroupBspFaceSummaryReader`, and `WmoGroupBspFaceRangeSummaryReader` without rebuilding temporary group files in the inspect tool
	- the optional group readers now expose internal `ReadMogpPayload(...)` entry points so embedded-root detail routing can reuse the same shared parsing logic directly on real root `MOGP` payloads
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `MOGP(root)[n]`, `MOBN(root)[n]`, `MOBR(root)[n]`, and `MOBN->MOBR(root)[n]` lines for Alpha monolithic roots with embedded groups
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupDetailReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now validates the per-group detail reader on `castle01.wmo.MPQ`
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `8` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports real per-group lines including:
		- `MOBN(root)[0]: payloadBytes=2032 nodes=127 ...`
		- `MOBR(root)[0]: payloadBytes=2290 refs=1145 ...`
		- `MOBN->MOBR(root)[0]: nodes=127 faceRefs=1145 zeroFaceNodes=92 coveredNodes=35 outOfRangeNodes=0 maxFaceEnd=1145`
		- `MOBN(root)[1]: payloadBytes=7296 nodes=456 ...`
		- `MOBR(root)[1]: payloadBytes=11142 refs=5571 ...`
		- `MOBN->MOBR(root)[1]: nodes=456 faceRefs=5571 zeroFaceNodes=237 coveredNodes=219 outOfRangeNodes=0 maxFaceEnd=5571`
- Important boundary:
	- this proves real per-embedded-group inspect routing for the existing shared BSP summaries on Alpha `MOMO` roots
	- it still does not expose full per-embedded-group routing for every group subchunk family or deeper BSP traversal semantics

## Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Expanded To Existing Shared Group Summaries

- Followed the first BSP-only per-group landing by broadening the shared embedded-group detail seam instead of adding another tool-local root formatter.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupDetail.cs` now also carries per-group shared summaries for `MLIQ`, `MOBA`, `MOPY`, `MOTV`, `MOCV`, `MODR`, `MOVI` or `MOIN`, `MOVT`, and `MONR`
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupLiquidSummaryReader.cs`, `WmoGroupBatchSummaryReader.cs`, `WmoGroupFaceMaterialSummaryReader.cs`, `WmoGroupUvSummaryReader.cs`, `WmoGroupVertexColorSummaryReader.cs`, `WmoGroupDoodadRefSummaryReader.cs`, `WmoGroupIndexSummaryReader.cs`, `WmoGroupVertexSummaryReader.cs`, and `WmoGroupNormalSummaryReader.cs` now expose internal `ReadMogpPayload(...)` entry points so root-embedded `MOGP` detail reads can reuse the same shared parsing logic directly on payload bytes
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupDetailReader.cs` now populates those additional shared group summaries when the per-group `MOGP` header reports the relevant counts or liquid presence
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints additional per-group root lines for `MONR(root)[n]`, `MOVT(root)[n]`, `MOVI(root)[n]` or `MOIN(root)[n]`, `MODR(root)[n]`, `MOCV(root)[n]`, `MOTV(root)[n]`, `MOPY(root)[n]`, `MOBA(root)[n]`, and `MLIQ(root)[n]` when present
	- synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupDetailReaderTests.cs` now proves those additional detail summaries on embedded synthetic `MOGP` payloads
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now proves the broadened detail reader against `castle01.wmo.MPQ`
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `4` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports additional real per-group lines including:
		- `MONR(root)[0]: payloadBytes=16488 normals=1374 ...`
		- `MOVT(root)[0]: payloadBytes=16488 vertices=1374 ...`
		- `MOIN(root)[0]: payloadBytes=2664 indices=1332 ...`
		- `MODR(root)[0]: payloadBytes=48 refs=24 ...`
		- `MOCV(root)[0]: payloadBytes=5496 primaryColors=1374 ...`
		- `MOTV(root)[0]: payloadBytes=10992 primaryUv=1374 ...`
		- `MOPY(root)[0]: payloadBytes=1832 entryBytes=4 faces=458 ...`
		- `MOBA(root)[0]: payloadBytes=192 entries=8 ...`
		- matching positive lines also appear for root group `1`, with `MODR(root)[1]` correctly absent because that embedded group has zero doodad refs
- Important boundary:
	- this proves the shared embedded-group detail seam can now surface the already-owned geometry or metadata group summaries directly on Alpha root `MOGP` payloads
	- real `castle01.wmo.MPQ` still does not positively prove `MOLR(root)[n]` or `MLIQ(root)[n]`, because its embedded groups report zero light refs and no liquid

## Mar 27, 2026 - `ironforge.wmo.MPQ` Added Positive Real Coverage For `MOLR(root)` And `MLIQ(root)`

- Switched the missing positive real-data proof from `castle01.wmo.MPQ` to `wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ`, because that real Alpha monolithic root actually exercises the remaining per-group light-ref and liquid seams.
- Landed pieces:
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now includes a real-data regression that proves `WmoEmbeddedGroupDetailReader` sees embedded groups with non-zero `LightRefSummary` and non-null `LiquidSummary` on `ironforge.wmo.MPQ`
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now treats invalid optional `MOLT` payload reads as non-fatal for inspect output, matching the existing optional-chunk behavior used for other root summaries, so the real Ironforge asset can continue through later root and embedded-group lines instead of aborting early
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests"` passed on Mar 27, 2026 with `4` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ` passed far enough on Mar 27, 2026 to print positive per-group lines including:
		- `MOLR(root)[120]: payloadBytes=10 refs=5 ...`
		- `MOLR(root)[121]: payloadBytes=20 refs=10 ...`
		- `MOLR(root)[123]: payloadBytes=2 refs=1 ...`
		- `MOLR(root)[124]: payloadBytes=10 refs=5 ...`
		- `MLIQ(root)[127]: payloadBytes=6457 verts=30x24 tiles=29x23 ... liquidType=Magma`
- Important boundary:
	- this proves positive real-data ownership for the remaining per-group `MOLR(root)` and `MLIQ(root)` inspect lines on an Alpha monolithic root
	- the underlying `MOLT` reader still does not claim full compatibility with Ironforge's real root-light payload layout; inspect now simply does not let that optional root-summary failure block later shared outputs

## Mar 27, 2026 - Shared `MOLT` Root-Light Summary Now Reads Real Alpha `ironforge.wmo.MPQ`

- Followed the non-fatal inspect guard with the actual shared-library fix: `WowViewer.Core.IO.Wmo.WmoLightSummaryReader` now supports both the legacy 32-byte Alpha light entries and the later 48-byte root-light entries instead of assuming only the later size.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightSummaryReader.cs` now infers `MOLT` entry size from version and payload shape, using 32-byte entries for Alpha `v14` roots and 48-byte entries for later roots
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLightSummary.cs` and `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now expose and print `attenStartRange`, a raw later-layout `headerFlagsWord` summary from bytes `2..3`, and later-layout rotation metrics (`rotationEntries`, `nonIdentityRotations`, `rotationLenRange`) alongside the existing intensity and `maxAttenEnd` metrics
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoLightSummaryReaderTests.cs` now covers both synthetic `v14` 32-byte `MOLT` payloads and synthetic `v17` 48-byte payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now verifies the real Ironforge root light summary directly, including the exact `218` light count, `6976` payload bytes, and positive attenuation-start range from `ironforge.wmo.MPQ`
	- the same real-data test surface now also loads `world/wmo/khazmodan/cities/ironforge/ironforge.wmo` from the shared `0.6.0` standard MPQ set via `MpqArchiveCatalog` + the vendored `wow-listfile`, proving that 48-byte standard `MOLT` entries carry a non-zero `headerFlagsWord` of `0x0101` at bytes `2..3`, quaternion rotation at offsets `24..39`, and attenuation values at offsets `40` and `44`
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/ArchiveVirtualFileReader.cs` now owns the shared “read a virtual file from standard archive roots” seam, and `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now consumes it for `wmo inspect --archive-root <dir> --virtual-path <world/...wmo>` with default vendored-listfile discovery
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ | Select-String '^(WMO semantics:|MOLT:|MFOG:)'` passed on Mar 27, 2026 and now reports:
		- `WMO semantics: ... lights=218 ...`
		- `MOLT: payloadBytes=6976 entries=218 distinctTypes=1 attenuated=218 intensityRange=[0.120, 1.000] attenStartRange=[1.306, 8.333] maxAttenEnd=29.611 ...`
		- `MFOG: payloadBytes=96 entries=2 ...`
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|Read_IronforgeAlphaPerAssetMpq_ProducesExpectedRootLightSummary|Read_IronforgeStandard060_RootLightSummary_UsesStandardTailAttenuationOffsets"` passed on Mar 27, 2026 with `4` targeted passing tests, including the real `0.6.0` standard-archive Ironforge root-light case
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo | Select-String '^(Version:|WMO semantics:|MOLT:|MFOG:)'` passed on Mar 27, 2026 and now reports the real standard root-light summary through the inspect CLI, including `headerFlagsWordRange=[0x0101, 0x0101]`, `headerFlagsWordDistinct=1`, `headerFlagsWordNonZero=218`, `rotationEntries=218`, `nonIdentityRotations=218`, and `rotationLenRange=[1.118, 1.118]`
- Important boundary:
	- this proves the shared root `MOLT` semantic-summary seam on a real Alpha monolithic root instead of only surviving past a failure
	- it also now proves the real standard `v16` attenuation offsets for 48-byte entries, so the shared reader no longer reports zero attenuation on standard roots
	- it also now proves that `WowViewer.Tool.Inspect` can consume the shared standard-archive seam directly for root WMO virtual paths instead of requiring an extracted loose file or per-asset Alpha MPQ wrapper
	- it still does not prove deeper light rendering semantics beyond the existing count, raw `headerFlagsWord`, attenuation, attenuation-start range, rotation-shape summary, intensity, and bounds contract; the current real proof only locks Ironforge's standard `0x0101` word, not the per-bit meaning or cross-asset variability yet
	- the follow-up per-light inspect dump has now landed; the next standard-root `MOLT` seam is to prove whether `headerFlagsWord` varies across additional real `v16` roots

## Mar 27, 2026 - Alpha `MOGI -> MOGP(root)` Linkage Summary Landed

- After landing the Alpha embedded-group aggregate, the next narrow follow-up linked root `MOGI` entries to the embedded top-level `MOGP` blocks by ordinal pairing instead of jumping straight to full monolithic group routing.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupLinkageSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupLinkageSummaryReader.cs` now own the narrow Alpha `MOGI -> MOGP(root)` linkage seam
	- the linkage summary reports `MOGI` entry count, embedded `MOGP` count, covered pairs, missing/extra groups, flag matches, bounds matches, and maximum bounds delta across paired groups
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints an `MOGI->MOGP(root)` line for Alpha monolithic roots when both surfaces are present
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupLinkageSummaryReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now validates the linkage summary on `castle01.wmo.MPQ`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `130` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupLinkageSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports:
		- `MOGI->MOGP(root): infos=2 groups=2 coveredPairs=2 missingGroups=0 extraGroups=0 flagMatches=0 boundsMatches=2 maxBoundsDelta=0.000`
- Important boundary:
	- this proves count/flag/bounds linkage across paired Alpha root group-info and embedded-group surfaces
	- it does not yet expose standalone per-embedded-group inspect routing or detailed per-group diff output

## Mar 27, 2026 - Alpha Monolithic Root Embedded-Group Aggregate Summary Landed

- With Alpha `MOMO` root support working on real `castle01.wmo.MPQ`, the next narrow follow-up landed on the root file's embedded top-level `MOGP` blocks instead of jumping straight to full monolithic group-consumer cutover.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupSummaryReader.cs` now own an aggregate embedded-group summary for Alpha monolithic root files with top-level `MOGP` chunks
	- the aggregate covers embedded-group count, header-size range, groups with portals, groups with liquid, total faces, vertices, indices, normals, batches, doodad refs, and aggregate bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now exposes a reusable internal `MOGP` payload summary helper so the embedded-root aggregate can reuse the same group-header interpretation instead of duplicating it
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints an `MOGP(root)` aggregate line when a root WMO actually contains embedded top-level `MOGP` chunks
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupSummaryReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now also validates the embedded-group aggregate against `castle01.wmo.MPQ`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `129` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports:
		- `MOGP(root): groups=2 headerBytes=128-128 groupsWithPortals=2 groupsWithLiquid=0 faces=2371 vertices=7113 indices=6195 normals=7113 batches=22 doodadRefs=24 ...`
- Important boundary:
	- this is an embedded-group aggregate seam for Alpha monolithic roots
	- it does not yet expose per-embedded-group detailed mesh summaries or direct monolithic-group selection/routing in inspect

## Mar 27, 2026 - Alpha MOMO Root WMO Support And Real 0.5.3 `.wmo.MPQ` Validation Landed

- Real Alpha-era WMO validation exposed an important boundary gap in the shared root-WMO readers:
	- `castle01.wmo.MPQ` from `wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/` extracts to a v14 Alpha monolithic WMO root
	- the file starts `MVER` then `MOMO`, not the later split-root `MVER` then `MOHD` layout
	- pre-fix `wow-viewer` classified the extracted bytes as `Unknown`, so real 0.5.3 root-WMO validation could not run through the shared root-summary stack
- Landed support:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOMO`
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/WowFileDetector.cs` now recognizes `MVER` + `MOMO` as a root `Wmo`
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRootReaderCommon.cs` now expands Alpha `MOMO` root subchunks into a flattened root-chunk view so shared root readers can keep using readable FourCC ownership on both Alpha monolithic roots and later split roots
	- shared root readers that previously only scanned top-level chunks now route through `WmoRootReaderCommon`, including `WmoSummaryReader`, `WmoGroupInfoSummaryReader`, `WmoMaterialSummaryReader`, `WmoTextureTableSummaryReader`, `WmoDoodadNameTableSummaryReader`, `WmoDoodadSetSummaryReader`, `WmoDoodadPlacementSummaryReader`, `WmoGroupNameTableSummaryReader`, `WmoSkyboxSummaryReader`, and the portal-root helper in `WmoPortalVertexSummaryReader`
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupInfoSummary.cs` now allows negative `MOGI` name offsets, which real Alpha data exposed as valid sentinel-style values
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/AlphaArchiveReader.cs` now builds broader non-map `World\...` internal-name candidates and uses them even when the input path itself ends with `.MPQ`
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now opens `.MPQ` WMO inputs through shared Alpha archive fallback and runs the shared stream-based readers, so `wmo inspect` works directly on real per-asset Alpha archives
	- real-data regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Concrete real-data proof now available:
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026
	- reported key semantic lines:
		- `Version: 14`
		- `WMO semantics: materials=11/11 groups=2/2 portals=1 ... doodadPlacements=24/24 doodadSets=1/1`
		- `MOPT->MOPV: portals=1 vertices=4 zeroVertexPortals=0 coveredPortals=1 outOfRangePortals=0 maxVertexEnd=4`
		- `MOPR->MOPT: refs=2 portals=1 coveredRefs=2 outOfRangeRefs=0 distinctPortalRefs=1 maxPortalIndex=0`
		- `MOPR->MOGI: refs=2 groups=2 coveredRefs=2 outOfRangeRefs=0 distinctGroupRefs=2 maxGroupIndex=1`
		- `MOMT: payloadBytes=484 entryBytes=44 entries=11 ...`
		- `MOGI: payloadBytes=80 entryBytes=40 entries=2 ... nameOffsetRange=-1--1 ...`
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `128` passing tests after the Alpha `MOMO` support and real-data WMO coverage
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AlphaArchiveReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` targeted passing tests
- Important boundary:
	- this adds shared Alpha-root summary ownership for `MOMO`-wrapped root chunks and direct inspect support for `.wmo.MPQ` inputs
	- it does not yet add Alpha monolithic-group mesh summary ownership beyond the root summaries already extracted from `MOMO`

## Mar 27, 2026 - Batched Root WMO Portal Linkage Summary Slices For MOPT->MOPV, MOPR->MOPT, And MOPR->MOGI Landed

- `wow-viewer` now has a portal-linkage focused batched root-WMO landing that builds on the earlier raw portal summaries instead of stopping at count-only payload ownership:
	- `MOPT -> MOPV` portal-vertex range coverage summary
	- `MOPR -> MOPT` portal-ref range coverage summary
	- `MOPR -> MOGI` portal-group range coverage summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalVertexRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalVertexRangeSummaryReader.cs` now own narrow `MOPT -> MOPV` linkage semantics for zero-vertex portals, covered portals, out-of-range portals, total visible portal vertices, and max vertex end
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalRefRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalRefRangeSummaryReader.cs` now own narrow `MOPR -> MOPT` linkage semantics for covered refs, out-of-range refs, distinct referenced portals, and max portal index
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalGroupRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalGroupRangeSummaryReader.cs` now own narrow `MOPR -> MOGI` linkage semantics for covered refs, out-of-range refs, distinct referenced groups, and max group index
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRootReaderCommon.cs` now exposes optional root-chunk reads so root readers can distinguish truly absent chunks instead of accidentally treating the first chunk as a match during optional lookup flows
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated portal-linkage lines for `MOPT->MOPV`, `MOPR->MOPT`, and `MOPR->MOGI`, and it now tolerates missing optional dependency chunks instead of aborting synthetic smoke cases
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalVertexRangeSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalRefRangeSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalGroupRangeSummaryReaderTests.cs`, plus a missing-`MOVV` regression in `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleVertexSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `125` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portal-linkage-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOPT->MOPV: portals=2 vertices=6 zeroVertexPortals=0 coveredPortals=1 outOfRangePortals=1 maxVertexEnd=8`
		- `MOPR->MOPT: refs=3 portals=2 coveredRefs=2 outOfRangeRefs=1 distinctPortalRefs=3 maxPortalIndex=4`
		- `MOPR->MOGI: refs=3 groups=3 coveredRefs=2 outOfRangeRefs=1 distinctGroupRefs=3 maxGroupIndex=5`
- Important boundary:
	- these seams prove portal-linkage range coverage only
	- they do not yet prove full portal topology validation, plane correctness, or runtime culling behavior

## Mar 27, 2026 - Batched Root WMO Visibility Summary Slices For MOVV, MOVB, And MOVB->MOVV Landed

- `wow-viewer` now has another batched root-WMO follow-up landing covering the two visibility-owner chunks plus their first narrow linkage seam together:
	- `MOVV` visible-vertex semantic summary
	- `MOVB` visible-block semantic summary
	- `MOVB -> MOVV` visible-block range coverage summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoVisibleVertexSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoVisibleVertexSummaryReader.cs` now own narrow `MOVV` semantics for payload size, visible-vertex counts, and computed bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoVisibleBlockSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoVisibleBlockSummaryReader.cs` now own narrow `MOVB` semantics for block counts, total vertex refs, per-block vertex-count range, first-vertex range, and max vertex end
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoVisibleBlockReferenceSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoVisibleBlockReferenceSummaryReader.cs` now own narrow `MOVB -> MOVV` linkage semantics for zero-vertex blocks, covered blocks, out-of-range blocks, visible-vertex counts, and max vertex end
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MOVV`, `MOVB`, and `MOVB->MOVV` semantic lines for root WMO files when those chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleVertexSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleBlockSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleBlockReferenceSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `121` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-visibility-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOVV: payloadBytes=72 vertices=6 boundsMin=(-4.00, -8.00, -6.00) boundsMax=(7.00, 5.00, 9.00)`
		- `MOVB: payloadBytes=12 blocks=3 vertexRefs=7 blockSizeRange=0-4 firstVertexRange=0-5 maxVertexEnd=8`
		- `MOVB->MOVV: blocks=3 vertices=6 zeroVertexBlocks=1 coveredBlocks=1 outOfRangeBlocks=1 maxVertexEnd=8`
- Important boundary:
	- these seams prove count, bounds, and simple block-to-vertex coverage only
	- they do not yet prove runtime visibility-volume semantics, convexity validation, or any write path

## Mar 27, 2026 - Batched Root WMO Linkage Summary Slices For MODD->MODN, MOGI->MOGN, And MODS->MODD Landed

- `wow-viewer` now has a linkage-focused batched root-WMO landing instead of another raw-payload-only step:
	- `MODD -> MODN` doodad-name reference summary
	- `MOGI -> MOGN` group-name reference summary
	- `MODS -> MODD` doodad-set range summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadNameReferenceSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadNameReferenceSummaryReader.cs` now own narrow `MODD -> MODN` linkage semantics for resolved-name counts, unresolved-name counts, distinct resolved names, and max resolved-name length
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupNameReferenceSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupNameReferenceSummaryReader.cs` now own narrow `MOGI -> MOGN` linkage semantics for resolved-name counts, unresolved-name counts, distinct resolved names, and max resolved-name length
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadSetRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadSetRangeSummaryReader.cs` now own narrow `MODS -> MODD` range semantics for empty-set counts, fully covered sets, out-of-range sets, placement counts, and max range end
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRootReaderCommon.cs` now centralizes shared root-WMO chunk reads, version reads, root-kind validation, and string-at-offset resolution used by the linkage readers
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated linkage lines for `MODD->MODN`, `MOGI->MOGN`, and `MODS->MODD` when the needed root chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadNameReferenceSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNameReferenceSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadSetRangeSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `118` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-linkage-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MODS->MODD: sets=3 placements=3 emptySets=1 coveredSets=0 outOfRangeSets=2 maxRangeEnd=18`
		- `MOGI->MOGN: entries=3 resolvedNames=2 unresolvedNames=1 distinctResolvedNames=2 maxNameLength=9`
		- `MODD->MODN: entries=3 resolvedNames=2 unresolvedNames=1 distinctResolvedNames=2 maxNameLength=7`
- Important boundary:
	- these seams prove narrow cross-chunk linkage and range validation only
	- they do not yet prove full root-name resolution ownership across every consumer path or any write path

## Mar 27, 2026 - Batched Root WMO Metadata Slices For MOLT, MFOG, And MCVP Landed

- `wow-viewer` now has another batched root-WMO metadata landing covering lights, fog, and one opaque trailing root chunk together:
	- `MOLT` light semantic summary
	- `MFOG` fog semantic summary
	- `MCVP` opaque-chunk byte summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLightSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightSummaryReader.cs` now own narrow `MOLT` semantics for entry counts, distinct light types, attenuation usage, intensity range, attenuation-end range, and light bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoFogSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoFogSummaryReader.cs` now own narrow `MFOG` semantics for entry counts, non-zero flag counts, radius ranges, fog-end range, and fog bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoOpaqueChunkSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoOpaqueChunkSummaryReader.cs` now provide a thin shared seam for byte-count reporting of opaque root chunks like `MCVP`
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOLT`, `MFOG`, `MCVP`, plus root `MOVV` and `MOVB` ids for continued root-chunk ownership work
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MOLT`, `MFOG`, and `MCVP` semantic lines for root WMO files when those chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoLightSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoFogSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoOpaqueChunkSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `115` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-meta-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOLT: payloadBytes=96 entries=2 distinctTypes=2 attenuated=1 intensityRange=[4.000, 8.000] maxAttenEnd=20.000 boundsMin=(-4.00, 2.00, -6.00) boundsMax=(1.00, 5.00, 3.00)`
		- `MFOG: payloadBytes=96 entries=2 nonZeroFlags=1 minSmallRadius=1.000 maxLargeRadius=7.000 maxFogEnd=11.000 boundsMin=(-4.00, 2.00, -6.00) boundsMax=(1.00, 5.00, 3.00)`
		- `MCVP: payloadBytes=12`
- Important boundary:
	- these seams prove light or fog count-level semantics plus a byte-count seam for opaque `MCVP`
	- they do not yet prove deeper light/fog rendering semantics, `MCVP` structure ownership, or any write path

## Mar 27, 2026 - Batched Root WMO Portal Summary Slices For MOPV, MOPT, And MOPR Landed

- `wow-viewer` now has a second batched root-WMO landing covering the three portal-owner chunks together:
	- `MOPV` portal-vertex semantic summary
	- `MOPT` portal-info semantic summary
	- `MOPR` portal-ref semantic summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalVertexSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalVertexSummaryReader.cs` now own narrow `MOPV` semantics for vertex counts and computed bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalInfoSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalInfoSummaryReader.cs` now own narrow `MOPT` semantics for portal-entry counts, max start vertex, max vertex count, and plane-D range
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalRefSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalRefSummaryReader.cs` now own narrow `MOPR` semantics for ref counts, distinct portal counts, max group index, and side distribution
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOPV`, `MOPT`, and `MOPR` chunk ids
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MOPV`, `MOPT`, and `MOPR` semantic lines for root WMO files when portal data is present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalVertexSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalInfoSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalRefSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `112` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `81` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portals-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOPV: payloadBytes=36 vertices=3 boundsMin=(-4.00, -8.00, -6.00) boundsMax=(7.00, 5.00, 9.00)`
		- `MOPT: payloadBytes=40 entries=2 maxStartVertex=10 maxVertexCount=4 planeDRange=[-2.000, 1.000]`
		- `MOPR: payloadBytes=24 entries=3 distinctPortals=2 maxGroupIndex=7 sides(+/-/0)=1/1/1`
- Important boundary:
	- these three seams prove portal-owner count and range semantics only
	- they do not yet prove full root-to-group portal routing behavior or any write path

## Mar 27, 2026 - Batched Root WMO Summary Slices For MODD, MOGN, And MOSB Landed

- `wow-viewer` now has a batched set of three additional narrow root-WMO seams instead of a one-slice landing:
	- `MODD` doodad-placement semantic summary
	- `MOGN` group-name table semantic summary
	- `MOSB` skybox semantic summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadPlacementSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadPlacementSummaryReader.cs` now own count-level `MODD` semantics for entry counts, distinct name indices, scale range, alpha range, and placement bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupNameTableSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupNameTableSummaryReader.cs` now own narrow `MOGN` string-table semantics for count, longest entry, and max offset
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoSkyboxSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoSkyboxSummaryReader.cs` now own the narrow `MOSB` seam for payload size and resolved skybox name
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOGN` and `MOSB` chunk ids
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MODD`, `MOGN`, and `MOSB` semantic lines for root WMO files when those chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadPlacementSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNameTableSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoSkyboxSummaryReaderTests.cs`

- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `109` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `78` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOSB: payloadBytes=4 skybox=Sky`
		- `MOGN: payloadBytes=31 names=3 longestEntry=10 maxOffset=21`
		- `MODD: payloadBytes=80 entries=2 distinctNameIndices=2 maxNameIndex=7 scaleRange=[1.250, 2.500] alphaRange=[170, 255] boundsMin=(-4.00, 2.00, -6.00) boundsMax=(1.00, 5.00, 3.00)`
- Important boundary:
	- these three seams prove string-table and placement-summary ownership only
	- they do not yet prove `MODD` linkage back to `MODN`, `MOGN` name resolution against group metadata, or any write path

## Mar 27, 2026 - Shared WMO Root Doodad-Set Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MODN`: a shared `MODS` doodad-set semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadSetSummary.cs` now owns the typed root `MODS` summary contract for payload size, set count, non-empty-set count, longest set-name length, total doodad refs, max start index, and max range end
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadSetSummaryReader.cs` now reads `MODS` payload semantics from root WMO files as a narrow doodad-set seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MODS` semantic line for root WMO files that contain doodad sets
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadSetSummaryReaderTests.cs` now covers a synthetic `MODS` table with empty and non-empty sets
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `106` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `75` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mods-test.wmo` passed on Mar 27, 2026 and reported `MODS: payloadBytes=96 entries=3 nonEmptySets=2 longestName=7 totalDoodadRefs=10 maxStartIndex=12 maxRangeEnd=18`
- Important boundary:
	- this proves shared `MODS` semantic summary for doodad-set counts and range signals only
	- this does not yet prove set-to-`MODD` linkage beyond count-level ranges or any write path

## Mar 27, 2026 - Shared WMO Root Doodad-Name Table Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MOTX`: a shared `MODN` doodad-name-table semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadNameTableSummary.cs` now owns the typed root `MODN` summary contract for payload size, name count, longest entry length, max string offset, distinct extension counts, and `.mdx` or `.m2` entry counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadNameTableSummaryReader.cs` now reads `MODN` payload semantics from root WMO files as a narrow string-table seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MODN` semantic line for root WMO files that contain doodad-name tables
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadNameTableSummaryReaderTests.cs` now covers a synthetic `MODN` table with mixed `.mdx` and `.m2` entries plus a nested path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `105` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `74` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-modn-test.wmo` passed on Mar 27, 2026 and reported `MODN: payloadBytes=31 names=3 longestEntry=15 maxOffset=15 extensions=2 mdxEntries=2 m2Entries=1`
- Important boundary:
	- this proves shared `MODN` semantic summary for string-table counts and extension-shape signals only
	- this does not yet prove offset resolution against `MODD`, path canonicalization, or any write path

## Mar 27, 2026 - Shared WMO Root Texture-Table Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MOMT`: a shared `MOTX` texture-table semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoTextureTableSummary.cs` now owns the typed root `MOTX` summary contract for payload size, texture count, longest entry length, max string offset, distinct extension counts, and `.blp` entry counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoTextureTableSummaryReader.cs` now reads `MOTX` payload semantics from root WMO files as a narrow table-summary seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOTX` semantic line for root WMO files that contain texture tables
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoTextureTableSummaryReaderTests.cs` now covers a synthetic `MOTX` table with mixed texture extensions and nested paths
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `104` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `73` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-motx-test.wmo` passed on Mar 27, 2026 and reported `MOTX: payloadBytes=33 textures=3 longestEntry=16 maxOffset=16 extensions=2 blpEntries=2`
- Important boundary:
	- this proves shared `MOTX` semantic summary for table counts and string-shape signals only
	- this does not yet prove offset resolution against `MOMT`, path canonicalization, or any write path

## Mar 27, 2026 - Shared WMO Root Material Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MOGI`: a shared `MOMT` material semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoMaterialSummary.cs` now owns the typed root `MOMT` summary contract for payload size, inferred entry size, entry count, distinct shader counts, distinct blend-mode counts, non-zero-flag counts, and maximum texture offsets across the first three slots
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoMaterialSummaryReader.cs` now reads standard, legacy, and vintage `MOMT` payload semantics from root WMO files using `MOHD` material-count guidance when available
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOMT` semantic line for root WMO files that contain material entries
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoMaterialSummaryReaderTests.cs` now covers synthetic standard 64-byte and legacy 44-byte `MOMT` payloads
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `103` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `72` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-momt-test.wmo` passed on Mar 27, 2026 and reported `MOMT: payloadBytes=128 entryBytes=64 entries=2 distinctShaders=2 distinctBlendModes=2 nonZeroFlags=1 maxTex1Ofs=24 maxTex2Ofs=20 maxTex3Ofs=88`
- Important boundary:
	- this proves shared `MOMT` semantic summary for material-entry layout and selected top-level fields only
	- this does not yet prove texture-name resolution against `MOTX`, color interpretation, or any write path

## Mar 27, 2026 - Shared WMO Root Group-Info Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after the group-level payload summaries: a shared `MOGI` group-info semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupInfoSummary.cs` now owns the typed root `MOGI` summary contract for payload size, inferred entry size, entry count, distinct-flag counts, non-zero-flag counts, name-offset range, and union bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupInfoSummaryReader.cs` now reads standard and legacy `MOGI` payload semantics from root WMO files using `MOHD` group-count guidance when available
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOGI` semantic line for root WMO files that contain group info
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupInfoSummaryReaderTests.cs` now covers synthetic standard 32-byte and legacy 40-byte `MOGI` payloads
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `101` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `70` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mogi-test.wmo` passed on Mar 27, 2026 and reported `MOGI: payloadBytes=64 entryBytes=32 entries=2 distinctFlags=2 nonZeroFlags=1 nameOffsetRange=12-40 boundsMin=(-7.00, -2.00, -3.00) boundsMax=(4.00, 8.00, 9.00)`
- Important boundary:
	- this proves shared `MOGI` semantic summary for root group-info entry counts, flag coverage, name-offset ranges, and union bounds only
	- this does not yet prove root-to-group file linkage beyond raw entry counts, name resolution against `MOGN`, or any write path

## Mar 27, 2026 - Shared WMO Group Normal Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOVT`: a shared `MONR` normal semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupNormalSummary.cs` now owns the typed normal-summary contract for payload size, normal count, component ranges, length ranges, average length, and near-unit counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupNormalSummaryReader.cs` now reads `MONR` payload semantics from WMO group files as a narrow count-and-range seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MONR` semantic line for WMO group files that contain normal payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNormalSummaryReaderTests.cs` now covers a synthetic `MONR` payload with two unit-length normals and one shorter vector
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `99` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `68` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-normal-test.wmo` passed on Mar 27, 2026 and reported `MONR: payloadBytes=36 normals=3 rangeX=[0.000, 1.000] rangeY=[-1.000, 0.500] rangeZ=[0.000, 0.500] lengthRange=[0.866, 1.000] avgLength=0.955 nearUnit=2`
- Important boundary:
	- this proves shared `MONR` semantic summary for count, component ranges, and length ranges only
	- this does not yet prove tangent-space ownership, generated-normal fallback logic, or any write path

## Mar 27, 2026 - Shared WMO Group Vertex Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOVI`: a shared `MOVT` vertex semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupVertexSummary.cs` now owns the typed vertex-summary contract for payload size, vertex count, and computed vertex bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupVertexSummaryReader.cs` now reads `MOVT` payload semantics from WMO group files as a narrow count-and-bounds seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOVT` semantic line for WMO group files that contain vertex payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexSummaryReaderTests.cs` now covers a synthetic `MOVT` payload with mixed positive and negative coordinates
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `98` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `67` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-vertex-test.wmo` passed on Mar 27, 2026 and reported `MOVT: payloadBytes=36 vertices=3 boundsMin=(-4.00, -8.00, -6.00) boundsMax=(7.00, 5.00, 9.00)`
- Important boundary:
	- this proves shared `MOVT` semantic summary for count and computed bounds only
	- this does not yet prove topology linkage, coordinate ownership beyond the payload, or any write path

## Mar 27, 2026 - Shared WMO Group Index Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MODR`: a shared `MOVI` or `MOIN` index semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupIndexSummary.cs` now owns the typed index-summary contract for chunk id, payload size, index count, triangle count, distinct index count, index range, and degenerate-triangle count
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupIndexSummaryReader.cs` now reads either `MOVI` or `MOIN` payload semantics from WMO group files as a narrow count-level seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOVI` or `MOIN` semantic line for WMO group files that contain index payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupIndexSummaryReaderTests.cs` now covers synthetic `MOVI` and `MOIN` payloads including a degenerate-triangle case
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `97` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `66` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-index-test.wmo` passed on Mar 27, 2026 and reported `MOVI: payloadBytes=12 indices=6 triangles=2 distinctIndices=4 indexRange=0-3 degenerateTriangles=1`
- Important boundary:
	- this proves shared `MOVI` or `MOIN` semantic summary for count, range, and degenerate-triangle coverage only
	- this does not yet prove topology ownership, face-material alignment, or any write path

## Mar 27, 2026 - Shared WMO Group Doodad-Ref Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOCV`: a shared `MODR` doodad-ref semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupDoodadRefSummary.cs` now owns the typed `MODR` summary contract for ref counts, distinct ref counts, min or max ref range, and duplicate-ref counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupDoodadRefSummaryReader.cs` now reads `MODR` payload semantics from WMO group files as a narrow count-level seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MODR` semantic line for WMO group files that contain doodad refs
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupDoodadRefSummaryReaderTests.cs` now covers a synthetic `MODR` payload with duplicate refs
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `95` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `64` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-doodadref-test.wmo` passed on Mar 27, 2026 and reported `MODR: payloadBytes=8 refs=4 distinctRefs=3 refRange=3-9 duplicateRefs=1`
- Important boundary:
	- this proves shared `MODR` semantic summary for doodad-ref counts and ranges only
	- this does not yet prove linkage back to root doodad tables, placement ownership, or any write path

## Mar 27, 2026 - Shared WMO Group Vertex-Color Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOTV`: a shared `MOCV` vertex-color semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupVertexColorSummary.cs` now owns the typed `MOCV` summary contract for primary color payload size, primary color count, BGRA-derived channel ranges, average alpha, and extra color-set counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupVertexColorSummaryReader.cs` now reads `MOCV` payload semantics from WMO group files while keeping the primary set separate from optional extra color sets
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOCV` semantic line for WMO group files that contain vertex colors
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexColorSummaryReaderTests.cs` now covers a synthetic WMO group with one primary and one extra `MOCV` set
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `63` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-color-test.wmo` passed on Mar 27, 2026 and reported `MOCV: payloadBytes=8 primaryColors=2 rangeR=[30, 70] rangeG=[20, 60] rangeB=[10, 50] rangeA=[40, 80] avgA=60 extraColorSets=1 totalExtraColors=3 maxExtraColors=3`
- Important boundary:
	- this proves shared `MOCV` semantic summary for count, channel-range, and extra-set coverage only
	- this does not yet prove runtime lighting interpretation, second color-set semantics, or any write path

## Mar 27, 2026 - Shared WMO Group UV Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOPY`: a shared `MOTV` UV semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupUvSummary.cs` now owns the typed `MOTV` UV-summary contract for primary UV payload size, primary UV count, primary U or V ranges, additional UV-set counts, and aggregate extra-UV counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupUvSummaryReader.cs` now reads `MOTV` payload semantics from WMO group files while keeping the primary set separate from optional extra UV sets
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOTV` semantic line for WMO group files that contain UV data
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupUvSummaryReaderTests.cs` now covers a synthetic WMO group with one primary and one extra `MOTV` set
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `93` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `62` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-uv-test.wmo` passed on Mar 27, 2026 and reported `MOTV: payloadBytes=24 primaryUv=3 rangeU=[-0.200, 0.800] rangeV=[0.200, 0.900] extraUvSets=1 totalExtraUv=2 maxExtraUv=2`
- Important boundary:
	- this proves shared `MOTV` UV semantic summary for counts and value ranges only
	- this does not yet prove runtime UV-set selection, secondary-set semantics, or any write path

## Mar 27, 2026 - Shared WMO Group Face-Material Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOBA`: a shared `MOPY` face-material semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupFaceMaterialSummary.cs` now owns the typed `MOPY` face-material summary contract for face counts, inferred entry size, distinct material ids, highest material id, hidden-face count, and flagged-face count
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupFaceMaterialSummaryReader.cs` now reads `MOPY` payload semantics from WMO group files while respecting v17 two-byte and v16 four-byte entry layouts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupReaderCommon.cs` now exposes shared `MOPY` entry-size inference used by both count-level and face-material readers
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOPY` semantic line for WMO group files that contain face-material entries
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupFaceMaterialSummaryReaderTests.cs` now covers synthetic v17-style and v16-style `MOPY` payloads
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `92` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `61` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-face-v17-test.wmo` passed on Mar 27, 2026 and reported `MOPY: payloadBytes=8 entryBytes=2 faces=4 distinctMaterials=2 highestMaterialId=7 hiddenFaces=1 flaggedFaces=2`
- Important boundary:
	- this proves shared `MOPY` face-material semantic summary for count, flag, and material-id coverage only
	- this does not yet prove face-to-batch reconstruction, material resolution against root tables, or any write path

## Mar 27, 2026 - Shared WMO Group Batch Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MLIQ`: a shared `MOBA` batch semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupBatchSummary.cs` now owns the typed `MOBA` batch-summary contract for entry counts, material-id coverage, total index count, first-index range, max index end, and flagged-batch counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupBatchSummaryReader.cs` now reads `MOBA` payload semantics from WMO group files without pretending to own full batch reconstruction
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOBA` semantic line for WMO group files that contain batches
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupBatchSummaryReaderTests.cs` now covers synthetic v17-style material-bearing batches and v16-style material-less batches
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `59` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-batch-test.wmo` passed on Mar 27, 2026 and reported `MOBA: payloadBytes=48 entries=2 hasMaterialIds=True distinctMaterials=2 highestMaterialId=7 totalIndexCount=15 firstIndexRange=10-20 maxIndexEnd=29 flaggedBatches=1`
- Important boundary:
	- this proves shared `MOBA` batch semantic summary for top-level batch-entry counts and index or material signals only
	- this does not yet prove full batch reconstruction, bounding-box interpretation, or write-path ownership

## Mar 27, 2026 - Shared WMO Group Liquid Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after the `MOGP` header summary: a shared `MLIQ` semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLiquidBasicType.cs` now owns the basic liquid-family enum used by the summary seam
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupLiquidSummary.cs` now owns the typed `MLIQ` semantic-summary contract for liquid dimensions, corner, material id, height range, tile-flag coverage, visible tile count, and inferred liquid family
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupReaderCommon.cs` now centralizes `MOGP` payload reads, header-size detection, subchunk enumeration, and shared helper logic used by both WMO group summary readers
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupLiquidSummaryReader.cs` now reads `MLIQ` payload semantics from WMO group files without pretending to own runtime mesh generation
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now consumes the new shared `WmoGroupReaderCommon` helper instead of carrying its own `MOGP` scanning copy
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MLIQ` semantic line for WMO group files that contain liquid
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupLiquidSummaryReaderTests.cs` now covers a synthetic `MLIQ` payload with ocean inference and height-range validation
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `88` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `57` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-liquid-test.wmo` passed on Mar 27, 2026 and reported `MLIQ: payloadBytes=63 verts=2x2 tiles=1x1 ... visibleTiles=1/1 ... liquidType=Ocean`
- Important boundary:
	- this proves shared `MLIQ` payload semantic summary for dimensions, height range, visible tile counts, and basic family inference only
	- this does not yet prove full WMO liquid mesh generation, orientation fitting, or any write path

## Mar 27, 2026 - Shared WMO Group Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO follow-up seam after the root summary: a shared WMO group semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupSummary.cs` now owns the typed WMO group semantic-summary contract for `MOGP` header fields, declared batch counts, geometry subchunk counts, optional extra UV-set count, doodad-ref count, and liquid presence
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now reads standard `MOGP` group files at count or presence level without pretending to own deep mesh reconstruction
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now also owns the shared readable WMO group subchunk ids used by the new seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/WowFileDetector.cs` now recognizes `MOGP`-first files as `WmoGroup` instead of treating them as unknown when `MVER` is absent
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now routes `wmo inspect` to either the root-WMO or group-WMO reader based on shared detection and prints a dedicated WMO group report for group files
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupSummaryReaderTests.cs` now covers synthetic `MVER + MOGP` and `MOGP`-first group files
	- `wow-viewer/tests/WowViewer.Core.Tests/WowFileDetectorTests.cs` now locks `MOGP`-first detection as `WmoGroup`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `56` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-summary-test.wmo` passed on Mar 27, 2026 and reported `Header: bytes=68 ... Geometry: faces=3 vertices=2 indices=3 ... hasLiquid=False`
- Important boundary:
	- this proves shared WMO group semantic summary for `MOGP` headers and top-level geometry or metadata subchunk counts only
	- this does not yet prove full group mesh decode, batch reconstruction, liquid payload ownership, or any write path

## Mar 27, 2026 - Shared ADT MCNK Semantic Summary And First WMO Root Summary Slices Landed

- `wow-viewer` now has its first chunk-internal ADT semantic seam plus its first shared WMO root semantic-summary seam.
- Landed ADT pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/AdtChunkIds.cs` now owns the shared readable `MCNK` subchunk ids used by the ADT MCNK summary seam
	- `wow-viewer/src/core/WowViewer.Core/Maps/AdtMcnkSummary.cs` now owns the typed ADT `MCNK` semantic-summary contract for root-header presence, index coverage, area-id coverage, hole or liquid or `MCCV` flags, subchunk presence, and per-chunk layer-count signals
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcnkSummaryReader.cs` now reads those `MCNK` signals from root, `_tex0.adt`, and `_obj0.adt` files while staying at count or presence level instead of deep payload decode
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports the shared ADT `MCNK` semantic summary for `map inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtMcnkSummaryReaderTests.cs` now covers synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus fixed real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Landed WMO pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now owns the shared readable root-WMO chunk ids used by the summary seam
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoSummary.cs` now owns the typed WMO root semantic-summary contract for `MOHD`-reported counts, string-table counts, top-level entry counts, flags, and bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoSummaryReader.cs` now reads those signals from standard chunked WMO root files without pretending to be group-file or deep payload ownership
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `wmo inspect --input <file.wmo>` as a thin shared-reader consumer
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoSummaryReaderTests.cs` now covers a synthetic WMO root summary case
- Current verified validation for these slices:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `53` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now reports the shared ADT `MCNK` semantic line `mcnk=256 zero=0 headerLike=0 distinctIndex=0 duplicateIndex=0 areaIds=0 holes=0 liquidFlags=0 mccvFlags=0 mcly=256 mcal=203 mcsh=174 totalLayers=775 maxLayers=4 multiLayerChunks=203`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-summary-test.wmo` passed on Mar 27, 2026 and reported `materials=2/2 groups=4/4 portals=1 lights=3 textures=2 doodadNames=5/5 doodadPlacements=6/6 doodadSets=2/2 flags=0x00001234`
- Important boundaries:
	- the ADT `MCNK` seam proves count or presence level ownership for root-header signals and split-file subchunk coverage, not full terrain payload decode, alpha decode, shadow decode, liquid decode, or writer support
	- the WMO seam proves root-file semantic summary only; it does not yet prove group-file parsing, material payload ownership beyond entry counts, or any write path

## Mar 27, 2026 - Shared ADT Semantic Summary Slice Landed

- `wow-viewer` now has its first shared ADT semantic-summary seam beyond raw top-level chunk inventory.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/AdtSummary.cs` now owns the typed ADT semantic-summary contract for terrain-chunk counts, texture-name counts, doodad or WMO name counts, placement counts, and top-level MFBO or MH2O or MAMP or MTXF presence
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtSummaryReader.cs` now reads those signals from root, `_tex0.adt`, and `_obj0.adt` files without pretending to be a deep payload parser
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/MapSummaryReaderCommon.cs` now centralizes the shared top-level chunk-payload and string-block helpers used by both `AdtSummaryReader` and `WdtSummaryReader`
	- `wow-viewer/src/core/WowViewer.Core/Maps/MapChunkIds.cs` now includes `MAMP` so texture-parameter presence is expressed as a shared map chunk id instead of a tool-local literal
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports the shared ADT semantic summary for `map inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtSummaryReaderTests.cs` now covers synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus fixed real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `77` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `46` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and reported `kind=AdtTex terrainChunks=256 textures=5 doodadNames=0 wmoNames=0 doodadPlacements=0 wmoPlacements=0 hasMfbo=False hasMh2o=False hasMamp=True hasMtxf=False`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 27, 2026 and reported `kind=AdtObj terrainChunks=256 textures=0 doodadNames=6 wmoNames=12 doodadPlacements=10 wmoPlacements=15 hasMfbo=False hasMh2o=False hasMamp=False hasMtxf=False`
- Important boundary:
	- this proves shared ADT semantic summary for top-level terrain-chunk counts, string-table counts, placement counts, and selected presence flags across root and split ADT-family files
	- this does not yet prove deep root ADT parsing, split-texture payload parsing, split-object payload parsing, chunk-internal MCNK semantics, or any write path

## Mar 27, 2026 - Shared WDT Semantic Summary Slice Landed

- `wow-viewer` now has its first shared WDT semantic-summary seam beyond raw top-level chunk inventory.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/WdtSummary.cs` now owns the typed WDT semantic-summary contract for MPHD WMO-based flags, MAIN occupancy, string-table counts, and top-level placement counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/WdtSummaryReader.cs` now reads those signals from either Alpha-style or standard WDT top-level chunks without pretending to be a full payload parser
	- `wow-viewer/src/core/WowViewer.Core/Maps/MapChunkIds.cs` now includes `MDNM` and `MONM` so the shared reader can treat Alpha name tables as first-class chunk ids instead of tool-local literals
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports the shared WDT semantic summary for `map inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/WdtSummaryReaderTests.cs` now covers synthetic standard WDT, synthetic Alpha WDT, and the fixed real-data `development.wdt` semantic signals
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `71` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt` passed on Mar 27, 2026 and reported `wmoBased=False tiles=1496/4096 mainCellBytes=8 doodadNames=0 wmoNames=0 doodadPlacements=0 wmoPlacements=0`
- Important boundary:
	- this proves shared WDT semantic summary for top-level MPHD, MAIN, string-table, and placement-count signals
	- this does not yet prove deep WDT payload parsing, WMO placement semantics beyond counts, or any write path

## Mar 27, 2026 - Shared AreaIdMapper Archive-Backed Loading Replaced Constructor-Time Extracted-Tree Probing

- The primary `AreaIdMapper` load path is now archive-backed instead of constructor-time test-data probing in `WoWMapConverter.Core.Converters.AlphaToLkConverter`.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` now exposes `TryLoadFromArchives(...)`, reading `AreaTable` and `Map` through shared `IArchiveReader` plus `DbClientFileReader` and feeding DBCD through an in-memory provider instead of a staged file tree
	- shorthand archive build inputs `0.5.3` and `3.3.5` now normalize to the full WoWDBDefs-compatible build strings the DBCD seam actually needs
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs` no longer calls `TryAutoLoadFromTestData()` in its constructor; it now initializes the mapper lazily from explicit DBC paths or explicit Alpha and LK archive roots, then falls back to CSV crosswalks only if those inputs fail
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` now accepts `--alpha-client` and `--lk-client` so converter runs can point directly at MPQ roots
	- `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` now covers synthetic archive-backed DBCD loading and explicit archive-missing diagnostics
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `37` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor
- Important boundary:
	- this proves the shared area-mapper seam can now consume archive-backed DBC bytes without extracted trees
	- this does not yet include a real client-root converter smoke test against Alpha and LK MPQ inputs in this workspace

## Mar 27, 2026 - Shared AreaIdMapper DBCD Wiring And Explicit Fallback Warning Landed

- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` now prefers real schema-aware loading through DBCD when extracted `AreaTable` and `Map` files are present and `WoWDBDefs` definitions can be discovered from the workspace.
- Landed pieces:
	- `WowViewer.Core.IO.csproj` now references the same vendored `gillijimproject_refactor/lib/wow.tools.local/DBCD/DBCD/DBCD.csproj` project the active viewer already uses, and bundles `gillijimproject_refactor/lib/WoWDBDefs/definitions` into output
	- shared `AreaIdMapper` now discovers `WoWDBDefs/definitions` from the bundled `definitions` output first, then from `gillijimproject_refactor/lib/WoWDBDefs/definitions`, `wow-viewer/libs/wowdev/WoWDBDefs/definitions`, `libs/wowdev/WoWDBDefs/definitions`, or legacy `lib/WoWDBDefs/definitions`
	- shared `AreaIdMapper.LoadDbcs(...)` now uses DBCD plus WoWDBDefs for known `0.5.3` and `3.3.5` paths when available, then falls back to the narrow raw `DbcReader` only when schema-backed loading is unavailable
	- shared `AreaIdMapper.TryAutoLoadFromTestData()` and `TryLoadKnownTestDataFromRoot(...)` now prefer `gillijimproject_refactor/test_data/*/tree/DBFilesClient` before legacy `test_data/*/tree/DBFilesClient`, and record explicit diagnostics instead of silently failing when extracted tables are missing
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs` now surfaces that missing-tree diagnostic as a runtime warning before falling back to crosswalk-only behavior
	- added focused shared-library regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` for explicit missing-tree reporting and a synthetic DBCD+WoWDBDefs-backed `AreaTable`/`Map` load path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `66` tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor and no new build break
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert i:/parp/parp-tools/gillijimproject_refactor/test_data/0.5.3/alphawdt/World/Maps/PVPZone01/PVPZone01.wdt -o i:/parp/parp-tools/output/pvpzone01-alpha-to-lk-smoke-dbcd-check3 -v` passed on Mar 27, 2026 and now emits one explicit warning that names the preferred `gillijimproject_refactor/test_data/*/tree/DBFilesClient` roots first when extracted DBC trees are absent
- Important boundary:
	- this proves the shared area-mapper seam is now actually wired to DBCD plus WoWDBDefs when the extracted table trees exist
	- the current real-data runtime smoke tests in this workspace still fall back because the extracted `gillijimproject_refactor/test_data/0.5.3/tree/DBFilesClient/*` and `gillijimproject_refactor/test_data/3.3.5/tree/DBFilesClient/*` files are absent here
	- this is still narrow `AreaTable` and `Map` ownership for the mapper seam, not broad general DBC or DB2 format ownership across all tables

## Mar 26, 2026 - Shared AreaIdMapper And Crosswalk Ownership Landed

- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` now owns the remaining live old-repo area-mapping seam plus the embedded area-crosswalk resource it depended on.
- Landed pieces:
	- added shared `AreaIdMapper`
	- moved `area_crosswalk.csv` into `wow-viewer/src/core/WowViewer.Core.IO/Resources/area_crosswalk.csv`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs` now uses shared `WowViewer.Core.IO.Dbc.AreaIdMapper`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/AreaIdMapper.cs`
	- deleted dead `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/AreaIdCrosswalk.cs`
	- deleted the old embedded `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Resources/area_crosswalk.csv`
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` for embedded-default loading, matching-report CSV parsing, and continent-hinted name matching
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `64` tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 26, 2026 with `3` warnings
- Important boundary:
	- this proves shared ownership of the active old-repo area-ID mapping seam and its embedded crosswalk data plus consumer compile validation
	- this does not prove broader DBC schema ownership beyond the narrow shared mapper or runtime converter signoff on real data
	- no runtime validation was run

## Mar 26, 2026 - Shared Alpha MPQ Old-Repo Caller Cutover Landed

- The shared `wow-viewer/src/core/WowViewer.Core.IO/Files/AlphaArchiveReader.cs` seam now owns the remaining active old-repo per-asset MPQ callers that were still using the deleted duplicate reader in `WoWMapConverter.Core`.
- Landed pieces:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now uses shared `AlphaArchiveReader`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` now uses shared `AlphaArchiveReader`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17ExtendedConverter.cs` now uses shared `AlphaArchiveReader`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/AlphaMpqReader.cs`
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AlphaArchiveReaderTests.cs` for per-asset MPQ block selection and companion `.MPQ` fallback
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `61` tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
- Important boundary:
	- this proves shared ownership of the active old-repo Alpha per-asset MPQ caller seam plus consumer compile validation
	- this does not prove broader WMO, MDX, or BLP format ownership beyond read access through the shared Alpha archive seam
	- no viewer runtime validation was run

## Mar 26, 2026 - Dead Old DBC Helper Cleanup Landed

- The old `WoWMapConverter.Core` archive or DBC helper layer left behind after the shared `Core.IO` cutovers has now been narrowed and cleaned up instead of being carried forward as dead compatibility code.
- Landed pieces:
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/NativeMpqService.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/Md5TranslateResolver.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/MapDbcService.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/GroundEffectService.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/DbcReader.cs`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/AreaIdMapper.cs` now reads tables through shared `WowViewer.Core.IO.Dbc.DbcReader`
- Current live-boundary result from the targeted review:
	- the deleted helper files were definition-only in the active `gillijimproject_refactor/src` tree after the earlier `Core.IO` cutovers
	- `AreaIdMapper` remains the only clearly live DBC-backed seam still owned by `WoWMapConverter.Core`
	- the current active consumer of that seam is `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs`
- Current verified validation for this slice:
	- workspace diagnostics for `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core` reported no errors after the cleanup
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `54` warnings and no new build break
- Important boundary:
	- this is old-repo cleanup plus dependency-boundary tightening, not a new `wow-viewer` library slice
	- no new `wow-viewer` tests were run in this pass because the shared library code did not change
	- the next real DBC ownership decision is whether `AreaIdMapper` and the Alpha-to-LK area crosswalk should move into `wow-viewer`

## Mar 26, 2026 - Shared DBC Lookup And VLM Archive Cutover Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the next narrow non-PM4 table-backed helper slice that was still stranded in `WoWMapConverter.Core`.
- Landed pieces:
	- shared `DbcReader`
	- shared `DbcHeader`
	- shared `MapDirectoryLookup`
	- shared `GroundEffectLookup`
	- expanded shared `DbClientFileReader` table probing to cover `DBFilesClient`, `DBC`, and root `.dbc` or `.db2` candidates
	- focused regression coverage for shared DBC lookup behavior in `wow-viewer/tests/WowViewer.Core.Tests/DbcLookupTests.cs`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now uses shared `IArchiveCatalog` or `IArchiveReader` instead of `WoWMapConverter.Core.Services.NativeMpqService`
	- `VlmDatasetExporter` now resolves `Map.dbc` through shared `MapDirectoryLookup`
	- `VlmDatasetExporter` now resolves ground-effect doodads through shared `GroundEffectLookup`
	- `VlmDatasetExporter` now loads MD5 minimap translation through shared callback-based `WowViewer.Core.IO.Files.Md5TranslateResolver`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `59` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with the existing warning floor plus no new build break
- Important boundary:
	- this proves shared DBC-backed lookup ownership plus active VLM consumer compile validation
	- this does not prove general DBC or DB2 format ownership, write support, or viewer runtime behavior
	- `MdxViewer` was not rebuilt in this slice because the change targeted `WowViewer.Core.IO` plus `WoWMapConverter.Core`
	- the old `MapDbcService`, `GroundEffectService`, local `DbcReader`, old `Md5TranslateResolver`, and `NativeMpqService` helper layer was later deleted from `WoWMapConverter.Core` once the active-path review showed it was dead in the current tree
	- `AreaIdMapper` is now the remaining active DBC-backed seam in `WoWMapConverter.Core`, and it already reads through shared `Core.IO`

## Mar 26, 2026 - Concrete Shared MPQ Catalog Port Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the concrete standard MPQ implementation used by the active `MdxViewer` path, not just the contracts and bootstrap helpers around it.
- Landed pieces:
	- shared `MpqArchiveCatalog`
	- shared `MpqArchiveCatalogFactory`
	- internal `MpqDiagnostics`
	- focused regression coverage for archive priority, patched-delete fallback, internal listfile extraction, and direct file-0 reads in `wow-viewer/tests/WowViewer.Core.Tests/MpqArchiveCatalogTests.cs`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now defaults to shared `MpqArchiveCatalogFactory`
	- deleted the active bridge file `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`
	- active `MdxViewer` `.cs` source no longer instantiates or references `WoWMapConverter.Core.Services.NativeMpqService` in its standard MPQ consumer path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `57` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing `32` warning floor
- Important boundary:
	- this proves concrete shared MPQ catalog ownership for the active `MdxViewer` path plus consumer compile validation
	- older `WoWMapConverter.Core.Services.NativeMpqService` code still exists for other non-migrated old-repo consumers, but it is no longer the active `MdxViewer` standard MPQ implementation path
	- no viewer runtime validation was run

## Mar 26, 2026 - Shared Archive Bootstrap And Alpha Wrapper Cutovers Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the next two archive-adjacent seams that `MpqDataSource` was still keeping locally or routing directly to old services.
- Landed pieces:
	- shared `ArchiveCatalogBootstrapper`
	- shared `ArchiveCatalogBootstrapResult`
	- shared `AlphaArchiveReader`
	- shared `PkwareExplode`
	- focused regression coverage for external listfile parsing, archive bootstrap aggregation, Alpha internal-name candidate generation, and direct-file fallback behavior in `wow-viewer/tests/WowViewer.Core.Tests`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now uses shared `ArchiveCatalogBootstrapper` instead of owning the standard archive bootstrap or external listfile parsing path locally
	- `MpqDataSource` now uses shared `AlphaArchiveReader` instead of directly calling `WoWMapConverter.Core.Services.AlphaMpqReader`
	- the active `MdxViewer` source no longer references the old Alpha wrapper reader in its MPQ data source path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `53` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutovers
- Important boundary:
	- this proves shared archive bootstrap and shared Alpha wrapper helper ownership plus consumer compile validation
	- `NativeMpqService` still remains behind `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`; the concrete standard MPQ implementation is not ported yet
	- no viewer runtime validation was run

## Mar 26, 2026 - Shared Archive-Reader MPQ Cutover Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the shared archive-reader or archive-catalog boundary that `MdxViewer` was still expressing directly through `WoWMapConverter.Core.Services.NativeMpqService`.
- Landed pieces:
	- shared `IArchiveReader`
	- shared `IArchiveCatalog`
	- shared `IArchiveCatalogFactory`
	- shared `DbClientFileReader` for `DBFilesClient` DBC or DB2 path probing
	- focused regression coverage for DBC or DB2 candidate ordering and first-match table reads in `wow-viewer/tests/WowViewer.Core.Tests`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now depends on shared archive interfaces instead of `NativeMpqService`
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDBCProvider.cs` now reads tables through shared `IArchiveReader` and `DbClientFileReader`
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now consumes `MpqDataSource.ArchiveReader` instead of `MpqService`
	- direct `NativeMpqService` ownership is isolated to `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `49` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutover
- Important boundary:
	- this proves shared archive-reader ownership plus consumer compile validation
	- `NativeMpqService` itself is not ported into `wow-viewer`; it is still the current implementation behind the compatibility adapter
	- Alpha wrapper reads still call `WoWMapConverter.Core.Services.AlphaMpqReader` directly inside `MpqDataSource`; that is a separate seam from standard MPQ archive access
	- no viewer runtime validation was run

## Mar 26, 2026 - Shared MD5 Minimap Translation Cutover Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the shared MD5 minimap translation seam that `MdxViewer` was still importing from `WoWMapConverter.Core.Services`.
- Landed pieces:
	- shared `Md5TranslateIndex`
	- shared `Md5TranslateResolver.TryLoad(...)` with archive read callbacks instead of direct `NativeMpqService` type ownership
	- shared `MinimapService.GetMinimapTilePath(...)` and `MinimapTileExists(...)`
	- focused regression coverage for map-specific TRS loading and `dir:` directory-context parsing in `wow-viewer/tests/WowViewer.Core.Tests`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now loads the MD5 minimap translation index through shared `WowViewer.Core.IO.Files.Md5TranslateResolver`
	- `gillijimproject_refactor/src/MdxViewer/Rendering/MinimapRenderer.cs` and `Export/MapGlbExporter.cs` now consume shared `Md5TranslateIndex` and `MinimapService`
	- `MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `ViewerApp` no longer pulls the default development-map directory from `WoWMapConverter.Core.Services.DevelopmentMapAnalyzer`; it now uses shared `Pm4CoordinateService.DefaultDevelopmentMapDirectory`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `47` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutover
- Important boundary:
	- this proves shared MD5 minimap translation ownership plus consumer compile validation
	- `MdxViewer` still depends on `WoWMapConverter.Core` for broader MPQ, terrain, converter, and VLM subsystems; that wider cutover is still open
	- no viewer runtime validation was run

## Mar 26, 2026 - PM4 Linked-Position-Ref Summary Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the linked MPRL position-ref summary seam that was still being aggregated inside `WorldScene`.
- Landed pieces:
	- shared `Pm4LinkedPositionRefSummary` contract
	- shared `Pm4PlacementMath.SummarizeLinkedPositionRefs(...)`
	- focused regression coverage for mixed normal-or-terminator linked refs and terminator-only fallback behavior
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now maps local `MprlEntry` values into shared `Core.PM4` position-ref entries and delegates linked-ref summary aggregation to `Core.PM4`
	- the viewer-local heading-range, floor-range, and circular-mean aggregation no longer owns that PM4 seam
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `31` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-linked-position-ref-summary-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this proves shared PM4 linked-position-ref summary ownership plus consumer compile validation
	- no PM4 inspect or viewer runtime validation was run in this slice because analyzer or report output did not change
	- this is not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Placement-Solution Consumer Hookup Landed

- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now delegates PM4 placement-solution assembly to the already-landed `Core.PM4` placement-solution seam.
- Landed pieces:
	- the CK24 overlay path now calls shared `Pm4PlacementMath.ResolvePlacementSolution(...)` instead of resolving planar transform, world pivot, and world yaw correction as separate consumer-owned steps
	- local per-piece consumer wrappers for those already-shared PM4 placement pieces no longer own that path
- Current verified validation for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-placement-solution-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this slice reuses an already-tested `Core.PM4` placement-solution seam; no new `wow-viewer` library code changed
	- the currently recorded `31` PM4-test and `45` total-test floor is the latest library-test proof
	- this is consumer compile validation only, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Connector-Key Consumer Hookup Landed

- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now delegates PM4 connector-key derivation to the already-landed `Core.PM4` connector-key seam.
- Landed pieces:
	- `BuildCk24ConnectorKeys()` now builds a shared `Pm4PlacementSolution` and delegates connector-key derivation to `Pm4PlacementMath.BuildConnectorKeys(...)`
	- local viewer-owned connector-point conversion and quantization logic no longer owns that PM4 grouping input path
- Current verified validation for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-connector-key-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this slice reuses an already-tested `Core.PM4` connector-key seam; no new `wow-viewer` library code changed
	- the currently recorded `31` PM4-test and `45` total-test floor is the latest library-test proof
	- this is consumer compile validation only, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Merge-Map Consumer Hookup Landed

- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now delegates PM4 merged-group resolution to the already-landed `Core.PM4` merge-map seam.
- Landed pieces:
	- `RebuildPm4MergedObjectGroups()` now builds shared `Pm4ConnectorMergeCandidate` inputs and delegates canonical merge resolution to `Pm4PlacementMath.BuildMergedGroupMap(...)`
	- local viewer-owned union-find and merge-heuristic logic no longer owns that PM4 grouping path
- Current verified validation for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-merge-map-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this slice reuses an already-tested `Core.PM4` merge-map seam; no new `wow-viewer` library code changed
	- the currently recorded `31` PM4-test and `45` total-test floor is the latest library-test proof
	- this is consumer compile validation only, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Correlation Geometry-Input Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the next PM4-only correlation seam: geometry-input assembly for shared object-state construction.
- Landed pieces:
	- shared `Pm4GeometryLineSegment` contract
	- shared `Pm4GeometryTriangle` contract
	- shared `Pm4CorrelationGeometryInput` contract
	- `Pm4CorrelationMath.BuildObjectStatesFromGeometry(...)`
	- regression coverage for building shared PM4 correlation object states directly from PM4 line or triangle geometry plus a transform
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now maps PM4 overlay lines and triangles into shared PM4 geometry-input contracts and delegates object-state construction to `Core.PM4`
	- local viewer-specific world-point flattening for PM4 correlation object-state assembly is no longer the owner of that seam
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `29` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-geometry-hookup/` passed on Mar 26, 2026
- Important boundary:
	- PM4-owned geometry, transforms, and shared object-state construction belong in `Core.PM4`
	- WMO-facing correlation report payloads stay in WMO or consumer space and should not be moved into PM4 just because the report compares PM4 against WMO
	- this proves shared PM4 geometry-input ownership plus consumer compile validation, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Correlation Object-State Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the next reusable correlation object-state seam for PM4 placement or report work.
- Landed pieces:
	- shared `Pm4CorrelationObjectDescriptor` contract
	- shared `Pm4CorrelationObjectInput` contract
	- shared `Pm4CorrelationObjectState` contract
	- `Pm4CorrelationMath.BuildObjectStates(...)`
	- public `Pm4CorrelationMath.BuildFootprintHull(...)`, `BuildTransformedFootprintHull(...)`, and `ComputeFootprintArea(...)`
	- regression coverage for synthetic object-state bounds or footprint derivation, empty-geometry fallback, and transformed footprint-hull construction
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now builds shared correlation object inputs and consumes shared correlation states
	- `BuildPm4WmoPlacementCorrelationReport(...)` now uses shared hull and metric helpers from `Core.PM4` instead of duplicating that scoring path locally
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `28` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `42` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-state-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this proves shared object-state, hull, and scoring consumption plus consumer compile validation, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Correlation-Math Library Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the next reusable correlation-scoring seam for PM4 placement or report work.
- Landed pieces:
	- shared `Pm4CorrelationMetrics` contract
	- shared `Pm4CorrelationCandidateScore` contract
	- `Pm4CorrelationMath.EvaluateMetrics(...)`
	- `Pm4CorrelationMath.CompareCandidateScores(...)`
	- library-owned planar-gap, vertical-gap, footprint-distance, polygon-overlap, footprint-area-ratio, planar-overlap, and AABB-overlap helpers extracted from the current `WorldScene` correlation logic
	- regression coverage for synthetic metric calculation and ranking precedence
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `25` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `39` total tests
- Important boundary:
	- no active-viewer consumer wiring changed in this slice
	- this proves library-owned correlation metrics and ranking, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Connector-Group Merge Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the first connector-based group-merge seam for PM4 grouping work.
- Landed pieces:
	- shared `Pm4ObjectGroupKey` contract
	- shared `Pm4ConnectorMergeCandidate` contract
	- `Pm4PlacementMath.BuildMergedGroupMap(...)`
	- library-owned connector-overlap, bounds-padding, and center-distance merge heuristics extracted from the current `WorldScene`
	- regression coverage for neighbor-tile merge resolution and same-tile non-merge behavior
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` total tests
- Important boundary:
	- no active-viewer consumer wiring changed in this slice
	- this proves library-owned merge heuristics, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Connector-Key Library Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the first reusable connector-key extraction seam for PM4 grouping or correlation work.
- Landed pieces:
	- shared `Pm4ConnectorKey` contract
	- `Pm4PlacementMath.BuildConnectorKeys(...)`
	- library-owned conversion of `MSUR.MdosIndex` exterior vertices into quantized world-space connector keys through typed `Pm4PlacementSolution`
	- dedupe and deterministic ordering of connector keys
	- regression coverage for distinct sorted connector extraction and yaw-corrected connector placement
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `20` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `34` total tests
- Important boundary:
	- no active-viewer consumer wiring changed in this slice
	- this is library-owned grouping or correlation groundwork, not viewer runtime PM4 signoff

## Mar 26, 2026 - wow-viewer Source-Of-Truth Reset

- The current default rule for `wow-viewer` changed: new implementation work should treat `WowViewer.Core.PM4`, `WowViewer.Core`, and `WowViewer.Core.IO` as the canonical owners, not `MdxViewer`.
- `MdxViewer` is now a historical, extraction, or consumer-compatibility input for `wow-viewer` work rather than the default runtime PM4 reference.
- Default validation for `wow-viewer` work is `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`, `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`, and the relevant inspect or converter command against the fixed development dataset.
- Build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln` only when a slice intentionally changes consumer compatibility or when the user explicitly asks for it.
- Older sections below that describe `MdxViewer` as the PM4 runtime reference are now historical context, not the live rule for new `wow-viewer` implementation work.

## Mar 26, 2026 - wow-viewer PM4 Fresh-Chat Handoff

- Treat the current `wow-viewer` PM4 state as library-first progress, not PM4 completion.
- What is real in `wow-viewer/src/core/WowViewer.Core.PM4` now:
	- research-seeded PM4 reader and inspect surface
	- working `pm4 inspect`, `pm4 audit`, `pm4 audit-directory`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, and `pm4 export-json`
	- shared placement-contract and placement-math slices for axis detection, planar-transform resolution, world-yaw correction, world-space centroid, pivot rotation, corrected world-position conversion, typed placement solutions, and typed coordinate-mode resolution
	- first reusable connector-key extraction seam for grouping or correlation work through typed placement solutions
	- first connector-based group-merge seam for PM4 grouping work through typed merge candidates and merge-map resolution
	- first narrow active-viewer consumer hookups for `ResolvePlanarTransform(...)`, `TryComputeWorldYawCorrectionRadians(...)`, and `ComputeSurfaceWorldCentroid(...)`
- Current verified validation floor:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `11` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` total tests
- Boundaries that still matter:
	- `MdxViewer` remains the runtime PM4 reference implementation
	- shared-library and compile validation are not viewer runtime signoff
	- final semantics are still open around `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR.Value1`, and full coordinate ownership
	- renderer-space conversion, broader object-group transforms, and final viewer object composition still remain outside `Core.PM4`
- Best next PM4 slice for a fresh chat:
	- hook the already-landed typed coordinate-mode resolver into the active `WorldScene` call site through the same narrow adapter pattern already used for planar-transform, yaw-correction, and centroid seams
	- keep that slice explicit about proof level: consumer hookup and regression preservation, not runtime PM4 closure

## Mar 25, 2026 - wow-viewer Tool Inventory And Cutover Plan

- Added a concrete inventory and cutover document at `plans/wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md`.
- New planning decisions captured there:
	- first-class survivors are the main viewer shell, one converter CLI, one inspect CLI, one optional catalog CLI, and a real PM4 library plus workspace from day one.
	- do not port duplicate legacy executables as permanent apps; merge WoWMapConverter with still-useful WoWRollback or AlphaLkToAlpha conversion seams, merge the Alpha WDT inspectors, and keep DBCTool.V2 behavior only.
	- PM4 correction: current `MdxViewer` behavior is the de facto PM4 runtime reference implementation, and `Pm4Research` should be ported as the future `Core.PM4` library family because PM4 semantics are still under active research.
	- keep parpToolbox, PM4Tool, ADTPrefabTool, and the legacy WoWRollback GUI or viewer surfaces in `parp-tools` as archaeology or reference unless a specific algorithm is deliberately re-homed.
	- immediate follow-up planning docs now exist for bootstrap layout, CLI or GUI surfaces, and the PM4 library direction:
		- `plans/wow_viewer_bootstrap_layout_plan_2026-03-25.md`
		- `plans/wow_viewer_cli_gui_surface_plan_2026-03-25.md`
		- `plans/wow_viewer_pm4_library_plan_2026-03-25.md`
	- migration emphasis is now effectively `1, 3, 2`: bootstrap layout and project skeleton, then dual-surface tool design, then deeper PM4 library consolidation work.
- This plan refines `plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md` rather than replacing it.
- Validation status:
	- planning and documentation only
	- no viewer, converter, or renderer code changed in this slice

## Mar 25, 2026 - wow-viewer Initial Skeleton Created In Workspace

- A first-pass `wow-viewer/` scaffold now exists directly under the workspace root.
- Created projects:
	- `src/viewer/WowViewer.App`
	- `src/core/WowViewer.Core`
	- `src/core/WowViewer.Core.IO`
	- `src/core/WowViewer.Core.Runtime`
	- `src/core/WowViewer.Core.PM4`
	- `src/tools-shared/WowViewer.Tools.Shared`
	- `tools/converter/WowViewer.Tool.Converter`
	- `tools/inspect/WowViewer.Tool.Inspect`
- Added first-pass repo files:
	- `WowViewer.slnx`
	- `Directory.Build.props`
	- `Directory.Packages.props`
	- `eng/Version.props`
	- `scripts/bootstrap.ps1`
	- `scripts/bootstrap.sh`
	- `scripts/validate-real-data.ps1`
- PM4-specific rule carried into the scaffold:
	- `Core.PM4` exists from day one
	- the placeholder code explicitly treats `MdxViewer` as the PM4 runtime reference and `Pm4Research` as the PM4 library seed
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- this is only a structure lock and placeholder-code build, not a real code-port or runtime signoff

## Mar 25, 2026 - First PM4 Code-Port Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first real PM4 code port from `src/Pm4Research.Core`.
- Landed pieces:
	- typed chunk models for the trusted PM4 chunk set
	- `Pm4ResearchDocument`
	- `Pm4ResearchReader`
	- `Pm4ResearchSnapshotBuilder`
- Important boundary:
	- this is still a raw research-facing PM4 reader layer
	- current `MdxViewer` behavior remains the runtime PM4 reference implementation for reconstruction, grouping, transforms, and viewer-facing semantics
	- no viewer PM4 logic has been re-homed onto `Core.PM4` yet
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after the PM4 port
	- no runtime validation or app integration has happened yet

## Mar 25, 2026 - PM4 Inspect Verbs Now Work In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first single-file PM4 analyzer and report layer on top of the earlier reader port.
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now has working PM4 commands:
	- `pm4 inspect --input <file.pm4>`
	- `pm4 export-json --input <file.pm4> [--output <report.json>]`
- Smoke-test result on the fixed reference tile:
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` succeeded
	- output included version `12304`, `54` chunks, `6318` `MSVT` vertices, `9990` `MSCN` points, and `2493` `MPRL` refs for `development_00_00.pm4`
- Important boundary:
	- this is still single-file research analysis, not viewer reconstruction or PM4 correctness closure
	- current `MdxViewer` behavior remains the runtime PM4 reference implementation

## Mar 25, 2026 - PM4 Audit And Placement Contracts Follow-Up

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first decode-audit path plus the first extracted MdxViewer-facing PM4 placement-contract seam.
- Landed pieces:
	- `Pm4ResearchAuditAnalyzer` with single-file and directory-level decode or corpus audit entry points
	- `WowViewer.Tool.Inspect` verbs for `pm4 audit --input <file.pm4>` and `pm4 audit-directory --input <directory>`
	- shared `Pm4AxisConvention`, `Pm4CoordinateMode`, `Pm4PlanarTransform`, `Pm4CoordinateService`, and `Pm4PlacementContract`
- New research note captured in the inspect layer:
	- CK24 low-16 object values, read as integers, appear to be plausible `UniqueID` candidates on the development map, but this remains a hypothesis until correlated against real placed-object data
- Important boundary:
	- this is still not the full MdxViewer PM4 reconstruction or transform solver port
	- current `MdxViewer` behavior remains the runtime reference implementation
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after this slice
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026 and scanned `616` PM4 files with no unknown chunks or diagnostics
	- early audit findings worth keeping visible:
		- `MDOS.buildingIndex->MDBH` shows real invalid references in the development corpus
		- `MSLK.RefIndex->MSUR` also shows corpus-level mismatches in nontrivial counts, which supports keeping linkage interpretation labeled as research

## Mar 25, 2026 - First wow-viewer PM4 Tests Landed

- `wow-viewer/tests/WowViewer.Core.PM4.Tests` now exists as the first real-data test project in the new repo.
- Current test coverage locks:
	- reader counts for `development_00_00.pm4`
	- current single-file analyzer summary and the `UniqueID` research note
	- current single-file decode-audit findings for `development_00_00.pm4`
	- current corpus-audit shape for `test_data/development/World/Maps/development`
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `6` passing tests
	- this is still fixed-dataset regression coverage only, not broad PM4 correctness closure

## Mar 25, 2026 - PM4 Linkage Slice And Placement-Math Helper Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first linkage-report family ported from `Pm4Research.Core`.
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now supports `pm4 linkage --input <directory> [--output <report.json>]`.
- Validated corpus result on the fixed development PM4 directory:
	- `616` files scanned
	- `150` files with ref-index mismatches
	- `58` files with bad `MDOS` refs
	- `4553` total ref-index mismatches
	- only `2` low16 object-id groups reused across multiple full CK24 values in this corpus slice
- Important interpretation boundary:
	- low16 CK24 object values may still align with expected `UniqueID` ranges, but the linkage report shows that range alignment alone is not enough to treat them as globally unique identifiers by themselves.
- First actual `WorldScene` helper port also landed in `Core.PM4`:
	- `Pm4PlacementMath.DetectAxisConventionByRanges`
	- `Pm4PlacementMath.IsLikelyTileLocal`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld`
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` tests

## Mar 25, 2026 - PM4 MSCN Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first MSCN relationship analyzer ported from `Pm4Research.Core`.
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now supports `pm4 mscn --input <directory> [--output <report.json>]`.
- Validated corpus result on the fixed development PM4 directory:
	- `616` files scanned
	- `309` files with MSCN
	- `1,342,410` total MSCN points
	- `MSUR.MdosIndex -> MSCN`: `511,891` fits and `6,201` misses
	- raw MSCN bounds overlap against mesh-backed CK24 groups: `1,162` fits and `724` misses
	- swapped-XY MSCN bounds overlap against mesh-backed CK24 groups: only `10` fits and `1,876` misses
- Important interpretation boundary:
	- this slice weakens the simple XY-swapped MSCN companion-space hypothesis for the fixed development corpus
	- it still does not make MSCN authoritative for final viewer reconstruction by itself
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` tests

## Mar 26, 2026 - PM4 Unknowns Slice And Normal-Based Axis Scoring Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first unknowns-report family ported from `Pm4Research.Core` plus the next extracted `WorldScene` solver seam for normal-based axis scoring.
- Landed pieces:
	- `Pm4ResearchUnknownsAnalyzer`
	- unknowns report records for relationship summaries, link-id patterns, MSPI interpretation, field distributions, and explicit open-question findings
	- `WowViewer.Tool.Inspect` verb `pm4 unknowns --input <directory> [--output <report.json>]`
	- `Pm4PlacementMath.DetectAxisConventionByTriangleNormals`
	- `Pm4PlacementMath.DetectAxisConventionBySurfaceNormals`
	- normal-based axis scoring helpers on triangles and surfaces
- Validated corpus result on the fixed development PM4 directory:
	- `616` files scanned
	- `309` non-empty geometry or link files
	- `1,273,335` `MSLK.LinkId` values, all currently fitting the sentinel-tile pattern in this corpus
	- `598,882` active `MSLK` path windows with `399,183` indices-only fits and `199,699` dual-fit windows
	- `MSLK.RefIndex -> MSUR` still partial with `1,268,782` fits and `4,553` misses
	- `MPRR.Value1` remains mixed-domain with partial fits against both `MPRL` and `MSVT`
- Important interpretation boundary:
	- this strengthens the decode-evidence base, but it still does not close the final semantics of `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR`, or PM4 coordinate ownership.
	- normal-based axis scoring is now reusable in `Core.PM4`, but the full viewer reconstruction and transform solver still live in current `WorldScene`.
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 26, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `8` tests

## Mar 26, 2026 - PM4 Planar-Transform Resolver Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next extracted `WorldScene` PM4 solver seam: planar-transform resolution against MPRL anchors.
- Landed pieces:
	- `Pm4PlacementMath.ResolvePlanarTransform`
	- MPRL centroid-distance scoring against planar candidates
	- MPRL footprint scoring for multi-anchor groups
	- MPRL heading/yaw comparison with quarter-turn fallback
	- reusable helpers for MPRL planar-point conversion and principal-yaw estimation
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `9` tests
	- current measured development-tile result for the whole-tile test slice: tile-local `XYPlaneZUp` resolves to planar transform `(swap=false, invertU=false, invertV=false)`
	- synthetic world-space regression case now also locks a quarter-turn candidate selection `(swap=true, invertU=true, invertV=false)`
- Important boundary:
	- this still does not port the full PM4 object-level placement pipeline or viewer yaw-correction layer.
	- active `WorldScene` remains the runtime reference implementation for full PM4 reconstruction behavior.

## Mar 26, 2026 - PM4 World-Yaw Correction Slice And First Viewer Consumer Wiring

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next extracted `WorldScene` solver seam: world-yaw correction against MPRL heading evidence.
- Landed pieces:
	- `Pm4PlacementMath.TryComputeWorldYawCorrectionRadians`
	- signed basis fallback against expected MPRL yaw
	- synthetic regression coverage for a meaningful non-zero yaw correction case
- Active viewer integration follow-up also landed:
	- `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.PM4`
	- `WorldScene.ResolvePlanarTransform(...)` now delegates to shared `Core.PM4` through a narrow adapter path
	- `WorldScene.TryComputeWorldYawCorrectionRadians(...)` now delegates to shared `Core.PM4` through the same adapter path
- Important boundary:
	- this is still a narrow consumer slice; `WorldScene` continues to own the broader PM4 placement/render path.
	- no runtime signoff has happened yet on viewer-visible PM4 behavior after the shared-library hookup.

## Mar 26, 2026 - PM4 World-Space Centroid Slice And Second Viewer Consumer Hookup

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next extracted `WorldScene` solver seam above world-yaw correction: world-space surface centroid computation.
- Landed pieces:
	- `Pm4PlacementMath.ComputeSurfaceWorldCentroid(...)`
	- synthetic tile-local regression coverage for the shared centroid helper
	- `WorldScene.ComputeSurfaceWorldCentroid(...)` now delegates to shared `Core.PM4` through the existing adapter path
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `11` tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `4` placement-focused tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-centroid-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this moves the shared world-space pivot helper only; renderer-space centroid handling and the broader PM4 object placement path still remain in `WorldScene`
	- no real-data runtime signoff yet on viewer-visible PM4 behavior after this additional shared-solver hookup

## Mar 26, 2026 - First Non-PM4 Shared Map Reader Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core` now contains the first shared non-PM4 map-format constants and summary contracts:
	- `MapChunkIds`
	- `MapFileKind`
	- `MapChunkLocation`
	- `MapFileSummary`
- `wow-viewer/src/core/WowViewer.Core.IO` now contains the first reusable WDT or ADT top-level reader layer:
	- `ChunkedFileReader`
	- `MapFileSummaryReader`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now has the first non-PM4 shared-I/O consumer verb:
	- `map inspect --input <file.wdt|file.adt>`
- Fixed-dataset scope for this slice:
	- `development.wdt`
	- `development_0_0.adt`
- Important boundary:
	- this is only top-level chunk order, version, and file-kind summarization for WDT or ADT-family files
	- it is not yet a full ADT or WDT semantic parser, writer, or runtime cutover

## Mar 26, 2026 - First Shared Cross-Family File Detector Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core` now contains the first cross-family file-detection contracts:
	- `WowFileKind`
	- `WowFileDetection`
- `wow-viewer/src/core/WowViewer.Core.IO` now contains the first shared cross-family detector:
	- `WowFileDetector`
- `MapFileSummaryReader` now routes WDT or ADT-family classification through that shared detector instead of owning its own kind heuristics.
- `wow-viewer/tools/converter/WowViewer.Tool.Converter` now has the first non-placeholder non-PM4 command:
	- `detect --input <file>`
- Fixed-dataset smoke coverage for this slice:
	- `development.wdt` -> `Wdt`
	- `development_00_00.pm4` -> `Pm4`
	- `development_0_0_tex0.adt` -> `AdtTex`
	- `development_0_0_obj0.adt` -> `AdtObj`
- Important boundary:
	- this is classification and version detection only
	- it is not yet a shared read or write implementation for WMO, M2, BLP, DBC, or DB2 payload semantics

## Mar 26, 2026 - PM4 World-Space Yaw Helper Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next library-only PM4 math slice adjacent to the earlier yaw solver and centroid helper.
- Landed pieces:
	- shared `Pm4PlacementMath.RotateWorldAroundPivot(...)`
	- shared `Pm4PlacementMath.ConvertPm4VertexToWorld(...)` overload that can apply yaw correction around a world pivot without any renderer-space dependency
	- synthetic regression coverage for pivot rotation and corrected world-position conversion
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `6` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `13` tests
- Important boundary:
	- this is a `wow-viewer` library slice only; no new `MdxViewer` consumer hookup was added in this step
	- renderer-space conversion and object-transform composition still remain outside `Core.PM4`

## Mar 26, 2026 - PM4 Placement-Solution Contract Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first typed placement-result contract that bundles the current library-owned PM4 placement decision into one object.
- Landed pieces:
	- `Pm4PlacementSolution`
	- `Pm4PlacementMath.ResolvePlacementSolution(...)`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld(Vector3, Pm4PlacementSolution)`
	- synthetic end-to-end regression coverage for world-space transform, pivot, and yaw-correction resolution
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `8` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `15` tests
- Important boundary:
	- this is still a `wow-viewer` library slice only; no new active-viewer consumer wiring was added here
	- object-group transforms, renderer-space conversion, and final viewer object composition still remain outside `Core.PM4`

## Mar 26, 2026 - wow-viewer Copilot Workflow Surface Updated

- The shared Copilot workflow surface now explicitly treats `wow-viewer` as a primary active path alongside `gillijimproject_refactor`.
- New shared continuation assets now live under `.github/`:
	- `.github/skills/wow-viewer-pm4-library/SKILL.md`
	- `.github/skills/wow-viewer-migration-continuation/SKILL.md`
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` now routes implementation-sized PM4 library asks to the dedicated PM4 library prompt instead of only the broader repo-planning prompts.
- Future-session workflow rule:
	- use the PM4 library prompt or skill when the ask is the next `Core.PM4` slice, inspect verb, regression update, or narrow shared-solver extraction
	- use the broader tool-suite prompt set only when the ask is repo-shape, tool inventory, CLI or GUI parity, or migration sequencing
- Important boundary:
	- this workflow update does not change runtime PM4 validation status
	- `wow-viewer` test or build passes are still library validation, not active-viewer runtime signoff

## Mar 26, 2026 - wow-viewer Shared I/O Copilot Workflow Surface Updated

- The shared Copilot workflow surface now has an explicit non-PM4 implementation path in addition to the earlier PM4-only route.
- New shared continuation assets now live under `.github/`:
	- `.github/skills/wow-viewer-shared-io-library/SKILL.md`
	- `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`
- `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md` now records the current shared `Core` or `Core.IO` source-of-truth, landed slices, validation surface, and immediate next seams.
- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` now routes implementation-sized non-PM4 shared-format work to the dedicated shared-I/O implementation prompt instead of only the broader shared-I/O planning prompt.
- `.github/copilot-instructions.md` now explicitly covers `wow-viewer` shared I/O guardrails and first reads, so new chats can distinguish PM4, shared-I/O implementation, and broader migration planning earlier.
- Future-session workflow rule:
	- use the PM4 library prompt or skill when the ask is the next `Core.PM4` slice
	- use the shared-I/O implementation prompt or skill when the ask is the next `Core` or `Core.IO` non-PM4 format slice
	- use the broader tool-suite prompt set only when the ask is repo-shape, ownership planning, or migration sequencing
	- whenever a new `wow-viewer` skill or implementation prompt is created, update `.github/copilot-instructions.md` and `wow-viewer/README.md` in the same slice so discovery stays automatic in future chats

## Mar 26, 2026 - PM4 Coordinate-Mode Resolver Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next library-only PM4 solver seam adjacent to the earlier placement-solution work: typed coordinate-mode resolution.
- Landed pieces:
	- `Pm4CoordinateModeResolution`
	- `Pm4PlacementMath.ResolveCoordinateMode(...)`
	- internal coordinate-mode score evaluation that reuses the shared planar-transform resolver, footprint score, and centroid score helpers instead of leaving the tile-local versus world-space decision loop only in `WorldScene`
	- regression coverage for the fixed development tile, a synthetic world-space case, and the missing-evidence fallback path
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `11` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `18` tests
- Important boundary:
	- this is still a `wow-viewer` library slice only; no new active-viewer consumer wiring was added here
	- the active viewer still owns the current coordinate-mode call site until a later narrow consumer slice explicitly re-homes it

## Mar 26, 2026 - wow-viewer Bootstrap And Non-PM4 Core Follow-Up

- The current concern about `wow-viewer` drifting into PM4-only work was valid.
- Verified repo state before correction:
	- `WowViewer.Core`, `WowViewer.Core.IO`, and `WowViewer.Core.Runtime` were still mostly placeholders
	- `libs/` was empty
	- `scripts/bootstrap.ps1` and `scripts/bootstrap.sh` were literal placeholders
- Corrective slice now landed:
	- `wow-viewer/scripts/bootstrap.ps1` and `wow-viewer/scripts/bootstrap.sh` now clone the baseline upstream repos called out in the migration draft
	- `wow-viewer/src/core/WowViewer.Core` now contains first non-PM4 chunk primitives: `FourCC` and `ChunkHeader`
	- `wow-viewer/src/core/WowViewer.Core.IO` now contains `ChunkHeaderReader`
	- `wow-viewer/tests/WowViewer.Core.Tests` now locks the FourCC and chunk-header boundary behavior
- Important boundary:
	- this only starts the non-PM4 shared-core path; it does not mean the broader map, object, terrain, WMO, model, texture, or runtime library families are migrated yet
	- the next corrective slices should target shared I/O ownership, not just more PM4 seams

## Mar 25, 2026 - Post-v0.4.5 Branch And Roadmap Prompt Bundle

- Post-release planning is now intentionally split onto branch `feature/v0.4.6-v0.5.0-roadmap` so the next milestone work can stay isolated from `main` until the first real slices are ready.
- Detailed Copilot prompt assets for the `wow-viewer` tool-suite/library refactor now live under workspace `.github/prompts/`, not under `gillijimproject_refactor/plans`.
- For this tool-suite migration work, treat `gillijimproject_refactor/plans` as scratchpad/archeology notes and `.github/prompts/` as the canonical prompt surface.
- Current dedicated prompt set:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-bootstrap-layout-plan.prompt.md`
	- `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-inventory-cutover-plan.prompt.md`
	- `.github/prompts/wow-viewer-cli-gui-surface-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-migration-sequence-plan.prompt.md`
- New prompt bundle captured under `plans/` for the next branch of work:
	- `post_v0_4_5_plan_set_2026-03-25.md`
	- `v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
	- `wowrollback_uniqueid_timeline_prompt_2026-03-25.md`
	- `alpha_core_sql_scene_liveness_prompt_2026-03-25.md`
	- `viewer_performance_recovery_prompt_2026-03-25.md`
	- `v0_5_0_new_repo_library_migration_prompt_2026-03-25.md`
	- `v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`
- Current intended milestone split:
	- `v0.4.6` should carry the first visible WoWRollback / `UniqueID` timeline filter slice inside the active viewer, plus Alpha-Core SQL caching/fidelity follow-up and an initial performance recovery pass.
	- `v0.5.0` should move into `https://github.com/akspa0/wow-viewer` as the new production repo with one canonical shared library plus split viewer/tool consumers.
- Important boundaries for future sessions:
	- keep WoWRollback integration on the active viewer UI/data-loading path; do not drift back to the older separate web-viewer plan as the primary delivery target.
	- treat `parp-tools` as the R&D / archaeology repo and `wow-viewer` as the intended production home for the next major milestone.
	- external constructive guidance now explicitly supports a sane top-level `wow-viewer` layout: the main renderer app should have one obvious root, with libraries/dependencies/tools split into their own clear folders instead of repeating the current nested sprawl.
	- latest user constraint: fully refactor and re-own the first-party read/parse/write/convert stack, including current base libraries such as `gillijimproject-csharp`; keep upstream projects like `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib` under `libs/` and track their original repos where practical.
	- repo bootstrap should automatically pull support repos like `wow-listfile` instead of relying on manual setup.
	- possible targeted integrations worth evaluating later include `MapUpconverter`, `ADTMeta`, `wow.export`, and `wow.tools.local`, but they should support the owned-library plan rather than replace it.
	- possible future upstream work on `Noggit` / `noggit-red` alpha-era support is interesting, but should stay an explicit stretch/outreach track rather than replacing the main `wow-viewer` migration target.
	- a concrete first-pass repo tree and migration order draft now exists in `plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`; future planning should refine that draft rather than re-deriving repo shape from scratch.
	- treat Alpha-Core SQL equipment correctness, animation-state handling, and pathing as separate seams.
	- do not assume SQL or PM4 already prove server-like NPC pathing; that remains a later research seam, not an implicit short-term deliverable.
	- performance recovery is now a first-class dependency, but the deeper overhaul should be planned against the new repo/library split instead of indefinite surgery inside the R&D tree.
- Documentation follow-up on the same slice:
	- root `README.md` was refreshed again to make the active support headline, conversion coverage, WMO `v14/v16/v17` handling, and built-in tooling more explicit.
	- screenshot reality remains unchanged: asset-catalog screenshot automation exists already, but a curated world/UI gallery is still future work.
- Validation status:
	- planning/documentation only
	- no viewer, converter, or renderer code changed in this slice

## Mar 24, 2026 - WMO Vertex-Light Prototype In Active Viewer

- First renderer-side object-lighting prototype is now in the active tree at `src/MdxViewer/Rendering/WmoRenderer.cs`.
- Scope of the implementation:
	- WMO group vertex buffers now carry a fourth attribute for baked vertex-light color.
	- `WmoRenderer` now prefers parsed `MOCV` vertex colors when they look usable.
	- if usable `MOCV` is missing but preserved v14 lightmap payloads exist (`MOLV` / `MOLD` / `MOLM`), the renderer now samples those on load into per-vertex baked-light modulation colors.
	- the fragment shader now modulates the existing diffuse/fog path by that baked-light color, so WMOs can show preserved object-light contribution instead of relying only on the generic ambient+directional path.
- Important limit:
	- this is not full `0.5.3` / early-client object-lightmap parity.
	- there is still no client-faithful group/batch lightmap texture pipeline, no recovered batch-to-lightmap index path, and no dedicated `RenderGroupLightmap` / `RenderGroupLightmapTex` analogue in the active renderer.
	- this is a first prototype using the data the active model already preserves.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after the change.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on affected WMOs.

## Mar 24, 2026 - 0.5.3 Terrain/Object Render Fast-Path And Viewer Perf Gap

- Reverse-engineering follow-up against the symbolized `0.5.3` client materially tightened the current performance/parity story; no viewer code changed in this slice.
- durable write-up extended in `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- high-confidence `0.5.3` terrain findings from decompilation:
	- `CreateRenderLists` (`0x00698230`) is a real precompute step that builds terrain texcoord tables and batch/render-list data instead of leaving chunk draw setup entirely to the frame loop
	- `RenderLayers` (`0x006a5d00`) and `RenderLayersDyn` (`0x006a64b0`) use locked GX buffers plus prebuilt chunk batches, not a fully generic per-layer rebuild path
	- terrain already has shader-assisted paths in `0.5.3`: the chunk draw path binds `CMap::psTerrain` / `CMap::psSpecTerrain` plus `shaderGxTexture` when terrain/specular shader support is enabled
	- terrain layer count is reduced by distance (`textureLodDist` can clamp the runtime draw to one layer), and the dynamic path also fades diffuse alpha before collapse
	- per-layer moving-texture behavior is confirmed in the terrain path itself: when runtime layer flag `0x40` is set, `RenderLayers` / `RenderLayersDyn` apply an extra texture transform indexed by low flag bits into the time-varying world transform tables updated by `FUN_006804b0`
	- terrain shadows are drawn as a separate modulation pass rather than being flattened into one generic terrain blend loop
- high-confidence `0.5.3` object/light findings from decompilation:
	- `RenderMapObjDefGroups` (`0x0066e030`) walks visible `CMapObjDefGroup` lists, sets transforms once per group, and dispatches `CMapObj::RenderGroup(...)`; this is more structured than the active viewer's generic instance loops
	- `CreateLightmaps` (`0x006adba0`) allocates per-group lightmap textures (`256x256`) and registers `UpdateLightmapTex`, which strongly supports a dedicated object-lightmap path in the client
	- `RenderGroupLightmap(...)` uses dedicated group lightmap vertex streams and batch-local lightmap texture binding rather than one generic object UV/material path
	- `RenderGroupLightmapTex(...)` splits the lightmap composition work into dedicated subpasses with lighting forced off, and `UpdateLightmapTex(...)` exposes row-stride plus CPU memory on `GxTex_Latch`; taken together, the object lightmap path is a real rendering subsystem, not just a texture on the generic WMO path
	- `CalcLightColors` (`0x006c4da0`) computes a much richer lighting state than the active viewer currently models: direct, ambient, six sky channels, five cloud channels, four water channels, fog end, fog-start scalar, and storm blending
- viewer-side implication from the same slice:
	- the active viewer remains structurally flatter than the client in the exact places that matter for both performance and fidelity:
		- `StandardTerrainAdapter` still actively uses `MPHD` only for big-alpha/profile selection and still flattens `MAIN` entries to boolean tile existence
		- `TerrainRenderer` is still a generic base+overlay pass loop that only interprets `MCLY 0x100`; it has no terrain shader-family split, no per-layer motion support, no layer-count LOD collapse, and no specular terrain path
		- `LightService` remains a simplified nearest-zone DBC interpolator rather than a full terrain/object/sky/runtime-light system
		- `WmoRenderer` / `MdxRenderer` still rely on shared generic shader families instead of the client's stronger specialization
		- `WorldScene` hot paths remain heavy: MDX transparent items are re-collected/sorted every frame, optional PM4 forensic budgets are still `int.MaxValue`, and the current render-queue abstraction is not yet the active world submission path
- practical priority order now supported by evidence:
	1. preserve `MAIN` / `MPHD` / `MCLY` semantics as first-class runtime metadata
	2. split terrain renderer responsibilities into fallback vs client-faithful material/shader path
	3. treat object/lightmap parity as a separate seam from terrain lighting
	4. reduce generic hot-path state churn before layering on more fidelity features
	5. use the existing `WorldAssetManager` read/path-probe counters as the basis for an explicit scene residency/prefetch policy
- validation status:
	- reverse engineering plus code audit only; no viewer build or runtime signoff was produced by this slice

## Mar 24, 2026 - WoW 2.0.0 Beta Ghidra Recon For M2 / Light / Particle Risk

- Static reverse-engineering pass only against a loaded beta `2.0.0` `WoW.exe` in Ghidra. No viewer/converter code changed in this slice.
- durable write-up: `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- High-confidence findings from decompilation:
	- `Model2` has an explicit BLS shader bootstrap in `FUN_00717b00` (`M2Cache.cpp` path string present) and loads both `shaders\vertex\Model2.bls` and `shaders\pixel\Model2.bls`.
	- map objects preload a dedicated bank of pixel BLS programs in `FUN_006b3b20`, including `MapObjOverbright`, `MapObjSpecular`, `MapObjMetal`, `MapObjEnv`, `MapObjEnvMetal`, `MapObjExtWater0`, `MapObjTransDiffuse`, and `MapObjTransSpecular`.
	- `M2Light.cpp`-anchored logic in `FUN_0072d1a0` does not treat model lights as a flat passive list: lights are inserted either into a spatial bucket structure or a general linked list depending on runtime mode/type, and companion mutators (`FUN_0072cc60`, `FUN_0072cc90`, `FUN_0072cdc0`) relink them when state/position changes.
	- particle runtime is a real engine-side system, not just file payload playback: `FUN_007c26c0` bootstraps `CParticleEmitter2_idx` and global pools, while `FUN_007ca9d0` / related constructors copy emitter payload regions into runtime `CParticle2` / `CParticle2_Model` objects.
	- the `Light*.dbc` family is loaded through strict `WDBC` schema-checked table loaders with ID-index maps, not ad-hoc parsing. Confirmed table shapes:
		- `LightFloatBand.dbc` and `LightIntBand.dbc`: `0x22` columns, `0x88` row size, two `0x40`-byte band payloads plus two leading scalars.
		- `LightParams.dbc`: `9` columns, `0x24` row size.
		- `Light.dbc`: `0xc` columns, `0x30` row size with a trailing `0x14`-byte block.
		- `LightSkybox.dbc`: `2` columns, `8` byte rows with string-table resolution.
- Practical viewer risk guidance from this RE pass:
	- do not collapse early/later `2.x` materials into one generic shader path if the goal is parity; the client uses distinct BLS programs for `Model2` and multiple map-object material families.
	- do not expect smoke / particle projection issues to close from parser tweaks alone; the particle and light systems are runtime-managed and likely need render-path/state investigation in addition to format parsing.
	- terrain follow-up is now split into two separate engine tracks:
		- cached per-layer terrain programs are now pinned down more precisely:
			- `terrain1..4` at `DAT_00caf304..310` are the one-pass layer-count table used when `DAT_00cb3594 == 0` and `DAT_00ca31b8 != 0`
			- `terrain1_s..4_s` at `DAT_00caf548..554` are the alternate one-pass layer-count table used when `DAT_00cb3594 != 0`
			- `terrainp` / `terrainp_s` belong to the slower manual terrain fallback path in `FUN_006cee30`, not the cached layer-count table
			- `terrainp_u` / `terrainp_us` are loaded at startup but are still untraced in an active draw branch
			- terrain also has a separate time-varying layer-transform path: `FUN_006c00f0` copies a source layer flag field into each runtime layer object, `FUN_006cee30` / `FUN_006cf590` apply an extra transform when bit `0x40` is present, and `FUN_006804b0` updates the transform tables every world tick
		- `XTextures\slime\slime.%d.blp` resolves into an animated `WCHUNKLIQUID` surface path, not yet proven to be a terrain diffuse-layer effect
		- latest `WCHUNKLIQUID` pass shows a real mode dispatcher: `FUN_006c65b0` splits modes `0/4/8` into animated texture-family rendering and modes `2/3/6/7` into a direct-coordinate/UV-style path
		- `FUN_006c65b0` passes the raw mode nibble into `FUN_0069b310`, so the liquid mode is also the animated family index
		- currently recovered family table entries:
			- `0 -> lake_a`
			- `1 -> ocean_h`
			- `2 -> lava`
			- `3 -> slime`
			- `4 -> lake_a` again
		- novelty/dead-content candidates:
			- `FUN_0069e690(2)` currently reaches `FUN_0069b310(6)`, but the family slot is still unresolved via data xrefs
			- `XTextures\river\fast_a.%d.blp` exists in strings but is not in the traced active family table
	- viewer-side audit against the active tree shows terrain flag under-parsing is real:
		- `StandardTerrainAdapter` currently uses `MPHD` only for big-alpha selection
		- `ReadMainChunk(...)` treats any non-zero `MAIN` entry as generic tile presence instead of keeping entry semantics like `has ADT` vs `all water`
		- raw `MCLY` flags are preserved into `TerrainLayer.Flags`, but `TerrainRenderer` only interprets `0x100` as the implicit-alpha hint
	- the dangerous seam for `2.x` support is downstream interpretation of light/material/particle IDs and runtime state, not raw DBC ingestion.
- Validation status:
	- reverse engineering only; no automated tests, no solution build, and no runtime real-data signoff were performed in this slice.

## Mar 24, 2026 - 0.12 Standalone Model Browser Recovery

- The latest standalone-model regression for the `0.12` client split into two separate seams in the active viewer:
	- `MpqDataSource` was no longer indexing Alpha-style nested model wrappers at all (`.mdx.MPQ`, `.mdl.MPQ`, `.m2.MPQ`), and it also skipped loose `.mdl` files entirely.
	- standalone `MD20` / `MD21` routing in `ViewerApp.LoadM2FromBytes(...)` still allowed an unsupported build with no resolved `M2Profile` to continue into the M2-family adapter path instead of failing cleanly.
- Root cause now fixed in the active tree:
	- `src/MdxViewer/DataSources/MpqDataSource.cs`
		- loose-file indexing now includes `.mdl`
		- Alpha nested wrapper scan now includes model wrappers (`.mdx.MPQ`, `.mdl.MPQ`, `.m2.MPQ`)
		- model wrappers now register extension aliases into the file set / Alpha wrapper cache so the browser and path resolver can find the same wrapped asset through `.mdx`, `.mdl`, or `.m2`
	- `src/MdxViewer/ViewerApp.cs`
		- the standalone browser's `.mdx` filter now aggregates early model files from both `.mdx` and `.mdl`
		- disk loads now accept `.mdl` through the same container-probe path already used by the data-source loader
		- `LoadM2FromBytes(...)` now hard-fails with a clear unsupported-build error when no `M2Profile` resolves for the active client build instead of continuing into an unsafe best-effort adapter path
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
		- the file-browser type label now reflects that the early-model bucket is `.mdx/.mdl`
- Scope boundary:
	- this fix restores file discovery/indexing and turns the unsupported `.m2` route into a safe load failure for pre-M2 builds; it is not proof that standalone `0.12` runtime model rendering is fully signed off across a real client dataset.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after this fix.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on actual `0.12` client browsing/loading because no fixed `0.12` data path is currently recorded in `memory-bank/data-paths.md`.

## Mar 24, 2026 - 0.6.0 Through 2.x Terrain Alpha Grid Regression Fix

- The terrain grid-pattern regression affecting standard ADT clients from `0.6.0` through the `2.x` era was not a newly proven shader/blend-style difference. The active viewer was still decoding that whole legacy band through a naive sequential 4-bit MCAL unpack path in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`.
- Root cause now fixed in the active tree:
	- `StandardTerrainAdapter.ExtractAlphaMaps(...)` for `TerrainAlphaDecodeMode.LegacySequential` now prefers the relaxed MCAL path (`Mcal.GetAlphaMapForLayerRelaxed(...)`) and preserves `DoNotFixAlphaMap` behavior.
	- the old naive legacy fallback now routes through the existing row-aware 4-bit decode + legacy edge-fix helpers instead of writing raw nibble pairs straight into the `64x64` output.
- Scope boundary:
	- this change is limited to the standard-terrain legacy band (`0.6.0` through `2.x`) and does not change the separate `AlphaTerrainAdapter` path for `0.5.x` or the strict `3.x` / Cataclysm `4.0.0` decode branches.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after this fix and after correcting unrelated compile breaks in the in-progress minimap candidate-path patch.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on affected `0.6.0` / `0.7.0` / `0.8.0` / `0.9.0` / `1.x` / `2.x` terrain tiles.

## Mar 24, 2026 - v0.4.5 Branding + MH2O LiquidType Classification Fix

- Active viewer branding/release metadata is now aligned toward `parp-tools WoW Viewer` version `0.4.5` without renaming the `MdxViewer` root namespace.
- Current user-facing changes in the active tree:
	- viewer window title now uses `parp-tools WoW Viewer`
	- Help -> About now opens a modal with author + credits instead of only writing a transient status line
	- project metadata now emits `ParpToolsWoWViewer` as the executable/assembly name
	- `.github/workflows/release-mdxviewer.yml` now packages/releases `parp-tools-wow-viewer-<version>-win-x64.zip` and uses the .NET 10 SDK required by the active project target
- MH2O follow-up on the same slice:
	- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` now classifies `MH2O` liquids from `LiquidType.dbc -> Type` when DBC metadata is available for the active client build
	- when DBC loading is unavailable or an ID is missing from the loaded table, the viewer now falls back to an expanded static family map that includes the real 3.3.5 / 4.0 IDs already used elsewhere in the repo (`13`, `14`, `17`, `19`, `20`)
	- `src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/LiquidConverter.cs` now recognizes those late-style IDs in the shared `LiquidTypeId -> MCLQ family` fallback path as well
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed on Mar 24, 2026
	- no automated tests were added or run
	- no runtime real-data signoff yet on 3.3.5 / 4.0 liquid visual parity; the build only proves the implementation compiles

## Mar 25, 2026 - Fullscreen Minimap Release Blocker Closed For v0.4.5

- The fullscreen/docked minimap repair is now treated as closed for `v0.4.5` after the final transpose-only follow-up and runtime user confirmation on the fixed development minimap dataset.
- Final landed behavior in the active tree:
	- the bad `WoWConstants.TileSize` minimap hypothesis stays reverted; the active `64x64` minimap grid continues to use `WoWConstants.ChunkSize`
	- the broad world-axis swap attempted during the first Designer Island follow-up was backed out
	- the landed fix instead keeps the direct world/click mapping and only transposes the screen-space marker placement seam that had drifted away from the drawn tile grid
	- docked and fullscreen minimap now agree well enough for the user to describe the bug as fixed after runtime checking the top-right Designer Island scenario
- Practical release consequence:
	- the fullscreen minimap is no longer an open `v0.4.5` blocker
	- remaining minimap work should be treated as future polish or new regressions, not as justification to keep `v0.4.5` open
- Validation status:
	- build plus targeted runtime user signoff: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-transpose-repair/"` passed on Mar 25, 2026 after the final transpose-only repair
	- runtime user feedback then confirmed the repaired minimap behavior on the fixed development minimap dataset
	- no automated tests were added or run
	- this is not broad automated minimap coverage; it is targeted real-data runtime confirmation for the previously broken release-blocker scenario

## Current Focus: v0.4.0 Recovery Branch (Mar 17, 2026)

Working branch is now reset in the main tree, not only in side worktrees.

- Branch: recovery/v0.4.0-surgical-main-tree
- Baseline tag/commit: v0.4.0 / 343dadf
- .github metadata restored from main and committed: 845748b
- .github restore was pushed to origin/recovery/v0.4.0-surgical-main-tree

### Tooling Path Reuse + Unified Format I/O Proposal (Mar 23)

- Viewer tool dialogs should stop forcing repeated folder browsing when the session already knows the active base client and loose overlay roots.
- Current viewer-side behavior now seeds tool inputs from the active session where practical:
	- `Generate VLM Dataset` pulls the active MPQ base client path and current map name.
	- `Terrain Texture Transfer` prefers the attached loose-overlay map directory as source and the base-client map directory as target when those roots exist.
	- `Map Converter` now seeds WDT/map-directory inputs from the currently loaded local WDT when available, otherwise from the current map under the active loose/base roots.
	- `WMO Converter` still seeds from the currently loaded standalone WMO when applicable.
- Important scope limit:
	- this is UI/tool input seeding only, not proof that all downstream conversion paths are correct for Alpha, LK 3.3.5, or 4.x data.
	- after the Mar 23 seeding follow-up, edited-file diagnostics were clean on `src/MdxViewer/ViewerApp.cs`, but no new full viewer build or runtime signoff was recorded yet for this slice.
- Larger project direction requested by the user:
	- consolidate terrain, ADT/WDT, M2/MDX, and WMO read/write knowledge into one shared library used by viewer, converter, and tooling instead of continuing to split capabilities across `MdxViewer` and `WoWMapConverter.Core`.
	- do not assume the existing map converter is already closed for Alpha placement writing: MODF/MDDF downconversion for Alpha WDT remains an explicit open seam until reimplemented and validated.
	- planning prompt captured in `plans/unified_format_io_overhaul_prompt_2026-03-23.md`.
	- new PM4 planning guardrail from Mar 24 viewer forensics/UI work:
		- the practical viewer hierarchy is `CK24 -> MSLK-linked subgroup -> optional MDOS subgroup -> connectivity part`
		- PM4 centroids are useful derived display anchors for those nodes, not proven raw PM4 node records
		- `MSUR.AttributeMask` colors should be surfaced as explicit value legends, but their semantics remain open and must not be hardcoded into format contracts prematurely

### Documentation Refresh + Render Quality Follow-Up (Mar 23)

- Repo-level docs were refreshed, but the first pass still contained bad assumptions.
- The user then rewrote `src/MdxViewer/README.md` to be more grounded and truthful.
- Current documentation/handoff rule:
	- treat the user-corrected viewer README as the authoritative public summary for support and usage claims
	- do not reintroduce speculative platform restrictions or inflated support statements without direct evidence
	- do not write branch-local language into README text intended for eventual `main`
- Important current README claims to preserve in future sessions:
	- support headline: `0.5.3` through `4.0.0.11927`
	- later `4.0.x` ADT support exists
	- later split-ADT support through `4.3.4` exists but remains explicitly untested
	- Alpha-Core SQL world NPC/gameobject support is relevant to the README and should not be dropped casually
	- asset-catalog screenshot automation exists already; broader UI/menu showcase capture is still future work
- Validation status:
	- docs were updated after the Mar 23 viewer build had already passed
	- the documentation update itself adds no runtime validation and should not be read as new visual signoff

### Viewer Debug/Workflow Follow-Up (Mar 22)

- Latest viewer-side work moved away from treating PM4 runtime streaming as the only inspection path.
- Current additions in the active tree:
	- PM4 offline OBJ export from `src/MdxViewer/Terrain/WorldScene.cs`, surfaced through `ViewerApp_Pm4Utilities.cs`, so per-tile/per-object PM4 geometry can be compared outside the live overlay window.
	- minimap interaction/caching follow-up in `ViewerApp_MinimapAndStatus.cs`, `ViewerApp.cs`, and `Rendering/MinimapRenderer.cs`:
		- teleport now requires triple-clicking the same tile instead of a single short click
		- minimap zoom/pan/window state now persist in viewer settings
		- decoded minimap tiles now cache on disk under `output/cache/minimap/<cache-segment>`
	- terrain-hole debug override in `TerrainMeshBuilder`, `TerrainManager`, `VlmTerrainManager`, and `ViewerApp_Sidebars.cs`:
		- viewer can ignore terrain hole masks globally or on the current camera tile by rebuilding loaded chunk meshes only
		- source ADT hole flags are unchanged; this is viewer-side inspection only
- Validation status:
	- file diagnostics were clean on the edited viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 22, 2026 after these viewer-side follow-ups were in the active tree
	- no automated tests were added or run
	- no runtime real-data signoff yet on PM4 OBJ correctness, minimap feel/cache benefit, or terrain-hole rebuild behavior while streaming

### Standalone PM4 Research Library (Mar 21)

- Added a new isolated project at `src/Pm4Research.Core` for fresh PM4 format work outside the current viewer/converter reconstruction path.
- Current scope of that library:
	- raw chunk walking with preserved signatures, offsets, sizes, and payload bytes
	- standalone typed decoding for `MVER`, `MSHD`, `MSLK`, `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR`, `MSCN`, `MPRL`, `MPRR`, `MDBH`, `MDBI`, `MDBF`, `MDOS`, and `MDSF`
	- lightweight exploration snapshot generation for counts and chunk bounds
	- raw decode-audit reporting for per-file and corpus-wide chunk consistency and cross-chunk reference checks
- Important boundary:
	- no viewer/world transform policy
	- no CK24 object reconstruction
	- no dependency on `MdxViewer` PM4 solver code or the current `WoWMapConverter.Core` PM4 models
- Preferred real-data reference tile for PM4 rediscovery:
	- use `test_data/development/World/Maps/development/development_00_00.pm4` first when checking raw chunk assumptions or viewer-forensics hypotheses
	- Mar 21 standalone analysis on that tile showed it is a dense PM4 file, not a degenerate edge case: `54` chunks, `MSPV=8778`, `MSVT=6318`, `MSCN=9990`, `MPRL=2493`
	- new Mar 21 audit result: `00_00` is also the only currently populated destructible-building payload tile in the in-repo development PM4 corpus; `MDBI` and `MDBF` are one-tile only, while `MDBH` / `MDOS` / `MDSF` mostly appear as empty or placeholder stubs elsewhere
	- the matching original ADTs are not present in this repo, so in-repo validation is currently PM4-side only; external visual cross-checks should still prefer this tile because the user has the trusted ADT placements for it
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Core/Pm4Research.Core.csproj -c Debug` PASSED on Mar 21, 2026.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and found zero file-walk/stride diagnostics across the 616-file corpus, but did surface `MSLK.RefIndex -> MSUR` mismatches in aggregate and the Wintergrasp-only destructible payload split described above.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and narrowed that open seam further: `150` files carry `4553` mismatches, `development_00_00.pm4` carries zero mismatches, and the bad values almost never fit `MPRL` counts but often still fit `MSLK`, `MSPI`, `MSVI`, and `MSCN` counts on the affected tiles.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and materially tightened the identity/hierarchy seam: the UI `Ck24ObjectId` is just the low 16 bits of `MSUR.PackedParams -> CK24`, it is almost always one-to-one with a full CK24 within a file (`2` reuse cases out of `1601` analyzed non-zero object-id groups), and `MSLK.GroupObjectId` remains very weak as the missing hierarchy/ownership key for the unresolved `RefIndex` population (`16` low16 matches and `15` low24 matches across `4553` mismatches).
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and materially tightened the MSCN seam: `MSUR.MdosIndex -> MSCN` is strong (`511891` fits, `6201` misses), `1886 / 1895` CK24 groups carry MSCN coverage, and in the standalone raw path raw MSCN bounds overlap CK24 mesh bounds far more often than swapped-XY MSCN bounds (`1162` vs `10` fits). Current standalone corpus evidence does not support the older blanket claim that MSCN is simply world-space plus XY swap.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-msur-geometry --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_msur_geometry_report.json` PASSED on Mar 21, 2026 and materially tightened a major decoder-trust seam: all `518092` analyzed `MSUR` surfaces had unit-length stored normals with strong positive alignment to geometry-derived polygon normals, and the trailing float currently named `Height` behaves like the negative plane-distance term along that normal (best candidate mean absolute error `0.00367829`).
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex-classifier --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_mslk_refindex_classifier_report.json` PASSED on Mar 21, 2026 and replaced the old all-or-nothing mismatch story with family buckets: `505` mismatch families are now classified beyond pure ambiguity, covering `2651` of `4553` mismatch rows, with the largest resolved family population currently landing in `probable-MSVT` plus smaller `MSPI` / `MSPV` / `MSVI` / `MSCN` / `MPRL` slices.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-structure-confidence --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and is now the explicit decode-trust guardrail for the standalone PM4 path: `13` tracked chunk families currently land in `high` layout confidence, but field semantics are much weaker (`1` high, `4` medium, `10` low, `4` very-low). The main hallucination-risk zone is semantic over-closure, not raw stride parsing.
	- refreshed `scan-structure-confidence` result after the new audits: field semantics are still weaker than layout confidence, but the picture improved materially (`2` high, `4` medium, `9` low, `4` very-low). Current highest-risk zones are `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible fields; `MSUR` bytes `4..19` are no longer in that top-risk bucket.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- pm4-validate-coords --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --json i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_pm4_coordinate_validation_report.json` PASSED on Mar 21, 2026 and materially strengthened `MPRL` against real placement truth on the fixed dataset: `206` tiles validated, `114301 / 114301` refs inside expected tile bounds (`100.0%`), `107907 / 114301` refs within `32` units of a nearest `_obj0.adt` placement (`94.4%`), average nearest placement distance `10.98`. This helps `MPRL`, not `MPRR`.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and now serves as the main corpus-scale PM4 unknowns map: it records verified raw edges, partial fits, field distributions, and open proof tasks in one place.
	- structure-confidence highlights to preserve for future PM4 work:
		- strongest byte+semantic anchors: `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR` plane fields, `MSUR -> MSVI`, and `MDSF -> {MSUR, MDOS}`
		- highest hallucination-risk fields: `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible payload fields such as `MDOS.buildingIndex`
		- explicit conflict inventory now exists for overstated legacy claims around `MSLK.LinkId`, `MSLK.RefIndex`, `MSUR.MdosIndex`, `MSUR.Normal + Height`, MSCN coordinate frame, and `MPRR.Value1`
	- no automated tests were added or run.
	- no real-data runtime signoff exists yet because this is a standalone decode/exploration foundation, not an integrated viewer fix.

### M2 Material Parity Slice: Explicit Env-Map + UV Selector Recovery (Mar 21)

### Archive I/O Performance Slice: Read-Path Probe Reduction + Useful Prefetch Instrumentation (Mar 21)

### ViewerApp Partial-Class Refactor (Mar 21)

- `src/MdxViewer/ViewerApp.cs` was reduced by extracting cohesive UI domains into partial-class files instead of doing a behavior rewrite:
	- `src/MdxViewer/ViewerApp_ClientDialogs.cs`
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
- The goal of this slice is maintainability only: keep existing viewer behavior while shrinking the single 6000+ line shell file and making future UI changes more localized.
- Current limit of the extraction:
	- the large world-objects body still lives behind `DrawWorldObjectsContentCore()` in `ViewerApp.cs`; the refactor did not attempt a full inspector redesign in this pass.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 after the split.
	- no automated tests were added or run.
	- no runtime real-data validation was done because this change is structural, not a terrain/data-path behavior fix.

### Viewer UI / Perf Slice: Hideable Chrome + Clipped Long Lists (Mar 21)

### Viewer UI Follow-Up: Dockspace Host + Dockable Side Panels (Mar 21)

### Viewer PM4/WMO Correlation Export (Mar 21)

- `MdxViewer` now exposes a viewer-side PM4/WMO correlation export in the existing `PM4 Alignment` window.
- Current implementation:
	- `ViewerApp_Pm4Utilities.cs` adds `Dump PM4/WMO Correlation JSON` next to the existing PM4 object dump.
	- `WorldScene.BuildPm4WmoPlacementCorrelationJson(...)` exports loaded ADT WMO placements, parsed WMO mesh summaries, and top nearby PM4 overlay object candidates per placement.
	- `WorldAssetManager` now exposes `WmoMeshSummary`, reusing the existing WMO v14/v17 parsing path to capture local bounds plus group/vertex/index/triangle counts without depending on a renderer instance.
- Scope / limit:
	- this is a correlation/export utility, not closure on PM4-to-WMO semantic identity.
	- current matching is still heuristic, but it is no longer AABB-only: ranking now uses transformed WMO footprint samples versus PM4 footprint hulls in addition to bounds-gap / overlap metrics and PM4 object metadata.
- Follow-up now landed on top of the export path:
	- `ViewerApp_Pm4Utilities.cs` now adds a real `PM4/WMO Correlation` window with refresh/filter controls, placement browsing, candidate inspection, PM4 selection, and camera framing actions.
	- `WorldScene` now exposes a typed PM4/WMO correlation report for viewer use instead of forcing the UI to go through JSON only.
	- `WorldScene.SelectPm4Object(...)` lets the panel drive live PM4 selection from a reported candidate row.
	- `WorldAssetManager.WmoMeshSummary` now caches sampled WMO geometry points so the correlation path can compare transformed footprint shape instead of only transformed bounds.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 after the interactive panel + footprint follow-up, with existing warnings.
	- no automated tests were added or run.
	- no runtime real-data signoff was performed yet for the new panel workflow or the footprint-based ranking changes.

- Latest user feedback after the clipped-list shell pass: `World Maps` starting collapsed was wrong, and the viewer still did not have a real dock-panel UI.
- Current correction in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- ImGui docking is now explicitly enabled in source instead of relying on stale layout state in `imgui.ini`.
	- the viewer now creates a real central dockspace host between the menu/toolbar region and the status bar.
	- the old fixed left/right sidebars can now render as normal dockable windows (`Navigator` and `Inspector`) when dock panels are enabled from the `View` menu.
	- `World Maps` now defaults open again on first draw.
	- scene viewport math no longer subtracts fixed sidebar widths, which was incompatible with docked/floating panels.
- Validation status for this follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on the docking workflow or interaction feel; do not over-claim the UI recovery from build success alone.

- Latest user priority shifted from PM4 transform tuning to viewer usability and frame-time friction while debugging PM4.
- Current implementation in `src/MdxViewer/ViewerApp.cs` is intentionally incremental, not a dockspace/UI-shell rewrite:
	- `Tab` now toggles a hide-chrome mode for the menu bar, toolbar, sidebars, status bar, and floating utility windows while keeping modal dialogs available.
	- left/right sidebar sections no longer all default open on first draw; the shell now starts less expanded by default.
	- large UI lists now use clipped child-list rendering instead of drawing every row every frame:
		- file browser
		- discovered maps
		- subobject/group visibility toggles
		- WMO / MDX placement lists
		- POI / taxi node / taxi route lists
- Scope / limit of this slice:
	- this reduces known UI hot spots and improves focus-mode usability, but it is not a full restoration of the older dockable UI and not proof yet of runtime frame-time recovery on the fixed development dataset.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on actual UI responsiveness or PM4-debugging flow; do not over-claim the perf impact from build success alone.

- Confirmed hot seam on the active viewer path:
	- `WorldAssetManager.ReadFileData(...)` was still issuing repeated alias/fallback `ReadFile(...)` probes on top of `MpqDataSource`, including duplicate lowercase and `.mpq` retries that the MPQ data source already handled internally through case-insensitive normalization and Alpha wrapper resolution.
	- `MpqDataSource` had a raw-byte cache and worker prefetch path already, but it did not expose exact counters for direct read cache behavior, resolution source, or prefetch queue latency.
- Current implementation change:
	- `MpqDataSource` now exposes precise archive-I/O counters through `MpqDataSourceStats`:
		- `FileExists` request/cache/source counters
		- `ReadFile` request/cache/source counters (`loose`, `alpha wrapper`, `MPQ`, `miss`)
		- average uncached read latency
		- prefetch enqueue/dedup/cache-skip/completion counters plus average queue-wait and worker-read latency
	- `WorldAssetManager` now exposes `WorldAssetReadStats` and caches the winning resolved asset path per requested model/WMO read so later retries can jump straight to the known-good candidate instead of replaying the whole fallback chain.
	- Redundant work removed from the active world-asset path:
		- removed duplicate lowercase retry in `WorldAssetManager.ReadFileData(...)`
		- removed duplicate `.mpq` retry there for Alpha wrapper reads because `MpqDataSource.ReadFile(...)` already resolves the wrapper path directly
		- deduped candidate enumeration before trying alternates / stripped-filename / prefixed fallbacks
	- Prefetch policy is now narrower and more scene-aligned:
		- prefetch uses the canonical resolved model path first
		- if that canonical path is known, it no longer fans out across all extension aliases
		- M2 prefetch now warms the best resolved `.skin` path first and only falls back to generic skin candidates when no indexed best match exists
	- Viewer terrain/world stats panel now surfaces both `WorldAssetManager` probe counters and `MpqDataSource` cache/prefetch counters for runtime measurement.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice.
	- no runtime real-data validation has been run yet on fixed MPQ-era data; do not claim generalized scene-streaming improvement from build success alone.

- The active M2-family renderer gap was confirmed to be material-state flattening inside `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`, not missing shader hooks in `ModelRenderer` first.
- Current landed slice recovers one explicit source seam instead of adding new transparency heuristics:
	- M2 skin batch metadata now preserves `textureCoordComboIndex` from raw `.skin` data and merges it back into the Warcraft.NET-derived skin path.
	- raw `MD20` vertex decode now preserves both UV sets instead of dropping everything to the first texture coordinate pair.
	- `textureCoordCombos` lookup now drives `MdlTexLayer.CoordId`; lookup value `-1` now marks the layer as `SphereEnvMap`, and lookup value `1` can select UV1 where present.
	- `ModelRenderer` now emits focused debug traces showing pass + resolved material family for M2-adapted batches when MDX debug focus is enabled.
- Scope of this slice:
	- improved family: reflective / env-mapped M2 surfaces, plus UV1-routed layers that were previously flattened to UV0
	- unchanged gaps: texture transform animation, color/transparency tracks, broader per-batch shader/material combo parity, and any runtime sorting issues beyond the existing pass split
- Validation status for this exact slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on reflective/env-mapped assets; do not claim PM4 matching benefit from this change alone

### M2 Material Parity Follow-Up: 4.0.0.11927 Wrap + Blend Correction (Mar 21)

- Follow-up runtime triage on Cataclysm-era M2 assets found two concrete material-state mismatches after the env-map / UV recovery slice:
	- `ModelRenderer` was only treating `WrapWidth` / `WrapHeight` as M2 repeat flags for the pre-release `3.0.1` profile, leaving later M2 builds on the old classic-MDX clamp interpretation.
	- `WarcraftNetM2Adapter.MapBlendMode(...)` was shifted after mode `2`, so M2 blend ids `4`..`7` were routed into the wrong local material families.
- Current correction:
	- all M2-adapted models now interpret wrap X/Y as repeat flags; classic MDX keeps the legacy clamp-flag behavior.
	- M2 blend ids now map as `0=Load`, `1=Transparent`, `2=Blend`, `3=Add` (`NoAlphaAdd`), `4=Add`, `5=Modulate`, `6=Modulate2X`, `7=AddAlpha` (`BlendAdd`).
	- the local renderer still has no distinct `NoAlphaAdd` or `BlendAdd` states, so those cases are now collapsed intentionally into the nearest additive families instead of landing there because of an off-by-one bug.
- Validation status for this exact follow-up slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on `4.0.0.11927` M2 assets; do not claim visual parity from build success alone

### PM4 Orientation Follow-Up: World-Space Solver No Longer Forces Mirrored Swap Fits (Mar 21)

### PM4 Link-Decode Follow-Up: Legacy `MSLK` Surface Index Defaults No Longer Leak As Real Data (Mar 21)

### PM4 MPRL Axis Contract Correction (Mar 21)

- Follow-up after comparing the active viewer path against older PM4 R&D exports and `WoWRollback/Pm4Reader` forensic notes.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the active viewer restores the older fixed `MSVT` viewer/world basis `(Y, X, Z)` for the common `XY+Zup` path instead of trying to recover that basis later with per-object planar heuristics.
	- axis convention is now held file-level again across CK24 groups instead of being redetected per CK24; this avoids neighboring PM4 pieces drifting into different mesh bases.
	- viewer-side `MPRL` positions are now converted to world as `(PositionX, PositionZ, PositionY)` so they line up with that restored `MSVT` basis during planar scoring, nearest-anchor comparisons, and PM4 position-ref marker rendering.
	- the previous viewer assumption that `MPRL` could be treated as ADT-style planar `X/Z`, vertical `Y` or as raw `Z/X/Y` world output is no longer the active contract.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet that this closes the reported PM4 placement failure.

### PM4 Render-Derivation Follow-Up: Overlay Objects Now Keep An Explicit Local Frame (Mar 21)

- Follow-up after runtime evidence that PM4 mesh pieces were effectively being treated as if they were already in final placed space, which makes it too easy to conflate object-local shape with world placement.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- `Pm4OverlayObject` now localizes its line/triangle geometry around a preserved pre-split linked-group placement anchor instead of storing only fully placed geometry.
	- each PM4 overlay object now carries a baked base placement transform that restores that anchored object-local geometry into the solved placed frame.
	- when one CK24 is split into linked-group / MDOS / connectivity-derived parts, those parts keep the original linked-group placement anchor instead of rebasing to per-fragment centers.
	- overlay rendering now applies that baked base transform first, then any global PM4 overlay transform and object-local alignment edits on top.
	- PM4 JSON export now rehydrates placed-space geometry from the baked base transform so the interchange dump still matches what the viewer is rendering.
- Scope / limit:
	- this is structural groundwork for the missing “mesh inside stable object frame” layer; it is not a claim that final PM4 natural-rotation decoding is solved.
	- the CK24 placement solve itself is unchanged in this slice.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet that this resolves the remaining PM4 orientation mismatch.

- Runtime investigation on `test_data/development/World/Maps/development/development_00_00.pm4` found a concrete active-path bug during PM4 rotation forensics:
	- `WoWMapConverter.Core.Formats.PM4.MslkEntry` exposes `MsurIndex`, `MsviFirstIndex`, and `MsviIndexCount`
	- `WorldScene` consults `MsurIndex` when grouping/linking surfaces and linked `MPRL` refs
	- but `Pm4File.PopulateLegacyView(...)` was never populating those legacy fields, so `MsurIndex` defaulted to `0`
- Current correction:
	- legacy `MSLK` entries created from the canonical decoder now explicitly set sentinel values for the unsupported fields (`MsurIndex = uint.MaxValue`, `MsviFirstIndex = -1`, `MsviIndexCount = 0`) instead of leaking fake `0` values into the viewer
	- this keeps `WorldScene` on the existing `RefIndex` fallback path unless a real surface index is available in the future
- PM4 rotation-forensics result from `development_00_00.pm4`:
	- raw `MPRL.Unk04` values only span about `0.01° .. 22.3°` on this tile
	- treat that field as a narrow local heading/placement signal on this file, not as proven absolute object yaw for the whole placed building set
	- `Unk06` is constant `0x8000` on this tile, and `Unk16` still behaves like normal-vs-terminator entry typing
	- `Unk14` continues to look like floor/level bucketing, not pitch/roll
- Viewer debugging follow-up:
	- selected PM4-object debug info now shows linked `MPRL` normal/terminator counts, floor range, and heading min/max/mean so runtime object picks can be compared against raw PM4 placement stats directly
- Validation status for this follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026
	- no automated tests were added or run
	- no runtime signoff yet that the selected `CK24=0x421809` object now has the final correct orientation; this slice improves link-data integrity and observability first

- Use `documentation/pm4-current-decoding-logic-2026-03-20.md` as the authoritative viewer-side PM4 reconstruction contract for the active branch.
- That doc was refreshed on Mar 21, 2026 to capture the current CK24 pipeline, the tile-local versus world-space planar candidate split, and the rollback of the linked-`MPRL` center-translation experiment.

### PM4 Tile-Local Orientation Follow-Up: Quarter-Turn Swap Solve No Longer Rotates Non-Origin Tiles (Mar 21)

- Latest runtime PM4 report narrowed a second orientation seam after the world-space solver fix: tiles beyond `0_0` / `0_1` were coherently rotated about `90°` counter-clockwise while origin-adjacent tiles still aligned.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the quarter-turn planar transform expansion was also being offered to tile-local PM4
	- tile-local PM4 already has a fixed south-west tile basis, so per-tile `swap` solving could rotate whole non-origin tiles even when the underlying tile basis was correct
- Current correction:
	- tile-local PM4 now tests only non-swapped mirror candidates inside the established tile basis
	- tile-local PM4 world assembly now applies the file tile indices in viewer-world order (`tileY -> worldX`, `tileX -> worldY`) instead of the naive unswapped pairing that only looked right on origin tiles
	- quarter-turn `swap` candidates remain world-space only
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet on the non-origin tile placement/orientation case; do not claim PM4 tile closure from build success alone.

- Runtime PM4 alignment evidence showed some objects resolving to mirrored planar transforms like `swap=True, invertU=False, invertV=False`, which reverses handedness and makes stairs/ramps wind the wrong way around structures.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- world-space PM4 candidate enumeration only tested `identity` and `swap`
	- rigid quarter-turn candidates were never considered, so some world-space objects could only be approximated by mirrored solutions
- Current correction:
	- world-space PM4 now evaluates the rigid planar set first: identity, 180 degree, +90 degree, and -90 degree basis changes
	- mirrored candidates are no longer part of the active PM4 planar solver; the viewer now stays on rigid candidates only to avoid reversed winding and opposite-facing fits
- Validation status for this exact PM4 solver slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on the guardtower staircase case; do not claim closure from build success alone

### PM4 Bounds Overlay Follow-Up: Per-Object PM4 Bounds Are Now Visible In-Scene (Mar 21)

### PM4 MPRL Frame Follow-Up: Linked-Center Translation Experiment Reverted (Mar 21)

- The earlier linked-`MPRL` frame experiment turned out to regress PM4 placement badly in runtime user validation.
- Latest runtime evidence also argues against the broader `MPRL` bounding-box/container paradigm itself: PM4 geometry and PM4 bounds are not conforming to that model in the viewer.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the viewer-side reconstruction path was translating whole CK24 groups into the linked `MPRL` world-bounds center after geometry pivot/yaw solve.
	- that shared translation was too aggressive and made PM4 alignment worse instead of better.
- Current correction:
	- the linked-center translation path was removed from `BuildPm4TileObjects(...)`.
	- CK24 rendering is back to the prior geometry-pivot path with the existing coarse yaw-correction logic.
	- this keeps the earlier `12°` suppression of small principal-axis yaw deltas, but no longer forces linked PM4 groups into an MPRL-center translation frame.
- Current interpretation:
	- user/domain correction: `MPRL` points are terrain/object collision-footprint intersections where ADT terrain is pierced by object collision geometry.
	- keep rejecting the old `MPRL` center/bounds translation experiment.
	- do not assume PM4 objects should fit inside an `MPRL` bounding box or container frame; use `MPRL` as footprint/collision reference data instead.
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 with existing solution warnings only.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet on whether PM4 alignment is restored; do not claim placement closure from build success alone.

### PM4 Yaw Follow-Up: Small Principal-Axis Corrections No Longer Override Near-Correct MPRL Rotation (Mar 21)

- Latest runtime user feedback on PM4 overlay alignment: objects were no longer wildly mis-rotated, but many still looked consistently off by roughly `5..10` degrees around the vertical axis.
- Root cause narrowed in `src/MdxViewer/Terrain/WorldScene.cs`:
	- PM4 MPRL yaw decode was already being rebased and then compared against a geometry-derived principal-axis yaw.
	- the follow-up CK24 world-yaw correction stage was still applying small residual deltas (`>= 2°`), which is too aggressive for irregular object footprints and can turn "almost correct" PM4 orientation into a visible small bias.
- Current correction:
	- CK24 continuous yaw correction is now treated as a coarse recovery step only.
	- residual yaw deltas below `12°` are ignored, leaving MPRL-derived orientation authoritative for near-correct objects.
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet after the threshold change; do not claim PM4 rotation closure from build success alone.

- Latest PM4 alignment feedback showed MPRL anchors lining up while other PM4 object extents still felt offset or nested inside the wrong container, making click-and-compare work too opaque.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs`:
	- PM4 per-object bounds that were already computed for picking/culling/debug info are now rendered directly in-scene through the existing `BoundingBoxRenderer` path.
	- the PM4 alignment controls now expose a dedicated `PM4 Bounds` toggle beside `PM4 MPRL Refs` and `PM4 Centroids`.
	- selected PM4 object groups get a highlighted bounds color, and the exact selected PM4 object gets a white bounds box.
- Important scope note:
	- these bounds are currently built from the rendered PM4 object geometry (`MSVT`/`MSVI`/`MSUR` path), not from `MSCN`.
	- treat this as a visibility/debugging aid for the current PM4 reconstruction path, not proof yet that the active PM4 extents are sourced from the final correct container.
- Validation status for this exact PM4 bounds slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 with existing solution warnings only.
	- no automated tests were added or run for this slice.
	- no runtime real-data signoff yet on PM4 bounds usefulness or on the MSCN-versus-MSVT extent question.

### MPQ Base Build Selection Recovery (Mar 21)

- The active viewer no longer relies only on `InferBuildFromPath(...)` for new MPQ loads.
- `ViewerApp` now restores explicit build selection before loading a game folder:
	- MPQ open flow now pauses on a build-selection dialog.
	- build choices come from `Terrain/BuildVersionCatalog.cs` using `WoWDBDefs/definitions/Map.dbd` when available, with a built-in fallback list that includes `4.0.0.11927` and `4.0.1.12304`.
	- path/build tokens are now treated as preselection hints, not authoritative routing.
- Known-good base-client entries now persist `BuildVersion` in viewer settings and reuse it when reopening a saved base or attaching a loose overlay against that base.
- Loose overlay attach now emits a PM4 build hint when the overlay contains PM4 files with known version markers:
	- `12304` => `4.0.1.12304`
	- `11927` => `4.0.0.11927`
	- if that hint disagrees with the active base build, the viewer logs a warning instead of silently continuing with no build-era signal.
- Validation status for this build-routing slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No automated tests were added or run for this slice.
	- No runtime real-data signoff yet on PM4/world-object matching with a `4.0.1.12304` base client.

### 4.0.0.11927 Terrain Blend Recovery (Mar 21)

- The earlier working assumption that 4.0 terrain texturing was effectively "3.3.5 MCAL decode with split files" is now documented as incomplete.
- Latest wow.exe RE confirms the missing behavior is runtime blend assembly, not only local MCAL byte decode:
	- `CMapChunk_UnpackChunkAlphaSet` stitches the current chunk with three linked neighbor chunks.
	- Neighbor alpha is matched by texture id, not only by local overlay slot index.
	- In 8-bit mode, layers without direct alpha payload can be synthesized as residual coverage `255 - other layer alphas`.
	- Blend textures are rebuilt through the `TerrainBlend` resource path (`CMapChunk_BuildSingleLayerBlendTexture`, `CMapChunk_BuildChunkBlendTextureSet`, `CMapChunk_RefreshBlendTextures`).
- Active viewer implementation now reflects the first verified slice of that model:
	- `FormatProfileRegistry.AdtProfile40xUnknown` routes to `TerrainAlphaDecodeMode.Cataclysm400`.
	- `StandardTerrainAdapter` captures per-layer source flags, synthesizes residual 8-bit alpha for missing direct payloads, and stitches same-tile chunk edges by matching neighbor layer texture ids.
	- `TerrainChunkData` now preserves `AlphaSourceFlags` for runtime post-processing.
- Documentation/handoff files updated for this recovery line:
	- `documentation/wow-400-terrain-blend-wow-exe-guide.md`
	- `docs/archive/WoW_400_ADT_Analysis.md`
	- `docs/archive/WoW_400_DeepDive_Analysis.md`
	- `docs/archive/WoW_301_DeepDive_Analysis.md`
	- `docs/ADT_WDT_Format_Specification.md`
	- `specifications/ghidra/prompt-400.md`
	- `.github/prompts/wow-400-terrain-blend-recovery.prompt.md`
- Validation status for this exact 4.0 recovery slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No real-data runtime signoff yet on the fixed development terrain after residual synthesis + edge stitching.
	- Do not claim 4.0 terrain correctness from build success or diagnostics alone.

### WMO Blend + Loose PM4 Overlay Follow-Up (Mar 21)

- WMO distant "foggy sheen" triage found one concrete renderer mismatch in `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- the active branch had flattened WMO material blend handling into opaque vs generic transparent
	- current code now maps raw WMO `BlendMode` to `EGxBlend` semantics (`Opaque`, `Blend`, `Add`, `AlphaKey`)
	- opaque pass now keeps `AlphaKey` with alpha-test, while transparent pass only handles `Blend` / `Add`
- Loose overlay PM4 resolution now gives precedence to the most recently attached overlay root in `src/MdxViewer/DataSources/MpqDataSource.cs`.
	- this matters when a base path and a later loose overlay both expose the same PM4 virtual path
	- older behavior searched loose roots in insertion order, so base loose files could shadow the attached overlay
	- current resolver searches newest overlay first and now traces PM4 loose-path misses like WMO misses
- Validation status for these viewer fixes:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No automated tests were added or run for these fixes.
	- No runtime real-data signoff yet for the WMO sheen symptom or the loose-overlay PM4 workflow.

### PM4 Decode Triage + Rendering Parity Program (Mar 21)

- Current PM4 overlay failure state has moved past indexing/attach into decode-or-reconstruction triage:
	- runtime symptom seen by the user: `PM4: 2674 files found, none decoded into overlay data.`
	- this means PM4 candidates are being found, but none produced renderable overlay objects
	- latest `WorldScene.LazyLoadPm4Overlay()` instrumentation now buckets that failure into:
		- tile-parse rejection
		- tile-range rejection
		- read failure
		- decode failure
		- parsed-but-zero-object files
- Working hypothesis for the `4.0` versus `3.3.5` split:
	- PM4 parsing/object assembly itself appears build-agnostic
	- the likely seam is build-dependent map discovery / WDT resolution / candidate-set selection through `_dbcBuild`
	- the observed `2674` candidate count is suspicious versus the fixed development dataset note in `memory-bank/data-paths.md` (`616 PM4 files`) and should be treated as a clue, not normal noise
- Rendering work is now explicitly grouped as one coordinated program because PM4 object-variant matching depends on visually trustworthy output, not only PM4 geometry placement.
- The ordered rendering program is now:
	1. M2 material, transparency, and reflective-surface parity
	2. lighting DBC expansion beyond the current `Light` + `LightData` subset
	3. skybox / environment parity so backdrop and lighting context stop misleading object matching
- Planning artifacts created for this program live under `.github/prompts/`:
	- `m2-material-parity-implementation-plan.prompt.md`
	- `lighting-dbc-expansion-implementation-plan.prompt.md`
	- `sky-environment-parity-implementation-plan.prompt.md`
- Validation status for this planning slice:
	- no rendering code changes landed yet from this program
	- no automated tests were added or run for the planning-only pass
	- no runtime real-data validation yet on the new PM4 failure-bucket diagnostics

### Recovery Work Completed On This Branch

- Re-established v0.4.0 baseline in the primary tree and validated build.
- Restored the project instruction stack from main:
	- copilot-instructions
	- instructions
	- prompts
	- terrain-alpha-regression skill files
- Applied profile-driven terrain alpha decode routing in viewer terrain path:
	- Added TerrainAlphaDecodeMode to AdtProfile in FormatProfileRegistry
	- 3.x profiles route to LichKingStrict
	- 0.x profiles route to LegacySequential
	- StandardTerrainAdapter alpha extraction now routes by profile mode
	- Strict path includes UseAlpha-first decode plus offset/span fallback for mis-set flags
	- Legacy path remains sequential 4-bit nibble expansion

### Validation Status

- Build: dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug PASSED
- Runtime real-data spot-check: PARTIAL PASS
	- user confirmed Alpha-era 0.5.3 terrain renders correctly again after the alpha-edge-fix restoration
	- user confirmed a 3.0.1 alpha build now renders correctly on the current profile-driven 3.x path
	- earlier 3.3.5 spot-check also looked correct, but broader cross-map signoff is still pending
- Do not claim full terrain regression safety beyond the validated samples above.

### Next Integration Queue (Ordered)

1. Commit and push the current profile/decode code slice if not already committed.
2. Broaden runtime-check alpha decode behavior beyond the currently validated 0.5.3 and 3.0.1 samples.
3. Continue commit-by-commit intake from v0.4.0..main with strict triage:
	 - SAFE first
	 - MIXED only with dependencies proven and build gates
	 - RISKY terrain renderer/decode rewrites skipped unless explicitly approved
4. Keep UI changes incremental; avoid broad layout rewrites.
5. Pull selected import/export functionality in small batches after profile/decode stabilization.

### Surgical Intake Triage (Mar 17)

- Commit triage against `v0.4.0..main` is now documented for the current queue:
	- `177f961`: RISKY, skip entire commit (terrain renderer + tile mesh + alpha decode rewrite)
	- `37f669c`: RISKY, skip entire commit (relaxed alpha heuristics + MPQ decompression changes)
	- `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, `62ecf64`: MIXED, only extract isolated safe slices
- First SAFE batch selected:
	- take only the corrected `TerrainImageIo` alpha-atlas helper from `62ecf64`
	- do not take the earlier `d50cfe7` version because it reintroduced atlas import/export edge remapping
	- do not take ViewerApp, TerrainRenderer, WorldScene, test-project, or alpha-decode hunks in the first batch
- Required gate after the first SAFE batch: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime terrain validation remains required after any terrain-adjacent batch; build success is not proof of terrain correctness.
- First SAFE batch status:
	- corrected `TerrainImageIo` helper has been applied in the recovery branch
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the change
	- runtime real-data validation is still pending

### Rendering Fix Batch (Mar 18)

- Applied the main-branch `WorldAssetManager` residency fix in the recovery branch:
	- MDX/WMO renderer residency now defaults to unlimited
	- only the raw file-data cache remains bounded
	- cached failed model loads are retried instead of becoming permanent null entries
	- lazy `GetMdx` / `GetWmo` lookups can now load on demand
- Applied the minimal main-branch skybox backdrop path without broad ViewerApp/UI churn:
	- skybox-like MDX/M2 placements are routed into a separate skybox instance list
	- nearest skybox placement renders as a camera-anchored backdrop before terrain
	- `ModelRenderer` now has a backdrop path that keeps depth test/write disabled for all layers
- Current branch already had the reflective M2 depth-flag fix and the guarded env-map backface handling, so those were not re-applied.
- Build gate passed again after this rendering batch: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation is still required for:
	- WMO/MDX disappearance when moving away and back
	- skybox model classification and backdrop behavior
	- MH2O liquid rendering on LK data

### MCCV + MPQ Recovery Batch (Mar 18)

- Restored active-branch MCCV terrain support in the chunk renderer path:
	- `StandardTerrainAdapter` now carries MCNK MCCV data into `TerrainChunkData`
	- `TerrainMeshBuilder` uploads per-vertex RGBA alongside position/normal/UV
	- `TerrainRenderer` consumes MCCV in the shader
- Follow-up correction after runtime feedback:
	- MCCV is now treated as BGRA, matching the repo's own `MinimapService.GenerateMccvData` documentation
	- neutral/no-tint MCCV is treated as mid-gray (`127`) rather than white
	- shader tinting now maps mid-gray to neutral and no longer relies on MCCV alpha as terrain tint strength
- Applied the isolated `NativeMpqService` slice from the mixed MPQ recovery commits:
	- broader patch archive priority ordering, including locale/custom patch variants
	- encrypted-file key derivation now tries the full normalized path first, then basename fallback
	- per-sector MPQ decompression now handles bitmask combinations instead of only single-byte cases
	- BZip2 sector decompression added via SharpZipLib
- Follow-up patch-chain fix:
	- `NativeMpqService.LoadArchives(...)` now discovers MPQs recursively instead of only scanning a few top-level directories
	- Alpha-style single-asset wrapper archives (`.wmo.mpq`, `.wdt.mpq`, `.wdl.mpq`) are still excluded from this generic path because `MpqDataSource` handles them separately
- Build gates passed after this batch:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime real-data validation is still required for:
	- 1.x+ patch-chain reads on patched client data
	- later-version encrypted MPQ entries
	- 3.x MCCV highlight/tint behavior on real LK terrain after the BGRA + mid-gray semantic correction

### WDL + Model Compatibility Follow-up (Mar 18)

- Follow-up after runtime feedback on the newly ported WDL preview cache:
	- the main WDL failure on 1.x/3.x was not only path lookup; `WoWMapConverter.Core.VLM.WdlParser` hard-rejected every non-`0x12` WDL
	- parser is now version-tolerant and scans for `MAOF`/`MARE` instead of requiring Alpha-only layout assumptions
	- parser also tolerates MAOF offsets that point either at a `MARE` chunk header or directly at the height payload
- Viewer-side WDL read paths are now unified through `WdlDataSourceResolver`:
	- both preview warmup and 3D WDL terrain now try `.wdl` and `.wdl.mpq`
	- MPQ-backed loads also use `MpqDataSource.FindInFileSet(...)` so listfile/casing recovery works consistently
- Remaining 3.x doodad extension parity gap closed in `WmoRenderer`:
	- canonical doodad resolution now tries `.m2` in addition to `.mdx`/`.mdl`
- Semi-translucent model follow-up in `ModelRenderer`:
	- shared texture cache entries now carry a simple alpha classification (`Opaque`, `Binary`, `Translucent`)
	- classic non-M2 layer-0 `Transparent` now stays on the hard alpha-cutout path only when the loaded texture alpha is binary
	- textures with intermediate alpha values now render through the blended path instead of the old foliage-style cutout heuristic
- Build gate passed after this batch:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation is still required before claiming:
	- non-Alpha WDL previews / WDL 3D terrain actually load on the user's real 1.x/3.x map set
	- 3.x `.mdx` WMO doodads now resolve correctly as M2-family assets on real data
	- the semi-translucent material heuristic fixes the reported visuals without regressing classic cutout foliage

### WDL Spawn Chooser Regression Handoff (Mar 20)

- Latest runtime report from the active branch: WDL heightmap spawn chooser is currently non-functional across tested versions.
- Treat earlier notes that framed the spawn chooser path as working as stale until revalidated.
- Scope this as a viewer flow regression, not a parser-complete claim:
	- likely touchpoints are spawn action enablement (`WdlPreviewWarmState` gating), preview readiness transitions, and preview dialog/open fallback routing
	- this may involve both UI state and async warmup timing, not just WDL decode
- Do not close this issue on build success or file-level diagnostics alone.
- Required signoff for closure:
	- real-data runtime verification on at least one Alpha-era map and one 3.x map
	- explicit proof that spawn chooser opens/commits a spawn point and that fallback load behavior still works when preview prep fails

### PM4 Tile Mapping Runtime Handoff (Mar 20)

- PM4 viewer tile assignment now follows direct filename indices (`map_x_y.pm4` maps to `(tileX=x, tileY=y)`).
- The old MPRL-based tile reassignment heuristic has been removed from the PM4 overlay load path.
- Duplicate PM4 files mapping to one tile now merge object payloads/stats/refs instead of replacing prior data.
- Immediate next step after restart is runtime validation on the reported adjacency mismatch (`00_00`, `01_00`, and `01_01`) before further PM4 transform work.
- Do not claim this fixed from build-only validation; runtime signoff is still pending.

### M2 Empty-Fallback Guardrail (Mar 18)

- Follow-up after the standalone 3.x model-load freeze fix: some M2-family assets could still appear to load while producing a blank viewport and model info with zero geometry.
- Current conclusion is narrow:
	- this is at least partly a false-positive success path, not necessarily a valid render of an odd pre-release asset
	- raw `MD20` fallback conversion can yield an `MDX` shell that parses but has no renderable geosets
- Recovery change applied:
	- shared geometry validation added for converted M2 fallback results
	- standalone `ViewerApp`, world `WorldAssetManager`, and WMO doodad `WmoRenderer` now reject empty converted fallback models and keep the real failure path visible in logs
- Validation status:
	- alternate-OutDir build passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
	- no runtime real-data validation yet
	- do not over-claim this as a full M2 render fix for pre-release `3.0.1`; it is a guardrail that removes a misleading blank-success outcome

### Pre-release 3.0.1 M2 + Shared Transparency Follow-up (Mar 18)

- User runtime verification now narrows the remaining model issue further:
	- most unresolved M2 failures are specific to the pre-release `3.0.1` model family, not the later `3.3.5` layout
	- the active working assumption is that this pre-release family may be a hybrid or transitional `MDX` + `M2` variant rather than a clean later-WotLK `M2`
- Treat this as a separate compatibility track:
	- do not assume later `3.3.5` `MD20` / `.skin` semantics are sufficient for pre-release `3.0.1`
	- keep profile/version-aware model parsing on the roadmap instead of broadening generic fallback heuristics
	- the empty-fallback guardrail remains useful, but it is only a diagnostics fix
- Separate rendering issue still confirmed by runtime evidence:
	- neon-pink transparent surfaces still reproduce on both classic `MDX` and M2-family assets
	- that means the pink/transparency bug is not only an M2 parser problem; it is likely in shared material, texture binding, blend, or shader behavior
- Practical next investigation split:
	1. pre-release `3.0.1` model-structure compatibility in `WarcraftNetM2Adapter` / profile routing
	2. shared transparent-material shader parity across `ModelRenderer` and any M2-converted runtime path

### Pre-release 3.0.1 wow.exe Guide Handoff (Mar 19)

- Latest Ghidra pass mapped the common model load chain in `wow.exe` build `3.0.1.8303`:
	- `FUN_0077e2c0` -> `FUN_0077d3c0` -> `FUN_0079bc70` -> `FUN_0079bc50` -> `FUN_0079bb30` -> `FUN_0079a8c0`
- High-confidence parser contract now documented in `documentation/pre-release-3.0.1-m2-wow-exe-guide.md`:
	- root must be `MD20`
	- accepted version range is `0x104..0x108`
	- parser layout splits at `0x108`
	- shared typed span validators use strides `1`, `2`, `4`, `8`, `0x0C`, `0x30`, and `0x44`
	- confirmed nested record families include `0x70`, `0x2C`, `0x38`, `0xD4`, and `0x7C`
	- legacy side uses `0xDC` + `0x1F8`; later side uses `0xE0` + `0x234`
- Fresh-chat prompts now exist for implementation, deeper Ghidra follow-up, and runtime triage:
	- `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`
- Do not treat the guide as proof that viewer support is implemented yet:
	- no new runtime validation happened in this documentation pass
	- Track B pink transparency remains separate

### Pre-release 3.0.1 Profile Routing Broadening (Mar 19)

- Active profile resolution is no longer restricted to exact build `3.0.1.8303`.
- `FormatProfileRegistry` now maps any parsed `3.0.1.x` build to the existing pre-release `3.0.1` ADT, WMO, and M2 profiles.
- Keep the scope narrow:
	- this is profile routing, not full parser completion for every remaining pre-release `3.0.1` model difference
	- other `3.0.x` builds still use the generic `3.0.x` fallback profile unless new binary evidence justifies a tighter mapping
- Validation status:
	- code change applied
	- build/runtime validation still pending for this exact routing update

### Pre-release 3.0.1 Parser + Fallback Alignment (Mar 19)

- `WarcraftNetM2Adapter` now has a dedicated pre-release `MD20` parse path based on the wow.exe contract instead of routing those files through Warcraft.NET's later-layout `MD21` parser.
- Current scope of the fix:
	- standalone model load
	- world doodad load
	- WMO doodad load
	- shared `M2ToMdxConverter` fallback for those entry points
- Important implementation boundary:
	- the prior profile-specific `.skin` parser path was disabled because its `0x70` / `0x2C` record-size assumptions were lifted from model-family validation, not proven `.skin` layout evidence
	- converter fallback now keeps pre-release handling geometry-focused by skipping later-layout animation / bone parsing and by not forcing optional fixed-stride `.skin` submesh / texture-unit parsing
- Current residual risk:
	- runtime validation on real `3.0.1` assets is still outstanding
	- active MPQ build selection still relies on path/build inference unless a more explicit selector is ported later

### 3.x Alpha Follow-up (Mar 18)

- The LK offset-0 fallback experiment in `StandardTerrainAdapter.ExtractAlphaMaps(...)` was reverted after runtime validation showed it was wrong for the active 3.x terrain path.
- Current conclusion:
	- the recent attempt to treat `AlphaMapOffset == 0` as a valid relaxed-LK fallback case was not the correct fix
	- keep that path reverted and continue investigating 3.x alpha sourcing/decode without broadening fallback heuristics blindly
- Alternate-output build validation passed after reverting the tweak because a live `MdxViewer` process still had the normal `bin/Debug` outputs locked.

### 3.x Profile-Driven Alpha Recovery (Mar 18)

- Follow-up investigation confirmed the active recovery branch was still missing two important 3.x inputs that existed in rollback code:
	- WDT/MPHD big-alpha detection should treat `0x4 | 0x80` as the effective big-alpha mask for 3.x profiles
	- 3.x layer/alpha/shadow sourcing may need to come from split `*_tex0.adt` MCNK data rather than the root ADT alone
- Recovery changes now applied:
	- `AdtProfile` carries `BigAlphaFlagsMask` and `PreferTex0ForTextureData`
	- 3.0.1 / 3.3.5 profiles use `0x4 | 0x80` and prefer `*_tex0.adt`
	- `StandardTerrainAdapter` can build a `*_tex0.adt` MCNK index map and source MTEX/layers/MCAL/MCSH from that file when the profile says to
	- `StandardTerrainAdapter` now passes the MCNK `0x8000` do-not-fix-alpha bit into MCAL decode and uses chunk-level big-alpha inference instead of the reverted offset-0 fallback
	- `WoWMapConverter.Core.Formats.LichKing.Mcal` now has the stronger compressed / big-alpha / 4-bit decode split with proper edge-fix suppression for big-alpha and do-not-fix chunks
- Build gates passed after this batch:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build "I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="I:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime signoff is still pending:
	- confirm real 3.x tiles stop falling back to obvious 4-bit Alpha-style layer-1-only behavior
	- confirm split `*_tex0.adt` sourcing is actually the missing piece on the user’s 3.x client data

### Commit 39799bf Model Slice (Mar 18)

- The M2 load-failure fix associated with `39799bf` was the `NativeMpqService` encrypted-read compatibility slice, which is now already applied.
- The only additional model-renderer change from that commit was also applied:
	- `ModelRenderer` no longer renders particles on the world-scene batched instance path
	- standalone model viewing still renders particles
- Rationale:
	- world-scene batch instancing does not yet propagate per-instance transforms into particle simulation/rendering
	- leaving particles enabled there can produce camera-locked billboard artifacts on placed models
- Build gate passed again after applying this renderer hunk: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`

## Current Focus: MDX Compatibility Port + Rendering Parity (Feb 14, 2026)

MdxViewer is the **primary project** in the tooling suite. It is a high-performance 3D world viewer supporting WoW Alpha 0.5.3, 0.6.0, and LK 3.3.5 game data.

### Recently Completed (Feb 14)

- **GEOS Port (wow-mdx-viewer parity)**: ✅ `MdxFile.ReadGeosets` now routes by version with strict paths for v1300/v1400 and v1500, with guarded fallback.
- **SEQS Name Recovery**: ✅ Counted 0x8C named-record detection broadened so playable models no longer fall into `Seq_{animId}` fallback names in many cases.
- **PRE2 Parser Expansion**: ✅ Particle emitter v2 parser now reads full scalar payload layout, spline block, and skips known anim-vector tails safely for alignment.
- **RIBB Parser Expansion**: ✅ Ribbon parser now processes known tail anim-vector chunks safely for alignment.
- **Specular/Env Orientation Fix (shader)**: ✅ MDX fragment shader now flips normals/view-normals on backfaces before sphere-env UV and lighting/specular, targeting inside-out dome reflections.

### Previously Completed (Feb 11-12)

- **Full-Load Mode**: ✅ `--full-load` (default) / `--partial-load` CLI flags — loads all tiles at startup
- **Specular Highlights**: ✅ Blinn-Phong specular in ModelRenderer fragment shader (shininess=32, intensity=0.3)
- **Sphere Environment Map**: ✅ `SphereEnvMap` flag (0x2) generates UVs from view-space normals for reflective surfaces
- **MDX Bone Parser**: ✅ BONE/HELP/PIVT chunks parsed with KGTR/KGRT/KGSC keyframe tracks + tangent data
- **MDX Animation Engine**: ✅ `MdxAnimator` — hierarchy traversal, keyframe interpolation (linear/hermite/bezier/slerp)
- **Animation Integration**: ✅ Per-frame bone matrix update in MdxRenderer.Render()
- **WoWDBDefs Bundling**: ✅ `.dbd` definitions copied to output via csproj Content items
- **Release Build**: ✅ `dotnet publish -c Release -r win-x64 --self-contained` verified working (1315 .dbd files bundled)
- **GitHub Actions**: ✅ `.github/workflows/release-mdxviewer.yml` — tag-triggered + manual dispatch, creates ZIP + GitHub Release
- **No StormLib**: ✅ Pure C# `NativeMpqService` handles all MPQ access — no native DLL dependency

### Previously Completed (Feb 9-10)

- WMO doodad culling (distance + cap + sort + fog passthrough)
- GEOS footer parsing (tag validation)
- Alpha cutout for trees, MDX fog skip for untextured
- AreaID fix (low 16-bit extraction + fallback)
- Directional tile loading with heading-based priority
- DBC lighting (Light.dbc + LightData.dbc)
- Replaceable texture DBC resolution with MPQ validation

### Mar 19, 2026 - PM4 Coordinate Validation Slice

- Active core PM4 support now has one explicit coordinate-validation path built around `MPRL` refs already stored in ADT placement order.
- New active-core pieces:
	- `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs` defines the authoritative PM4 placement helpers for this first validation pass.
	- `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs` validates transformed `MPRL` refs against real `_obj0.adt` placements from the fixed development dataset.
	- `WoWMapConverter.Cli` now exposes `pm4-validate-coords`.
- Real-data validation status for this slice:
	- `wowmapconverter pm4-validate-coords --tile-limit 100` validated 100 PM4 tiles with placements from the fixed development dataset
	- 38,133 `MPRL` refs landed in expected tile bounds (100.0%)
	- 36,070 refs landed within a 32-unit nearest-placement threshold (94.6%)
	- average nearest-placement distance was 10.86 units
- Scope boundary:
	- this validates the `MPRL` anchor path only
	- cross-tile CK24 aggregation is still pending
	- MSCN/world-space semantics are still not the validated contract for active core code
- Do not claim PM4 world placement is fully solved beyond this `MPRL` path until CK24 aggregation and MSCN semantics are also validated.

### Mar 20, 2026 - PM4 Viewer Overlay Diagnostics + Grouping/Winding Pass

- PM4 support advanced from coordinate-validation-only into active viewer diagnostics in `src/MdxViewer/Terrain/WorldScene.cs` + `src/MdxViewer/ViewerApp.cs`.
- New viewer PM4 overlay capabilities now include:
	- multi-mode color classification (`CK24` type/object/key, tile, dominant group/attribute, height)
	- optional `MPRL` reference pins and PM4-object centroid pins
	- selected-object PM4 metadata readout (dominant group key, attribute mask, `MdosIndex`, planar transform flags, winding parity)
	- CK24 disjoint-geometry splitting toggles: connectivity split and optional `MdosIndex` pre-split
- Orientation correction changed from translation-first nudging to per-object planar transform solving with parity-aware triangle winding correction.
- Scope boundary for this pass:
	- still a viewer-side PM4 debug/reconstruction layer, not final cross-tile object identity
	- map-wide CK24 registry + MSCN semantics remain pending
- Validation status:
	- repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime real-data visual signoff remains pending for merged/disjoint object edge cases

### Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Alpha 0.5.3 WDT terrain | ✅ | Monolithic format, 256 MCNK chunks per tile |
| 0.6.0 split ADT terrain | ✅ | StandardTerrainAdapter, MCNK with header offsets |
| 0.6.0 WMO-only maps | ✅ | MWMO+MODF parsed from WDT |
| 3.3.5 split ADT terrain | ⚠️ | Loading freeze — needs investigation |
| WMO v14 rendering | ✅ | 4-pass: opaque/doodads/liquids/transparent |
| WMO liquid (MLIQ) | ✅ | matId-based type detection, correct positioning |
| Terrain liquid (MCLQ) | ✅ | Per-vertex sloped heights, absolute world Z |
| MDX rendering | ✅ | Two-pass, alpha cutout, blend modes 0-6 |
| Async tile streaming | ✅ | 9×9 AOI, directional lookahead, persistent cache |
| Frustum culling | ✅ | View-frustum + distance + fade |
| DBC Lighting | ✅ | Zone-based ambient/fog/sky colors |
| Minimap overlay | ✅ | BLP tiles, zoom, click-to-teleport |

### Known Issues / Next Steps

1. **Runtime validation pending (critical handoff item)** — verify PRE2/RIBB-heavy models visually after parser expansion.
2. **Specular/env dome check pending** — confirm Dalaran dome-like materials now reflect outward after backface normal correction.
3. **Residual SEQS/material parity work** — continue porting edge-case behavior from `lib/wow-mdx-viewer` if specific models still diverge.
4. **WMO semi-transparent window materials** — Stormwind glass still maps to wrong geometry (root cause unknown).
5. **MDX cylindrical texture stretching** — barrels/tree trunks still show stretched planks on some assets.
6. **3.3.5 ADT loading freeze** — needs investigation.
7. **WMO culling too aggressive** — objects outside WMO not visible from inside.

---

## Key Architecture Decisions

### Coordinate System (Confirmed via Ghidra)
- WoW: right-handed, X=North, Y=West, Z=Up, Direct3D CW front faces
- OpenGL: CCW front faces
- Fix: Reverse winding at GPU upload + 180° Z rotation in placement
- Terrain: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### Performance Constants

| Constant | Value | Location |
|----------|-------|----------|
| DoodadCullDistance (world) | 1500f | WorldScene.cs |
| DoodadSmallThreshold | 10f | WorldScene.cs |
| WmoCullDistance | 2000f | WorldScene.cs |
| NoCullRadius | 150f | WorldScene.cs |
| WMO DoodadCullDistance | 500f | WmoRenderer.cs |
| WMO DoodadMaxRenderCount | 64 | WmoRenderer.cs |
| AoiRadius | 4 (9×9) | TerrainManager.cs |
| AoiForwardExtra | 3 | TerrainManager.cs |
| MaxGpuUploadsPerFrame | 8 | TerrainManager.cs |
| MaxConcurrentMpqReads | 4 | TerrainManager.cs |

### Key Files

| File | Purpose |
|------|---------|
| `WorldScene.cs` | Placement transforms, instance management, culling |
| `WmoRenderer.cs` | WMO v14 GPU rendering, doodad culling, liquid |
| `ModelRenderer.cs` | MDX GPU rendering, alpha cutout, fog skip |
| `AlphaTerrainAdapter.cs` | Alpha 0.5.3 WDT terrain + AreaID + liquid type |
| `StandardTerrainAdapter.cs` | 0.6.0 / 3.3.5 split ADT terrain + MCLQ + WMO-only maps |
| `TerrainManager.cs` | AOI streaming, persistent cache, MPQ throttling |
| `LiquidRenderer.cs` | MCLQ/MLIQ liquid mesh rendering |
| `AreaTableService.cs` | AreaID → name with MapID filtering |
| `LightService.cs` | DBC Light/LightData zone-based lighting |
| `ReplaceableTextureResolver.cs` | DBC-based replaceable texture resolution |
| `MdxFile.cs` | MDX parser (GEOS, BONE, PIVT, HELP with KGTR/KGRT/KGSC tracks) |
| `MdxAnimator.cs` | Skeletal animation engine (hierarchy, interpolation, bone matrices) |
| `MdxViewer.csproj` | Project file with WoWDBDefs bundling |
| `.github/workflows/release-mdxviewer.yml` | CI/CD release workflow |
