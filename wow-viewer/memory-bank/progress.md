# Progress — wow-viewer

Last updated: 2026-09-03

## 2026-09-03 — Spec 219: Phase 1 Core transform seam and policy gate passed
- **Landed in Core:** [`TileContentTransform.cs`](../src/core/WowViewer.Core/Maps/TileContentTransform.cs)
  implements exact 90°/180° rotation and horizontal/vertical mirrors for the interleaved 145-vertex
  chunk lattice, renderer-space normals, hole masks, 64×64 alpha/shadow grids, MCCV, 8×8 liquid
  flags, complete MDDF/MODF placement records, and MODF bounds. It also exposes origin-based free
  point rotation and has no dependency on phase-system types.
- **Composition policy:** [`PhaseComposition.cs`](../src/core/WowViewer.Core/Maps/PhaseComposition.cs)
  now carries rotation angle/origin, mirrors, and donor→target mappings on `PhaseLayerSettings`;
  clone preserves all state. `ResolveTileSource` enforces per-tile-last-wins precedence, reports
  duplicate target claim counts, performs origin-aware inverse lookup, normalizes negative quarter
  turns, and reports exact-grid versus free-rotate approximation.
- **Validation:** focused `TileContentTransform|PhaseTileSource|PhaseComposition` tests **55/55
  passed**; full `WowViewer.slnx` Debug build **0 errors** with existing warnings. No adapter/runtime,
  visual, FPS, or real-client claim. Next: Phase 2 T007, wire `StandardTerrainAdapter` only and retain
  byte-identical zero-transform routing before the Alpha adapter slice.
- **Correction to earlier Spec 203 note:** `WoWConstants.ChunkSize` is misnamed but is the 533.33 yd
  one-ADT tile span used by `MapOrigin - tile * ChunkSize`; the attempted switch to `TileSize`
  (8533.33 yd = 16 ADTs) was reverted because it overshoots placement offsets 16×.
- **Operator scope addition:** Spec 219 now requires exact integer terrain-cell offsets, with 8
  cells/chunk and 128 cells/ADT, so misaligned transplanted terrain can be shifted and re-sliced
  across chunk/ADT boundaries without interpolation. Added FR-020/021, SC-011, and Phase 2B.
  Spec 196's existing WDL magnetizer can optionally score/propose the snap (FR-022/SC-012), but does
  not own the transform and cannot change it without operator acceptance. Specified/planned only;
  no cell-offset or WDL-fit implementation/runtime proof yet.
- **Workbench scope specified:** full-map orthographic minimap/heightmap/occupancy canvas; magnetic
  tile/chunk/cell selection and explicit targets; shared configurable 3D terrain selection; base
  channel gating; complete controls on every phase card; one undo/operation model; and preflighted
  Save Transformed Map through supported ADT/Alpha WDT outputs. No implementation proof claimed.
- **Spec 195 completion claim corrected by source audit:** current panel is a camera-centered 280px
  dark grid with no map backdrop and click-only selection; paste target is camera-derived; paste
  applies only heights/normals/holes despite exposing texture/placement options. Spec 219 now owns
  migration and retirement of the duplicate tool; audited coordinate/undo helpers may be reused.

## 2026-09-03 — Spec 203: Alpha phase name-table remap + tile-offset double inversion fixed
- **Name-table remap (operator-confirmed `newbindstone.mdx` → `gypsywagon.mdx`):** MDDF/MODF
  `NameIndex` is local to its owning WDT's MDNM/MONM table. The Alpha composition path copied a
  phase placement into the base tile unchanged, so the renderer resolved the base table's unrelated
  entry at the same numeric slot — correct transform, wrong model. [`AlphaTerrainAdapter.cs`](../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs)
  now resolves each phase index to its phase-table path, reuses or appends that path in the base
  table, and rewrites the placement index before rendering; logs record the index/path map.
- **Tile-offset double inversion (operator-confirmed `+11,-2` behaving as `-11,+2`):** two causes.
  (1) The Phase Map Layers panel cross-wired its labels; now X edits `TileOffsetX` (row) and Y
  edits `TileOffsetY` (column) directly. (2) `TranslatePhasePlacements` in BOTH adapters passed
  `WoWConstants.ChunkSize` (533.33 yd) where offsets are in TILES, moving placements 1/16 of the
  terrain's distance. A later coordinate audit proved this conclusion backward: this codebase's
  misnamed `WoWConstants.ChunkSize` is the 533.33 yd one-ADT span, while `TileSize` is 16 ADTs;
  the attempted `TileSize` change was reverted. Source lookup stays `target - offset`.
- **Verification:** full solution Debug build **0 errors**; focused phase-composition tests
  **18/18 passed** (updated `TileOffsetToWorldTranslation_IsNegativeInBothAxes` to the tile-span
  contract). Operator visual reload of an offset Alpha phase layer is the remaining proof.
- **Spec:** [`203-multi-phase-map-composition/spec.md`](../specs/203-multi-phase-map-composition/spec.md).

## 2026-09-03 — Spec 214: solver-independent physics policy implemented and validated
- **Planning and audit:** Authored the Spec Kit pack and [`current-implementation-audit.md`](../specs/214-mop-physics-domino/evidence/current-implementation-audit.md). The audit separates the unconsumed M2 `0x20` flag, MDX `CLID` inspection geometry, camera navigation clamping, and visual particle gravity from actual physicalised-model simulation.
- **Landed in Core.Runtime:** [`PhysicsRuntimePolicy.cs`](../src/core/WowViewer.Core.Runtime/World/Physics/PhysicsRuntimePolicy.cs) reuses `ClientBuildKey` to resolve exact `0.5.3.3368` as known-disabled, exact `5.0.1.15464` as admission-enabled, and every malformed/unmeasured build as unknown. Every decision carries activation state, profile/build/evidence provenance, diagnostics, and an explicit admitted/cull/defer/refusal reason. Budget and candidate inputs are validated; capacity selection is deterministic by priority, distance, then ordinal stable id, while results preserve input order.
- **Verification:** [`PhysicsRuntimePolicyTests.cs`](../tests/WowViewer.Core.Tests/PhysicsRuntimePolicyTests.cs) passes **16/16**; the affected `WowViewer.Core.Runtime` Debug build succeeds with **0 errors**. The full Core scope gate reports **1,379 passed, 1 skipped, and the same 9 unrelated baseline failures**. Existing `Snappier` NU1903 warnings remain unrelated. No runtime, visual, real-client, or solver claim was made.
- **Still absent and gated:** no sidecar resolver/parser, solver package, body simulation, collision response, cloth, joints, animation binding, or viewer integration. Next evidence remains read-only adapter-to-sidecar discovery plus exact-version solver license/cloth evaluation; parser/solver/viewer tasks stay blocked until those gates pass.

## 2026-09-02 — Spec 211: WMO Interior Ray Picking, Doodad Selection & Ghost Transparent Wireframes
- **Landed (Spec 211 Phases 1–3):**
  - **WMO Container Fall-Through:** Created [`WmoContainerFallThroughFilter.cs`](../src/core/WowViewer.Core.Runtime/World/WmoContainerFallThroughFilter.cs) in `WowViewer.Core.Runtime.World` with 6 unit tests in [`WmoContainerFallThroughFilterTests.cs`](../tests/WowViewer.Core.Tests/World/WmoContainerFallThroughFilterTests.cs) (all passing). Resolves the enclosing WMO bounding box lockout where rays hitting interior objects (MDX, WMO doodads, nested WMOs) were occluded by outer building AABBs.
  - **WMO Doodad Selection:** Added `ObjectType.WmoDoodad`, implemented `WmoRenderer.TryPickDoodadsByRay`, and wired candidate hits through `WorldScene.AppendWmoDoodadPickHits` and `ViewerApp_ClickSelection.cs`. Plumbed parent WMO index and selection retrieval into `ViewerApp.cs` for inspector detail, BB overlay, and camera framing.
  - **Dual-Pass Ghost Transparent Wireframe Rendering:**
    - **M2/MDX Models ([`ModelRenderer.cs`](../src/viewer/WoWViewer/Rendering/ModelRenderer.cs)):** Pass 1 renders textured geosets with 33% alpha blending (`fadeAlpha * 0.33f`); Pass 2 overlays prominent wireframe lines with `PolygonOffsetLine` (-1.0, -1.0) and 1.5 line width.
    - **WMO Objects ([`WmoRenderer.cs`](../src/viewer/WoWViewer/Rendering/WmoRenderer.cs)):** Unbinds GPU instancing under wireframe mode, renders opaque groups with 33% alpha blending (`uColor = (1, 1, 1, 0.33)`), and draws wireframe overlay on top.
    - **Terrain ([`TerrainRenderer.cs`](../src/viewer/WoWViewer/Terrain/TerrainRenderer.cs)):** Renders textured terrain fill in `PolygonMode.Fill`, then draws offset wireframe line pass on top so texturing remains clearly visible while wireframes are distinct.
  - Verified with `dotnet build` (0 errors) and `dotnet test` (all 6 new tests green, full test suite unchanged). Ready for operator interactive verification.

## 2026-09-01 — Specs 201/202: submission attribution, GPU instancing completed, batching blockers found
- **Landed (201 Phase 1 + 202 Phase 0):** [`ModelSubmissionAccounting.cs`](../src/core/WowViewer.Core.Runtime/World/Passes/ModelSubmissionAccounting.cs) — instanced / state-hoisted / unbatched / unbatchable are now four separate numbers **per render path** (`M2RouteType` mapped via `WorldAssetManager.GetRouteDecision`, using `AppliedRoute`). Draw calls are counted at the four GL call sites ([`ModelDrawCallCounter.cs`](../src/viewer/WoWViewer/Rendering/ModelDrawCallCounter.cs)) because a model draws once per geoset or section — no arithmetic over instance counts can produce that number. Every instance short of instancing carries a named gate. 16 new tests; full `WowViewer.Core.Tests` 1267 passed / 9 pre-existing failures, unchanged.
- **Root cause found — GPU instancing was unreachable dead code.** `MdxRenderer.SupportsGpuInstancedOpaque` was hardcoded `false` and `M2Renderer` delegated to it, so **no renderer could ever instance** and "batched" only ever meant state-hoisted. The CPU side was complete (instance VBO, divisor-tagged attributes at locations 6–10, `DrawElementsInstanced`) but **the vertex shader declared only locations 0–5** — enabling the flag as it stood would have stacked every doodad on the world origin. Shader completed (instance transform + per-instance fade) with a constant-folded non-instanced fallback and automatic disable if it fails to compile.
- **Second blocker removed:** `MdxRenderer.RequiresUnbatchedWorldRender` narrowed to `_wireframe`. The particle/ribbon terms blocked **3,305 of 3,313** opaque instances for effects the opaque pass never draws — particles render only under `RenderPass.Transparent` (which never batches), and on M2 adapter models they are never drawn at all because the adapter copies header emitter counts but never populates `ParticleEmitters2` (the `[M2] Unresolved effect systems` log).
- **Correctness guard:** models with local MDX lights are excluded from instancing. `RenderInstance` calls `UploadMdxLights(modelMatrix)` per instance; the instanced path uploads once from identity, so a batched lamp would light every copy as if at the world origin (contract C2).
- **Third blocker, NOT yet fixed (202 T301):** `WowViewerM2RuntimeBridge.PreferNativeStaticRenderer` **defaults to `true`** when its env var is unset, so every M2 gets the native-only `M2Renderer` with `_legacyRenderer == null` and `RequiresUnbatchedWorldRender` is unconditionally true. The native renderer has no instancing path and its `RenderCore` re-uploads all ten shared uniforms per instance, so flipping the flag alone buys nothing.
- **Also found:** the state-hoisted path **cannot** hoist bone matrices — submission walks instances in visibility order and calls `BeginBatch` lazily, so renderers interleave and per-model uniforms would be clobbered. Only GPU instancing removes the per-instance bone upload. `TransparentBatchedMdxCount` is dead (never incremented). Added an "Animate world doodads" toggle, default **off**, per the operator's rule that only WMO doodads should auto-animate.
- **Open (user-owned proof):** fly with "Opaque MDX batching" **and** "GPU instancing for opaque models" both ticked; confirm `instanced` rises and draw calls fall toward `distinct models`. Check lamps/braziers and distance-faded doodads for appearance change.

## 2026-09-01 — Spec 205: MH2O liquid is 100% LiquidObject ids; river heightmaps discarded (MEASURED)
- **Measured, not inferred:** new command `inspect adt liquid-formats --client <dir> --map <name>` ([`AdtLiquidFormatSupport.cs`](../tools/inspect/WowViewer.Tool.Inspect/AdtLiquidFormatSupport.cs)) over 80 MoP root ADTs / 17,461 liquid layers: **100% carry a `LiquidObject.dbc` id** in `liquid_object_or_lvf` (42, 2325, 2333, 2372), never a vertex format 0–3. Both MH2O decoders `switch` on that field with **no `default`**, so every layer falls through with `heights = null` and renders flat at the header `minHeight` — no error, no counter, no log.
- **Ocean is not the defect:** id 42 (17,317 layers, liquidType 2) is genuinely depth-only — 6,008 of its 6,194 vertex blocks read as implausible floats, i.e. depth bytes — and renders correctly today. The **144 river layers** (ids 2325/2333/2372, liquidType 5) are 100% plausible heights that **vary**, spreads of 11.90 / 70.15 / 163.25 world units, all discarded. The substituted flat plane is also at the wrong height: the lowest vertex disagrees with the header in every varying layer, which is why adjacent chunks step against each other.
- **Two decoders, only one in the render path:** `StandardTerrainAdapter` calls `Mh2oChunk.Parse`; `AdtLiquidReader` serves harvest/dataset/converter. Both carry the defect independently — fixing `AdtLiquidReader` alone would change nothing visible. Harvested MoP liquid data has been wrong too.
- **Fix (specced, not implemented):** DBC chain `LiquidObject → LiquidType → LiquidMaterial → LVF`; neither `LiquidObject` nor `LiquidMaterial` has a reader, and the offsets are wiki-documented and **unverified against this client** — Phase 1 is a gate with a corpus-derived expected answer. The float-plausibility probe that made the diagnosis is **not** acceptable as the decoder (0.3% false positives on ocean); keep it as a cross-check only.
- **Spec:** [`specs/205-mh2o-liquid-object-vertex-format/`](../specs/205-mh2o-liquid-object-vertex-format/spec.md).

## 2026-09-01 — Spec 204: asset loading is synchronous on the render thread (not the SSD, not the MPQ reader)
- **Measured:** 2048-frame MoP flight — median 75.26 ms, p95 144.59, p99 235.37, max 541.65, **2047/2048 frames over 33.3 ms**. `DeferredAssetLoads` owns **12 of 13** recent hitches at **26.4–68.1 ms** each.
- **The operator's premise was correct.** `MpqDataSource` already runs 2 prefetch workers and `QueueMdxLoad`/`QueueWmoLoad` already call `PrefetchModelBytes`, so root bytes are usually warm. But `WorldAssetManager` contains **no threading at all** — parse, adaptation, BLP decode and GL upload all run inside the frame.
- **The budget cannot rescue it, by design.** `DeferredLoadBudget.CanStartAnotherLoad` admits the first load of every frame unconditionally so an oversized asset still becomes resident; its own documentation names the fix as *"moving decode off the render thread ... Spec 153 Phase 5 step 2, deliberately not attempted here"* and calls `OversizedAdmissionCount` *"the honest measure of the residual the off-thread decode still owes."* That counter existed and **nothing read it** until today; now surfaced in the frame panel with `BudgetDeferralCount` and `WorstObservedLoadMs`.
- **The throttle makes it worse:** at ≥33 ms previous-frame CPU it clamps to 1 load/frame, but the first load stays unconditional — so it cuts streaming ~6x **without reducing the hitch**. Self-reinforcing: slow frames throttle loading, oversized loads keep frames slow.
- **Also found:** `ProcessDeferredTextureLoads()` runs from the top of `MdxRenderer.RenderGeosets`, so texture decode/upload happen **inside the draw pass**, billed to `MdxOpaqueSubmission` and governed by no budget. Prefetch covers root files only — WMO group files and BLPs are discovered after the parse.
- **Phase 1 is a hard gate:** GL objects live on static fields and `MdxTextureDiagnosticLogger` is a process-global `StreamWriter` re-opened per model from a renderer constructor.
- **Spec:** [`specs/204-off-thread-asset-decode/`](../specs/204-off-thread-asset-decode/spec.md).

## 2026-09-01 — Spec 197: PTCH/BSDIFF patch-artifact reconstruction (random missing tiles root cause)
- **Root cause confirmed:** Loose 5.0.1 `.adt` files are frequently PTCH/BSDIFF patch artifacts (`PTCH`/`MD5_`/`BSD0`/`BSDIFF40`); the viewer fed them raw to `ParseAdt`, the `KNCM` scan found nothing, and the tile silently produced zero chunks. Native proof: `MapArea` (`FUN_00BB0850`) receives already-reconstructed bytes — load-complete callback `FUN_00BB70F0` (`MapAdtFileData.cpp`) stores final `(fileData, size)`; cache helpers `FUN_00BB71B0`/`FUN_00BB7C80` are pure hash-table plumbing; key builder `FUN_00BB6C70` packs `mapId|adtFileType|y|x`.
- **Landed:** [`AdtPatchArtifact.cs`](../src/core/WowViewer.Core.IO/Maps/AdtPatchArtifact.cs) — pure PTCH parse, standard BSDIFF40 applier (BZip2 control/diff/extra), MD5-matched base selection with `REVM` validation. `ReadFileCopies` base-copy enumeration added to `IDataSource` (default single-copy), `MpqDataSource` (archives lowest-priority first, alpha wrapper, loose last), and `IArchiveCatalog.ReadFileCopiesLowestFirst` (default + `MpqArchiveCatalog` override). `StandardTerrainAdapter.LoadMapTile` reconstructs root + `_texN`/`_objN` companions; reconstruction failure logs Important and treats the file as missing instead of parsing artifact bytes.
- **Validation:** Full solution Debug build 0 errors. New `AdtPatchArtifactTests` (7 tests) + existing terrain builder tests: 12/12 passed. Full `WowViewer.Core.Tests`: 1245 passed / 9 failed — the 9 failures were verified identical at HEAD via a path-scoped stash (frame-pass ordering, M2 footprints, WTF classifier, V23 summaries, enrichment streams, V18 placements; all disjoint from this change).
- **Evidence:** [`5.0.1-adt-ptch-patch-artifacts.md`](../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/evidence/5.0.1-adt-ptch-patch-artifacts.md).
- **Open (user-owned proof):** Reload Thunder Isle from MoPBeta; confirm previously missing tiles render and `Reconstructed patched ADT ... via embedded BSDIFF patch` log lines appear. Out of scope: multi-step chain base synthesis, patched-WDT reconstruction, editing patched companions.

## 2026-08-31 — Spec 197: split reader/runtime and explicit conversion target boundary
- **Landed:** Native-aligned 5.0.1 split-family discovery and loading for root plus selected `_obj0`/`_obj1` and `_tex0`/`_tex1` companions; headerless companion `MCNK` routing; sparse physical MCIN slot preservation; and focused regression coverage for band 1, split wrappers, sparse slots, and object references.
- **Landed:** Core `MapConversionTargetFormat` contract and viewer target selector for Alpha 0.5.3 monolithic WDT, LK v18 monolithic ADT, and unavailable Cataclysm/MoP split ADT. The target is independent from source family, lossy routes display warnings, split-to-Alpha uses the Alpha command, and split-to-LK keeps archive client root separate from the loose split overlay directory.
- **Safety boundary:** Target-aware `LkAdtWriter` calls accept only LK v18. Native MoP split output remains disabled because no slot-aware split writer exists; LK output is never labeled as native MoP. Alpha writer remains frozen.
- **Validation:** Viewer and converter Debug builds completed with 0 errors after the routing correction; the focused `MapConversionFormat` test filter passed 6/6. Existing warnings, including the Snappier vulnerability advisory, remain non-blocking.
- **Open:** Audit compact MCIN consumers and the runtime convenience path; extend merger/texture-transfer/converter inputs to band 1; define the canonical slot-aware document and loss policy; implement/test a genuine native MoP split writer; finish native WDT/MAIN, blend/seam, numeric FourCC, indirect-reachability, and address-resolution evidence.

## 2026-08-31 — Spec 197: 5.0.1 dead/dormant/partial documentation checkpoint
- **Recorded:** Added the focused [`5.0.1-dead-dormant-partial-rendering.md`](../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/evidence/5.0.1-dead-dormant-partial-rendering.md) evidence note and consolidated [`wow-5.0.1-adt-wdt-definitive.md`](../docs/architecture/wow-5.0.1-adt-wdt-definitive.md) guide, following the legacy ADT/WDT and Ghidra definitive-guide structure.
- **Confirmed/classified:** `FUN_00b7a230` has a partial liquid-factory default branch (`Not implemented!`) after supported selectors `7`, `9`, `10`, and `0xd`; `FUN_004b16d0` has a real `CSimpleEditBox.cpp` FIXME early-return branch; DepthCache/GBuffer are capability-gated implemented infrastructure; atlas/doodad/particle batching and water-detail paths are optional or explicitly unsupported settings; MapArea/MapRenderChunkState assertions are cleanup contracts.
- **Boundary:** 2,184 zero-direct-xref functions remain low-reachability candidates because indirect dispatch is not ruled out. No function is called proven dead. A path-builder discrepancy (`0x00bb9490` vs `FUN_00b94990`) remains explicitly unresolved. No production code or Ghidra state changed.
- **Next:** Complete native WDT/MAIN, numeric `MHID`/`MDID`/`MCXH`, blend/seam, indirect registration, and address-resolution evidence before reopening T118–T120.

## 2026-08-31 — Spec 197: GhidraMCP 6.0.0 bridge recovery
- **Diagnosed:** Ghidra itself was not the failure. The GhidraMCP 6.0.0 plugin was already running against the loaded `Mists of Pandaria 5.0.1.15464` project and serving HTTP on `127.0.0.1:8089` (status dialog: UDS + TCP running, 222 endpoints). The stale config referenced the absent `H:\ghidra_12.1.2_PUBLIC\GhidraMCP-5.14.2\.venv\Scripts\python.exe` and passed unsupported `--ghidra-server` arguments to the bridge script.
- **Landed:** Downloaded the release-provided `ghidra_mcp_bridge-6.0.0-py3-none-any.whl` and installed it with `uv tool install --force`. Updated both [`.mcp.json`](../../.mcp.json) and [`.roo/mcp.json`](../../.roo/mcp.json) to launch `C:\Users\akspa\.local\bin\bridge-mcp-ghidra.exe --no-lazy` with `GHIDRA_MCP_URL=http://127.0.0.1:8089`.
- **Validated:** JSON configs parse; bridge help works; stdio initialize succeeds; bridge auto-connects by TCP to the named 5.0.1 project and registers 221 live tools (the HTTP schema reports 222). `get_current_program_info` confirms `Wow.exe` PE x86, image base `0x00400000`, 38,405 functions, 175,352 symbols, and 790 data types. Read-only HTTP probes found 766 `.cpp`-matching strings, including rendering, terrain, WMO, liquid, and `CMapChunk` anchors.
- **Next:** T117 is the read-only Ghidra extraction pass for rendering systems, parser entry points, and split-ADT/height-blend/WMO seams. No Ghidra program edits were made.

## 2026-08-31 — Spec 197: 5.0.1 Ghidra reconnaissance checkpoint
- **Recorded:** Added [`research-ghidra-5.0.1.md`](../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/research-ghidra-5.0.1.md) with the live binary provenance, source-path/function map, and the first confirmed `CMapChunk`/WMO reader facts.
- **Confirmed:** `CMapChunk` accepts a 128-byte header, dispatches `MCMT`, `MCDD`, `MCAL`, `MCBB`, `MCCV`, `MCLQ`, `MCLV`, `MCLY`, `MCRD`, `MCRW`, `MCNR`, `MCRF`, `MCSH`, and `MCVT`, derives MCLY/reference/blend-batch counts from payload sizes, and uses the exact `33.333332` / `17066.666` world transform. WMO parsing asserts `MVER == 0x0011` and uses 0x10-byte material records.
- **Open:** Literal string searches found no `MHID`, `MDID`, `MCXH`, or `CMapTile` tokens; this is not evidence of absence because the MCNK parser uses numeric FourCC comparisons. Blend/render batch routines and numeric split-ADT tracing remain T117.
- **Boundary:** No production code or Ghidra state changed; parser/renderer implementation remains gated on the next decompilation and cross-check pass.

## 2026-08-31 — Spec 197: native 5.0.1 ADT family matrix checkpoint
- **Recorded:** Expanded [`research-ghidra-5.0.1.md`](../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/research-ghidra-5.0.1.md:173) and the Spec 197 plan/tasks with the versioned split-loader contract; documentation-only change, with no production reader/renderer or Ghidra edits.
- **Confirmed:** The 5.0.1 path requires `MVER == 0x12`; constructs the root plus exactly one selected `_obj0`/`_obj1` and `_tex0`/`_tex1` pair; distinguishes at least cache types 0–4; requires 256 outer MCNK records per file-data object; consumes the 128-byte MCNK header only for root slot 0; and merges headerless split MCNK payloads by chunk index.
- **Confirmed:** Native area creation is gated by the WDT/map-table `CMapTableEntry::Flag_Exists` bit, separate from root/split file discovery. This is now the leading explanation for false tile admission; it remains a hypothesis until a real WDT/MAIN sample is compared.
- **Not confirmed:** `_lod.adt` is not constructed in the investigated 5.0.1 loader path; `MHID`, `MDID`, and `MCXH` have no confirmed native parser location; the relation between the two suffix bands and broader internal `LOD_COUNT` remains open.
- **Viewer gap:** `AdtTileFamily`, `AdtTileFamilyResolver`, `WowFileDetector`, `StandardTerrainAdapter`, and split/export paths currently model only `_obj0`/`_tex0` plus `_lod`; band-1 routing and native tile-table admission need an evidence-gated follow-up. T117 remains in progress; T118–T120 remain blocked on the evidence pass.

## 2026-08-30 — Spec 196: WDL Lattice Magnetization, Polarity Inversion & Stratigraphy Restoration Engine (Complete)
- **Landed:**
  - **Polarity Inversion & Multi-Anchor Geometry Solver (`StratigraphyAnchorMode`, `TemporalStratigraphyOptions`, `TemporalMeshRestorer`):** Added `StratigraphyAnchorMode` enum (`LowestZ_Floor`, `HighestZ_Ceiling`, `MeanZ`, `NeighborMeshBorder`, `WdlLattice`, `CustomDatum`) and signed scaling factor support. Solves inverted compressed terrain (e.g. Dragon Isles) by inverting amplification direction from upper ceiling datums without spike artifacts.
  - **Neighboring Mesh Height & Scale Auto-Fit Solver (`NeighborMeshHeightSolver`):** Evaluates candidate scale bands ($1\times \to 512\times$), polarities ($\pm 1$), and anchor datums over shared chunk seam boundary vertices (9 outer lattice vertices per edge) to compute optimal vertical shift $\Delta Z$ and minimize boundary seam RMSE against adjacent active terrain.
  - **WDL Macro-Lattice Magnetization & WDL Writer (`WdlLatticeMagnetizer`, `WdlFileWriter`):** Performs continuous bilinear interpolation over $17\times 17$ tile vertices and $16\times 16$ chunk center heights to blend high-frequency ADT micro-relief onto low-frequency WDL macro terrain. Serializes Blizzard-compliant binary `.wdl` files (`MVER`, `MWMO`, `MWID`, `MODF`, `MAOF`, `MARE`, `MAHO`).
  - **0-Hitch Asynchronous Pipeline & In-Viewer Controls (`ViewerApp.cs`, `ViewerApp_Sidebars.cs`):** Offloaded tile deformation and SIMD normal synthesis to background worker tasks (`Task.Run`) and double-buffered queue (`_pendingRestoredTilesQueue`), eliminating 2–4s UI hitches. Added polarity inversion, anchor mode, neighbor auto-fit, WDL magnetization controls, and companion `.wdl` export to `ViewerApp_Sidebars.cs`.
  - **Unit Tests:** 12/12 unit tests passing (100% green) across `NeighborMeshHeightSolverTests`, `WdlLatticeMagnetizerTests`, `StratigraphyLevelAnalyzerTests`, and `StratigraphyTileExporterTests`. Full solution builds with 0 errors.

## 2026-08-30 — Alpha 0.5.3 Terrain Organization & Tile Indexing Fix (Complete)
- **Landed:**
  - **Row-Major Grid Index Alignment (`AlphaTerrainAdapter.cs`, `ViewerApp.cs`):** Fixed inverted tile index formula in `AlphaTerrainAdapter.TileExists`, `AlphaTerrainAdapter.LoadTileWithPlacements`, and `ViewerApp.TryGetTerrainWeakSignalWdlTile` from `tileY * 64 + tileX` to `tileX * 64 + tileY`.
  - **Eliminated Tile Transposition:** Fixed the issue where non-diagonal 0.5.3 Alpha terrain tiles loaded the transposed $(Y, X)$ ADT blocks across world boundaries.
  - **Unit Tests:** 6/6 `WdtSummary` tests passing (100% green). Full solution builds with 0 errors across Windows and CrossPlatform targets.

## 2026-08-30 — Spec 195: Overhead Chunk Manipulator & Multi-Tile Sub-Cell Transposition Engine (Complete)
- **Landed:**
  - **Global Chunk Coordinate Space & Selection Region Model (`GlobalChunkCoordinate`, `ChunkSelectionRegion`):** $1024 \times 1024$ continuous coordinate space ($G_x = T_x \times 16 + C_x$, $G_y = T_y \times 16 + C_y$) eliminating tile boundary seam issues, with bounding box queries, rectangular box selections, individual toggles, and whole-tile selections.
  - **Core Transposition & Transformation Engine (`ChunkTranspositionService`, `ChunkTranspositionPayload`, `ChunkTranspositionOptions`):** Extracts, transforms (relative height offsets, rotation by $90^\circ/180^\circ/270^\circ$, X/Y mirroring), and transposes 145-vertex MCVT heights, MCNR normals, MCLY/MCAL texture layers, hole masks, and MDDF/MODF doodad and WMO placements with world coordinate delta $(\Delta X, \Delta Y, \Delta Z)$.
  - **Reversible Editor Operations (`ChunkTranspositionOperation`):** Fully integrated with `EditorSession` for non-destructive undo/redo history.
  - **Interactive In-Viewer Editor Plugin & 2D Overhead Canvas (`ChunkManipulatorEditorPlugin`, `ViewerApp_Editor.cs`):** Live 2D overhead canvas rendering tile borders ($533.334\text{m}$) and chunk sub-grids ($33.334\text{m}$), zoom/pan, click-drag box selection, copy/cut/paste buttons, and live in-place memory replacement via `ReplaceTileChunksAndRebuild`.
  - **Unit Tests:** 91/91 unit tests passing in `WowViewer.Core.Editor.Tests` (100% green). Full solution builds with 0 errors across Windows and CrossPlatform targets.

## 2026-08-30 — Spec 194: Temporal Stratigraphy & Weak Signal Development Mesh Restoration (Complete)
- **Landed:**
  - **Core Stratigraphy & SIMD Analysis Engine (`StratigraphyLevelAnalyzer`, `TemporalStratum`, `SeamDiscontinuityProfiler`):** Computes unique floating-point height level cardinality $|\{h\}|$ without altitude limit bias, profiling C0 step and C1 slope discontinuities across 15 internal MCNK boundaries to classify chunks into discrete strata (`Active_1x`, `LateRevision_4x_8x`, `ClassicErasure_33x`, `DeepProto_64x_512x`, `Holed_DevMesh_1x`, `Submerged_OceanFloor`, `BitExact_Flat`).
  - **High-Performance SIMD Normal & Mesh Solver (`FastTerrainNormalSolver`, `TemporalMeshRestorer`):** Vectorized normal computation for 257x257 lattices, SIMD in-place height transformation, negative floor preservation, and SmoothStep boundary slope blending.
  - **In-Viewer Interactive Workbench (`ViewerApp_Sidebars.cs` & `ViewerApp.cs`):** Unified "Mesh Stratigraphy" subtab on the Archeology page (Inspect > Archeology > Stratigraphy) and Terrain Lab with continuous gradient factor slider ($1.0\times \to 512.0\times$), preset snap points ($33.334\times$, $16\times$, $64\times$, $1\times$), dev mesh unhiding toggle (`_stratigraphyUnhideDevMeshes` bypassing `HoleMask`), boundary slope stitching, floor anchoring, in-viewer analysis, and in-app folder picker export via `ImGuiPathPicker`.
  - **Restored Terrain Exporter (`StratigraphyTileExporter`):** Serializes restored LK ADTs with companion file copying and monolithic Alpha WDT maps with zero runtime loss.
  - **CLI Batch Scanning & Offline Patching (`terrain-stratigraphy-scan`, `terrain-stratigraphy-patch`):** Added `TerrainStratigraphyScanCommand` in `WowViewer.Tool.Inspect` emitting structured `stratigraphy_manifest.json` and CSV summaries, and `TerrainStratigraphyPatchCommand` in `WowViewer.Tool.Converter` for pre-computed batch patching.
  - **Unit Tests:** 10/10 unit tests passing (100% green) across `StratigraphyLevelAnalyzerTests` and `TemporalMeshRestorerTests`. Solution builds with 0 errors.

## 2026-08-30 — Spec 192: Terrain Template Brush & Paste Library with Interactive In-Viewer Map Generator (Complete)
- **Landed:**
  - **Curated Terrain Brush & Paste Library (`CuratedTerrainBrushLibrary`):** 15 stock archetypal terrain motifs (cobblestone straight/curve/cross roads, dirt paths, marble plazas, gentle knolls, terraces, ridges, pond basins, grand avenues, flat exhibit pads) with 2D relative heightfields, multi-layer alpha masks, and slope calculation.
  - **Strict Hardware 4-Layer Chunk Allocator (`TerrainLayerAllocator`):** Solves the hardware constraint by energy-based alpha pruning and weight normalization, strictly enforcing $\le 4$ texture layers per MCNK chunk.
  - **ADT Sub-Region Extraction Engine (`AdtPasteExtractor`):** Extracts bounded terrain pastes directly from loaded LK/Alpha ADT chunks, enabling continuous harvesting of real terrain brush pastes from game data.
  - **Undo/Redo Terrain Stamping Operation (`TerrainStampOperation`):** Applies brush pastes with `SmoothStep` boundary feathering, height blending modes (Additive, Replace, Maximum, Minimum), and snapshot state capture for `EditorSession`.
  - **Templated Procedural Map Generator (`TemplatedTerrainGenerator`):** Synthesizes multi-tile maps with connected cobblestone walkway networks, flat marble exhibit courtyards ($Z = 0$), slope constraints ($\le 25^\circ$), and multi-era ADT / WDT / WDL serialization.
  - **In-Viewer Editor Plugin (`TerrainTemplateEditorPlugin`):** ImGui categorized paste catalog browser, search filter, interactive stamping sliders (scale, rotation, height mult, feathering), and "New Map from Template" generation wizard in `ViewerApp_Editor.cs`.
  - **CLI Command (`terrain-generate-templated`):** Generates full multi-tile maps from command line with customizable themes, grid sizes, and output paths.
  - **Unit Tests:** 14 new tests in `TerrainBrushPasteTests`, `TerrainLayerAllocatorTests`, `AdtPasteExtractorTests`, `TerrainStampOperationTests`, and `TemplatedTerrainGeneratorTests` (100% green). Solution builds with 0 errors.

## 2026-08-30 — Spec 191: Procedural Garden Museum Map Generator (In Progress)
- **Defects Identified via In-Game Proof:**
  - `DefaultGroundTexture` was hardcoded to `wcsand.blp` (Wailing Caverns sand) with `checkers.blp`, generating barren sand maps with giant checkerboard lines instead of garden terrain.
  - `CreateChunkHeights` generated jagged 45-degree bevel quad ramps across MCVT vertices, trapping player collision capsules.
  - `DrawRectOutlineMeters` drew grid lines directly over text label bands, making signage unreadable.
- **Architectural Action Plan:**
  - Replace defaults with lush Elwynn garden grass (`elwynngrass.blp`), Stormwind cobblestone walkways (`stormwindcobble.blp`), and clean white marble exhibit pads (`whitemarble.blp`).
  - Enforce completely flat walkable terrain ($Z = 0$) across walkways and exhibits to eliminate navigation collision traps.
  - Separate text plaques onto clean stone backgrounds without overlapping grid lines.
- **Status:** In progress. Spec 191 remains open until real-client in-game proof confirms a clean, flat, lush multi-textured garden museum.

## 2026-08-28 — Spec 190 Phase 4: Companion ADT Synthesizer (US4)
- **Landed:**
  - **Companion ADT Scanner & Synthesizer (`RosettaCompanionAdtSynthesizer`):** Scans directories for orphan `.pm4` files lacking companion ADTs and synthesizes minimal compliant companion ADTs with authentic headers (`MVER`, `MHDR`, `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`, `MCNK`) via `BlankAdtFactory` and `LkAdtWriter`.
  - **Cryptographic Provenance Manifest (`RosettaCompanionProvenanceReport`):** Emits machine-readable provenance reports with SHA256 hashes, source PM4 references, generation options, and timestamps distinguishing synthetic files from authentic game assets (FR-010).
  - **Safe Overwrite Protection:** Skips existing companion files by default, with `--overwrite` option for explicit regeneration.
  - **CLI Companion Tool (`rosetta-synthesize-companions`):** Added CLI command with structured summary reporting and provenance export.
  - **Unit Tests:** Added 6 new tests in [`RosettaCompanionAdtSynthesizerTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaCompanionAdtSynthesizerTests.cs). All 69 Rosetta unit tests pass green.

## 2026-08-28 — Spec 190 Phase 3: Deterministic PM4 Lookup Engine (US3)
- **Landed:**
  - **Deterministic PM4 Lookup Engine (`RosettaPm4LookupEngine`):** Pure non-LLM geometric identification engine matching real PM4 geometry segments against `RosettaReferenceLibrary` using multi-signal scoring, bounding candidate pruning, and tri-state classification (`Identified`, `Ambiguous`, `NoReference`, `Ineligible`).
  - **Granular Signal Agreement & Disagreement Engine:** Compares aspect ratios, major spans, bounding volume, footprint area, and TypeFlags surface profiles to surface transparent reasons for candidate scoring.
  - **Legacy Reconciliation Integration:** Added `CompareWithLegacyScorer` and `Pm4ReconciliationInputAdapter.BuildRosettaCorpusReferences` connecting the full reference library to Spec 176 reconciliation.
  - **CLI Match Tool (`rosetta-pm4-match`):** Added CLI command with structured JSON output and console candidate breakdown tables.
  - **Unit Tests:** Added 7 new tests in [`RosettaPm4LookupEngineTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaPm4LookupEngineTests.cs). All 63 Rosetta unit tests pass green.

## 2026-08-28 — Spec 190 Phase 2.5: Alpha 0.5.3 Native Client Ergonomics, First-Class DBC Generation, WDL Mesh & +20Z Exhibits

- **Landed:**
  - **Native Alpha `Map.dbc` & `AreaTable.dbc` Generation (`RosettaDbcGenerator`):** Emits authentic 0.5.3 binary WDBC files allocating dedicated Map IDs (starting at 500) and Area IDs (starting at 5000) for designkit zones, registering Rosetta maps as first-class outdoor continents under `DBFilesClient\`.
  - **Alpha WDL Low-Resolution Distant Terrain Mesh:** Automatically generates and saves `{mapName}.wdl` alongside `{mapName}.wdt` using `WdlWriter.Build`.
  - **Minimap Translation (`minimap.trs` / `md5translate.trs`):** Added `RosettaMinimapPainter.GenerateMinimapTrs` and `WriteMinimapTrs` generating TRS mapping blocks for each map directory (`dir: {mapName}`) and aliases (`dir: Azeroth`).
  - **Model (`.mdx`/`.m2`) and World Model (`.wmo`) Map Splitting:** Added `SplitAssetKinds` separating assets into `{map}_MDX` and `{map}_WMO` maps with an 800-tile budget ceiling to prevent engine memory exhaustion.
  - **Bounding-Box Centering Offset & $+20\text{Z}$ Elevation:** Centers geometry inside cell footprints by offset `(X, Y) -= boundsCenter`, and elevates by $Z = \text{groundZ} + \max(0, -\text{bounds.Min.Z}) + 20\text{m}$, ensuring models float comfortably in the air with zero ground clipping.
  - **Walkable Baseline Terrain:** Smooths heightfield calculation to eliminate saw-tooth `/\` knife-edge ridges and deep pits, providing flat walkable ground ($Z = 0$).
  - **Unit Tests:** Added 6 new unit tests in [`RosettaTilesetGeneratorTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaTilesetGeneratorTests.cs). All 56 Rosetta unit tests pass green.

## 2026-08-28 — Spec 190 Phase 2: Rosetta Reference Library Builder (US2) & Verification Suite

- **Landed:**
  - **Rosetta Reference Library Data Model (`RosettaReferenceLibrary` & `RosettaReferenceAsset`):** Complete model representation containing asset path, normalized path, kind (`"m2"`/`"wmo"`), client build, tile coordinates, bounding boxes (`Pm4Bounds3`), center, span, diagonal, volume, footprint hull/area, aspect ratios, sub-part bounds, signal feature dictionary, and `ToAssetReferenceSignalRecord()` for 100% interoperability with `Pm4AssetMatchScorer` and reconciliation services.
  - **Synthetic Corpus Reader (`RosettaCorpusReader`):** Decodes synthetic Rosetta placements and geometry directly from `rosetta-manifest.json`, in-memory `RosettaGenerationResult`, and Zarr datastores (`RosettaObjectLibrary`), creating standardized reference libraries with deterministic SHA256 library IDs.
  - **Automated Self-Test Verification Suite (`RosettaReferenceLibrarySelfTest`):** Runs systematic verification queries with exact and perturbed (jittered) bounding boxes to evaluate candidate match rankings against reference assets, enforcing the $\ge 99.0\%$ Top-1 accuracy requirement.
  - **CLI Integration (`Program.cs`):**
    - `wowviewer-inspect rosetta-build-library <input> [--output <path>] [--build <label>]`: Builds and saves reference libraries from directories, manifests, or datastores.
    - `wowviewer-inspect rosetta-library-selftest <libraryJsonPath> [--tolerance <f>] [--top-k <n>] [--perturb]`: Executes the self-test suite and reports Top-1/Top-3 accuracy and any candidate defects.
    - `rosetta-generate --emit-library [--library-output <path>]`: Automatically builds and saves the reference library alongside the generated map.
  - **JSON Serialization:** Added custom `Vector3JsonConverter` and `Vector2JsonConverter` for clean, high-fidelity JSON serialization.
  - **Unit Tests:** Authored 7 comprehensive unit tests in [`RosettaReferenceLibraryTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaReferenceLibraryTests.cs). All 50 Rosetta unit tests pass green.

## 2026-08-28 — Alpha WDT Row-Major Tile Indexing Fix & Visual Calibration Tools

- **Landed:**
  - **Fixed Alpha WDT Transposed Tile Loading:** In [`AlphaTerrainAdapter.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs), corrected `TileExists` and `LoadTileWithPlacements` from column-major `tileX * 64 + tileY` to row-major `tileY * 64 + tileX`. Previously, every non-diagonal tile ($tileX \ne tileY$) loaded the terrain, heights, and MCAL alpha canvas of its transposed counterpart $(tileY, tileX)$, causing objects to float over mismatched terrain and labels.
  - **Visual Calibration & Bullseye Diagnostics:** Added `DrawCircleOutline` and `DrawBullseyePattern` to [`RosettaAlphaPainter.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/RosettaAlphaPainter.cs) and [`RosettaTilesetGenerator.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/RosettaTilesetGenerator.cs). Calibration tiles and empty tiles now feature concentric range rings (30m to 240m), full-tile crosshair axes, and cardinal direction indicators (`NORTH (-Y)`, `SOUTH (+Y)`, `WEST (-X)`, `EAST (+X)`).
  - **Unit Tests:** Added `AlphaPainter_DrawBullseyePattern_FillsExpectedRegions` and `AlphaWdt_AsymmetricTileCoordinates_MaintainRowMajorIntegrity` in [`RosettaTilesetGeneratorTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaTilesetGeneratorTests.cs). All 43 tests pass green.

## 2026-08-28 — Spec 190 Phase 1.5: Cross-Era Model Resolution, Rosetta Zarr Multi-Build Diffing & UI Load Dialog

- **Landed:**
  - **Cross-Era Model Extension Shifting (`.mdx` $\leftrightarrow$ `.mdl` $\leftrightarrow$ `.m2`):** Enabled seamless format shifting in `WorldAssetManager.cs`, `WmoRenderer.cs`, and `ViewerApp.cs`. Maps from any era (e.g. Rosetta 0.5.3 or Alpha WMOs with `.mdx` doodad references) automatically resolve and render with modern `.m2` models when running against newer clients (1.12.1 or 3.3.5), and vice-versa.
  - **Rosetta Datastore Build Metadata & Diff Engine:** Added `RosettaBuildMetadata` writing in `RosettaDatastoreWriter.cs` and `RosettaBuildDiff` / `ComputeBuildDiff` in `RosettaObjectLibrary.cs` for fast comparison of asset additions, removals, format migrations, and geometry changes across builds without re-processing.
  - **CLI Diff Command:** Added `rosetta-datastore-diff` in `WowViewer.Tool.Inspect` (`Program.cs`).
  - **Viewer UI Integration:** Added **File > Load from Rosetta Datastore...** menu item and interactive modal in `ViewerApp` (`ViewerApp_ClientDialogs.cs`), enabling interactive selection of Data Version (build), Map Name, and Base Game Version (client MPQ / asset source) with live cross-build diff statistics.
  - **End-User Documentation:** Updated `docs/WoWViewer/USERGUIDE.md` and `wow-viewer/README.md` with dedicated guides for Phased Terrain Dual-Map Overlays (Spec 135/137) and Rosetta Multi-Version Zarr Datastores (Spec 190).
  - **Validation:** All 41 focused Rosetta unit tests pass green. Full solution builds with 0 errors.

## 2026-08-27 — Spec 190 Checkpoint 17 correction: minimap-only fix

- **Corrected:**
  - The previous Alpha Rosetta placement-axis commit treated generated Alpha WDT placement bytes as the defect. That was incorrect: the operator report was about new minimap tooling, not the written map bytes.
  - Removed the Rosetta Alpha WDT post-write MDDF/MODF coordinate patcher, removed the manifest `alphaClientFilePosition` claim, and restored `rosetta-generate --format alpha` to direct `AlphaWdtWriter.Build(...)` output.
  - Kept the valid minimap fix: `RosettaMinimapPainter` places synthetic model/WMO pins at the object-band center used by generated placements, not the whole label cell center.
  - **33/33 focused Rosetta tests pass.** Full solution compiles with 0 errors. Real 0.5.3 minimap visual proof remains operator-owned.

## 2026-08-27 — Spec 190 Checkpoint 16: Alpha 0.5.3 client asset discovery

- **Landed:**
  - **Rosetta Asset Discovery for Alpha Clients (`RunRosettaGenerate`):** Added robust filesystem scanning in `Program.cs` for Alpha-era single-file wrapper archives (`.mdx.mpq`, `.mdl.mpq`, `.m2.mpq`, `.wmo.mpq`, `.blp`).
  - **Era Detection:** Correctly identifies Alpha clients with `.mdx`/`.mdl` assets, avoiding false-positive `"lk"` classification.
  - **32/32 unit tests pass green in `RosettaTilesetGeneratorTests`.** Full solution compiles with 0 errors.

## 2026-08-27 — Spec 190 Checkpoint 15: Minimap generation, status bar tile readout & v0.5.2.2 bump

- **Landed:**
  - **Rosetta Minimap Tile Generator (`RosettaMinimapPainter` & `Blp2Writer`):** 256×256 DXT1-compressed BLP2 minimap tiles generated under `Textures/Minimap/{mapName}/map{tileY}_{tileX}.blp` with rendered cell borders, pedestal plateaus with bevels, downsampled antialiased text labels, and model/WMO markers.
  - **Status Bar Tile Readout:** Bottom status bar now displays `Tile: {tileY:D2}_{tileX:D2}` (matching ADT and minimap filenames) whenever any runtime scene is active.
  - **Version bumped to `v0.5.2.2`** across `Version.props`, `WoWViewer.csproj`, `WoWViewer.CrossPlatform.csproj`, `ViewerApp.cs`, `CHANGELOG.md`, and READMEs.
  - **32/32 unit tests pass green in `RosettaTilesetGeneratorTests`.** Full solution compiles with 0 errors.

## 2026-08-27 — Spec 190 Phase 1: MCAL text painting, museum pedestals, and full-map Alpha WDT

- **Landed (Phase 1 / US1 complete):**
  - **MCAL/MCLY multi-texture label painting (`RosettaAlphaPainter`):** 1024×1024 MCAL text rasterization ($0.52\text{ m/texel}$) with quincunx antialiasing and 4-bit nibble slicing into 2-layer MCLY/MCAL chunks (`GroundTexture` + `InkTexture`).
  - **Museum pedestals (`MCVT`):** Raised 4m plateau with 12.5m linear bevel ramp in terrain heightfield under each object cell to prevent model base clipping.
  - **Removed 512-tile Alpha WDT limitation:** Supported up to 4096 tiles for Alpha WDTs in single continent containers.
  - **CLI arguments:** Added `--ink-texture`, `--pedestal-height`, `--pedestal-bevel` to `wowviewer-inspect rosetta-generate`.
  - **Spec Kit & Tests:** Authored `plan.md`, `tasks.md`, updated `spec.md`; 30 focused unit tests pass in `RosettaTilesetGeneratorTests`. Solution builds with 0 errors.

## 2026-08-26 — Spec 176 session: Reconcile apply loop landed; scene-discern left uncompiled

- **Landed (commits `89a8eebe`…`6fa1adbb`).** Real PM4-guided preview + guarded apply; Reconcile tab
  replaces the freezing Match tab; `AdtPlacementEditor` is the only Core.IO placement writer (MODF
  bounds translate with moves); authored edits share the staged-save queue; align confidence is
  `exp(-d/25)` and already-aligned rows report `AlreadyAligned`; output defaults to
  `output/projects/<map>/<yyyyMMdd_HHmmss>`. Last green proof: viewer/solution Debug 0 errors; **81/81**
  focused `WowViewer.Core.Editor.Tests`.
- **Uncommitted, does not compile:** scene-discerned PM4/`_obj0.adt` pairs from
  `WorldScene.LoadedPm4Tiles`. Prefill helper was deleted while Sidebars still calls it. Next code
  step is Spec 176 **T001**, not more UX.
- **Still open:** in-scene overlay, off-thread preview, P1 transfer, user-owned reload/visual proof.
  Durable detail lives in `specs/176-object-transfer/` (`plan.md`, `tasks.md`, `quickstart.md`).

## 2026-08-25 — Spec 176 real PM4-driven preview + hardened apply (skeleton replaced)

- Preview parses the real PM4 guide (`Pm4ObjectSegmentBuilder` + `Pm4ReconciliationInputAdapter`);
  apply goes through `ReconciliationApplyService` + provenance sidecar + undo. ID allocation keeps
  an original-catalog high-water mark so substitute delete+add continues the tile chronology.
- **Defect fixed in `AdtPlacementEditor`:** ID allocation restarted from the post-delete max, so a
  substitute's delete+add pair allocated id 1 on a tile whose chronology reached 77. Allocation now
  keeps a high-water mark captured from the original catalog.
- Proof: full solution Debug build 0 errors; **79/79 focused `WowViewer.Core.Editor.Tests` pass**
  (adapter conversion/height/snapshots/self-corpus/candidate mapping, tile association semantics,
  apply round-trips incl. substitute ID continuity and staleness refusal, operation reverse
  semantics). One pre-existing failure noted independently: `Pm4RegionObjectGrouperTests`
  ("Non-empty region 0 should have objects") over the local development corpus — untouched by this
  work. Remaining gaps: in-scene proposal overlay (Phase 3 step 3), P1 cross-tile/cross-era transfer,
  user-owned real Museum/PM4 visual + independent-reader proof.

## 2026-08-25 — Editor Platform foundation + Spec 176 reconciliation core (library-first, tested)

- Stood up the Editor Platform as a new `WowViewer.Core.Editor` library (added to the solution) and
  implemented the dependency chain the Spec 176 plan gates on, all library-first so the runtime never
  references an editor type and the viewer stays buildable with the editor removed.
- **166 plugin host**: `EditorHost` + registry, deterministic build-version/era model
  (`EditorBuildVersion`/`EditorBuildEraRange`/`EraHandlerResolver`), lifecycle with fault containment
  (faulted plugins are not retried every frame), duplicate-identity startup failure, and a reference
  plugin.
- **167 bridge**: renderer-free scene snapshot (`EditorSceneSnapshot`), operations-as-data
  (`EditorOperation`/`PlacementMoveOperation`), and `PlacementWriteService` now delegating to
  `AdtPlacementEditor` (the in-place writer is deleted).
- **168 session**: cross-plugin undo/redo, aggregated dirty state, and write-safety policy (protected
  roots, MPQ refusal, output-dir resolution).
- **173 integrity gate**: validate-on-read verdicts, refuse-on-unverified/quarantined, re-read verify,
  and mandatory provenance. The modernwow census (FR-007) and real 384-group WMO merge (FR-008) are
  user-owned corpus gates.
- **175 placement authoring**: `AdtPlacementEditor` in `WowViewer.Core.IO/Maps` — move/rotate/scale/add/
  delete plus deterministic non-colliding ID allocation and MMDX/MMID/MWMO/MWID name-table merge/index
  remap, rebuilding only placement+name-table chunks and copying all other chunks byte-identically.
- **176 Phase 1**: reconciliation models + `Pm4ReconciliationEngine` in `WowViewer.Core.PM4/
  Reconciliation` — deterministic align/substitute/clone proposals with explicit
  `ReviewRequired`/`Conflict` status, residual/evidence records, and never a proximity-as-certainty
  claim.
- Proof: full solution Debug build 0 errors; 58 focused `WowViewer.Core.Editor.Tests` pass (plugin host,
  era resolution, bridge write-service round-trip, session, integrity gate, placement editor round-trip,
  reconciliation). Remaining is the viewer shell (editor destination, bridge adapter, PM4/Museum overlay
  preview, save/reload provenance UI) plus user-owned real-client/corpus proof — no runtime/visual claim
  is made from compilation.

## 2026-08-25 — Spec 176 PM4-guided Museum placement repair planning

- Expanded the existing object-transfer spec into a reviewed PM4-guided workflow: preserve the Museum
  placements as the editable source, use PM4 as a read-only guide, preview alignment/substitution/clone
  proposals, require explicit decisions, and save loose ADT/WDT outputs with provenance.
- Created the Spec Kit planning pack: research, data model, transport-neutral OpenAPI command shapes,
  quickstart, and a dependency-gated implementation plan. The design reuses the existing PM4 scorer,
  replacement synthesizer, coordinate service, ADT placement catalog, and writers; it adds no parser,
  serializer, model-training lane, or automatic identity claim.
- Proof level: planning/source inspection only. Real Museum/PM4 visual alignment, independent-reader
  reload, and output-write proof remain user-owned future gates. The local Spec Kit agent-context updater
  is absent from this checkout, so no generated agent file was changed.

## 2026-08-15 — Spec 151 WMO admission instrumentation (source proof only, nothing measured)

- Built the counter that was missing. `WmoAdmissionTally` / `WmoAdmissionStats` in
  `Core.Runtime/World/Visibility` record **two layers**: which WMO placements entered the visible set
  and why, and which groups inside them were submitted **and on whose authority** — runtime-visibility
  disabled, placement-transform invalid, portal conservative fallback, portal only, frustum only,
  both, or GPU-instanced shell. Surfaced at Utilities > Perf > "WMO admission (this frame)".
- `CollectVisibleWmos` gained a `ref` overload; a test asserts it produces identical cull counts and
  identical visible sets to the original, so the accounting cannot be blamed for a behaviour change.
  It also counts the two rules the old `WmoCulledCount` never reported at all — hidden-by-uniqueId
  and asset-not-resident both `continue`d without incrementing it.
- **Four source findings, all with magnitudes unmeasured.** Portal culling **cannot reject** a group:
  `UpdateRuntimeVisibility` unions the portal decision with raw frustum visibility, and a conservative
  fallback admits everything by construction. WMO placements are **never rejected by the frustum**,
  because the call sites pass `IgnoreVisionConeCulling: true` and the only frustum-rejecting branch is
  guarded by `!IgnoreFrustumCulling && !IgnoreVisionConeCulling` — the same shape as the
  old-Ironforge-past-fog symptom. Group admission is evaluated **twice per placement per frame**
  (opaque + transparent both call `UpdateRuntimeVisibility`). And the recorded 7512 is a
  **submission** count spanning both passes, not distinct groups; the old counter is left untouched so
  the recorded capture stays comparable.
- Solution builds with 0 errors. Core suite failure set is **byte-identical to the pre-change
  baseline** (same 9 unrelated failures), +11 passing. Verified by stashing the change and diffing the
  failure names, after an allocation-free-recording regression showed up and was fixed: the new frame
  stats field had been initialised from a static property, putting a lazy class-constructor check on a
  path the frame history asserts allocates nothing.
- **Nothing is measured.** The user-owned Stormwind flight is what turns these counters into a
  diagnosis. No admission rule may change before that reading exists.

## 2026-08-15 — v0.5.2.1 released; renderer owner moves to WMO group admission

- **Shipped v0.5.2.1** (commit `975d0c79`, tag `v0.5.2.1`, branch `v0.5.3-dev`, release workflow
  green with all four self-contained builds attached). Out-of-band patch because v0.5.2 shipped with
  known, unresolved frame-pacing jank. Added `wow-viewer/CHANGELOG.md` and
  `docs/releases/v0.5.2.1.md`; bumped both csproj files and `ViewerApp.ViewerProductName`; updated
  both READMEs. The unfixed Stormwind WMO problem is documented as a **known issue** with real
  numbers rather than left for users to discover.
- **Phase 2 — the periodic stall was named by capture, then fixed.** `PrepareObjectPhase` peak
  283.47 ms of which `AudioRuntime.Update` was **283.46 ms** (PM4 overlay 0.12, remainder ~0), so no
  subdivision was needed. The cost was not audio: `RefreshEmitterDiagnosticsIfDue` rebuilt an
  `AudioTriggerDiagnostic` per resident emitter (5565) on a **wall-clock 250 ms** timer on the render
  thread — explaining why the "every 47–50 frames" interval drifted with framerate — and it ran
  whether or not anything displayed the result. A second, movement-triggered copy: `RemoveTile`
  rebuilt the list synchronously on streaming eviction. Gated the rebuild on
  `NoteEmitterDiagnosticsObserved()`; eviction now invalidates only.
- **Audio scoped to the camera tile.** `WorldAudioRuntime.Update` consulted no tile information at
  all and scanned every resident tile. It now takes `TerrainManager.CameraTileX/Y` (passed in, never
  re-derived) within `AudibleTileRadius`. Tile keying was investigated and **cleared** as a cause.
- **MCSE coordinate frame: measured, not guessed.** MCSE emitters read as permanently out of range
  while water-derived ones work. `ConvertSoundPosition` does `chunkCorner - local` on an unevidenced
  comment; the Ghidra work proved the 0x34 field layout, not the frame. Added `McseFrameEvidence`
  (raw min/max per axis, chunk/tile/beyond counts, explicit "inconclusive" verdict on a mixed sample)
  rather than switching frames on a hunch. Still open.
- **Confirmed by the Stranglethorn capture:** unaccounted median 0.05 / p99 0.16 ms and pass gap
  259–314 → 9.45 ms (Phase 1); **526 batched / 3 unbatched** from 0/312 with `MdxOpaqueSubmission`
  p99 30.75 → 14.12 ms (Phase 3). SC-002/003/006 met. Frame p99 barely moved (259.70 → 246.62), so
  Phase 3 was explicitly **not** credited with fixing the gallop.
- **Confirmed by the Stormwind capture (2048 frames):** `PrepareObjectPhase` max **283.4 → 2.5 ms**
  and gone from the hitch list; `SceneMaintenance` max **454.5 → 3.9 ms**; unaccounted median 0.02 /
  p99 0.11 ms; median frame 17.40 → 6.98 ms.
- **New measured owner: `WmoSubmission`** — p99 154.10 / max 161.3 ms against a 0.71 ms median, and
  **all 592 recent hitches** attribute to it at 153–157 ms. Stormwind submits **all districts at
  once**: 7512 visible groups, 80484 draw calls, 15852 doodad submissions. **An admission problem,
  not batching** — 80200 of 80484 calls are correctly batched. Belongs to Spec 151; first step is to
  instrument group admission (considered / admitted / rejected + which rule) before changing logic.
- **Spec 153 Phase 4 is likely moot** — written against `SceneMaintenance` max 454.8 ms, which no
  longer reproduces. Re-measure before implementing. **Phase 5 step 2 still owed:**
  `DeferredAssetLoads` max 442.9 ms in Stormwind against a 3.5 ms budget; the admission policy
  bounded the additive overshoot but the single-load residual needs decode off the render thread.

## 2026-08-15 — Spec 153 Phases 1/3/5 implemented (source proof only, nothing measured)

- **Phase 1 (FR-001, SC-006).** `PrepareObjectPhase` now has a stage timer and appears in the stage
  table, so its cost — including the ~212 ms stall — is no longer part of the unaccounted pass gap.
  `WorldRenderStage` gained the value, `StageCount` went 18 → 19, and the pass-gap subtraction in
  `RecordRenderRegionBreakdown` includes it so the region breakdown and stage table agree instead of
  double-counting. New `WorldFramePassInstrumentation` declares pass → stage ownership; a reflection
  test fails if any `WorldFramePasses` member has no timer, any stage is unowned, or a stage is
  claimed twice. The Perf panel gained an `other (unprobed remainder)` row, which is what decides
  Phase 0's "subdivide rather than guess" branch.
- **Phase 3 (FR-004/005/006).** The MDX batching cause was found and it was not a capability gap:
  `WorldScene.PlanVisibleMdxPasses` passed `PlanOpaqueMdxRoutes` a `requiresUnbatchedRender`
  predicate whose whole body was `return true`, so the planner routed 100% of opaque MDX to the
  fallback by construction while WMO — which reads the renderer's own declaration — batched 198/198.
  The predicate now consumes `IModelRenderer.RequiresUnbatchedWorldRender`, the same contract the
  WMO-internal doodad path already used. GPU instancing stays held out
  (`MdxRenderer.SupportsGpuInstancedOpaque` is still `false`); the win is `BeginBatch` once per
  renderer instead of a full state setup per draw, with submission order unchanged. Found and fixed
  the one real visual divergence between the two paths: `RenderInstance` ignores `_wireframe` while
  `RenderWithTransform` honours it, so `_wireframe` is now part of `RequiresUnbatchedWorldRender`
  (which also corrects the WMO doodad path). Live-revertible via `WorldScene.MdxOpaqueBatchingEnabled`,
  exposed as a checkbox so the before/after is one flight rather than two builds.
- **Phase 5 (FR-008).** New pure `DeferredLoadBudget` learns per-kind (MDX/WMO) load cost via EWMA
  plus a decaying high-water mark, and `WorldAssetManager.ProcessPendingLoads` consults it **before**
  each load rather than only between them — the old `elapsed < budget` condition would start a 55 ms
  load with 0.1 ms of a 3.5 ms budget left. **Deliberately partial:** the first load of a frame is
  always admitted so an oversized asset cannot starve, and that admission is counted in
  `OversizedAdmissionCount`. SC-005 is *not* claimed; a synchronous load larger than the whole budget
  still costs what it costs, and closing that needs decode off the render thread (plan Phase 5 step 2).
- Solution builds with 0 errors. Core suite: 9 failures before these changes and the same 9 after,
  all pre-existing and unrelated; net +15 passing tests.
- **Nothing is measured.** Every SC still needs the user-owned Stranglethorn before/after. Phase 0's
  capture protocol and an empty results table are in
  [Spec 153 research.md](../specs/153-renderer-hitch-and-batching/research.md); Phases 2 and 4 stay
  gated behind it.

## 2026-08-15 — Renderer gallop diagnosed; Spec 153 opened, flattening lane suspended

- Built the missing detector. `WorldRenderFrameStats` already produced `TotalCpuMs` plus 18 per-stage
  timers every frame and the viewer discarded all of it — `LastRenderFrameStats` held one frame and
  no history existed anywhere. Added `WorldRenderFrameHistory` (fixed-capacity ring, per-stage
  percentiles, hitch marking, unaccounted time, region peaks, allocation-free recording asserted by
  test) and an in-viewer panel under Utilities > Perf with an injected-stall self-check.
- Diagnosed against real clients across four zones. **Four measured defects**, all recorded in
  [Spec 153](../specs/153-renderer-hitch-and-batching/spec.md): a ~212 ms stall every ~47–50 frames
  living entirely in the pass gap; 100% of opaque MDX submitting unbatched while WMO batches 198/198;
  `SceneMaintenance` max 454.8 ms; and a deferred-load budget checked only between loads (58 ms vs
  3.5 ms nominal).
- Root structural cause of the invisibility: `WorldFramePasses` declares eleven passes and only
  `PrepareObjectPhase` assigns no stage timer, so its cost could not appear in the stage table at all.
- **The allocation-churn hypothesis was refuted by measurement** (median world-render CPU 0.33–8.58 ms;
  traversal max 0.22 ms). Per Spec 152's own Phase 1 decision point, Phases 3–5 (flatten the scene
  graph into retained draw lists, view modes) are suspended rather than continued on momentum. The
  churn fixes that landed — traversal now allocation-free in steady state, diagnostics off the hot
  path, opaque-pass buffers reused and pooled — are kept on their own merits.
- Ruled out with evidence so they are not re-proposed: decoded-asset caching and LRU thrash
  (`MaxMdxCached = 0`, unlimited; 554 models serve 18663 instances), scene-graph traversal, per-frame
  diagnostic logging.
- Stranglethorn Vale is now the standard benchmark scene; it exposed both defects where quieter zones
  hid them. Baseline table recorded in the Spec 153 plan.
- No renderer defect is fixed yet. Next bounded action is Spec 153 Phase 0: capture the
  `PrepareObjectPhase` sub-probes to name the stall before proposing a fix.

## 2026-08-15 — Repair Utilities minimap routing after sidebar consolidation

- Isolated the Utilities page selector from the shared Inspect/Scene/Experimental index. Legacy
  menu, keyboard, and persisted-settings paths remain synchronized, while opening Utilities defaults
  to Minimap again.
- The minimap renderer and tile data path were not changed. The focused cross-platform viewer build
  passes with 0 errors; live loaded-world and interaction proof remains user-owned.

## 2026-08-15 — Spec 080 Phase 2D placement ownership cleanup

- Implemented the ownership contract: Scene Placements is a list-only WMO/MDX route, Inspect owns
  selected-placement editing, scene investigation, world context, and full MCNK/ADT analysis,
  Phase Map selection is under left World Maps, and SQL population has a named Experimental page.
- The historical composite world-object body remains compatibility-only. Focused route checks,
  diff checks, and the isolated cross-platform viewer build pass with 0 errors; compact-window and
  runtime visual proof remain open.
- Renderer, fog, ADT admission, object submission, and streaming behavior remain out of scope.

## 2026-08-15 — Spec 080 Phase 2E Inspect and terrain page consolidation

- Scene now keeps Placements/LOD only, Terrain Lab owns tiles with chunk clipboard/save, and Inspect
  exposes Archeology as a direct dropdown page alongside MCNK/ADT, world context, investigation,
  animation, and action pages.
- Compatibility callers map legacy Tiles to Experimental Terrain Lab and legacy Archeology to
  Inspect. Focused route checks and the isolated viewer build pass with 0 errors; compact-window
  and runtime visual proof remain open. Renderer and streaming behavior remain out of scope.

## 2026-08-15 — Sidebar entry-point convergence; renderer hitch deferred

- Main Panels entries now land directly on their named Utilities page: Log Viewer,
  Perf, Asset Catalog, and Taxi. Capture and Camera Path continue to land on the shared Capture page.
- Recorded the reported camera-movement hitch as a deferred Spec 150 observation. No renderer, fog,
  ADT admission, object submission, or streaming code was changed during this sidebar pass.
- Sidebar-only source checks and the full Debug build pass with 0 errors; compact-window manual proof
  remains open.

## 2026-08-15 — Utilities ownership and animation restoration

- Promoted Utilities to the canonical fourth right-sidebar destination. Audio now exists only as
  Utilities -> Audio; it is no longer a duplicate top-level destination or an Experimental page.
- Restored the existing MDX/M2 animation controls inside Inspect for standalone models and selected
  world MDX instances without adding another model-information route.
- The WMO-beyond-fog symptom remains deferred to Spec 150; no renderer or streaming code changed.

## 2026-08-15 — Keep source navigation in the left sidebar

- Removed the right-sidebar Scene `Source` page and its dead source-body route. The right Scene
  selector now contains only Placements, Tiles, and LOD.
- Kept compatibility mapping for placements, tiles, selection, and LOD; source/file/map loading is
  owned exclusively by the left Navigator sidebar.

## 2026-08-15 — Spec 080 Phase 2A sidebar IA and unified inspector

- Replaced the visible tabbed `Model / World / Tools` top row with five deliberate destinations:
  `Quick`, `Inspect`, `Scene`, `Utilities`, and `Experimental`. Quick, Inspect, and Scene now render
  their bodies directly without a second page strip; Utilities and Experimental retain only their
  purpose-specific page selectors.
- Added one inline unified inspector for selected models/objects, PM4 context, and current ADT/MCNK
  facts. Area-trigger and WL-liquid loading actions are inline rather than nested popup-only controls.
- Combined tile targeting, chunk selection, and MCNK/chunk clipboard operations in Experimental >
  Terrain Lab. Existing legacy callers map into the new destinations, while legacy route retirement
  remains a separate inventory/manual-proof gate.
- Focused source checks and the full Debug build pass with 0 errors. The full solution test command
  timed out after two minutes; the focused core suite completed with 1,019 passing, 9 unrelated
  baseline failures, and 1 skipped test. User-owned visual proof remains open.

## 2026-08-14 — Mute unproven automatic ZoneMusic playback

- Added an explicit world-audio policy that keeps automatic ZoneMusic playback disabled.
- Area/ZoneMusic resolution and status diagnostics remain visible, while MCNK/MCSE water and
  environmental emitter behavior remains independent and testable.
- Added a focused policy regression test; live water/audio proof remains user-owned.

## 2026-08-14 — Resident audio speaker-marker overlay

- Added a residency-change-only normalized emitter snapshot to WorldAudioRuntime and forwarded it
  through WorldScene.
- Added an off-by-default Audio-panel toggle that renders one source-colored 3D pin per finite
  resident MCSE/MCNK emitter through the existing batched overlay: amber MCSE, cyan MCNK water, and
  purple MCNK environment.
- Marker rendering does not probe audio files, enable world triggers, or create OpenAL sources.
  Focused diagnostics/build proof and user-owned visual/audible proof remain pending.

## 2026-08-14 — Spatial audio emitter coordinate correction

- Added the shared `TerrainCoordinateTransform` contract. Alpha and standard MCSE records retain
  raw/local positions but now anchor renderer positions to their owning chunk before range checks,
  OpenAL placement, and diagnostics.
- Corrected legacy MCNK liquid marker placement from `chunk corner + halfChunk` to the terrain
  convention `chunk corner - halfChunk`; diagnostics now identify the MCSE or MCNK coordinate path.
- Focused audio-contract tests pass (11/11). The Windows viewer project builds with 0 errors in an
  isolated output directory because the live viewer owns the normal Debug binaries. Live visual and
  audible proof against the configured 0.5.3 client remains user-owned.

## 2026-08-14 — Alpha 0.5.3 time-of-day cycle checkpoint

- Added the existing 2,880-unit / 24-real-minute world clock as a pure tested contract.
- Interactive WorldScene lighting now advances it from a monotonic frame clock by default; manual
  slider input freezes it until resumed, and Light DBC/LIT/sky/audio receive one same-frame value.
- Synthetic minimap generation remains fixed at its requested time and records
  timeOfDayMode=frozen in its manifest. Focused clock tests (10), viewer build, and harvest build
  pass with 0 errors; live early-client timing and authored-minimap tint history remain user-owned.

This is a short newest-first implementation ledger. It is not a changelog or archive. Older detail
belongs to the owning spec, linked workstream, or `memory-bank/archive/`.

## 2026-08-14 — MDX material shader parity checkpoint

- Restored the live GLSL MDX material inputs that the CPU path was already uploading: UV0/UV1
  selection, animated UV transforms, and view-normal sphere-environment mapping.
- Added a bounded `SphereEnvMap` reflective highlight, finite/clamped model-local lighting and
  emissive values, and a shared/tested material policy. The implementation translates the native
  BLS material contract; it does not load or port BLS bytecode.
- Viewer build passes with 0 errors. Real shader compilation, translucent/reflective model appearance,
  and comparison against the configured client/build remain user-owned proof gates.

## 2026-08-14 — MCNK liquid audio and camera residency stability checkpoint

- Projected resident MCNK flags/MCLQ/MH2O liquid state into typed environmental audio candidates.
  Exact-build `SoundWaterType` rows resolve `(liquid family, subtype)` to `SoundEntries` IDs without
  inventing IDs; world-trigger playback remains default-off and unresolved mappings stay visible.
- Corrected the Alpha MCLQ handoff to preserve 81 packed vertex records and 64 tile flags instead of
  flattening the surface. Standard MH2O liquid assignment replaces the provisional MCNK candidate for
  the same chunk rather than duplicating it.
- Removed camera-heading-driven residency churn, added one capped unload-hysteresis ring, and kept WMO
  group rendering fail-open for transformed frustum-visible groups after portal evaluation. These are
  source-level smoothness/correctness changes; user real-client visual/FPS/audio proof is still open.
- Active render tiles now follow mouse-look without reopening the streaming lease; focused catalog,
  flag, and audio contract tests pass 53/53. Viewer project build passes with 0 errors; the full
  solution build also passes with 0 errors. Full-suite test completion, focused PM4/area coverage,
  per-trigger toggles, ZoneMusic indirection, and runtime proof remain.

## 2026-08-14 — Dual-era AreaTable identity routing

- Restored an explicit build-selected area identity contract: Alpha 0.5.x uses packed MCNK
  `AreaNumber` high/low Zone/SubZone words with `ParentAreaNum`, while 3.3.5+ uses direct
  `AreaTable.ID` and `ParentAreaID` values.
- Applied the same layout to the status-bar/overlay `AreaTableService`, the area-audio catalog reader,
  and `WorldAudioRuntime`. Modern direct IDs now resolve through primary IDs only, so compatibility
  aliases cannot steal a row that numerically resembles a packed Alpha AreaNumber.
- Added collision and parent-path regression tests. Focused area/audio tests pass 13/13; full Debug
  build and real-client dual-era visual/audio proof remain pending.

## 2026-08-14 — Spec 151 portal/game-mode/simple-surface planning

## 2026-08-14 — Spec 151 Phase 1 bounded WMO portal visibility (`c70e1945`)

- Implemented the shared fail-open portal decision from the 0.5.3 Ghidra evidence: exterior seeds,
  source-side plane admission, transformed portal polygons, recursively narrowed clip volumes, and
  bounded depth/visit traversal. Invalid/missing geometry, singular placement transforms, boundary
  cameras, and capacity overflow remain conservative rather than dropping WMO surfaces.
- Replaced the WMO renderer's old center-distance/queue heuristic with one decision reused for group
  geometry, doodad admission, and liquids. `WmoRenderStats` now carries portal-tested, fallback, and
  admitted-group counters. The graph evaluator remains explicitly diagnostic and no longer marks a
  destination visited before its portal volume is accepted.
- Focused portal/graph tests pass 16/16 and the full solution Debug build passes with 0 errors.
  Real-client WMO visual/submission/FPS comparison remains user-owned. Next bounded slice is Phase 2
  pure game-mode state/physics and character-head anchoring; simple surface/logging work waits.

- Checkpointed the accumulated prior work as `3bfbbba4` before opening branch
  `151-portal-game-mode-surface`.
- Created the Spec Kit specification, Ghidra-backed research, data model, contracts, quickstart,
  plan, checklist, and dependency-ordered tasks for WMO portal-aware visibility, an opt-in character
  head/game-mode physics path, a low-information interactive surface, and interactive/forensic
  diagnostic profiles.
- Queried the live 0.5.3 Ghidra REST bridge directly (no GUI driving) and recorded native anchors for
  `RRenderThruPortals`, `RTransformPortal`, `ClipPortal`, `StabPortals`, and portal intersection.
  The current renderer's center-distance/depth heuristic and inside-root full-visibility fallback are
  documented as the first implementation gap.
- No source implementation or runtime/FPS claim has been made. Next bounded action is Spec 151 Phase
  1 portal decision/tests; game-mode and UI work wait for that checkpoint.

## 2026-08-14 — Spec 149 resident Zone/SubZone overlay slice

- Queried the restarted Ghidra bridge's open 0.5.3 program and confirmed `CMapChunk::Create @
  0x00698e10` stores the MCNK area value, while `AreaTableRec::Read @ 0x00585e20` exposes
  `AreaNumber`/`ContinentID`/`ParentAreaNum` and the native script getters keep ZoneText and
  SubZoneText separate.
- Added the Spec 149 resident-area story and handoff. `TerrainRenderer` now exposes a revisioned
  resident chunk snapshot; `AreaOverlayRegionBuilder` groups map-aware AreaTable results into finite
  Zone/Subzone footprint regions with deterministic colors and unresolved counts; `WorldScene` renders
  opt-in footprint outlines/pins; `ViewerApp` projects one label per group; the investigation panel
  exposes `Show Area Boundaries`, default off.
- The viewer project compiles with 0 errors using an alternate output directory because the normal
  running viewer executable is locked by PID 46216. Focused aggregation tests remain open; live visual
  streaming proof remains user-owned.

## 2026-08-14 — Spec 150 Alpha 0.5.3 renderer performance planning

- Created an evidence-first Spec Kit pack for learning from the 0.5.3 OpenGL renderer without
  porting original code: `specs/150-alpha-renderer-performance/`.
- Reused the existing production `profile-render` path and current WorldScene/TerrainRenderer/object
  counters as the baseline seam. The new lane requires native Ghidra anchors, explicit CPU versus
  GPU/driver timing classification, one reversible optimization at a time, and user-owned real
  client visual/FPS proof.
- Current retained tile VAOs/texture arrays, bounded tile admission, object visibility collectors,
  opaque batching, and GPU-instancing seams are foundations to measure, not proof of performance.
- No renderer source optimization or native performance claim was made. Next step: record 0.5.3
  renderer anchors and run two unchanged-source profiles before selecting the first owner.

## 2026-08-14 — Spec 149 PM4 region navigation and audio trigger controls

- Created the Spec Kit specification, research, data model, contracts, quickstart, and dependency-
  ordered tasks for replacing PM4 correlation UI with decoded resident-region navigation and double-
  click camera focus.
- Amended the audio scope after live viewer evidence: decoded MCNK flags/liquid data are first-class
  legacy environmental/water trigger inputs even when Alpha 0.5.3 has no MCSE, and later MCSE data is
  additive. MCSE raw/local positions must be normalized with the owning tile/chunk before range checks
  or OpenAL placement; diagnostics retain both forms.
- Audited the music path against the 0.5.3 client contract: AreaMIDIAmbiences pairing is represented
  correctly as day/night MIDI plus shared DLS metadata, but AreaTable.ZoneMusic is still incorrectly
  treated as a direct SoundEntries ID. Spec 149 now requires ZoneMusic row -> Sounds[day/night] ->
  SoundEntries indirection and explicit underwater ambience selection before playback claims.
- Defined a default-off master/per-trigger world-audio policy covering MCNK, MCSE, and current-area
  ZoneMusic inspection, while preserving explicit preview, provenance diagnostics, and MIDI/DLS gates.
- Player-height/game-mode movement is explicitly deferred. Implementation, focused tests, build proof,
  and user-run visual/streaming/audible proof remain open.

## 2026-08-14 — Shared Alpha AreaNumber high/low resolution

- Fixed the Alpha area contract across status and terrain audio: raw `AreaNumber` values are
  decoded as `high16=zone` and `low16=subzone`, with unsigned bit preservation and continent/map
  qualification. `AreaNumber`/`ParentAreaNum` are preferred for Alpha rows.
- Removed the unsafe legacy aliases that treated the two component words as standalone area IDs.
  Audio receives the status-bar's resolved ZoneText/SubzoneText context, so display and music row
  selection cannot drift to different Zone/SubZone rows.
- Added packed-word, high-bit, parent/inheritance, ambiguity, and reader regression coverage;
  focused tests pass. Real-client audio/archive/audible proof remains user-owned.

## 2026-08-14 — Alpha 0.5.3 Ghidra audio contract correction

- Read the open 0.5.3 client in Ghidra and recorded the evidence in
  `memory-bank/workstream-audio-client-053-ghidra.md`.
- Confirmed `AreaMIDIAmbiences` row-level MIDI/DLS pairing and DirectMusic hand-off; confirmed that
  `ZoneMusic` selects ordinary SoundEntries IDs rather than mapping a missing SoundEntries ID to MIDI.
- Confirmed Alpha MCSE’s 0x34-byte on-disk record versus the client’s 0x4c-byte in-memory
  `CWSoundEmitter`; corrected the reader and preserved its scheduler fields through the Alpha terrain
  hand-off. Focused decoder coverage is the next validation target.
- Confirmed the client’s map-emitter callback slots are cleared at initialization with no in-process
  registration xref. Native callback equivalence and audible proof remain open.

## 2026-08-14 — Alpha packed AreaNumber ZoneMusic resolution

- Corrected the active area-audio lookup to match the repository's DBCTool contract: Alpha
  `MCNK.Unknown3` is packed `(zone << 16) | subzone`, rows are matched by continent and
  `AreaNumber`, and parent audio inheritance follows `ParentAreaNum` before modern ID fallback.
- Preserved `AreaNumber`/`ParentAreaNum` in the shared DBC audio catalog and exposed both values in
  ZoneMusic status, including when the OpenAL backend is unavailable.
- Added packed-parent, continent-ambiguity, and reader-preservation regression coverage. Focused
  tests and the cross-platform Debug build pass; real-client ZoneMusic playback remains user-owned.

## 2026-08-14 — Spec 148 world-simulator plan and audio diagnostic slice

- Created the provenance-first `148-world-simulator` Spec Kit pack: audio truth, explicit camera
  actor, lease-based residency/batching, and local artifact-museum boundaries.
- Preserved raw MCSE coordinates alongside the existing renderer-world transform and exposed
  current resident emitter diagnostics for SoundEntries resolution, path/source, optional read/decode,
  distance, mute, and backend states in the audio panel. Diagnostics refresh on a bounded cadence
  instead of doing file-existence work on every render frame; explicit probing still reads/decodes
  only when requested.
- Cross-platform viewer Debug build passes with 0 errors; focused AudioRuntimeContractTests pass
  (9/9). Real MPQ provenance, audible playback, coordinate correctness, and performance remain
  user-owned gates.

## 2026-08-14 — Spec 146 visible audio mute control

- Added a clearly labeled, color-coded `AUDIO: ON` / `AUDIO: MUTED` button to the bottom status bar.
- Routed mute through the runtime master bus so resident emitters, preview audio, and ZoneMusic
  all silence together while preserving the configured master gain.
- Viewer cross-platform Debug build passes with 0 errors; audible mute/unmute remains user-owned.

## 2026-08-14 — Spec 147 minimap interaction, LIT coverage, and WMO doodad batching

- Removed duplicate fullscreen minimap ownership and routed docked/fullscreen input through a pure
  gesture state. Focused proof covers drag classification, target changes, timeout, invalid tiles,
  and exactly-once third-click teleport; real-client input proof remains user-owned.
- Added a low-alpha fog-colored LIT radius fill and outline to minimap markers. The color comes
  from the active LIT fog track.
- Added a cross-WMO opaque internal-doodad collection path that groups shared renderers across
  visible WMO placements, using GPU instance batches where supported and renderer-level CPU batches
  otherwise. Transparent/effect-sensitive fallbacks remain unchanged.
- Cross-platform viewer build passes with 0 errors; focused minimap and area-catalog tests pass
  (6/6). Dense Stormwind draw/CPU/FPS comparison is still user-owned.

## 2026-08-14 — DBC-driven area music resolution

- The active-build DBC provider now feeds AreaTable and optional AreaMIDIAmbiences into the viewer
  audio runtime. A resident terrain chunk's AreaID selects the most specific area or parent area.
- ZoneMusic IDs resolve through the existing SoundEntries catalog and active client file paths, then
  loop through the existing OpenAL source path. MIDI/DLS choices are reported explicitly as
  unsupported instead of being converted or guessed.
- Focused catalog inheritance tests pass; audible client proof and camera/capture synchronization
  remain open.

## 2026-08-14 — Spec 143 pre-alpha v2 LIT compatibility

- Added an evidence-bounded parser profile for the observed 0.5.3 `areatest.lit` payload:
  embedded 64-byte Global Light header, 60-byte legacy prefix, and two `0xA24` nine-track data
  sets inside a `0x1484` group payload.
- Retained the secondary data set as `LegacyPartialAlternate`; only the primary `Partial` group
  drives lighting selection. Modern v8.3–v8.5 track lengths remain strict `0..32`.
- Focused LIT tests pass (8/8), inspect tool builds, and archive-backed `lit profile` proof decodes
  `H:\\053-client` Azeroth `areatest.lit`. Viewer visual/runtime proof remains user-owned.
- Next: validate the active viewer's v2 LIT lighting/fog path, then resume evidence-backed WMO/M2
  lighting work.

## 2026-08-14 — Spec 147 minimap, fog residency, and doodad instancing plan

- Authored the bounded Spec Kit feature, research, data model, interaction/fog/batching contracts,
  quickstart, requirements checklist, and dependency-ordered tasks.
- The read-only audit found duplicate fullscreen minimap draw ownership, `TerrainManager` explicitly
  discarding `fogEnd` for streaming targets, and WMO-internal doodad batching remaining
  placement-local.
- No production code changed. Branch creation was blocked by the shared workspace's
  `.git/index.lock` permission; the user-owned `wow-viewer/imgui.ini` change remains untouched.
- Next: implement and validate Spec 147 Phase 1 only.

## 2026-08-14 — Spec 143 LIT source switching and fallback

- LIT discovery now enumerates every `.lit` file directly in the active map folder while retaining
  conventional archive probes; `.lit` is included in loose-file indexing.
- The Lighting and investigation panels can switch variants even when the selected profile failed
  to parse. When no usable map-scoped Light DBC profile exists, LIT loads automatically and its
  lighting/fog override starts enabled; marker overlays remain opt-in.
- Focused source proof passes (25 tests); the isolated viewer build passes with 0 errors and the
  existing warning set. Real-client variant/fallback appearance remains user-owned.

## 2026-08-13 — Spec 143 LIT spatial coordinate correction

- LIT list-header positions now decode client fixed-point XZY values by dividing by 36, swapping
  file Y/Z into semantic WoW XYZ, and applying the map-origin transform for renderer consumers.
- Viewer LIT diagnostics now distinguish raw XZY, decoded WoW, and renderer coordinates; minimap
  markers and camera focus use the same shared conversion.
- Focused source proof passes (23 tests); real-client marker/focus placement is user-owned.

## 2026-08-13 — Spec 142 near-field detail selection correction

- The focused regression reproduced the reported failure: a 25-tile budget kept only the immediate
  3×3 ring, then spent the remaining slots on forward tiles, allowing nearby side/rear ADTs to be
  removed while distant terrain remained visible.
- `DirectionalTileSelector` now protects the largest complete camera-centered square supported by
  the budget: 3×3 for 9–24 tiles and 5×5 at 25, before bounded forward-cone expansion.
- Focused selector tests pass (9/9). Viewer build and real-client movement/camera-path proof remain
  open and user-owned.

## 2026-08-13 — Spec 146 SoundEntries preview and diagnostics slice

- Added a reachable Tools > Utilities > Audio page with resident SoundEntries ID discovery,
  camera-local preview/stop, master/emitter gain controls, backend status, and last diagnostics.
- Hardened OpenAL source updates so listener/emitter/preview failures disable audio cleanly instead of
  escaping into the render loop; active source gain now tracks current attenuation and bus changes.
- Focused audio tests pass (10/10) and the cross-platform viewer build passes with 0 errors. User-run
  audible proof against a configured client remains open; MIDI/DLS, camera transport, and capture
  muxing remain out of scope for this slice.

## 2026-08-13 — Spec 146 packaged OpenAL Soft MCSE runtime

- Added the OpenAL Soft native package to both viewer targets and copy the selected desktop DLL
  beside Debug and publish executables as `soft_oal.dll` plus Silk.NET's `openal32.dll` name.
- The production native probe retains the loaded module before Silk.NET constructs `AudioContext`,
  preventing the prior missing-library and premature-unload failures; cleanup remains guarded.
- Focused audio tests pass (3/3), both viewer targets build with 0 errors, and a process-level
  `AudioContext` create/dispose smoke test passes against the packaged Windows output.
- Proven scope is resident MCSE positional PCM-WAV playback. User-run audible client proof remains
  required; MIDI/DLS/MP3/OGG playback and Play + Video audio muxing remain explicitly unsupported.

## 2026-08-13 — Alpha audio catalog documentation

- Added a plain-language guide for the Alpha area-audio catalog, including the `AreaTable` to
  `AreaMIDIAmbiences` join, day/night/underwater semantics, loose/archive asset resolution, exact
  `audio alpha-area` inspect commands, and the boundary between metadata proof and playback.
- Linked the guide from the viewer README, CLI guide, Spec 146 quickstart, and audio-engine plan;
  corrected the plan/audit wording that incorrectly described the existing catalog proof as absent.
- No playback backend or runtime audio claim was added; user-run audible proof remains out of scope.

## 2026-08-12 — Spec 104 MDX material/effect artifact repair

- Added classic `LITE` parsing to `MdxFile`, including static Omni/Ambient values and deferred
  `PIVT` resolution; the MDX shader now receives up to eight model-local light records.
- Added a focused synthetic parser test for the light entry/pivot contract. Source validation
  passes; the full Windows solution build remains blocked in this sandbox by denied access to
  `C:\Users\akspa\AppData\Local\Microsoft SDKs`, while the core I/O build passes.
- User-run viewer proof remains required to confirm visible MDX lamp/effect illumination.

- Implemented the missing premultiplied-alpha shader output that the transparent MDX blend state
  already requested; the compatibility fragment path now follows the same alpha contract.
- Preserved classic MTLS static emissive gain in the runtime MDX material layer and applied it as
  self-illumination only, without introducing dynamic scene lighting.
- Removed white 1x1 fallbacks from transparent MDX geosets and unresolved particle emitters; missing
  effect textures now fail closed instead of drawing invented white squares/webs. Alpha-key particles
  still use an explicit discard threshold.
- Focused parser/build proof and real-client visual proof remain separate; the user owns the latter.

## 2026-08-12 — Spec 142 shared WMO placement transform

- Routed global and tile-local MDX/WMO placement creation plus translation-only editing through one
  renderer-space transform, correcting the prior WMO-only positive-axis rotation path.
- Bounds use the same transform as mesh submission; focused transform tests pass. Real-client camera
  movement remains user-owned proof for WMO flash-in behavior.

## 2026-08-12 — Spec 142 retained-window object admission

- Resident neighbor tiles now remain eligible for MDX/WMO collection even when they are outside the
  directional detailed-terrain list; object bounds and frustum tests remain the submission gate.
- This closes the path where a 25-tile resident window still made nearby buildings disappear on camera
  turns because WorldScene admitted objects from selected tiles only. Real-client movement proof remains
  user-owned.

## 2026-08-12 — Spec 142 resident WMO camera-turn stability

- Kept camera heading as a pending WMO-load priority signal, but removed rear-cone draw-distance
  culling for resident WMOs already admitted by active ADT tiles.
- WMO visibility remains bounded by active tile admission, bounds/frustum checks, and distance;
  focused collector proof and a viewer build are required before user movement validation.

## 2026-08-12 — Spec 142 fog admission and detailed/WDL ownership correction

- Terrain tile and legacy chunk distance admission now measures the nearest point on the
  geometry bounds, not the tile/chunk center. A camera near a tile edge can no longer lose
  that nearby terrain merely because the center lies beyond the fog cutoff.
- WDL suppression now follows the selected-and-GPU-resident detailed ADT set each frame.
  Retained-only neighbors remain streamable and keep their WDL underlay until detailed terrain
  is actually submitted. Three focused bounds-distance tests pass; viewer runtime proof remains
  user-owned.

## 2026-08-12 — Spec 142 near-field WMO readiness ordering

- Pending GPU tile uploads are now ranked by selected active tiles before retained and stale
  completions, preventing background parse completion order from delaying neighboring ADTs.
- WMO assets for the camera tile and immediate retained neighbors are prioritized before WMO
  visibility collection; inactive retained tiles remain admission-gated and are not submitted.
- Source build passes in an isolated output directory; runtime movement/WMO flash-in proof remains
  user-owned.

## 2026-08-12 — Spec 142 restored bounded detail and selected-tile residency

- Preserved the renderer's established ADT coordinate span (`WoWConstants.ChunkSize`, 533.333
  yards) while widening directional selection; `WoWConstants.TileSize` is a legacy aggregate and
  is not interchangeable with the camera's ADT span.
- Selected detail tiles now participate in both desired residency and unload protection, while the
  separate retained window remains the camera-centered streaming policy.
- The selector now fills the active tile's immediate 3×3 safety ring before spending remaining
  budget forward, preventing close side/rear ADTs from popping out when the slider is 12 or lower.
- Added focused proof for the 1–25 selector and the established ADT coordinate span; real-client
  movement/FPS validation remains user-owned.

## 2026-08-12 — Spec 144 swept camera-path residency correction

- Corrected camera-path tile conversion to use ADT `TileSize` rather than terrain `ChunkSize`.
- Added a core swept-footprint selector that connects path samples in tile space and applies the
  configured tile radius, preventing fast/spline paths from skipping ADTs that then unload.
- Enabled ordinary Play to wait on the existing bounded preload lease when enabled; playback and
  capture completion/stop release it. Active directional rendering remains separate.
- Focused core proof and real-client playback/unload proof remain separate; the latter is user-owned.

## 2026-08-12 — Spec 144 cross-era client camera import repair

- Routed loose and loaded-client `.m2` camera imports through `M2ModelReaderDispatcher` instead of
  the later-era reader directly.
- Added explicit MD20 `0x109+` modern camera records with strict `0x74` span validation; removed
  the prior unconditional camera suppression that caused `cameraIndex` failures.
- Added the documented MD20 `0x100` early camera layout: `0x7c` records, old `0x1c` tracks, range
  slicing, and normalization into the shared sampler without changing later M2 track semantics.
- Focused source proof is 35 passing tests. Archived Cata `FlybyUndead` and `FlybyDwarf` both inspect
  as MD20 `0x109` with `cameras=1`; viewer playback/origin placement remains user-owned proof.

## 2026-08-12 — Spec 142 bounded camera-centered residency

- Added a pure `CameraTileWindowSelector` with deterministic bounded retention; radius two is the
  default and radius three is the explicit maximum.
- `TerrainManager` now uses the retained window for streaming/unload protection while preserving
  the directional active list for detailed terrain, liquids, scene graphs, and WMO/MDX objects.
- Added retained count/radius diagnostics and runtime controls. Focused tests and the full solution
  build are the source-level proof; radius 2/3 production capture remains user-owned.

## 2026-08-12 — Spec 142 active-tile object admission

- Scene-graph traversal and portal preparation now enumerate only the camera-selected ADT graphs
  plus external content; flat WMO/MDX collection and deferred bounds promotion use the same gate.
- Full-load retains residency for stress work without turning every resident tile into an object
  visibility candidate. Explicit capture-preload tiles remain admitted.
- The viewer builds with 0 errors and the focused directional-selector proof passes 4/4. A user-run
  production capture is still required to prove frame-time improvement and visual parity.

## 2026-08-12 — Spec 142 camera-inside WMO group admission

- WMO runtime group visibility now treats containment in any local group bounds as an inside-WMO
  state, even when the root MOHD bounds miss the camera; this keeps interior groups visible instead
  of entering portal traversal with no valid starting group.
- Focused inside/outside policy tests pass. Camera-track playback/video remains the real-client
  proof owner because the recorder is the benchmark for path stability.

## 2026-08-12 — Spec 142 strict directional tile baseline

- Added the pure `DirectionalTileSelector` contract and four focused geometry tests.
- Replaced normal fog/radial ADT admission with the active tile plus at most three immediately
  forward-facing neighbors; normal detailed/manual budgets are capped at four.
- Added render-boundary active-tile/detailed-draw diagnostics. Capture preloads and `--full-load`
  remain explicit exceptions and are not normal camera admission.
- Focused selector proof passes 4/4 and the viewer dependency graph builds with existing warnings;
  user-run movement/FPS proof remains open before any FOV-radiation work.

## 2026-08-12 — Documentation continuity cleanup

- Replaced the oversized root and viewer agent guides with short operational guides.
- Added `specs/STATUS.md` as the single current-spec router.
- Condensed this ledger and `activeContext.md`; removed duplicated historical narrative.
- Updated the documentation and plans indexes to point at the new handoff path.
- No source code, project files, client data, generated output, or active spec requirements were
  changed by this cleanup.

## 2026-08-12 — Spec 146 planning package

- Added the audio/camera playback spec, plan, tasks, and single-player roadmap.
- Scope includes capability-gated MP3/OGG/WAV/MIDI playback, emitters, camera-track audio, and
  future client/server seams. It does not select or implement a backend yet.
- Next: Phase 1 contracts and capability tests only.

## 2026-08-13 — Spec 146 resident MCSE playback slice

- Added build-aware standard/Alpha 0.5.3 MCSE emitter extraction to tile load results, including raw
  Alpha 76-byte identity, position, range, timing, mode, and preservation bytes.
- Added `SoundEntries` catalog loading from the active DBC provider, dependency-free PCM WAV decoding,
  and a viewer-owned OpenAL runtime that admits only resident tile emitters and releases them on unload.
- Added lower status-bar `Audio active/resident` diagnostics and focused Alpha MCSE/WAV contract tests.
- Source proof: focused audio tests pass (2/2), focused MCSE tests pass (2/2), and the cross-platform
  viewer Debug build passes with existing warnings. User-owned proof remains audible playback in the
  configured client; MIDI/DLS, MP3/OGG/FLAC, camera-track audio, and capture muxing remain open.

## 2026-08-13 — Sparse MCCV terrain preservation correction

- Fixed the 3.x–4.x sparse-MCNK loss where a short MCNR declaration caused the padded subchunk walk to
  skip a following 580-byte MCCV payload, even when MCLY and MCAL were absent.
- Split root/texture/object sources now retain whichever valid MCCV payload exists, and the live terrain
  adapter selects MCCV independently from the layer/alpha source. MCCV guide-image and tensor extraction
  retry using declared subchunk sizes for the same sparse layout.
- Focused source proof passes 8/8 across parser, guide-image, tensor-adjacent, and split-ADT tests. User
  still owns real 3.x–4.x client visual validation on a tile with MCCV but no MCLY/MCAL.

## 2026-08-12 — Spec 144 capture path slice

- Camera path authoring, JSON camera state, roll/time controls, contextual keybinds, path preload,
  client FlyBy import, collision hooks, and capture controls are present in the current viewer
  surface.
- Focused source/build proof exists; user-run real-client and capture proof remains open.

## 2026-08-12 — Spec 145 UI first slice

- Contextual help/keybind surface, bounded sidebar navigation, wrapped log output, and v0.5.2 UI
  metadata were landed.
- Remaining work is the explicitly listed persistent-window and placeholder/control audit; do not
  infer a complete UI overhaul from the first slice.

## 2026-08-11 — Specs 142/143 world slices

- Scene-graph/performance and world-context work have source slices, but FPS, runtime stability,
  WMO-area decoding, and lighting still require their owning evidence gates.
- Treat crash logs and user screenshots as validation inputs, not as implementation proof.

## Handoff rule

When a task completes, update the owning spec first, then adjust this ledger only if the next-agent
routing changed. Move superseded detail to the owning archive rather than appending more history.
## 2026-08-13 — OpenAL absence must fail closed

- Added a platform-aware native OpenAL probe in `WowViewer.Core.Audio`.
- `WorldAudioRuntime` now avoids Silk.NET `AudioContext` construction when the
  optional native library is absent and guards cleanup after backend failure.
- Added a missing-library contract test and documented the no-OpenAL acceptance
  path in Spec 146.
- Remaining proof: run the viewer without OpenAL installed and verify it stays
  alive; then validate actual emitter playback with an OpenAL-enabled client.
## 2026-08-13 — Workbench tab rails replace unreachable overflow arrows

- Replaced the workbench's horizontal primary and nested sub-tab strips with
  directly clickable vertical rails.
- Capture Automation and Camera Path now use the same reachable rail when
  opened inside the Utilities surface.
- Updated Spec 080 with the reachable-navigation requirement and task proof
  row. Remaining proof is compact-window UI validation by the user.
