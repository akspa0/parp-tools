# Progress — V16.x Terrain Training Pipeline

## Completed
- 2026-05-27: Expanded Spec 001 (`specs/001-v18-dataset-spec/`) into a fuller V18 dataset canonical contract and then revised it to reflect the simpler direction: V18 is the direct versioned successor to the V16 dataset creation flow, with decoded metadata and currently patched-on signals promoted into the main build contract
- 2026-05-27: Added `specs/001-v18-dataset-spec/plan.md` and `tasks.md` manually on `v0.5.0-dev` after bypassing Speckit branch-gated scripts
- 2026-05-27: Created `data-harvester/scripts/build_v18_dataset.py` as the copy-forward V18 builder, changed it to write under `output/datasets/v18`, added canonical V18 artifact/finalization helpers, and integrated optional renderer-truth promotion into the `build` command
- 2026-05-27: Upstreamed object-roof support into the shared harvest/tensor-pack contract by adding roof arrays plus roof-mask-source metadata to the C# pack and serializer surfaces, and taught `build_v18_dataset.py` to treat those arrays as canonical streamed V18 signals
- 2026-05-27: Added an explicit `--experimental-renderer-truth-promotion` gate so capture-derived V18 signals are not treated as proven canonical outputs before refreshed object-loading/capture validation
- 2026-05-27: Locked scope so the parser → decoded → dataset direct-pipeline redesign is deferred to a future V20 dataset effort instead of being folded into V18
- 2026-05-27: Bounded `build_v18_dataset.py build --limit 1` proof succeeded on staged `3_3_5_12340 / Azeroth`; emitted `finalization.json`, `signal_validation.json`, and `decoded_metadata_validation.json` with pass status
- 2026-05-27: Validation-capture dry-run readiness passed on staged `0_5_3_3368 / Azeroth_30_48` and `3_3_5_12340 / Azeroth_30_48`, but non-dry-run `gpu-viewer-style` capture on staged `3_3_5_12340 / Azeroth_30_48` still produced flat/uniform renders and an all-black object-visibility artifact
- 2026-05-27: Corrected the proof-owner boundary for roof/object visual evidence — `wow-viewer` command-path success is not enough; bounded `MdxViewer` compatibility proof remains the credible visual lane until parity is demonstrated 🧭
- 2026-05-23: V16.1.2 refiner added (later abandoned — gradient can't flow through detached graph)
- 2026-05-23: V16.1.3 height-channel normal model (4ch input) added and trained (plateaued epoch 123)
- 2026-05-24: V16.1.4 `V161NormalHeightCombinedModel` added to v16_1_models.py
- 2026-05-24: `_combined_loss` + `_preview_combined` + `combined` task registered in TASKS
- 2026-05-24: CLI flags `--normal-weight`, `--height-weight` added
- 2026-05-24: Spec 020 written (renderer culling fix → tile-level capture → V16.2 masks)
- 2026-05-24: Memory bank updated with MotherShip direction

## In Progress
- Bounded V18 dataset-builder validation still needed on staged client roots
- Real object-rendering proof for wow-viewer validation capture is still not closed despite command-path success
- Renderer culling fix needed before V16.2 dataset generation can proceed
- V16.1.4 combined model implemented but not yet trained
- Context-drift prevention: keep route, proof owner, and scope stated explicitly when the lane changes ⚠️

## Next Up
- Run a bounded `build_v18_dataset.py build` proof and inspect `finalization.json`, `signal_validation.json`, and `decoded_metadata_validation.json`
- Refresh real object-loading and capture proof on the bounded staged anchors before widening renderer-truth promotion claims
- Route the next capture fix through the existing renderer/culling specs instead of pretending the emitted flat artifacts are acceptable proof
- Simplify or retire the Python-only `patch_v18_object_roof_masks.py` workflow where the shared C# roof arrays now cover the same contract
- Use a bounded `MdxViewer` proof artifact before calling full ADT MCLY/object-inclusive data trustworthy 🧪
- Fix coordinate bug in `ComputeTilePlanarMin/Max` so single-tile capture works
- Tile-level loading (not full-map)
- Batch capture (Noggit-red composite pattern)
- V16.2 precise object mask generation
- V16.2 model with height-channel, combined heads, proper object weighting
- V18 object-roof curation + minimap sieve spec now drafted; next step is to split it into bounded implementation slices
- V18 object-roof lane now explicitly includes MdxViewer per-asset capture improvements and a separate object-visual Zarr store
- V18 object-roof lane now also explicitly calls for a Python `uv` + transformers object-identification model that feeds the main V18 model
- V18 object-roof lane now treats SAM2 as the initial promptable mask host and SAM3 as a gated follow-on if the Hugging Face token unlocks it
- **2026-06-09 051 text + semantics doc**: Spec 051 visualization (Phase 1) was already landed in a prior session. This session rewrote the in-program glossary + color-token doc comments + spec Findings 1-7 + plan to reflect the 2026-06-09 visual+code reading of the PM4 streams. Authoritative doc is `wow-viewer/docs/architecture/pm4-chunk-semantics.md`. The "MSCN = polygon centroids / MSPV = shared vertices / PM4 = navmesh graph" interpretation is dead. Build verified: 0 errors. PM4 unit tests: 4/4 pass.
- **2026-06-09 051 MdosIndex → MscnRefIndex rename**: full rename across `wow-viewer/src/core/WowViewer.Core.PM4/Models/`, `WowViewer.Core.PM4/Research/`, `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`, `ViewerApp_Pm4Utilities.cs`, `ViewerApp.cs`, `Pm4OverlayCacheService.cs`, `tools/inspect/.../Pm4MatchSupport.cs`, `tools/inspect/.../Program.cs`, and `tests/.../Pm4ResearchIntegrationTests.cs`. PM4 overlay cache version bumped from 6 to 7 (cache format breaking — old cached overlays will be invalidated on next load). The real `MDOS` chunk (destructible object state) and its references in `Pm4MdsfEntry.MdosIndex` and the `MDSF.MdosIndex -> MDOS` edge stay correctly named. Build verified: 0 errors. PM4 tests: 10/10 pass.
- **2026-06-09 053 Phase 0+1 ship + pause for 054**: Spec 053 (M2/MDX animation pose farm) Phase 0 (research) and Phase 1 (library skeleton + loaders) shipped. 21/21 tests pass. New `WowViewer.Core.Anim` library + test project added to `WowViewer.slnx`. One pre-existing build error in `pm4PerFileCacheService.cs` (missing import of `WowViewer.Core.PM4.Models`) fixed as a one-line unblock. Phase 2-9 deferred to a future session.
- **2026-06-09 054 partial land, viewer wiring pending**: Spec 054 (PM4 per-file camera window cache) library code landed as untracked files: `Pm4PerFileCache` (T002), `Pm4PerFileCacheEntry` + `Pm4CachedTile` + `Pm4CachedObject` (T001), `Pm4PerFileCacheService` (T007), `pm4PerFileCacheTests` (T003, 8 tests). The critical viewer wiring (T004-T006, T008-T012) is **not** done and the working tree is in a half-merged state on `WorldScene.cs` and `Pm4OverlayCacheService.cs` — finishing those in a fresh session is recommended. This is the root cause of the reported PM4 tile navigation hard crash.
- **2026-06-09 054 viewer wiring + stamp-folding fix**: This session finished the 054 viewer integration that was pending. The in-memory `Pm4PerFileCache` and on-disk `Pm4PerFileCacheService` are now both wired into `WorldScene.LoadPm4OverlayAsync`'s per-file loop. The status format now shows `(mem-cache N hit, disk-cache M hit)`. The architecture doc has a new "PM4 overlay cache layout" section. **Critical bug fixed**: the in-memory cache read was passing `lastWriteTicks = 0L` while the writer stored a folded split-flag stamp — this made the in-memory cache silently miss in the default config (split-by-MscnRef on, split-by-connectivity off), which was the root cause of the reported "PM4 tile navigation hard crash." The fix computes the same fold at the read site. 18/18 PM4 per-file cache tests pass; 21/21 anim farm tests pass; viewer builds clean. T015 (real-data smoke) and T018 (manual end-to-end UX timings) are deferred to a session with the staged 3.3.5 client and an interactive viewer.
- **2026-06-09 MSCN/MSPV lazy guard fix**: The `EnsurePm4MscnData` / `EnsurePm4MspvData` lazy extractors at `WorldScene.cs:10680` and 10703 had an inverted early-return: `if (_pm4TileMscnPoints.Count > 0 || _pm4TileObjects.Count == 0) return;` bailed out as soon as ANY tile was populated, so subsequent camera shifts that added new tiles never got their MSCN/MSPV points extracted. The fix drops the `Count > 0` short-circuit and trusts the inner per-tile `ContainsKey` guard. **The "MSCN/MSPV nodes require clicking Reload PM4" symptom is resolved.** This is a pure correctness fix, not a cache fix; it does not change the cache architecture.
- **2026-06-09 Spec consolidation + memory bank merge**: Consolidated 52 specs → 25 active + 19 archived. Archived obsolete V16/V17 model specs, merged 050/052 into 046. Decommissioned `gillijimproject_refactor/memory-bank/` in favor of `wow-viewer/memory-bank/`. Created new wow-viewer memory-bank files: projectbrief, techContext, systemPatterns, dataPaths, codingStandards. Wrote WoWViewer README.md, CLI-TOOLS.md advanced guide, and PLANS-OVERVIEW.md. Updated activeContext.md to current state.

## MotherShip Direction
Long-range: `theMothership/game-engine/` — universal game engine with WoW plugin. See `wow-engine-modernization-plan-2026-05-14.md`.

## Spec 056 — ViewerApp + GPU + LOD Modernization (NEW 2026-06-10)

Authored the full Spec Kit pack under `wow-viewer/specs/056-viewerapp-gpu-lod-modernization/`:

- `spec.md` (7 user stories, 20 FRs, 10 SCs; locked D1-D8; supersedes 036)
- `plan.md` (9 phases, 0-8; max 10 steps each; constitution-clean)
- `research.md` (Phase 0; maps existing `WowViewer.Core.Runtime.World` LOD/visibility surface; per-phase porting risks; reuse-and-adapt map)
- `data-model.md` (Phase 1; `RenderScene`, `RenderBackend`, `RenderResources`, `TextureCache`, `PerFrameRenderStats`, LOD settings, **WMO pass dispatch types from Ghidra correctness oracle**)
- `contracts/RenderScene.md`, `RenderBackend.md`, `RenderResources.md`, `TextureCache.md` (Phase 1 backend-neutral interfaces)
- `quickstart.md` (build/test/validate commands, per-phase baselines, per-phase validation checklist, source-of-truth map)
- `tasks.md` (81 tasks, 9 phases, max 10 per sub-phase; US1-US7 mapped; MVP = Phase 0+1)

Plus the audit: `docs/architecture/spec056-viewerapp-gpu-lod-modernization-analysis-2026-06-10.md`.

**2026-06-10 source-of-truth correction** (D1 amended, D8 added):

1. **Source of truth = `wow-viewer/src/viewer/WoWViewer/Rendering/*` and `WoWViewer/Terrain/*`**. The new shared library is built by *improving and moving* this code. The legacy `MdxViewer/Rendering/*` is read-only reference (RULE 1) and is **not** the source of truth.
2. **`wow-viewer/src/viewer/WowViewer.App.Defunct/*` is forbidden** (user instruction 2026-06-10). Do not read, do not port, do not reference. Treated as a poisoned source.
3. **Ghidra doc is the correctness oracle** for the new WMO renderer (`docs/architecture/wmo-render-pass-architecture-2026-05-30.md`). The new WMO renderer conforms to the dispatch logic in that doc (interior/exterior by `flags & 0x48`, per-batch MOMT flag testing, lightmap pass split, liquid type dispatch, portal-walk visibility, group flag filtering). It is **not** a code source. Added a 4a. WMO pass dispatch section to `data-model.md` (`WmoGroupRenderMode`, `WmoGroupFlags`, `WmoBatchMomtFlags`, `WmoBatchRenderPass`, `WmoLiquidType`, `WmoPortalWalkState`, `WmoInteriorFogState`, `WmoDayNightInfo`). Added US3 acceptance scenarios 5-10 to `spec.md` to test Ghidra conformance.

**Current state**: spec authored, source-of-truth corrected, not yet implemented. Next step when implementation starts: Phase 0 (T001-T009) creates the new test project, locks the contract seams, and records the pre-spec capture baseline.

## 2026-06-10 Small-Wins Pass

Three bounded slices landed outside the active spec work. Each is independently committable, runs the build green, and clears warning debt without scope creep.

- **B — dead-field cleanup in `ViewerApp.cs` and `WorldScene.cs`.** Removed 8 write-only private fields from `ViewerApp.cs` (`_showDemoWindow`, `_showBottomDrawer`, `_listfileInputBuf`, `_mouseOverViewport`, `_terrainWeakSignalRestoreUseShadowHeuristic`, four `_mkHarvest*`) plus the lone write-only field `_pm4ZeroCk24SplitComponentCount` and its reset assignment in `WorldScene.cs`. Verified no serialization, reflection, or external reader touched any of them. Build: 163 → 123 warnings (40 cleared), 0 errors. The `_pm4PositionRefCount` / `_pm4RejectedLongEdges` / `_pm4TotalMsurCount` / `_pm4DroppedShortIndexCount` / `_pm4DroppedOutOfRangeMsviCount` / `_pm4DroppedEmptyComponentCount` / `_pm4MinObjectZ` / `_pm4MaxObjectZ` siblings were left alone — they have public accessor properties and are part of a coherent PM4 diagnostic surface.

- **F — `Pm4MatchRunOptions` for the 046 PM4 matching lane.** Created `wow-viewer/src/tools-shared/WowViewer.Tools.Shared/Pm4Matching/Pm4MatchRunOptions.cs` carrying the typed options for the three PM4 commands (`match-assets`, `synthesize-placements`, `export-asset-signals`). Includes `Pm4MatchCommand` enum, the record itself, and a `Validation` nested class with an `ErrorCode` enum and human-readable `FormatErrorMessage` for each error path. Closes the missing-file gap called out in the 046 progress entry. Caveat: no `WowViewer.Tools.Shared` test project exists today; the new validator is verified by build only. **Follow-up**: spin up a focused test project for the shared lib when a future slice calls for it.

- **D — honest attribution on the `TerrainManager.SetSubObjectVisible` TODO stub.** The stub is required by the `ISceneRenderer` interface and is intentional while per-tile visibility is un-specced. Reworded the inline TODO comment to make the deferral explicit and tracked, not lost. The actual per-tile-visibility feature is a future-scope change requiring: `Visible` on `TerrainTileMesh`, a visibility map in `TerrainManager`, and render-path filtering — out of scope for this pass per Rule 8.

- **2026-06-10 click-freeze dedup.** `ViewerApp.PickObjectAtMouse` ran `_worldScene.TryPickPm4ObjectByRay` unconditionally, then `TryHandleSceneClickSelection` ran the same pick again internally. On dense development maps the per-click PM4 pick loop (`_pm4TileObjects` walk + per-object `BuildPm4ObjectTransform`) was paying for itself twice. Moved the outer pick into the `addPm4ToCollection` (Shift+LMB) branch where it is actually used; the normal-click path now relies on the inner pick only. Halves the per-click PM4 work in the common case. Build green; behavior contract preserved.

- **2026-06-10 hot-swap missing-map fallback.** `RestoreWorldAfterDataSourceReload` would fall back to the previous-client local cache (`localPath` from `TryGetLoadedLocalWdtPath`) whenever the in-MPQ load via `LoadFileFromDataSource` left `_worldScene == null`, even if the new data source did not have the same map. The on-disk cache was written by the *prior* data source; loading it through the new source can hang the viewer when the StandardTerrainAdapter then queries ADTs that the new source does not have. Probed the new data source with `MpqDataSource.FileExists(virtualPath)` first; only fall back to the local cache when the new source confirms the WDT is present; otherwise set a clear status line (`Map "<name>" not present in the new client; previous world cleared.`) and return without touching the cache. Build green; the click handler is unaffected because the gating happens before any terrain-mesh construction. This is the first follow-up task from spec 057.
