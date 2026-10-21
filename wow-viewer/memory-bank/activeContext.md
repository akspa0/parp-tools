# Active Context — wow-viewer

Last updated: 2026-10-21 · **Spec 280 UI refactor implemented; startup crash fixed in code, operator smoke owed**:
- **Spec 280 UI refactor — implementation landed, ledger reconciled 2026-10-21** (branch
  `280-ui-rework`; spec, plan, tasks, audit and 18 receipts under
  [specs/280-ui-audit-refactor/](../specs/280-ui-audit-refactor/)). **Landed:** one
  `ViewerUiManager` owns every chrome window (one token set, one draw order, one input rule, one
  popup registry); three destinations (Viewer / Editor / Archaeology) behind a slim bottom bar with
  an Options sub-menu; top menu reduced to File / View / Help; the whole Tools menu and all
  export/import trees deleted with their actions moved to their owning destinations; 7
  compatibility page enums, `Canonicalize`, the legacy tab redirects and `ShellLayoutService`
  deleted, with versioned `output/settings/ui-layout-v1.json` persistence; the 12 duplicate
  floating windows deleted (Log and Perf only remain); archaeology's legacy 5-tab strip, floating
  UniqueId window and Quick launchers deleted (the destination is now **5** pages, PM4 is a
  cross-destination button); Editor sections renamed to their real owners; PM4
  matching/alignment/reconciliation removed and moved to [Spec 281](../specs/281-museum-adt-patcher/spec.md).
  **Where the code lives:** `src/viewer/WoWViewer/Workbench/Services/UiManager/`
  (`ViewerUiManager.cs`, `.Input.cs`, `.Persistence.cs`, `UiLayout.cs`, `UiPopupRegistry.cs`,
  `UiTokens.cs`, `UiControlRegistryService.cs`), plus the content-only providers
  (`Chrome/`, `Navigator/`, `WorkbenchPanels/`, `MainMenu/`, `FileOperations/`).
  **Receipts:** [final receipt](../specs/280-ui-audit-refactor/evidence/final-receipt.md),
  [tasks ledger](../specs/280-ui-audit-refactor/tasks.md),
  [audit §0.11 / §2.1](../specs/280-ui-audit-refactor/audit.md).
  **Verified 2026-10-21:** solution build 0 errors; tests Core 7 failed / 1822 passed / 1 skipped
  (the 7 are the pre-existing format/data set from the P0 baseline, none UI), Editor 91/91,
  Curation 39/39, PM4 165/165; `eng/ui/Verify-UiReference.ps1` exit 0
  (`inventory_rows=125 distinct_labels=117 registry_entries=117`, 0 errors, 0 warnings) and a
  117-control `docs/WoWViewer/UI-REFERENCE.md` generated from `eng/ui/ui-controls.json`.
  **Startup crash found and fixed in code:** `ViewerUiManager..ctor` captured
  `host.MinimapAndStatus.DrawStatusContent` before `ViewerApp` assigned `_minimapAndStatus`
  (`ArgumentException: Delegate to an instance method cannot have null 'this'`). The manager now
  resolves the provider at draw time (`ViewerUiManager.cs:24-30`). The viewer has **not** been
  launched, so startup is still operator-owned proof.
  **Operator-owned / open (T092, T093):** no real-client run, no visual/FPS/audio proof, no
  in-viewer export/import proof. The `Ctrl+K` command palette (T106) was not implemented. The
  type scale (T100) is defined but not applied because the viewer uses one ImGui font. Bespoke
  styling remains outside the five manager-owned surfaces (34 files / 179 hits) and the
  documentation `WC3 Beta` rename plus the final dead-control sweep are not done.
  **Operator feedback 2026-10-21 (first real run) and fixes:** the first launch showed a 4K UI
  with 280 px sidebars, a 12 px font, a floating thumbnail minimap, a chat that resized off-screen,
  bottom bars below the client area, and sidebars whose long lists pushed everything else out of
  reach. Fixed in code: `UiLayout.ScaleFor` scales every metric and the font by display height
  (1.0 at 1080p, 2.0 at 4K); the layout uses the window client size instead of ImGui's
  `DisplaySize` (which could be the framebuffer size); the minimap is **docked at the top of the
  right sidebar by default** and sized to the sidebar; the chat is movable, resizable, clamped to
  the display and persisted; an **Options** button sits in the chat header; **Fog Start/End** are
  back in the Workspace section; the left sidebar order is Workspace → destination sections →
  Diagnostics → World Overview → File Browser (long nested lists last); layout persistence bumped
  to `ui-layout-v2.json` so the stale tiny layout is ignored.
  **Hover-jump fix (operator report 2026-10-21):** hovering the right sidebar made the Inspector
  content collapse and the controls move. Cause: scene hover picking ran every frame even when the
  pointer was over a sidebar, so the hovered asset cleared, the Inspector payload changed, and the
  controls moved under the cursor. Hover picking now runs only while
  `CanSceneConsumeMouse` is true (`ViewerApp.cs:1039-1047`).
  **Spec 282 (operator-originated 2026-10-21):** real client cursor model
  (`Interface\Cursor\cursor.mdx`/`.m2`) as the in-scene pointer, plus precise PM4 picking.
  [Spec](../specs/282-client-cursor-and-pm4-pick/spec.md) ·
  [Receipt](../specs/282-client-cursor-and-pm4-pick/evidence/receipt-2026-10-21.md).
  **PM4 picking fixed:** the pick was AABB-only, so a large object's box could steal a click; it
  now rejects on the AABB and then tests the actual PM4 triangles (Moller-Trumbore), selecting the
  nearest triangle under the ray (`Pm4OverlayScene.Selection.cs`, `Pm4OverlayScene.cs`).
  **Client cursor model implemented:** `SceneCursorRenderer` dispatches candidates by extension —
  MDX through `MdxRenderer`, M2 through `WowViewerM2RuntimeBridge` + `M2Renderer` with skin
  resolution and an Alpha-era inline-geometry fallback — and `ViewerApp.OnUpdate` re-points it at
  the live data source (it was constructed before any client opened and never re-pointed, which is
  why the authentic cursor never loaded). Options > Objects reports the resolved model or the load
  error and offers `Reload cursor model`. Build 0 errors; `Verify-UiReference.ps1` PASSED; PM4
  165/165; UI doc tests 3/3. **Not proven:** no real client cursor asset has been loaded in a run,
  and PM4 selection accuracy needs an operator spot check.
  **Next bounded step:** operator relaunches, opens a client, and confirms the cursor model and PM4
  click accuracy.
- **Spec 280 UI audit (2026-10-20, operator-directed):** the full nesting map is in
  [audit.md](../specs/280-ui-audit-refactor/audit.md). File menu 18 entries/3 levels; View 15/8
  duplicates; Tools 36/5 levels; Editor 4 pages/13 sections (3 host archaeology-named content);
  Archaeology 6 pages with **four parallel navigation systems** (destination strip, legacy 5-tab
  strip, floating UniqueId window, Quick launchers) and a 4-level Cartography tree; 12 Tools >
  Panels floating windows duplicate destination pages. This audit was the implementation contract
  for the 2026-10-21 pass above.
- **Known bugs to fix (operator-reported 2026-10-20, not yet fixed):**
  1. **Sky / skyboxes** — no proper skybox support. Use the wowdecomp 3.3.5 / 4.3.4 sky code. Add
     Quick-tab controls with a list of known-good settings from DBC/DB2 plus the current blank-sky
     default for simple viewer use.
  2. **Population model fallback** — 0.5.3 and 3.3.5 population tries to load `humanmale.mdx/m2`
     for every quest-giver / gossip NPC. Suspect the bark/`bunnies` handling is not selecting the
     real creature display.
  3. **3.3.5 creature textures** — all 3.3.5 creature textures are missing; 0.5.3 works. Likely a
     texture-path or BLP-version resolution difference between the two clients.
- **PM4 open-question follow-up (2026-10-20):** `MSUR._0x10 = -dot(normal, v0)` in raw MSVT space
  (518,080/518,092, median error 0); `MPRR` first two records are ordered spatial bounds in a
  768×768 tile grid (3,171,133/3,171,912, 99.98%); `_0x1c` matches a placement Z anywhere in the
  map for 69.7% of nonzero surfaces; `MSHD` spans vs raw placement spans r = 0.39–0.58;
  `MPRL._0x14` is not the run index, node level, group size, link count, or run length. Receipt:
  [phase-5](../specs/279-decode-pm4-format/evidence/phase-5-open-questions-2026-10-20.md).
- **PM4 UI consolidation + matcher removal (2026-10-20, operator-directed):** PM4 now lives in the
  **Inspector window** as a tab (Inspector, World, Model, Settings, PM4). The tab exposes the eight
  PM4 pages (Overlay, Selection, Correlation, Info, Reconcile, Alignment, Outliner, Object Groups).
  The duplicate PM4 content in the Editor Placement page, Editor PM4 workspace, and Archaeology PM4
  sub-tab is replaced with a navigation shortcut. The hover matcher and the workbench object-match
  report are removed; matching now goes through the patched-ADT placement services (`Find WMO
  Match` / `Shape Search`). **Human curation:** `Pm4ObjectMatchStore` writes
  `output/pm4_object_matches.json`; the PM4 Selection tab has a Match this object section (path
  field, save/clear, saved-match list, Suggest nearest model), and the Rosetta object library has a
  Match to selected PM4 object button. Viewer build 0 errors.
- **Spec 279 (PM4 comprehension + reference):** 16 parsed chunks, 17 open question groups.
  [Reference](../docs/research/pm4-format-reference.md) - [spec](../specs/279-decode-pm4-format/spec.md).
  **2A** raw MSLK +0x01 = source-edge ordinal; **2B** +0x04 is not a unique edge ID; **MSHD**
  five of eight words always zero; **M2 partition** isolates M2-only surfaces but the per-object
  split is **negative** (ownership remains U); **MPRR** is a quadtree index over MPRL points;
  `MPRL.Unk04` = `MSLK.GroupObjectId` (98.97%); object-to-model join 85.1% within 25 units;
  MPRL points are not MSPV path vertices. Reference consolidated 2026-10-20; opt-in `modelPriors`
  added to `Pm4AssetMatchScorer`. Receipts: [2A](../specs/279-decode-pm4-format/evidence/phase-2a-edge-ordinal-2026-10-20.md),
  [2B](../specs/279-decode-pm4-format/evidence/phase-2b-edge-identity-2026-10-20.md),
  [MSHD](../specs/279-decode-pm4-format/evidence/phase-3-mshd-2026-10-20.md),
  [M2 partition](../specs/279-decode-pm4-format/evidence/phase-3-m2-partition-2026-10-20.md),
  [path nodes](../specs/279-decode-pm4-format/evidence/phase-3-path-nodes-2026-10-20.md),
  [model prior](../specs/279-decode-pm4-format/evidence/phase-4-model-prior-2026-10-20.md).
  Full detail: [archive/2026-10-21-pre-spec280-active-context.md](archive/2026-10-21-pre-spec280-active-context.md).
- **PM4 structural findings (2026-10-20):** 616 files probed; exact MSUR-to-MSLK windows; packed
  link target = tile X/Y + surface index; 15,773 cross-tile links; 32 focused tests pass.
  Unsolved: editor-authored path-node identity, remaining raw fields, M2 object partition.
  [Ledger](../specs/270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md).
- **Terrain Texture & Blending Fix (WoW 3.3.5 / Modern ADTs)**:
  - Completely resolved chunk-level checkerboarding, pitch black/dark green slopes, and corrupting layer textures across 3.3.5 maps (Dark Portal, Crystalsong, Dragonblight, Valgarde 18_23).
  - Fixed MCNK subchunk scanning desync in `Mcnk.cs`: `Header.SizeMcsh` is the raw payload (512 bytes), removed erroneous `- 8` which caused an 8-byte advance overrun into shadow data on 71 chunks; prioritized `Header.SizeMcal - 8` when declared size is 0, 4, or 512 to prevent overruns; adjusted walk loop to `while (remaining >= 8)` so 0-payload chunks (`MCSE`) don't break exact walk tracking.
  - Fixed format profile resolution: `FormatProfileRegistry.cs` now routes all `major == 3` clients to `AdtProfile33512340`; relaxed `TryParseBuild` for 2-4 parts; added `TryDetectLichKingAdt` to `StandardTerrainAdapter.cs` checking `MH2O`, `ofsMcnr == 0x520`, or `sizeMcal > 0`.
  - Restored sequential `mix()` layer blending with accumulated alpha tracking in `TerrainRenderer.cs` (strict ASCII).
  - Real-client validation on `Northrend_23_18.adt`: 256/256 chunks parsed with 0 errors; 6/6 `McnkSubchunkWalkTests` pass.
- **PM4 M2 Surface Correlation & Unauthored Placement**:
  - Added PM4 collision surface and M2 mesh correlation services (`Pm4SurfaceCorrelationExtractor.cs`, `Pm4SurfaceCorrelationMatcher.cs`, `Pm4UnauthoredPlacementResolver.cs`).
  - Added M2 collision and attachment metadata extraction in `M2ModelReader.cs` and `M2ModelDocument.cs`.
  - Integrated world dialogue and spawn streaming refinements.

**Older release and regression detail** (2026-10-12 to 2026-10-20) is archived in
[archive/2026-10-21-pre-spec280-active-context.md](archive/2026-10-21-pre-spec280-active-context.md):
the 2026-10-20 committed feature sets, the WC3 browser unification (Spec 272), the release
packaging fix, the ground-effect tint rollback, the 1.60.1 taxi mount fix, the open WMO area
lookup issue, the M2 texture-resolver rollback, and the M2 camera origin/facing work. Live
constraints from those notes: do not reintroduce tint propagation without a diagnosed approach;
do not refactor `ReadStringArray` without a narrowly scoped proven change; the WMO area lookup
and the `DD21` magic remain unexplained.

**Next on `v0.6.1-dev` (fresh chat start here):**
1. **PM4 next:** test raw +0x04/subtype and point/reference chunk relationships against the now
   correctly tile-addressed surface graph. Identify authored path-node layer independently; compare
   any proposed M2 ownership partition with Museum copies/scales. Large export remains operator-owned
   ([usage](../specs/270-pm4-object-association-ledger-and-placement/evidence/map-context-usage.md));
   lightweight whole-map structural summaries have already run and must not be described as unrun.
2. Operator runs the viewer on 1.60.1 inside `MD_CryptOneRm.wmo` and provides `[WMOAreaLookup]` diagnostics; do not implement a key/fallback change before reading the matching row IDs.
3. Operator confirms character M2s render again after the `ReplaceableTextureResolver` rollback; FileDataID texture support remains unresolved.
4. Operator verifies the actual 1.60.1 `CinematicCamera` row for FileDataID 116912 and plays/warmups FlybyUndead on base Azeroth with intended phase layers; record whether origin/facing and tile `(28,28)` agree.
5. Operator-requested public format docs: prepare wowdev.wiki-style articles for the 0.5.5 prototype ADT, WMO v14, MDX, Alpha WDT, and a substantial PM4/PD4 refresh. Ground each in repo evidence; distinguish verified facts from open interpretations.
6. Spec 271 Phase B — integrated Help + image guide (operator captures screenshots; agent prepares shot list and `HelpGuideService`).
7. Spec 270 open gap — objects on tiles adjacent to PM4-covered tiles that exist in museum data are still missing from restored tiles (operator report 2026-10-11).
8. Resolved: `Mcnk.cs` subchunk scanner desync fixed (MCSH raw size 512, MCAL header size fallback, and loop boundary). Alpha/shadow offsets now parse 100% cleanly without tool-local offset repairs.
9. Operator smoke owed on regenerated `output/restored_development` (alpha masks, no orange tint, 63_5 huts flush).

The repeatable procedure is now documented in [Development Map Restoration](../docs/WoWViewer/development-map-restoration.md);
that guide does not close the operator visual smoke item above. The README uses its native Markdown support matrix
instead of a separate graphic; worklog/spec and old release-note references were removed from viewer-facing documentation.

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — 7 active epics (248–254), all **triage pending**.
2. This compact handoff.
3. [TRIAGE.md](../specs/TRIAGE.md) — the operator's Want / Drop / Later decisions.
4. Only then one epic's `spec.md` → `plan.md` → `tasks.md`, and only the archived design documents
   its `plan.md` adopts for the selected item.

[Progress ledger](progress.md), [memory archive](archive/README.md) and `specs/archived/` are
on-demand history, never default reading.

## Current lanes — operator P1s (both marked Want 2026-09-23)

| Lane | Code | Owed (operator) | Next agent step |
|---|---|---|---|
| **R-10** modern-data lighting perf (Epic 249) | T002, T005, T006 landed 2026-09-23; **T004's per-placement WMO gate reverted 2026-09-27** (operator saw <1 FPS; T009) | check FPS after the revert on `wow_classic_beta` 1.60.1 `Azeroth` (0 WMO placements batched while lights exist); T003/T007 captures | redesign WMO instancing only after a before/after capture |
| **R-39** CASC minimap streaming cost (Epic 249) | code landed and **reverted** 2026-09-27 (operator: performance worse); code = `c2e9517` | say whether to add per-tile read/decode/upload timing + GC counts, then capture on a CASC client | no minimap change before measurement |
| **Spec 256** modern-data streaming & M2 submission | P0 `77f6f02`, minimap `b15e553`, P1 `4f1e6b4`; capture showed ≈ 57 ms/M2 in texture work → P2a shared textures `c9cf7b6`, P2b off-thread decode `6a326d3`, P2c compressed DXT + authored mips `bad4884`; P2b fallback fix `328866d`; missing textures → error texture, no search `983c8ac`; D WMO doodad loads + shared cache `b09654b`; P3 native M2 instancing `2c609f3`; Perf panel out of PM4 service `93d7461`; A animation: cached key frames + GPU skinning `e437673`; T060/T092 file splits landed (`WorldAssetManager.cs` 1,576 lines, `WmoRenderer.cs` 1,937 lines) | T102 capture (MdxAnimation, entering animated areas) + T042 visual A/B (Utilities > Perf > Frame history > Submission efficiency) | operator captures (T015, T022, T042, T081, T102) |
| **Spec 257** unified video recording & automation | **Complete 2026-10-02**: fixed taxi node stop-crash (null deref), unified `RecordingCoordinatorService` in `Recording/`, CLI video recording automation (`--record-taxi-route`, `--record-camera-path`, `--record-feature-tour`), camera path & feature tour integration | Operator smoke test of manual and automated recording | None (code & spec complete; receipt in `evidence/receipt-spec257.md`) |
| **Spec 258** taxi playlists & marketing showreel tour | **Complete 2026-10-02**: multi-route taxi playlists, continuous multi-segment recording, dynamic showreel telemetry HUD, large centered cinematic zone transition banners, landmark callouts; strict tour-mode viewport gating | Operator smoke test of playlist playback and showreel overlays | None (code & spec complete; receipt in `evidence/receipt-spec258.md`) |
| **Spec 273** ground-truth renderer performance | **Implemented & Verified 2026-10-16**: ground-truth 3.3.5a decompilation (H:\wowdecomps\3.3.5a\wowdecomp-main) refinements: WMO portal occlusion & interior culling (R-15), decoupled shell instancing (R-10), spatial M2 doodad submission, and terrain shader constant alignment; receipt in `evidence/receipt-spec273.md` | Operator in-viewer smoke owed | None (code complete) |
| **Spec 274** minimap hillshade correlation & domain translation | **Active 2026-10-16**: Batch-export clean paired dataset across Zarr v3 map revisions (real minimap, synthetic DXT1 hillshade, WDL lattice, ADT heightmap), train lightweight style-transfer network for object stripping and texture normalization, and correlate clean hillshade directly with 3D terrain implication data | Operator review of Spec 274 plan | Phase 1 implementation (T001–T004) |
| **Spec 259** ground effects & WMO detail doodads | **Complete 2026-10-02**: dynamic grass/foliage on terrain and WMO surfaces (`MDDL`), client slope ($Z \ge 0.4$) & MCCV rules from recent wowdev.wiki disclosures, 1.60 CASC support, and instanced GPU rendering; UI controls in RenderQuality; receipt in `evidence/receipt-spec259.md` | Operator visual smoke on terrain maps and WMO surfaces | None (code & spec complete) |
| **Spec 260** engine diagnostics & hitch telemetry HUD | **Complete 2026-10-02**: real-time engine pipeline metrics (M2 instancing %, flora count, WMO batches, ADT chunks, RAM & draw calls), rolling hitch detection with dominant stage attribution, flashing crimson/amber alert badge, multi-stage expanded timing panel, UI toggles, and CLI flags; receipt in `evidence/receipt-spec260.md` | Operator smoke test during recording and live viewport preview | None (code & spec complete) |
| **Spec 261** WoW Forever terrain hole fidelity | **Complete 2026-10-04**: verified ADT v18 format conformance, 64-bit high-res hole decoding at MCNK offset 0x14 with Little-Endian mapping (`cellY * 8 + cellX`), 100% seam boundary continuity, real CASC test cases for WoW Forever 1.60.1 / 11.2.7 (Deathknell church crypt and graves); receipt in `evidence/receipt-spec261.md` | Operator visual smoke on WoW Forever (1.60.1) Deathknell/caves/crypts | None (code & spec complete) |
| **Spec 262** minimap residual model & 3D fractal brush reconstruction | **Complete 2026-10-07**: Photometric Blinn-Phong specular solver, Rosetta catalog vision indexer, ComfyUI SAM 3.1 sieve & inpainting, closed-loop shadow difference $\Delta S$, discrete 3D fractal brush discovery & matching pursuit, and $\ge 75\%$ terrain reconstruction benchmark; operator smoke verified live on RTX 4070 Ti SUPER (`cuda:0`, 12.6 GB VRAM free, 2,201 segmented pixels, 97.37% geometric fidelity); receipt in `evidence/receipt-spec262.md` | Operator smoke test passed | None (code & spec complete; 28/28 Python + 218/218 C# tests green) |
| **Spec 263** 1.60 MCCV terrain shadow validation & synthesis | **Complete 2026-10-07**: empirical cross-validation between 1.12 bare minimap residual shadows and 1.60 MCCV vertex data (145 vertices BGRA: 580 bytes), geographic sampling stratification (Westfall/Elwynn/STV as Tier 1 baselines vs composite cities/Wetlands/Badlands as Tier 2 excluded), bidirectional ADT synthesis & injection bridge (`MccvTerrainShadowService.cs`, `mccv_shadow_comparator.py`, `v60_compare_mccv_residuals.py`); road-safe sieve preserving 2D alpha texture splats, labeled 4-panel diagnostic sheets with headers/legends, and 3D Wavefront OBJ + binary glTF 2.0 (`.glb`) mesh exports for side-by-side surface comparison in Windows 3D Viewer/Blender; verified NCC 0.9252 (Westfall) / 0.8778 (STV) and 100% / 87.5% ridge coincidence; receipt in `evidence/receipt-spec263.md` | Operator smoke verified | None (code & spec complete; all 6 C# + 11 Python tests green) |
| **Spec 264** authentic development shadow-to-height & Rosetta object placement | **In Progress (Active Dead End on Poisson SFS)**: Shape-from-shading / Poisson integration hits a mathematical dead end: land-only Pearson $r = 0.3345$, land MAE $= 31.74$ yds, max error $= 179.30$ yds (prior $0.94$ metric was skewed by flat $Z=0$ water plane). Pure shading integration cannot resolve macro-elevation or texture-albedo confusion. Next step: train an end-to-end multi-scale elevation network directly mapping authentic minimap RGB $\to$ absolute Z yards on the 198 development tiles without relying on test-time ground truth. | Operator review | Build deep elevation network on 198 authentic development tiles |
| **Spec 265** two-stage residual elevation refinement & scale calibration | **Active 2026-10-08**: Cascaded Stage 2 residual refiner predicting $\Delta Z = Z_{\text{gt}} - Z_{\text{initial}}$ to uncompress squashed mountain spans (39 yds $\to >200$ yds on `development_16_33`) and correct systematic texture/altitude offsets. Spec Kit authored (`specs/265-residual-elevation-refinement-and-scale-calibration/`); Phase 1 residual dataset extraction underway. | Operator review | Implement Phase 1 residual dataset caching |
| **Spec 266** WDL trestle elevation reconstruction & lineage synthesis | **Complete 2026-10-08**: Modern lean successor to Feb 2026 v7 WDL trestle model (4.7M params vs 100M+); synthesizes missing development WDL lattices from Northrend 3.0.1.8303 prototype matches (2,035 tiles transferred); restores authentic $>250$ yd mountain relief (measured 424.8 yds on `development_16_33`); 100% upward CCW mesh normals (+Z OBJ, +Y GLB); validation MAE 2.87 yds, $r=0.9943$. Fixed oversized sieve: replaced AABB bounding boxes with tight oriented OBB footprinting ($31\times$ reduction on doodads, 0 painted text masked, PM4 collision extraction disabled). Batch generated all 2,252 development LK ADTs with full multi-band SfS Poisson meso-relief, 220° solar normals, and ridge crest spines in 134.7s (16.7 tiles/sec) in `output/development_reconstructed_lk/` with updated `development.wdt`. Receipts in `specs/266-wdl-trestle-elevation-reconstruction/tasks.md`. | Operator smoke test | None (code & spec complete; 6/6 tests green) |
| **Spec 267** Trellis.2 / Pixel3D ComfyUI ADT refinement | **Active / Phases 1-3 Complete 2026-10-08**: Discovered running ComfyUI models (Trellis.2 / Pixel3D on RTX 4070 Ti SUPER, 2,933 nodes cataloged); created `comfyui_parp_nodes` package with `WoW_AdtLoader`, `WoW_ObjectSieveConditioner`, `WoW_HardZCalibrator`, `WoW_MeshExporter` (5/5 tests green). **Complete Chunk & Data Transfer**: Monolithic 3.3.5 ADT preservation for all 2,252 tiles — keeps 100% of authentic chunks (`MTEX`, `MCLY`, `MCAL`, `MMDX`, `MWMO`, `MDDF`, `MODF`, `MCRF`, `MCSE`, `MCSH`, `MCLQ`/`MH2O`, `MFBO`, `MCCV`). Cleaned 447 stray `_obj0`/`_tex0` companion files from output root, enforcing `AdtProfile_335` so WoWViewer loads all textures and objects. **Tileset Texture Stripping & De-Bumping**: Decomposed minimap luminance into structural illumination ($\sigma=4.5$) vs tileset texture noise ($T = I - S$); gated ridge detection to genuine macro relief ($|\nabla Z| \ge 0.08$); smoothed meso SfS relief ($\sigma=2.0$, $1.0$ yd amplitude) and wrote smooth surface normals into `MCNR`. Batch generated all 2,252 monolithic ADTs in 791.0s with updated `development.wdt`. | Operator reload & visual smoke test in WoWViewer | Complete Phase 4 end-to-end ComfyUI workflow JSON |
| **Spec 268** quilt canvas terrain reconstruction & MCAL deciphering | **Phases 1-5 Complete 2026-10-09**: Continuous multi-tile quilt canvas stitching, 1px shared tie-edge boundary solver (C0/C1, $\Delta Z = 0.0000$ yds, normal alignment $0.9995$), minimap albedo de-mixing & dynamic MCAL layer stacking ($\le 4$ per chunk), authentic WDL global macro-trestle lattice interpolation, 36x inches-scale 3D fractal refiner (8 brush stamps + scars), monolithic LK ADT materialization, and textured 3D OBJ/GLB mesh exports; verified on alpine massif `(16, 32)`/`(16, 33)`, 2x2 cluster, and unauthored minimap-only tiles `(17, 13)`/`(17, 14)` via pure V7 foundation; all post-V7 models purged; receipt in `evidence/receipt-spec268.md`. | Operator smoke passed | Complete (2026-10-09) |
| [**Spec 269** v60 classic terrain Zarr ingestion & Felwood clean elevation training](../specs/269-v60-classic-terrain-ingestion-and-elevation-dataset/spec.md) | **Complete 2026-10-10**: Ingest Wrath 3.0.1.8303 terrain (37 maps, 4,448 tiles) via C# harvest engine into `output/datasets/v60/`, train `TrestleElevationUNet` on authentic Felwood canyons/indentations (120 epochs, MAE 4.62 yds, $r=0.8842$), and carve $-96.13$ yd canyon into unauthored quilt tile `(17, 14)`. Receipt in `evidence/receipt-spec269.md`. | Operator harvest & training runs | Complete (2026-10-10) |
| [**Spec 270** PM4 object association ledger & placement pipeline](../specs/270-pm4-object-association-ledger-and-placement/spec.md) | **Complete 2026-10-11**: Harvested 1,929 ground-truth objects into Rosetta Ledger (`pm4_object_association_ledger.json`). **Phase 4 & 5 Monolithic LK Pipeline**: Strict barrier preserving 255 hand-authored Blizzard tiles; 2D PCA Yaw; repaired Schlumpf index defect; restored unauthored PM4 tiles. **Phase 5 Multi-Source Placement Convergence & Magnetic Collision Fix**: Fixed model pivot origin vs bounding centroid Z offset in Strategy 0 and vertical attraction; preserved authentic authored rotations over synthetic PCA; deduplicated stacked instances with proximity gating (20 yds WMO, 5 yds M2); sampled 256x256 minimap PNGs into 256 145-vertex MCCV chunks via `AdtMccvImageSampler` (195 untextured tiles colored, 0 black tiles). Full map batch executed: 616 tiles processed, 1,654 WMOs, 64,301 M2s, 616 PASS / 0 FAIL round-trip. | Operator review & visual smoke in WoWViewer | Complete (2026-10-11) |
| **U-01** god-class decomposition (Epic 251) | **E1 + E3 + ViewerApp campaign landed 2026-09-25**: PM4 overlay → `Terrain/Pm4/` (`WorldScene.cs` 17,175 → 8,326); `ViewerApp` class 42,973 lines / 29 files → 2,512 / 6 (51 services + 3 static helpers under `Workbench/Services/`) | U01-T003 E1 smoke; U01-T007 / T013 smoke of every moved ViewerApp surface; U01-T009 E4 hover/click smoke; **Spec 255 T018** smoke of W0–W7 | E4 landed 2026-09-26; **Spec 255 W0–W7 landed 2026-09-27** (`WorldScene.cs` 8,275 → 4,922; 7 scene services + PM4 helpers). W8/W9 and E2 `Render()` split wait for R-10's after-capture |

Receipts: [E1](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-e1-pm4-extraction-2026-09-25.md) ·
[ViewerApp campaign](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-viewerapp-extraction-2026-09-25.md) ·
[Spec 255 W0–W7](../specs/255-worldscene-decomposition/evidence/w0-w7-extraction-2026-09-27.md) ·
[E4 selection](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-e4-selection-2026-09-26.md).
**Where code lives now:** viewer features are owned services under `src/viewer/WoWViewer/Workbench/Services/<Feature>/`
(namespace `WoWViewer`); they reach app state only through `IViewerAppHost` (implemented in
`ViewerApp_Host.cs`). `ViewerApp.cs` is the shell: fields, composition root, window lifecycle. New behaviour
goes into the owning service; if it needs more app state, add one `IViewerAppHost` member + its one-line
implementation. PM4 overlay code lives in `Terrain/Pm4/` (`WorldScene.Pm4Overlay.X`). World-scene features live in
`Terrain/Scene/<Feature>/` services (`WorldScene.HoverPick`, `.TaxiActors`, `.ObjectFilters`, `.Selection`, `.Atmosphere`,
`.ExternalSpawns`, `.TerrainQueries`); they read scene state only through `IWorldSceneHost` (explicit impls in `WorldScene.cs`).

All other epic items remain untriaged in [TRIAGE.md](../specs/TRIAGE.md); nothing else is scheduled.
Receipt for the 2026-09-23 reconciliation:
[reconciliation README](../specs/archived/reconciliation-2026-09-23/README.md).

**Linux build note (agent sessions):** the viewer builds and tests on Linux with
`-p:EnableWindowsTargeting=true` after `git submodule update --init` of the six `wow-viewer/libs/*`
submodules. 26 tests fail on `HEAD` there for environmental reasons (missing `test_data/`,
Windows-path expectations) — compare against that set, not zero. One Curation test rewrites
`data-harvester/tests/fixtures/spec122_curation_manifest/.../curation_run.json` with the local path;
revert it before staging.

## Release state

`eng/Version.props` is restored to `0.6.1` / `0.6.1-dev` for active development. Stable tags: replacement `v0.6.0.1` at `5ba5eb2a` and base `v0.6.0` (2026-10-11).

**Shipped unverified in alpha2** (operator verification, tracked as V-tasks in the epics): the M2
texture-wrap fix (every M2 era); `LkAdtWriter` chunk-completeness fixes (17 call sites); no exported
map loaded in a client or Noggit; no DAT Cartography layer composed on screen; new export menu and
sidebar buttons never clicked.

## Measured gaps worth knowing before any work (details: TRIAGE §1)

- No map save pipeline exists (`MapSaveService` absent; `EditorSession.SaveAll()` writes nothing).
- Editor undo reverses only 2 of 5+ operation kinds.
- Zone music is disabled by policy and still reads `ZoneMusic` as a SoundEntries id.
- Modern data: the scene-light gate disables WMO instancing (~5.5 FPS, 16,431 WMO draws pre-R-10); R-10's per-placement gate made it worse (<1 FPS, operator) and was reverted 2026-09-27.
- v22 DAT blends layer 0 only; `AMAP` codec unidentified.
- `WorldScene.cs` 4,922 lines (after Spec 255 W0–W7, 2026-09-27), `ViewerApp` class 2,512 lines in 6 files — no new members (AGENTS.md §10).

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video and client-data proof are operator-owned.
- Training, harvests, GPU and cloud runs are operator-run; hand over PowerShell-ready commands.
- Preserve unrelated dirty work (`imgui.ini`); stage named files only.
- New scope needs operator-approved wording (AGENTS.md §9.1); receipts per §9.2.

## Handoff

**Immediate:** operator runs R10-T003/T007 captures and the U-01 smoke passes (U01-T003 PM4 overlay;
U01-T007/T013 every moved ViewerApp surface — checklist in the campaign receipt). U01-T009 E4 hover/click smoke and
[Spec 255](../specs/255-worldscene-decomposition/spec.md) T018 (W0–W7 smoke — checklist in its
[receipt](../specs/255-worldscene-decomposition/evidence/w0-w7-extraction-2026-09-27.md)) are owed too.
Next agent-owned U-01 work: Spec 255 W8/W9 and E2, after R-10's after-capture (R10-T007). Do not start any other epic
item before it is marked Want. Superseded dashboard:
[archive/2026-09-23-pre-reconciliation-active-context.md](archive/2026-09-23-pre-reconciliation-active-context.md).
