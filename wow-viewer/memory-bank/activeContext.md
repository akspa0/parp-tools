# Active Context — wow-viewer

Last updated: 2026-10-20 · **Terrain MCAL/MCSH Parsing & PM4 Surface Correlation**:
- **Spec 279 (operator: full PM4 comprehension + readable writeup, Spec Kit):**
  [Current PM4 reference](../docs/research/pm4-format-reference.md) covers 16 parsed chunks,
  raw fields/evidence levels and 17 unresolved question groups. Spec/plan/tasks/checklist on
  `279-decode-pm4-format`. **2A:** raw MSLK +0x01 is a source-edge ordinal (674,013 boundary matches
  vs shifted control 137). **2B:** +0x04 is not a unique edge ID (889,594 reused values); type-1
  reciprocal records share it 98.92%, type 17 100%. **MSHD:** five of eight words always zero;
  Field00/08 correlate with Mdbh ~0.76–0.79. **M2 partition found:** authored `_obj0.adt` placements
  show `Ck24Type == 0` / `GroupKey == 3` selects 14,489 M2-only surfaces vs 57 WMO-only; WMO is
  64–67 / 16–19. The zero bucket is the M2 class, not a per-object identity (only 12.8% of zero
  surfaces cover an authored M2 origin). Receipts: [2A](../specs/279-decode-pm4-format/evidence/phase-2a-edge-ordinal-2026-10-20.md),
  [2B](../specs/279-decode-pm4-format/evidence/phase-2b-edge-identity-2026-10-20.md),
  [MSHD](../specs/279-decode-pm4-format/evidence/phase-3-mshd-2026-10-20.md),
  [placements](../specs/279-decode-pm4-format/evidence/phase-3-placement-ground-truth-2026-10-20.md).
  **M2 per-object split is negative:** topology, vertex ranges, MSUR order, Padding, +0x18, Normal,
  Height and GroupObjectId all interleave objects (2,000 mixed-model components; 6,576 placements
  share vertices; 9,570 overlapping MSUR ranges; Padding always 0; 7,315 placements multi-GroupObjectId).
  [Receipt](../specs/279-decode-pm4-format/evidence/phase-3-m2-partition-2026-10-20.md). Per-object M2
  ownership remains U. **Path layer:** MPRL 178,588 points, 99.4% inside the mesh, `Unk02 = −1` and
  `Unk06 = 0x8000` constant; MPRR 17,150,640 records at 4 bytes each. **MPRR is a quadtree spatial
  index of the MPRL points:** each run is a node (first two records = bounds, middle = MPRL indices,
  last = terminator), and the whole-map node width/height is 1/2/4/8/16/32 (powers of two). The
  `4k+3` run lengths and ~4:1 decay are the quadtree hierarchy. **Scene-graph hierarchy:** `Unk00`
  (region, 265 values) → `Unk04` (object, 42,092 values; every group has exactly one `Unk00`) →
  `Unk14`/`Unk16` (per-point type/flags). **Object partition found:** `MPRL.Unk04` is
  `MSLK.GroupObjectId` (+0x04) for 176,748/178,588 points (98.97%), and 98.8% of MPRR runs group
  MPRL points sharing one `Unk04` (100% share one `Unk00`). The id is a group/model id, not a unique
  placement id. The direct group→surface link is not containment (0.4%); the points sit on M2
  surfaces (66.7%), WMO surfaces (29.0%) and other surfaces (6.7%). **Object→model join:** the
  nearest-placement join covers every group — 32.2% within 5 units, 59.7% within 10, 85.1% within
  25. **Class-consistency precision:** 19,991 groups (68.5%) have a consistent match (≤25 units and
  model class matches ground-surface class), 5,363 (18.4%) are class-mismatched, 3,662 (12.6%) have
  no match within 25. **Ranked matcher:** 895 models; top are `WINTERSPRINGROCK01.M2` (931 groups),
  `SHADOWMOON_OVERHANGROCK_SMALL_02.M2` (753), `STONETREELOG02.M2` (618). **Viewer:** new
  Experimental > PM4 > Object Groups tab lists the selected tile's groups with filter,
  consistent-only and camera focus. **MPRL points are not MSPV path vertices:** only 0.8% within 0.1
  and 13.1% within 1.0; the path geometry is the separate MSPV/MSPI stream.
  [Receipt](../specs/279-decode-pm4-format/evidence/phase-3-path-nodes-2026-10-20.md). Editor-node
  identity remains unknown. Operator wants checkpoints committed. **Reference consolidated
  (2026-10-20):** [pm4-format-reference.md](../docs/research/pm4-format-reference.md) rewritten top to
  bottom as a standalone reader/renderer specification using the current wowdev.wiki field names
  (`PM4`, `PD4`) as primary names; proposed names live only in §9. No commentary about earlier names
  remains. **Opt-in model prior:** `Pm4AssetMatchScorer` accepts an optional
  `modelPriors` map (exact path, then filename, clamped 0–1) and adds at most +0.10 to the overall
  score; `priorBonus` is 0 and the default path is unchanged when no map is supplied. 5/5 scorer
  tests pass, including `ScoreSegment_ModelPriorBreaksTieOnlyWhenSupplied`.
  [Receipt](../specs/279-decode-pm4-format/evidence/phase-4-model-prior-2026-10-20.md).
- **PM4 structural findings (2026-10-20, newest):** summary probes read all **616 files**; exact
  MSUR→MSLK start/count windows; packed link target is tile X=low byte, Y=next byte + surface index.
  **Not universally file-local:** 15,773 cross-tile links; 14,659 resolve. Same-tile addressed links
  are 1,257,562/1,257,562 reciprocal. Paired wrong-local comparison: 9,748 vs 15 back-links on
  10,222 identical records. Raw 0x01/0x11 links behave as boundary adjacency; other kinds carry
  connection geometry. **32 focused tests pass.** Research audit now retains addressed hypotheses.
  [Measured/refuted/unknown ledger](../specs/270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md).
  **Unsolved:** editor-authored path-node chunk identity, remaining raw fields, M2 object partition.
  Screenshot-reported NPC route nodes are a separate hypothesis, not automatically MPRL/MSCN.
  Existing readers/matchers/viewer untouched; ~70% WMO accuracy remains operator-reported.
  Prior [instruments/pools](../specs/270-pm4-object-association-ledger-and-placement/evidence/map-context-instrument-2026-10-20.md)
  remain available; large Museum/raw export, viewer pool UI and WL footprint mapping are still open.
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

**Committed Feature Sets on `v0.6.1-dev` (2026-10-20)**:
1. `fix(terrain,pm4)`: Resolved MCNK MCAL/MCSH subchunk parsing desynchronization, 3.3.5 ADT format profile auto-detection, authentic sequential mix terrain blending, and PM4 M2 surface correlation pipeline.
2. `feat(archaeology,spec-277)`: UniqueId Stratigraphy & Cross-Map Chronological Fork Analysis. Unified `UniqueIdRecord` contract, quantile calculation ($p_{10}..p_{90}$), adaptive layer clustering, spatial hash twin placement matching, divergence boundary ($UID_{div}$) detection, CLI profiler (`map stratigraphy`), and interactive in-viewer Archaeology Time Machine slider. 5/5 focused unit tests pass; CLI divergence report verified on Map 451 vs Map 0 ($UID_{div} = 400111$, 100% matched overworld branch).
3. `perf(population,viewer)`: Resolved 0.5.3 MDX/WMO loading regression. Made SQL spawn auto-streaming opt-in (`_sqlAutoLoadOnMapChange = false`), switched partition loading to non-blocking background `Task.Run`, deduplicated applied spawn sets to avoid spurious scene-graph invalidations on camera movement, replaced synchronous WMO disk parsing with cached mesh summary / MOHD bounds in `ExternalSpawnLayer` and `WorldAssetManager.TryGetWmoPlacementBounds`, and added immediate `.mdx`/`.mdl` priority resolution for classic clients in `WorldAssetPathResolver`.
4. `feat(population,spec-276)`: TrinityCore multi-version SQL population extraction pipeline, full multi-map streaming (all 113 maps in 3.3.5a, all 31 maps in 0.5.3), automatic client-version detection & map loading, Map 451 developer test spawns + alpha-core corner dev transposition, authentic NPC/GameObject nameplates and Info tab inspector display, default Stand/Idle animation selection, and ground-truth 4.3.4 detail doodad alpha blending. 12/12 focused tests pass.
5. `feat(harvester,spec-274)`: Minimap Hillshade Correlation & Domain Translation dataset pipeline across 37 Zarr v3 map revisions; 228/228 pytest tests pass.
6. `feat(inspector)`: Integrated 4.3.4 client ground-truth format flags inspection (`FormatFlagsDescriber.cs`) across Terrain, Object, Model, and Standalone inspection panels.
7. `perf(renderer)`: Optimized spatial M2 lighting collection, terrain streaming MCNK FourCC fast-path, and asset path resolver negative caching.
8. `feat(renderer,spec-273)`: Completed Spec 273 performance pipeline with fail-safe WMO frustum preservation; 48/48 portal tests pass.

**WC3 file support unified in main browser (2026-10-15):** [Spec 272](../specs/272-wc2-wc3-viewer-support/spec.md) WC3 loose folders and MPQs now load as first-class `IDataSource` instances directly into the main File Browser (`Wc3DataSource`). Textures (BLP, BLP0 with external mips, TGA, PNG) render directly in the main viewer scene viewport (`TextureViewRenderer`) with transparency checkerboard, pan, and zoom, instead of cramped sidebar previews. Models (`.mdx`) render through the main MDX viewport renderer. Arbitrary text/binary files can be browsed with the new "All files" (`*`) filter. Multiple BLP0 mip levels (`.b00` through `.b15`) are grouped into a single `<texture>.blp [N mips]` row with level 0 as the full-resolution primary. Added PNG batch/single export via `TextureExportService` with an option to write all companion mip levels. Viewer build passes 0 errors; focused core tests pass 70/70.

**Immediate release fix:** the release workflow previously published only the viewer `.csproj`. All nine `wow-viewer/tools/**/*.csproj` projects now publish into one shared `publish/<rid>/tools/` directory inside the platform's main build tree; common DBDs, listfiles, runtime DLLs and libraries are emitted once. Viewer and CLI tools are separate ZIPs. `V22Enrich` and `WmoMinimap` are registered in `WowViewer.slnx`. The packaging matrix passed in `38506799386`; the final baseline replacement passed in `38512662293` and release `38513490254`. The replacement is eight assets. Downloaded Windows tool bundle verification from the first split release found nine apphosts, 1,342 DBDs, one common `WowViewer.Core.dll`, and Inspector `--help` exit 0.

**Ground-effect regression response:** the per-instance tint/API changes were removed after the operator reported the viewer could no longer load anything. The affected files match the pre-tint baseline; no crash log or stack trace was available, so the runtime cause is not independently confirmed. Do not reintroduce tint propagation until a separate diagnosed approach exists. The replacement release uses the pre-tint renderer path; operator must confirm a real client/map loads.

**Immediate taxi mount issue (1.60.1):** the gryphon never loaded because
[`TaxiPathLoader.LoadCreatureModelPaths`](wow-viewer/src/viewer/WoWViewer/Terrain/TaxiPathLoader.cs:231)
read only `ModelName`/`ModelPath` — absent from 6.0.0 — so every mount resolved to null and every
actor fell onto the hardcoded `Creature\Gryphon\Gryphon.mdx` default. Fixed with the same both-era
pattern: string first, else `FileDataID` → listfile. The default candidate list now also probes the
`.m2` variant of each legacy `.mdx` path. 3 new tests; 18/18 focused tests pass.

**Immediate WMO area issue (open):** inside `MD_CryptOneRm.wmo` (header UniqueId 85944) in Tirisfal,
the status bar reads `Area: Crypt84 [Riverglades]`. The 1.60.1 DBD layout declares a relation
`WMOID<u16>` and has no `InlineWMOID`; it does **not** establish that this relation shares the
header UniqueId's numeric domain. The global `WMOGroupID` fallback can collide across WMOs, but the
bad row/match is not yet confirmed. A temporary log probe was removed: operator reports the viewer
log is not selectable/exportable and does not provide usable evidence. The operator's hypothesis is
a client-data remap/copy-link error; treat it as plausible, not proven. Next proof must be available
in-viewer or extracted directly from the relevant client rows.

**M2 texture resolver status (2026-10-12):** the broad FileDataID attempt and ItemDisplayInfo
resource-chain changes were reverted after the operator reported that many character M2 textures
stopped rendering. `ReplaceableTextureResolver.cs` is restored to the proven baseline; its new tests
were removed. No FileDataID correction for creature textures or item-display texture chains is
currently active. Operator must confirm character M2s load again after the rollback. **Do not refactor
`ReadStringArray` or its call sites again** without a narrowly scoped, proven change.

**Immediate M2 camera issue:** M2/MDX camera keys are tile-local until their matching
`CinematicCamera` origin/facing is applied. Warm Path's 64×64 grid spacing was corrected from
`TileSize` (8,533 units) to `ChunkSize` (533.333 units); FlybyUndead's documented origin maps to
tile `(28,28)`. Modern 1.60.1 `CinematicCamera` rows have `FileDataID`, not `Model`; the resolver now
matches the selected CASC asset's ID (community listfile: FlybyUndead is 116912). If lookup fails,
tracks remain local and playback, key captures, and warmup reject them. A missed guard was found in
plain Play and fixed. Focused camera tests pass (12/12), including local-origin-to-world-tile mapping.
Phase layers merge donor content onto base-map target coordinates; camera binding still uses the
base `TerrainManager.MapName`, so there is no code evidence that phase composition itself relocates
the path. Operator must verify the actual client DB2 row/origin and visual playback on
`wow_classic_beta` 1.60.1/Azeroth. Separately, the earlier reported `DD21` (`0x31324444`) magic is
still unexplained; exact bytes remain unavailable, so do not widen the reader based on that report.

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
