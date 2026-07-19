# Tasks: Viewer Stabilization

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), [contract](contracts/viewer-stabilization-contract.md), and
[quickstart.md](quickstart.md)

## Dependencies

```text
US1 Fog/terrain visibility and LIT inspection
  -> US6 Terrain-derived minimap export
  -> US7 Visible fog and Archeology controls
  -> US2 Native M2 routes
  -> US3 Tools menu/current tool entry points
  -> US4 Explicit conversion capability publication
```

US1, US6, and US7 are one revised Phase 1 stabilization slice. The remaining stories are planned
but blocked until this slice has focused test/build and user visual/export proof.

## Phase 1: Setup

- [x] T001 Record the active viewer-stabilization route and contract links in `specs/110-viewer-stabilization/plan.md`

## Phase 2: Foundational fog contract

- [x] T002 Add a finite, non-zero fog-range normalizer in `src/core/WowViewer.Core/Terrain/TerrainLightingMath.cs`
- [x] T003 [P] Add invalid, equal, reversed, and valid-range tests in `tests/WowViewer.Core.Tests/TerrainLightingMathTests.cs`

## Phase 3: User Story 1 - Keep every loaded map visible (Priority: P1)

**Goal**: Lighting recommendations never hide terrain, while a user can activate/reset a visible
fog range from the Lighting surface.

**Independent Test**: Focused Core tests demonstrate safe ranges; the Debug build has no errors;
the user can test one LIT and one no-LIT map with the quickstart.

- [x] T004 [US1] Add user fog override state, source reporting, and active-range resolution in `src/viewer/WoWViewer/Terrain/WorldScene.cs`
- [x] T005 [US1] Route all LIT/DBC/no-source outputs through the active range before terrain, WDL, object, and shader consumers in `src/viewer/WoWViewer/Terrain/WorldScene.cs`
- [x] T006 [US1] Add active Fog Start/Fog End controls, override reset, and source/status text in `src/viewer/WoWViewer/ViewerApp_Lighting.cs`
- [x] T007 [US1] Clarify settings as load defaults rather than active override controls in `src/viewer/WoWViewer/ViewerApp_Settings.cs`
- [x] T008 [US1] Add opt-in LIT minimap marker state and lazy-load behavior in `src/viewer/WoWViewer/Terrain/WorldScene.cs`
- [x] T009 [US1] Render selected and unselected LIT entry markers through the shared minimap surface in `src/viewer/WoWViewer/MinimapHelpers.cs`
- [x] T010 [US1] Select LIT markers from both minimap modes and focus navigable entries safely in `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs`
- [x] T011 [US1] Add virtualized LIT entry list, marker toggle, selection, and double-click camera focus in `src/viewer/WoWViewer/ViewerApp_Lighting.cs`
- [x] T012 [US1] Run focused fog tests and the Debug Viewer build documented in `specs/110-viewer-stabilization/quickstart.md`

## Phase 3b: User Story 6 - Export terrain-derived minimaps (Priority: P1)

**Goal**: Make a terrain-only, time-of-day-aware minimap for maps that have terrain but no usable
client minimap asset, with one direct Core.IO synthesis path for individual tiles and a combined map.

**Independent Test**: Focused Core.IO tests prove renderer-equivalent MCAL/MCLY composition,
projection/minification, lighting, and transparent-hole stitching; a user-run real-client export
records its build and lighting provenance.

- [x] T013 [US6] Add the reusable weighted terrain compositor and explicit lighting input in `src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs`
- [x] T014 [US6] Add the reusable transparent-hole stitcher and explicit bounds/result contract in `src/core/WowViewer.Core.IO/Maps/TerrainMinimapStitcher.cs`
- [x] T015 [P] [US6] Add compositor and stitcher coverage in `tests/WowViewer.Core.Tests/TerrainMinimapCompositorTests.cs`
- [x] T016 [US6] Replace the Harvest `synthetic-minimap` stub with client-map, per-tile, whole-map, manifest, and explicit lighting-provenance support in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`; T027q later fixes that provenance to the all-era noon-white contract.
- [x] T017 [US6] Add an in-repository-resolving Tools > Export dialog in `src/viewer/WoWViewer/ViewerApp.cs` and `src/viewer/WoWViewer/ViewerApp_SynthesizedMinimapExport.cs`
- [x] T018 [US1] [US6] Run focused fog/minimap tests and the Debug Viewer/Harvest build documented in `specs/110-viewer-stabilization/quickstart.md`
- [ ] T019 [US1] [US6] [US7] Run the two-map LIT/no-LIT time-of-day, overlay, synthesized-minimap, visible-fog-slider, and Archeology playback visual proof in `specs/110-viewer-stabilization/quickstart.md`

## Phase 3c: User Story 7 - Keep interactive controls visible and reachable (Priority: P1)

**Goal**: Fog is visibly draggable, while UniqueId range playback is owned by Archeology and can
always be paused or stopped.

**Independent Test**: Build the Viewer and use the quickstart to verify visible fog grabs plus
Tools > Archeology nested-tab and playback transport behavior in both UI modes.

- [x] T020 [US7] Replace drag-only fog controls with visible sliders in `src/viewer/WoWViewer/ViewerApp_Lighting.cs` and `src/viewer/WoWViewer/ViewerApp_Settings.cs`
- [x] T021 [US7] Give Archeology independent nested-tab state and remove the duplicate World controls in `src/viewer/WoWViewer/ViewerApp.cs`, `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`, and `src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs`
- [x] T022 [US7] Keep playback transport reachable while active and stop it safely on unavailable world/range state in `src/viewer/WoWViewer/ViewerApp.cs` and `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`
- [x] T023 [US7] Build the Viewer without replacing a running user app in `specs/110-viewer-stabilization/quickstart.md`

## Phase 3d: User Story 6 - Correct terrain minimap fidelity (Priority: P1)

**Goal**: Synthesized terrain minimaps preserve the active terrain renderer's ordered MCAL blend
contract while using phase-independent material averaging instead of producing static, moire, or
interpolation artifacts.

**Independent Test**: Focused Core.IO tests prove material averaging is invariant under texture
phase, overlapping-overlay order, and high-frequency input; the Harvest build succeeds without an
external binary.

- [x] T024 [US6] Replace renderer-UV texture projection with cached BLP material averaging in `src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs`
- [x] T025 [US6] Replace normalized MCAL layer weights with ordered overlay composition; normalize Alpha MCLY to row-major tensor-pack coordinates and honor MCLY layer presence in `src/core/WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs` and `src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs`
- [x] T026 [P] [US6] Add material-average, overlay-order, phase-invariance, Alpha coordinate-layout, and absent-layer regression coverage in `tests/WowViewer.Core.Tests/`
- [x] T027 [US6] Make `--limit` count emitted minimap PNGs instead of skipped WDT candidates; keep MCSH out of normal synthesized RGB while preserving it as a separate target; compose a readable missing-MCAL tile as base-only, bound-check partial MCNR masks, and recover a missing terrain BLP from a verified same-stem `_s` companion or a decoded related diffuse candidate scanned from the archive/listfile catalog; rank exact/strong basename matches before directory-theme similarity so moved historical assets can repair stale ADT links; retain MTEX identity/proxy metadata and apply the same order in `src/viewer/WoWViewer/Terrain/TerrainRenderer.cs`; emit conservative authored-minimap tint/MCSH/time-bucket provenance in full/V22 streams with name-aligned MTEX sidecars; run the focused minimap/serializer suite and Debug Harvest/Viewer builds; retain real-client visual re-export in T019
- [x] T027a [US6] Add shared minute-precise export-time parsing: accept `HHmm`, `HH:mm`, and legacy decimal-hour CLI inputs; record canonical clock plus decimal hours in the manifest; expose exact hour/minute fields in the viewer export dialog; add Core parser coverage and build Harvest/Viewer.
- [x] T027b [US6] Keep Alpha WDT `MAIN` enumeration row-major (`tileY * 64 + tileX`) in `WdtTileIndexReader`, matching `AlphaWdtReader`; add an asymmetric 16-byte-MAIN regression test so sparse Alpha maps do not generate transposed false decode skips.
- [x] T027c [US6] Replace the fixed-bias authored solar direction with a shared terrain/raster-axis path and include the failed pipeline stage in tile manifest diagnostics; this is the synthesized-minimap direction contract.
- [x] T027d [US6] Emit aligned terrain-only and `_liquid` per-tile/whole-map artifacts from decoded unified liquid coverage and basic types; normalize Alpha's 16×16 liquid-type grid to its 257² MCLQ surface before unified composition; record liquid paths/pixel counts/render profile and add focused Alpha/liquid-compositor proof.
- [x] T027e [US6] Respect Alpha MCLQ 8×8 liquid-cell visibility when deriving unified coverage, rasterize minimap liquid only for four-corner-complete source cells, and add `synthetic-minimap --tile-x/--tile-y` plus the first relevant source frame to residual failed-tile diagnostics in `src/core/WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs`, `src/core/WowViewer.Core.IO/Maps/TerrainMinimapLiquidCompositor.cs`, `tools/harvest/WowViewer.Tool.Harvest/Program.cs`, and focused tests.
- [x] T027f [US6] Clip Alpha WMO/object footprint painters to their actual buffer dimensions so 256² roof masks cannot fail at cross-tile edge placements; add deterministic decoded catalog-RGB last-resort recovery for stale/missing MTEX paths and missing material grids in `AlphaTensorPackBuilder.cs`, `TerrainMinimapCompositor.cs`, `TerrainTextureFallbackPolicy.cs`, Harvest, and the live terrain renderer, with focused regression coverage and manifest provenance.
- [x] T027g [US6] Preserve Alpha MCLQ's visible per-cell liquid type nibbles through the tensor pack, with MCNK type flags as the fallback only, so synthesized liquid companions distinguish water, ocean, magma, and slime. Add focused cell-precedence and Alpha round-trip type coverage.

## Phase 3e: User Story 6 - Empty-tile, native-light, and WL* correction (Priority: P1)

**Goal**: Preserve declared empty terrain without fabricating material color, use a north/top-edge
white minimap light independent of world LIT, and make WL* liquid continuous.

- [x] T027h [US6] Treat a tile with no non-empty MTEX names as an unlit solid-white empty tile in the Core compositor and Harvest; preserve catalog recovery only for named but unresolved materials, with focused proof under non-neutral lighting.
- [x] T027i [US6] Correct the shared authored solar vector so it cannot source from the raster south/bottom half.
- [x] T027j [US6] Correct raw MCLQ cell decoding and Alpha writer round-trip mapping: `0x01=Ocean`, `0x03=Slime`, `0x04=River/Water`, and `0x06=Magma`; prove that a visible river overrides a slime MCNK fallback and remains blue.
- [x] T027k [US6] Preserve the recovered 0.5.3.3368 native world-light ray as diagnostic research, but remove it and LIT colors from synthesized-minimap input. Use pure-white north/top-edge terrain lighting and prove the source cannot enter positive terrain X.
- [x] T027l [US6] Replace sparse WL* origin/vertex stamps with a shared world-geometry triangle rasterizer for all nine 4x4-block quads in both loose and archive-backed liquid paths; clip each raster sample against aligned terrain height; resolve WLW/WLQ header and WLM/WLL family liquid types into `LiquidBasicType257`; stamp contiguous-surface, above-terrain, and typed provenance markers; and make V16/V18/V50 builders reject any incomplete WL fallback so historic checkerboard, through-terrain, or default-water masks cannot enter a liquid-aware dataset.
- [x] T027m [US6] Correct the shared authored solar vector's cardinal sign using the traced 1.0.0 world-light ghidra proof (`docs/architecture/wow-1.0.0-world-lighting-shadow-model-2026-07-15.md`): raw MCNR/MCVT world axes are +X = North, +Y = West, +Z = Up (`AdtTensorPackBuilder.AssembleNormals` applies no axis swap; `TerrainMeshBuilder` derives vertex world-X from row/tileY-indexed quantities that decrease southward), so the north-locked horizontal bias must be positive X, not negative. T027i/T027k had locked the source to negative X and mislabeled it "raster north," which actually sourced the sun from the south and inverted hillshade relief. Update `TerrainSolarDirection`, its shared-consumer doc comments, and the compositor test/spec/contract wording that encoded the inverted claim.
- [x] T027n [US6] Stop sweeping the authored solar bearing through zero at solar noon/midnight. A user-run side-by-side of a synthesized tile against the real 0.5.3 client minimap for the same crater/lake feature showed the client keeps a persistent bright-north/dark-south hillshade at every sampled time, while the swept formula (`cos(sunAngle - pi/2)`) went exactly horizontal-less (straight overhead) at noon/midnight, washing out the relief on bowl/crater terrain. Lock the horizontal bearing to a fixed north-west share at every time of day (matching the traced ray's constant azimuth, §2.1) and vary only elevation. Add `TerrainSolarDirectionTests` regression coverage for the non-collapsing horizontal magnitude and fixed bearing ratio, and correct `AuthoredTerrainDayNightProfileTests`' now-inverted "vertical at noon" assertion.
- [x] T027o [US6] Transform raw ADT MCNR normals to renderer coordinates before every compositor
  Lambert dot. Wire the shared exact-build Light DBC resolver into the interactive viewer for 2.x+
  no-LIT clients, expose active source/status and anomaly recoveries, and prove the real
  2.4.3.8606 catalog plus focused asymmetric-normal regressions. Minimap use was removed by T027q.
- [x] T027p [US6] Replace the inadequate WDT-occupied one-tile visual handoff with bounded
  `--tile-list` plus `--authored-reference`. Emit native authored, synthetic, liquid, and
  authored-vs-synthetic images per tile; reject missing/all-black authored references and all-black
  synthetic results. Vet the 2.4.3 Expansion01 comparison set by WDT occupancy, nonblack authored
  decode, and 5-10 nonblack decoded terrain BLPs; explicitly retire black tile 32,32.
- [x] T027q [US6] Restore the minimap/viewer lighting ownership boundary after the Expansion01
  proof showed local Light DBC colors crushing and purple-tinting synthetic terrain. Make every
  `synthetic-minimap` path use one fixed 12:00 achromatic global light, reject non-noon/DBC options,
  bump the manifest to v6, retain exact-build LIT/Light DBC only in the interactive viewer, and add
  a focused no-darkening regression.
- [x] T027r [US1] Make the interactive viewer global directional/ambient light unconditional and
  default it to noon. Preserve raw exact-build DBC/LightData profiles only as spatially weighted
  local overlays, reset departed local fog to the global range, expose both layers independently in
  Lighting, and prove zero-weight identity plus local blending with focused tests and a Debug build.
  User-run 3.x visual confirmation remains part of T019.

## Phase 4: User Story 2 - Render M2 assets through their native path (Priority: P1)

**Goal**: M2 runtime rendering is native or diagnostic, never converted MDX.

**Independent Test**: Focused M2 reader/runtime tests plus Debug build; user visual proof at all
four representative client eras.

- [ ] T028 [US2] Make the runtime bridge unconditionally select native M2 drawing in `src/viewer/WoWViewer/Rendering/WowViewerM2RuntimeBridge.cs`
- [ ] T029 [US2] Route 1.0.0 embedded M2 divisions before skin probing in `src/viewer/WoWViewer/Terrain/WorldAssetManager.cs`
- [ ] T030 [US2] Route 1.0.0 embedded WMO doodads before skin probing in `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`
- [ ] T031 [US2] Remove M2-to-MDX and adapter-backed MdxRenderer runtime fallback construction from `src/viewer/WoWViewer/Terrain/WorldAssetManager.cs`
- [ ] T032 [US2] Remove M2-to-MDX and adapter-backed MdxRenderer runtime fallback construction from `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`
- [ ] T033 [P] [US2] Add native-route/diagnostic coverage in `tests/WowViewer.Core.Tests/M2RuntimeTests.cs`
- [ ] T034 [US2] Run focused M2 reader/runtime tests and Debug build from `specs/110-viewer-stabilization/quickstart.md`

## Phase 5: User Story 3 - Use only working modern tool surfaces (Priority: P2)

**Goal**: The Tools menu contains current owners only, and Inspect/Converter have reliable entry points.

**Independent Test**: Inventory each menu item in both UI modes; every retained action resolves or
reports an actionable missing dependency.

- [ ] T035 [US3] Inventory and classify main-menu Tools actions in `src/viewer/WoWViewer/ViewerApp.cs`
- [ ] T036 [US3] Remove MK Dataset/VLM Dataset and other dead launchers from `src/viewer/WoWViewer/ViewerApp.cs`
- [ ] T037 [US3] Implement supported Inspect/Converter resolution and dependency diagnostics in `src/viewer/WoWViewer/ViewerApp.cs`
- [ ] T038 [US3] Add source-level or integration coverage for retained menu routes in `tests/WowViewer.Core.Tests/`
- [ ] T039 [US3] Build the Viewer, Inspect, and Converter projects with their retained launch contracts

## Phase 6: User Story 4 - Export conversion is explicit and bounded (Priority: P2)

**Goal**: Publish what conversion directions do today, without overstating fixture proof as
real-client reliability.

**Independent Test**: Both WMO conversion test classes and M2-to-MDX test class pass; documentation
contains the exact evidence label for each direction.

- [ ] T040 [P] [US4] Run and record WMO v14→v17 coverage in `tests/WowViewer.Core.Tests/WmoV14ToV17ConverterTests.cs`
- [ ] T041 [P] [US4] Run and record WMO v17→v14 coverage in `tests/WowViewer.Core.Tests/WmoV17ToV14ConverterTests.cs`
- [ ] T042 [P] [US4] Run and classify M2-to-MDX export coverage in `tests/WowViewer.Core.Tests/M2ToMdxConverterTests.cs`
- [ ] T043 [US4] Publish source/target fidelity statuses in `docs/architecture/viewer-conversion-capability-2026-07-16.md`
- [ ] T044 [US4] Ensure M2 export commands do not appear in runtime route code under `src/viewer/WoWViewer/`

## Phase 7: Polish and continuity

- [x] T045 Update task states and exact proof commands in `specs/110-viewer-stabilization/tasks.md`
- [x] T046 Update active work and next bounded phase in `wow-viewer/memory-bank/activeContext.md`
- [x] T047 Update completed work and next phase in `wow-viewer/memory-bank/progress.md`

## Implementation Strategy

1. Deliver and validate Phase 3/3b/3c (US1/US6/US7) first: it restores immediate map usability,
   keeps its controls reachable, and makes missing minimaps recoverable without relying on an
   authored minimap asset.
2. Only after US1/US6/US7 proof, remove incorrect M2 runtime fallback routes and recover the native paths.
3. Then reduce the UI surface to working tools and document conversion evidence separately.
