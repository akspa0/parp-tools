# Tasks: Cartography — Multi-Map, Multi-Tile Composition Workbench (Spec 222)

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)
Checklist order = execution order. Each phase ends at a verification gate before the next begins.

## Phase 1 — Footprints & tile placements visible (P1 foundation)

- [x] 222-T101: Add `GetOccupiedTiles(mapName)` + `TryResolveMap(mapName)` to [`ITerrainAdapter`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/ITerrainAdapter.cs); implemented on `AlphaTerrainAdapter` (phase WDT MAIN offsets via `MapFootprint.FromMainOffsets`), `StandardTerrainAdapter` (own tile set for the base map, cached 4096-probe scan for overlays), and `RosettaDatastoreTerrainAdapter` (its own map only). Pure math extracted to Core ([`MapFootprint`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Maps/MapFootprint.cs)) so it is unit-testable without the GL viewer. **13/13 tests green** (`MapFootprintTests`), including the Shadowfang-over-Azeroth overlap regression case.
- [x] 222-T102: `PhaseLayerResolution` enum + `FootprintColorIndex` added to `PhaseLayerSettings` (Core, clone-safe); `TerrainManager` wiring: `GetLayerFootprint`, `GetBaseFootprint`, `FootprintPalette` (8 saturated hues), `RefreshLayerResolutionStates` (called from `RefreshPhaseLayers` so badges are always current after an edit). Viewer build 0 errors.
- [x] 222-T103: Minimap overlay footprint draw pass landed in [`MinimapHelpers.RenderPhaseFootprints`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/MinimapHelpers.cs) — per-layer colored rects (offset applied, so the operator sees where content LANDS), selected-layer highlight, culled to the view window, drawn under the camera indicator. WorldScene gained `SelectedPhaseLayerIndex` + `GetLayerFootprints()`. Single-tile placement rects land with Phase 2's drag interaction (T105) since a placement without a drag target has nothing to show yet.
- [x] 222-T104: Inline status badges in [`ViewerApp_PhaseLayers`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) layer rows — footprint color swatch + resolution outcome: "resolving…", orange "map not found — the layer contributes nothing", tile count with "overlapping the base map" or orange "none overlap the base map; set an offset or align to see anything". Expanding a row selects it (drives the minimap highlight); removal/reorder re-syncs the selection.
- [ ] **Gate 1**: Shadowfang over Azeroth (0.5.3) shows its 10-tile footprint with zero input; a single donor tile from an unoverlapping map is visible and droppable; unresolvable maps error inline. Build green.

## Phase 2 — Drag-to-align (P1 interaction)

- [x] 222-T105: Minimap footprint drag landed in [`ViewerApp_MinimapAndStatus`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs): press on the SELECTED layer's footprint grabs it (`TryGetFootprintTileAt` hit-test); while dragging, the offset follows the pointer live via `MapFootprint.ApplyDragDelta` (clamped, recomputed from the drag-start base each frame so wiggle cannot accumulate); release calls `RefreshPhaseLayers()` exactly once. Pans still work when no footprint is grabbed; click-without-drag on base terrain still teleports (FR-9 preserved — drag release returns before the click path).
- [x] 222-T106: 7 new tests in [`MapFootprintTests`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/Maps/MapFootprintTests.cs): drag delta zero/rounded/clamped/idempotent-per-frame, offset→source-tile round-trip, per-tile-placement-claims-target-over-offset, and unmapped-target-still-routes-through-offset. **20/20 tests green**; viewer build 0 errors.
- [ ] **Gate 2**: Dragging a layer N tiles shifts 3D composition by exactly N tiles on both Alpha and Standard bases; dragging a placement moves only that placement (operator visual check; unit tests green).

## Phase 3 — Transform tools + consolidation (P1/P2)

### Operator directives 2026-09-04 (post-Gate-1/2, all P1 for the Cartography build)

- **Gate 1/2 PASSED**: Shadowfang composes on Azeroth — 25 tiles, real terrain rendering after the
  cache-versioning fix re-extracted the true 6.9 MB alphaWDT.
- **New defect report (culling/camera): sticky tiles.** Tiles sometimes refuse to unload/reload when
  the phase layer is moved around or over the camera position. **ROOT-CAUSED AND FIXED** (same day):
  composition happens at tile-parse time and [`TerrainManager._tileCache`](../../src/viewer/WoWViewer/Terrain/TerrainManager.cs)
  is keyed only by target tile coords. `EvictAllTiles()` cleared the caches, but background tile
  loads queued under the OLD composition finished afterwards and wrote their stale composed results
  back into `_tileCache` + `_pendingTiles` — re-materializing old content as "sticky" tiles.
  **Fix**: `_compositionGeneration` counter — bumped at the top of `EvictAllTiles` (before cache
  clearing), `_pendingTiles` drained, and each background load captures the generation at queue
  time and DISCARDS its result when the generation moved on. Build green; operator should confirm
  the sticky behavior is gone by dragging a phase layer around and across the camera.
- **222-T109 AMENDED — Cartography becomes a top-level feature**: add a **Cartography tab to the
  top tab bar** (beside Viewer/Editor/Archaeology), simple, not overly dramatic. The phase/layer
  panel lives in the **RIGHT sidebar** — explicitly NOT the left sidebar, where it is hidden away
  from everything. Delete the left-sidebar Phase Map Layers panel in the same change (FR-11).
- **222-T109a (new) — Synthesized minimap moves INTO Cartography**: the synthesized-minimap tooling
  must (a) render minimaps from the **merged phase-map layer stack using whatever channels are
  selected for each layer** (not just the base map), and (b) live as a **tab inside the right
  sidebar under Cartography**, not as a separate floating window. Floating windows are unreliable —
  they disappear and cannot be recalled by any simple means (except the Settings window) — and are
  retired as a surface for this feature.
- Defunct-project retirement (ViewerIoService / WorldGpuPreviewRenderer / runtime-bridge migration)
  continues as a background lane; it does not gate the Cartography UI work.

### Operator directive 2026-09-04 (later) — synthesized 0.5.3 minimap tint still off vs authored

- Synthesized minimap (left) is brighter, less saturated, purple-shifted vs the authored client
  minimap (right): warmer, darker brown. Operator suspects the generation sun time differs from
  what Blizzard used for the authored maps.
- **The measurement instrument already exists**: `synthetic-minimap --measure-sun
  --authored-reference` scores each authored tile across BOTH sun bearings and hours and reports
  which bearing best explains the authored shading (`MinimapShadingMatch.SweepSolarAzimuth`);
  `--match-time` renders at the inferred hour; documented contrast knobs (`--ambient`,
  `--bearing`, `--score`) sweep tint after the sun is measured.
- **Recorded knowledge gap**: `MinimapEraProfile.Alpha053` carries
  `AzimuthProvenance: AssumedFromOtherBuild` — the 0.5.3 solar model is inherited from the 1.0.0
  debugger trace, never measured for 0.5.3. The profile's own doc says to measure against authored
  0.5.3 minimaps, then set `SolarModelProvenance.MeasuredFromAuthoredMinimaps`.
- Next bounded step: operator runs the sweep on the side-by-side tile, we read the measured
  bearing/hour, update `Alpha053`, re-render. If the tint still differs with a converged sun,
  audit the compositor albedo path next (texture decode + DXT1 quantization + tint ratios) — sun
  time alone may not explain the hue/saturation gap.

- [-] 222-T107 (in progress): Transform toolbar wrapping `TileContentTransform` (Spec 219 core, validated). **Landed**: Core `TileContentTransform.TransformTileChunks` — composed content transforms + 16×16 slot re-mapping in one pass (Core's `TerrainChunkData`), ready for unit tests and the harvest/converter paths. **Blocked on a viewer↔Core chunk mapping**: the Alpha adapter's chunks are the VIEWER's `TerrainChunkData` (`WoWViewer/Terrain/TerrainChunkData.cs`), a different type from Core's — the validated seam operates on Core's type, so the adapter wiring needs an explicit field-mapping (or unifying the two types under Core, which is the cleaner fix and should be decided with the operator). The naive wiring attempt was REVERTED to the proven offset path before it shipped; rotation/mirror fields therefore remain inert until this lands. Do not re-attempt the wiring without resolving the type split first.
- [ ] 222-T108: Donor tile-grid picker in the add flow: browse a donor map's 64×64 grid, click tiles to create single-tile placements with a chosen target.
- [ ] 222-T109: Rebuild the layer stack UI as the Cartography panel in the right sidebar: rows with swatch, state badge, enable toggle, expandable details (all existing channel/transform controls, Spec 219 included). **Delete the left-sidebar Phase Map Layers panel in the same change** (FR-11).
- [ ] 222-T110: Move DBC child-map suggestions into Cartography's add flow.
- [ ] 222-T111: Chunk-manipulator (Spec 195) parity checklist: every 195 capability is either present in Cartography or listed as deferred with an owner spec; then **retire the 195 UI**. Settle plan open questions Q1–Q3 before this task.
- [ ] **Gate 3**: 30-second add→align criterion met; copy→rotate→paste keeps all channels; exactly one manipulation surface exists (old panel + 195 UI gone).

## Phase 0.5 (found during Gate 1) — UNVERSIONED CACHE was the root cause; stride "fix" REVERTED

- [x] 222-T100: **ROOT CAUSE — the extraction cache was unversioned.** `output/cache/Shadowfang.wdt` (32,828 bytes) was a stale file extracted from a DIFFERENT client version — the real 0.5.3 `Shadowfang.wdt` in `Shadowfang.wdt.MPQ` is ~6.9 MB with embedded ADTs (same as Azeroth.wdt at 752 MB, which extracted fully and always worked). The phase resolver (`TerrainManager.ResolvePhaseWdtPath`) preferred the existing cache file, so every "diagnosis" of the 32 KB file — the "8-byte MAIN cells", the "5×5 footprint", the "WMO-only map" badge — was analyzing garbage from the wrong client version. **Operator identified it; the probes confirmed**: MPQ header shows `Shadowfang.wdt.MPQ` = 6,897,752 bytes holding 3 entries; the stale cache copy ended exactly at the end of MAIN.
- [x] 222-T100b: **Cache versioning fix** — [`ViewerApp.LoadVirtualFile`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/ViewerApp.cs) now writes extracted files to `output/cache/<client-root-hash>/<file>` using the same `BuildCacheSegment(BuildWdlPreviewCacheIdentity())` client-root identity the minimap/WDL caches already use. Files from different client roots can no longer shadow each other. Phase WDTs land in the same versioned segment (they're written next to the base WDT path).
- [x] 222-T100c: **REVERTED** the `MainAlpha.cs` / `WdtAlpha.cs` stride "fixes" — they were fitted to the stale cross-version cache file, not the real 0.5.3 alphaWDT. Both files are back to their committed state (`git checkout`), and the alphaWDT reader is untouched. Lesson recorded: never fit format fixes to cache files without verifying the cache against its source first.
- [ ] 222-T100d: Operator re-verification — delete the stale `output/cache/Shadowfang.wdt` (or any root-level flat cache files), reload Azeroth 0.5.3, re-add Shadowfang: the WDT should re-extract at ~6.9 MB into the versioned segment and compose its real terrain.

## Phase 4 — Cleanup + docs

- [ ] 222-T112: Remove dead code paths (left panel, 195 UI remnants); full suite gate — no new failures vs the 1441-pass/10-fail baseline.
- [ ] 222-T113: Update [`specs/STATUS.md`](file:///I:/parp/parp-tools/wow-viewer/specs/STATUS.md) + [`activeContext.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md); record the 2026-09-04 Shadowfang diagnosis as the motivating evidence and the 195/219 consolidation map.
- [ ] 222-T114: Operator interactive verification — Alpha (Shadowfang over Azeroth: footprint, drag, align, compose), one 3.3.5 map, and a tile copy→rotate→paste flight. Visual proof is operator-owned.
