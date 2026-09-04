# Tasks: Cartography — Multi-Map, Multi-Tile Composition Workbench (Spec 222)

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)
Checklist order = execution order. Each phase ends at a verification gate before the next begins.

## Phase 1 — Footprints & tile placements visible (P1 foundation)

- [ ] 222-T101: Add `GetOccupiedTiles(mapName)` + `TryResolveMap(mapName)` to [`ITerrainAdapter`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/ITerrainAdapter.cs); implement on `AlphaTerrainAdapter` (phase WDT MAIN offsets) and `StandardTerrainAdapter` (overlay WDT existence set). Unit tests on synthetic WDT fixtures.
- [ ] 222-T102: Add `ResolvedState` + `FootprintColor` to `PhaseLayerSettings`; wire adapter resolution results into layer state via `TerrainManager`.
- [ ] 222-T103: Minimap overlay draw pass — per-layer footprint rects AND single-tile placement rects (donor-side + target ghost), per-layer colors, selected highlight, non-overlapping badge. Reuse the teleport path's tile↔pixel mapping from [`MinimapHelpers`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/MinimapHelpers.cs).
- [ ] 222-T104: Inline state badges in layer rows (active / non-overlapping / unresolved-with-error / missing-donor-tile).
- [ ] **Gate 1**: Shadowfang over Azeroth (0.5.3) shows its 10-tile footprint with zero input; a single donor tile from an unoverlapping map is visible and droppable; unresolvable maps error inline. Build green.

## Phase 2 — Drag-to-align (P1 interaction)

- [ ] 222-T105: Minimap drag interaction: hover selects layer/placement; drag moves layer offset OR placement target; release applies clamped values (±63) and calls `RefreshPhaseLayers()` exactly once. Click-without-drag on base terrain still teleports (FR-9).
- [ ] 222-T106: Unit tests: delta math, clamping, placement-claim-over-offset, offset→source-tile round-trip (`target − offset = source`).
- [ ] **Gate 2**: Dragging a layer N tiles shifts 3D composition by exactly N tiles on both Alpha and Standard bases; dragging a placement moves only that placement (operator visual check; unit tests green).

## Phase 3 — Transform tools + consolidation (P1/P2)

- [ ] 222-T107: Transform toolbar wrapping `TileContentTransform` (Spec 219 core, validated): rotate 90/45/free, mirror, copy/paste on the current selection (layer | tile set | single tile). Paste applies ALL channels the source carries, gated by the target layer's channel checkboxes — the Spec 195 partial-paste defect cannot recur. 45°/free-angle results report their approximation mode in the UI.
- [ ] 222-T108: Donor tile-grid picker in the add flow: browse a donor map's 64×64 grid, click tiles to create single-tile placements with a chosen target.
- [ ] 222-T109: Rebuild the layer stack UI as the Cartography panel in the right sidebar: rows with swatch, state badge, enable toggle, expandable details (all existing channel/transform controls, Spec 219 included). **Delete the left-sidebar Phase Map Layers panel in the same change** (FR-11).
- [ ] 222-T110: Move DBC child-map suggestions into Cartography's add flow.
- [ ] 222-T111: Chunk-manipulator (Spec 195) parity checklist: every 195 capability is either present in Cartography or listed as deferred with an owner spec; then **retire the 195 UI**. Settle plan open questions Q1–Q3 before this task.
- [ ] **Gate 3**: 30-second add→align criterion met; copy→rotate→paste keeps all channels; exactly one manipulation surface exists (old panel + 195 UI gone).

## Phase 4 — Cleanup + docs

- [ ] 222-T112: Remove dead code paths (left panel, 195 UI remnants); full suite gate — no new failures vs the 1441-pass/10-fail baseline.
- [ ] 222-T113: Update [`specs/STATUS.md`](file:///I:/parp/parp-tools/wow-viewer/specs/STATUS.md) + [`activeContext.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md); record the 2026-09-04 Shadowfang diagnosis as the motivating evidence and the 195/219 consolidation map.
- [ ] 222-T114: Operator interactive verification — Alpha (Shadowfang over Azeroth: footprint, drag, align, compose), one 3.3.5 map, and a tile copy→rotate→paste flight. Visual proof is operator-owned.
