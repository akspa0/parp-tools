# Implementation Plan: Phase Layer Rotation Tools

**Branch**: `219-phase-layer-rotation`
**Spec**: [spec.md](spec.md)
**Created**: 2026-09-03

## Technical Context

- **Runtime**: .NET 10, C# 13 (matches repo standard; see `wow-viewer/memory-bank/techContext.md`)
- **Graphics**: OpenGL 3.3 via Silk.NET; UI via ImGui.NET
- **Testing**: xUnit under `wow-viewer/tests/WowViewer.Core.Tests` — **no test project references
  the viewer**, so all rotation/mapping logic that must be tested lives in `WowViewer.Core*`
- **Composition seam**: Spec 203's `PhaseLayerSettings` (Core, `WowViewer.Core/Maps/PhaseComposition.cs`)
  plus the per-adapter `MergePhaseTile` in `StandardTerrainAdapter` (split ADT) and
  `AlphaTerrainAdapter` (monolithic Alpha WDT)
- **Scope**: viewer-side composition only; no client-file writes

## Constitution Check

| Gate | Status | Notes |
|---|---|---|
| Library-first (readers/contracts in Core, tools thin) | PASS | Rotation math and per-tile mapping policy live in `WowViewer.Core/Maps`; adapters consume them |
| No test project references the viewer | PASS | New policy types in Core; adapters stay thin callers |
| Do not modify working format readers | PASS | No MPQ/ADT/WMO/M2 reader changes; composition is layered above parse output |
| Layer new tooling above proven base | PASS | Extends 203's `PhaseLayerSettings` + `MergePhaseTile` seam; no renderer/camera/terrain-loading behaviour change for unrotated layers |
| 0.5.3 behaviour unchanged for no-rotation layers | PASS | FR-011; rotation defaults to zero and short-circuits |
| One phase at a time, validated | PASS | Phases below each end in a build/test gate |
| Spec Kit first | PASS | This plan follows the spec |

## Summary of Design Decisions

Full rationale in [research.md](research.md); the decisions the plan depends on:

1. **Rotation representation**: `RotationDegrees` (float) + `RotationOriginTileX/Y` (float, tile
   space) on `PhaseLayerSettings`. Zero degrees short-circuits everywhere — no behaviour change.
2. **45° approximation (FR-009 decision): free-rotate with grid-snapped tile *lookup*, placements
   transformed exactly.** Terrain chunks are re-sourced from the donor tile whose rotated position
   covers the target; chunk-level terrain is NOT resampled (no height interpolation) in Phase 1.
   The approximation mode is reported per layer. Resampling is a later, opt-in mode.
3. **Transform order (FR-007, fixed and stated)**: **rotate about origin → then apply tile
   offset → then per-tile placement overrides**. Per-tile placement is the last word for a target
   tile it claims; conflicts are reported.
4. **Per-tile placement**: a list of `(DonorTileX, DonorTileY, TargetTileX, TargetTileY)` on the
   layer. Target lookup consults the mapping before the whole-layer offset.
5. **Straddling placements**: an object belongs to the donor tile it was parsed from (the ADT that
   listed it); it moves with that tile. Verified against real data in Phase 2 (T-gate).
6. **General-purpose transform seam (FR-019, operator clarification 2026-09-03)**: the rotate /
   mirror / offset math is factored into a standalone Core seam — `WowViewer.Core/Maps/TileContentTransform.cs`
   operating on `TerrainChunkData` + placement lists — that knows nothing about phases. The phase
   layer is the first consumer; later editing tooling calls the same primitives. Mirroring is
   exact-grid: heights mirror in place, the mirrored normal axis component negates, alpha maps
   mirror, and placement positions mirror with orientation adjusted for the handedness flip.
   Mirrors compose as an involution (mirror twice = identity, SC-009).
7. **Translation precision (FR-020/021, operator clarification 2026-09-03)**: canonical whole-layer
   offsets are integer terrain cells, with 8 cells per chunk and 128 cells per ADT axis. Tile and
   chunk controls are views over the same cell offset, not separate transforms. Applying a
   non-boundary offset re-slices the 145-vertex lattice and associated per-cell channels across
   target chunks/ADTs; authored heights are moved, never interpolated. World-float/sub-cell shifts
   require a future stated resampling mode and are not silently approximated.
8. **Optional WDL alignment (FR-022)**: reuse Spec 196's existing `WdlLatticeMagnetizer` sampling as
   a fit/proposal signal. Search bounded integer cell offsets, score donor micro-relief against the
   target WDL macro surface, and present the best candidate for operator acceptance. The accepted
   integer cell offset remains the transform; magnetization never silently rewrites terrain and its
   implementation is not copied into this spec.
9. **One selection, two views (FR-023–027)**: one Core.Editor selection contract owns tile/chunk/cell
   units, modifiers, source and explicit target. The orthographic full-map canvas and the 3D terrain
   selector are projections of that contract, not peers with synchronization glue between duplicate
   state. Minimap, heightmap, and occupancy are canvas backdrops; none owns selection semantics.
10. **One composition stack (FR-028–030)**: base is a non-removable first layer with channel gates
    applied before overlays. Every phase card owns its whole-layer rotation/mirror controls; selected-
    content transforms are separate operations and never substitute for the phase controls.
11. **Spec 195 migration, not coexistence (FR-031/032)**: reuse audited coordinate/undo components,
    then redirect and retire the broken camera-centered canvas and incomplete paste path. There is one
    clipboard/preview/commit/undo model and every paste has an explicit target.
12. **Save the preview through proven writers (FR-033–035)**: materialize one canonical composition,
    preflight representability, then route supported LK-v18 ADT and Alpha WDT outputs through the
    existing writer/converter seams. Native split MoP remains refused until Spec 197 provides its
    writer. Source files are not overwritten by default.

## Phases

### Phase 1 — Core transform seam + rotation policy + 90° exact-grid path (US1, US4, FR-019)

All new logic in `WowViewer.Core/Maps/`:

- **`TileContentTransform.cs` (the general-purpose seam, FR-019)**: pure functions over
  `TerrainChunkData` + placement lists — `RotateContent90(chunk, placements, origin, clockwise)`,
  `MirrorContentH(...)`, `MirrorContentV(...)`, `TranslateContent(...)` — with no phase-system
  type in scope. Heights mirror/rotate in place; the mirrored normal axis component negates;
  alpha maps mirror; placement positions transform and orientations adjust for handedness.
  A Core unit test exercises this seam directly with no phase type in scope (SC-010).
- `PhaseLayerRotation` helpers: `RotateTile90(tileX, tileY, originX, originY, clockwise)` — exact
  integer grid mapping; `RotatePoint90`, `RotateDegreesToQuadrant`.
- `PhaseLayerSettings` gains `RotationDegrees`, `RotationOriginTileX/Y`, `MirrorHorizontal`,
  `MirrorVertical` (defaults: off — no behaviour change).
- `PhaseCompositionPolicy.ResolveTileSource(layer, targetX, targetY)` — the single function both
  adapters call to answer "which donor tile fills target (tx,ty)?", applying per-tile placements
  first, then offset+rotation(+mirror). Returns source tile + the transform to apply to its content.
- Placement transform: `RotatePlacement(position, rotation, angle, origin)` rotating renderer-space
  position and adding the angle to the placement's Z rotation (degrees), for 90° multiples exact;
  mirror variants flip the position and negate the appropriate orientation components.
- Focused tests: 90° CW/CCW round-trip identity, mirror involution (SC-009), corner/edge tiles,
  origin handling, conflict reporting, seam-without-phases test (SC-010).

**Gate**: `dotnet build` 0 errors; focused `PhaseComposition` tests green; unrotated/unmirrored
layers byte-identical behaviour (existing 18 tests still pass).

### Phase 2 — Adapter wiring: 90° rotation + per-tile placement (US1, US5)

- `StandardTerrainAdapter` and `AlphaTerrainAdapter` replace their inline
  `tileX - layer.TileOffsetX` source lookup with `PhaseCompositionPolicy.ResolveTileSource`.
- Placement translation path extended: after the existing offset translation, apply rotation to
  placement positions and orientations (90° exact), and re-chunk placements whose rotated position
  lands in a different tile than the one being composed (they belong to their donor tile and move
  with it).
- Per-tile placement list on `PhaseLayerSettings`; adapters consult it in `TileExists` and
  `LoadTileWithPlacements`.
- Diagnostics: `[Phase] tile source (tx,ty) <- donor (dx,dy) via <offset|rotation|per-tile>`
  log lines; conflict reports.
- **Real-data gate (T-gate)**: load a real donor map, verify straddling objects land on their
  donor tile's terrain; record the observed rule in research.md.

**Gate**: build 0 errors; focused tests green; operator loads a 90°-rotated phase layer and
confirms terrain+objects rotate as a unit (SC-001).

### Phase 3 — Shared selection model + orthographic full-map canvas (US6)

- Migrate useful Spec 195 coordinate/selection behavior into a single Core.Editor selection contract
  supporting tile/chunk/cell units, mixed non-rectangular regions, explicit target, and modifier modes.
- Build a full-map orthographic canvas with minimap, heightmap, and occupancy backdrops; pan/zoom must
  never collapse it into a camera-centered local grid.
- Implement magnetic click/check, paint, box, and lasso selection with accessible source/target/
  preview/conflict/empty/committed states.

**Gate**: coordinate/property tests green; operator can select and target a multi-ADT region without
raw coordinates in under 30 seconds.

### Phase 4 — 3D selection + Spec 195 retirement (US7, US9)

- Project the same selection contract onto terrain in 3D with explicit tool-mode input arbitration.
- Redirect the old Chunk Manipulator entry to the workbench, migrate only proven coordinate/undo
  services, and remove/disable its independent selection, clipboard, camera-target, and paste path.

**Gate**: one selection and undo history are demonstrated across both views; no second active
transform pipeline remains.

### Phase 5 — Cell-granular translation and boundary re-slicing (FR-020/021/022)

- Add canonical `CellOffsetX/Y` state to `PhaseLayerSettings`; preserve legacy `TileOffsetX/Y` as
  exact 128-cell compatibility views. Add chunk controls as exact 8-cell views.
- Implement phase-agnostic content translation/re-slicing in Core across source/target chunk and
  ADT boundaries for heights, normals, holes, texturing, shadows, MCCV, liquid, and placements.
- Wire the same translation result into both adapters; do not fork Alpha and Standard math.
- Add optional bounded WDL-assisted offset search by consuming Spec 196's sampler; report candidate
  offset + fit score and apply only after explicit operator acceptance.
- Add exact equivalence tests: 8 cells = 1 chunk, 128 cells = 1 ADT, inverse round-trip, and seams
  crossing both chunk and ADT boundaries. Add WDL decline/no-change coverage.

**Gate**: build 0 errors; focused equivalence tests green; operator confirms a terrain patch that
does not share source/target chunk or ADT boundaries can be aligned at one-cell precision, with WDL
snap remaining optional and inspectable.

### Phase 6 — 45° named tools + free angle (US2, US3)

- 45°/free rotation: terrain re-sourced by free-rotated coverage (no resampling); placements
  transformed exactly (position rotate, orientation add).
- `RotationApproximation` reported per layer in the panel and logs (FR-009).
- UI: preset buttons (90 CW, 90 CCW, 45 CW, 45 CCW) + free-angle input + origin display, all
  font-scale legible (FR-012, FR-017), on the existing Phase Map Layers panel.
- Minimap: rotated layer tiles drawn at rotated positions (US4 scenario 2).

**Gate**: build 0 errors; tests green; operator confirms 45° result matches the reported mode
(SC-003) and objects stay seated (SC-004).

### Phase 7 — Save Transformed Map (US10)

- Define save preflight and composition materialization independent of the UI.
- Add the workbench button and supported-format selection. Use output copies and explicit overwrite.
- Read back supported ADT and Alpha WDT output and compare channels, placement transforms, and tile
  inventory against the preview. Refuse native split MoP and every unrepresentable route before write.

**Gate**: tests and operator reload pass for one ADT and one Alpha WDT output; no source overwrite.

### Phase 8 — Composition order + regression gates (US4, SC-005/006)

- Lock the stated order (rotate → offset → per-tile) in policy code with tests proving
  determinism across reloads.
- Pixel-comparison regression: no-rotation map identical to pre-feature (SC-006); 90 CW then
  90 CCW round-trip (SC-002).
- Update Spec 203 cross-reference and continuity files.

**Gate**: full solution build + focused tests; operator reload verification.

## Risks / Open Items

- **Straddling placements** (spec assumption): Phase 2's real-data gate decides the final rule;
  the plan's default (object belongs to the ADT that listed it) may need adjustment if operator
  data shows otherwise.
- **Rotation + presence gating interaction**: a rotated donor tile whose channels are partially
  gated must not rotate channels it does not contribute; covered by Phase 1 tests.
- **Performance**: rotation must not force loading every donor tile; per-tile re-sourcing keeps
  the existing AOI-driven lazy load intact (only the lookup changes).
- **Minimap**: rotated minimap tiles reuse the existing per-layer minimap path; if the minimap
  renderer cannot express rotation cheaply, the plan falls back to offset-only minimap with a
  stated limitation rather than a wrong visual.
- **Legacy implementation claims are unreliable**: Spec 195 marks drag selection, complete channel
  paste, and live integration complete, but source audit shows click-only UI, camera-derived target,
  no map backdrop, and an incomplete paste path. Phase 4 uses code evidence, not old task checkmarks.
- **Save representability**: Alpha WDT and LK-v18 ADT have proven writer paths; modern split output
  does not. Preflight refusal is required until Spec 197 changes that fact.
