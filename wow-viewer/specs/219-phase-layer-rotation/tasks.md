# Tasks: Map Composition Selection & Transform Workbench

**Feature**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)
**Created**: 2026-09-03

Dependency-ordered checklist. Each phase ends in a validation gate; do not start the next phase
before the current gate passes.

## Phase 1 — Core transform seam + rotation policy (US1, US4, FR-019)

- [x] T001: Create `WowViewer.Core/Maps/TileContentTransform.cs` — pure static seam over
      `TerrainChunkData` + complete `MddfPlacement`/`ModfPlacement` records. Exact-grid 90°/180°
      rotation + H/V mirrors cover heights, normals, holes, alpha, MCSH, MCCV, liquid flags,
      placement position/yaw, and MODF bounds. Free-angle point rotation is present; terrain
      free-rotate sourcing and offset orchestration remain Phase 3/adapter work. No phase type in scope.
- [x] T002: Extend `PhaseLayerSettings` with `RotationDegrees`, `RotationOriginTileX/Y`,
      `MirrorHorizontal`, `MirrorVertical` (defaults: 0°, false, false).
- [x] T003: Add `PhaseTilePlacement` record + `TilePlacements` list on the layer.
- [x] T004: Add `PhaseTileSource` result + `PhaseTileSourceKind` / `PhaseRotationApproximation`
      enums; implement `PhaseCompositionPolicy.ResolveTileSource(layer, tx, ty)` — per-tile first,
      then offset+rotation+mirror; last-wins conflict rule with report.
- [x] T005: Placement transform helpers — complete MDDF/MODF position + Z-rotation, mirror
      variants with handedness-correct orientation adjustment.
- [x] T006: Focused tests — 90° CW/CCW round-trip identity; mirror involution (SC-009);
      corner/edge tiles; origin handling; per-tile-beats-offset; duplicate-target last-wins with
      report; direct phase-agnostic seam tests (SC-010); untransformed routing regression.
- [x] **Gate 1**: full `WowViewer.slnx` Debug build **0 errors** (302 existing warnings);
      focused `TileContentTransform|PhaseTileSource|PhaseComposition` tests **55/55 passed**,
      including the existing 18 PhaseComposition tests. No runtime/visual/real-client claim.

## Phase 2 — Adapter wiring + visible whole-phase controls (US1, US5, US8)

- [ ] T007: `StandardTerrainAdapter` — replace inline `tileX - layer.TileOffsetX` lookups in
      `TileExists` / `LoadTileWithPlacements` with `ResolveTileSource`; apply
      `TileContentTransform` to sourced chunk content and placements.
- [ ] T008: `AlphaTerrainAdapter` — same wiring for the monolithic Alpha WDT path.
- [ ] T009: Diagnostics — `[Phase] tile source (tx,ty) <- donor (dx,dy) via <kind>` and
      `[Phase] rotation origin (ox,oy), approximation <mode>` log lines; conflict reports.
- [ ] T010: Add controls to **every phase-layer card**: 90° CW/CCW, 45° CW/CCW, Mirror H/V,
      free angle, editable/visible origin, approximation, and reset. Exact 90°/mirror controls must
      become usable immediately; free-angle remains visibly marked preview-limited until Phase 6.
- [ ] T011: Add the base map as a non-removable first composition layer with the same channel
      checkboxes; apply base gates before phase composition. Add Base Only, Phases Only, All, None,
      Terrain, and Objects presets.
- [ ] T012: **Real-data gate (T-gate)** — operator loads a real donor map; verify straddling
      objects land on their donor tile's terrain; record the observed rule in research.md (R4).
- [ ] **Gate 2**: build 0 errors; tests green; operator confirms 90° rotation + mirrors render
      terrain+objects as a rigid unit from phase-card controls, and base channels can be suppressed
      independently (SC-001, SC-009, SC-016, SC-017 visual).

## Phase 3 — Shared selection contract + orthographic whole-map canvas (US6)

- [ ] T013: Audit and migrate useful parts of `GlobalChunkCoordinate` / `ChunkSelectionRegion` into
      one Core.Editor `MapContentSelection` contract supporting tile, chunk, and cell units; source,
      explicit target, add/subtract/replace/toggle, rectangle, lasso rasterization, and stable order.
- [ ] T014: Build a full 64×64 orthographic canvas with pan/zoom and selectable minimap, heightmap,
      and occupancy backdrops; never depend on the camera tile for canvas extent or target.
- [ ] T015: Add magnetic tile/chunk/cell hover and click/check, paint, rectangle-drag, lasso,
      add/subtract/replace/toggle gestures; use distinct source/target/preview/conflict/empty/committed
      states with patterns/outlines plus a legend.
- [ ] T016: Add direct donor→target mapping workflow on the canvas and synchronized transformed
      minimap preview for exact-grid transforms.
- [ ] T017: Focused selection tests: 64×64 bounds; cross-ADT rectangles; mixed granularity;
      shared-edge ownership; lasso; modifiers; explicit targets; 100 boundary-coordinate cases.
- [ ] **Gate 3**: build 0 errors; focused selection tests green; operator selects a non-rectangular
      multi-ADT source and explicit target in under 30 seconds without raw coordinates (SC-013/014).

## Phase 4 — Configurable 3D selection + legacy manipulator retirement (US7, US9)

- [ ] T018: Project the shared selection onto terrain in 3D with configurable tile/chunk/cell hover,
      click, paint, rectangle/frustum drag, add/subtract/replace/toggle, and source/target overlays.
- [ ] T019: Gate terrain selection behind an explicit tool mode and integrate with camera navigation,
      scene/object picking, UI mouse capture, Escape/cancel, and selection visibility controls.
- [ ] T020: Make 2D and 3D surfaces read/write the same selection instance and operation preview;
      add synchronization and no-duplicate-state tests.
- [ ] T021: Replace `ChunkManipulatorEditorPlugin` UI/clipboard/paste target with a compatibility
      redirect into the workbench; migrate only proven coordinate/undo services and delete or disable
      the incomplete parallel paste path.
- [ ] **Gate 4**: build 0 errors; operator creates in 3D, edits in 2D, and verifies one undo history;
      source audit finds no second active transform/paste pipeline (SC-015, SC-018, SC-020).

## Phase 5 — Cell-granular translation + boundary re-slicing (FR-020–FR-022)

- [ ] T022: Add canonical integer `CellOffsetX/Y` to `PhaseLayerSettings`; define exact conversion
      helpers and compatibility views: 8 cells = 1 chunk, 128 cells = 1 ADT tile.
- [ ] T023: Add phase-agnostic Core translation/re-slicing over source + target chunk/ADT
      neighborhoods; move heights/normals, holes, MCLY/MCAL, MCSH, MCCV, liquid, MDDF/MODF, and
      MODF bounds together without interpolation.
- [ ] T024: Wire cell offsets into `ResolveTileSource`, selected-content operations, and both
      adapters while retaining exact zero-offset, whole-chunk, and whole-tile fast paths.
- [ ] T025: Add UI units for Tile / Chunk / Cell over the single canonical cell offset; show the
      effective cell, chunk, tile, and world-yard displacement.
- [ ] T026: Add optional WDL-assisted bounded cell-offset search by consuming Spec 196's existing
      sampler; display candidate offset + fit score and require acceptance.
- [ ] T027: Tests — 8-cell equals one-chunk translation; 128-cell equals one-tile translation;
      inverse round-trip; seams across chunk/ADT boundaries; placements seated; disabled channels
      untouched; declining WDL snap makes no change.
- [ ] **Gate 5**: build 0 errors; equivalence tests green; operator aligns a real patch whose
      boundaries do not match at one-cell precision.

## Phase 6 — 45°/free angle + approximation preview (US2, US3)

- [ ] T028: Free-rotate terrain re-sourcing (grid-snapped lookup, no resampling); transform
      placements exactly and report `RotationApproximation` per layer.
- [ ] T029: Extend orthographic and 3D target previews for 45°/free-angle transformed coverage;
      never depict an exact rotated terrain surface when only grid-snapped sourcing is available.
- [ ] T030: Add determinism tests for rotate → mirror → cell offset → per-selection/per-tile target
      override, including reload and two independent phase layers.
- [ ] **Gate 6**: build 0 errors; operator confirms 45° matches its stated preview mode and objects
      remain seated (SC-003/004/005).

## Phase 7 — Save Transformed Map + writer/readback gates (US10)

- [ ] T031: Define Core save-preflight result: target format, destination, affected tiles, enabled/
      omitted channels, off-map data, conflicts, lossy conversions, refusal reasons, and output paths.
- [ ] T032: Build one composition materializer from the preview state; route supported LK-v18 ADT
      output through the existing ADT writer and Alpha output through the existing frozen Alpha WDT
      writer/conversion seam. Do not add parallel writers or native-MoP relabeling.
- [ ] T033: Add **Save Transformed Map...** to the workbench page with target selection, output-copy
      default, overwrite confirmation, complete preflight, progress, cancellation, and final report.
- [ ] T034: Add writer readback tests for terrain channels, placement positions/orientations/bounds,
      name tables, affected-tile set, refusal atomicity, and unsupported native split target.
- [ ] T035: Operator saves and reloads one supported ADT composition and one Alpha WDT composition;
      record exact output roots/builds and compare against preview.
- [ ] **Gate 7**: build/tests/readback green; operator reload proof passes; no source client file is
      overwritten by default and unsupported output creates no mislabeled files (SC-019).

## Phase 8 — Regression, performance, and continuity gate

- [ ] T036: Pixel-comparison regression — no-transform map identical to pre-feature; 90° CW then
      CCW round-trip; mirror involution; base and phase preset restoration.
- [ ] T037: Validate orthographic/3D hover and drag responsiveness without synchronous whole-map
      loading; record frame-time and allocation evidence without claiming visual proof from metrics.
- [ ] T038: Update Specs 195/196/197/203 cross-references, `STATUS.md`, `progress.md`, and
      `activeContext.md`; mark 195 superseded only after compatibility migration passes.
- [ ] **Gate 8**: full solution build + affected tests; operator canvas/3D/save reload verification.
