# Feature Specification: Overhead Orthographic World View with Grid Overlays (Spec 225)

**Feature Branch**: `225-overhead-ortho-view`
**Created**: 2026-09-06
**Status**: Draft — authored from operator directive, not yet implemented
**Input**: Operator directive 2026-09-06 (verbatim intent): "the overhead view needs to work in the
renderer or a specialized view that uses the minimap synthesizer orthogonal view, with a grid overlay
generated from the existing overlay data we have for Tiles/Chunks/Cells. We should also put the
options for each in that order on the bottom toolbar, it's otherwise really confusing."

## Context & Motivation

The viewer has no working top-down view of the loaded world. The only "overhead" surface today is
the Chunk Manipulator's ImGui canvas, which is not the renderer and does not use the map-wide
imagery the minimap synthesizer already produces. Meanwhile the Tiles/Chunks/Cells grid toggles
live on the bottom toolbar in an arbitrary order (Chunks, Tiles, Cells), which the operator finds
confusing.

The minimap synthesizer already renders an orthographic map-wide view of loaded tiles; the terrain
renderer already draws screen-space Tiles/Chunks/Cells grid overlays. This spec composes those
existing capabilities into one overhead viewing mode instead of inventing a parallel pipeline.

## User Stories & Testing

### US1 — Overhead view over the loaded world (P1)

The operator toggles an **Overhead** option (first position, before Tiles) on the bottom toolbar.
The renderer switches to an orthographic top-down view of the loaded world — either by driving the
existing world renderer with an orthographic top-down projection, or by presenting the minimap
synthesizer's orthogonal output as a specialized view. Toggling it off returns exactly to the prior
perspective camera state.

**Acceptance criteria**:
- Overhead is a bottom-toolbar toggle in the order **Overhead | Tiles | Chunks | Cells**.
- Entering overhead does not lose camera position/orientation; exiting restores them.
- The world streams tiles around the overhead focus (not the old camera position) or shows the
  already-loaded map imagery — no freeze, no blank view.
- The 2D shell, selection, and capture behavior are unchanged when overhead is off.

### US2 — Grid overlays in overhead (P1)

While overhead is active, the existing Tiles/Chunks/Cells overlay data renders as a grid on top of
the map view, driven by the same toggles in the new toolbar order.

**Acceptance criteria**:
- Tile boundaries (533.33 yd), chunk boundaries (33.33 yd), and cell boundaries (2.083 yd) each
  draw from the existing overlay sources — no second grid implementation.
- Grid lines follow the distance-fade/anti-aliased treatment shipped for the Cells overlay
  (2026-09-06): no moiré, no sub-pixel shimmer, glow may be toggled per grid.
- Toggles persist with the existing renderer state (single source of truth; FR-6 of Spec 223).

### US3 — Fog-independent legibility (P2)

The overhead view remains legible regardless of fog settings because it is a map view, not a flight
view; but any 3D-perspective fallback rendering inside it still respects the user's fog override.

## Functional Requirements

- **FR-1 — Toolbar order**: bottom-bar grid controls read `Overhead, Tiles, Chunks, Cells`.
  (Tiles/Chunks/Cells reordering shipped 2026-09-06; the Overhead toggle ships with this spec.)
- **FR-2 — Reuse, do not fork**: the orthographic imagery comes from the world renderer with an
  ortho projection or the minimap synthesizer — no third map-rendering path.
- **FR-3 — Grids from existing data**: overlays come from the same tile/chunk/cell sources the
  3D grid toggles already use.
- **FR-4 — State preservation**: toggling overhead preserves and restores the perspective camera.

## Success Criteria

- **SC-1**: With a map loaded, one click on Overhead shows the whole loaded area top-down with
  selectable grid density; one more click returns to the exact prior view.
- **SC-2**: No new shader/mesh pipeline exists solely for the overhead grids.

## Dependencies

- Minimap synthesizer (orthographic tile imagery) — existing.
- TerrainRenderer grid overlays (`ShowTileGrid`/`ShowChunkGrid`/`ShowCellGrid`) — existing,
  anti-aliased + fog-faded as of 2026-09-06.
- Spec 223 FR-6 (mirror, never fork) applies to the toolbar controls.

## Stakeholders

- Operator: requested 2026-09-06; accepts the mode after a real-client walkthrough.
