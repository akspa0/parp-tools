# Quickstart: Phase Layer Rotation Tools

> The delivery target is now the Map Composition Selection & Transform Workbench. The checks below
> remain valid and are extended by the whole-map/3D/base-layer/save gates in `tasks.md`.

**Feature**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)
**Created**: 2026-09-03

Operator-facing verification for each phase. All visual proof is operator-owned; the agent
prepares builds and reads logs.

## Phase 1–2: 90° rotation + per-tile placement

1. Load a base map with a phase layer (Spec 203 flow).
2. In **Phase Map Layers**, expand the layer and set **Rotation 90° CW**.
3. Confirm:
   - The layer's terrain features appear rotated 90° clockwise about the reported origin.
   - Objects (doodads/WMOs) rotate with the terrain — nothing floats or detaches.
   - The log shows `[Phase] tile source (tx,ty) <- donor (dx,dy) via Rotation`.
4. Switch to **90° CCW**: the map returns to the unrotated arrangement (SC-002).
5. Set rotation back to 0: the composed map is pixel-identical to before (SC-006).
6. **Per-tile placement**: pick a donor tile and drop it on a target base tile.
   - The target tile shows the donor tile's content; all other tiles unchanged (SC-007).
   - The log shows `via PerTile` for that target.
7. Map two donor tiles onto the same target: the last one wins and the conflict is reported
   (FR-015).
8. Confirm every phase-layer card exposes 90° CW/CCW, 45° CW/CCW, Mirror H/V, free angle, visible
   origin, approximation, and reset controls.
9. Disable base Heightmap, Objects, or all base channels and confirm suppression occurs before phase
   composition; exercise Base Only and Phases Only presets.
10. In the orthographic whole-map canvas, switch among minimap/heightmap/occupancy backdrops and
    tile/chunk/cell grids; create one non-rectangular source selection and an explicit target.
11. Edit the same selection in 3D and verify the orthographic canvas updates immediately.
12. Open the legacy Chunk Manipulator entry and confirm it redirects to the same workbench state.
13. Run Save Transformed Map preflight. Reload supported ADT and Alpha WDT output copies; verify an
    unsupported native split target is refused before any file is written.

## Mirroring (Phase 1–2)

1. Set **Mirror Horizontal** on the layer.
2. Confirm terrain features mirror left-right; objects mirror position AND face the mirrored
   direction (handedness flip — an object that faced East now faces West).
3. Toggle Mirror Horizontal off: the map returns to the unmirrored composition.
4. Repeat for **Mirror Vertical** (top-bottom).
5. Mirror twice in the same axis: the result is identical to unmirrored (SC-009 involution).

## Phase 3: 45° and free angle

1. Set **Rotation 45° CW** on the layer.
2. Confirm the panel/log states the approximation mode (`FreeRotate`) — never silent (SC-003).
3. Confirm objects remain seated on the donor terrain they came from (SC-004).
4. Enter an arbitrary angle (e.g. 17°): terrain re-sources by rotated coverage; objects rotate
   exactly; the result is stable across reloads (SC-005).

## Phase 4: Regression gates

1. With rotation 0 and no per-tile mappings, confirm the map renders identically to the
   pre-feature build (SC-006).
2. Combine an offset and a rotation on one layer: the result is deterministic and the applied
   order (rotate → offset → per-tile) is stated in the logs (SC-005).
3. Check the minimap: rotated layer tiles appear at their rotated positions (90° exact; 45° per
   the stated minimap limitation, if any).

## Log lines to send with any defect report

- `[Phase] tile source (tx,ty) <- donor (dx,dy) via <kind>` — which donor filled which target.
- `[Phase] rotation origin (ox,oy), approximation <mode>` — per-layer rotation state.
- `[Phase] tile placement conflict: target (tx,ty) claimed by ...` — conflict reports.
- `[AlphaADT] Phase offset mapping ...` / `[StandardADT] Phase patch ...` — existing 203 lines,
  unchanged, for composition context.
