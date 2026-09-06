# Tasks: Overhead Orthographic World View (Spec 225)

**Spec**: [spec.md](spec.md)

## Phase 0 — Shipped with the 2026-09-06 session (receipts inline)

- [x] 225-T001: Reorder bottom-toolbar grid controls to **Tiles, Chunks, Cells** (FR-1 partial).
      **Receipt**: `ViewerApp_Sidebars.cs` `DrawBottomBar()` reordered; viewer Debug build 0 errors.
      Operator visual check owed (runtime).
- [x] 225-T002: Cells overlay anti-aliasing + neon glow + fog fade in both terrain shaders, so the
      grid no longer moirés in the distance (US2 legibility prerequisite).
      **Receipt**: `TerrainRenderer.cs` main + tile shaders rewritten with `fwidth`-based width,
      glow halo, sub-pixel detail fade, and `fogFactor²` dissolve; build 0 errors. Operator visual
      check owed (runtime).

## Phase 1 — Overhead view mode (not started)

- [ ] 225-T101: Decide the projection owner — world renderer with ortho top-down projection vs.
      specialized minimap-synthesizer view (US1). Record the decision + evidence before code.
- [ ] 225-T102: Implement the Overhead bottom-toolbar toggle (first position) and camera
      save/restore around the mode (FR-4).
- [ ] 225-T103: Render the map imagery + route the existing Tiles/Chunks/Cells overlay data into
      the overhead view (FR-2/FR-3).
- [ ] **Gate 1**: SC-1 walkthrough by the operator; build 0 errors; no new grid pipeline (SC-2).
