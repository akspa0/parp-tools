# Tasks — Spec 232 Cartography Composition

Phases per the [spec](spec.md). Receipts per AGENTS.md §9.2. Implementation in a fresh session.

## Phase 1 — Cell-level alignment (FR-1)

- [ ] T010 Core: extend `PhaseLayerSettings` with `CellOffsetX/Y` (0–15 cells, composing into the
      tile lookup); extend `PhaseComposition.ResolveTileSource`/new `ResolveChunkSource` so a
      composed chunk can pull from the neighboring donor tile when a cell shift crosses a border;
      carry the within-chunk offset to the content transform
- [ ] T011 Adapters: apply the cell shift to terrain channels (heights, normals, holes, alpha,
      shadows, MCCV, liquid, area) and to placement world positions in both Alpha and Standard
      adapters via the shared policy
- [ ] T012 UI: cell nudge controls in the layers panel (per-axis, with fine steps + reset), and
      minimap footprint rendering at cell granularity
- [ ] T013 Gate: SC-1 visual receipt (roadway alignment case — DeadminesInstance vs Azeroth
      Moonbrook), build/test clean

## Phase 2 — Project persistence (FR-2)

- [ ] T020 Serialization: per-base-map project JSON (layer stack: order, channels, offsets, cell
      offsets, rotation + origin, mirrors, per-tile mappings, presence gates, locks) under the
      viewer project output area; save on demand + auto-load on launch/map load
- [ ] T021 Locks: locked badge + edit rejection in the layers panel; explicit unlock
- [ ] T022 Gate: SC-2 reload receipt (save → restart → identical composition); SC-3 lock receipt

## Phase 3 — Full-map export (FR-3)

- [ ] T030 Export pipeline: compose every occupied base tile through `PhaseCompositionPolicy`
      (same path as live rendering) and write a complete client-format map set (Alpha WDT/ADT for
      0.x; LK route via existing converters where the target build needs it) — full tiles, not
      diffs; background execution + progress + re-entrancy guard
- [ ] T031 Export UI: format/build selection, output path (project-managed), progress, and a
      post-export integrity report (tiles written, channels per tile, placements counts)
- [ ] T032 Gate: SC-4 client-load validation (operator) + SC-5 parity spot-check (exported tile
      vs live composition on sampled tiles); receipt

## Dependencies

T010 → T011 → T012 → T013; T020 → T021 → T022; T030 → T031 → T032. Phases are independent of
each other after Phase 1.
