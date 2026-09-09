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

## Phase 4 — T015 seam fix: full-tile lattice rotation (KNOWN DEFECT, highest priority)

The quarter-turn rotation re-slots chunks correctly but rotates each chunk's content
independently — ADT chunks share edge vertices, so the seams scramble (operator screenshot
2026-09-07). Fix: rotate the donor tile's FULL-TILE lattices as single grids, then slice.

- [x] T015a Add `AlphaTileData.RotateQuarterTurn(int quarterTurns, bool mirrorH, bool mirrorV)`
      rotating every full-tile lattice about the tile center with ONE consistent index map:
      `Heightmap` 257×257 `[py, px]`, `McnrNormalXyz[py, px, 0..2]` (positions rotated + normal
      components per the kind's normal map), `McshShadowMask1024` 1024×1024, `McshShadowMask256`,
      `McalAlphaPack[256, 256, layer]` (first two axes per layer), `MclyTextureIds`/
      `MclyLayerMask` 16×16×L, `HoleMask`/`HoleFullMasks`/`AreaIds`/`McnkFlags16` 16×16 (doc says
      `[chunkY, chunkX]`), `MclqSurfaceHeight`/`MclqTypeMask` if square, `LiquidChunks`
      (IndexX/IndexY via the 16×16 map; local Heights 9×9 / TileGrid 4×4 / TileFlags 8×8 rotated
      about their own centers). Mirrors: reflect after rotation, same axes as
      `InverseRotateCellDelta` documents.
- [x] T015b Verify direction consistency BEFORE rendering: unit test — rotate a synthetic
      257×257 heightmap 90°, slice via `ToTileLoadResult`, and assert chunk (0,0) heights equal
      the source lattice region that `TransformChunkSlot(Rotate90CW)` names for slot (0,0). If
      the direction disagrees with `ResolveTileSource`'s tile mapping, flip the rotation — the
      tile map and the content map MUST agree.
- [x] T015c Alpha adapter: in the transformed path, replace per-chunk
      `TransformChunksForTarget` with tile-level rotation — expose
      `AlphaTerrainAdapter.GetTileData(tileX, tileY)` (parse without slicing), rotate via T015a,
      then `ToTileLoadResult(tileX, tileY)` (target tile coords → WorldPositions re-homed for
      free). Cell offsets still resolve the supplying tile via `ResolveCellShiftedChunk`.
- [x] T015e MCAL alpha repair (operator defect report 2026-09-08: "MCLY layers on overlapped maps
      broke; worked in the last build"): `ToTileLoadResult` sliced MCAL alpha from the 256×256
      downsampled pack at a 64-px-per-chunk stride — chunks past (3,3) decoded silent zero alpha,
      so every transformed overlapped tile rendered flat single-texture patches. The reader now
      also carries the full-resolution 1024×1024 pack (`McalAlphaPackFull`), `ToTileLoadResult`
      slices from it (256-pack 4× nearest-upsample fallback), and `RotateQuarterTurn` rotates the
      full plane with the same index map. Synthetic tests cover chunk-level alpha slicing, the
      packed fallback, and rotation equivalence with the proven per-chunk alpha transform.
      Receipt: [t015e mcal alpha repair](evidence/t015e-mcal-alpha-repair-receipt.md).
- [ ] T015d Gate: rotated DeadminesInstance renders seam-free (operator screenshot: coastline and
      roadway continuous across chunk boundaries) AND its texture layers blend correctly (the
      T015e witness); cell fine-tune still works on the rotated layer; build/test clean; receipt.
      T015e is build/test-verified only — the visual MCLY witness remains operator-owned.

## Phase 5 — Composition UX completion

- [x] T050 FR-13: donor-tile picker places ONLY the requested tiles — per-layer mode where
      placed tiles compose exclusively (whole-layer offset ignored for unplaced targets), UI
      toggle to return to offset mode. Receipt: [t050 placed tiles only](evidence/t050-placed-tiles-only-receipt.md).
- [ ] T051 FR-14: per-tile locks — locked composed tiles are claimed exclusively by their owning
      layer; later layers skip locked targets; locks persist in the project and get a minimap
      badge. Implementation receipt: [t051 per-tile locks](evidence/t051-per-tile-lock-implementation-receipt.md);
      unchecked pending its operator minimap badge/override witness.
- [ ] T052 FR-15: locked layers protected from deletion (DONE 2026-09-08 — Remove disabled for
      locked, Clear keeps locked with a status message; commit 82889edd).
- [ ] T053 REGRESSION: WL* liquid click-inspector broken with maps placed — audit and repair
      recorded in [t053 WL inspector fall-through](evidence/t053-wl-inspector-fallthrough-receipt.md).
      The minimap footprint path was not the click consumer; composed terrain occlusion cleared
      the WL hover identity before viewport click selection. Unchecked pending the required
      operator with/without-layer-stack inspector witness.
- [ ] T054 FR-4: Archaeology workbench defaults to the Map Layers page. Implementation receipt:
      [t054 Archaeology Map Layers default](evidence/t054-archaeology-map-layers-default-receipt.md).
      Unchecked pending the operator's default-entry/remembered-page UI witness.
- [ ] T055 FR-5: UniqueId era color-coding — objects in a UniqueId range tinted per-range.

## Phase 6 — Renderer + output

- [ ] T060 FR-6 REGRESSION: MDX fire missing, water dark — audit the object-emissive/light
      pipeline changes for the break; restore fire/water/light-casting effects.
- [ ] T061 FR-12: synthesized minimaps consume the COMPOSED loaded map (base + layers,
      post-transform), not only a folder or client map.
- [ ] T062 Phase 3 (FR-3): full-map export — compose every occupied tile through
      `PhaseCompositionPolicy` and write complete client-format map sets (Alpha WDT/ADT for
      0.x; LK route via existing converters), background execution + progress + integrity
      report; exported tiles must match live composition (SC-5).
- [ ] T063 FR-10: texture restoration for stripped phase maps — per-layer/per-tile remapping of
      base-map textures onto texture-less MCLY layers, persisted in the project.

## Dependencies

T010 → T011 → T012 → T013; T020 → T021 → T022; T030 → T031 → T032. Phases are independent of
each other after Phase 1. Phase 4 (T015a–d) is the highest-priority defect fix; Phase 5/6 build
on it.
