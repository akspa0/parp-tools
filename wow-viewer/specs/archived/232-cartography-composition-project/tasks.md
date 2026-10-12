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

- [ ] T050 FR-13: donor-tile picker places ONLY the requested tiles — per-layer mode where
      placed tiles compose exclusively (whole-layer offset ignored for unplaced targets), UI
      toggle to return to offset mode. Receipt: [t050 placed tiles only](evidence/t050-placed-tiles-only-receipt.md).
      **Un-checked by 2026-09-10 governance audit (224-T201)**: receipt marks the minimap-visual
      criterion "Pass" on source + full solution build alone, with no operator visual evidence —
      routing/toggle logic is real and unit-tested, but the visual claim violates AGENTS.md §6.
- [ ] T051 FR-14: per-tile locks — locked composed tiles are claimed exclusively by their owning
      layer; later layers skip locked targets; locks persist in the project and get a minimap
      badge. Implementation receipt: [t051 per-tile locks](evidence/t051-per-tile-lock-implementation-receipt.md);
      unchecked pending its operator minimap badge/override witness.
- [ ] T052 FR-15: locked layers protected from deletion (DONE 2026-09-08 — Remove disabled for
      locked, Clear keeps locked with a status message; commit 82889edd).
      **Governance audit 2026-09-10 (224-T201)**: no `evidence/` receipt exists for this task —
      an inline commit reference with no files-changed list, verification commands, or criterion
      table does not meet the AGENTS.md §9.2 receipt bar. Not a claim the guard logic is wrong,
      only that it is unreceipted; already correctly unchecked.
- [ ] T053 REGRESSION: WL* liquid click-inspector broken with maps placed — audit and repair
      recorded in [t053 WL inspector fall-through](evidence/t053-wl-inspector-fallthrough-receipt.md).
      The minimap footprint path was not the click consumer; composed terrain occlusion cleared
      the WL hover identity before viewport click selection. Unchecked pending the required
      operator with/without-layer-stack inspector witness.
- [ ] T054 FR-4: Archaeology workbench defaults to the Map Layers page. Implementation receipt:
      [t054 Archaeology Map Layers default](evidence/t054-archaeology-map-layers-default-receipt.md).
      Unchecked pending the operator's default-entry/remembered-page UI witness.
- [ ] T055 FR-5: UniqueId era color-coding — objects in a UniqueId range tinted per-range.
- [ ] T056 OPERATOR DIRECTIVE 2026-09-09 ("donor tiles are not where they belong... it doesn't put
      the tile at the target xx_yy location"): the donor-tile picker interpreted its X/Y inputs in
      internal (row, column) order while the operator enters ADT-name order (xx = column, yy =
      row — the status bar prints `{tileY}_{tileX}`). Every placement landed on the diagonal
      mirror of the requested location. The picker now labels and reads xx/yy and maps them to
      (tileX = yy, tileY = xx); the placed-lock labels print in xx_yy order. Receipt:
      [t056 placement coordinate order](evidence/t056-placement-coordinate-order-receipt.md).
      **Un-checked by 2026-09-10 governance audit (224-T201)**: the linked receipt filename does
      not exist on disk (phantom link) — real content lives in the combined
      `t056-t059-placement-z-minimap-receipt.md`, which itself marks the actual placement-lands-
      correctly criterion "Open, operator-owned."
- [ ] T057 OPERATOR DIRECTIVE 2026-09-09 ("we need a way for phase maps to have their base Z
      changed, or maybe even scaling up the Z"): per-layer `ZOffset` (world-Z translation, yards)
      and `ZScale` (multiplier) applied as `z' = z * ZScale + ZOffset` to contributed terrain
      heights, liquid surfaces, and placement Z in both adapters; layer-card inputs + project
      persistence. Receipt: [t057 layer Z transform](evidence/t057-layer-z-transform-receipt.md).
      Unchecked pending the operator's visual witness (e.g. Teldrassil raised onto Kalimdor).
      **Governance audit 2026-09-10 (224-T201)**: the task's own text already said "Unchecked
      pending..." while the checkbox read `[x]` — a literal self-contradiction. Corrected to `[ ]`.
      Also: the linked receipt filename does not exist on disk (real content is in the combined
      `t056-t059-placement-z-minimap-receipt.md`).
- [ ] T058 OPERATOR DIRECTIVE 2026-09-09 ("the full screen minimap's teleport (3 clicks) didn't
      work, at all"): every minimap surface shared one `MinimapInteractionState`, so the other
      surfaces re-processed and cleared each press/release before three clicks could accumulate.
      Each surface now owns its pointer state. Build-verified; interactive witness operator-owned.
      **Governance audit 2026-09-10 (224-T201)**: checkbox read `[x]` while the task's own text
      says "interactive witness operator-owned" — self-contradiction, corrected to `[ ]`.
- [ ] T059 OPERATOR DIRECTIVE 2026-09-09 ("let us drag and drop the tile on the full-screen
      minimap, and place it/lock it to a region"): placed-only layers are now grabbable on the
      minimap — dragging moves every placement's target by the pointer delta (donor slot fixed),
      re-streams once on release, and the lock flags ride along. Interactive witness operator-owned.
      **Governance audit 2026-09-10 (224-T201)**: checkbox read `[x]` while the task's own text
      says "interactive witness operator-owned" — self-contradiction, corrected to `[ ]`.
- [ ] T064 OPERATOR DIRECTIVE 2026-09-09: magnetic WDL microlattice edge-snapping — implemented
      as a per-layer `EdgeBlendWdl` strength (0 = off): the tile-boundary outer vertices of
      contributed heightmap chunks blend toward the base map's WDL 17×17 macro lattice
      ([PhaseEdgeBlender](../../../src/core/WowViewer.Core.Runtime/World/Terrain/Stratigraphy/PhaseEdgeBlender.cs),
      Core.Runtime beside WdlLatticeMagnetizer), applied in both adapters' `MergePhaseTile` after
      the Z transform. Edges shared with a neighbor target tile that also receives the layer's
      heights stay untouched — only footprint-boundary edges blend. Host wiring feeds the parsed
      base WDL to the adapters (`BaseWdlTileLookup`, shared parse with the stratigraphy path);
      layer-card slider + project persistence. Maps tests 181/181; build 0 errors. Receipt:
      [t064 wdl edge snap](evidence/t064-wdl-edge-snap-receipt.md). Visual witness operator-owned
      (e.g. Teldrassil on Kalimdor).
      **Governance audit 2026-09-10 (224-T201)**: checkbox read `[x]` while the task's own text
      says "Visual witness operator-owned" and the receipt's decisive criterion ("cut tile meets
      surrounding terrain without manual Z nudge") is marked "Open, operator-owned" — only the
      underlying blend-math sub-checks passed via unit test. Corrected to `[ ]`.
- [ ] T065 OPERATOR DIRECTIVE 2026-09-09: re-audit chunk-level MCAL/MCLY "off-by-one scrambling"
      and cross-chunk height smoothness inside composed tiles after T056 (the transposed targets
      made composed content appear scrambled); witness-driven.
- [ ] T066 OPERATOR DIRECTIVE 2026-09-09 ("cell finetuning shifts the cells around instead of
      just moving the TILE to the cells... applying the heightmap still does this horrific shit";
      amended after the first pass: "it's aligned right, but we're missing stuff in between"):
      the cell-shifted path re-resolved a supply tile PER CHUNK (ResolveCellShiftedChunk +
      ResolveTileSource), which under rotation/cell offset composed checkerboard garbage and
      pulled border chunks from unrelated donor tiles. Both adapters' `BuildCellShiftedTile` are
      now LAYER-RIGID: each target tile collects its own donor tile's retained content plus the
      3×3-neighborhood spill (contributors resolve their own donor through the shared tile map,
      rotation included; per-axis ranges disjoint at |offset| ≤ 15), so nothing drops between
      tiles and no chunk is re-picked from an unrelated tile. Receipt:
      [t066 tile-rigid cell shift](evidence/t066-tile-rigid-cell-shift-receipt.md). Visual
      witness operator-owned.
      **Governance audit 2026-09-10 (224-T201)**: checkbox read `[x]` while the task's own text
      says "Visual witness operator-owned" and the receipt marks BOTH of its acceptance criteria
      "Open, operator-owned" — 0 of 2 rows have evidence beyond code review. Corrected to `[ ]`.

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

## Phase 7 — Operator-reported gaps 2026-09-10

Operator directive, verbatim intent preserved per item. None of these have been investigated yet
— recorded here so they are not lost, per AGENTS.md §9.1 (operator-originated wording may enter a
spec directly). No implementation claimed.

- [ ] T067 OPERATOR DIRECTIVE 2026-09-10: remove the "Bake MCSH shadows" checkbox from the
      synthesized-minimap export UI permanently
      (`ViewerApp_SynthesizedMinimapExport.cs:115`, `_synthesizedMinimapBakeMcsh`). Operator:
      "We should not ever have any option to bake MCSH shadows, as they have no use in minimaps,
      ever." This matches prior measurement that MCSH is not encoded in real client minimaps
      (near-zero correlation with minimap luminance) — the option produces misleading output by
      construction, not just an unused feature. Remove the checkbox and the `bakeMcsh` parameter
      path, not just default it off.
- [ ] T068 OPERATOR DIRECTIVE 2026-09-10: "Include WMO geometry" checkbox
      (`ViewerApp_SynthesizedMinimapExport.cs:111`, `_synthesizedMinimapIncludeWmos`) does not
      work — operator report, not yet root-caused. Audit the `includeWmos` path from checkbox
      through to the render/export call.
- [ ] T069 OPERATOR DIRECTIVE 2026-09-10: "our no-water minimaps have weird shading glitches that
      the normal minimaps render fine" — operator suspects terrain shading, unconfirmed. Not yet
      root-caused; reproduce with `--no-water`-equivalent export path (`castShadows`/water toggle
      interaction in `ViewerApp_SynthesizedMinimapExport.cs`) against a normal export on the same
      tile.
- [ ] T070 OPERATOR DIRECTIVE 2026-09-10: cell-level alignment fine-tune needs TRUE 1-cell (not
      8-cell) granularity in both X and Y. Operator: "we currently seem to be batching the tile in
      8x8 cell increments, which, while that's right for rendering, is not ample for perfectly
      lining up the oddball copy/pasted map data." Concrete failing case: Hellfire Ramparts does
      not perfectly overlap its Expansion01 tiles — off by 1–3 cells in one direction and 1–2 in
      the other, and current tooling cannot close that gap. This overlaps Phase 1 (T010–T013,
      `CellOffsetX/Y` 0–15 cells, not yet implemented) and the already-shipped-but-buggy cell
      fine-tune referenced in T066 — reconcile which mechanism this is/should be before
      implementing; do not assume they are the same code path without checking.
- [ ] T071 OPERATOR DIRECTIVE 2026-09-10: numeric offset/transform counters (at minimum the cell
      and Z-offset controls in the layers panel) are too small to display a two-digit value with
      its sign at 100% UI zoom — operator cannot tell if a value is positive or negative. Controls
      must scale with the UI text size, not stay a fixed pixel width.

## Dependencies

T010 → T011 → T012 → T013; T020 → T021 → T022; T030 → T031 → T032. Phases are independent of
each other after Phase 1. Phase 4 (T015a–d) is the highest-priority defect fix; Phase 5/6 build
on it.
