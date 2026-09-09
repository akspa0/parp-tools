# Receipt — Spec 232 T056–T059: placement coordinate order, layer Z transform, minimap teleport + placed-tile drag

**Date**: 2026-09-09 · **Spec**: [232](../spec.md) · **Tasks**: T056, T057, T058, T059 (operator directives)

## Defects / directives

1. **T056** — "donor tiles are not where they belong, at all... it doesn't put the tile at the
   target xx_yy location": the donor-tile picker read its X/Y inputs as internal
   (tileX = row, tileY = column), while the operator enters ADT-name order (xx = column W-E, yy =
   row N-S — the status bar prints `Tile: {tileY}_{tileX}`). Every placement landed on the
   diagonal mirror of the requested location, so no setting ever placed the tile where asked.
2. **T057** — "we need a way for phase maps to have their base Z changed, or maybe even scaling up
   the Z" (Teldrassil cut from Kalidar onto Kalimdor has no way to meet its neighbors).
3. **T058** — "the full screen minimap's teleport (3 clicks) didn't work, at all": all minimap
   surfaces (sidebar, cartography panel, fullscreen) shared one
   [`MinimapInteractionState`](../../../src/core/WowViewer.Core.Runtime/World/Minimap/MinimapInteractionState.cs),
   so other surfaces re-processed/cleared every press-release before three clicks accumulated.
4. **T059** — "let us drag and drop the tile on the full-screen minimap, and place it/lock it":
   the footprint hit-test explicitly excluded placed-only layers and drags only moved
   `TileOffset`, which placed-only layers ignore.

## Files changed

| File | Change |
|---|---|
| [ViewerApp_PhaseLayers.cs](../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | T056: picker labels/reads xx (column) and yy (row) and maps them to (tileX = yy, tileY = xx); placement summary and placed-lock labels print in ADT xx_yy order. T057: "Z offset / scale" inputs (offset yards, scale multiplier, Reset Z) in the layer card. |
| [PhaseComposition.cs](../../../src/core/WowViewer.Core/Maps/PhaseComposition.cs) | T057: `PhaseLayerSettings.ZOffset`/`ZScale` (+ clone). |
| [PhaseLayerProjectFile.cs](../../../src/core/WowViewer.Core/Maps/PhaseLayerProjectFile.cs) | T057: `zOffset`/`zScale` project round-trip. |
| [PhaseLayerZ.cs](../../../src/viewer/WoWViewer/Terrain/PhaseLayerZ.cs) | T057 (new): owned service applying `z' = z * ZScale + ZOffset` to chunk heights, liquid surface heights, and MDDF/MODF placement Z of a phase tile before the merge. |
| [AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs) / [StandardTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs) | T057: `PhaseLayerZ.Apply` at the top of each adapter's `MergePhaseTile` (covers direct and cell-shifted routes). |
| [ViewerApp_MinimapAndStatus.cs](../../../src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs) | T058: per-surface `MinimapInteractionState` dictionary keyed by interaction id; `PrepareFullscreenMinimapState`/`ClearPendingMinimapTeleport` reset all. T059: placed-only layers are grabbable; drag snapshots placements and rewrites each placement's target by the pointer delta (donor slot + lock fixed), re-streams once on release. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**. |
| `dotnet test ... --filter "FullyQualifiedName~PhaseLayerZTests"` | 0 | **3/3 passed** (round-trip, identity defaults, clone). |
| `dotnet test ... --filter "FullyQualifiedName~Maps"` | 0 | **177/177 passed**. |
| Full-suite run | — | 1521 passed, 10 failed — all 10 in subsystems untouched by this diff (WTF classifier, WorldFramePassCoordinator, enrichment/ModelFootprint readers, AdtV23 summary, LkToAlpha flag contract); pre-existing, tracked for a separate audit. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| T056: a placement entered as xx_yy lands at that ADT location | Code-level coordinate mapping (picker → internal row/col) + operator re-test. | Open, operator-owned |
| T057: a cut tile raised/lowered by Z offset meets adjacent terrain | Requires a real configured map; visual witness operator-owned. | Open, operator-owned |
| T057: Z settings persist across restart | `LayerProjectFile_RoundTripsZSettings` (unit). | Pass (unit) |
| T058: triple-click on the fullscreen minimap teleports | Requires the running viewer; interactive witness operator-owned. | Open, operator-owned |
| T059: dragging a placed tile moves it to a new target and re-streams | Requires the running viewer; interactive witness operator-owned. | Open, operator-owned |

## Notes

- Self-map donor (layer "Azeroth" on base Azeroth) resolves through
  `ResolvePhaseAdapter` → `PhaseWdtPathResolver`, which returns the same extracted WDT — no code
  change required; re-verify with the T056 fix in place.
- The operator's chunk-level MCAL/MCLY "off-by-one scrambling" and cross-chunk height smoothness
  reports are recorded as T065 for re-audit after T056 (transposed targets made composed content
  appear scrambled).
- Magnetic WDL microlattice edge-snapping recorded as T064 (design needed; pairs with T057).
- Earlier persisted layer projects may hold placements created with the transposed convention;
  re-create or re-drag them with the fixed picker.
