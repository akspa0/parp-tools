# Receipt — Spec 232 T064: magnetic WDL edge-snap for composed layers

**Date**: 2026-09-09 · **Spec**: [232](../spec.md) · **Task**: T064 (operator directive)

## Directive

"magnetic WDL microlattices that we have for other alignment uses" — a cut tile raised onto a base
map (Teldrassil out of Kalidar onto Kalimdor) leaves a height cliff at the seam; the operator
asked for the existing WDL magnetization mechanism to blend composed tile borders into the
surrounding terrain without manual Z nudging.

## Design

- New pure-math service [`PhaseEdgeBlender`](../../../src/core/WowViewer.Core.Runtime/World/Terrain/Stratigraphy/PhaseEdgeBlender.cs)
  (Core.Runtime, beside [`WdlLatticeMagnetizer`](../../../src/core/WowViewer.Core.Runtime/World/Terrain/Stratigraphy/WdlLatticeMagnetizer.cs)):
  blends ONLY the tile-boundary outer vertices of a contributed chunk's 145-height array toward
  `WdlLatticeMagnetizer.SampleWdlHeight` on the base map's WDL 17×17 lattice at the same world
  position. Edges shared with a neighbor target tile that also receives this layer's heights are
  left untouched (interior seams stay intact); interior chunks are never modified.
- Per-layer setting `PhaseLayerSettings.EdgeBlendWdl` (0 = off, 0..1 strength) + project
  round-trip + layer-card slider ("WDL edge-snap (magnetic)") with reset.
- Both adapters apply the blend in `MergePhaseTile` after `PhaseLayerZ.Apply` (Z moves the tile
  first; the edge blend then pulls footprint-boundary edges toward the base surface), gated on
  the layer contributing the heightmap channel and strength > 0.
- Host wiring: `BaseWdlTileLookup` on both adapters, fed by
  [`ViewerApp.WireBaseWdlEdgeBlendLookup`](../../../src/viewer/WoWViewer/ViewerApp.cs) from the
  already-parsed base-map WDL (`_terrainWeakSignalWdlData` — the same cache the stratigraphy
  weak-signal path uses; no second parse).

## Files changed

| File | Change |
|---|---|
| [PhaseEdgeBlender.cs](../../../src/core/WowViewer.Core.Runtime/World/Terrain/Stratigraphy/PhaseEdgeBlender.cs) | New: boundary-vertex blend math (pure, testable). |
| [PhaseComposition.cs](../../../src/core/WowViewer.Core/Maps/PhaseComposition.cs) | `PhaseLayerSettings.EdgeBlendWdl` + clone. |
| [PhaseLayerProjectFile.cs](../../../src/core/WowViewer.Core/Maps/PhaseLayerProjectFile.cs) | `edgeBlendWdl` project round-trip. |
| [AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs) / [StandardTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs) | `BaseWdlTileLookup` property + `ApplyEdgeBlend` in `MergePhaseTile`. |
| [ViewerApp.cs](../../../src/viewer/WoWViewer/ViewerApp.cs) | `WireBaseWdlEdgeBlendLookup` + `ResolveBaseWdlTile` (shares the stratigraphy WDL parse); wired at both terrain-manager assignment sites. |
| [ViewerApp_PhaseLayers.cs](../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | "WDL edge-snap (magnetic)" slider + reset in the layer card. |
| [PhaseEdgeBlenderTests.cs](../../../tests/WowViewer.Core.Tests/Maps/PhaseEdgeBlenderTests.cs) | 4 tests: west-edge blend, contributing-neighbor skip, zero-strength no-op, interior-chunk no-op. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet test ... --filter "FullyQualifiedName~PhaseEdgeBlenderTests"` | 0 | **4/4 passed**. |
| `dotnet test ... --filter "FullyQualifiedName~Maps"` | 0 | **181/181 passed**. |
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| Boundary vertices blend toward the base WDL lattice at footprint edges | `BlendTileBoundary_PullsWestEdgeTowardTheWdlLattice` (unit). | Pass (unit) |
| Interior seams between contributing tiles are untouched | `BlendTileBoundary_SkipsEdgesSharedWithAContributingNeighbor` (unit). | Pass (unit) |
| A cut tile meets surrounding terrain without a manual Z nudge | Requires the real configured map (e.g. Teldrassil on Kalimdor); visual witness operator-owned. | Open, operator-owned |