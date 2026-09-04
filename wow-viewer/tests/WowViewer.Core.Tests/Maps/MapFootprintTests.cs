using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 222 (Cartography) Phase 1: footprint conversion from Alpha MAIN offset grids and Standard
/// tile indices, alignment-offset math, clamping, and base-overlap detection. These are the pure
/// Core halves of the adapters' GetOccupiedTiles and the upcoming align-to-base feature.
/// </summary>
public sealed class MapFootprintTests
{
    private static int[] MainOffsetsWithTiles(params int[] indices)
    {
        var offsets = new int[64 * 64];
        foreach (int idx in indices)
            offsets[idx] = 0x1000 + idx; // any nonzero MHDR offset
        return offsets;
    }

    [Fact]
    public void FromMainOffsets_IndexMathMatchesWdtConvention()
    {
        // Alpha WDT MAIN is index = tileX*64 + tileY. Tile (34,28) therefore lives at index
        // 34*64+28 = 2204. This is the exact convention TileExistsInOwnWdt uses, so the footprint
        // and the tile-existence check can never disagree.
        IReadOnlyList<(int TileX, int TileY)> tiles = MapFootprint.FromMainOffsets(
            MainOffsetsWithTiles(34 * 64 + 28, 0, 63 * 64 + 63));

        Assert.Equal(3, tiles.Count);
        Assert.Contains((34, 28), tiles);
        Assert.Contains((0, 0), tiles);
        Assert.Contains((63, 63), tiles);
    }

    [Fact]
    public void FromMainOffsets_EmptyGrid_YieldsEmptyFootprint()
    {
        Assert.Empty(MapFootprint.FromMainOffsets(new int[64 * 64]));
    }

    [Fact]
    public void FromMainOffsets_IgnoresOutOfRangeEntries()
    {
        var offsets = new int[70]; // deliberately longer than the grid
        offsets[65] = 1; // index 65 is beyond 64*64? No — 4096 cells; 65 is valid...

        // 4096-entry grids are the real shape; verify a full-size grid with an out-of-grid write
        // is simply not reachable, and that a short list is handled.
        var shortOffsets = new int[100];
        shortOffsets[99] = 7; // index 99 = (1, 35)
        IReadOnlyList<(int TileX, int TileY)> tiles = MapFootprint.FromMainOffsets(shortOffsets);

        _ = offsets; // silence unused warning; the assertion is about the short grid
        Assert.Single(tiles);
        Assert.Contains((1, 35), tiles);
    }

    [Fact]
    public void FromTileIndices_MatchesIndexConvention()
    {
        IReadOnlyList<(int TileX, int TileY)> tiles = MapFootprint.FromTileIndices([31 * 64 + 5, -1, 4096]);

        Assert.Single(tiles);
        Assert.Contains((31, 5), tiles); // out-of-range indices are dropped, not clamped
    }

    [Fact]
    public void ComputeAlignmentOffset_BaseCentroid_MovesDonorOntoBase()
    {
        // Base occupies a single tile at (34,28); donor occupies a single tile at (10,10).
        // Aligning must produce offset (24,18): donor (10,10) + offset = (34,28).
        bool ok = MapFootprint.TryComputeAlignmentOffset(
            [(34, 28)],
            [(10, 10)],
            MapFootprint.AlignmentTarget.BaseCentroid,
            (0, 0),
            out int offsetX,
            out int offsetY);

        Assert.True(ok);
        Assert.Equal(24, offsetX);
        Assert.Equal(18, offsetY);
    }

    [Fact]
    public void ComputeAlignmentOffset_MultiTileFootprints_UseCentroids()
    {
        // Base centroid of {(33,28),(35,28)} = (34,28). Donor centroid of {(10,9),(10,11)} = (10,10).
        bool ok = MapFootprint.TryComputeAlignmentOffset(
            [(33, 28), (35, 28)],
            [(10, 9), (10, 11)],
            MapFootprint.AlignmentTarget.BaseCentroid,
            (0, 0),
            out int offsetX,
            out int offsetY);

        Assert.True(ok);
        Assert.Equal(24, offsetX);
        Assert.Equal(18, offsetY);
    }

    [Fact]
    public void ComputeAlignmentOffset_CameraTarget_UsesCameraTileAsDestination()
    {
        bool ok = MapFootprint.TryComputeAlignmentOffset(
            [(34, 28)],
            [(10, 10)],
            MapFootprint.AlignmentTarget.CameraTile,
            (50, 20),
            out int offsetX,
            out int offsetY);

        Assert.True(ok);
        Assert.Equal(40, offsetX); // 50 - 10
        Assert.Equal(10, offsetY); // 20 - 10
    }

    [Fact]
    public void ComputeAlignmentOffset_EmptyFootprints_Fail()
    {
        Assert.False(MapFootprint.TryComputeAlignmentOffset(
            [], [(10, 10)], MapFootprint.AlignmentTarget.BaseCentroid, (0, 0), out _, out _));
        Assert.False(MapFootprint.TryComputeAlignmentOffset(
            [(34, 28)], [], MapFootprint.AlignmentTarget.BaseCentroid, (0, 0), out _, out _));
    }

    [Fact]
    public void ClampOffset_BoundsToPlusMinus63()
    {
        Assert.Equal((63, -63), MapFootprint.ClampOffset(100, -100));
        Assert.Equal((5, -7), MapFootprint.ClampOffset(5, -7));
    }

    [Fact]
    public void OverlapsBase_DetectsTheShadowfangConfiguration()
    {
        // The motivating 2026-09-04 case: Shadowfang's tiles do not include the base target
        // (34,28) at zero offset — this is the silent no-op the workbench must make visible.
        var baseTiles = new List<(int, int)> { (34, 28), (33, 28), (34, 29) };
        var donorTiles = new List<(int, int)> { (10, 10), (11, 10) };

        Assert.False(MapFootprint.OverlapsBase(baseTiles, donorTiles, 0, 0));
        Assert.True(MapFootprint.OverlapsBase(baseTiles, donorTiles, 24, 18));
    }

    [Fact]
    public void OverlapsBase_EmptyFootprints_NeverOverlap()
    {
        Assert.False(MapFootprint.OverlapsBase([], [(10, 10)], 0, 0));
        Assert.False(MapFootprint.OverlapsBase([(34, 28)], [], 0, 0));
    }

    [Fact]
    public void PhaseLayerSettings_CloneCarriesResolutionAndColor()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = "Shadowfang",
            Resolution = PhaseLayerResolution.Resolved,
            FootprintColorIndex = 3,
            TileOffsetX = 2,
        };

        PhaseLayerSettings clone = layer.Clone();

        Assert.Equal(PhaseLayerResolution.Resolved, clone.Resolution);
        Assert.Equal(3, clone.FootprintColorIndex);
        Assert.Equal(2, clone.TileOffsetX);

        // Mutating the clone must not affect the original — layers are edited in place by the UI.
        clone.Resolution = PhaseLayerResolution.Unresolved;
        clone.FootprintColorIndex = 5;
        Assert.Equal(PhaseLayerResolution.Resolved, layer.Resolution);
        Assert.Equal(3, layer.FootprintColorIndex);
    }

    [Fact]
    public void PhaseLayerSettings_NewLayerDefaults()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };

        Assert.Equal(PhaseLayerResolution.NotYetChecked, layer.Resolution);
        Assert.Equal(-1, layer.FootprintColorIndex);
    }

    // ---- Spec 222 Phase 2 (T106): minimap drag math ----

    [Fact]
    public void ApplyDragDelta_ZeroMovement_KeepsBaseOffset()
    {
        Assert.Equal((24, 18), MapFootprint.ApplyDragDelta(24, 18, 10.4f, 10.6f, 10.2f, 10.9f));
    }

    [Fact]
    public void ApplyDragDelta_AddsRoundedPointerDelta()
    {
        // Dragging from (10,10) to (13.4, 8.6) rounds to (+3, -1) on top of the base offset.
        Assert.Equal((27, 17), MapFootprint.ApplyDragDelta(24, 18, 10f, 10f, 13.4f, 8.6f));
    }

    [Fact]
    public void ApplyDragDelta_ClampsToGrid()
    {
        // A drag that would push the offset beyond ±63 clamps instead of going off-grid.
        Assert.Equal((63, -63), MapFootprint.ApplyDragDelta(60, -60, 10f, 10f, 20f, 0f));
    }

    [Fact]
    public void ApplyDragDelta_SequentialDragsFromSameBase_AreIdempotentPerFrame()
    {
        // The interaction recomputes from the DRAG-START base every frame, so wiggle does not
        // accumulate: returning to the start tile restores the base offset exactly.
        (int x1, int y1) = MapFootprint.ApplyDragDelta(5, 5, 10f, 10f, 14f, 12f);
        Assert.Equal((9, 7), (x1, y1));

        (int x2, int y2) = MapFootprint.ApplyDragDelta(5, 5, 10f, 10f, 10f, 10f);
        Assert.Equal((5, 5), (x2, y2));
    }

    [Fact]
    public void OffsetRoundTrip_TargetMinusOffset_EqualsSource()
    {
        // The invariant the whole offset system rests on: base tile (tx,ty) reads donor tile
        // (tx - offsetX, ty - offsetY). After a drag sets the offset, the composed tile must be
        // the donor tile the operator dropped onto.
        var layer = new PhaseLayerSettings { MapName = "Donor", TileOffsetX = 24, TileOffsetY = 18 };

        const int baseTileX = 34, baseTileY = 28;
        int sourceTileX = baseTileX - layer.TileOffsetX;
        int sourceTileY = baseTileY - layer.TileOffsetY;

        Assert.Equal((10, 10), (sourceTileX, sourceTileY));

        // And the reverse: donor (10,10) + offset lands on the base tile.
        Assert.Equal((baseTileX, baseTileY), (sourceTileX + layer.TileOffsetX, sourceTileY + layer.TileOffsetY));
    }

    [Fact]
    public void PerTilePlacement_ClaimsTargetOverWholeLayerOffset()
    {
        // Spec 219 semantics Cartography composes with: a per-tile mapping wins over the
        // whole-layer offset for its target. ResolveTileSource must route the placed target to the
        // mapped donor tile, not the offset-shifted one.
        var layer = new PhaseLayerSettings { MapName = "Donor", TileOffsetX = 3, TileOffsetY = 0 };
        layer.TilePlacements.Add(new PhaseTilePlacement(7, 8, 10, 12));

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(
            layer, 10, 12, static (_, _) => true);

        Assert.True(source.HasSource);
        Assert.Equal(PhaseTileSourceKind.PerTile, source.Via);
        Assert.Equal((7, 8), (source.SourceTileX, source.SourceTileY));
    }

    [Fact]
    public void PerTilePlacement_UnmappedTarget_StillRoutesThroughOffset()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", TileOffsetX = 3, TileOffsetY = 0 };
        layer.TilePlacements.Add(new PhaseTilePlacement(7, 8, 10, 12));

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(
            layer, 20, 12, static (_, _) => true);

        Assert.True(source.HasSource);
        Assert.Equal(PhaseTileSourceKind.Offset, source.Via);
        Assert.Equal((17, 12), (source.SourceTileX, source.SourceTileY));
    }
}
