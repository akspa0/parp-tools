namespace WowViewer.Core.Maps;

/// <summary>
/// Cartography (Spec 222): tile-footprint helpers shared by the Alpha and Standard terrain
/// adapters. Lives in Core so the footprint math and the alignment computation are unit-testable
/// without the OpenGL viewer.
/// </summary>
public static class MapFootprint
{
    /// <summary>
    /// Converts an Alpha WDT MAIN offset grid (4096 entries, index = tileX*64+tileY, entry = MHDR
    /// file offset where nonzero means the tile exists) into (tileX, tileY) pairs.
    /// </summary>
    public static IReadOnlyList<(int TileX, int TileY)> FromMainOffsets(IReadOnlyList<int> mainOffsets)
    {
        var tiles = new List<(int, int)>(mainOffsets.Count);
        for (int i = 0; i < mainOffsets.Count && i < 64 * 64; i++)
        {
            if (mainOffsets[i] != 0)
                tiles.Add((i / 64, i % 64));
        }

        return tiles;
    }

    /// <summary>
    /// Converts a set of existing tile indices (tileX*64+tileY) into (tileX, tileY) pairs — the
    /// Standard-side counterpart of <see cref="FromMainOffsets"/>.
    /// </summary>
    public static IReadOnlyList<(int TileX, int TileY)> FromTileIndices(IEnumerable<int> tileIndices)
    {
        var tiles = new List<(int, int)>();
        foreach (int idx in tileIndices)
        {
            if (idx >= 0 && idx < 64 * 64)
                tiles.Add((idx / 64, idx % 64));
        }

        return tiles;
    }

    /// <summary>Target anchor for <see cref="ComputeAlignmentOffset"/>.</summary>
    public enum AlignmentTarget
    {
        /// <summary>The centroid of the base map's occupied tiles (default).</summary>
        BaseCentroid = 0,

        /// <summary>The camera's current tile.</summary>
        CameraTile = 1,
    }

    /// <summary>
    /// Computes the whole-layer tile offset that moves a donor footprint onto the base map:
    /// offset = target − centroid(donorTiles). Deterministic and pure — the caller clamps and
    /// applies it to <see cref="PhaseLayerSettings.TileOffsetX"/>/<see cref="PhaseLayerSettings.TileOffsetY"/>.
    /// </summary>
    /// <returns>False when either footprint is empty (nothing to align to).</returns>
    public static bool TryComputeAlignmentOffset(
        IReadOnlyList<(int TileX, int TileY)> baseTiles,
        IReadOnlyList<(int TileX, int TileY)> donorTiles,
        AlignmentTarget target,
        (int TileX, int TileY) cameraTile,
        out int offsetX,
        out int offsetY)
    {
        offsetX = 0;
        offsetY = 0;
        if (baseTiles.Count == 0 || donorTiles.Count == 0)
            return false;

        if (target == AlignmentTarget.CameraTile)
        {
            offsetX = cameraTile.TileX;
            offsetY = cameraTile.TileY;
        }
        else
        {
            double baseSumX = 0, baseSumY = 0;
            foreach ((int bx, int by) in baseTiles)
            {
                baseSumX += bx;
                baseSumY += by;
            }

            offsetX = (int)Math.Round(baseSumX / baseTiles.Count, MidpointRounding.AwayFromZero);
            offsetY = (int)Math.Round(baseSumY / baseTiles.Count, MidpointRounding.AwayFromZero);
        }

        double donorSumX = 0, donorSumY = 0;
        foreach ((int dx, int dy) in donorTiles)
        {
            donorSumX += dx;
            donorSumY += dy;
        }

        offsetX -= (int)Math.Round(donorSumX / donorTiles.Count, MidpointRounding.AwayFromZero);
        offsetY -= (int)Math.Round(donorSumY / donorTiles.Count, MidpointRounding.AwayFromZero);
        return true;
    }

    /// <summary>Clamps a tile offset to the 64×64 grid's ±63 range.</summary>
    public static (int OffsetX, int OffsetY) ClampOffset(int offsetX, int offsetY)
        => (Math.Clamp(offsetX, -63, 63), Math.Clamp(offsetY, -63, 63));

    /// <summary>
    /// Cartography minimap drag: computes the layer offset after dragging from a start tile to the
    /// current pointer tile, starting from the offset the layer had when the drag began. The
    /// pointer delta is rounded per axis, added to the base offset, and the result clamped to ±63.
    /// Pure — the minimap interaction calls this every frame while dragging; the composition
    /// re-streams only on release.
    /// </summary>
    public static (int OffsetX, int OffsetY) ApplyDragDelta(
        int baseOffsetX,
        int baseOffsetY,
        float startTileX,
        float startTileY,
        float currentTileX,
        float currentTileY)
    {
        int dx = (int)MathF.Round(currentTileX - startTileX);
        int dy = (int)MathF.Round(currentTileY - startTileY);
        return ClampOffset(baseOffsetX + dx, baseOffsetY + dy);
    }

    /// <summary>
    /// True when at least one donor tile, after applying the offset, lands on a base tile — i.e.
    /// the layer will actually contribute something at the current configuration.
    /// </summary>
    public static bool OverlapsBase(
        IReadOnlyList<(int TileX, int TileY)> baseTiles,
        IReadOnlyList<(int TileX, int TileY)> donorTiles,
        int offsetX,
        int offsetY)
    {
        if (baseTiles.Count == 0 || donorTiles.Count == 0)
            return false;

        var baseSet = new HashSet<(int, int)>(baseTiles);
        foreach ((int tx, int ty) in donorTiles)
        {
            if (baseSet.Contains((tx + offsetX, ty + offsetY)))
                return true;
        }

        return false;
    }
}
