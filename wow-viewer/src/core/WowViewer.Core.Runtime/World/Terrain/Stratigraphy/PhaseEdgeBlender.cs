namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// Operator directive 2026-09-09 (Spec 232 T064): magnetic WDL microlattice edge-snapping for
/// composed phase layers. A cut tile (e.g. Teldrassil out of Kalidar) raised onto a base map
/// leaves a height cliff at the seam; this service blends ONLY the tile-boundary outer vertices
/// of contributed chunks toward the base map's WDL 17×17 macro lattice sampled at the same world
/// position, so the seam follows the low-frequency surface both sides already agree on. Interior
/// vertices and interior-tile chunks are untouched — the tile keeps its relief.
/// </summary>
public static class PhaseEdgeBlender
{
    /// <summary>
    /// Blends the tile-boundary outer vertices of one chunk's 145-height array toward the WDL
    /// macro lattice. <paramref name="height17"/> is the base map's WDL lattice for the chunk's
    /// target tile (world yards, [row = y, col = x]). <paramref name="neighborContributes"/>
    /// answers whether the adjacent target tile in a direction also receives this layer's
    /// heights — edges shared with a contributing neighbor are left alone (that seam is inside
    /// the layer), only footprint-boundary edges blend.
    /// </summary>
    /// <returns>True when any vertex was modified.</returns>
    public static bool BlendTileBoundary(
        float[] heights145,
        short[,] height17,
        int chunkX,
        int chunkY,
        float strength,
        Func<int, int, bool> neighborContributes)
    {
        ArgumentNullException.ThrowIfNull(heights145);
        ArgumentNullException.ThrowIfNull(height17);
        ArgumentNullException.ThrowIfNull(neighborContributes);

        if (strength <= 0f || heights145.Length < 145)
            return false;

        float clamped = Math.Clamp(strength, 0f, 1f);
        bool modified = false;

        // Outer-lattice vertices only; inner vertices never lie on a tile boundary.
        // West edge: chunkX == 0, col == 0. East: chunkX == 15, col == 8.
        // North edge: chunkY == 0, row == 0. South: chunkY == 15, row == 8.
        if (chunkX == 0 && !neighborContributes(-1, 0))
            modified |= BlendEdge(heights145, height17, chunkX, chunkY, strength, edgeColumn: 0);
        if (chunkX == 15 && !neighborContributes(1, 0))
            modified |= BlendEdge(heights145, height17, chunkX, chunkY, strength, edgeColumn: 8);
        if (chunkY == 0 && !neighborContributes(0, -1))
            modified |= BlendRow(heights145, height17, chunkX, chunkY, strength, edgeRow: 0);
        if (chunkY == 15 && !neighborContributes(0, 1))
            modified |= BlendRow(heights145, height17, chunkX, chunkY, strength, edgeRow: 8);

        return modified;
    }

    private static bool BlendEdge(
        float[] heights145, short[,] height17, int chunkX, int chunkY, float strength, int edgeColumn)
    {
        bool modified = false;
        for (int row = 0; row < 9; row++)
        {
            int idx = row * 17 + edgeColumn;
            float normX = (chunkX + (edgeColumn / 8f)) / 16f;
            float normY = (chunkY + (row / 8f)) / 16f;
            modified |= BlendVertex(heights145, idx, height17, normX, normY, strength);
        }

        return modified;
    }

    private static bool BlendRow(
        float[] heights145, short[,] height17, int chunkX, int chunkY, float strength, int edgeRow)
    {
        bool modified = false;
        for (int col = 0; col < 9; col++)
        {
            int idx = edgeRow * 17 + col;
            float normX = (chunkX + (col / 8f)) / 16f;
            float normY = (chunkY + (edgeRow / 8f)) / 16f;
            modified |= BlendVertex(heights145, idx, height17, normX, normY, strength);
        }

        return modified;
    }

    private static bool BlendVertex(
        float[] heights145, int idx, short[,] height17, float normX, float normY, float strength)
    {
        float wdlMacro = WdlLatticeMagnetizer.SampleWdlHeight(height17, normX, normY);
        float blended = (1f - strength) * heights145[idx] + strength * wdlMacro;
        bool changed = MathF.Abs(blended - heights145[idx]) > 1e-4f;
        heights145[idx] = blended;
        return changed;
    }
}