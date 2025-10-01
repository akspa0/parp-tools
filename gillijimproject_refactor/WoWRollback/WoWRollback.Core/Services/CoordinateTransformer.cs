using System;

namespace WoWRollback.Core.Services;

/// <summary>
/// Provides helpers to convert WoW world coordinates into tile-relative normalized or pixel positions.
/// Formulas follow the viewer spec and wow.tools orientation.
/// </summary>
public static class CoordinateTransformer
{
    private const double TilesPerSide = 64.0;
    private const double HalfTiles = TilesPerSide / 2.0; // 32
    private const double TileSpanYards = 533.33333; // ADT tile span

    /// <summary>
    /// Computes the ADT tile indices (row, col) for a world position.
    /// </summary>
    public static (int TileRow, int TileCol) ComputeTileIndices(double worldX, double worldY)
    {
        var col = (int)Math.Floor(HalfTiles - (worldX / TileSpanYards));
        var row = (int)Math.Floor(HalfTiles - (worldY / TileSpanYards));
        return (ClampTile(row), ClampTile(col));
    }

    /// <summary>
    /// Computes normalized coordinates relative to the specified tile (0..1 range) using wow.tools mapping.
    /// - Tile indices: row = floor(32 - worldY/533.33333), col = floor(32 - worldX/533.33333)
    /// - Local coords within tile:
    ///     localX = 1 - frac(32 - worldX/533.33333)  // flip X to match minimap texture orientation (west to the right in image)
    ///     localY = frac(32 - worldY/533.33333)
    /// </summary>
    public static (double LocalX, double LocalY) ComputeLocalCoordinates(double worldX, double worldY, int tileRow, int tileCol)
    {
        // Compute continuous tile coordinates
        var tx = HalfTiles - (worldX / TileSpanYards);
        var ty = HalfTiles - (worldY / TileSpanYards);

        static double Frac(double v) => v - Math.Floor(v);

        // wow.tools orientation: X is flipped relative to world axis when rendered on minimap textures
        var localX = 1.0 - Frac(tx);
        var localY = Frac(ty);

        return (ClampUnit(localX), ClampUnit(localY));
    }

    /// <summary>
    /// Converts normalized coordinates into pixel coordinates for a tile image.
    /// Maps local [0..1] directly to pixel [0..W-1]/[0..H-1] with top-left origin (no axis flip).
    /// </summary>
    public static (double PixelX, double PixelY) ToPixels(double localX, double localY, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");

        var w1 = Math.Max(1, width - 1);
        var h1 = Math.Max(1, height - 1);
        var px = ClampUnit(localX) * w1;
        var py = ClampUnit(localY) * h1;
        return (px, py);
    }

    private static int ClampTile(int value)
    {
        if (value < 0) return 0;
        if (value >= TilesPerSide) return (int)TilesPerSide - 1;
        return value;
    }

    private static double ClampUnit(double value)
    {
        if (value < 0d) return 0d;
        if (value > 1d) return 1d;
        return value;
    }
}
