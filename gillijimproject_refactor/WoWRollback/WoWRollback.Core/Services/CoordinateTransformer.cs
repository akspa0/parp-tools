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
    /// Computes normalized world coordinates (±17066 range) from tile indices and local [0..1] coordinates.
    /// Inverse of ComputeLocalCoordinates following the same wow.tools orientation.
    /// </summary>
    public static (double WorldX, double WorldY) ComputeWorldFromTileLocal(int tileRow, int tileCol, double localX, double localY)
    {
        // tx = HalfTiles - (worldX / TileSpan) = tileCol + (1 - localX)
        // ty = HalfTiles - (worldY / TileSpan) = tileRow + localY
        var tx = tileCol + (1.0 - ClampUnit(localX));
        var ty = tileRow + ClampUnit(localY);
        var worldX = (HalfTiles - tx) * TileSpanYards;
        var worldY = (HalfTiles - ty) * TileSpanYards;
        return (worldX, worldY);
    }

    /// <summary>
    /// Computes normalized world coordinates from tile indices and pixel coordinates.
    /// </summary>
    public static (double WorldX, double WorldY) ComputeWorldFromTilePixel(int tileRow, int tileCol, double pixelX, double pixelY, int width, int height)
    {
        var w1 = Math.Max(1, width - 1);
        var h1 = Math.Max(1, height - 1);
        var localX = ClampUnit(pixelX / w1);
        var localY = ClampUnit(pixelY / h1);
        return ComputeWorldFromTileLocal(tileRow, tileCol, localX, localY);
    }

    /// <summary>
    /// Computes raw local coordinates (unclamped) using wow.tools mapping for the specified tile.
    /// - Tile indices: row = floor(32 - worldY/533.33333), col = floor(32 - worldX/533.33333)
    /// - Local coords within tile:
    ///     localX = 1 - frac(32 - worldX/533.33333)
    ///     localY = frac(32 - worldY/533.33333)
    /// </summary>
    public static (double LocalX, double LocalY) ComputeLocalCoordinatesRaw(double worldX, double worldY, int tileRow, int tileCol)
    {
        // Compute continuous tile coordinates
        var tx = HalfTiles - (worldX / TileSpanYards);
        var ty = HalfTiles - (worldY / TileSpanYards);

        static double Frac(double v) => v - Math.Floor(v);

        // wow.tools orientation: X is flipped relative to world axis when rendered on minimap textures
        var localX = 1.0 - Frac(tx);
        var localY = Frac(ty);
        return (localX, localY);
    }

    /// <summary>
    /// Computes normalized coordinates relative to the specified tile (0..1 range) using wow.tools mapping.
    /// Calls ComputeLocalCoordinatesRaw and clamps to [0..1].
    /// </summary>
    public static (double LocalX, double LocalY) ComputeLocalCoordinates(double worldX, double worldY, int tileRow, int tileCol)
    {
        var (lx, ly) = ComputeLocalCoordinatesRaw(worldX, worldY, tileRow, tileCol);
        return (ClampUnit(lx), ClampUnit(ly));
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
