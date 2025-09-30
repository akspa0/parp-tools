using System;

namespace WoWRollback.Core.Services;

/// <summary>
/// Provides helpers to convert WoW world coordinates into tile-relative normalized or pixel positions.
/// Formulas follow the viewer spec in <c>memory-bank/plans/viewer-diff-plan.md</c>.
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
    /// Computes normalized coordinates relative to the specified tile (0..1 range).
    /// World coords are absolute. Tile covers a 533.33 yard square.
    /// </summary>
    public static (double LocalX, double LocalY) ComputeLocalCoordinates(double worldX, double worldY, int tileRow, int tileCol)
    {
        // Calculate tile's world position (center of tile)
        // Tile 32,32 is at world origin (0,0)
        // Formula: worldPos = (32 - tileIndex) * 533.33333
        var tileCenterX = (HalfTiles - tileCol) * TileSpanYards;
        var tileCenterY = (HalfTiles - tileRow) * TileSpanYards;
        
        // Calculate offset from tile center, then normalize to [0,1]
        // Tile spans ±266.666 yards from center = 533.33 total
        var offsetX = worldX - tileCenterX;
        var offsetY = worldY - tileCenterY;
        
        // Convert to [0,1] range: center is 0.5, edges are 0 and 1
        var localX = 0.5 + (offsetX / TileSpanYards);
        var localY = 0.5 + (offsetY / TileSpanYards);
        
        return (ClampUnit(localX), ClampUnit(localY));
    }

    /// <summary>
    /// Converts normalized coordinates into pixel coordinates for a tile image.
    /// Leaflet Y increases downward, WoW Y+ is West/up on map, so invert Y.
    /// </summary>
    public static (double PixelX, double PixelY) ToPixels(double localX, double localY, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");

        var px = ClampUnit(localX) * width;
        // Keep Y inversion: localY=0 → bottom (height), localY=1 → top (0)
        var py = (1.0 - ClampUnit(localY)) * height;
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
