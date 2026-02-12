using System;

namespace WoWRollback.Core.Services;

/// <summary>
/// Provides helpers to convert WoW world coordinates into tile-relative normalized or pixel positions.
/// Uses precise constants from ADT file format reference documentation.
/// 
/// Coordinate System (Right-handed, origin at map center):
/// - Positive X-axis points NORTH, Positive Y-axis points WEST
/// - NW corner (tile 0,0): (+17066.66656, +17066.66656)
/// - Center (tiles 31/32 intersection): (0, 0)
/// - SE corner (tile 63,63): (-17066.66656, -17066.66656)
/// - Each tile: 533.33333 yards (exactly 1600 feet)
/// </summary>
public static class CoordinateTransformer
{
    // Precise WoW coordinate system constants (from ADT reference documentation)
    private const double TilesPerSide = 64.0;
    private const double HalfTiles = TilesPerSide / 2.0; // 32.0
    private const double TileSpanYards = 533.33333; // Exact: 1600 feet / 3 = 533.33333 yards
    public const double MapHalfSize = HalfTiles * TileSpanYards; // 17066.66656 yards (±map boundaries)
    public const double MapTotalSize = TilesPerSide * TileSpanYards; // 34133.33312 yards (full map width/height)

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
    /// 
    /// WoW coordinate system: +X=North, +Y=West, origin at map center
    /// Tile 0,0 is NW corner (+17066, +17066)
    /// Minimap texture: pixel (0,0) is NW corner (top-left), (width,height) is SE corner (bottom-right)
    /// 
    /// Within a tile:
    /// - localX: 0.0 at tile's west edge, 1.0 at tile's east edge
    /// - localY: 0.0 at tile's north edge, 1.0 at tile's south edge
    /// </summary>
    public static (double LocalX, double LocalY) ComputeLocalCoordinates(double worldX, double worldY, int tileRow, int tileCol)
    {
        // Compute continuous tile coordinates
        var tx = HalfTiles - (worldX / TileSpanYards);
        var ty = HalfTiles - (worldY / TileSpanYards);

        static double Frac(double v) => v - Math.Floor(v);

        // Extract fractional position within tile (no flipping needed - the tile index formula handles orientation)
        var localX = Frac(tx);
        var localY = Frac(ty);

        return (ClampUnit(localX), ClampUnit(localY));
    }

    /// <summary>
    /// Converts normalized tile coordinates into pixel coordinates for a minimap tile image.
    /// Maps [0..1] range to [0..width-1] and [0..height-1] to avoid bleeding into adjacent tiles.
    /// </summary>
    public static (double PixelX, double PixelY) ToPixels(double localX, double localY, int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");

        // Map normalized [0..1] to pixel coordinates [0..W-1] / [0..H-1]
        // Subtract 1 to prevent coordinates at exactly 1.0 from landing on the next tile
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
