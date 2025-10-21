namespace WoWDataPlot.Helpers;

/// <summary>
/// WoW coordinate system transformations.
/// 
/// WoW uses a right-handed coordinate system:
/// - Positive X-axis points NORTH
/// - Positive Y-axis points WEST
/// - Z-axis is vertical (0 = sea level)
/// - Origin is at CENTER of map
/// - Map bounds: ±17066.66656 yards
/// - Map is 64x64 blocks, each 533.33333 yards
/// - Each block is 16x16 chunks, each 33.3333 yards
/// </summary>
public static class CoordinateTransform
{
    // WoW map constants
    public const float MAP_SIZE = 34133.33312f;          // Total map size (yards)
    public const float MAP_HALF_SIZE = 17066.66656f;     // Half map size (±)
    public const int BLOCKS_PER_AXIS = 64;               // 64x64 blocks
    public const float BLOCK_SIZE = 533.33333f;          // Block size (yards)
    public const int CHUNKS_PER_BLOCK = 16;              // 16x16 chunks per block
    public const float CHUNK_SIZE = 33.3333f;            // Chunk size (yards)
    
    /// <summary>
    /// Convert WoW world coordinates to normalized 0-1 range.
    /// (0,0) = bottom-left corner, (1,1) = top-right corner
    /// </summary>
    public static (double x, double y) WorldToNormalized(float worldX, float worldY)
    {
        // WoW coords: Top-left = (17066, 17066), Bottom-right = (-17066, -17066)
        // Normalized: (0,0) = bottom-left, (1,1) = top-right
        
        double normX = (worldX + MAP_HALF_SIZE) / MAP_SIZE;
        double normY = (worldY + MAP_HALF_SIZE) / MAP_SIZE;
        
        return (normX, normY);
    }
    
    /// <summary>
    /// Convert WoW world coordinates to tile indices.
    /// Returns (tileX, tileY) in 0-63 range.
    /// </summary>
    public static (int tileX, int tileY) WorldToTile(float worldX, float worldY)
    {
        // Formula from wiki: floor((32 - (axis / 533.33333)))
        int tileX = (int)Math.Floor(32 - (worldX / BLOCK_SIZE));
        int tileY = (int)Math.Floor(32 - (worldY / BLOCK_SIZE));
        
        // Clamp to valid range
        tileX = Math.Clamp(tileX, 0, BLOCKS_PER_AXIS - 1);
        tileY = Math.Clamp(tileY, 0, BLOCKS_PER_AXIS - 1);
        
        return (tileX, tileY);
    }
    
    /// <summary>
    /// Convert WoW world coordinates to plot coordinates for top-down visualization.
    /// 
    /// For top-down view matching in-game orientation:
    /// - Plot X-axis = East-West (maps to WoW Y-axis)
    /// - Plot Y-axis = North-South (maps to WoW X-axis)
    /// 
    /// This makes:
    /// - Top of image = North (high X in WoW)
    /// - Right of image = East (low Y in WoW)
    /// - Bottom of image = South (low X in WoW)
    /// - Left of image = West (high Y in WoW)
    /// </summary>
    public static (double plotX, double plotY) WorldToPlot(float worldX, float worldY)
    {
        // Simple: just flip Y-axis for correct orientation
        double plotX = worldX;
        double plotY = -worldY;
        
        return (plotX, plotY);
    }
    
    /// <summary>
    /// Get tile center coordinates in world space.
    /// </summary>
    public static (float centerX, float centerY) TileToWorldCenter(int tileX, int tileY)
    {
        // Reverse of WorldToTile formula
        // tileX = floor(32 - (worldX / 533.33))
        // So: worldX = (32 - tileX) * 533.33 - (533.33 / 2) for center
        
        float worldX = (32 - tileX - 0.5f) * BLOCK_SIZE;
        float worldY = (32 - tileY - 0.5f) * BLOCK_SIZE;
        
        return (worldX, worldY);
    }
    
    /// <summary>
    /// Get descriptive label for coordinates.
    /// </summary>
    public static string GetCoordinateLabel(float worldX, float worldY)
    {
        var (tileX, tileY) = WorldToTile(worldX, worldY);
        
        string northSouth = worldX >= 0 ? "North" : "South";
        string westEast = worldY >= 0 ? "West" : "East";
        
        return $"Tile [{tileX},{tileY}] ({Math.Abs(worldX):F1}y {northSouth}, {Math.Abs(worldY):F1}y {westEast})";
    }
}
