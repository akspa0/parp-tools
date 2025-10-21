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
    /// Convert WoW world coordinates to minimap tile pixel coordinates (0-imageSize).
    /// EXACT port of tileCanvas.js worldToPixel() with FLIP X and FLIP Y applied.
    /// </summary>
    public static (double pixelX, double pixelY) WorldToTilePixel(float worldX, float worldY, int imageWidth, int imageHeight)
    {
        // Exact formula from tileCanvas.js:
        // const localX = (32 - (worldX / tileSize)) - Math.floor(32 - (worldX / tileSize));
        // const localY = (32 - (worldY / tileSize)) - Math.floor(32 - (worldY / tileSize));
        // const pixelX = localX * this.width;
        // const pixelY = (1 - localY) * this.height;
        
        double tileCoordX = 32 - (worldX / BLOCK_SIZE);
        double tileCoordY = 32 - (worldY / BLOCK_SIZE);
        
        // Get fractional part (position within tile, 0-1)
        double localX = tileCoordX - Math.Floor(tileCoordX);
        double localY = tileCoordY - Math.Floor(tileCoordY);
        
        // Convert to pixel coordinates
        double pixelX = localX * imageWidth;
        double pixelY = (1 - localY) * imageHeight;
        
        // CRITICAL FIX: Both X and Y need to be flipped to align with minimap orientation
        // Verified through interactive debugging controls
        pixelX = imageWidth - pixelX;
        pixelY = imageHeight - pixelY;
        
        return (pixelX, pixelY);
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
    /// Get the world coordinate bounds for a tile.
    /// Returns (minX, maxX, minY, maxY) in world coordinates.
    /// </summary>
    public static (float minX, float maxX, float minY, float maxY) GetTileBounds(int tileX, int tileY)
    {
        // Each tile is BLOCK_SIZE wide (533.33 yards)
        // Tile formula: tileX = floor(32 - (worldX / 533.33))
        // Reverse: worldX = (32 - tileX) * 533.33 (top edge)
        //          worldX = (32 - tileX - 1) * 533.33 (bottom edge)
        
        float maxX = (32 - tileX) * BLOCK_SIZE;
        float minX = (32 - tileX - 1) * BLOCK_SIZE;
        float maxY = (32 - tileY) * BLOCK_SIZE;
        float minY = (32 - tileY - 1) * BLOCK_SIZE;
        
        return (minX, maxX, minY, maxY);
    }
    
    /// <summary>
    /// Check if world coordinates actually belong to the specified tile.
    /// Returns true if the coordinate is within the tile's bounds.
    /// This filters out "spanned" placements that appear on multiple tiles but don't geometrically belong.
    /// </summary>
    public static bool IsCoordinateInTile(float worldX, float worldY, int tileX, int tileY)
    {
        var (minX, maxX, minY, maxY) = GetTileBounds(tileX, tileY);
        
        // Check if coordinate is within tile bounds (inclusive of min, exclusive of max)
        return worldX >= minX && worldX < maxX && worldY >= minY && worldY < maxY;
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
