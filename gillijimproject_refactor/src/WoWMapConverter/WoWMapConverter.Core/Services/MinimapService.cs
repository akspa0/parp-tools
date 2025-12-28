namespace WoWMapConverter.Core.Services;

/// <summary>
/// Minimap extraction and conversion service.
/// Provides utilities for extracting minimap tiles and generating MCCV vertex colors.
/// </summary>
public static class MinimapService
{
    /// <summary>
    /// Standard minimap tile size.
    /// </summary>
    public const int TileSize = 256;

    /// <summary>
    /// Get the minimap tile path for a given map and coordinates.
    /// </summary>
    public static string GetMinimapTilePath(string mapName, int x, int y)
    {
        return $"textures/minimap/{mapName.ToLowerInvariant()}/map{x:D2}_{y:D2}.blp";
    }

    /// <summary>
    /// Check if a minimap tile exists at the given path.
    /// </summary>
    public static bool MinimapTileExists(string basePath, string mapName, int x, int y)
    {
        var tilePath = Path.Combine(basePath, GetMinimapTilePath(mapName, x, y));
        return File.Exists(tilePath);
    }

    /// <summary>
    /// Extract vertex colors from a minimap tile for MCCV painting.
    /// Returns a 17x17 array of RGB values sampled from the minimap.
    /// </summary>
    /// <param name="minimapPath">Path to the minimap PNG/BLP file</param>
    /// <returns>17x17 array of (R,G,B) tuples, or null if extraction fails</returns>
    public static (byte r, byte g, byte b)[,]? ExtractVertexColors(string minimapPath)
    {
        // Stub implementation - full version would:
        // 1. Load the minimap image (PNG or BLP)
        // 2. Sample 17x17 points across the tile
        // 3. Return the RGB values for MCCV vertex coloring
        
        if (!File.Exists(minimapPath))
            return null;

        // Return default white colors as placeholder
        var colors = new (byte r, byte g, byte b)[17, 17];
        for (int y = 0; y < 17; y++)
            for (int x = 0; x < 17; x++)
                colors[x, y] = (127, 127, 127); // Neutral gray

        return colors;
    }

    /// <summary>
    /// Generate MCCV chunk data from vertex colors.
    /// </summary>
    /// <param name="colors">17x17 array of RGB values</param>
    /// <returns>MCCV chunk data (145 * 4 bytes = 580 bytes)</returns>
    public static byte[] GenerateMccvData((byte r, byte g, byte b)[,] colors)
    {
        // MCCV format: 145 vertices (9x9 outer + 8x8 inner, interleaved)
        // Each vertex: 4 bytes (B, G, R, A)
        const int vertexCount = 145;
        var data = new byte[vertexCount * 4];

        int idx = 0;
        for (int row = 0; row < 17; row++)
        {
            int cols = (row % 2 == 0) ? 9 : 8;
            int startX = (row % 2 == 0) ? 0 : 0;
            
            for (int col = 0; col < cols; col++)
            {
                int x = (row % 2 == 0) ? col * 2 : col * 2 + 1;
                int y = row;
                
                if (x < 17 && y < 17)
                {
                    var (r, g, b) = colors[x, y];
                    data[idx++] = b;     // Blue
                    data[idx++] = g;     // Green
                    data[idx++] = r;     // Red
                    data[idx++] = 127;   // Alpha (unused, set to neutral)
                }
                else
                {
                    // Fallback for out-of-bounds
                    data[idx++] = 127;
                    data[idx++] = 127;
                    data[idx++] = 127;
                    data[idx++] = 127;
                }
            }
        }

        return data;
    }
}
