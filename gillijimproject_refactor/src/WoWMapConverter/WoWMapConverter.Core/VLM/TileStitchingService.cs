using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Image = SixLabors.ImageSharp.Image;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Tile stitching service - combines chunk-level data into full tile images.
/// ADT tiles have 16×16 chunks. Each chunk is 64×64 pixels for shadows/alphas.
/// Result: 1024×1024 tile images.
/// </summary>
public static class TileStitchingService
{
    private const int ChunksPerRow = 16;
    private const int ChunksPerTile = 256;
    private const int ChunkSize = 64;  // 64×64 for shadows and alphas
    private const int TileSize = ChunksPerRow * ChunkSize;  // 1024×1024

    /// <summary>
    /// Stitch 256 chunk shadow PNGs into a single 1024×1024 tile shadow image.
    /// </summary>
    public static byte[] StitchShadows(string shadowsDir, string tileName)
    {
        using var tileImage = new Image<L8>(TileSize, TileSize);
        
        for (int chunkIndex = 0; chunkIndex < ChunksPerTile; chunkIndex++)
        {
            var chunkPath = Path.Combine(shadowsDir, $"{tileName}_c{chunkIndex}.png");
            if (!File.Exists(chunkPath)) continue;
            
            int cx = chunkIndex % ChunksPerRow;
            int cy = chunkIndex / ChunksPerRow;
            int px = cx * ChunkSize;
            int py = cy * ChunkSize;
            
            try
            {
                using var chunkImage = Image.Load<L8>(chunkPath);
                // Copy chunk to tile at correct position
                for (int y = 0; y < ChunkSize && y < chunkImage.Height; y++)
                {
                    for (int x = 0; x < ChunkSize && x < chunkImage.Width; x++)
                    {
                        tileImage[px + x, py + y] = chunkImage[x, y];
                    }
                }
            }
            catch { }
        }
        
        using var ms = new MemoryStream();
        tileImage.SaveAsPng(ms);
        return ms.ToArray();
    }

    /// <summary>
    /// Stitch alpha masks for a specific layer into a 1024×1024 tile image.
    /// </summary>
    public static byte[] StitchAlphaLayer(string masksDir, string tileName, int layer)
    {
        using var tileImage = new Image<L8>(TileSize, TileSize);
        // Initialize to transparent/black (0)
        tileImage.Mutate(ctx => ctx.BackgroundColor(SixLabors.ImageSharp.Color.Black));
        
        for (int chunkIndex = 0; chunkIndex < ChunksPerTile; chunkIndex++)
        {
            var chunkPath = Path.Combine(masksDir, $"{tileName}_c{chunkIndex}_l{layer}.png");
            if (!File.Exists(chunkPath)) continue;
            
            int cx = chunkIndex % ChunksPerRow;
            int cy = chunkIndex / ChunksPerRow;
            int px = cx * ChunkSize;
            int py = cy * ChunkSize;
            
            try
            {
                using var chunkImage = Image.Load<L8>(chunkPath);
                for (int y = 0; y < ChunkSize && y < chunkImage.Height; y++)
                {
                    for (int x = 0; x < ChunkSize && x < chunkImage.Width; x++)
                    {
                        tileImage[px + x, py + y] = chunkImage[x, y];
                    }
                }
            }
            catch { }
        }
        
        using var ms = new MemoryStream();
        tileImage.SaveAsPng(ms);
        return ms.ToArray();
    }

    /// <summary>
    /// Stitch all layers for a tile, creating separate 1024×1024 images per layer.
    /// Returns paths to created files.
    /// </summary>
    public static async Task<List<string>> StitchAllLayersAsync(
        string masksDir, string tileName, int maxLayers = 4)
    {
        var createdPaths = new List<string>();
        
        for (int layer = 1; layer <= maxLayers; layer++)
        {
            // Check if any chunks have this layer
            bool hasAny = false;
            for (int c = 0; c < ChunksPerTile && !hasAny; c++)
            {
                var chunkPath = Path.Combine(masksDir, $"{tileName}_c{c}_l{layer}.png");
                if (File.Exists(chunkPath)) hasAny = true;
            }
            
            if (!hasAny) continue;
            
            var stitched = StitchAlphaLayer(masksDir, tileName, layer);
            var outputPath = Path.Combine(masksDir, $"{tileName}_stitched_l{layer}.png");
            await File.WriteAllBytesAsync(outputPath, stitched);
            createdPaths.Add(outputPath);
        }
        
        return createdPaths;
    }

    /// <summary>
    /// Generate stitched tile images for shadow and all alpha layers.
    /// </summary>
    public static async Task<(string? shadowPath, List<string> alphaPaths)> StitchTileAsync(
        string shadowsDir, string masksDir, string tileName, string outputDir)
    {
        string? shadowPath = null;
        var alphaPaths = new List<string>();
        
        // Stitch shadows
        var shadowFile = Path.Combine(shadowsDir, $"{tileName}_c0.png");
        if (File.Exists(shadowFile))
        {
            var stitchedShadow = StitchShadows(shadowsDir, tileName);
            shadowPath = Path.Combine(outputDir, $"{tileName}_shadow.png");
            await File.WriteAllBytesAsync(shadowPath, stitchedShadow);
        }
        
        // Stitch alpha layers
        for (int layer = 1; layer <= 4; layer++)
        {
            bool hasAny = false;
            for (int c = 0; c < ChunksPerTile && !hasAny; c++)
            {
                if (File.Exists(Path.Combine(masksDir, $"{tileName}_c{c}_l{layer}.png")))
                    hasAny = true;
            }
            
            if (!hasAny) continue;
            
            var stitched = StitchAlphaLayer(masksDir, tileName, layer);
            var alphaPath = Path.Combine(outputDir, $"{tileName}_alpha_l{layer}.png");
            await File.WriteAllBytesAsync(alphaPath, stitched);
            alphaPaths.Add(alphaPath);
        }
        
        return (shadowPath, alphaPaths);
    }
    /// <summary>
    /// Stitch liquid heights for a tile (1024x1024 L8).
    /// </summary>
    public static (byte[] image, float min, float max) StitchLiquidHeights(List<VlmLiquidData> liquids, string tileName)
    {
        using var tileImage = new Image<L8>(TileSize, TileSize);
        // Initialize to 0
        tileImage.Mutate(ctx => ctx.BackgroundColor(SixLabors.ImageSharp.Color.Black));
        
        // Find global min/max for the tile
        float minH = float.MaxValue;
        float maxH = float.MinValue;
        bool hasLiquid = false;
        
        foreach (var l in liquids)
        {
            if (l.Heights != null)
            {
                foreach (var h in l.Heights)
                {
                    if (h < minH) minH = h;
                    if (h > maxH) maxH = h;
                    hasLiquid = true;
                }
            }
        }
        
        if (!hasLiquid)
        {
            using var emptyMs = new MemoryStream();
            tileImage.SaveAsPng(emptyMs);
            return (emptyMs.ToArray(), 0f, 0f);
        }
        
        float range = maxH - minH;
        
        foreach (var liquid in liquids)
        {
            if (liquid.Heights == null) continue;
            
            int cx = liquid.ChunkIndex % ChunksPerRow;
            int cy = liquid.ChunkIndex / ChunksPerRow;
            int px = cx * ChunkSize;
            int py = cy * ChunkSize;
            
            // Render 9x9 grid to 64x64 area (8x8 blocks)
            for (int ly = 0; ly < 8; ly++)
            {
                for (int lx = 0; lx < 8; lx++)
                {
                    float hVal = liquid.Heights[ly * 9 + lx];
                    
                    byte val = 0;
                    if (range > 0.001f)
                        val = (byte)((hVal - minH) / range * 255);
                    else
                        val = 127; // Flat liquid
                        
                    var color = new L8(val);
                    for (int by = 0; by < 8; by++)
                        for (int bx = 0; bx < 8; bx++)
                            tileImage[px + lx * 8 + bx, py + ly * 8 + by] = color;
                }
            }
        }
        
        using var ms = new MemoryStream();
        tileImage.SaveAsPng(ms);
        return (ms.ToArray(), minH, maxH);
    }
    
    /// <summary>
    /// Stitch liquid mask (Water/Lava presence) to 1024x1024.
    /// </summary>
    public static byte[] StitchLiquidMask(List<VlmLiquidData> liquids, string tileName)
    {
        using var tileImage = new Image<L8>(TileSize, TileSize);
        tileImage.Mutate(ctx => ctx.BackgroundColor(SixLabors.ImageSharp.Color.Black)); // No liquid
        
        foreach (var liquid in liquids)
        {
            int cx = liquid.ChunkIndex % ChunksPerRow;
            int cy = liquid.ChunkIndex / ChunksPerRow;
            int px = cx * ChunkSize;
            int py = cy * ChunkSize;
            
            // Fill chunk with "Liquid Present" (255)
            // Later we can support specific liquid types
            // For now, just binary mask where liquid exists.
            
            // Check valid mask if exists, otherwise fill whole chunk
            // Currently VlmLiquidData doesn't store per-pixel mask, just assumed from MCLQ
            // MCLQ implies 8x8 flags. 
            // We should ideally use those flags.
            // But VlmLiquidData currently just stores "Heights" if existing.
            // If Heights != null, liquid exists.
            
            // Fill the 64x64 area for this chunk
            var white = new L8(255);
            for (int y = 0; y < ChunkSize; y++)
                for (int x = 0; x < ChunkSize; x++)
                    tileImage[px + x, py + y] = white;
        }
        
        using var ms = new MemoryStream();
        tileImage.SaveAsPng(ms);
        return ms.ToArray();
    }
}
