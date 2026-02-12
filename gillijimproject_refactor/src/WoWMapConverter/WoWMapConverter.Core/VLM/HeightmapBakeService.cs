using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Generates heightmap images from VLM dataset JSON exports.
/// Companion to MinimapBakeService - produces paired heightmap/minimap training data.
/// 
/// Output options:
/// - 256x256 per tile (16 pixels per chunk)
/// - 4096x4096 full resolution (256 pixels per chunk)
/// - 16-bit grayscale for precision
/// - Includes height bounds metadata for absolute height reconstruction
/// </summary>
public class HeightmapBakeService
{
    private readonly string _datasetRoot;
    
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public HeightmapBakeService(string datasetRoot)
    {
        _datasetRoot = datasetRoot;
    }

    /// <summary>
    /// Bakes a 256x256 heightmap from VLM JSON (training resolution).
    /// Uses Noggit's exact algorithm for smooth output.
    /// </summary>
    public async Task<(Image<L16> Heightmap, float HeightMin, float HeightMax)> BakeHeightmap256Async(string jsonPath)
    {
        var (heights, heightMin, heightMax) = await LoadHeightsFromJsonAsync(jsonPath);
        
        // Noggit uses 257x257 (16 chunks * 16 + 1), we'll resize to 256 at the end
        var heightmap257 = new Image<L16>(257, 257);
        float range = heightMax - heightMin;
        
        // Constants from Noggit: LONG=9 (outer), SHORT=8 (inner), SUM=17
        const int LONG = 9, SHORT = 8, SUM = LONG + SHORT, DSUM = SUM * 2;
        
        heightmap257.ProcessPixelRows(accessor =>
        {
            for (int chunkY = 0; chunkY < 16; chunkY++)
            {
                for (int chunkX = 0; chunkX < 16; chunkX++)
                {
                    int chunkIdx = chunkY * 16 + chunkX;
                    var mcvt = heights[chunkIdx];
                    
                    if (mcvt == null || mcvt.Length < 145)
                        continue;
                    
                    // Noggit iterates y then x within chunk, each 0-16 (SUM=17 values)
                    for (int y = 0; y < SUM; y++)
                    {
                        int pixelY = chunkY * 16 + y;
                        if (pixelY >= 257) continue;
                        
                        var row = accessor.GetRowSpan(pixelY);
                        
                        for (int x = 0; x < SUM; x++)
                        {
                            int pixelX = chunkX * 16 + x;
                            if (pixelX >= 257) continue;
                            
                            // Noggit's indexing logic
                            int plain = y * SUM + x;
                            bool isVirtual = (plain % 2) == 1;
                            bool erp = (plain % DSUM) / SUM == 1;
                            int idx = (plain - (isVirtual ? (erp ? SUM : 1) : 0)) / 2;
                            
                            float value;
                            if (isVirtual)
                            {
                                // Virtual vertex: interpolate between two real vertices
                                int idx2 = idx + (erp ? SUM : 1);
                                if (idx < 145 && idx2 < 145)
                                    value = (mcvt[idx] + mcvt[idx2]) / 2.0f;
                                else if (idx < 145)
                                    value = mcvt[idx];
                                else
                                    value = 0;
                            }
                            else
                            {
                                // Real vertex: direct lookup
                                value = idx < 145 ? mcvt[idx] : 0;
                            }
                            
                            // Normalize
                            float normalized = range > 1e-6f 
                                ? (value - heightMin) / range 
                                : 0.5f;
                            normalized = Math.Clamp(normalized, 0f, 1f);
                            row[pixelX] = new L16((ushort)(normalized * 65535));
                        }
                    }
                }
            }
        });
        
        // Resize from 257 to 256 for training
        var heightmap = heightmap257.Clone();
        heightmap.Mutate(ctx => ctx.Resize(256, 256, KnownResamplers.Lanczos3));
        heightmap257.Dispose();
        
        return (heightmap, heightMin, heightMax);
    }

    /// <summary>
    /// Bakes a 4096x4096 full-resolution heightmap (256 pixels per chunk).
    /// Provides maximum detail for terrain visualization.
    /// </summary>
    public async Task<(Image<L16> Heightmap, float HeightMin, float HeightMax)> BakeHeightmap4096Async(string jsonPath)
    {
        var (heightmap256, heightMin, heightMax) = await BakeHeightmap256Async(jsonPath);
        
        // Upscale to 4096x4096 using bicubic interpolation
        var heightmap4096 = heightmap256.Clone();
        heightmap4096.Mutate(x => x.Resize(4096, 4096, KnownResamplers.Bicubic));
        
        heightmap256.Dispose();
        return (heightmap4096, heightMin, heightMax);
    }

    /// <summary>
    /// Loads height data from VLM JSON file.
    /// </summary>
    private async Task<(float[][] Heights, float HeightMin, float HeightMax)> LoadHeightsFromJsonAsync(string jsonPath)
    {
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("JSON tile not found", jsonPath);

        var jsonContent = await File.ReadAllTextAsync(jsonPath);
        var sample = JsonSerializer.Deserialize<VlmTrainingSample>(jsonContent, _jsonOptions);
        
        if (sample?.TerrainData?.Heights == null)
            throw new Exception("Invalid VLM JSON data: missing heights.");

        var heights = new float[256][];
        float globalMin = float.MaxValue;
        float globalMax = float.MinValue;
        
        foreach (var chunk in sample.TerrainData.Heights)
        {
            if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= 256)
                continue;
            
            heights[chunk.ChunkIndex] = chunk.Heights;
            
            if (chunk.Heights != null)
            {
                foreach (var h in chunk.Heights)
                {
                    if (h < globalMin) globalMin = h;
                    if (h > globalMax) globalMax = h;
                }
            }
        }
        
        // Use stored bounds if available, otherwise use computed
        float heightMin = sample.TerrainData.HeightMin;
        float heightMax = sample.TerrainData.HeightMax;
        
        // If stored bounds are invalid (same value), use computed
        if (Math.Abs(heightMax - heightMin) < 1e-6f)
        {
            heightMin = globalMin;
            heightMax = globalMax;
        }
        
        return (heights, heightMin, heightMax);
    }

    /// <summary>
    /// Scan all tiles in a dataset to find global height bounds for the entire map.
    /// This is required for Noggit-compatible heightmap export.
    /// </summary>
    public async Task<(float GlobalMin, float GlobalMax, int TileCount)> ScanMapHeightBoundsAsync(string datasetDir)
    {
        var datasetFolder = Path.Combine(datasetDir, "dataset");
        if (!Directory.Exists(datasetFolder))
            throw new DirectoryNotFoundException($"Dataset folder not found: {datasetFolder}");
        
        float globalMin = float.MaxValue;
        float globalMax = float.MinValue;
        int tileCount = 0;
        
        foreach (var jsonPath in Directory.EnumerateFiles(datasetFolder, "*.json"))
        {
            try
            {
                var (heights, tileMin, tileMax) = await LoadHeightsFromJsonAsync(jsonPath);
                
                // Also scan actual vertex values for accuracy
                foreach (var chunk in heights)
                {
                    if (chunk == null) continue;
                    foreach (var h in chunk)
                    {
                        if (h < globalMin) globalMin = h;
                        if (h > globalMax) globalMax = h;
                    }
                }
                tileCount++;
            }
            catch { /* Skip invalid tiles */ }
        }
        
        return (globalMin, globalMax, tileCount);
    }
    
    /// <summary>
    /// Bakes heightmap using specified global height bounds (for map-wide consistency).
    /// Uses ALPHA ADT MCVT format: 81 outer (9x9) then 64 inner (8x8) - NOT interleaved.
    /// RAW vertex placement only - no interpolation (best for ML training).
    /// </summary>
    public async Task<Image<L16>> BakeHeightmapWithBoundsAsync(string jsonPath, float globalMin, float globalMax)
    {
        var (heights, _, _) = await LoadHeightsFromJsonAsync(jsonPath);
        
        var output = new float[256, 256];
        var hasData = new bool[256, 256];
        
        for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
        {
            var mcvt = heights[chunkIdx];
            if (mcvt == null || mcvt.Length < 145)
                continue;
            
            int chunkY = chunkIdx / 16;
            int chunkX = chunkIdx % 16;
            int baseX = chunkX * 16;
            int baseY = chunkY * 16;
            
            // Place 9x9 outer vertices at even positions (raw data only)
            for (int oy = 0; oy < 9; oy++)
            {
                for (int ox = 0; ox < 9; ox++)
                {
                    int px = baseX + ox * 2;
                    int py = baseY + oy * 2;
                    if (px < 256 && py < 256)
                    {
                        float val = mcvt[oy * 9 + ox];
                        if (hasData[py, px])
                            output[py, px] = (output[py, px] + val) / 2; // Average overlapping edges
                        else
                            output[py, px] = val;
                        hasData[py, px] = true;
                    }
                }
            }
            
            // Place 8x8 inner vertices at odd positions (raw data only)
            for (int iy = 0; iy < 8; iy++)
            {
                for (int ix = 0; ix < 8; ix++)
                {
                    int px = baseX + ix * 2 + 1;
                    int py = baseY + iy * 2 + 1;
                    if (px < 256 && py < 256)
                    {
                        output[py, px] = mcvt[81 + iy * 8 + ix];
                        hasData[py, px] = true;
                    }
                }
            }
        }
        
        // Fill gaps with nearest-neighbor (no interpolation - just copy nearest real value)
        for (int pass = 0; pass < 3; pass++)
        {
            var filled = (float[,])output.Clone();
            var filledData = (bool[,])hasData.Clone();
            
            for (int y = 0; y < 256; y++)
            {
                for (int x = 0; x < 256; x++)
                {
                    if (!hasData[y, x])
                    {
                        // Find nearest neighbor with data
                        float nearest = 0;
                        float minDist = float.MaxValue;
                        for (int dy = -2; dy <= 2; dy++)
                        {
                            for (int dx = -2; dx <= 2; dx++)
                            {
                                if (dy == 0 && dx == 0) continue;
                                int ny = y + dy, nx = x + dx;
                                if (ny >= 0 && ny < 256 && nx >= 0 && nx < 256 && hasData[ny, nx])
                                {
                                    float dist = dy * dy + dx * dx;
                                    if (dist < minDist)
                                    {
                                        minDist = dist;
                                        nearest = output[ny, nx];
                                    }
                                }
                            }
                        }
                        if (minDist < float.MaxValue)
                        {
                            filled[y, x] = nearest;
                            filledData[y, x] = true;
                        }
                    }
                }
            }
            output = filled;
            hasData = filledData;
        }
        
        // Render to image with global bounds
        var result = new Image<L16>(256, 256);
        float range = globalMax - globalMin;
        
        result.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < 256; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < 256; x++)
                {
                    float normalized = range > 1e-6f 
                        ? (output[y, x] - globalMin) / range 
                        : 0.5f;
                    normalized = Math.Clamp(normalized, 0f, 1f);
                    row[x] = new L16((ushort)(normalized * 65535));
                }
            }
        });
        
        return result;
    }
    
    /// <summary>
    /// Export all tiles in a dataset using map-wide height bounds.
    /// Creates a map_bounds.json with global min/max for reconstruction.
    /// </summary>
    public async Task ExportMapHeightmapsAsync(string datasetDir, string outputDir, IProgress<string>? progress = null)
    {
        progress?.Report("Scanning map for global height bounds...");
        var (globalMin, globalMax, tileCount) = await ScanMapHeightBoundsAsync(datasetDir);
        
        progress?.Report($"Found {tileCount} tiles, height range: {globalMin:F2} to {globalMax:F2}");
        
        Directory.CreateDirectory(outputDir);
        
        // Write map-level bounds metadata
        var mapMeta = new
        {
            map_name = Path.GetFileName(datasetDir),
            height_min = globalMin,
            height_max = globalMax,
            height_range = globalMax - globalMin,
            tile_count = tileCount,
            export_date = DateTime.UtcNow.ToString("o")
        };
        await File.WriteAllTextAsync(
            Path.Combine(outputDir, "map_bounds.json"),
            JsonSerializer.Serialize(mapMeta, _jsonOptions));
        
        // Export each tile
        var datasetFolder = Path.Combine(datasetDir, "dataset");
        int exported = 0;
        
        foreach (var jsonPath in Directory.EnumerateFiles(datasetFolder, "*.json"))
        {
            try
            {
                var tileName = Path.GetFileNameWithoutExtension(jsonPath);
                progress?.Report($"Exporting {tileName}...");
                
                var heightmap = await BakeHeightmapWithBoundsAsync(jsonPath, globalMin, globalMax);
                await heightmap.SaveAsPngAsync(Path.Combine(outputDir, $"{tileName}_heightmap.png"));
                heightmap.Dispose();
                
                exported++;
            }
            catch (Exception ex)
            {
                progress?.Report($"Failed {Path.GetFileName(jsonPath)}: {ex.Message}");
            }
        }
        
        progress?.Report($"Exported {exported}/{tileCount} tiles");
    }

    /// <summary>
    /// Exports heightmap with metadata sidecar JSON containing height bounds.
    /// This allows reconstruction of absolute world heights from normalized image.
    /// </summary>
    public async Task ExportWithMetadataAsync(string jsonPath, string outputDir)
    {
        var (heightmap, heightMin, heightMax) = await BakeHeightmap256Async(jsonPath);
        
        var tileName = Path.GetFileNameWithoutExtension(jsonPath);
        var heightmapPath = Path.Combine(outputDir, $"{tileName}_heightmap.png");
        var metadataPath = Path.Combine(outputDir, $"{tileName}_heightmap_meta.json");
        
        Directory.CreateDirectory(outputDir);
        
        // Save 16-bit heightmap
        await heightmap.SaveAsPngAsync(heightmapPath);
        
        // Save metadata
        var metadata = new
        {
            tile_name = tileName,
            height_min = heightMin,
            height_max = heightMax,
            height_range = heightMax - heightMin,
            image_path = Path.GetFileName(heightmapPath),
            resolution = 256
        };
        
        await File.WriteAllTextAsync(metadataPath, 
            JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
        
        heightmap.Dispose();
    }
}
