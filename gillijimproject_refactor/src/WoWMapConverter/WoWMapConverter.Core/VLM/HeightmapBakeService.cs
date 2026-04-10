using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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
    /// Uses the shared viewer-compatible tile bake path.
    /// </summary>
    public async Task<(Image<L16> Heightmap, float HeightMin, float HeightMax)> BakeHeightmap256Async(string jsonPath)
    {
        VlmTrainingSample sample = await LoadSampleAsync(jsonPath);
        TerrainTileBakeService.TileHeightmap257 tileHeightmap = BuildTileHeightmap(sample.TerrainData);
        var (heightMin, heightMax) = ResolveHeightRange(sample.TerrainData, tileHeightmap);
        Image<L16> image = TerrainTileBakeService.CreateHeightmapImage(tileHeightmap.Heights, heightMin, heightMax, 256);
        return (image, heightMin, heightMax);
    }

    /// <summary>
    /// Bakes a 4096x4096 full-resolution heightmap (256 pixels per chunk).
    /// Provides maximum detail for terrain visualization.
    /// </summary>
    public async Task<(Image<L16> Heightmap, float HeightMin, float HeightMax)> BakeHeightmap4096Async(string jsonPath)
    {
        VlmTrainingSample sample = await LoadSampleAsync(jsonPath);
        TerrainTileBakeService.TileHeightmap257 tileHeightmap = BuildTileHeightmap(sample.TerrainData);
        var (heightMin, heightMax) = ResolveHeightRange(sample.TerrainData, tileHeightmap);
        Image<L16> image = TerrainTileBakeService.CreateHeightmapImage(tileHeightmap.Heights, heightMin, heightMax, 4096);
        return (image, heightMin, heightMax);
    }

    /// <summary>
    /// Loads VLM JSON sample data.
    /// </summary>
    private async Task<VlmTrainingSample> LoadSampleAsync(string jsonPath)
    {
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("JSON tile not found", jsonPath);

        var jsonContent = await File.ReadAllTextAsync(jsonPath);
        VlmTrainingSample? sample = JsonSerializer.Deserialize<VlmTrainingSample>(jsonContent, _jsonOptions);
        
        if (sample?.TerrainData?.Heights == null)
            throw new Exception("Invalid VLM JSON data: missing heights.");

        return sample;
    }

    private static TerrainTileBakeService.TileHeightmap257 BuildTileHeightmap(VlmTerrainData terrainData)
    {
        var heights = new Dictionary<int, float[]>();
        foreach (VlmChunkHeights chunk in terrainData.Heights ?? Array.Empty<VlmChunkHeights>())
        {
            if (chunk.Heights == null || chunk.Heights.Length < 145)
                continue;

            heights[chunk.ChunkIndex] = chunk.Heights;
        }

        return TerrainTileBakeService.BuildTileHeightmap257(heights, terrainData.IsInterleaved);
    }

    private static (float HeightMin, float HeightMax) ResolveHeightRange(VlmTerrainData terrainData, TerrainTileBakeService.TileHeightmap257 tileHeightmap)
    {
        float heightMin = terrainData.HeightMin;
        float heightMax = terrainData.HeightMax;

        if (Math.Abs(heightMax - heightMin) >= 1e-6f)
            return (heightMin, heightMax);

        return (tileHeightmap.MinHeight, tileHeightmap.MaxHeight);
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
                VlmTrainingSample sample = await LoadSampleAsync(jsonPath);
                
                // Also scan actual vertex values for accuracy
                foreach (VlmChunkHeights chunk in sample.TerrainData.Heights ?? Array.Empty<VlmChunkHeights>())
                {
                    if (chunk.Heights == null) continue;
                    foreach (float h in chunk.Heights)
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
    /// Uses the shared viewer-compatible tile bake path for coherent tile edges.
    /// </summary>
    public async Task<Image<L16>> BakeHeightmapWithBoundsAsync(string jsonPath, float globalMin, float globalMax)
    {
        VlmTrainingSample sample = await LoadSampleAsync(jsonPath);
        TerrainTileBakeService.TileHeightmap257 tileHeightmap = BuildTileHeightmap(sample.TerrainData);
        return TerrainTileBakeService.CreateHeightmapImage(tileHeightmap.Heights, globalMin, globalMax, 256);
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
