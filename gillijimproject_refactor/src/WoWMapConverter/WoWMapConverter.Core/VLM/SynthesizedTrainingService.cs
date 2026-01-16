using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Generates synthesized training pairs from VLM dataset.
/// Combines MinimapBakeService + HeightmapBakeService to create perfectly matched pairs
/// where the minimap is deformed/shaded based on the heightmap.
/// 
/// Output:
/// - Deformed minimap with hillshading, shadows, and terrain effects
/// - Clean heightmap (ground truth)
/// - Metadata with height bounds
/// 
/// This gives the model PERFECT ground truth pairs with no noise from real minimap
/// artifacts, lighting variations, or data corruption.
/// </summary>
public class SynthesizedTrainingService
{
    private readonly string _datasetRoot;
    private readonly MinimapBakeService _minimapBaker;
    private readonly HeightmapBakeService _heightmapBaker;
    
    // Lighting parameters for hillshading
    public float LightAzimuth { get; set; } = 315f;  // degrees, NW light
    public float LightAltitude { get; set; } = 45f;  // degrees above horizon
    public float HillshadeStrength { get; set; } = 0.4f;  // 0-1, blend with original
    public float AmbientOcclusion { get; set; } = 0.2f;  // darkening in valleys
    public float HeightTint { get; set; } = 0.15f;  // subtle color shift by elevation

    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true
    };

    public SynthesizedTrainingService(string datasetRoot)
    {
        _datasetRoot = datasetRoot;
        _minimapBaker = new MinimapBakeService(datasetRoot);
        _heightmapBaker = new HeightmapBakeService(datasetRoot);
    }

    /// <summary>
    /// Generate a complete synthesized training pair from a VLM JSON tile.
    /// </summary>
    public async Task<SynthesizedPair> GeneratePairAsync(string jsonPath, int resolution = 256)
    {
        // 1. Bake the clean minimap (4096x4096 full res)
        var minimap = await _minimapBaker.BakeTileAsync(jsonPath);
        
        // 2. Bake the heightmap
        var (heightmap16, heightMin, heightMax) = await _heightmapBaker.BakeHeightmap256Async(jsonPath);
        
        // 3. Convert heightmap to float array for processing
        var heightData = HeightmapToFloatArray(heightmap16);
        
        // 4. Apply deformation effects to minimap based on heightmap
        var deformedMinimap = ApplyTerrainEffects(minimap, heightData, resolution);
        
        // 5. Resize heightmap to match output resolution
        var heightmapOut = heightmap16.Clone();
        if (resolution != 256)
        {
            heightmapOut.Mutate(x => x.Resize(resolution, resolution, KnownResamplers.Bicubic));
        }
        
        // 6. Convert heightmap to 8-bit for training (normalized)
        var heightmap8 = ConvertTo8Bit(heightmapOut);
        
        minimap.Dispose();
        heightmap16.Dispose();
        heightmapOut.Dispose();
        
        return new SynthesizedPair
        {
            DeformedMinimap = deformedMinimap,
            Heightmap = heightmap8,
            HeightMin = heightMin,
            HeightMax = heightMax,
            TileName = Path.GetFileNameWithoutExtension(jsonPath),
            Resolution = resolution
        };
    }

    /// <summary>
    /// Applies terrain-based visual effects to the minimap.
    /// </summary>
    private Image<Rgba32> ApplyTerrainEffects(Image<Rgba32> minimap, float[,] heightData, int outputRes)
    {
        int hWidth = heightData.GetLength(1);
        int hHeight = heightData.GetLength(0);
        
        // Calculate gradients (normals) from heightmap
        var (gradX, gradY) = CalculateGradients(heightData);
        
        // Calculate hillshade
        var hillshade = CalculateHillshade(gradX, gradY);
        
        // Calculate ambient occlusion (valleys are darker)
        var ao = CalculateAmbientOcclusion(heightData);
        
        // Resize minimap to output resolution
        var result = minimap.Clone();
        result.Mutate(x => x.Resize(outputRes, outputRes, KnownResamplers.Lanczos3));
        
        // Apply effects
        result.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                
                // Map pixel coords to heightmap coords
                float hy = (float)y / accessor.Height * hHeight;
                int hyInt = Math.Clamp((int)hy, 0, hHeight - 1);
                
                for (int x = 0; x < row.Length; x++)
                {
                    float hx = (float)x / row.Length * hWidth;
                    int hxInt = Math.Clamp((int)hx, 0, hWidth - 1);
                    
                    // Get height-based factors
                    float shade = hillshade[hyInt, hxInt];
                    float occlusion = ao[hyInt, hxInt];
                    float heightNorm = heightData[hyInt, hxInt];
                    
                    // Apply hillshading
                    float shadeFactor = 1.0f - HillshadeStrength + (shade * HillshadeStrength);
                    
                    // Apply ambient occlusion (darken valleys)
                    float aoFactor = 1.0f - (AmbientOcclusion * (1.0f - occlusion));
                    
                    // Height-based tint (higher = slightly warmer, lower = cooler)
                    float warmShift = (heightNorm - 0.5f) * HeightTint;
                    
                    // Combine factors
                    float r = row[x].R / 255f;
                    float g = row[x].G / 255f;
                    float b = row[x].B / 255f;
                    
                    r = r * shadeFactor * aoFactor + warmShift * 0.5f;
                    g = g * shadeFactor * aoFactor;
                    b = b * shadeFactor * aoFactor - warmShift * 0.3f;
                    
                    row[x].R = (byte)Math.Clamp(r * 255, 0, 255);
                    row[x].G = (byte)Math.Clamp(g * 255, 0, 255);
                    row[x].B = (byte)Math.Clamp(b * 255, 0, 255);
                }
            }
        });
        
        return result;
    }

    /// <summary>
    /// Calculate X and Y gradients from heightmap.
    /// </summary>
    private (float[,] gradX, float[,] gradY) CalculateGradients(float[,] height)
    {
        int h = height.GetLength(0);
        int w = height.GetLength(1);
        var gradX = new float[h, w];
        var gradY = new float[h, w];
        
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                // Sobel-like gradient
                float left = x > 0 ? height[y, x - 1] : height[y, x];
                float right = x < w - 1 ? height[y, x + 1] : height[y, x];
                float up = y > 0 ? height[y - 1, x] : height[y, x];
                float down = y < h - 1 ? height[y + 1, x] : height[y, x];
                
                gradX[y, x] = (right - left) * 0.5f;
                gradY[y, x] = (down - up) * 0.5f;
            }
        }
        
        return (gradX, gradY);
    }

    /// <summary>
    /// Calculate hillshade based on light direction and terrain gradients.
    /// </summary>
    private float[,] CalculateHillshade(float[,] gradX, float[,] gradY)
    {
        int h = gradX.GetLength(0);
        int w = gradX.GetLength(1);
        var hillshade = new float[h, w];
        
        // Convert light angles to radians
        float azimuthRad = LightAzimuth * MathF.PI / 180f;
        float altitudeRad = LightAltitude * MathF.PI / 180f;
        
        // Light direction vector
        float lightX = MathF.Cos(altitudeRad) * MathF.Sin(azimuthRad);
        float lightY = MathF.Cos(altitudeRad) * MathF.Cos(azimuthRad);
        float lightZ = MathF.Sin(altitudeRad);
        
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                // Surface normal from gradients
                float nx = -gradX[y, x] * 10f;  // Scale factor for steepness
                float ny = -gradY[y, x] * 10f;
                float nz = 1f;
                
                // Normalize
                float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
                nx /= len; ny /= len; nz /= len;
                
                // Dot product with light direction
                float dot = nx * lightX + ny * lightY + nz * lightZ;
                hillshade[y, x] = Math.Clamp(dot, 0f, 1f);
            }
        }
        
        return hillshade;
    }

    /// <summary>
    /// Calculate ambient occlusion (valleys darker, ridges lighter).
    /// </summary>
    private float[,] CalculateAmbientOcclusion(float[,] height)
    {
        int h = height.GetLength(0);
        int w = height.GetLength(1);
        var ao = new float[h, w];
        
        // Simple AO: compare to local average
        int radius = 3;
        
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0;
                int count = 0;
                
                for (int dy = -radius; dy <= radius; dy++)
                {
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int ny = Math.Clamp(y + dy, 0, h - 1);
                        int nx = Math.Clamp(x + dx, 0, w - 1);
                        sum += height[ny, nx];
                        count++;
                    }
                }
                
                float avg = sum / count;
                float diff = height[y, x] - avg;
                
                // Positive diff = ridge (lighter), negative = valley (darker)
                ao[y, x] = Math.Clamp(0.5f + diff * 5f, 0f, 1f);
            }
        }
        
        return ao;
    }

    /// <summary>
    /// Convert 16-bit heightmap to float array [0-1].
    /// </summary>
    private float[,] HeightmapToFloatArray(Image<L16> heightmap)
    {
        var result = new float[heightmap.Height, heightmap.Width];
        
        heightmap.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    result[y, x] = row[x].PackedValue / 65535f;
                }
            }
        });
        
        return result;
    }

    /// <summary>
    /// Convert 16-bit heightmap to 8-bit grayscale.
    /// </summary>
    private Image<L8> ConvertTo8Bit(Image<L16> heightmap16)
    {
        var result = new Image<L8>(heightmap16.Width, heightmap16.Height);
        
        heightmap16.ProcessPixelRows(result, (src, dst) =>
        {
            for (int y = 0; y < src.Height; y++)
            {
                var srcRow = src.GetRowSpan(y);
                var dstRow = dst.GetRowSpan(y);
                for (int x = 0; x < srcRow.Length; x++)
                {
                    dstRow[x] = new L8((byte)(srcRow[x].PackedValue >> 8));
                }
            }
        });
        
        return result;
    }

    /// <summary>
    /// Export synthesized training pair to disk.
    /// </summary>
    public async Task ExportPairAsync(string jsonPath, string outputDir, int resolution = 256)
    {
        var pair = await GeneratePairAsync(jsonPath, resolution);
        
        Directory.CreateDirectory(outputDir);
        
        var baseName = pair.TileName;
        var minimapPath = Path.Combine(outputDir, $"{baseName}_synth_minimap.png");
        var heightmapPath = Path.Combine(outputDir, $"{baseName}_synth_heightmap.png");
        var metadataPath = Path.Combine(outputDir, $"{baseName}_synth_meta.json");
        
        await pair.DeformedMinimap.SaveAsPngAsync(minimapPath);
        await pair.Heightmap.SaveAsPngAsync(heightmapPath);
        
        var metadata = new
        {
            tile_name = pair.TileName,
            resolution = pair.Resolution,
            height_min = pair.HeightMin,
            height_max = pair.HeightMax,
            height_range = pair.HeightMax - pair.HeightMin,
            minimap_path = Path.GetFileName(minimapPath),
            heightmap_path = Path.GetFileName(heightmapPath),
            light_azimuth = LightAzimuth,
            light_altitude = LightAltitude,
            hillshade_strength = HillshadeStrength,
            ambient_occlusion = AmbientOcclusion
        };
        
        await File.WriteAllTextAsync(metadataPath, JsonSerializer.Serialize(metadata, _jsonOptions));
        
        pair.Dispose();
    }

    /// <summary>
    /// Export with multiple lighting variations for data augmentation.
    /// </summary>
    public async Task ExportWithVariationsAsync(string jsonPath, string outputDir, int resolution = 256)
    {
        var baseName = Path.GetFileNameWithoutExtension(jsonPath);
        
        // Variation 1: Default NW light
        LightAzimuth = 315f;
        await ExportPairAsync(jsonPath, outputDir, resolution);
        
        // Variation 2: NE light
        LightAzimuth = 45f;
        var pair2 = await GeneratePairAsync(jsonPath, resolution);
        await pair2.DeformedMinimap.SaveAsPngAsync(Path.Combine(outputDir, $"{baseName}_synth_minimap_ne.png"));
        pair2.Dispose();
        
        // Variation 3: SE light  
        LightAzimuth = 135f;
        var pair3 = await GeneratePairAsync(jsonPath, resolution);
        await pair3.DeformedMinimap.SaveAsPngAsync(Path.Combine(outputDir, $"{baseName}_synth_minimap_se.png"));
        pair3.Dispose();
        
        // Variation 4: SW light
        LightAzimuth = 225f;
        var pair4 = await GeneratePairAsync(jsonPath, resolution);
        await pair4.DeformedMinimap.SaveAsPngAsync(Path.Combine(outputDir, $"{baseName}_synth_minimap_sw.png"));
        pair4.Dispose();
        
        // Reset to default
        LightAzimuth = 315f;
    }
}

/// <summary>
/// A synthesized training pair with deformed minimap and ground truth heightmap.
/// </summary>
public class SynthesizedPair : IDisposable
{
    public Image<Rgba32> DeformedMinimap { get; set; } = null!;
    public Image<L8> Heightmap { get; set; } = null!;
    public float HeightMin { get; set; }
    public float HeightMax { get; set; }
    public string TileName { get; set; } = "";
    public int Resolution { get; set; }
    
    public void Dispose()
    {
        DeformedMinimap?.Dispose();
        Heightmap?.Dispose();
    }
}
