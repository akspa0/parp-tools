using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Reconstructs high-resolution minimap tiles from VLM dataset exports.
/// Composes 16x16 chunks using their layer metadata, textures, and alpha masks.
/// Supports shadow map subtraction (de-baking) and application (re-baking).
/// </summary>
public class MinimapBakeService
{
    private readonly string _datasetRoot;
    private readonly string _tilesetsDir;
    private readonly string _masksDir;
    private readonly string _shadowsDir;
    
    /// <summary>
    /// Shadow intensity for baking (0.0 = no shadow, 1.0 = full black).
    /// Default 0.5 gives a balanced shadow effect.
    /// </summary>
    public float ShadowIntensity { get; set; } = 0.5f;
    
    /// <summary>
    /// If true, inverts alpha mask values (255-value).
    /// Default false - alpha values used directly from MCAL.
    /// </summary>
    public bool InvertAlpha { get; set; } = false;
    
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public MinimapBakeService(string datasetRoot)
    {
        _datasetRoot = datasetRoot;
        _tilesetsDir = Path.Combine(datasetRoot, "tilesets");
        _masksDir = Path.Combine(datasetRoot, "masks");
        _shadowsDir = Path.Combine(datasetRoot, "shadows");
    }

    /// <summary>
    /// Bakes a full ADT tile minimap (4096x4096px).
    /// </summary>
    public async Task<Image<Rgba32>> BakeTileAsync(string jsonPath)
    {
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("JSON tile not found", jsonPath);

        var jsonContent = await File.ReadAllTextAsync(jsonPath);
        var sample = JsonSerializer.Deserialize<VlmTrainingSample>(jsonContent, _jsonOptions);
        
        if (sample?.TerrainData?.ChunkLayers == null)
            throw new Exception("Invalid VLM JSON data: missing chunk layers.");

        var fullTile = new Image<Rgba32>(16 * 256, 16 * 256);

        // Process chunks (0-255)
        // Row = index / 16, Col = index % 16
        foreach (var chunk in sample.TerrainData.ChunkLayers)
        {
            var row = chunk.ChunkIndex / 16;
            var col = chunk.ChunkIndex % 16;
            
            using var chunkImage = await BakeChunkAsync(chunk);
            
            // Place in the big image
            fullTile.Mutate(ctx => ctx.DrawImage(chunkImage, new Point(col * 256, row * 256), 1.0f));
        }

        return fullTile;
    }

    /// <summary>
    /// Bakes a tile and exports ALL layer combinations as separate PNGs for ViT training.
    /// Exports comprehensive layer data including shadows applied at each stage:
    /// 1. Raw texture layers (texture only, no alpha)
    /// 2. Weighted blend layers (texture * WoW weight, with alpha channel = weight)
    /// 3. Cumulative blend layers (progressive composite up to layer N)
    /// 4. Cumulative + shadows (progressive composite with shadows applied)
    /// 5. Shadow masks (per-chunk and full tile)
    /// </summary>
    /// <param name="jsonPath">Path to VLM JSON file</param>
    /// <param name="outputDir">Output directory for layer PNGs</param>
    /// <param name="applyShadows">If true, also exports shadowed versions</param>
    /// <returns>Tuple of (composite image, layer count, stats)</returns>
    public async Task<(Image<Rgba32> Composite, int LayerCount, string Stats)> BakeTileWithLayersAsync(
        string jsonPath, string outputDir, bool applyShadows = true)
    {
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("JSON tile not found", jsonPath);

        var jsonContent = await File.ReadAllTextAsync(jsonPath);
        var sample = JsonSerializer.Deserialize<VlmTrainingSample>(jsonContent, _jsonOptions);
        
        if (sample?.TerrainData?.ChunkLayers == null)
            throw new Exception("Invalid VLM JSON data: missing chunk layers.");

        var tileName = sample.TerrainData.AdtTile;
        var layersDir = Path.Combine(outputDir, $"{tileName}_layers");
        var rawDir = Path.Combine(layersDir, "raw");              // Raw textures only
        var weightedDir = Path.Combine(layersDir, "weighted");    // Texture * weight with alpha
        var cumulativeDir = Path.Combine(layersDir, "cumulative"); // Progressive blend (no shadow)
        var shadowedDir = Path.Combine(layersDir, "shadowed");    // Progressive blend + shadows
        var shadowMaskDir = Path.Combine(layersDir, "shadow_masks"); // Shadow masks
        
        Directory.CreateDirectory(rawDir);
        Directory.CreateDirectory(weightedDir);
        Directory.CreateDirectory(cumulativeDir);
        Directory.CreateDirectory(shadowedDir);
        Directory.CreateDirectory(shadowMaskDir);

        // Determine max layer count across all chunks
        int maxLayers = sample.TerrainData.ChunkLayers.Max(c => c.Layers?.Length ?? 0);
        
        // Create per-layer full-tile images for each export type
        var rawLayers = new Image<Rgba32>[maxLayers];
        var weightedLayers = new Image<Rgba32>[maxLayers];
        var cumulativeLayers = new Image<Rgba32>[maxLayers];
        var shadowedLayers = new Image<Rgba32>[maxLayers];
        
        for (int i = 0; i < maxLayers; i++)
        {
            rawLayers[i] = new Image<Rgba32>(16 * 256, 16 * 256);
            weightedLayers[i] = new Image<Rgba32>(16 * 256, 16 * 256);
            cumulativeLayers[i] = new Image<Rgba32>(16 * 256, 16 * 256);
            shadowedLayers[i] = new Image<Rgba32>(16 * 256, 16 * 256);
        }

        // Full tile shadow mask (composite of all chunk shadows)
        var fullShadowMask = new Image<L8>(16 * 256, 16 * 256);
        // Initialize to white (no shadow)
        fullShadowMask.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                    row[x] = new L8(255);
            }
        });

        // Final composite images (with and without shadows)
        var composite = new Image<Rgba32>(16 * 256, 16 * 256);
        var compositeWithShadows = new Image<Rgba32>(16 * 256, 16 * 256);
        
        int chunksProcessed = 0;
        int layersApplied = 0;
        int masksApplied = 0;
        int shadowsApplied = 0;

        foreach (var chunk in sample.TerrainData.ChunkLayers)
        {
            var row = chunk.ChunkIndex / 16;
            var col = chunk.ChunkIndex % 16;
            var point = new Point(col * 256, row * 256);
            
            // Load shadow mask for this chunk if available
            Image<L8>? chunkShadow = null;
            if (applyShadows && !string.IsNullOrEmpty(chunk.ShadowPath))
            {
                var shadowPath = Path.Combine(_datasetRoot, chunk.ShadowPath);
                if (File.Exists(shadowPath))
                {
                    chunkShadow = await Image.LoadAsync<L8>(shadowPath);
                    // Upscale 64x64 -> 256x256
                    if (chunkShadow.Width != 256 || chunkShadow.Height != 256)
                        chunkShadow.Mutate(x => x.Resize(256, 256));
                    
                    // Copy to full shadow mask
                    fullShadowMask.ProcessPixelRows(chunkShadow, (fullAccessor, chunkAccessor) =>
                    {
                        for (int y = 0; y < 256; y++)
                        {
                            var fullRow = fullAccessor.GetRowSpan(row * 256 + y);
                            var chunkRow = chunkAccessor.GetRowSpan(y);
                            for (int x = 0; x < 256; x++)
                                fullRow[col * 256 + x] = chunkRow[x];
                        }
                    });
                    shadowsApplied++;
                }
            }
            
            // Bake composite chunk using WoW weighted blend
            using var compositeChunk = await BakeChunkAsync(chunk);
            composite.Mutate(ctx => ctx.DrawImage(compositeChunk, point, 1.0f));
            
            // Create shadowed version
            var shadowedChunk = compositeChunk.Clone();
            if (chunkShadow != null)
            {
                ApplyShadowOverlay(shadowedChunk, chunkShadow);
            }
            compositeWithShadows.Mutate(ctx => ctx.DrawImage(shadowedChunk, point, 1.0f));
            shadowedChunk.Dispose();
            
            // Generate per-layer exports with WoW weighted blend data
            if (chunk.Layers != null && chunk.Layers.Length > 0)
            {
                var chunkLayerData = await BakeChunkLayersAsync(chunk);
                
                for (int layerIdx = 0; layerIdx < chunkLayerData.Length && layerIdx < maxLayers; layerIdx++)
                {
                    var (rawTex, weightedTex, cumulativeTex) = chunkLayerData[layerIdx];
                    
                    if (rawTex != null)
                    {
                        rawLayers[layerIdx].Mutate(ctx => ctx.DrawImage(rawTex, point, 1.0f));
                        rawTex.Dispose();
                    }
                    if (weightedTex != null)
                    {
                        weightedLayers[layerIdx].Mutate(ctx => ctx.DrawImage(weightedTex, point, 1.0f));
                        weightedTex.Dispose();
                        masksApplied++;
                    }
                    if (cumulativeTex != null)
                    {
                        // Save unshadowed cumulative
                        cumulativeLayers[layerIdx].Mutate(ctx => ctx.DrawImage(cumulativeTex, point, 1.0f));
                        
                        // Create shadowed version of cumulative
                        var shadowedCumulative = cumulativeTex.Clone();
                        if (chunkShadow != null)
                        {
                            using var shadowClone = chunkShadow.Clone();
                            ApplyShadowOverlay(shadowedCumulative, shadowClone);
                        }
                        shadowedLayers[layerIdx].Mutate(ctx => ctx.DrawImage(shadowedCumulative, point, 1.0f));
                        shadowedCumulative.Dispose();
                        
                        cumulativeTex.Dispose();
                    }
                    layersApplied++;
                }
            }
            
            chunkShadow?.Dispose();
            chunksProcessed++;
        }

        // Save all layer images
        for (int i = 0; i < maxLayers; i++)
        {
            await rawLayers[i].SaveAsPngAsync(Path.Combine(rawDir, $"{tileName}_layer{i}_raw.png"));
            await weightedLayers[i].SaveAsPngAsync(Path.Combine(weightedDir, $"{tileName}_layer{i}_weighted.png"));
            await cumulativeLayers[i].SaveAsPngAsync(Path.Combine(cumulativeDir, $"{tileName}_layer{i}_cumulative.png"));
            await shadowedLayers[i].SaveAsPngAsync(Path.Combine(shadowedDir, $"{tileName}_layer{i}_shadowed.png"));
            
            rawLayers[i].Dispose();
            weightedLayers[i].Dispose();
            cumulativeLayers[i].Dispose();
            shadowedLayers[i].Dispose();
        }
        
        // Save full tile shadow mask
        await fullShadowMask.SaveAsPngAsync(Path.Combine(shadowMaskDir, $"{tileName}_shadow_mask.png"));
        fullShadowMask.Dispose();
        
        // Save final composites
        await composite.SaveAsPngAsync(Path.Combine(outputDir, $"{tileName}_composite_noshadow.png"));
        await compositeWithShadows.SaveAsPngAsync(Path.Combine(outputDir, $"{tileName}_composite_shadowed.png"));
        compositeWithShadows.Dispose();

        var stats = $"Chunks: {chunksProcessed}, Layers: {maxLayers}, Applied: {layersApplied}, Masks: {masksApplied}, Shadows: {shadowsApplied}";
        return (composite, maxLayers, stats);
    }

    /// <summary>
    /// Bakes individual layer data for a chunk using WoW's weighted blend algorithm.
    /// Returns arrays of (raw texture, weighted texture with alpha, cumulative blend).
    /// </summary>
    private async Task<(Image<Rgba32>? Raw, Image<Rgba32>? Weighted, Image<Rgba32>? Cumulative)[]> BakeChunkLayersAsync(VlmChunkLayers chunk)
    {
        const int size = 256;
        
        if (chunk.Layers == null || chunk.Layers.Length == 0)
            return Array.Empty<(Image<Rgba32>?, Image<Rgba32>?, Image<Rgba32>?)>();

        var results = new (Image<Rgba32>?, Image<Rgba32>?, Image<Rgba32>?)[chunk.Layers.Length];
        
        // Load all textures and alpha masks first
        var textures = new Image<Rgba32>?[chunk.Layers.Length];
        var alphas = new byte[chunk.Layers.Length][];
        
        for (int i = 0; i < chunk.Layers.Length; i++)
        {
            var layer = chunk.Layers[i];
            if (string.IsNullOrEmpty(layer.TexturePath)) continue;

            var texName = Path.GetFileNameWithoutExtension(layer.TexturePath) + ".png";
            var texPath = Path.Combine(_tilesetsDir, texName);
            if (!File.Exists(texPath)) continue;

            textures[i] = await Image.LoadAsync<Rgba32>(texPath);
            if (textures[i]!.Width != size || textures[i]!.Height != size)
                textures[i]!.Mutate(x => x.Resize(size, size));

            // Load alpha mask for layers 1+
            if (i > 0 && !string.IsNullOrEmpty(layer.AlphaPath))
            {
                var maskPath = Path.Combine(_datasetRoot, layer.AlphaPath);
                if (File.Exists(maskPath))
                {
                    using var maskImage = await Image.LoadAsync<L8>(maskPath);
                    if (maskImage.Width != size || maskImage.Height != size)
                        maskImage.Mutate(x => x.Resize(size, size));
                    
                    alphas[i] = new byte[size * size];
                    for (int y = 0; y < size; y++)
                        for (int x = 0; x < size; x++)
                            alphas[i][y * size + x] = maskImage[x, y].PackedValue;
                }
            }
        }

        // Calculate weights per pixel using WoW algorithm
        var weights = new float[chunk.Layers.Length][]; 
        for (int i = 0; i < chunk.Layers.Length; i++)
            weights[i] = new float[size * size];

        for (int pixelIdx = 0; pixelIdx < size * size; pixelIdx++)
        {
            // Calculate alpha sum for layers 1+
            float alphaSum = 0f;
            for (int i = 1; i < chunk.Layers.Length; i++)
            {
                if (alphas[i] != null)
                    alphaSum += alphas[i][pixelIdx] / 255f;
            }
            alphaSum = Math.Clamp(alphaSum, 0f, 1f);

            // Layer 0 gets remainder, others get their alpha
            weights[0][pixelIdx] = 1f - alphaSum;
            for (int i = 1; i < chunk.Layers.Length; i++)
            {
                weights[i][pixelIdx] = alphas[i] != null ? alphas[i][pixelIdx] / 255f : 0f;
            }
        }

        // Generate outputs for each layer
        var cumulative = new Image<Rgba32>(size, size);
        
        for (int layerIdx = 0; layerIdx < chunk.Layers.Length; layerIdx++)
        {
            if (textures[layerIdx] == null)
            {
                results[layerIdx] = (null, null, null);
                continue;
            }

            // 1. Raw texture (clone)
            var rawTex = textures[layerIdx]!.Clone();

            // 2. Weighted texture (RGB * weight, A = weight * 255)
            var weightedTex = new Image<Rgba32>(size, size);
            var layerWeights = weights[layerIdx];
            
            weightedTex.ProcessPixelRows(textures[layerIdx]!, (dstAccessor, srcAccessor) =>
            {
                for (int y = 0; y < size; y++)
                {
                    var dstRow = dstAccessor.GetRowSpan(y);
                    var srcRow = srcAccessor.GetRowSpan(y);
                    for (int x = 0; x < size; x++)
                    {
                        int pixelIdx = y * size + x;
                        float w = layerWeights[pixelIdx];
                        var src = srcRow[x];
                        
                        // Store weighted color with weight as alpha
                        dstRow[x] = new Rgba32(
                            (byte)Math.Clamp(src.R * w, 0, 255),
                            (byte)Math.Clamp(src.G * w, 0, 255),
                            (byte)Math.Clamp(src.B * w, 0, 255),
                            (byte)Math.Clamp(w * 255, 0, 255)
                        );
                    }
                }
            });

            // 3. Cumulative blend (progressive composite up to this layer)
            // Add this layer's weighted contribution to cumulative
            cumulative.ProcessPixelRows(textures[layerIdx]!, (cumAccessor, srcAccessor) =>
            {
                for (int y = 0; y < size; y++)
                {
                    var cumRow = cumAccessor.GetRowSpan(y);
                    var srcRow = srcAccessor.GetRowSpan(y);
                    for (int x = 0; x < size; x++)
                    {
                        int pixelIdx = y * size + x;
                        float w = layerWeights[pixelIdx];
                        var src = srcRow[x];
                        var cum = cumRow[x];
                        
                        // Add weighted contribution
                        cumRow[x] = new Rgba32(
                            (byte)Math.Clamp(cum.R + src.R * w, 0, 255),
                            (byte)Math.Clamp(cum.G + src.G * w, 0, 255),
                            (byte)Math.Clamp(cum.B + src.B * w, 0, 255),
                            255
                        );
                    }
                }
            });

            // Clone cumulative for this layer's output
            var cumulativeTex = cumulative.Clone();

            results[layerIdx] = (rawTex, weightedTex, cumulativeTex);
        }

        // Cleanup textures
        foreach (var tex in textures)
            tex?.Dispose();

        return results;
    }

    /// <summary>
    /// Bakes a single 256x256 chunk from its layers using WoW's weighted blend algorithm.
    /// Based on wow.export's adt.fragment.shader:
    /// - Layer 0 weight = 1.0 - sum(layer1..N alphas)
    /// - Layer N weight = alpha[N]
    /// - Final color = sum(layer[i].rgb * weight[i]) / sum(weights)
    /// </summary>
    public async Task<Image<Rgba32>> BakeChunkAsync(VlmChunkLayers chunk)
    {
        const int size = 256;
        var chunkImage = new Image<Rgba32>(size, size);
        
        if (chunk.Layers == null || chunk.Layers.Length == 0)
            return chunkImage;

        // Load all textures and alpha masks
        var textures = new Image<Rgba32>[chunk.Layers.Length];
        var alphas = new byte[chunk.Layers.Length][];
        
        for (int i = 0; i < chunk.Layers.Length; i++)
        {
            var layer = chunk.Layers[i];
            if (string.IsNullOrEmpty(layer.TexturePath)) continue;

            var texName = Path.GetFileNameWithoutExtension(layer.TexturePath) + ".png";
            var texPath = Path.Combine(_tilesetsDir, texName);

            if (!File.Exists(texPath)) continue;

            textures[i] = await Image.LoadAsync<Rgba32>(texPath);
            if (textures[i].Width != size || textures[i].Height != size)
                textures[i].Mutate(x => x.Resize(size, size));

            // Load alpha mask for layers 1+
            if (i > 0 && !string.IsNullOrEmpty(layer.AlphaPath))
            {
                var maskPath = Path.Combine(_datasetRoot, layer.AlphaPath);
                if (File.Exists(maskPath))
                {
                    using var maskImage = await Image.LoadAsync<L8>(maskPath);
                    // Alpha masks are 64x64, upscale to 256x256
                    if (maskImage.Width != size || maskImage.Height != size)
                        maskImage.Mutate(x => x.Resize(size, size));
                    
                    alphas[i] = new byte[size * size];
                    for (int y = 0; y < size; y++)
                        for (int x = 0; x < size; x++)
                            alphas[i][y * size + x] = maskImage[x, y].PackedValue;
                }
            }
        }

        // WoW blending algorithm (from adt.fragment.shader):
        // layer_weights[0] = 1.0 - clamp(sum(alphas[1..N]), 0, 1)
        // layer_weights[i] = alphas[i] for i > 0
        // final_color = sum(layer[i].rgb * weight[i]) normalized
        
        chunkImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < size; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < size; x++)
                {
                    int pixelIdx = y * size + x;
                    
                    // Calculate alpha sum for layers 1+
                    float alphaSum = 0f;
                    for (int i = 1; i < chunk.Layers.Length; i++)
                    {
                        if (alphas[i] != null)
                            alphaSum += alphas[i][pixelIdx] / 255f;
                    }
                    alphaSum = Math.Clamp(alphaSum, 0f, 1f);

                    // Calculate weights
                    float[] weights = new float[chunk.Layers.Length];
                    weights[0] = 1f - alphaSum;  // Base layer gets remainder
                    for (int i = 1; i < chunk.Layers.Length; i++)
                    {
                        weights[i] = alphas[i] != null ? alphas[i][pixelIdx] / 255f : 0f;
                    }

                    // Blend colors
                    float r = 0, g = 0, b = 0;
                    float weightSum = 0;
                    
                    for (int i = 0; i < chunk.Layers.Length; i++)
                    {
                        if (textures[i] == null) continue;
                        
                        var texPixel = textures[i][x, y];
                        r += texPixel.R * weights[i];
                        g += texPixel.G * weights[i];
                        b += texPixel.B * weights[i];
                        weightSum += weights[i];
                    }

                    // Normalize
                    if (weightSum > 0)
                    {
                        r /= weightSum;
                        g /= weightSum;
                        b /= weightSum;
                    }

                    row[x] = new Rgba32(
                        (byte)Math.Clamp(r, 0, 255),
                        (byte)Math.Clamp(g, 0, 255),
                        (byte)Math.Clamp(b, 0, 255),
                        255
                    );
                }
            }
        });

        // Cleanup
        foreach (var tex in textures)
            tex?.Dispose();

        return chunkImage;
    }

    private void ApplyAlphaMask(Image<Rgba32> image, Image<L8> mask)
    {
        bool invert = InvertAlpha;
        image.ProcessPixelRows(mask, (imgAccessor, maskAccessor) =>
        {
            for (int y = 0; y < imgAccessor.Height; y++)
            {
                var imgRow = imgAccessor.GetRowSpan(y);
                var maskRow = maskAccessor.GetRowSpan(y);
                for (int x = 0; x < imgRow.Length; x++)
                {
                    byte alpha = maskRow[x].PackedValue;
                    imgRow[x].A = invert ? (byte)(255 - alpha) : alpha;
                }
            }
        });
    }

    /// <summary>
    /// Bakes a full ADT tile minimap with shadows applied (4096x4096px).
    /// </summary>
    /// <param name="jsonPath">Path to VLM JSON file</param>
    /// <param name="applyShadows">If true, applies shadow maps to darken shadowed areas</param>
    public async Task<Image<Rgba32>> BakeTileWithShadowsAsync(string jsonPath, bool applyShadows = true)
    {
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("JSON tile not found", jsonPath);

        var jsonContent = await File.ReadAllTextAsync(jsonPath);
        var sample = JsonSerializer.Deserialize<VlmTrainingSample>(jsonContent, _jsonOptions);
        
        if (sample?.TerrainData?.ChunkLayers == null)
            throw new Exception("Invalid VLM JSON data: missing chunk layers.");

        var fullTile = new Image<Rgba32>(16 * 256, 16 * 256);

        foreach (var chunk in sample.TerrainData.ChunkLayers)
        {
            var row = chunk.ChunkIndex / 16;
            var col = chunk.ChunkIndex % 16;
            
            using var chunkImage = await BakeChunkAsync(chunk);
            
            // Apply shadow if available
            if (applyShadows && !string.IsNullOrEmpty(chunk.ShadowPath))
            {
                var shadowPath = Path.Combine(_datasetRoot, chunk.ShadowPath);
                if (File.Exists(shadowPath))
                {
                    using var shadowImage = await Image.LoadAsync<L8>(shadowPath);
                    ApplyShadowOverlay(chunkImage, shadowImage);
                }
            }
            
            fullTile.Mutate(ctx => ctx.DrawImage(chunkImage, new Point(col * 256, row * 256), 1.0f));
        }

        return fullTile;
    }

    /// <summary>
    /// Applies shadow overlay to an image. Shadow map is 64x64, upscaled to match image.
    /// WoW shadow convention: white (255) = transparent/no shadow, black (0) = opaque/full shadow.
    /// </summary>
    private void ApplyShadowOverlay(Image<Rgba32> image, Image<L8> shadow)
    {
        // Upscale shadow to match image size (64x64 -> 256x256)
        if (shadow.Width != image.Width || shadow.Height != image.Height)
        {
            shadow.Mutate(x => x.Resize(image.Width, image.Height));
        }

        float intensity = ShadowIntensity;
        
        image.ProcessPixelRows(shadow, (imgAccessor, shadowAccessor) =>
        {
            for (int y = 0; y < imgAccessor.Height; y++)
            {
                var imgRow = imgAccessor.GetRowSpan(y);
                var shadowRow = shadowAccessor.GetRowSpan(y);
                for (int x = 0; x < imgRow.Length; x++)
                {
                    // WoW shadow: white (255) = transparent, black (0) = opaque shadow
                    // So shadowValue directly represents how much light gets through
                    float lightFactor = shadowRow[x].PackedValue / 255f;
                    // Apply intensity: lerp between full light (1.0) and shadow
                    float darken = 1.0f - (1.0f - lightFactor) * intensity;
                    
                    imgRow[x].R = (byte)(imgRow[x].R * darken);
                    imgRow[x].G = (byte)(imgRow[x].G * darken);
                    imgRow[x].B = (byte)(imgRow[x].B * darken);
                }
            }
        });
    }

    /// <summary>
    /// Removes shadow from an image (de-baking). Useful for extracting clean textures.
    /// This is the inverse of ApplyShadowOverlay - it brightens shadowed areas.
    /// </summary>
    private void RemoveShadowOverlay(Image<Rgba32> image, Image<L8> shadow)
    {
        // Upscale shadow to match image size
        if (shadow.Width != image.Width || shadow.Height != image.Height)
        {
            shadow.Mutate(x => x.Resize(image.Width, image.Height));
        }

        float intensity = ShadowIntensity;
        
        image.ProcessPixelRows(shadow, (imgAccessor, shadowAccessor) =>
        {
            for (int y = 0; y < imgAccessor.Height; y++)
            {
                var imgRow = imgAccessor.GetRowSpan(y);
                var shadowRow = shadowAccessor.GetRowSpan(y);
                for (int x = 0; x < imgRow.Length; x++)
                {
                    float shadowFactor = shadowRow[x].PackedValue / 255f;
                    float darken = 1.0f - (1.0f - shadowFactor) * intensity;
                    
                    // Inverse: divide instead of multiply (with clamp)
                    if (darken > 0.01f)
                    {
                        imgRow[x].R = (byte)Math.Min(255, imgRow[x].R / darken);
                        imgRow[x].G = (byte)Math.Min(255, imgRow[x].G / darken);
                        imgRow[x].B = (byte)Math.Min(255, imgRow[x].B / darken);
                    }
                }
            }
        });
    }

    /// <summary>
    /// De-bakes shadows from a minimap tile PNG, producing a "clean" unshadowed version.
    /// </summary>
    /// <param name="minimapPath">Path to source minimap PNG (256x256 per chunk)</param>
    /// <param name="jsonPath">Path to VLM JSON with shadow paths</param>
    /// <returns>Minimap with shadows removed</returns>
    public async Task<Image<Rgba32>> DebakeShadowsFromMinimapAsync(string minimapPath, string jsonPath)
    {
        if (!File.Exists(minimapPath))
            throw new FileNotFoundException("Minimap not found", minimapPath);
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("JSON tile not found", jsonPath);

        var jsonContent = await File.ReadAllTextAsync(jsonPath);
        var sample = JsonSerializer.Deserialize<VlmTrainingSample>(jsonContent, _jsonOptions);
        
        if (sample?.TerrainData?.ChunkLayers == null)
            throw new Exception("Invalid VLM JSON data: missing chunk layers.");

        using var sourceImage = await Image.LoadAsync<Rgba32>(minimapPath);
        var result = sourceImage.Clone();

        // Process each chunk's shadow
        foreach (var chunk in sample.TerrainData.ChunkLayers)
        {
            if (string.IsNullOrEmpty(chunk.ShadowPath)) continue;
            
            var shadowPath = Path.Combine(_datasetRoot, chunk.ShadowPath);
            if (!File.Exists(shadowPath)) continue;

            var row = chunk.ChunkIndex / 16;
            var col = chunk.ChunkIndex % 16;
            int startX = col * 256;
            int startY = row * 256;

            using var shadowImage = await Image.LoadAsync<L8>(shadowPath);
            // Upscale shadow 64x64 -> 256x256
            shadowImage.Mutate(x => x.Resize(256, 256));

            // Extract chunk region, remove shadow, put back
            result.ProcessPixelRows(shadowImage, (imgAccessor, shadowAccessor) =>
            {
                for (int y = 0; y < 256 && (startY + y) < imgAccessor.Height; y++)
                {
                    var imgRow = imgAccessor.GetRowSpan(startY + y);
                    var shadowRow = shadowAccessor.GetRowSpan(y);
                    for (int x = 0; x < 256 && (startX + x) < imgRow.Length; x++)
                    {
                        float shadowFactor = shadowRow[x].PackedValue / 255f;
                        float darken = 1.0f - (1.0f - shadowFactor) * ShadowIntensity;
                        
                        if (darken > 0.01f)
                        {
                            int px = startX + x;
                            imgRow[px].R = (byte)Math.Min(255, imgRow[px].R / darken);
                            imgRow[px].G = (byte)Math.Min(255, imgRow[px].G / darken);
                            imgRow[px].B = (byte)Math.Min(255, imgRow[px].B / darken);
                        }
                    }
                }
            });
        }

        return result;
    }

    /// <summary>
    /// Bakes shadows onto a clean minimap tile, producing a shadowed version.
    /// </summary>
    /// <param name="cleanMinimapPath">Path to unshadowed minimap PNG</param>
    /// <param name="jsonPath">Path to VLM JSON with shadow paths</param>
    /// <returns>Minimap with shadows applied</returns>
    public async Task<Image<Rgba32>> BakeShadowsOntoMinimapAsync(string cleanMinimapPath, string jsonPath)
    {
        if (!File.Exists(cleanMinimapPath))
            throw new FileNotFoundException("Clean minimap not found", cleanMinimapPath);
        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("JSON tile not found", jsonPath);

        var jsonContent = await File.ReadAllTextAsync(jsonPath);
        var sample = JsonSerializer.Deserialize<VlmTrainingSample>(jsonContent, _jsonOptions);
        
        if (sample?.TerrainData?.ChunkLayers == null)
            throw new Exception("Invalid VLM JSON data: missing chunk layers.");

        using var sourceImage = await Image.LoadAsync<Rgba32>(cleanMinimapPath);
        var result = sourceImage.Clone();

        foreach (var chunk in sample.TerrainData.ChunkLayers)
        {
            if (string.IsNullOrEmpty(chunk.ShadowPath)) continue;
            
            var shadowPath = Path.Combine(_datasetRoot, chunk.ShadowPath);
            if (!File.Exists(shadowPath)) continue;

            var row = chunk.ChunkIndex / 16;
            var col = chunk.ChunkIndex % 16;
            int startX = col * 256;
            int startY = row * 256;

            using var shadowImage = await Image.LoadAsync<L8>(shadowPath);
            shadowImage.Mutate(x => x.Resize(256, 256));

            result.ProcessPixelRows(shadowImage, (imgAccessor, shadowAccessor) =>
            {
                for (int y = 0; y < 256 && (startY + y) < imgAccessor.Height; y++)
                {
                    var imgRow = imgAccessor.GetRowSpan(startY + y);
                    var shadowRow = shadowAccessor.GetRowSpan(y);
                    for (int x = 0; x < 256 && (startX + x) < imgRow.Length; x++)
                    {
                        float shadowFactor = shadowRow[x].PackedValue / 255f;
                        float darken = 1.0f - (1.0f - shadowFactor) * ShadowIntensity;
                        
                        int px = startX + x;
                        imgRow[px].R = (byte)(imgRow[px].R * darken);
                        imgRow[px].G = (byte)(imgRow[px].G * darken);
                        imgRow[px].B = (byte)(imgRow[px].B * darken);
                    }
                }
            });
        }

        return result;
    }
}
