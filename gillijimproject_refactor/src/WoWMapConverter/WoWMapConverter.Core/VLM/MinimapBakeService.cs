using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Reconstructs high-resolution minimap tiles from VLM dataset exports.
/// Composes 16x16 chunks using their layer metadata, textures, and alpha masks.
/// </summary>
public class MinimapBakeService
{
    private readonly string _datasetRoot;
    private readonly string _tilesetsDir;
    private readonly string _masksDir;
    
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public MinimapBakeService(string datasetRoot)
    {
        _datasetRoot = datasetRoot;
        _tilesetsDir = Path.Combine(datasetRoot, "tilesets");
        _masksDir = Path.Combine(datasetRoot, "masks");
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
    /// Bakes a single 256x256 chunk from its layers.
    /// </summary>
    public async Task<Image<Rgba32>> BakeChunkAsync(VlmChunkLayers chunk)
    {
        var chunkImage = new Image<Rgba32>(256, 256);

        foreach (var layer in chunk.Layers)
        {
            if (string.IsNullOrEmpty(layer.TexturePath)) continue;

            // 1. Resolve Texture Path
            // TexturePath is e.g. "Tileset\Kalidar\KalidarWood.blp"
            // We expect "tilesets\KalidarWood.png"
            var texName = Path.GetFileNameWithoutExtension(layer.TexturePath) + ".png";
            var texPath = Path.Combine(_tilesetsDir, texName);

            if (!File.Exists(texPath)) continue;

            using var texImage = await Image.LoadAsync<Rgba32>(texPath);
            
            // Ensure texture is 256x256
            if (texImage.Width != 256 || texImage.Height != 256)
            {
                texImage.Mutate(x => x.Resize(256, 256));
            }

            // 2. Resolve Alpha Mask
            if (layer.TextureId == 0 || string.IsNullOrEmpty(layer.AlphaPath))
            {
                // Layer 0 is opaque base
                chunkImage.Mutate(ctx => ctx.DrawImage(texImage, 1.0f));
            }
            else
            {
                // Layers 1+ have alpha masks
                var maskPath = Path.Combine(_datasetRoot, layer.AlphaPath);
                if (File.Exists(maskPath))
                {
                    using var maskImage = await Image.LoadAsync<L8>(maskPath);
                    if (maskImage.Width != 256 || maskImage.Height != 256)
                    {
                        maskImage.Mutate(x => x.Resize(256, 256));
                    }

                    // Apply mask to texture
                    ApplyAlphaMask(texImage, maskImage);
                    
                    // Blend onto chunk
                    chunkImage.Mutate(ctx => ctx.DrawImage(texImage, 1.0f));
                }
            }
        }

        return chunkImage;
    }

    private void ApplyAlphaMask(Image<Rgba32> image, Image<L8> mask)
    {
        image.ProcessPixelRows(mask, (imgAccessor, maskAccessor) =>
        {
            for (int y = 0; y < imgAccessor.Height; y++)
            {
                var imgRow = imgAccessor.GetRowSpan(y);
                var maskRow = maskAccessor.GetRowSpan(y);
                for (int x = 0; x < imgRow.Length; x++)
                {
                    imgRow[x].A = maskRow[x].PackedValue;
                }
            }
        });
    }
}
