using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// High-precision sampler for rasterizing 2D tile images (such as 256x256 minimap PNGs)
/// directly into 145-vertex MCCV BGRA chunks (580 bytes each) across all 256 MCNKs of an ADT tile.
/// </summary>
public static class AdtMccvImageSampler
{
    public const int TileChunks = 16;
    public const int VerticesPerChunk = 145;
    public const int BytesPerChunk = 580; // 145 * 4 BGRA
    public const int McnkHasMccvFlag = 0x40;

    /// <summary>
    /// Samples an image into 256 MCCV chunk arrays, indexed by (chunkY * 16 + chunkX).
    /// </summary>
    public static Dictionary<int, byte[]> SampleMccvFromImage(Image<Rgba32> image)
    {
        ArgumentNullException.ThrowIfNull(image);

        if (image.Width <= 0 || image.Height <= 0)
            throw new ArgumentException("Image must have non-zero dimensions.", nameof(image));

        var result = new Dictionary<int, byte[]>(TileChunks * TileChunks);

        for (int cy = 0; cy < TileChunks; cy++)
        {
            for (int cx = 0; cx < TileChunks; cx++)
            {
                byte[] chunkBytes = SampleChunkMccv(image, cx, cy);
                result[cy * TileChunks + cx] = chunkBytes;
            }
        }

        return result;
    }

    /// <summary>
    /// Loads a PNG from disk and samples it into 256 MCCV chunk arrays.
    /// </summary>
    public static Dictionary<int, byte[]>? SampleMccvFromPng(string pngPath)
    {
        if (string.IsNullOrWhiteSpace(pngPath) || !File.Exists(pngPath))
            return null;

        try
        {
            using var image = Image.Load<Rgba32>(pngPath);
            return SampleMccvFromImage(image);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Injects synthesized MCCV vertex colors into an existing LkAdtData structure,
    /// setting the McnkHasMccvFlag (0x40) on each chunk and providing a neutral
    /// base texture layer if the ADT chunks have zero texture layers.
    /// </summary>
    public static LkAdtData InjectMccvIntoLkAdt(
        LkAdtData baseAdt,
        Dictionary<int, byte[]> mccvData,
        string fallbackTexture = "tileset/ocean/westfallseafloor.blp")
    {
        ArgumentNullException.ThrowIfNull(baseAdt);
        ArgumentNullException.ThrowIfNull(mccvData);

        var textureNames = baseAdt.TextureNames.Count > 0
            ? baseAdt.TextureNames.ToList()
            : [fallbackTexture];

        var updatedChunks = new List<LkMcnkData>(baseAdt.Chunks.Count);

        for (int i = 0; i < baseAdt.Chunks.Count; i++)
        {
            var chunk = baseAdt.Chunks[i];
            int chunkIndex = (chunk.IndexY * TileChunks) + chunk.IndexX;

            byte[]? colors = mccvData.TryGetValue(chunkIndex, out byte[]? val)
                ? val
                : chunk.MccvColors;

            var layers = chunk.Layers;
            int nLayers = chunk.NLayers;
            if (layers.Count == 0)
            {
                layers = [new LkMclyEntry(TextureId: 0, Flags: 0, AlphaOffset: 0, EffectId: 0)];
                nLayers = 1;
            }

            int flags = chunk.Flags;
            if (colors != null && colors.Length >= BytesPerChunk)
            {
                flags |= McnkHasMccvFlag;
            }

            updatedChunks.Add(new LkMcnkData
            {
                IndexX = chunk.IndexX,
                IndexY = chunk.IndexY,
                Flags = flags,
                AreaId = chunk.AreaId,
                NLayers = nLayers,
                HoleMask = chunk.HoleMask,
                BaseHeight = chunk.BaseHeight,
                Heights = chunk.Heights,
                Normals = chunk.Normals,
                ShadowMap = chunk.ShadowMap,
                AlphaMapData = chunk.AlphaMapData,
                AlphaMapSize = chunk.AlphaMapSize,
                Layers = layers,
                DoodadRefs = chunk.DoodadRefs,
                WorldModelRefs = chunk.WorldModelRefs,
                LiquidData = chunk.LiquidData,
                MccvColors = colors,
                MclvLighting = chunk.MclvLighting,
                PosX = chunk.PosX,
                PosY = chunk.PosY,
                PosZ = chunk.PosZ,
            });
        }

        return new LkAdtData
        {
            MapName = baseAdt.MapName,
            TileX = baseAdt.TileX,
            TileY = baseAdt.TileY,
            TextureNames = textureNames,
            ModelNames = baseAdt.ModelNames,
            WorldModelNames = baseAdt.WorldModelNames,
            ModelPlacements = baseAdt.ModelPlacements,
            WorldModelPlacements = baseAdt.WorldModelPlacements,
            Chunks = updatedChunks,
            MhdrFlags = baseAdt.MhdrFlags,
            MfboFlightBounds = baseAdt.MfboFlightBounds,
        };
    }

    private static byte[] SampleChunkMccv(Image<Rgba32> image, int chunkX, int chunkY)
    {
        var data = new byte[BytesPerChunk];
        int vertexIndex = 0;

        // 9x9 Outer Grid (0..80)
        for (int row = 0; row < 9; row++)
        {
            for (int col = 0; col < 9; col++)
            {
                var color = SampleImageBilinear(image, chunkX + (col / 8f), chunkY + (row / 8f));
                WriteVertexBgra(data, vertexIndex++, color);
            }

            if (row >= 8)
                continue;

            // 8x8 Inner Grid (81..144)
            for (int col = 0; col < 8; col++)
            {
                var color = SampleImageBilinear(image, chunkX + ((col + 0.5f) / 8f), chunkY + ((row + 0.5f) / 8f));
                WriteVertexBgra(data, vertexIndex++, color);
            }
        }

        return data;
    }

    private static Rgba32 SampleImageBilinear(Image<Rgba32> image, float tileChunkX, float tileChunkY)
    {
        float pixelX = Math.Clamp(tileChunkX / TileChunks * (image.Width - 1), 0f, image.Width - 1);
        float pixelY = Math.Clamp(tileChunkY / TileChunks * (image.Height - 1), 0f, image.Height - 1);

        int x0 = (int)MathF.Floor(pixelX);
        int y0 = (int)MathF.Floor(pixelY);
        int x1 = Math.Min(x0 + 1, image.Width - 1);
        int y1 = Math.Min(y0 + 1, image.Height - 1);
        float tx = pixelX - x0;
        float ty = pixelY - y0;

        var c00 = image[x0, y0];
        var c10 = image[x1, y0];
        var c01 = image[x0, y1];
        var c11 = image[x1, y1];

        return new Rgba32(
            ToByte(Bilerp(c00.R, c10.R, c01.R, c11.R, tx, ty)),
            ToByte(Bilerp(c00.G, c10.G, c01.G, c11.G, tx, ty)),
            ToByte(Bilerp(c00.B, c10.B, c01.B, c11.B, tx, ty)),
            ToByte(Bilerp(c00.A, c10.A, c01.A, c11.A, tx, ty)));
    }

    private static float Bilerp(float c00, float c10, float c01, float c11, float tx, float ty)
    {
        float top = c00 + (c10 - c00) * tx;
        float bottom = c01 + (c11 - c01) * tx;
        return top + (bottom - top) * ty;
    }

    private static void WriteVertexBgra(byte[] data, int vertexIndex, Rgba32 color)
    {
        // WoW MCCV format is BGRA (145 vertices * 4 bytes = 580 bytes)
        int offset = vertexIndex * 4;
        data[offset + 0] = color.B;
        data[offset + 1] = color.G;
        data[offset + 2] = color.R;
        data[offset + 3] = 255;
    }

    private static byte ToByte(float value) => (byte)Math.Clamp((int)MathF.Round(value), 0, 255);
}
