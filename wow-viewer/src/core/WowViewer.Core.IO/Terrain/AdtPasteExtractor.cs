using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Terrain;

/// <summary>
/// Extracts reusable terrain brush pastes directly from loaded authentic ADT chunk data (MCVT heights + MCLY/MCAL splats).
/// Zeroes height baselines and extracts individual layer alpha masks to create standalone <see cref="TerrainBrushPaste"/> instances.
/// </summary>
public static class AdtPasteExtractor
{
    public const int ChunkVertexCount = 145; // 9x9 outer + 8x8 inner
    public const int AlphaResolution = 64;
    public const int AlphaPixelCount = AlphaResolution * AlphaResolution;

    /// <summary>
    /// Extracts a single-chunk (33.33m x 33.33m) terrain paste from an LK MCNK chunk.
    /// </summary>
    public static TerrainBrushPaste ExtractFromChunk(
        LkMcnkData chunk,
        IReadOnlyList<string> textureTable,
        string id,
        string name,
        string category = "General",
        string[]? tags = null,
        string sourceMap = "",
        string sourceBuild = "")
    {
        ArgumentNullException.ThrowIfNull(chunk);
        ArgumentNullException.ThrowIfNull(textureTable);

        // 1. Resample 145 MCVT vertices into a 17x17 grid (interleaving 9x9 outer and 8x8 inner vertices)
        float[] gridHeights = Resample145HeightsTo17x17(chunk.Heights);

        // 2. Zero height baseline so deltas are relative
        float baseElevation = gridHeights.Length > 0 ? gridHeights[0] : 0f;
        for (int i = 0; i < gridHeights.Length; i++)
            gridHeights[i] -= baseElevation;

        // 3. Extract texture layers and alpha masks
        var layers = new List<TerrainPasteLayer>();
        for (int i = 0; i < chunk.Layers.Count; i++)
        {
            LkMclyEntry entry = chunk.Layers[i];
            string texPath = (entry.TextureId < textureTable.Count) ? textureTable[(int)entry.TextureId] : string.Empty;
            if (string.IsNullOrWhiteSpace(texPath))
                continue;

            byte[] alphaMask = (i == 0)
                ? CreateSolidAlpha(AlphaResolution, 255)
                : ExtractLayerAlpha(chunk, entry, i);

            layers.Add(new TerrainPasteLayer
            {
                TexturePath = texPath,
                Resolution = AlphaResolution,
                AlphaMask = alphaMask,
                EffectId = entry.EffectId,
                Flags = entry.Flags
            });
        }

        var paste = new TerrainBrushPaste
        {
            Id = id,
            Name = name,
            Category = category,
            Tags = tags ?? [],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = 17,
            ResolutionY = 17,
            HeightDeltas = gridHeights,
            Layers = layers,
            SourceMap = sourceMap,
            SourceBuild = sourceBuild
        };

        paste.CalculateMaxSlopeDegrees();
        return paste;
    }

    /// <summary>
    /// Resamples 145 MCVT heights (9x9 outer + 8x8 inner) into a continuous 17x17 grid.
    /// </summary>
    public static float[] Resample145HeightsTo17x17(float[] heights145)
    {
        var grid = new float[17 * 17];
        if (heights145 == null || heights145.Length < 145)
            return grid;

        // Fill 9x9 outer grid points at even (x, y) coordinates
        for (int oy = 0; oy < 9; oy++)
        {
            for (int ox = 0; ox < 9; ox++)
            {
                int srcIdx = oy * 17 + ox;
                grid[(oy * 2) * 17 + (ox * 2)] = heights145[srcIdx];
            }
        }

        // Fill 8x8 inner grid points at odd (x, y) coordinates
        for (int iy = 0; iy < 8; iy++)
        {
            for (int ix = 0; ix < 8; ix++)
            {
                int srcIdx = 9 + iy * 17 + ix;
                grid[(iy * 2 + 1) * 17 + (ix * 2 + 1)] = heights145[srcIdx];
            }
        }

        // Interpolate remaining edge midpoint cells
        for (int y = 0; y < 17; y++)
        {
            for (int x = 0; x < 17; x++)
            {
                if ((x % 2 == 1 && y % 2 == 0) || (x % 2 == 0 && y % 2 == 1))
                {
                    // Average 2 adjacent cardinal points
                    float sum = 0f;
                    int count = 0;
                    if (x > 0) { sum += grid[y * 17 + (x - 1)]; count++; }
                    if (x < 16) { sum += grid[y * 17 + (x + 1)]; count++; }
                    if (y > 0) { sum += grid[(y - 1) * 17 + x]; count++; }
                    if (y < 16) { sum += grid[(y + 1) * 17 + x]; count++; }
                    grid[y * 17 + x] = count > 0 ? sum / count : 0f;
                }
            }
        }

        return grid;
    }

    private static byte[] ExtractLayerAlpha(LkMcnkData chunk, LkMclyEntry entry, int layerIndex)
    {
        var alpha = new byte[AlphaPixelCount];
        if (chunk.AlphaMapData == null || chunk.AlphaMapData.Length == 0)
            return alpha;

        int offset = (int)entry.AlphaOffset;
        if (offset < 0 || offset >= chunk.AlphaMapData.Length)
            return alpha;

        bool isCompressed = (entry.Flags & 0x200) != 0;
        if (!isCompressed)
        {
            // Uncompressed 64x64 8-bit (4096 bytes) or 4-bit (2048 bytes)
            if (offset + 4096 <= chunk.AlphaMapData.Length)
            {
                Array.Copy(chunk.AlphaMapData, offset, alpha, 0, 4096);
            }
            else if (offset + 2048 <= chunk.AlphaMapData.Length)
            {
                // 4-bit unpack
                for (int i = 0; i < 2048; i++)
                {
                    byte b = chunk.AlphaMapData[offset + i];
                    byte low = (byte)((b & 0x0F) * 17);
                    byte high = (byte)(((b >> 4) & 0x0F) * 17);
                    alpha[i * 2] = low;
                    alpha[i * 2 + 1] = high;
                }
            }
        }
        else
        {
            // RLE Compressed MCAL
            int readPos = offset;
            int writePos = 0;
            byte[] src = chunk.AlphaMapData;

            while (writePos < AlphaPixelCount && readPos < src.Length)
            {
                byte tag = src[readPos++];
                bool fill = (tag & 0x80) != 0;
                int count = tag & 0x7F;

                if (fill)
                {
                    if (readPos >= src.Length) break;
                    byte val = src[readPos++];
                    for (int k = 0; k < count && writePos < AlphaPixelCount; k++)
                        alpha[writePos++] = val;
                }
                else
                {
                    for (int k = 0; k < count && writePos < AlphaPixelCount && readPos < src.Length; k++)
                        alpha[writePos++] = src[readPos++];
                }
            }
        }

        return alpha;
    }

    private static byte[] CreateSolidAlpha(int resolution, byte value)
    {
        var buf = new byte[resolution * resolution];
        Array.Fill(buf, value);
        return buf;
    }
}
