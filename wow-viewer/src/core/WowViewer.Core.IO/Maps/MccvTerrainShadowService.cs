using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Service for extracting, rasterizing, synthesizing, and injecting 1.60 MCCV
/// terrain vertex shadow data (145 vertices per chunk, 580 bytes BGRA)
/// per Spec 263 (WoW: Forever 1.60 terrain self-shadow and ambient occlusion engine).
/// </summary>
public static class MccvTerrainShadowService
{
    public const int TileChunks = 16;
    public const int VerticesPerChunk = 145;
    public const int BytesPerChunk = 580; // 145 * 4 BGRA
    public const int McnkHasMccvFlag = 0x40;
    public const byte NeutralChannelValue = 127;
    public const float NeutralLuminance = 0.5f;

    /// <summary>
    /// Generates normalized (U, V) coordinates in [0, 1] for all 145 vertices of a chunk.
    /// Index 0..80: 9x9 outer grid (spacing 1/8).
    /// Index 81..144: 8x8 inner grid (spacing 1/8, offset 1/16).
    /// </summary>
    public static (float U, float V)[] GetChunkVertexCoordinates()
    {
        var coords = new (float U, float V)[VerticesPerChunk];

        // 9x9 Outer grid (0..80)
        int idx = 0;
        for (int y = 0; y < 9; y++)
        {
            for (int x = 0; x < 9; x++)
            {
                coords[idx++] = (x / 8.0f, y / 8.0f);
            }
        }

        // 8x8 Inner grid (81..144)
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                coords[idx++] = ((x + 0.5f) / 8.0f, ((y + 0.5f) / 8.0f));
            }
        }

        return coords;
    }

    /// <summary>
    /// Extracts and rasterizes MCCV vertex colors from a collection of chunks into a continuous 2D luminance float grid.
    /// Luminance is in [0.0, 1.0], where 0.5 (~127) represents neutral 1.0x terrain lighting multiplier.
    /// </summary>
    public static float[,] ExtractMccvTileLuminance(IReadOnlyDictionary<int, byte[]> chunkColors, int tileRes = 256)
    {
        ArgumentNullException.ThrowIfNull(chunkColors);
        ArgumentOutOfRangeException.ThrowIfLessThan(tileRes, 16);

        var raster = new float[tileRes, tileRes];
        int chunkPixels = tileRes / TileChunks;
        byte[] neutralChunk = CreateNeutralChunkBytes();

        for (int cy = 0; cy < TileChunks; cy++)
        {
            for (int cx = 0; cx < TileChunks; cx++)
            {
                int chunkIndex = (cy * TileChunks) + cx;
                byte[] chunkData = chunkColors.TryGetValue(chunkIndex, out byte[]? data) && data.Length >= BytesPerChunk
                    ? data
                    : neutralChunk;

                float[] vertexLuma = ExtractVertexLuminance(chunkData);

                // Sample 145 vertices over chunk pixel grid
                for (int py = 0; py < chunkPixels; py++)
                {
                    float localY = py / (float)Math.Max(1, chunkPixels - 1);
                    for (int px = 0; px < chunkPixels; px++)
                    {
                        float localX = px / (float)Math.Max(1, chunkPixels - 1);
                        float luma = SampleChunkLuminance(vertexLuma, localX, localY);

                        int outX = (cx * chunkPixels) + px;
                        int outY = (cy * chunkPixels) + py;
                        if (outX < tileRes && outY < tileRes)
                        {
                            raster[outY, outX] = luma;
                        }
                    }
                }
            }
        }

        return raster;
    }

    /// <summary>
    /// Extracts and rasterizes MCCV vertex colors from a list of LkMcnkData chunks.
    /// </summary>
    public static float[,] ExtractMccvTileLuminance(IReadOnlyList<LkMcnkData> chunks, int tileRes = 256)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        var dict = new Dictionary<int, byte[]>(chunks.Count);
        foreach (LkMcnkData chunk in chunks)
        {
            if (chunk.MccvColors is { Length: >= BytesPerChunk })
            {
                int index = (chunk.IndexY * TileChunks) + chunk.IndexX;
                dict[index] = chunk.MccvColors;
            }
        }

        return ExtractMccvTileLuminance(dict, tileRes);
    }

    /// <summary>
    /// Synthesizes 1.60-compliant MCCV vertex color chunks (145 vertices BGRA: 580 bytes)
    /// from a 2D residual shadow luminance grid (typically 256x256).
    /// </summary>
    public static Dictionary<int, byte[]> SynthesizeMccvChunks(float[,] residualShadow)
    {
        ArgumentNullException.ThrowIfNull(residualShadow);

        int height = residualShadow.GetLength(0);
        int width = residualShadow.GetLength(1);
        if (height < 16 || width < 16)
            throw new ArgumentException("Residual shadow map must be at least 16x16.", nameof(residualShadow));

        (float U, float V)[] vertexCoords = GetChunkVertexCoordinates();
        var result = new Dictionary<int, byte[]>(TileChunks * TileChunks);

        for (int cy = 0; cy < TileChunks; cy++)
        {
            for (int cx = 0; cx < TileChunks; cx++)
            {
                byte[] bgra = new byte[BytesPerChunk];

                for (int vIdx = 0; vIdx < VerticesPerChunk; vIdx++)
                {
                    (float u, float v) = vertexCoords[vIdx];

                    // Map to continuous grid pixel coordinates
                    float gx = ((cx + u) / TileChunks) * (width - 1);
                    float gy = ((cy + v) / TileChunks) * (height - 1);

                    float luma = SampleBilinear(residualShadow, gx, gy, width, height);
                    byte byteVal = (byte)Math.Clamp((int)MathF.Round(luma * 255.0f), 0, 255);

                    int byteOffset = vIdx * 4;
                    bgra[byteOffset + 0] = byteVal; // Blue
                    bgra[byteOffset + 1] = byteVal; // Green
                    bgra[byteOffset + 2] = byteVal; // Red
                    bgra[byteOffset + 3] = 255;     // Alpha
                }

                int chunkIndex = (cy * TileChunks) + cx;
                result[chunkIndex] = bgra;
            }
        }

        return result;
    }

    /// <summary>
    /// Injects synthesized 1.60 MCCV vertex shadow arrays into a list of LkMcnkData chunks,
    /// setting the McnkHasMccvFlag (0x40).
    /// </summary>
    public static IReadOnlyList<LkMcnkData> InjectMccvIntoChunks(
        IReadOnlyList<LkMcnkData> chunks,
        float[,] residualShadow)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(residualShadow);

        Dictionary<int, byte[]> synthesized = SynthesizeMccvChunks(residualShadow);
        var output = new List<LkMcnkData>(chunks.Count);

        foreach (LkMcnkData chunk in chunks)
        {
            int index = (chunk.IndexY * TileChunks) + chunk.IndexX;
            byte[] mccv = synthesized.TryGetValue(index, out byte[]? val)
                ? val
                : CreateNeutralChunkBytes();

            output.Add(new LkMcnkData
            {
                IndexX = chunk.IndexX,
                IndexY = chunk.IndexY,
                Flags = chunk.Flags | McnkHasMccvFlag,
                AreaId = chunk.AreaId,
                NLayers = chunk.NLayers,
                HoleMask = chunk.HoleMask,
                BaseHeight = chunk.BaseHeight,
                Heights = chunk.Heights,
                Normals = chunk.Normals,
                ShadowMap = chunk.ShadowMap,
                AlphaMapData = chunk.AlphaMapData,
                AlphaMapSize = chunk.AlphaMapSize,
                Layers = chunk.Layers,
                DoodadRefs = chunk.DoodadRefs,
                WorldModelRefs = chunk.WorldModelRefs,
                LiquidData = chunk.LiquidData,
                MccvColors = mccv,
                MclvLighting = chunk.MclvLighting,
                PosX = chunk.PosX,
                PosY = chunk.PosY,
                PosZ = chunk.PosZ,
            });
        }

        return output;
    }

    /// <summary>
    /// Computes the Normalized Cross-Correlation (NCC) between two 2D scalar fields of identical dimensions.
    /// </summary>
    public static float ComputeNormalizedCrossCorrelation(float[,] s1, float[,] s2)
    {
        ArgumentNullException.ThrowIfNull(s1);
        ArgumentNullException.ThrowIfNull(s2);

        int h1 = s1.GetLength(0);
        int w1 = s1.GetLength(1);
        int h2 = s2.GetLength(0);
        int w2 = s2.GetLength(1);

        if (h1 != h2 || w1 != w2)
            throw new ArgumentException("Matrices must have identical dimensions for NCC computation.");

        double sum1 = 0.0;
        double sum2 = 0.0;
        int count = h1 * w1;

        for (int y = 0; y < h1; y++)
        {
            for (int x = 0; x < w1; x++)
            {
                sum1 += s1[y, x];
                sum2 += s2[y, x];
            }
        }

        double mean1 = sum1 / count;
        double mean2 = sum2 / count;

        double num = 0.0;
        double den1 = 0.0;
        double den2 = 0.0;

        for (int y = 0; y < h1; y++)
        {
            for (int x = 0; x < w1; x++)
            {
                double d1 = s1[y, x] - mean1;
                double d2 = s2[y, x] - mean2;

                num += d1 * d2;
                den1 += d1 * d1;
                den2 += d2 * d2;
            }
        }

        double denom = Math.Sqrt(den1 * den2);
        if (denom < 1e-9)
            return 0.0f;

        return (float)(num / denom);
    }

    /// <summary>
    /// Creates a neutral 580-byte BGRA chunk (B=127, G=127, R=127, A=255).
    /// </summary>
    public static byte[] CreateNeutralChunkBytes()
    {
        byte[] bytes = new byte[BytesPerChunk];
        for (int i = 0; i < VerticesPerChunk; i++)
        {
            int off = i * 4;
            bytes[off + 0] = NeutralChannelValue;
            bytes[off + 1] = NeutralChannelValue;
            bytes[off + 2] = NeutralChannelValue;
            bytes[off + 3] = 255;
        }

        return bytes;
    }

    private static float[] ExtractVertexLuminance(byte[] chunkData)
    {
        var luma = new float[VerticesPerChunk];
        for (int i = 0; i < VerticesPerChunk; i++)
        {
            int off = i * 4;
            byte b = chunkData[off + 0];
            byte g = chunkData[off + 1];
            byte r = chunkData[off + 2];
            luma[i] = ((0.299f * r) + (0.587f * g) + (0.114f * b)) / 255.0f;
        }

        return luma;
    }

    private static float SampleChunkLuminance(float[] vertexLuma, float localX, float localY)
    {
        float gridX = localX * 8.0f;
        float gridY = localY * 8.0f;

        int ix = Math.Clamp((int)gridX, 0, 7);
        int iy = Math.Clamp((int)gridY, 0, 7);
        float dx = Math.Clamp(gridX - ix, 0.0f, 1.0f);
        float dy = Math.Clamp(gridY - iy, 0.0f, 1.0f);

        float topLeft = vertexLuma[(iy * 9) + ix];
        float topRight = vertexLuma[(iy * 9) + ix + 1];
        float bottomLeft = vertexLuma[((iy + 1) * 9) + ix];
        float bottomRight = vertexLuma[((iy + 1) * 9) + ix + 1];
        float center = vertexLuma[81 + (iy * 8) + ix];

        if (dy < dx && dy < 1.0f - dx)
            return (topLeft * (1.0f - dx - dy)) + (topRight * (dx - dy)) + (center * (2.0f * dy));

        if (dy > dx && dy > 1.0f - dx)
            return (bottomLeft * (dy - dx)) + (bottomRight * (dx + dy - 1.0f)) + (center * (2.0f * (1.0f - dy)));

        if (dx < dy && dx < 1.0f - dy)
            return (topLeft * (1.0f - dx - dy)) + (bottomLeft * (dy - dx)) + (center * (2.0f * dx));

        return (topRight * (dx - dy)) + (bottomRight * (dy + dx - 1.0f)) + (center * (2.0f * (1.0f - dx)));
    }

    private static float SampleBilinear(float[,] grid, float x, float y, int width, int height)
    {
        int x0 = Math.Clamp((int)MathF.Floor(x), 0, width - 1);
        int y0 = Math.Clamp((int)MathF.Floor(y), 0, height - 1);
        int x1 = Math.Clamp(x0 + 1, 0, width - 1);
        int y1 = Math.Clamp(y0 + 1, 0, height - 1);

        float fx = x - x0;
        float fy = y - y0;

        float v00 = grid[y0, x0];
        float v10 = grid[y0, x1];
        float v01 = grid[y1, x0];
        float v11 = grid[y1, x1];

        float top = (v00 * (1.0f - fx)) + (v10 * fx);
        float bottom = (v01 * (1.0f - fx)) + (v11 * fx);

        return (top * (1.0f - fy)) + (bottom * fy);
    }
}
