using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MdxViewer.Export;

public static class TerrainMccvIo
{
    public const int TileMccvImageSize = 145;
    private const int TileChunks = 16;
    private const int VerticesPerChunk = 145;
    private const byte NeutralChannelValue = 127;

    public static Image<Rgba32> BuildTileImage(IReadOnlyList<Terrain.TerrainChunkData> chunks)
    {
        var chunkColors = new Dictionary<int, byte[]>(TileChunks * TileChunks);
        var neutralChunk = CreateNeutralChunkColors();
        foreach (var chunk in chunks)
        {
            if ((uint)chunk.ChunkX >= TileChunks || (uint)chunk.ChunkY >= TileChunks)
                continue;

            int chunkIndex = chunk.ChunkY * TileChunks + chunk.ChunkX;
            chunkColors[chunkIndex] = NormalizeChunkColors(chunk.MccvColors);
        }

        var image = new Image<Rgba32>(TileMccvImageSize, TileMccvImageSize);
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < TileMccvImageSize; y++)
            {
                float v = y / (float)(TileMccvImageSize - 1);
                float chunkY = v * TileChunks;
                int chunkIy = Math.Clamp((int)chunkY, 0, TileChunks - 1);
                float localY = Math.Clamp(chunkY - chunkIy, 0f, 1f);
                var row = accessor.GetRowSpan(y);

                for (int x = 0; x < TileMccvImageSize; x++)
                {
                    float u = x / (float)(TileMccvImageSize - 1);
                    float chunkX = u * TileChunks;
                    int chunkIx = Math.Clamp((int)chunkX, 0, TileChunks - 1);
                    float localX = Math.Clamp(chunkX - chunkIx, 0f, 1f);
                    int chunkIndex = chunkIy * TileChunks + chunkIx;

                    byte[] chunkData = chunkColors.TryGetValue(chunkIndex, out var data)
                        ? data
                        : neutralChunk;

                    row[x] = SampleChunkImage(chunkData, localX, localY);
                }
            }
        });

        return image;
    }

    public static List<Terrain.TerrainChunkData> ApplyTileImageToChunks(
        IReadOnlyList<Terrain.TerrainChunkData> chunks,
        Image<Rgba32> image)
    {
        if (image.Width <= 0 || image.Height <= 0)
            throw new ArgumentException("Image must have non-zero dimensions.", nameof(image));

        var result = new List<Terrain.TerrainChunkData>(chunks.Count);
        foreach (var chunk in chunks)
        {
            result.Add(new Terrain.TerrainChunkData
            {
                TileX = chunk.TileX,
                TileY = chunk.TileY,
                ChunkX = chunk.ChunkX,
                ChunkY = chunk.ChunkY,
                Heights = chunk.Heights,
                Normals = chunk.Normals,
                HoleMask = chunk.HoleMask,
                Layers = chunk.Layers,
                AlphaMaps = chunk.AlphaMaps,
                ShadowMap = chunk.ShadowMap,
                MccvColors = SampleChunkColors(image, chunk.ChunkX, chunk.ChunkY),
                Liquid = chunk.Liquid,
                WorldPosition = chunk.WorldPosition,
                AreaId = chunk.AreaId,
                McnkFlags = chunk.McnkFlags,
            });
        }

        return result;
    }

    private static byte[] NormalizeChunkColors(byte[]? raw)
    {
        if (raw != null && raw.Length >= VerticesPerChunk * 4)
            return raw;

        return CreateNeutralChunkColors();
    }

    private static byte[] CreateNeutralChunkColors()
    {
        var data = new byte[VerticesPerChunk * 4];
        for (int index = 0; index < VerticesPerChunk; index++)
        {
            int offset = index * 4;
            data[offset + 0] = NeutralChannelValue;
            data[offset + 1] = NeutralChannelValue;
            data[offset + 2] = NeutralChannelValue;
            data[offset + 3] = NeutralChannelValue;
        }

        return data;
    }

    private static Rgba32 SampleChunkImage(byte[] chunkData, float localX, float localY)
    {
        float gridX = localX * 8f;
        float gridY = localY * 8f;

        int ix = Math.Clamp((int)gridX, 0, 7);
        int iy = Math.Clamp((int)gridY, 0, 7);
        float dx = Math.Clamp(gridX - ix, 0f, 1f);
        float dy = Math.Clamp(gridY - iy, 0f, 1f);

        var topLeft = GetVertex(chunkData, iy * 9 + ix);
        var topRight = GetVertex(chunkData, iy * 9 + ix + 1);
        var bottomLeft = GetVertex(chunkData, (iy + 1) * 9 + ix);
        var bottomRight = GetVertex(chunkData, (iy + 1) * 9 + ix + 1);
        var center = GetVertex(chunkData, 81 + iy * 8 + ix);

        (float r, float g, float b, float a) blended;
        if (dy < dx && dy < 1.0f - dx)
        {
            float wTopLeft = 1 - dx - dy;
            float wTopRight = dx - dy;
            float wCenter = 2 * dy;
            blended = Combine(topLeft, topRight, center, wTopLeft, wTopRight, wCenter);
        }
        else if (dy > dx && dy > 1.0f - dx)
        {
            float wBottomLeft = dy - dx;
            float wBottomRight = dx + dy - 1;
            float wCenter = 2 * (1 - dy);
            blended = Combine(bottomLeft, bottomRight, center, wBottomLeft, wBottomRight, wCenter);
        }
        else if (dx < dy && dx < 1.0f - dy)
        {
            float wTopLeft = 1 - dx - dy;
            float wBottomLeft = dy - dx;
            float wCenter = 2 * dx;
            blended = Combine(topLeft, bottomLeft, center, wTopLeft, wBottomLeft, wCenter);
        }
        else
        {
            float wTopRight = dx - dy;
            float wBottomRight = dy + dx - 1;
            float wCenter = 2 * (1 - dx);
            blended = Combine(topRight, bottomRight, center, wTopRight, wBottomRight, wCenter);
        }

        return new Rgba32(
            ToByte(blended.r),
            ToByte(blended.g),
            ToByte(blended.b),
            ToByte(blended.a));
    }

    private static (float r, float g, float b, float a) Combine(
        Rgba32 colorA,
        Rgba32 colorB,
        Rgba32 colorC,
        float weightA,
        float weightB,
        float weightC)
    {
        return (
            Math.Clamp(colorA.R * weightA + colorB.R * weightB + colorC.R * weightC, 0f, 255f),
            Math.Clamp(colorA.G * weightA + colorB.G * weightB + colorC.G * weightC, 0f, 255f),
            Math.Clamp(colorA.B * weightA + colorB.B * weightB + colorC.B * weightC, 0f, 255f),
            Math.Clamp(colorA.A * weightA + colorB.A * weightB + colorC.A * weightC, 0f, 255f));
    }

    private static Rgba32 GetVertex(byte[] chunkData, int vertexIndex)
    {
        int offset = vertexIndex * 4;
        return new Rgba32(
            chunkData[offset + 0],
            chunkData[offset + 1],
            chunkData[offset + 2],
            chunkData[offset + 3]);
    }

    private static byte[] SampleChunkColors(Image<Rgba32> image, int chunkX, int chunkY)
    {
        var data = new byte[VerticesPerChunk * 4];
        int vertexIndex = 0;

        for (int row = 0; row < 9; row++)
        {
            for (int col = 0; col < 9; col++)
            {
                var color = SampleTileImage(image, chunkX + (col / 8f), chunkY + (row / 8f));
                WriteVertex(data, vertexIndex++, color);
            }

            if (row >= 8)
                continue;

            for (int col = 0; col < 8; col++)
            {
                var color = SampleTileImage(image, chunkX + ((col + 0.5f) / 8f), chunkY + ((row + 0.5f) / 8f));
                WriteVertex(data, vertexIndex++, color);
            }
        }

        return data;
    }

    private static Rgba32 SampleTileImage(Image<Rgba32> image, float tileChunkX, float tileChunkY)
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

    private static void WriteVertex(byte[] data, int vertexIndex, Rgba32 color)
    {
        int offset = vertexIndex * 4;
        data[offset + 0] = color.R;
        data[offset + 1] = color.G;
        data[offset + 2] = color.B;
        data[offset + 3] = color.A;
    }

    private static byte ToByte(float value) => (byte)Math.Clamp((int)MathF.Round(value), 0, 255);
}