using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Rectangle = SixLabors.ImageSharp.Rectangle;

namespace MdxViewer.Export;

public static class TerrainImageIo
{
    public const int ChunkAlphaSize = 64;
    public const int TileChunks = 16;
    public const int TileAlphaAtlasSize = ChunkAlphaSize * TileChunks;

    // Packed RGBA convention:
    // R = alpha1, G = alpha2, B = alpha3, A = shadow

    public static Image<Rgba32> BuildAlphaAtlasFromChunks(IReadOnlyList<Terrain.TerrainChunkData> chunks)
    {
        var image = new Image<Rgba32>(TileAlphaAtlasSize, TileAlphaAtlasSize);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    row[x] = new Rgba32(255, 255, 255, 0);
                }
            }
        });

        foreach (var chunk in chunks)
        {
            int cx = chunk.ChunkX;
            int cy = chunk.ChunkY;
            if ((uint)cx >= TileChunks || (uint)cy >= TileChunks)
            {
                continue;
            }

            chunk.AlphaMaps.TryGetValue(1, out var alpha1);
            chunk.AlphaMaps.TryGetValue(2, out var alpha2);
            chunk.AlphaMaps.TryGetValue(3, out var alpha3);
            var shadow = chunk.ShadowMap;

            int dstX0 = cx * ChunkAlphaSize;
            int dstY0 = cy * ChunkAlphaSize;

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < ChunkAlphaSize; y++)
                {
                    var row = accessor.GetRowSpan(dstY0 + y);
                    for (int x = 0; x < ChunkAlphaSize; x++)
                    {
                        int src = y * ChunkAlphaSize + x;
                        byte r = alpha1 != null && alpha1.Length >= 4096 ? alpha1[src] : (byte)255;
                        byte g = alpha2 != null && alpha2.Length >= 4096 ? alpha2[src] : (byte)255;
                        byte b = alpha3 != null && alpha3.Length >= 4096 ? alpha3[src] : (byte)255;
                        byte a = shadow != null && shadow.Length >= 4096 ? shadow[src] : (byte)0;
                        row[dstX0 + x] = new Rgba32(r, g, b, a);
                    }
                }
            });
        }

        return image;
    }

    public static Dictionary<(int chunkX, int chunkY), Image<Rgba32>> BuildAlphaChunkImagesFromAtlas(Image<Rgba32> atlas)
    {
        if (atlas.Width != TileAlphaAtlasSize || atlas.Height != TileAlphaAtlasSize)
        {
            throw new ArgumentException($"Expected atlas {TileAlphaAtlasSize}x{TileAlphaAtlasSize}.");
        }

        var result = new Dictionary<(int, int), Image<Rgba32>>(TileChunks * TileChunks);
        for (int cy = 0; cy < TileChunks; cy++)
        {
            for (int cx = 0; cx < TileChunks; cx++)
            {
                var chunk = atlas.Clone(ctx => ctx.Crop(new Rectangle(cx * ChunkAlphaSize, cy * ChunkAlphaSize, ChunkAlphaSize, ChunkAlphaSize)));
                result[(cx, cy)] = chunk;
            }
        }

        return result;
    }

    public static byte[] DecodeAlphaShadowArrayFromAtlas(Image<Rgba32> atlas)
    {
        if (atlas.Width != TileAlphaAtlasSize || atlas.Height != TileAlphaAtlasSize)
        {
            throw new ArgumentException($"Expected atlas {TileAlphaAtlasSize}x{TileAlphaAtlasSize}.");
        }

        var alphaShadow = new byte[ChunkAlphaSize * ChunkAlphaSize * TileChunks * TileChunks * 4];

        for (int i = 0; i < alphaShadow.Length; i += 4)
        {
            alphaShadow[i + 0] = 255;
            alphaShadow[i + 1] = 255;
            alphaShadow[i + 2] = 255;
            alphaShadow[i + 3] = 0;
        }

        atlas.ProcessPixelRows(accessor =>
        {
            for (int cy = 0; cy < TileChunks; cy++)
            {
                for (int cx = 0; cx < TileChunks; cx++)
                {
                    int slice = cy * TileChunks + cx;
                    int sliceBase = slice * ChunkAlphaSize * ChunkAlphaSize * 4;
                    int srcX0 = cx * ChunkAlphaSize;
                    int srcY0 = cy * ChunkAlphaSize;

                    for (int y = 0; y < ChunkAlphaSize; y++)
                    {
                        var row = accessor.GetRowSpan(srcY0 + y);
                        for (int x = 0; x < ChunkAlphaSize; x++)
                        {
                            var px = row[srcX0 + x];
                            int dst = (y * ChunkAlphaSize + x) * 4;
                            alphaShadow[sliceBase + dst + 0] = px.R;
                            alphaShadow[sliceBase + dst + 1] = px.G;
                            alphaShadow[sliceBase + dst + 2] = px.B;
                            alphaShadow[sliceBase + dst + 3] = px.A;
                        }
                    }
                }
            }
        });

        return alphaShadow;
    }
}