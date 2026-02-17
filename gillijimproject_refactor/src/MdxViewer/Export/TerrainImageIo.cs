using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Rectangle = SixLabors.ImageSharp.Rectangle;

namespace MdxViewer.Export;

public static class TerrainImageIo
{
    public const int ChunkAlphaSize = 64;
    public const int TileChunks = 16;
    public const int TileAlphaAtlasSize = ChunkAlphaSize * TileChunks; // 1024

    // Packed RGBA convention:
    // R = alpha1, G = alpha2, B = alpha3, A = shadow

    private static int EdgeFixedIndex64(int x, int y)
    {
        if (x >= 63) x = 62;
        if (y >= 63) y = 62;
        return y * 64 + x;
    }

    public static Image<Rgba32> BuildAlphaAtlasFromChunks(IReadOnlyList<Terrain.TerrainChunkData> chunks)
    {
        var image = new Image<Rgba32>(TileAlphaAtlasSize, TileAlphaAtlasSize);

        // Default: overlay alpha 255, shadow 0.
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                    row[x] = new Rgba32(255, 255, 255, 0);
            }
        });

        foreach (var chunk in chunks)
        {
            int cx = chunk.ChunkX;
            int cy = chunk.ChunkY;
            if ((uint)cx >= 16u || (uint)cy >= 16u) continue;

            chunk.AlphaMaps.TryGetValue(1, out var a1);
            chunk.AlphaMaps.TryGetValue(2, out var a2);
            chunk.AlphaMaps.TryGetValue(3, out var a3);
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
                        int src = EdgeFixedIndex64(x, y);
                        byte r = (a1 != null && a1.Length >= 4096) ? a1[src] : (byte)255;
                        byte g = (a2 != null && a2.Length >= 4096) ? a2[src] : (byte)255;
                        byte b = (a3 != null && a3.Length >= 4096) ? a3[src] : (byte)255;
                        byte a = (shadow != null && shadow.Length >= 4096) ? shadow[src] : (byte)0;
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
            throw new ArgumentException($"Expected atlas {TileAlphaAtlasSize}x{TileAlphaAtlasSize}.");

        var result = new Dictionary<(int, int), Image<Rgba32>>(256);
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                var chunk = atlas.Clone(ctx => ctx.Crop(new Rectangle(cx * 64, cy * 64, 64, 64)));
                result[(cx, cy)] = chunk;
            }
        }
        return result;
    }

    public static byte[] DecodeAlphaShadowArrayFromAtlas(Image<Rgba32> atlas)
    {
        if (atlas.Width != TileAlphaAtlasSize || atlas.Height != TileAlphaAtlasSize)
            throw new ArgumentException($"Expected atlas {TileAlphaAtlasSize}x{TileAlphaAtlasSize}.");

        // 64*64*256*4
        var alphaShadow = new byte[64 * 64 * 256 * 4];

        // Default: overlay alpha 255, shadow 0.
        for (int i = 0; i < alphaShadow.Length; i += 4)
        {
            alphaShadow[i + 0] = 255;
            alphaShadow[i + 1] = 255;
            alphaShadow[i + 2] = 255;
            alphaShadow[i + 3] = 0;
        }

        atlas.ProcessPixelRows(accessor =>
        {
            for (int cy = 0; cy < 16; cy++)
            {
                for (int cx = 0; cx < 16; cx++)
                {
                    int slice = cy * 16 + cx;
                    int sliceBase = slice * 64 * 64 * 4;

                    int srcX0 = cx * 64;
                    int srcY0 = cy * 64;

                    for (int y = 0; y < 64; y++)
                    {
                        var row = accessor.GetRowSpan(srcY0 + y);
                        for (int x = 0; x < 64; x++)
                        {
                            // Apply Noggit-style edge fix even on import (duplicate 63 from 62)
                            int fx = x >= 63 ? 62 : x;
                            int fy = y >= 63 ? 62 : y;

                            var px = row[srcX0 + fx];
                            int dst = (y * 64 + x) * 4;
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
