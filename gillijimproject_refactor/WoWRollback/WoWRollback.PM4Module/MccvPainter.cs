using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWRollback.PM4Module;

/// <summary>
/// Paints MCCV (vertex color) chunks from minimap images.
/// 
/// MCCV format: 145 vertices per MCNK (9x9 outer + 8x8 inner grid)
/// Each vertex: 4 bytes BGRA (Blue, Green, Red, Alpha)
/// 
/// Minimap tiles are 256x256 pixels per ADT tile.
/// Each MCNK covers 16x16 pixels of the minimap (256/16 = 16).
/// </summary>
public static class MccvPainter
{
    private const int VerticesPerChunk = 145; // 9*9 + 8*8
    private const int BytesPerVertex = 4;     // BGRA
    private const int McnkPerRow = 16;        // 16x16 MCNKs per ADT
    private const int MinimapSize = 256;      // 256x256 pixels per minimap tile
    private const int PixelsPerMcnk = MinimapSize / McnkPerRow; // 16 pixels

    /// <summary>
    /// Generate MCCV data for a single MCNK from a minimap image.
    /// </summary>
    /// <param name="image">256x256 minimap image for the ADT tile</param>
    /// <param name="mcnkX">MCNK X index (0-15)</param>
    /// <param name="mcnkY">MCNK Y index (0-15)</param>
    /// <returns>580 bytes of MCCV data (145 vertices * 4 bytes)</returns>
    public static byte[] GenerateMccvFromImage(Image<Rgba32> image, int mcnkX, int mcnkY)
    {
        var data = new byte[VerticesPerChunk * BytesPerVertex];
        
        // Calculate pixel region for this MCNK
        int basePixelX = mcnkX * PixelsPerMcnk;
        int basePixelY = mcnkY * PixelsPerMcnk;
        
        int vertexIndex = 0;
        
        // Outer grid: 9x9 vertices (corners and edges)
        for (int row = 0; row < 9; row++)
        {
            for (int col = 0; col < 9; col++)
            {
                // Map vertex to pixel position
                float pixelX = basePixelX + (col / 8.0f) * (PixelsPerMcnk - 1);
                float pixelY = basePixelY + (row / 8.0f) * (PixelsPerMcnk - 1);
                
                var color = SamplePixel(image, pixelX, pixelY);
                WriteVertex(data, vertexIndex++, color);
            }
        }
        
        // Inner grid: 8x8 vertices (centers)
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                // Inner vertices are offset by 0.5 cell
                float pixelX = basePixelX + ((col + 0.5f) / 8.0f) * (PixelsPerMcnk - 1);
                float pixelY = basePixelY + ((row + 0.5f) / 8.0f) * (PixelsPerMcnk - 1);
                
                var color = SamplePixel(image, pixelX, pixelY);
                WriteVertex(data, vertexIndex++, color);
            }
        }
        
        return data;
    }

    /// <summary>
    /// Generate MCCV data for all 256 MCNKs from a minimap image.
    /// </summary>
    /// <param name="imagePath">Path to 256x256 minimap image (PNG, BLP, etc.)</param>
    /// <returns>Array of 256 MCCV byte arrays, indexed by MCNK index (Y*16+X)</returns>
    public static byte[][] GenerateAllMccvFromImage(string imagePath)
    {
        using var image = Image.Load<Rgba32>(imagePath);
        return GenerateAllMccvFromImage(image);
    }

    /// <summary>
    /// Generate MCCV data for all 256 MCNKs from a minimap image.
    /// </summary>
    public static byte[][] GenerateAllMccvFromImage(Image<Rgba32> image)
    {
        // Resize if not 256x256
        if (image.Width != MinimapSize || image.Height != MinimapSize)
        {
            image.Mutate(x => x.Resize(MinimapSize, MinimapSize));
        }
        
        var result = new byte[256][];
        
        for (int y = 0; y < McnkPerRow; y++)
        {
            for (int x = 0; x < McnkPerRow; x++)
            {
                int mcnkIndex = y * McnkPerRow + x;
                result[mcnkIndex] = GenerateMccvFromImage(image, x, y);
            }
        }
        
        return result;
    }

    /// <summary>
    /// Generate neutral gray MCCV data (no tinting).
    /// </summary>
    public static byte[] GenerateNeutralMccv()
    {
        var data = new byte[VerticesPerChunk * BytesPerVertex];
        for (int i = 0; i < VerticesPerChunk; i++)
        {
            int offset = i * BytesPerVertex;
            data[offset + 0] = 0x7F; // B
            data[offset + 1] = 0x7F; // G
            data[offset + 2] = 0x7F; // R
            data[offset + 3] = 0x00; // A (unused)
        }
        return data;
    }

    private static Rgba32 SamplePixel(Image<Rgba32> image, float x, float y)
    {
        // Clamp to image bounds
        int px = Math.Clamp((int)x, 0, image.Width - 1);
        int py = Math.Clamp((int)y, 0, image.Height - 1);
        return image[px, py];
    }

    private static void WriteVertex(byte[] data, int index, Rgba32 color)
    {
        int offset = index * BytesPerVertex;
        // MCCV format is BGRA
        data[offset + 0] = color.B;
        data[offset + 1] = color.G;
        data[offset + 2] = color.R;
        data[offset + 3] = 0x00; // Alpha unused in MCCV
    }
}
