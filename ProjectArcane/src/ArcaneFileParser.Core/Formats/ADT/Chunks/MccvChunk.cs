using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Vertex Colors subchunk containing vertex color data.
/// </summary>
public class MccvChunk : ChunkBase
{
    public override string ChunkId => "MCCV";

    /// <summary>
    /// The number of vertices in each dimension of the color grid.
    /// </summary>
    public const int COLOR_GRID_DIM = 17;

    /// <summary>
    /// The total number of vertices in the color grid (17x17).
    /// </summary>
    public const int TOTAL_VERTICES = COLOR_GRID_DIM * COLOR_GRID_DIM;

    /// <summary>
    /// Gets the vertex colors array (17x17 grid).
    /// </summary>
    public ColorBGRA[] Colors { get; } = new ColorBGRA[TOTAL_VERTICES];

    public override void Parse(BinaryReader reader, uint size)
    {
        // Each color is 4 bytes (BGRA)
        var colorCount = Math.Min(size / 4, TOTAL_VERTICES);

        // Read all vertex colors
        for (int i = 0; i < colorCount; i++)
        {
            Colors[i] = reader.ReadColorBGRA();
        }

        // Clear any remaining colors if the chunk was smaller than expected
        for (int i = (int)colorCount; i < TOTAL_VERTICES; i++)
        {
            Colors[i] = new ColorBGRA(255, 255, 255, 255); // Default to white
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write all vertex colors
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            writer.WriteColorBGRA(Colors[i]);
        }
    }

    /// <summary>
    /// Gets a vertex color at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-16).</param>
    /// <param name="y">Y coordinate (0-16).</param>
    /// <returns>The vertex color if coordinates are valid, white otherwise.</returns>
    public ColorBGRA GetColor(int x, int y)
    {
        if (x < 0 || x >= COLOR_GRID_DIM || y < 0 || y >= COLOR_GRID_DIM)
            return new ColorBGRA(255, 255, 255, 255); // Return white for invalid coordinates

        return Colors[y * COLOR_GRID_DIM + x];
    }

    /// <summary>
    /// Sets a vertex color at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-16).</param>
    /// <param name="y">Y coordinate (0-16).</param>
    /// <param name="color">The color to set.</param>
    public void SetColor(int x, int y, ColorBGRA color)
    {
        if (x < 0 || x >= COLOR_GRID_DIM || y < 0 || y >= COLOR_GRID_DIM)
            return;

        Colors[y * COLOR_GRID_DIM + x] = color;
    }

    /// <summary>
    /// Sets a vertex color at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-16).</param>
    /// <param name="y">Y coordinate (0-16).</param>
    /// <param name="r">Red component (0-255).</param>
    /// <param name="g">Green component (0-255).</param>
    /// <param name="b">Blue component (0-255).</param>
    /// <param name="a">Alpha component (0-255).</param>
    public void SetColor(int x, int y, byte r, byte g, byte b, byte a = 255)
    {
        SetColor(x, y, new ColorBGRA(r, g, b, a));
    }

    /// <summary>
    /// Fills all vertex colors with a single color.
    /// </summary>
    /// <param name="color">The color to fill with.</param>
    public void Fill(ColorBGRA color)
    {
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            Colors[i] = color;
        }
    }

    /// <summary>
    /// Gets the average color of the chunk.
    /// </summary>
    /// <returns>The average color.</returns>
    public ColorBGRA GetAverageColor()
    {
        long totalR = 0, totalG = 0, totalB = 0, totalA = 0;

        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            var color = Colors[i];
            totalR += color.R;
            totalG += color.G;
            totalB += color.B;
            totalA += color.A;
        }

        return new ColorBGRA(
            (byte)(totalR / TOTAL_VERTICES),
            (byte)(totalG / TOTAL_VERTICES),
            (byte)(totalB / TOTAL_VERTICES),
            (byte)(totalA / TOTAL_VERTICES)
        );
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Vertex Colors (17x17 grid):");

        var avgColor = GetAverageColor();
        builder.AppendLine($"Average Color: R:{avgColor.R} G:{avgColor.G} B:{avgColor.B} A:{avgColor.A}");

        // Sample a few points to show variation
        for (int y = 0; y < COLOR_GRID_DIM; y += 4)
        {
            for (int x = 0; x < COLOR_GRID_DIM; x += 4)
            {
                var color = GetColor(x, y);
                if (color.R != 255 || color.G != 255 || color.B != 255 || color.A != 255)
                {
                    builder.AppendLine($"\nPosition ({x}, {y}):");
                    builder.AppendLine($"  R:{color.R} G:{color.G} B:{color.B} A:{color.A}");
                }
            }
        }

        return builder.ToString();
    }
} 