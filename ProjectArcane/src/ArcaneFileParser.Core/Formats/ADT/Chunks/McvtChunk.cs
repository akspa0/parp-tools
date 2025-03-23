using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Vertex Heights subchunk containing height values for the terrain.
/// </summary>
public class McvtChunk : ChunkBase
{
    public override string ChunkId => "MCVT";

    /// <summary>
    /// The number of vertices in each dimension of the height map.
    /// </summary>
    public const int CHUNK_VERTEX_SIZE = 9;

    /// <summary>
    /// The total number of vertices in the height map (9x9 + 8x8).
    /// </summary>
    public const int TOTAL_VERTICES = 145;

    /// <summary>
    /// Gets the height values for the terrain vertices.
    /// First 81 entries are for the 9x9 grid, followed by 64 entries for the 8x8 grid.
    /// </summary>
    public float[] Heights { get; } = new float[TOTAL_VERTICES];

    /// <summary>
    /// Gets the minimum height value in the chunk.
    /// </summary>
    public float MinHeight { get; private set; } = float.MaxValue;

    /// <summary>
    /// Gets the maximum height value in the chunk.
    /// </summary>
    public float MaxHeight { get; private set; } = float.MinValue;

    public override void Parse(BinaryReader reader, uint size)
    {
        MinHeight = float.MaxValue;
        MaxHeight = float.MinValue;

        // Read all height values
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            Heights[i] = reader.ReadSingle();

            // Update min/max heights
            if (Heights[i] < MinHeight) MinHeight = Heights[i];
            if (Heights[i] > MaxHeight) MaxHeight = Heights[i];
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write all height values
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            writer.Write(Heights[i]);
        }
    }

    /// <summary>
    /// Gets the height value at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="y">Y coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="inner">True to access the 9x9 inner grid, false for the 8x8 outer grid.</param>
    /// <returns>The height value at the specified coordinates.</returns>
    public float GetHeight(int x, int y, bool inner = true)
    {
        if (inner)
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE || y < 0 || y >= CHUNK_VERTEX_SIZE)
                throw new ArgumentOutOfRangeException($"Inner grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 1}");

            return Heights[y * CHUNK_VERTEX_SIZE + x];
        }
        else
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE - 1 || y < 0 || y >= CHUNK_VERTEX_SIZE - 1)
                throw new ArgumentOutOfRangeException($"Outer grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 2}");

            return Heights[81 + y * (CHUNK_VERTEX_SIZE - 1) + x];
        }
    }

    /// <summary>
    /// Sets the height value at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="y">Y coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="height">The height value to set.</param>
    /// <param name="inner">True to access the 9x9 inner grid, false for the 8x8 outer grid.</param>
    public void SetHeight(int x, int y, float height, bool inner = true)
    {
        if (inner)
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE || y < 0 || y >= CHUNK_VERTEX_SIZE)
                throw new ArgumentOutOfRangeException($"Inner grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 1}");

            Heights[y * CHUNK_VERTEX_SIZE + x] = height;
        }
        else
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE - 1 || y < 0 || y >= CHUNK_VERTEX_SIZE - 1)
                throw new ArgumentOutOfRangeException($"Outer grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 2}");

            Heights[81 + y * (CHUNK_VERTEX_SIZE - 1) + x] = height;
        }

        // Update min/max heights
        if (height < MinHeight) MinHeight = height;
        if (height > MaxHeight) MaxHeight = height;
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Vertex Heights:");
        builder.AppendLine($"Min Height: {MinHeight:F2}");
        builder.AppendLine($"Max Height: {MaxHeight:F2}");
        builder.AppendLine($"Height Range: {MaxHeight - MinHeight:F2}");

        builder.AppendLine("\nInner Grid (9x9):");
        for (int y = 0; y < CHUNK_VERTEX_SIZE; y++)
        {
            for (int x = 0; x < CHUNK_VERTEX_SIZE; x++)
            {
                builder.Append($"{GetHeight(x, y):F2} ");
            }
            builder.AppendLine();
        }

        builder.AppendLine("\nOuter Grid (8x8):");
        for (int y = 0; y < CHUNK_VERTEX_SIZE - 1; y++)
        {
            for (int x = 0; x < CHUNK_VERTEX_SIZE - 1; x++)
            {
                builder.Append($"{GetHeight(x, y, false):F2} ");
            }
            builder.AppendLine();
        }

        return builder.ToString();
    }
} 