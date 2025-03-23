using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Liquid subchunk containing liquid data.
/// </summary>
public class MclqChunk : ChunkBase
{
    public override string ChunkId => "MCLQ";

    /// <summary>
    /// The number of vertices in each dimension of the liquid grid.
    /// </summary>
    public const int LIQUID_GRID_DIM = 9;

    /// <summary>
    /// The total number of vertices in the liquid grid (9x9).
    /// </summary>
    public const int TOTAL_VERTICES = LIQUID_GRID_DIM * LIQUID_GRID_DIM;

    /// <summary>
    /// Liquid vertex information.
    /// </summary>
    public struct LiquidVertex
    {
        public float Height;        // Liquid height
        public byte Depth;          // Liquid depth
        public byte Flow0;          // Flow direction 0
        public byte Flow1;          // Flow direction 1
        public byte Flags;          // Liquid flags
    }

    /// <summary>
    /// Gets the liquid vertices array (9x9 grid).
    /// </summary>
    public LiquidVertex[] Vertices { get; } = new LiquidVertex[TOTAL_VERTICES];

    /// <summary>
    /// Gets or sets the minimum liquid height.
    /// </summary>
    public float MinHeight { get; private set; } = float.MaxValue;

    /// <summary>
    /// Gets or sets the maximum liquid height.
    /// </summary>
    public float MaxHeight { get; private set; } = float.MinValue;

    public override void Parse(BinaryReader reader, uint size)
    {
        MinHeight = float.MaxValue;
        MaxHeight = float.MinValue;

        // Read all liquid vertices
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            var vertex = new LiquidVertex
            {
                Height = reader.ReadSingle(),
                Depth = reader.ReadByte(),
                Flow0 = reader.ReadByte(),
                Flow1 = reader.ReadByte(),
                Flags = reader.ReadByte()
            };

            Vertices[i] = vertex;

            // Update min/max heights
            if (vertex.Height < MinHeight) MinHeight = vertex.Height;
            if (vertex.Height > MaxHeight) MaxHeight = vertex.Height;
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write all liquid vertices
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            var vertex = Vertices[i];
            writer.Write(vertex.Height);
            writer.Write(vertex.Depth);
            writer.Write(vertex.Flow0);
            writer.Write(vertex.Flow1);
            writer.Write(vertex.Flags);
        }
    }

    /// <summary>
    /// Gets a liquid vertex at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8).</param>
    /// <param name="y">Y coordinate (0-8).</param>
    /// <returns>The liquid vertex if coordinates are valid, a default vertex otherwise.</returns>
    public LiquidVertex GetVertex(int x, int y)
    {
        if (x < 0 || x >= LIQUID_GRID_DIM || y < 0 || y >= LIQUID_GRID_DIM)
            return new LiquidVertex();

        return Vertices[y * LIQUID_GRID_DIM + x];
    }

    /// <summary>
    /// Sets a liquid vertex at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8).</param>
    /// <param name="y">Y coordinate (0-8).</param>
    /// <param name="vertex">The liquid vertex to set.</param>
    public void SetVertex(int x, int y, LiquidVertex vertex)
    {
        if (x < 0 || x >= LIQUID_GRID_DIM || y < 0 || y >= LIQUID_GRID_DIM)
            return;

        Vertices[y * LIQUID_GRID_DIM + x] = vertex;

        // Update min/max heights
        if (vertex.Height < MinHeight) MinHeight = vertex.Height;
        if (vertex.Height > MaxHeight) MaxHeight = vertex.Height;
    }

    /// <summary>
    /// Sets the liquid height at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8).</param>
    /// <param name="y">Y coordinate (0-8).</param>
    /// <param name="height">The height value to set.</param>
    public void SetHeight(int x, int y, float height)
    {
        if (x < 0 || x >= LIQUID_GRID_DIM || y < 0 || y >= LIQUID_GRID_DIM)
            return;

        var vertex = GetVertex(x, y);
        vertex.Height = height;
        SetVertex(x, y, vertex);
    }

    /// <summary>
    /// Sets the liquid flow at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8).</param>
    /// <param name="y">Y coordinate (0-8).</param>
    /// <param name="flow0">First flow direction.</param>
    /// <param name="flow1">Second flow direction.</param>
    public void SetFlow(int x, int y, byte flow0, byte flow1)
    {
        if (x < 0 || x >= LIQUID_GRID_DIM || y < 0 || y >= LIQUID_GRID_DIM)
            return;

        var vertex = GetVertex(x, y);
        vertex.Flow0 = flow0;
        vertex.Flow1 = flow1;
        SetVertex(x, y, vertex);
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Liquid Data (9x9 grid):");
        builder.AppendLine($"Height Range: {MinHeight:F2} - {MaxHeight:F2}");

        for (int y = 0; y < LIQUID_GRID_DIM; y++)
        {
            for (int x = 0; x < LIQUID_GRID_DIM; x++)
            {
                var vertex = GetVertex(x, y);
                if (vertex.Flags != 0)
                {
                    builder.AppendLine($"\nPosition ({x}, {y}):");
                    builder.AppendLine($"  Height: {vertex.Height:F2}");
                    builder.AppendLine($"  Depth: {vertex.Depth}");
                    builder.AppendLine($"  Flow: ({vertex.Flow0}, {vertex.Flow1})");
                    builder.AppendLine($"  Flags: 0x{vertex.Flags:X2}");
                }
            }
        }

        return builder.ToString();
    }
} 