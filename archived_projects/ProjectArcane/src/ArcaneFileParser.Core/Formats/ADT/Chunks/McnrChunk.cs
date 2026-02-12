using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Normals subchunk containing normal vectors for the terrain vertices.
/// </summary>
public class McnrChunk : ChunkBase
{
    public override string ChunkId => "MCNR";

    /// <summary>
    /// The number of vertices in each dimension of the normal map.
    /// </summary>
    public const int CHUNK_VERTEX_SIZE = 9;

    /// <summary>
    /// The total number of vertices in the normal map (9x9 + 8x8).
    /// </summary>
    public const int TOTAL_VERTICES = 145;

    /// <summary>
    /// Gets the normal vectors for the terrain vertices.
    /// First 81 entries are for the 9x9 grid, followed by 64 entries for the 8x8 grid.
    /// </summary>
    public Vector3F[] Normals { get; } = new Vector3F[TOTAL_VERTICES];

    public override void Parse(BinaryReader reader, uint size)
    {
        // Read all normal vectors
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            // Normals are stored as signed bytes (-127 to 127) and need to be normalized
            float x = reader.ReadSByte() / 127f;
            float y = reader.ReadSByte() / 127f;
            float z = reader.ReadSByte() / 127f;

            Normals[i] = new Vector3F(x, y, z);
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write all normal vectors
        for (int i = 0; i < TOTAL_VERTICES; i++)
        {
            // Convert normalized vectors back to signed bytes
            writer.Write((sbyte)(Normals[i].X * 127f));
            writer.Write((sbyte)(Normals[i].Y * 127f));
            writer.Write((sbyte)(Normals[i].Z * 127f));
        }
    }

    /// <summary>
    /// Gets the normal vector at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="y">Y coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="inner">True to access the 9x9 inner grid, false for the 8x8 outer grid.</param>
    /// <returns>The normal vector at the specified coordinates.</returns>
    public Vector3F GetNormal(int x, int y, bool inner = true)
    {
        if (inner)
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE || y < 0 || y >= CHUNK_VERTEX_SIZE)
                throw new ArgumentOutOfRangeException($"Inner grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 1}");

            return Normals[y * CHUNK_VERTEX_SIZE + x];
        }
        else
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE - 1 || y < 0 || y >= CHUNK_VERTEX_SIZE - 1)
                throw new ArgumentOutOfRangeException($"Outer grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 2}");

            return Normals[81 + y * (CHUNK_VERTEX_SIZE - 1) + x];
        }
    }

    /// <summary>
    /// Sets the normal vector at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="y">Y coordinate (0-8 for inner vertices, 0-7 for outer vertices).</param>
    /// <param name="normal">The normal vector to set.</param>
    /// <param name="inner">True to access the 9x9 inner grid, false for the 8x8 outer grid.</param>
    public void SetNormal(int x, int y, Vector3F normal, bool inner = true)
    {
        if (inner)
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE || y < 0 || y >= CHUNK_VERTEX_SIZE)
                throw new ArgumentOutOfRangeException($"Inner grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 1}");

            Normals[y * CHUNK_VERTEX_SIZE + x] = normal;
        }
        else
        {
            if (x < 0 || x >= CHUNK_VERTEX_SIZE - 1 || y < 0 || y >= CHUNK_VERTEX_SIZE - 1)
                throw new ArgumentOutOfRangeException($"Outer grid coordinates must be between 0 and {CHUNK_VERTEX_SIZE - 2}");

            Normals[81 + y * (CHUNK_VERTEX_SIZE - 1) + x] = normal;
        }
    }

    /// <summary>
    /// Recalculates normal vectors based on height values.
    /// </summary>
    /// <param name="mcvt">The MCVT chunk containing height values.</param>
    public void RecalculateNormals(McvtChunk mcvt)
    {
        // Calculate normals for inner grid
        for (int y = 0; y < CHUNK_VERTEX_SIZE; y++)
        {
            for (int x = 0; x < CHUNK_VERTEX_SIZE; x++)
            {
                Vector3F normal = CalculateNormalAt(x, y, mcvt);
                SetNormal(x, y, normal);
            }
        }

        // Calculate normals for outer grid
        for (int y = 0; y < CHUNK_VERTEX_SIZE - 1; y++)
        {
            for (int x = 0; x < CHUNK_VERTEX_SIZE - 1; x++)
            {
                Vector3F normal = CalculateNormalAt(x, y, mcvt, false);
                SetNormal(x, y, normal, false);
            }
        }
    }

    private Vector3F CalculateNormalAt(int x, int y, McvtChunk mcvt, bool inner = true)
    {
        // Get heights of surrounding vertices
        float center = mcvt.GetHeight(x, y, inner);
        float left = x > 0 ? mcvt.GetHeight(x - 1, y, inner) : center;
        float right = x < (inner ? CHUNK_VERTEX_SIZE - 1 : CHUNK_VERTEX_SIZE - 2) ? mcvt.GetHeight(x + 1, y, inner) : center;
        float top = y > 0 ? mcvt.GetHeight(x, y - 1, inner) : center;
        float bottom = y < (inner ? CHUNK_VERTEX_SIZE - 1 : CHUNK_VERTEX_SIZE - 2) ? mcvt.GetHeight(x, y + 1, inner) : center;

        // Calculate normal using cross products of tangent vectors
        Vector3F tangentX = new Vector3F(2.0f, 0, right - left);
        Vector3F tangentY = new Vector3F(0, 2.0f, bottom - top);

        Vector3F normal = Vector3F.Cross(tangentX, tangentY);
        normal.Normalize();

        return normal;
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Normal Vectors:");

        builder.AppendLine("\nInner Grid (9x9):");
        for (int y = 0; y < CHUNK_VERTEX_SIZE; y++)
        {
            for (int x = 0; x < CHUNK_VERTEX_SIZE; x++)
            {
                var normal = GetNormal(x, y);
                builder.Append($"({normal.X:F2}, {normal.Y:F2}, {normal.Z:F2}) ");
            }
            builder.AppendLine();
        }

        builder.AppendLine("\nOuter Grid (8x8):");
        for (int y = 0; y < CHUNK_VERTEX_SIZE - 1; y++)
        {
            for (int x = 0; x < CHUNK_VERTEX_SIZE - 1; x++)
            {
                var normal = GetNormal(x, y, false);
                builder.Append($"({normal.X:F2}, {normal.Y:F2}, {normal.Z:F2}) ");
            }
            builder.AppendLine();
        }

        return builder.ToString();
    }
} 