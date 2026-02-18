using System.Numerics;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Builds GPU mesh data (VAO/VBO/EBO) from <see cref="TerrainChunkData"/>.
/// Each MCNK chunk has 145 vertices arranged in a 9-8-9-8... pattern (17 rows).
/// Outer rows have 9 vertices, inner rows have 8 center vertices.
/// Each 2×2 outer quad is split into 4 triangles via the center inner vertex → 256 triangles total.
/// </summary>
public class TerrainMeshBuilder
{
    private readonly GL _gl;

    public TerrainMeshBuilder(GL gl)
    {
        _gl = gl;
    }

    /// <summary>
    /// Build a GPU mesh for a single terrain chunk.
    /// Returns null if the chunk has no valid height data.
    /// </summary>
    public TerrainChunkMesh? BuildChunkMesh(TerrainChunkData chunk)
    {
        if (chunk.Heights.Length < 145) return null;

        // Build vertex data: position(3) + normal(3) + texcoord(2) + color(4) = 12 floats per vertex
        var vertices = new float[145 * 12];
        float cellSize = WoWConstants.ChunkSize / 16f; // size of one chunk
        float subCellSize = cellSize / 8f;             // size of one cell within chunk

        // Track AABB for frustum culling
        var bMin = new Vector3(float.MaxValue);
        var bMax = new Vector3(float.MinValue);

        for (int i = 0; i < 145; i++)
        {
            // Determine row and column in the interleaved layout
            GetVertexPosition(i, out int row, out int col, out bool isInner);

            float x, y;
            if (!isInner)
            {
                // Outer vertex: on the grid corners
                x = col * subCellSize;
                y = row / 2 * subCellSize;
            }
            else
            {
                // Inner vertex: at cell centers (offset by half cell)
                x = (col + 0.5f) * subCellSize;
                y = (row / 2 + 0.5f) * subCellSize;
            }

            float z = chunk.Heights[i];

            // World position = chunk corner + local offset
            float wx = chunk.WorldPosition.X - y;
            float wy = chunk.WorldPosition.Y - x;
            vertices[i * 12 + 0] = wx;
            vertices[i * 12 + 1] = wy;
            vertices[i * 12 + 2] = z;

            // Update AABB
            bMin = Vector3.Min(bMin, new Vector3(wx, wy, z));
            bMax = Vector3.Max(bMax, new Vector3(wx, wy, z));

            // Normal
            var n = i < chunk.Normals.Length ? chunk.Normals[i] : Vector3.UnitZ;
            vertices[i * 12 + 3] = n.X;
            vertices[i * 12 + 4] = n.Y;
            vertices[i * 12 + 5] = n.Z;

            // Texture coordinates (0-1 across the chunk)
            if (!isInner)
            {
                vertices[i * 12 + 6] = col / 8f;
                vertices[i * 12 + 7] = (row / 2) / 8f;
            }
            else
            {
                vertices[i * 12 + 6] = (col + 0.5f) / 8f;
                vertices[i * 12 + 7] = (row / 2 + 0.5f) / 8f;
            }

            // MCCV vertex color (RGBA, normalized). Default = white when MCCV is absent.
            float r = 1f, g = 1f, b = 1f, a = 1f;
            if (chunk.MccvColors != null)
            {
                int cBase = i * 4;
                if (cBase + 3 < chunk.MccvColors.Length)
                {
                    r = chunk.MccvColors[cBase + 0] / 255f;
                    g = chunk.MccvColors[cBase + 1] / 255f;
                    b = chunk.MccvColors[cBase + 2] / 255f;
                    a = chunk.MccvColors[cBase + 3] / 255f;
                }
            }
            vertices[i * 12 + 8] = r;
            vertices[i * 12 + 9] = g;
            vertices[i * 12 + 10] = b;
            vertices[i * 12 + 11] = a;
        }

        // Build index buffer: 8×8 cells, each split into 4 triangles via center vertex = 256 triangles
        var indices = BuildIndices(chunk.HoleMask);

        if (indices.Length == 0) return null;

        return UploadMesh(vertices, indices, chunk, bMin, bMax);
    }

    /// <summary>
    /// Determine the row, column, and type (outer/inner) for a vertex index in interleaved layout.
    /// Interleaved: row0(9 outer), row1(8 inner), row2(9 outer), ... row16(9 outer)
    /// </summary>
    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        // Walk through rows to find which row this index belongs to
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int r = 0; r < 17; r++)
        {
            int rowSize = (r % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = r;
                col = remaining;
                isInner = (r % 2 != 0);
                return;
            }
            remaining -= rowSize;
        }
    }

    /// <summary>
    /// Get the vertex index in the interleaved 145-vertex layout for an outer vertex at (outerRow, outerCol).
    /// </summary>
    private static int OuterIndex(int outerRow, int outerCol)
    {
        // Each pair of rows (outer + inner) = 9 + 8 = 17 vertices
        return outerRow * 17 + outerCol;
    }

    /// <summary>
    /// Get the vertex index for an inner vertex at (innerRow, innerCol).
    /// Inner row i sits between outer row i and outer row i+1.
    /// </summary>
    private static int InnerIndex(int innerRow, int innerCol)
    {
        return innerRow * 17 + 9 + innerCol;
    }

    /// <summary>
    /// Build triangle indices for the 8×8 cell grid.
    /// Each cell has 4 corners (outer) and 1 center (inner), forming 4 triangles.
    /// Holes are skipped based on the hole mask (4×4 groups of 2×2 cells).
    /// </summary>
    private static ushort[] BuildIndices(int holeMask)
    {
        var indices = new List<ushort>(256 * 3);

        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                // Check hole mask: 4×4 groups, each covering 2×2 cells
                if (holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << (holeY * 4 + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue; // This cell is a hole
                }

                // Four outer corners of this cell
                ushort tl = (ushort)OuterIndex(cellY, cellX);         // top-left
                ushort tr = (ushort)OuterIndex(cellY, cellX + 1);     // top-right
                ushort bl = (ushort)OuterIndex(cellY + 1, cellX);     // bottom-left
                ushort br = (ushort)OuterIndex(cellY + 1, cellX + 1); // bottom-right

                // Center inner vertex
                ushort center = (ushort)InnerIndex(cellY, cellX);

                // 4 triangles (fan from center), CW winding for our coordinate system
                // Top triangle
                indices.Add(center);
                indices.Add(tr);
                indices.Add(tl);

                // Right triangle
                indices.Add(center);
                indices.Add(br);
                indices.Add(tr);

                // Bottom triangle
                indices.Add(center);
                indices.Add(bl);
                indices.Add(br);

                // Left triangle
                indices.Add(center);
                indices.Add(tl);
                indices.Add(bl);
            }
        }

        return indices.ToArray();
    }

    private unsafe TerrainChunkMesh UploadMesh(float[] vertices, ushort[] indices, TerrainChunkData chunk, Vector3 boundsMin, Vector3 boundsMax)
    {
        uint vao = _gl.GenVertexArray();
        _gl.BindVertexArray(vao);

        uint vbo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* ptr = vertices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        uint ebo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        fixed (ushort* ptr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        uint stride = 12 * sizeof(float);
        // Position (location 0)
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        // Normal (location 1)
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        // TexCoord (location 2)
        _gl.EnableVertexAttribArray(2);
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));
        // Vertex color (location 5)
        _gl.EnableVertexAttribArray(5);
        _gl.VertexAttribPointer(5, 4, VertexAttribPointerType.Float, false, stride, (void*)(8 * sizeof(float)));

        _gl.BindVertexArray(0);

        return new TerrainChunkMesh
        {
            Vao = vao,
            Vbo = vbo,
            Ebo = ebo,
            IndexCount = (uint)indices.Length,
            ChunkX = chunk.ChunkX,
            ChunkY = chunk.ChunkY,
            TileX = chunk.TileX,
            TileY = chunk.TileY,
            WorldPosition = chunk.WorldPosition,
            BoundsMin = boundsMin,
            BoundsMax = boundsMax,
            AreaId = chunk.AreaId,
            McnkFlags = chunk.McnkFlags,
            Layers = chunk.Layers,
            AlphaMaps = chunk.AlphaMaps,
            ShadowMap = chunk.ShadowMap
        };
    }
}

/// <summary>
/// GPU-resident mesh for a single terrain chunk.
/// </summary>
public class TerrainChunkMesh : IDisposable
{
    public uint Vao { get; init; }
    public uint Vbo { get; init; }
    public uint Ebo { get; init; }
    public uint IndexCount { get; init; }
    public int ChunkX { get; init; }
    public int ChunkY { get; init; }
    public int TileX { get; init; }
    public int TileY { get; init; }
    public Vector3 WorldPosition { get; init; }
    public Vector3 BoundsMin { get; init; }
    public Vector3 BoundsMax { get; init; }
    public int AreaId { get; init; }
    public int McnkFlags { get; init; }
    public TerrainLayer[] Layers { get; init; } = Array.Empty<TerrainLayer>();
    public Dictionary<int, byte[]> AlphaMaps { get; init; } = new();

    /// <summary>MCSH shadow map: 64×64 bytes (0=lit, 255=shadowed). Null if no shadow data.</summary>
    public byte[]? ShadowMap { get; init; }

    /// <summary>GL texture handles for alpha maps (layer index → GL texture).</summary>
    public Dictionary<int, uint> AlphaTextures { get; } = new();

    /// <summary>GL texture handle for shadow map. 0 if no shadow data.</summary>
    public uint ShadowTexture { get; set; }

    internal GL? Gl { get; set; }

    public void Dispose()
    {
        if (Gl == null) return;
        Gl.DeleteVertexArray(Vao);
        Gl.DeleteBuffer(Vbo);
        Gl.DeleteBuffer(Ebo);
        foreach (var tex in AlphaTextures.Values)
            Gl.DeleteTexture(tex);
        if (ShadowTexture != 0)
            Gl.DeleteTexture(ShadowTexture);
    }
}
