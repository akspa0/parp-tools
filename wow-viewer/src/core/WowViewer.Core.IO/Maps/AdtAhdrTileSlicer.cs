using System.Numerics;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Spec 237: converts whole-tile AHDR-family grids into the per-chunk 145-vertex interleaved
/// layout (9 outer, 8 inner, … 17 rows) used by the terrain renderer.
/// <para>
/// Measured for DAT v26: the outer grid is row-major, and chunk (IndexX, IndexY) covers outer rows
/// IndexY*8..IndexY*8+8 and columns IndexX*8..IndexX*8+8. The inner grid is assumed to use the same
/// orientation (unverified; a wrong inner order shows up as spikes).
/// </para>
/// </summary>
public static class AdtAhdrTileSlicer
{
    public const int VerticesPerChunk = 145;
    private const int CellsPerChunk = 8;

    public static float[] SliceHeights(AdtAhdrTile tile, int chunkX, int chunkY)
    {
        var heights = new float[VerticesPerChunk];
        if (!HasGrids(tile))
            return heights;

        int outerWidth = tile.VerticesX;
        int innerWidth = tile.VerticesX - 1;
        int index = 0;
        for (int row = 0; row < 17; row++)
        {
            int r = row / 2;
            if ((row & 1) == 0)
            {
                int baseRow = (chunkY * CellsPerChunk + r) * outerWidth;
                for (int c = 0; c <= CellsPerChunk; c++)
                    heights[index++] = tile.OuterHeights[baseRow + chunkX * CellsPerChunk + c];
            }
            else
            {
                int baseRow = (chunkY * CellsPerChunk + r) * innerWidth;
                for (int c = 0; c < CellsPerChunk; c++)
                    heights[index++] = tile.InnerHeights[baseRow + chunkX * CellsPerChunk + c];
            }
        }

        return heights;
    }

    /// <summary>
    /// Normals computed from the outer height grid by central differences, in the renderer frame
    /// where a grid row step moves along -X and a column step moves along -Y. ANRM is not used
    /// because its v26 encoding is unverified.
    /// </summary>
    public static Vector3[] ComputeNormals(AdtAhdrTile tile, int chunkX, int chunkY, float vertexSpacing)
    {
        var normals = new Vector3[VerticesPerChunk];
        if (!HasGrids(tile))
        {
            Array.Fill(normals, Vector3.UnitZ);
            return normals;
        }

        int index = 0;
        for (int row = 0; row < 17; row++)
        {
            int r = row / 2;
            int gridRow = chunkY * CellsPerChunk + r;
            if ((row & 1) == 0)
            {
                for (int c = 0; c <= CellsPerChunk; c++)
                    normals[index++] = OuterNormal(tile, gridRow, chunkX * CellsPerChunk + c, vertexSpacing);
            }
            else
            {
                for (int c = 0; c < CellsPerChunk; c++)
                {
                    int gridCol = chunkX * CellsPerChunk + c;
                    Vector3 sum = OuterNormal(tile, gridRow, gridCol, vertexSpacing)
                        + OuterNormal(tile, gridRow, gridCol + 1, vertexSpacing)
                        + OuterNormal(tile, gridRow + 1, gridCol, vertexSpacing)
                        + OuterNormal(tile, gridRow + 1, gridCol + 1, vertexSpacing);
                    normals[index++] = Vector3.Normalize(sum);
                }
            }
        }

        return normals;
    }

    /// <summary>
    /// Stored ANRM normals for one chunk in the renderer frame. MEASURED for DAT v26: components are
    /// (column axis, vertical, row axis) with 127 = 1.0 (mean dot product 0.985 with normals computed from AVTX
    /// in inches, next-best order 0.658; script anrm_acvt_v26.py). Renderer X decreases along rows and Y along
    /// columns, so the renderer normal is (-row, -column, vertical). The inner grid is assumed to use the same
    /// order as the outer grid. Returns null when ANRM is absent or the wrong size.
    /// </summary>
    public static Vector3[]? SliceStoredNormals(AdtAhdrTile tile, int chunkX, int chunkY)
    {
        if (!HasGrids(tile) || tile.NormalsRaw is not { Length: (129 * 129 + 128 * 128) * 3 } raw)
            return null;

        var normals = new Vector3[VerticesPerChunk];
        int index = 0;
        for (int row = 0; row < 17; row++)
        {
            int r = row / 2;
            bool outer = (row & 1) == 0;
            int width = outer ? 129 : 128;
            int gridOffset = outer ? 0 : 129 * 129;
            int count = outer ? CellsPerChunk + 1 : CellsPerChunk;
            for (int c = 0; c < count; c++)
            {
                int vertex = gridOffset + (chunkY * CellsPerChunk + r) * width + chunkX * CellsPerChunk + c;
                float column = (sbyte)raw[vertex * 3];
                float vertical = (sbyte)raw[vertex * 3 + 1];
                float rowComponent = (sbyte)raw[vertex * 3 + 2];
                var n = new Vector3(-rowComponent, -column, vertical);
                normals[index++] = n.LengthSquared() > 0 ? Vector3.Normalize(n) : Vector3.UnitZ;
            }
        }

        return normals;
    }

    /// <summary>
    /// ACVT vertex colours for one chunk as 145 × 4 bytes in the renderer's MCCV layout. MEASURED for DAT v26:
    /// three bytes centre on 127 (neutral, as MCCV) and the fourth is always 255; which of the first and third
    /// bytes is red is not established (the corpus is almost entirely neutral), so bytes are passed through in
    /// file order and treated as MCCV's BGRA. Returns null when ACVT is absent or the wrong size.
    /// </summary>
    public static byte[]? SliceVertexColors(AdtAhdrTile tile, int chunkX, int chunkY)
    {
        if (!HasGrids(tile) || tile.VertexShadingRaw is not { Length: (129 * 129 + 128 * 128) * 4 } raw)
            return null;

        var colors = new byte[VerticesPerChunk * 4];
        int index = 0;
        for (int row = 0; row < 17; row++)
        {
            int r = row / 2;
            bool outer = (row & 1) == 0;
            int width = outer ? 129 : 128;
            int gridOffset = outer ? 0 : 129 * 129;
            int count = outer ? CellsPerChunk + 1 : CellsPerChunk;
            for (int c = 0; c < count; c++)
            {
                int vertex = gridOffset + (chunkY * CellsPerChunk + r) * width + chunkX * CellsPerChunk + c;
                Buffer.BlockCopy(raw, vertex * 4, colors, index * 4, 4);
                index++;
            }
        }

        return colors;
    }

    /// <summary>Inches per outer-grid cell: a chunk is 1200 inches (33.33 yd) wide and has 8 cells.</summary>
    public const float InchesPerCell = 150f;

    /// <summary>Mean of the chunk's 145 AVTX heights (81 outer + 64 inner), in file units (inches for v26).</summary>
    public static float ChunkMeanHeight(AdtAhdrTile tile, int chunkX, int chunkY)
    {
        if (!HasGrids(tile))
            return 0f;

        double sum = 0;
        for (int r = 0; r <= CellsPerChunk; r++)
            for (int c = 0; c <= CellsPerChunk; c++)
                sum += tile.OuterHeights[(chunkY * CellsPerChunk + r) * tile.VerticesX + chunkX * CellsPerChunk + c];
        for (int r = 0; r < CellsPerChunk; r++)
            for (int c = 0; c < CellsPerChunk; c++)
                sum += tile.InnerHeights[(chunkY * CellsPerChunk + r) * (tile.VerticesX - 1) + chunkX * CellsPerChunk + c];

        return (float)(sum / VerticesPerChunk);
    }

    /// <summary>
    /// Resolves an ACDO placement to tile-grid coordinates: fractional outer-grid column and row (0..128) and
    /// the absolute height in file units. Frame measured for DAT v26 (see <see cref="AdtAhdrObjectDefinition"/>).
    /// </summary>
    public static (float Column, float Row, float Height) ResolveObjectGridPosition(AdtAhdrTile tile, AdtAhdrChunk chunk, AdtAhdrObjectDefinition obj)
    {
        float column = chunk.IndexX * CellsPerChunk + CellsPerChunk / 2f + obj.LocalPositionInches.X / InchesPerCell;
        float row = chunk.IndexY * CellsPerChunk + CellsPerChunk / 2f + obj.LocalPositionInches.Z / InchesPerCell;
        float height = ChunkMeanHeight(tile, chunk.IndexX, chunk.IndexY) + obj.LocalPositionInches.Y;
        return (column, row, height);
    }

    private static bool HasGrids(AdtAhdrTile tile) =>
        tile.VerticesX == 129 && tile.VerticesY == 129
        && tile.OuterHeights.Length == 129 * 129
        && tile.InnerHeights.Length == 128 * 128;

    private static Vector3 OuterNormal(AdtAhdrTile tile, int row, int col, float spacing)
    {
        int width = tile.VerticesX;
        int maxRow = tile.VerticesY - 1;
        int maxCol = tile.VerticesX - 1;
        int r0 = Math.Max(row - 1, 0), r1 = Math.Min(row + 1, maxRow);
        int c0 = Math.Max(col - 1, 0), c1 = Math.Min(col + 1, maxCol);

        float dhdRow = (tile.OuterHeights[r1 * width + col] - tile.OuterHeights[r0 * width + col]) / ((r1 - r0) * spacing);
        float dhdCol = (tile.OuterHeights[row * width + c1] - tile.OuterHeights[row * width + c0]) / ((c1 - c0) * spacing);

        // World X decreases as row increases and world Y decreases as column increases,
        // so dh/dX = -dh/dRow and dh/dY = -dh/dCol; the surface normal is (-dh/dX, -dh/dY, 1).
        return Vector3.Normalize(new Vector3(dhdRow, dhdCol, 1f));
    }
}
