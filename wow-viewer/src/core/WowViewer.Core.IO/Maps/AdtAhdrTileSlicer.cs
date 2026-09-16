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
