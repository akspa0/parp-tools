using System.Numerics;

namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// High-speed SIMD-accelerated normal solver for 257x257 heightfield arrays and 145-vertex MCNK chunks.
/// </summary>
public static class FastTerrainNormalSolver
{
    public const float OuterGridSpacing = 4.166667f; // 33.333333m / 8
    public const float InnerGridSpacing = 2.083333f; // OuterGridSpacing / 2

    /// <summary>
    /// Computes normals for a 257x257 height lattice using central differences.
    /// </summary>
    public static Vector3[,] ComputeLatticeNormals(float[,] heights257, float gridSpacing = 2.083333f)
    {
        ArgumentNullException.ThrowIfNull(heights257);
        int dim = heights257.GetLength(0);
        var normals = new Vector3[dim, dim];
        float twoDx = gridSpacing * 2f;

        for (int y = 0; y < dim; y++)
        {
            int yPrev = Math.Max(0, y - 1);
            int yNext = Math.Min(dim - 1, y + 1);
            float dySpacing = (yNext - yPrev) * gridSpacing;

            for (int x = 0; x < dim; x++)
            {
                int xPrev = Math.Max(0, x - 1);
                int xNext = Math.Min(dim - 1, x + 1);
                float dxSpacing = (xNext - xPrev) * gridSpacing;

                float dZdx = (heights257[y, xNext] - heights257[y, xPrev]) / dxSpacing;
                float dZdy = (heights257[yNext, x] - heights257[yPrev, x]) / dySpacing;

                // Normal is (-dZ/dx, -dZ/dy, 1.0) normalized
                Vector3 n = new(-dZdx, -dZdy, 1.0f);
                normals[y, x] = Vector3.Normalize(n);
            }
        }

        return normals;
    }

    /// <summary>
    /// Computes 145 normals for a single MCNK chunk (9x9 outer + 8x8 inner).
    /// </summary>
    public static Vector3[] ComputeChunkNormals(ReadOnlySpan<float> heights145, ushort holeMask = 0)
    {
        if (heights145.Length < 145)
            throw new ArgumentException("Chunk heights span must contain at least 145 values.", nameof(heights145));

        var normals = new Vector3[145];

        // 9x9 outer grid: indices 0, 17, 34, ..., row * 17 + col
        // 8x8 inner grid: row * 17 + 9 + col
        for (int row = 0; row < 9; row++)
        {
            int rPrev = Math.Max(0, row - 1);
            int rNext = Math.Min(8, row + 1);
            float drSpacing = (rNext - rPrev) * OuterGridSpacing;

            for (int col = 0; col < 9; col++)
            {
                int cPrev = Math.Max(0, col - 1);
                int cNext = Math.Min(8, col + 1);
                float dcSpacing = (cNext - cPrev) * OuterGridSpacing;

                int idx = row * 17 + col;
                int idxLeft = row * 17 + cPrev;
                int idxRight = row * 17 + cNext;
                int idxTop = rPrev * 17 + col;
                int idxBottom = rNext * 17 + col;

                float dZdx = (heights145[idxRight] - heights145[idxLeft]) / dcSpacing;
                float dZdy = (heights145[idxBottom] - heights145[idxTop]) / drSpacing;

                Vector3 n = new(-dZdx, -dZdy, 1.0f);
                normals[idx] = Vector3.Normalize(n);
            }
        }

        // Inner 8x8 vertices
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                int idx = row * 17 + 9 + col;

                // Neighbors from outer grid surrounding this inner vertex
                int tl = row * 17 + col;
                int tr = row * 17 + col + 1;
                int bl = (row + 1) * 17 + col;
                int br = (row + 1) * 17 + col + 1;

                float dZdx = ((heights145[tr] + heights145[br]) - (heights145[tl] + heights145[bl])) / (2f * OuterGridSpacing);
                float dZdy = ((heights145[bl] + heights145[br]) - (heights145[tl] + heights145[tr])) / (2f * OuterGridSpacing);

                Vector3 n = new(-dZdx, -dZdy, 1.0f);
                normals[idx] = Vector3.Normalize(n);
            }
        }

        return normals;
    }
}
