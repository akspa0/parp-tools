using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace MdxViewer.Export;

public static class TerrainHeightmapIo
{
    public const int TileHeightmapSize = 257; // 16 chunks * 16 half-steps + 1
    private const int HalfStepsPerChunk = 16;

    public sealed class TileHeightmap257
    {
        public float[] Heights { get; init; } = Array.Empty<float>(); // size=257*257
        public float MinHeight { get; init; }
        public float MaxHeight { get; init; }
    }

    public static TileHeightmap257 BuildTileHeightmap257(IReadOnlyList<Terrain.TerrainChunkData> chunks)
    {
        int w = TileHeightmapSize;
        int h = TileHeightmapSize;

        var sum = new float[w * h];
        var count = new ushort[w * h];

        foreach (var chunk in chunks)
        {
            int cx = chunk.ChunkX;
            int cy = chunk.ChunkY;
            if ((uint)cx >= 16u || (uint)cy >= 16u) continue;
            if (chunk.Heights == null || chunk.Heights.Length < 145) continue;

            int baseX = cx * HalfStepsPerChunk;
            int baseY = cy * HalfStepsPerChunk;

            for (int i = 0; i < 145; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                int hx;
                int hy;
                if (!isInner)
                {
                    hx = col * 2;
                    hy = (row / 2) * 2;
                }
                else
                {
                    hx = col * 2 + 1;
                    hy = (row / 2) * 2 + 1;
                }

                int px = baseX + hx;
                int py = baseY + hy;
                if ((uint)px >= (uint)w || (uint)py >= (uint)h) continue;

                int idx = py * w + px;
                sum[idx] += chunk.Heights[i];
                count[idx]++;
            }
        }

        var grid = new float[w * h];
        float min = float.MaxValue;
        float max = float.MinValue;

        for (int i = 0; i < grid.Length; i++)
        {
            if (count[i] > 0)
            {
                float v = sum[i] / count[i];
                grid[i] = v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            else
            {
                grid[i] = float.NaN;
            }
        }

        // Fill mixed-parity points for a visually continuous grid.
        // Authoritative samples are at (even,even) outer corners and (odd,odd) inner centers.
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int idx = y * w + x;
                if (!float.IsNaN(grid[idx]))
                    continue;

                if ((x & 1) == 1 && (y & 1) == 0)
                {
                    // Between two outer/derived samples horizontally.
                    float l = grid[y * w + (x - 1)];
                    float r = grid[y * w + (x + 1)];
                    if (!float.IsNaN(l) && !float.IsNaN(r))
                        grid[idx] = (l + r) * 0.5f;
                }
                else if ((x & 1) == 0 && (y & 1) == 1)
                {
                    // Between two outer/derived samples vertically.
                    float u = grid[(y - 1) * w + x];
                    float d = grid[(y + 1) * w + x];
                    if (!float.IsNaN(u) && !float.IsNaN(d))
                        grid[idx] = (u + d) * 0.5f;
                }
            }
        }

        // Fallback: fill any remaining NaNs (e.g., missing chunks) with nearest available sample.
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int idx = y * w + x;
                if (!float.IsNaN(grid[idx]))
                    continue;

                if (TryFindNearest(grid, w, h, x, y, out float nearest))
                    grid[idx] = nearest;
                else
                    grid[idx] = 0f;
            }
        }

        if (min == float.MaxValue || max == float.MinValue)
        {
            min = 0f;
            max = 0f;
        }

        return new TileHeightmap257 { Heights = grid, MinHeight = min, MaxHeight = max };
    }

    public static Image<L16> EncodeL16(float[] heights, float minHeight, float maxHeight)
    {
        if (heights == null) throw new ArgumentNullException(nameof(heights));
        if (heights.Length < TileHeightmapSize * TileHeightmapSize)
            throw new ArgumentException($"Expected {TileHeightmapSize}x{TileHeightmapSize} height array.");

        float range = maxHeight - minHeight;
        if (range <= 1e-6f)
            range = 1f;

        var img = new Image<L16>(TileHeightmapSize, TileHeightmapSize);
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < TileHeightmapSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                int baseIdx = y * TileHeightmapSize;
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    float v = heights[baseIdx + x];
                    float t = (v - minHeight) / range;
                    if (t < 0f) t = 0f;
                    if (t > 1f) t = 1f;
                    ushort u = (ushort)Math.Clamp((int)MathF.Round(t * 65535f), 0, 65535);
                    row[x] = new L16(u);
                }
            }
        });
        return img;
    }

    public static float[] DecodeL16(Image<L16> img, float minHeight, float maxHeight)
    {
        if (img.Width != TileHeightmapSize || img.Height != TileHeightmapSize)
            throw new ArgumentException($"Expected {TileHeightmapSize}x{TileHeightmapSize} L16 image.");

        float range = maxHeight - minHeight;
        if (range <= 1e-6f)
            range = 1f;

        var heights = new float[TileHeightmapSize * TileHeightmapSize];
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < TileHeightmapSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                int baseIdx = y * TileHeightmapSize;
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    float t = row[x].PackedValue / 65535f;
                    heights[baseIdx + x] = minHeight + t * range;
                }
            }
        });

        return heights;
    }

    public static List<Terrain.TerrainChunkData> ApplyHeightmap257ToChunks(IReadOnlyList<Terrain.TerrainChunkData> chunks, float[] tileHeightmap)
    {
        if (tileHeightmap == null) throw new ArgumentNullException(nameof(tileHeightmap));
        if (tileHeightmap.Length < TileHeightmapSize * TileHeightmapSize)
            throw new ArgumentException($"Expected {TileHeightmapSize}x{TileHeightmapSize} height array.");

        var result = new List<Terrain.TerrainChunkData>(chunks.Count);
        foreach (var chunk in chunks)
        {
            int cx = chunk.ChunkX;
            int cy = chunk.ChunkY;
            if ((uint)cx >= 16u || (uint)cy >= 16u)
            {
                result.Add(chunk);
                continue;
            }

            int baseX = cx * HalfStepsPerChunk;
            int baseY = cy * HalfStepsPerChunk;

            var newHeights = new float[145];
            for (int i = 0; i < 145; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                int hx;
                int hy;
                if (!isInner)
                {
                    hx = col * 2;
                    hy = (row / 2) * 2;
                }
                else
                {
                    hx = col * 2 + 1;
                    hy = (row / 2) * 2 + 1;
                }

                int px = baseX + hx;
                int py = baseY + hy;
                px = Math.Clamp(px, 0, TileHeightmapSize - 1);
                py = Math.Clamp(py, 0, TileHeightmapSize - 1);

                newHeights[i] = tileHeightmap[py * TileHeightmapSize + px];
            }

            var newNormals = GenerateNormals(chunk, newHeights);

            result.Add(new Terrain.TerrainChunkData
            {
                TileX = chunk.TileX,
                TileY = chunk.TileY,
                ChunkX = chunk.ChunkX,
                ChunkY = chunk.ChunkY,
                Heights = newHeights,
                Normals = newNormals,
                HoleMask = chunk.HoleMask,
                Layers = chunk.Layers,
                AlphaMaps = chunk.AlphaMaps,
                ShadowMap = chunk.ShadowMap,
                MccvColors = chunk.MccvColors,
                Liquid = chunk.Liquid,
                WorldPosition = chunk.WorldPosition,
                AreaId = chunk.AreaId,
                McnkFlags = chunk.McnkFlags
            });
        }

        return result;
    }

    private static bool TryFindNearest(float[] grid, int w, int h, int x, int y, out float value)
    {
        value = 0f;
        const int maxRadius = 24;

        for (int r = 1; r <= maxRadius; r++)
        {
            int y0 = Math.Max(0, y - r);
            int y1 = Math.Min(h - 1, y + r);
            int x0 = Math.Max(0, x - r);
            int x1 = Math.Min(w - 1, x + r);

            // Top/bottom rows
            for (int xx = x0; xx <= x1; xx++)
            {
                float a = grid[y0 * w + xx];
                if (!float.IsNaN(a)) { value = a; return true; }
                float b = grid[y1 * w + xx];
                if (!float.IsNaN(b)) { value = b; return true; }
            }

            // Left/right cols
            for (int yy = y0 + 1; yy <= y1 - 1; yy++)
            {
                float a = grid[yy * w + x0];
                if (!float.IsNaN(a)) { value = a; return true; }
                float b = grid[yy * w + x1];
                if (!float.IsNaN(b)) { value = b; return true; }
            }
        }

        return false;
    }

    // ---- Normals ----

    private static Vector3[] GenerateNormals(Terrain.TerrainChunkData chunk, float[] heights)
    {
        var positions = new Vector3[145];
        for (int i = 0; i < 145; i++)
            positions[i] = GetVertexWorldPosition(chunk, heights, i);

        var indices = BuildIndices(chunk.HoleMask);
        var accum = new Vector3[145];

        for (int t = 0; t + 2 < indices.Length; t += 3)
        {
            int i0 = indices[t + 0];
            int i1 = indices[t + 1];
            int i2 = indices[t + 2];

            var p0 = positions[i0];
            var p1 = positions[i1];
            var p2 = positions[i2];

            var e1 = p1 - p0;
            var e2 = p2 - p0;
            var n = Vector3.Cross(e1, e2);
            float lenSq = n.LengthSquared();
            if (lenSq < 1e-10f)
                continue;

            n = Vector3.Normalize(n);
            accum[i0] += n;
            accum[i1] += n;
            accum[i2] += n;
        }

        var normals = new Vector3[145];
        for (int i = 0; i < 145; i++)
        {
            var n = accum[i];
            float lenSq = n.LengthSquared();
            normals[i] = lenSq > 1e-10f ? Vector3.Normalize(n) : Vector3.UnitZ;
        }

        return normals;
    }

    private static Vector3 GetVertexWorldPosition(Terrain.TerrainChunkData chunk, float[] heights, int index)
    {
        GetVertexPosition(index, out int row, out int col, out bool isInner);

        float cellSize = Rendering.WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        float x, y;
        if (!isInner)
        {
            x = col * subCellSize;
            y = row / 2 * subCellSize;
        }
        else
        {
            x = (col + 0.5f) * subCellSize;
            y = (row / 2 + 0.5f) * subCellSize;
        }

        float z = (index < heights.Length) ? heights[index] : 0f;
        float wx = chunk.WorldPosition.X - y;
        float wy = chunk.WorldPosition.Y - x;
        return new Vector3(wx, wy, z);
    }

    // ---- Chunk mesh topology helpers (copied from TerrainMeshBuilder) ----

    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
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

    private static int OuterIndex(int outerRow, int outerCol) => outerRow * 17 + outerCol;
    private static int InnerIndex(int innerRow, int innerCol) => innerRow * 17 + 9 + innerCol;

    private static int[] BuildIndices(int holeMask)
    {
        var indices = new List<int>(256 * 3);

        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                if (holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << (holeY * 4 + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                int tl = OuterIndex(cellY, cellX);
                int tr = OuterIndex(cellY, cellX + 1);
                int bl = OuterIndex(cellY + 1, cellX);
                int br = OuterIndex(cellY + 1, cellX + 1);
                int center = InnerIndex(cellY, cellX);

                indices.Add(center);
                indices.Add(tr);
                indices.Add(tl);

                indices.Add(center);
                indices.Add(br);
                indices.Add(tr);

                indices.Add(center);
                indices.Add(bl);
                indices.Add(br);

                indices.Add(center);
                indices.Add(tl);
                indices.Add(bl);
            }
        }

        return indices.ToArray();
    }
}
