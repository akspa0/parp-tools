using WowViewer.Core.IO.Maps;
using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWViewer.Terrain.Vlm;

/// <summary>
/// Shared tile bake path for VLM terrain outputs.
/// Mirrors the viewer's coherent 257x257 tile reconstruction so exporter outputs
/// and standalone heightmap baking use the same chunk stitching rules.
/// </summary>
public static class TerrainTileBakeService
{
    public const int TileHeightmapSize = 257;
    private const int HalfStepsPerChunk = 16;
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16f;

    public sealed class TileHeightmap257
    {
        public float[] Heights { get; init; } = Array.Empty<float>();
        public float MinHeight { get; init; }
        public float MaxHeight { get; init; }
    }

    public static TileHeightmap257 BuildTileHeightmap257(IReadOnlyList<VlmChunkHeights> chunks, bool isInterleaved)
    {
        ArgumentNullException.ThrowIfNull(chunks);

        var heightsByChunk = new Dictionary<int, float[]>(chunks.Count);
        foreach (var chunk in chunks)
        {
            if (chunk.ChunkIndex is < 0 or >= 256)
                continue;

            if (chunk.Heights == null || chunk.Heights.Length < 145)
                continue;

            heightsByChunk[chunk.ChunkIndex] = chunk.Heights;
        }

        return BuildTileHeightmap257(heightsByChunk, isInterleaved);
    }

    public static TileHeightmap257 BuildTileHeightmap257(IReadOnlyDictionary<int, float[]> heightsByChunk, bool isInterleaved)
    {
        ArgumentNullException.ThrowIfNull(heightsByChunk);

        int width = TileHeightmapSize;
        int height = TileHeightmapSize;

        var sum = new float[width * height];
        var count = new ushort[width * height];

        foreach ((int chunkIndex, float[] rawHeights) in heightsByChunk)
        {
            if (chunkIndex is < 0 or >= 256)
                continue;

            if (rawHeights == null || rawHeights.Length < 145)
                continue;

            float[] heights = NormalizeChunkHeights(rawHeights, isInterleaved);
            int chunkX = chunkIndex % 16;
            int chunkY = chunkIndex / 16;
            int baseX = chunkX * HalfStepsPerChunk;
            int baseY = chunkY * HalfStepsPerChunk;

            for (int i = 0; i < 145; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;

                int px = baseX + sampleX;
                int py = baseY + sampleY;
                if ((uint)px >= (uint)width || (uint)py >= (uint)height)
                    continue;

                int gridIndex = (py * width) + px;
                sum[gridIndex] += heights[i];
                count[gridIndex]++;
            }
        }

        var grid = new float[width * height];
        float min = float.MaxValue;
        float max = float.MinValue;

        for (int i = 0; i < grid.Length; i++)
        {
            if (count[i] > 0)
            {
                float value = sum[i] / count[i];
                grid[i] = value;
                if (value < min) min = value;
                if (value > max) max = value;
            }
            else
            {
                grid[i] = float.NaN;
            }
        }

        FillHeightGaps(grid, width, height);

        if (min == float.MaxValue || max == float.MinValue)
        {
            min = 0f;
            max = 0f;
        }

        return new TileHeightmap257
        {
            Heights = grid,
            MinHeight = min,
            MaxHeight = max
        };
    }

    public static Vector3[] BuildTileNormals257(float[] tileHeightmap, IReadOnlyDictionary<int, int>? holeMasks = null)
    {
        ArgumentNullException.ThrowIfNull(tileHeightmap);

        if (tileHeightmap.Length < TileHeightmapSize * TileHeightmapSize)
            throw new ArgumentException($"Expected {TileHeightmapSize}x{TileHeightmapSize} height array.", nameof(tileHeightmap));

        int width = TileHeightmapSize;
        int height = TileHeightmapSize;
        var sum = new Vector3[width * height];
        var count = new ushort[width * height];
        var grid = new Vector3[width * height];
        var hasSample = new bool[width * height];

        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
        {
            int chunkX = chunkIndex % 16;
            int chunkY = chunkIndex / 16;
            int holeMask = 0;
            if (holeMasks != null && holeMasks.TryGetValue(chunkIndex, out int mask))
                holeMask = mask;

            float[] chunkHeights = ExtractChunkHeights(tileHeightmap, chunkX, chunkY);
            Vector3[] chunkNormals = GenerateChunkNormals(chunkX, chunkY, holeMask, chunkHeights);

            int baseX = chunkX * HalfStepsPerChunk;
            int baseY = chunkY * HalfStepsPerChunk;

            for (int i = 0; i < 145; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;

                int px = baseX + sampleX;
                int py = baseY + sampleY;
                if ((uint)px >= (uint)width || (uint)py >= (uint)height)
                    continue;

                int gridIndex = (py * width) + px;
                sum[gridIndex] += chunkNormals[i];
                count[gridIndex]++;
            }
        }

        for (int i = 0; i < grid.Length; i++)
        {
            if (count[i] == 0)
                continue;

            grid[i] = Vector3.Normalize(sum[i] / count[i]);
            hasSample[i] = true;
        }

        FillNormalGaps(grid, hasSample, width, height);
        return grid;
    }

    public static Image<L16> CreateHeightmapImage(float[] heights, float minHeight, float maxHeight, int outputSize)
    {
        ArgumentNullException.ThrowIfNull(heights);

        if (heights.Length < TileHeightmapSize * TileHeightmapSize)
            throw new ArgumentException($"Expected {TileHeightmapSize}x{TileHeightmapSize} height array.", nameof(heights));

        float range = maxHeight - minHeight;
        if (range <= 1e-6f)
            range = 1f;

        var image = new Image<L16>(TileHeightmapSize, TileHeightmapSize);
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < TileHeightmapSize; y++)
            {
                Span<L16> row = accessor.GetRowSpan(y);
                int baseIndex = y * TileHeightmapSize;
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    float value = heights[baseIndex + x];
                    float normalized = Math.Clamp((value - minHeight) / range, 0f, 1f);
                    row[x] = new L16((ushort)Math.Clamp((int)MathF.Round(normalized * 65535f), 0, 65535));
                }
            }
        });

        if (outputSize != TileHeightmapSize)
            image.Mutate(ctx => ctx.Resize(outputSize, outputSize, KnownResamplers.Lanczos3));

        return image;
    }

    public static Image<Rgba32> CreateNormalmapImage(Vector3[] normals, int outputSize)
    {
        ArgumentNullException.ThrowIfNull(normals);

        if (normals.Length < TileHeightmapSize * TileHeightmapSize)
            throw new ArgumentException($"Expected {TileHeightmapSize}x{TileHeightmapSize} normal array.", nameof(normals));

        var image = new Image<Rgba32>(TileHeightmapSize, TileHeightmapSize);
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < TileHeightmapSize; y++)
            {
                Span<Rgba32> row = accessor.GetRowSpan(y);
                int baseIndex = y * TileHeightmapSize;
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    Vector3 normal = NormalizeSafe(normals[baseIndex + x]);
                    byte r = (byte)Math.Clamp((int)MathF.Round((normal.X * 0.5f + 0.5f) * 255f), 0, 255);
                    byte g = (byte)Math.Clamp((int)MathF.Round((normal.Y * 0.5f + 0.5f) * 255f), 0, 255);
                    byte b = (byte)Math.Clamp((int)MathF.Round((normal.Z * 0.5f + 0.5f) * 255f), 0, 255);
                    row[x] = new Rgba32(r, g, b, 255);
                }
            }
        });

        if (outputSize != TileHeightmapSize)
            image.Mutate(ctx => ctx.Resize(outputSize, outputSize, KnownResamplers.Bicubic));

        return image;
    }

    private static float[] NormalizeChunkHeights(float[] heights, bool isInterleaved)
    {
        if (isInterleaved || heights.Length < 145)
            return heights;

        var interleaved = new float[145];
        int destination = 0;
        for (int outerRow = 0; outerRow < 9; outerRow++)
        {
            int outerOffset = outerRow * 9;
            Array.Copy(heights, outerOffset, interleaved, destination, 9);
            destination += 9;

            if (outerRow >= 8)
                continue;

            int innerOffset = 81 + (outerRow * 8);
            Array.Copy(heights, innerOffset, interleaved, destination, 8);
            destination += 8;
        }

        return interleaved;
    }

    private static float[] ExtractChunkHeights(float[] tileHeightmap, int chunkX, int chunkY)
    {
        var chunkHeights = new float[145];
        int baseX = chunkX * HalfStepsPerChunk;
        int baseY = chunkY * HalfStepsPerChunk;

        for (int i = 0; i < 145; i++)
        {
            GetVertexPosition(i, out int row, out int col, out bool isInner);

            int sampleX = isInner ? (col * 2) + 1 : col * 2;
            int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;

            int px = Math.Clamp(baseX + sampleX, 0, TileHeightmapSize - 1);
            int py = Math.Clamp(baseY + sampleY, 0, TileHeightmapSize - 1);
            chunkHeights[i] = tileHeightmap[(py * TileHeightmapSize) + px];
        }

        return chunkHeights;
    }

    private static Vector3[] GenerateChunkNormals(int chunkX, int chunkY, int holeMask, float[] heights)
    {
        var positions = new Vector3[145];
        for (int i = 0; i < 145; i++)
            positions[i] = GetVertexWorldPosition(chunkX, chunkY, heights, i);

        int[] indices = BuildIndices(holeMask);
        var accumulated = new Vector3[145];

        for (int triangle = 0; triangle + 2 < indices.Length; triangle += 3)
        {
            int i0 = indices[triangle];
            int i1 = indices[triangle + 1];
            int i2 = indices[triangle + 2];

            Vector3 p0 = positions[i0];
            Vector3 p1 = positions[i1];
            Vector3 p2 = positions[i2];

            Vector3 edge1 = p1 - p0;
            Vector3 edge2 = p2 - p0;
            Vector3 normal = Vector3.Cross(edge1, edge2);
            if (normal.LengthSquared() < 1e-10f)
                continue;

            normal = Vector3.Normalize(normal);
            accumulated[i0] += normal;
            accumulated[i1] += normal;
            accumulated[i2] += normal;
        }

        var normals = new Vector3[145];
        for (int i = 0; i < 145; i++)
            normals[i] = NormalizeSafe(accumulated[i]);

        return normals;
    }

    private static Vector3 GetVertexWorldPosition(int chunkX, int chunkY, float[] heights, int index)
    {
        GetVertexPosition(index, out int row, out int col, out bool isInner);

        float cellSize = ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        float localX;
        float localY;
        if (!isInner)
        {
            localX = col * subCellSize;
            localY = (row / 2) * subCellSize;
        }
        else
        {
            localX = (col + 0.5f) * subCellSize;
            localY = ((row / 2) + 0.5f) * subCellSize;
        }

        float baseWorldX = -(chunkY * ChunkSize);
        float baseWorldY = -(chunkX * ChunkSize);
        float height = index < heights.Length ? heights[index] : 0f;
        return new Vector3(baseWorldX - localY, baseWorldY - localX, height);
    }

    private static void FillHeightGaps(float[] grid, int width, int height)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width) + x;
                if (!float.IsNaN(grid[index]))
                    continue;

                if ((x & 1) == 1 && (y & 1) == 0 && x > 0 && x + 1 < width)
                {
                    float left = grid[(y * width) + x - 1];
                    float right = grid[(y * width) + x + 1];
                    if (!float.IsNaN(left) && !float.IsNaN(right))
                        grid[index] = (left + right) * 0.5f;
                }
                else if ((x & 1) == 0 && (y & 1) == 1 && y > 0 && y + 1 < height)
                {
                    float up = grid[((y - 1) * width) + x];
                    float down = grid[((y + 1) * width) + x];
                    if (!float.IsNaN(up) && !float.IsNaN(down))
                        grid[index] = (up + down) * 0.5f;
                }
            }
        }

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width) + x;
                if (!float.IsNaN(grid[index]))
                    continue;

                if (TryFindNearestHeight(grid, width, height, x, y, out float nearest))
                    grid[index] = nearest;
                else
                    grid[index] = 0f;
            }
        }
    }

    private static void FillNormalGaps(Vector3[] grid, bool[] hasSample, int width, int height)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width) + x;
                if (hasSample[index])
                    continue;

                if ((x & 1) == 1 && (y & 1) == 0 && x > 0 && x + 1 < width)
                {
                    int leftIndex = (y * width) + x - 1;
                    int rightIndex = (y * width) + x + 1;
                    if (hasSample[leftIndex] && hasSample[rightIndex])
                    {
                        grid[index] = NormalizeSafe((grid[leftIndex] + grid[rightIndex]) * 0.5f);
                        hasSample[index] = true;
                    }
                }
                else if ((x & 1) == 0 && (y & 1) == 1 && y > 0 && y + 1 < height)
                {
                    int upIndex = ((y - 1) * width) + x;
                    int downIndex = ((y + 1) * width) + x;
                    if (hasSample[upIndex] && hasSample[downIndex])
                    {
                        grid[index] = NormalizeSafe((grid[upIndex] + grid[downIndex]) * 0.5f);
                        hasSample[index] = true;
                    }
                }
            }
        }

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width) + x;
                if (hasSample[index])
                    continue;

                if (TryFindNearestNormal(grid, hasSample, width, height, x, y, out Vector3 nearest))
                {
                    grid[index] = nearest;
                    hasSample[index] = true;
                }
                else
                {
                    grid[index] = Vector3.UnitZ;
                }
            }
        }
    }

    private static bool TryFindNearestHeight(float[] grid, int width, int height, int x, int y, out float value)
    {
        value = 0f;
        const int maxRadius = 24;

        for (int radius = 1; radius <= maxRadius; radius++)
        {
            int minY = Math.Max(0, y - radius);
            int maxY = Math.Min(height - 1, y + radius);
            int minX = Math.Max(0, x - radius);
            int maxX = Math.Min(width - 1, x + radius);

            for (int sampleX = minX; sampleX <= maxX; sampleX++)
            {
                float top = grid[(minY * width) + sampleX];
                if (!float.IsNaN(top))
                {
                    value = top;
                    return true;
                }

                float bottom = grid[(maxY * width) + sampleX];
                if (!float.IsNaN(bottom))
                {
                    value = bottom;
                    return true;
                }
            }

            for (int sampleY = minY + 1; sampleY <= maxY - 1; sampleY++)
            {
                float left = grid[(sampleY * width) + minX];
                if (!float.IsNaN(left))
                {
                    value = left;
                    return true;
                }

                float right = grid[(sampleY * width) + maxX];
                if (!float.IsNaN(right))
                {
                    value = right;
                    return true;
                }
            }
        }

        return false;
    }

    private static bool TryFindNearestNormal(Vector3[] grid, bool[] hasSample, int width, int height, int x, int y, out Vector3 value)
    {
        value = Vector3.UnitZ;
        const int maxRadius = 24;

        for (int radius = 1; radius <= maxRadius; radius++)
        {
            int minY = Math.Max(0, y - radius);
            int maxY = Math.Min(height - 1, y + radius);
            int minX = Math.Max(0, x - radius);
            int maxX = Math.Min(width - 1, x + radius);

            for (int sampleX = minX; sampleX <= maxX; sampleX++)
            {
                int topIndex = (minY * width) + sampleX;
                if (hasSample[topIndex])
                {
                    value = grid[topIndex];
                    return true;
                }

                int bottomIndex = (maxY * width) + sampleX;
                if (hasSample[bottomIndex])
                {
                    value = grid[bottomIndex];
                    return true;
                }
            }

            for (int sampleY = minY + 1; sampleY <= maxY - 1; sampleY++)
            {
                int leftIndex = (sampleY * width) + minX;
                if (hasSample[leftIndex])
                {
                    value = grid[leftIndex];
                    return true;
                }

                int rightIndex = (sampleY * width) + maxX;
                if (hasSample[rightIndex])
                {
                    value = grid[rightIndex];
                    return true;
                }
            }
        }

        return false;
    }

    private static Vector3 NormalizeSafe(Vector3 value)
    {
        return value.LengthSquared() > 1e-10f ? Vector3.Normalize(value) : Vector3.UnitZ;
    }

    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int rowIndex = 0; rowIndex < 17; rowIndex++)
        {
            int rowSize = rowIndex % 2 == 0 ? 9 : 8;
            if (remaining < rowSize)
            {
                row = rowIndex;
                col = remaining;
                isInner = rowIndex % 2 != 0;
                return;
            }

            remaining -= rowSize;
        }
    }

    private static int OuterIndex(int outerRow, int outerCol) => (outerRow * 17) + outerCol;

    private static int InnerIndex(int innerRow, int innerCol) => (innerRow * 17) + 9 + innerCol;

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
                    int holeBit = 1 << ((holeY * 4) + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                int topLeft = OuterIndex(cellY, cellX);
                int topRight = OuterIndex(cellY, cellX + 1);
                int bottomLeft = OuterIndex(cellY + 1, cellX);
                int bottomRight = OuterIndex(cellY + 1, cellX + 1);
                int center = InnerIndex(cellY, cellX);

                indices.Add(center);
                indices.Add(topRight);
                indices.Add(topLeft);

                indices.Add(center);
                indices.Add(bottomRight);
                indices.Add(topRight);

                indices.Add(center);
                indices.Add(bottomLeft);
                indices.Add(bottomRight);

                indices.Add(center);
                indices.Add(topLeft);
                indices.Add(bottomLeft);
            }
        }

        return indices.ToArray();
    }
}
