using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Pure terrain-chunk math shared by chunk editing, weak-signal restore and terrain queries: grids, normals, indices, cloning.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01); stateless, no host access.
/// </summary>
internal static class TerrainChunkMath
{

    internal static float[] RotateFloatGrid(float[] src, int w, int h, int rot, out int outW, out int outH)
    {
        rot = ((rot % 4) + 4) % 4;
        if (rot == 0)
        {
            outW = w;
            outH = h;
            return src;
        }

        if (rot == 2)
        {
            outW = w;
            outH = h;
            var dst = new float[w * h];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int sx = x;
                    int sy = y;
                    int dx = (w - 1 - sx);
                    int dy = (h - 1 - sy);
                    dst[dy * w + dx] = src[sy * w + sx];
                }
            }
            return dst;
        }

        outW = h;
        outH = w;
        var outGrid = new float[outW * outH];

        if (rot == 1)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int dx = (h - 1 - y);
                    int dy = x;
                    outGrid[dy * outW + dx] = src[y * w + x];
                }
            }
        }
        else
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int dx = y;
                    int dy = (w - 1 - x);
                    outGrid[dy * outW + dx] = src[y * w + x];
                }
            }
        }

        return outGrid;
    }

    internal static int RotateHoleMask(int holeMask, int rot)
    {
        rot = ((rot % 4) + 4) % 4;
        if (rot == 0 || holeMask == 0)
            return holeMask;

        int GetBit(int x, int y) => (holeMask >> (y * 4 + x)) & 1;
        int SetBit(int x, int y) => 1 << (y * 4 + x);

        int outMask = 0;
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                if (GetBit(x, y) == 0)
                    continue;

                int rx;
                int ry;
                switch (rot)
                {
                    case 1:
                        rx = 3 - y;
                        ry = x;
                        break;
                    case 2:
                        rx = 3 - x;
                        ry = 3 - y;
                        break;
                    case 3:
                        rx = y;
                        ry = 3 - x;
                        break;
                    default:
                        rx = x;
                        ry = y;
                        break;
                }

                outMask |= SetBit(rx, ry);
            }
        }

        return outMask;
    }

    internal static Vector3[] GenerateNormalsForChunk(Terrain.TerrainChunkData chunk, float[] heights, int holeMask)
    {
        var positions = new Vector3[145];
        for (int i = 0; i < 145; i++)
            positions[i] = GetChunkVertexWorldPosition(chunk, heights, i);

        var indices = BuildChunkIndices(holeMask);
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

    internal static Vector3 GetChunkVertexWorldPosition(Terrain.TerrainChunkData chunk, float[] heights, int index)
    {
        GetChunkVertexLocalPosition(index, out float x, out float y);

        float z = (index < heights.Length) ? heights[index] : 0f;
        float wx = chunk.WorldPosition.X - y;
        float wy = chunk.WorldPosition.Y - x;
        return new Vector3(wx, wy, z);
    }

    internal static void GetChunkVertexLocalPosition(int index, out float x, out float y)
    {
        GetChunkVertexPosition(index, out int row, out int col, out bool isInner);

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        if (!isInner)
        {
            x = col * subCellSize;
            y = (row / 2) * subCellSize;
            return;
        }

        x = (col + 0.5f) * subCellSize;
        y = (row / 2 + 0.5f) * subCellSize;
    }

    internal static int OuterIndex(int outerRow, int outerCol) => outerRow * 17 + outerCol;
    private static int InnerIndex(int innerRow, int innerCol) => innerRow * 17 + 9 + innerCol;

    internal static int[] BuildChunkIndices(int holeMask)
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

    internal static Terrain.TerrainChunkData CloneTerrainChunk(
        Terrain.TerrainChunkData source,
        float[]? heights = null,
        Vector3[]? normals = null,
        int? holeMask = null,
        WoWViewer.Terrain.TerrainLayer[]? layers = null,
        Dictionary<int, byte[]>? alphaMaps = null,
        byte[]? shadowMap = null,
        byte[]? mccvColors = null)
        => new()
        {
            McinIndex = source.McinIndex,
            TileX = source.TileX,
            TileY = source.TileY,
            ChunkX = source.ChunkX,
            ChunkY = source.ChunkY,
            Heights = heights ?? source.Heights,
            Normals = normals ?? source.Normals,
            HoleMask = holeMask ?? source.HoleMask,
            Layers = layers ?? source.Layers,
            AlphaMaps = alphaMaps ?? source.AlphaMaps,
            ShadowMap = shadowMap ?? source.ShadowMap,
            MccvColors = mccvColors ?? source.MccvColors,
            Liquid = source.Liquid,
            WorldPosition = source.WorldPosition,
            AreaId = source.AreaId,
            McnkFlags = source.McnkFlags,
            AlphaSourceFlags = source.AlphaSourceFlags,
            McrdReferences = source.McrdReferences?.ToArray() ?? Array.Empty<int>(),
            McrwReferences = source.McrwReferences?.ToArray() ?? Array.Empty<int>(),
        };

    internal static List<Terrain.TerrainChunkData> CloneTerrainChunkList(IReadOnlyList<Terrain.TerrainChunkData> chunks)
    {
        var cloned = new List<Terrain.TerrainChunkData>(chunks.Count);
        foreach (var chunk in chunks)
        {
            cloned.Add(CloneTerrainChunk(
                chunk,
                heights: chunk.Heights?.ToArray(),
                normals: chunk.Normals?.ToArray(),
                layers: chunk.Layers?.ToArray(),
                alphaMaps: CloneChunkAlphaMaps(chunk.AlphaMaps),
                shadowMap: chunk.ShadowMap?.ToArray(),
                mccvColors: chunk.MccvColors?.ToArray()));
        }

        return cloned;
    }

    private static Dictionary<int, byte[]> CloneChunkAlphaMaps(Dictionary<int, byte[]>? alphaMaps)
    {
        if (alphaMaps == null || alphaMaps.Count == 0)
            return new Dictionary<int, byte[]>();

        var cloned = new Dictionary<int, byte[]>(alphaMaps.Count);
        foreach (var entry in alphaMaps)
            cloned[entry.Key] = entry.Value?.ToArray() ?? Array.Empty<byte>();

        return cloned;
    }

    internal static float ComputeAverageHeight(float[] heights)
    {
        if (heights == null || heights.Length == 0)
            return 0f;
        double sum = 0;
        for (int i = 0; i < heights.Length; i++)
            sum += heights[i];
        return (float)(sum / heights.Length);
    }

    internal static float SampleHeightOuterGrid(Terrain.TerrainChunkData chunk, float localX, float localY)
    {
        if (chunk.Heights == null || chunk.Heights.Length < 145) return chunk.WorldPosition.Z;

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        Span<float> grid = stackalloc float[9 * 9];
        grid.Clear();

        for (int i = 0; i < 145; i++)
        {
            GetChunkVertexPosition(i, out int row, out int col, out bool isInner);
            if (isInner) continue;
            int gy = row / 2;
            if ((uint)gy >= 9u || (uint)col >= 9u) continue;
            grid[gy * 9 + col] = chunk.Heights[i];
        }

        float gx = localX / subCellSize;
        float gyf = localY / subCellSize;
        int ix = Math.Clamp((int)MathF.Floor(gx), 0, 7);
        int iy = Math.Clamp((int)MathF.Floor(gyf), 0, 7);
        float fx = Math.Clamp(gx - ix, 0f, 1f);
        float fy = Math.Clamp(gyf - iy, 0f, 1f);

        float h00 = grid[iy * 9 + ix];
        float h10 = grid[iy * 9 + (ix + 1)];
        float h01 = grid[(iy + 1) * 9 + ix];
        float h11 = grid[(iy + 1) * 9 + (ix + 1)];

        float h0 = h00 + (h10 - h00) * fx;
        float h1 = h01 + (h11 - h01) * fx;
        return h0 + (h1 - h0) * fy;
    }

    internal static void GetChunkVertexPosition(int index, out int row, out int col, out bool isInner)
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
                isInner = (r % 2 == 1);
                return;
            }
            remaining -= rowSize;
        }
    }

    internal static bool AreLayersCompatible(Terrain.TerrainLayer[] a, Terrain.TerrainLayer[] b)
    {
        if (a.Length != b.Length)
            return false;

        for (int i = 0; i < a.Length; i++)
        {
            if (a[i].TextureIndex != b[i].TextureIndex)
                return false;
        }

        return true;
    }

    internal static Dictionary<int, byte[]> CloneAlphaMaps(Dictionary<int, byte[]> maps)
    {
        var clone = new Dictionary<int, byte[]>(maps.Count);
        foreach (var (k, v) in maps)
            clone[k] = (byte[])v.Clone();
        return clone;
    }
}
