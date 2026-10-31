using System;
using System.Collections.Generic;

namespace WowViewer.Core.Maps;

/// <summary>
/// Detects solid burgundy-red underside fog vertex colors in MCCV data and derives
/// terrain chunk hole flags for clusters of 2 or more contiguous chunks (Spec 287 Phase 2).
/// </summary>
public static class MccvHoleDetection
{
    public const int VerticesPerChunk = 145;
    public const int BytesPerVertex = 4; // BGRA format
    public const int ExpectedChunkMccvBytes = VerticesPerChunk * BytesPerVertex; // 580 bytes

    /// <summary>
    /// Checks whether an individual BGRA vertex color matches the burgundy underside fog profile:
    /// dominant Red (R >= 70), low Green and Blue (G <= 35, B <= 35), with Red significantly higher than G + B.
    /// </summary>
    public static bool IsBurgundyVertex(byte b, byte g, byte r, byte a)
    {
        return r >= 70 && g <= 35 && b <= 35 && r > (g + b) * 1.3f;
    }

    /// <summary>
    /// Evaluates all 145 vertices of a chunk to verify if the chunk is uniformly solid burgundy red.
    /// Excludes textured hills or slope-shaded surfaces where color variance is high or red is blended.
    /// </summary>
    public static bool IsSolidBurgundyChunk(byte[]? mccvColors)
    {
        if (mccvColors == null || mccvColors.Length < ExpectedChunkMccvBytes)
            return false;

        float sumR = 0f, sumG = 0f, sumB = 0f;
        int burgundyVertexCount = 0;

        for (int i = 0; i < VerticesPerChunk; i++)
        {
            int offset = i * BytesPerVertex;
            byte b = mccvColors[offset + 0];
            byte g = mccvColors[offset + 1];
            byte r = mccvColors[offset + 2];
            byte a = mccvColors[offset + 3];

            sumR += r;
            sumG += g;
            sumB += b;

            if (IsBurgundyVertex(b, g, r, a))
                burgundyVertexCount++;
        }

        // At least 88% of vertices must individually qualify as burgundy
        if (burgundyVertexCount < 128)
            return false;

        float avgR = sumR / VerticesPerChunk;
        float avgG = sumG / VerticesPerChunk;
        float avgB = sumB / VerticesPerChunk;

        if (avgR < 70f || avgG > 35f || avgB > 35f || avgR <= (avgG + avgB) * 1.3f)
            return false;

        // Check standard deviation to ensure solid, untextured color
        float varR = 0f, varG = 0f, varB = 0f;
        for (int i = 0; i < VerticesPerChunk; i++)
        {
            int offset = i * BytesPerVertex;
            float dr = mccvColors[offset + 2] - avgR;
            float dg = mccvColors[offset + 1] - avgG;
            float db = mccvColors[offset + 0] - avgB;

            varR += dr * dr;
            varG += dg * dg;
            varB += db * db;
        }

        float stdR = MathF.Sqrt(varR / VerticesPerChunk);
        float stdG = MathF.Sqrt(varG / VerticesPerChunk);
        float stdB = MathF.Sqrt(varB / VerticesPerChunk);

        // Solid underside fog has low variance (<= 18 for R, <= 15 for G and B)
        return stdR <= 18f && stdG <= 15f && stdB <= 15f;
    }

    /// <summary>
    /// Clusters candidate chunk coordinates into connected components on the 16x16 tile grid.
    /// Rejects isolated single chunks, returning only chunks belonging to clusters of size >= 2.
    /// </summary>
    public static HashSet<(int ChunkX, int ChunkY)> FilterContiguousClusters(
        IReadOnlyCollection<(int ChunkX, int ChunkY)> candidateCoords,
        int minClusterSize = 2)
    {
        ArgumentNullException.ThrowIfNull(candidateCoords);
        var coordSet = new HashSet<(int ChunkX, int ChunkY)>(candidateCoords);
        var visited = new HashSet<(int ChunkX, int ChunkY)>();
        var admitted = new HashSet<(int ChunkX, int ChunkY)>();

        int[] dx = [1, -1, 0, 0];
        int[] dy = [0, 0, 1, -1];

        foreach (var coord in candidateCoords)
        {
            if (visited.Contains(coord))
                continue;

            var cluster = new List<(int ChunkX, int ChunkY)>();
            var queue = new Queue<(int ChunkX, int ChunkY)>();
            queue.Enqueue(coord);
            visited.Add(coord);

            while (queue.Count > 0)
            {
                var curr = queue.Dequeue();
                cluster.Add(curr);

                for (int d = 0; d < 4; d++)
                {
                    var neighbor = (curr.ChunkX + dx[d], curr.ChunkY + dy[d]);
                    if (coordSet.Contains(neighbor) && visited.Add(neighbor))
                    {
                        queue.Enqueue(neighbor);
                    }
                }
            }

            if (cluster.Count >= minClusterSize)
            {
                for (int i = 0; i < cluster.Count; i++)
                    admitted.Add(cluster[i]);
            }
        }

        return admitted;
    }

    /// <summary>
    /// Analyzes chunks across a tile, detecting solid burgundy MCCV colors and setting
    /// HoleMask = 0xFFFF and HoleMask64 = 0xFFFFFFFFFFFFFFFF on clusters of >= 2 contiguous chunks.
    /// Returns the list of chunks whose hole masks were updated.
    /// </summary>
    public static List<TerrainChunkData> ApplyBurgundyHolesToChunks(
        IEnumerable<TerrainChunkData> chunks,
        int minClusterSize = 2)
    {
        ArgumentNullException.ThrowIfNull(chunks);

        var chunkMap = new Dictionary<(int, int), TerrainChunkData>();
        var candidates = new List<(int ChunkX, int ChunkY)>();

        foreach (var chunk in chunks)
        {
            chunkMap[(chunk.ChunkX, chunk.ChunkY)] = chunk;
            if (IsSolidBurgundyChunk(chunk.MccvColors))
            {
                candidates.Add((chunk.ChunkX, chunk.ChunkY));
            }
        }

        var admittedCoords = FilterContiguousClusters(candidates, minClusterSize);
        var holedChunks = new List<TerrainChunkData>(admittedCoords.Count);

        foreach (var coord in admittedCoords)
        {
            if (chunkMap.TryGetValue(coord, out var chunk))
            {
                chunk.HoleMask = 0xFFFF;
                chunk.HoleMask64 = 0xFFFFFFFFFFFFFFFFUL;
                holedChunks.Add(chunk);
            }
        }

        return holedChunks;
    }
}

