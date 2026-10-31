using System;
using System.Collections.Generic;
using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests.Maps;

public sealed class MccvHoleDetectionTests
{
    private static byte[] CreateUniformMccv(byte b, byte g, byte r, byte a = 255)
    {
        var data = new byte[MccvHoleDetection.ExpectedChunkMccvBytes];
        for (int i = 0; i < MccvHoleDetection.VerticesPerChunk; i++)
        {
            data[i * 4 + 0] = b;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = r;
            data[i * 4 + 3] = a;
        }
        return data;
    }

    [Fact]
    public void IsSolidBurgundyChunk_AcceptsSolidBurgundyUndersideColor()
    {
        // Dark burgundy: R=110, G=20, B=20
        byte[] burgundy = CreateUniformMccv(b: 20, g: 20, r: 110);
        Assert.True(MccvHoleDetection.IsSolidBurgundyChunk(burgundy));
    }

    [Fact]
    public void IsSolidBurgundyChunk_RejectsGrayOrWhiteChunk()
    {
        // Gray / default lighting: R=128, G=128, B=128
        byte[] gray = CreateUniformMccv(b: 128, g: 128, r: 128);
        Assert.False(MccvHoleDetection.IsSolidBurgundyChunk(gray));
    }

    [Fact]
    public void IsSolidBurgundyChunk_RejectsHighVarianceTexturedChunk()
    {
        // Noisy pattern alternating between high red and blue/green
        var noisy = new byte[MccvHoleDetection.ExpectedChunkMccvBytes];
        for (int i = 0; i < MccvHoleDetection.VerticesPerChunk; i++)
        {
            if (i % 2 == 0)
            {
                noisy[i * 4 + 0] = 20;
                noisy[i * 4 + 1] = 20;
                noisy[i * 4 + 2] = 110;
            }
            else
            {
                noisy[i * 4 + 0] = 80;
                noisy[i * 4 + 1] = 80;
                noisy[i * 4 + 2] = 30;
            }
            noisy[i * 4 + 3] = 255;
        }

        Assert.False(MccvHoleDetection.IsSolidBurgundyChunk(noisy));
    }

    [Fact]
    public void FilterContiguousClusters_RequiresAtLeastTwoAdjacentChunks()
    {
        var candidates = new List<(int, int)>
        {
            // Cluster of 2 connected chunks: (2, 2) and (2, 3)
            (2, 2), (2, 3),

            // Isolated single chunk at (10, 10)
            (10, 10),

            // Another isolated single chunk at (14, 2)
            (14, 2)
        };

        var admitted = MccvHoleDetection.FilterContiguousClusters(candidates, minClusterSize: 2);

        Assert.Equal(2, admitted.Count);
        Assert.Contains((2, 2), admitted);
        Assert.Contains((2, 3), admitted);
        Assert.DoesNotContain((10, 10), admitted);
        Assert.DoesNotContain((14, 2), admitted);
    }

    [Fact]
    public void ApplyBurgundyHolesToChunks_AppliesFullHoleMaskToConnectedClusters()
    {
        var chunk1 = new TerrainChunkData { ChunkX = 4, ChunkY = 5, MccvColors = CreateUniformMccv(20, 20, 110) };
        var chunk2 = new TerrainChunkData { ChunkX = 4, ChunkY = 6, MccvColors = CreateUniformMccv(20, 20, 110) };
        var isolated = new TerrainChunkData { ChunkX = 8, ChunkY = 8, MccvColors = CreateUniformMccv(20, 20, 110) };
        var nonBurgundy = new TerrainChunkData { ChunkX = 4, ChunkY = 7, MccvColors = CreateUniformMccv(100, 100, 100) };

        var chunks = new List<TerrainChunkData> { chunk1, chunk2, isolated, nonBurgundy };

        List<TerrainChunkData> holed = MccvHoleDetection.ApplyBurgundyHolesToChunks(chunks, minClusterSize: 2);

        Assert.Equal(2, holed.Count);
        Assert.Contains(chunk1, holed);
        Assert.Contains(chunk2, holed);

        // Verify full hole masks
        Assert.Equal(0xFFFF, chunk1.HoleMask);
        Assert.Equal(0xFFFFFFFFFFFFFFFFUL, chunk1.HoleMask64);
        Assert.Equal(0xFFFF, chunk2.HoleMask);
        Assert.Equal(0xFFFFFFFFFFFFFFFFUL, chunk2.HoleMask64);

        // Verify isolated chunk was NOT holed
        Assert.Equal(0, isolated.HoleMask);
        Assert.Equal(0UL, isolated.HoleMask64);
    }
}

