using System.Numerics;
using WowViewer.Core.Editor.Operations;
using Xunit;

namespace WowViewer.Core.Editor.Tests.Operations;

public sealed class ChunkTranspositionServiceTests
{
    [Fact]
    public void ExtractPayload_ComputesRelativeOriginsCorrectly()
    {
        var selection = new List<GlobalChunkCoordinate>
        {
            new(100, 200),
            new(101, 200),
            new(100, 201),
            new(101, 201),
        };

        var payload = ChunkTranspositionService.ExtractPayload(selection, coord =>
        {
            return new TransposedChunkRecord
            {
                Heights = new float[145],
                HoleMask = 0,
            };
        });

        Assert.Equal(new GlobalChunkCoordinate(100, 200), payload.Origin);
        Assert.Equal(2, payload.WidthInChunks);
        Assert.Equal(2, payload.HeightInChunks);
        Assert.Equal(4, payload.Chunks.Count);

        var c00 = payload.Chunks.First(c => c.RelativeGx == 0 && c.RelativeGy == 0);
        Assert.NotNull(c00);
        var c11 = payload.Chunks.First(c => c.RelativeGx == 1 && c.RelativeGy == 1);
        Assert.NotNull(c11);
    }

    [Fact]
    public void TransformPayload_Rotation90_SwapsDimensionsAndCoordinates()
    {
        var selection = new List<GlobalChunkCoordinate>
        {
            new(10, 10),
            new(11, 10),
            new(12, 10), // 3x1 chunk strip
        };

        var payload = ChunkTranspositionService.ExtractPayload(selection, coord => new TransposedChunkRecord());
        Assert.Equal(3, payload.WidthInChunks);
        Assert.Equal(1, payload.HeightInChunks);

        var options = new ChunkTranspositionOptions { RotationDegrees = 90 };
        var rotated = ChunkTranspositionService.TransformPayload(payload, options);

        Assert.Equal(1, rotated.WidthInChunks);
        Assert.Equal(3, rotated.HeightInChunks);
        Assert.Equal(3, rotated.Chunks.Count);
    }

    [Fact]
    public void TransformPayload_HeightOffset_OffsetsHeightsAndPlacements()
    {
        var payload = new ChunkTranspositionPayload
        {
            WidthInChunks = 1,
            HeightInChunks = 1,
            Origin = new GlobalChunkCoordinate(0, 0),
        };

        var heights = new float[145];
        Array.Fill(heights, 10f);

        payload.Chunks.Add(new TransposedChunkRecord
        {
            RelativeGx = 0,
            RelativeGy = 0,
            Heights = heights,
        });

        payload.Placements.Add(new TransposedObjectPlacement
        {
            AssetPath = "World\\Generic\\Doodad\\Tree.m2",
            RelativePosition = new Vector3(100f, 200f, 15f),
        });

        var options = new ChunkTranspositionOptions { HeightOffset = 25f };
        var transformed = ChunkTranspositionService.TransformPayload(payload, options);

        Assert.Equal(35f, transformed.Chunks[0].Heights![0]);
        Assert.Equal(40f, transformed.Placements[0].RelativePosition.Z);
    }
}
