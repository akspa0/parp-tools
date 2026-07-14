using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Tests;

public sealed class WorldTerrainVisualSnapshotBuilderTests
{
    [Fact]
    public void Build_TileWithoutHeightSamples_ProducesDeterministicEmptySnapshot()
    {
        WorldTerrainTileData terrainTile = new("synthetic_0_0.adt", WowViewer.Core.Maps.MapFileKind.Adt, Array.Empty<WorldTerrainChunkData>(), null);

        WorldTerrainVisualSnapshot snapshot = WorldTerrainVisualSnapshotBuilder.Build(terrainTile);

        Assert.Equal(257, snapshot.Width);
        Assert.Equal(257, snapshot.Height);
        Assert.Equal(0, snapshot.SampledPixelCount);
        Assert.Equal(64, snapshot.VisualHash.Length);
    }
}