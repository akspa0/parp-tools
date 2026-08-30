using WowViewer.Core.Editor.Operations;
using WowViewer.Core.IO.Terrain;
using Xunit;

namespace WowViewer.Core.Editor.Tests.Operations;

public class TerrainStampOperationTests
{
    [Fact]
    public void CreateReverse_SwapsSnapshotsAndMaintainsReversibility()
    {
        var before = new List<ChunkTerrainSnapshot>
        {
            new() { ChunkIndex = 0, Heights = [0f, 0f, 0f], Normals = [0, 0, 1] }
        };
        var after = new List<ChunkTerrainSnapshot>
        {
            new() { ChunkIndex = 0, Heights = [5f, 5f, 5f], Normals = [0, 0, 1] }
        };

        var options = new TerrainStampOptions { CenterWorldX = 0f, CenterWorldY = 0f, Scale = 1.0f };

        var op = new TerrainStampOperation(
            "op-stamp-01",
            "World/Maps/TestMap/TestMap_30_30.adt",
            "hill_gentle_knoll_01",
            options,
            before,
            after);

        Assert.True(op.Undoable);
        Assert.Equal("terrain.stamp", op.OriginPluginId);
        Assert.Single(op.AffectedPaths);

        var reverse = (TerrainStampOperation)op.CreateReverse();
        Assert.True(reverse.Undoable);
        Assert.Equal(after, reverse.BeforeSnapshots);
        Assert.Equal(before, reverse.AfterSnapshots);
    }

    [Fact]
    public void StampHeightmap_AppliesGentleKnollWithFeathering()
    {
        TerrainBrushLibrary lib = CuratedTerrainBrushLibrary.Instance;
        Assert.True(lib.TryGetPaste("hill_gentle_knoll_01", out TerrainBrushPaste? hill));
        Assert.NotNull(hill);

        float[] existingHeights = new float[145]; // Flat base Z=0
        var options = new TerrainStampOptions
        {
            CenterWorldX = 16.66666f,
            CenterWorldY = 16.66666f,
            Scale = 1.0f,
            HeightMultiplier = 1.0f,
            FeatherRadiusMeters = 4.0f,
            BlendMode = TerrainStampBlendMode.Additive
        };

        float[] result = TerrainStampOperation.StampHeightmap(
            existingHeights,
            chunkOriginWorldX: 0f,
            chunkOriginWorldY: 0f,
            hill,
            options);

        Assert.Equal(145, result.Length);

        // Center vertex (vertex index 40 for 9x9 row 4 col 4) should be elevated near ~4.5m
        float centerElevation = result[40];
        Assert.InRange(centerElevation, 4.0f, 4.6f);

        // Outer corner (vertex index 0) should be close to 0 due to feathering
        Assert.Equal(0f, result[0]);
    }
}
