using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 232 Phase 2 (FR-2): the layer-project file round-trips every transform, channel, and
/// lock field — a saved project must reload bit-identical.
/// </summary>
public sealed class PhaseLayerProjectFileTests
{
    [Fact]
    public void RoundTrip_PreservesAllFields()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = "DeadminesInstance",
            Enabled = true,
            Channels = PhaseDataChannel.Normals | PhaseDataChannel.TextureLayers | PhaseDataChannel.WorldObjects,
            OnlyTakeWhatThePhaseCarries = true,
            TileOffsetX = 19,
            TileOffsetY = -3,
            CellOffsetX = 2,
            CellOffsetY = -5,
            RotationDegrees = 90f,
            RotationOriginTileX = 32.5f,
            RotationOriginTileY = 32.5f,
            GroupTransform = new DonorGroupTransform(17.5f, 2f, -1f, 32.5f, 32.5f),
            MirrorHorizontal = true,
            UsePlacedTilesOnly = true,
            Locked = true,
            FootprintColorIndex = 3,
        };
        PhaseChunkMask mask = PhaseChunkMask.All
            .With(0, 0, false)
            .With(15, 15, false)
            .With(7, 8, false);
        layer.TilePlacements.Add(new PhaseTilePlacement(32, 32, 10, 12, Locked: true, ChunkMask: mask));

        PhaseLayerProjectFile project = PhaseLayerProjectFile.FromLayers("Azeroth", [layer], PhaseDataChannel.Terrain);
        string path = Path.Combine(Path.GetTempPath(), $"carto-{Guid.NewGuid():N}.json");
        try
        {
            project.Save(path);
            PhaseLayerProjectFile loaded = PhaseLayerProjectFile.Load(path);

            Assert.Equal("Azeroth", loaded.BaseMap);
            Assert.Equal(PhaseDataChannel.Terrain, loaded.BaseChannelKeep);
            PhaseLayerSettings roundTripped = Assert.Single(loaded.ToLayers());

            Assert.Equal(layer.MapName, roundTripped.MapName);
            Assert.True(roundTripped.Enabled);
            Assert.Equal(layer.Channels, roundTripped.Channels);
            Assert.True(roundTripped.OnlyTakeWhatThePhaseCarries);
            Assert.Equal((19, -3), (roundTripped.TileOffsetX, roundTripped.TileOffsetY));
            Assert.Equal((2, -5), (roundTripped.CellOffsetX, roundTripped.CellOffsetY));
            Assert.Equal(90f, roundTripped.RotationDegrees);
            Assert.Equal((32.5f, 32.5f), (roundTripped.RotationOriginTileX, roundTripped.RotationOriginTileY));
            Assert.Equal(layer.GroupTransform, roundTripped.GroupTransform);
            Assert.True(roundTripped.MirrorHorizontal);
            Assert.False(roundTripped.MirrorVertical);
            Assert.True(roundTripped.UsePlacedTilesOnly);
            Assert.True(roundTripped.Locked);
            Assert.Equal(3, roundTripped.FootprintColorIndex);
            PhaseTilePlacement placement = Assert.Single(roundTripped.TilePlacements);
            Assert.Equal((32, 32, 10, 12), (placement.DonorTileX, placement.DonorTileY, placement.TargetTileX, placement.TargetTileY));
            Assert.True(placement.Locked);
            Assert.NotNull(placement.ChunkMask);
            Assert.False(placement.ChunkMask!.Value.IsIncluded(0, 0));
            Assert.False(placement.ChunkMask.Value.IsIncluded(15, 15));
            Assert.False(placement.ChunkMask.Value.IsIncluded(7, 8));
            Assert.True(placement.ChunkMask.Value.IsIncluded(1, 1));
            Assert.Equal(253, placement.ChunkMask.Value.IncludedCount);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ChunkMask_AllNoneAndHexRoundTrip()
    {
        Assert.Equal(256, PhaseChunkMask.All.IncludedCount);
        Assert.Equal(0, PhaseChunkMask.None.IncludedCount);
        Assert.True(PhaseChunkMask.All.IsIncluded(0, 0));
        Assert.True(PhaseChunkMask.All.IsIncluded(15, 15));
        Assert.False(PhaseChunkMask.None.IsIncluded(0, 0));

        PhaseChunkMask mask = PhaseChunkMask.All.With(3, 4, false);
        PhaseChunkMask parsed = PhaseChunkMask.FromHex(mask.ToHex());
        Assert.Equal(mask, parsed);
        Assert.False(parsed.IsIncluded(3, 4));
        Assert.True(parsed.IsIncluded(4, 3));
    }

    [Fact]
    public void ChunkMask_InverseTransformMapsTargetSlotBackToDonorSlot()
    {
        PhaseChunkMask mask = PhaseChunkMask.All.With(0, 0, false);

        // Rotate90CW maps donor (0,0) to target (15,0); the inverse must find the excluded donor slot.
        Assert.False(mask.IsIncludedAfterTransforms(15, 0, [TileTransformKind.Rotate90CW]));
        Assert.True(mask.IsIncludedAfterTransforms(0, 0, [TileTransformKind.Rotate90CW]));

        // MirrorH maps donor (0,0) to target (15,0).
        Assert.False(mask.IsIncludedAfterTransforms(15, 0, [TileTransformKind.MirrorH]));
        Assert.True(mask.IsIncludedAfterTransforms(0, 0, [TileTransformKind.MirrorH]));
    }
}
