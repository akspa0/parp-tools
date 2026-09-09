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
            MirrorHorizontal = true,
            UsePlacedTilesOnly = true,
            Locked = true,
            FootprintColorIndex = 3,
        };
        layer.TilePlacements.Add(new PhaseTilePlacement(32, 32, 10, 12, Locked: true));

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
            Assert.True(roundTripped.MirrorHorizontal);
            Assert.False(roundTripped.MirrorVertical);
            Assert.True(roundTripped.UsePlacedTilesOnly);
            Assert.True(roundTripped.Locked);
            Assert.Equal(3, roundTripped.FootprintColorIndex);
            PhaseTilePlacement placement = Assert.Single(roundTripped.TilePlacements);
            Assert.Equal((32, 32, 10, 12), (placement.DonorTileX, placement.DonorTileY, placement.TargetTileX, placement.TargetTileY));
            Assert.True(placement.Locked);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
