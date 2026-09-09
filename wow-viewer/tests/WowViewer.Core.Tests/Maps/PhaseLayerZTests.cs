using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Operator directive 2026-09-09: per-layer world-Z offset/scale persists with the layer project
/// (the Teldrassil-on-Kalimdor raise-to-fit case). The viewer-side affine application over chunk
/// heights/liquid/placements is build-verified in WoWViewer.Terrain.PhaseLayerZ; the Core-side
/// contract is the round-trip below.
/// </summary>
public sealed class PhaseLayerZTests
{
    [Fact]
    public void LayerProjectFile_RoundTripsZSettings()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = "Donor",
            ZOffset = -17.5f,
            ZScale = 1.25f,
        };

        PhaseLayerProjectEntry entry = PhaseLayerProjectEntry.FromLayer(layer);
        PhaseLayerSettings roundTripped = entry.ToLayer();

        Assert.Equal(-17.5f, roundTripped.ZOffset);
        Assert.Equal(1.25f, roundTripped.ZScale);
    }

    [Fact]
    public void LayerDefaults_AreIdentityZ()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };

        Assert.Equal(0f, layer.ZOffset);
        Assert.Equal(1f, layer.ZScale);
    }

    [Fact]
    public void Clone_CarriesZSettings()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", ZOffset = 30f, ZScale = 2f };

        PhaseLayerSettings clone = layer.Clone();

        Assert.Equal(30f, clone.ZOffset);
        Assert.Equal(2f, clone.ZScale);
    }
}
