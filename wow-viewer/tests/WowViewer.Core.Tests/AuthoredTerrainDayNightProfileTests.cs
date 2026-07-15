using System.Numerics;
using WowViewer.Core.Renderer.Terrain;

namespace WowViewer.Core.Tests;

public sealed class AuthoredTerrainDayNightProfileTests
{
    [Fact]
    public void Evaluate_LabelsAuthoredColorsAndWrapsGameTime()
    {
        TerrainLightingSample sample = AuthoredTerrainDayNightProfile.Evaluate(1.35f);

        Assert.Equal(AuthoredTerrainDayNightProfile.ProfileRevision, sample.ProfileRevision);
        Assert.Equal(AuthoredTerrainDayNightProfile.EvidenceState, sample.EvidenceState);
        Assert.Equal(0.35f, sample.GameTime, 5);
        Assert.Equal(1f, sample.LightDirection.Length(), 5);
    }

    [Fact]
    public void Shade_UsesOneSidedLambertAndMcshOnlyModulatesDirectionalLight()
    {
        TerrainLightingSample sample = AuthoredTerrainDayNightProfile.Evaluate(0.5f);
        Vector3 albedo = Vector3.One;

        Vector3 facing = AuthoredTerrainDayNightProfile.Shade(albedo, sample.LightDirection, 0f, sample);
        Vector3 facingShadowed = AuthoredTerrainDayNightProfile.Shade(albedo, sample.LightDirection, 1f, sample);
        Vector3 backFacing = AuthoredTerrainDayNightProfile.Shade(albedo, -sample.LightDirection, 0f, sample);

        Assert.True(facing.X > facingShadowed.X);
        Assert.True(facingShadowed.X > backFacing.X);
        Assert.Equal(sample.AmbientColor.X * sample.AmbientIntensity, backFacing.X, 5);
        Assert.Equal(sample.AmbientColor.Y * sample.AmbientIntensity, backFacing.Y, 5);
        Assert.Equal(sample.AmbientColor.Z * sample.AmbientIntensity, backFacing.Z, 5);
    }

    [Fact]
    public void Evaluate_ChangesWhiteLightTintAcrossTimeOfDay()
    {
        TerrainLightingSample night = AuthoredTerrainDayNightProfile.Evaluate(0f);
        TerrainLightingSample noon = AuthoredTerrainDayNightProfile.Evaluate(0.5f);

        Assert.NotEqual(night.DirectionalColor, noon.DirectionalColor);
        Assert.NotEqual(night.AmbientColor, noon.AmbientColor);
        Assert.True(night.DirectionalColor.Z > night.DirectionalColor.X);
        Assert.True(noon.DirectionalColor.X > noon.DirectionalColor.Z);
    }
}
