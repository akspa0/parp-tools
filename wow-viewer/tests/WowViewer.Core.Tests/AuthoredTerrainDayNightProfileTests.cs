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

    [Fact]
    public void EvaluateLightDirection_KeepsAFixedNorthWestBearingInsteadOfGoingVerticalAtNoon()
    {
        Vector3 noon = AuthoredTerrainDayNightProfile.EvaluateLightDirection(0.5f);
        Vector3 preNoon = AuthoredTerrainDayNightProfile.EvaluateLightDirection(11f / 24f);
        Vector3 postNoon = AuthoredTerrainDayNightProfile.EvaluateLightDirection(13f / 24f);

        // A previous implementation swept the horizontal bearing with time of day and collapsed to a
        // fully vertical, shadow-less sun exactly at solar noon (a real 0.5.3 minimap comparison
        // showed this washes out the hillshade instead of matching the client's persistent
        // bright-north/dark-south relief). The traced native ray holds a constant azimuth, so noon
        // must keep the same meaningful north-west horizontal bearing as every other time of day.
        Assert.True(noon.X > 0.2f);
        Assert.True(noon.Y > 0.2f);
        Assert.Equal(1f, noon.X / noon.Y, 4);
        Assert.Equal(noon.X / noon.Y, preNoon.X / preNoon.Y, 4);
        Assert.Equal(noon.X / noon.Y, postNoon.X / postNoon.Y, 4);
    }
}
