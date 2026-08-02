using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.Tests;

public sealed class MinimapEraProfileTests
{
    [Theory]
    [InlineData("0.5.3.3368", "alpha_0_5_3")]
    [InlineData("0_5_3_3368", "alpha_0_5_3")]
    [InlineData("0.6.0.3592", "beta1_0_6_0")]
    [InlineData("1.0.0.3368", "release_1_0_0")]
    [InlineData("3.3.5.12340", "release_1_0_0")]
    public void ResolveForBuild_MapsBuildsToTheirGenerationEra(string build, string expectedName)
    {
        MinimapEraProfile era = MinimapEraProfile.ResolveForBuild(build, out bool exact);

        Assert.True(exact);
        Assert.Equal(expectedName, era.Name);
    }

    /// <summary>
    /// An unrecognised build must be flagged, not silently handed Alpha's rules. Minimap generation
    /// changed across eras, so quietly defaulting would misattribute a corpus to the wrong era.
    /// </summary>
    [Theory]
    [InlineData("")]
    [InlineData(null)]
    [InlineData("weird-custom-build")]
    public void ResolveForBuild_FlagsAnUnrecognisedBuildInsteadOfSilentlyDefaulting(string? build)
    {
        MinimapEraProfile era = MinimapEraProfile.ResolveForBuild(build, out bool exact);

        Assert.False(exact);
        Assert.Same(MinimapEraProfile.Default, era);
    }

    /// <summary>
    /// Only the 1.0.0 era's solar model came out of a client trace. Alpha and Beta 1 inherit it,
    /// and that inheritance must stay visible so nobody reads it as a finding about those builds.
    /// </summary>
    [Fact]
    public void OnlyTheReleaseEraClaimsATracedSolarModel()
    {
        Assert.Equal(SolarModelProvenance.TracedFromClient, MinimapEraProfile.Release100.AzimuthProvenance);
        Assert.False(MinimapEraProfile.Release100.HasUnverifiedSolarModel);

        Assert.True(MinimapEraProfile.Alpha053.HasUnverifiedSolarModel);
        Assert.True(MinimapEraProfile.Beta060.HasUnverifiedSolarModel);
    }

    [Fact]
    public void RenderProfile_RecordsTheEraAndTheSolarModelsProvenance()
    {
        string profile = MinimapEraProfile.Alpha053.RenderProfile;

        Assert.Contains("alpha_0_5_3", profile, StringComparison.Ordinal);
        Assert.Contains("assumedfromotherbuild", profile, StringComparison.Ordinal);
    }

    [Fact]
    public void FixedModel_HoldsOneBearingAllDay()
    {
        MinimapEraProfile era = MinimapEraProfile.Alpha053 with { AzimuthModel = SolarAzimuthModel.Fixed };

        Assert.Equal(era.NoonAzimuthDegrees, era.ResolveAzimuthDegrees(0.25f), 3);
        Assert.Equal(era.NoonAzimuthDegrees, era.ResolveAzimuthDegrees(0.5f), 3);
        Assert.Equal(era.NoonAzimuthDegrees, era.ResolveAzimuthDegrees(0.75f), 3);
    }

    /// <summary>
    /// The sweep must rotate the bearing while keeping horizontal magnitude constant. A previous
    /// attempt scaled the horizontal bias by cos(sunAngle), which collapsed to a straight-overhead
    /// sun at solar noon and produced a symmetric shadow ring instead of a one-sided hillshade.
    /// </summary>
    [Fact]
    public void SweepModel_RotatesBearingWithoutCollapsingHorizontalMagnitudeAtNoon()
    {
        MinimapEraProfile era = MinimapEraProfile.Alpha053 with { AzimuthModel = SolarAzimuthModel.EastToWestSweep };

        float morning = era.ResolveAzimuthDegrees(0.25f);
        float noon = era.ResolveAzimuthDegrees(0.5f);
        float evening = era.ResolveAzimuthDegrees(0.75f);

        Assert.Equal(era.NoonAzimuthDegrees, noon, 3);
        Assert.Equal(90f, noon - morning, 3);
        Assert.Equal(90f, evening - noon, 3);

        // Horizontal magnitude must stay constant across the day, including at noon.
        foreach (float gameTime in new[] { 0.25f, 0.4f, 0.5f, 0.6f, 0.75f })
        {
            Vector3 direction = era.ResolveLightDirection(gameTime);
            float horizontal = new Vector2(direction.X, direction.Y).Length();
            Assert.True(horizontal > 0.05f, $"Horizontal bias collapsed at gameTime {gameTime} (got {horizontal}).");
        }
    }

    [Fact]
    public void Evaluate_WithoutAnAzimuthKeepsTheTracedBearingExactly()
    {
        // Every pre-existing caller must be untouched by the azimuth parameterisation.
        Vector3 traced = TerrainSolarDirection.Evaluate(0.5f);
        Vector3 explicitTraced = TerrainSolarDirection.Evaluate(0.5f, TerrainSolarDirection.TracedSourceAzimuthDegrees);

        Assert.Equal(traced.X, explicitTraced.X, 5);
        Assert.Equal(traced.Y, explicitTraced.Y, 5);
        Assert.Equal(traced.Z, explicitTraced.Z, 5);

        // Traced bearing is north-west: positive on both X (North) and Y (West).
        Assert.True(traced.X > 0f);
        Assert.True(traced.Y > 0f);
    }

    [Theory]
    [InlineData(TerrainSolarDirection.EastAzimuthDegrees, false, true)]   // East = -Y
    [InlineData(TerrainSolarDirection.WestAzimuthDegrees, false, false)]  // West = +Y
    [InlineData(0f, true, false)]                                          // North = +X
    public void Evaluate_PlacesTheSourceOnTheExpectedCompassAxis(
        float azimuth,
        bool expectNorth,
        bool expectEast)
    {
        Vector3 direction = TerrainSolarDirection.Evaluate(0.5f, azimuth);

        if (expectNorth)
            Assert.True(direction.X > 0.1f, $"Expected a northward (+X) source, got X={direction.X}.");
        if (expectEast)
            Assert.True(direction.Y < -0.1f, $"East is -Y in world axes, got Y={direction.Y}.");
        else if (!expectNorth)
            Assert.True(direction.Y > 0.1f, $"West is +Y in world axes, got Y={direction.Y}.");
    }

    [Theory]
    [InlineData(10f, 350f, 20f)]
    [InlineData(45f, 225f, 180f)]
    [InlineData(0f, 90f, 90f)]
    public void AngularSeparation_WrapsAroundTheCompass(float first, float second, float expected)
    {
        Assert.Equal(expected, MinimapShadingMatch.AngularSeparationDegrees(first, second), 3);
    }
}
