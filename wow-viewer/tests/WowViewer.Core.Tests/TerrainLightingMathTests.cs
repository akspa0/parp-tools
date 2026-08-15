using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class TerrainLightingMathTests
{
    [Fact]
    public void Evaluate_ClampsBackFacingLambertToAmbientOnly()
    {
        Vector3 result = TerrainLightingMath.Evaluate(
            -Vector3.UnitZ,
            Vector3.UnitZ,
            new Vector3(0.8f),
            new Vector3(0.2f),
            shadowMask: 0f);

        Assert.Equal(new Vector3(0.2f), result);
    }

    [Fact]
    public void Evaluate_McshModulatesDirectionalTermWithoutDarkeningAmbient()
    {
        Vector3 result = TerrainLightingMath.Evaluate(
            Vector3.UnitZ,
            Vector3.UnitZ,
            new Vector3(0.8f),
            new Vector3(0.2f),
            shadowMask: 1f,
            shadowStrength: 0.6f);

        Assert.Equal(0.52f, result.X, 5);
        Assert.Equal(0.52f, result.Y, 5);
        Assert.Equal(0.52f, result.Z, 5);
    }

    [Fact]
    public void ComputeFogRange_UsesNativeFoggedFractionSemantics()
    {
        (float start, float end) = TerrainLightingMath.ComputeFogRange(2000f, 0.25f);

        Assert.Equal(1500f, start);
        Assert.Equal(2000f, end);
    }

    [Fact]
    public void ComputeFogRange_FallsBackAndClampsMalformedValues()
    {
        Assert.Equal((1125f, 1500f), TerrainLightingMath.ComputeFogRange(float.NaN, float.NaN));
        Assert.Equal((0f, 2000f), TerrainLightingMath.ComputeFogRange(2000f, 2f));
    }

    [Fact]
    public void NormalizeFogRange_CollapsedOrReversedSource_UsesVisibleFallback()
    {
        Assert.Equal((200f, 1500f), TerrainLightingMath.NormalizeFogRange(0f, 0f));
        Assert.Equal((300f, 1800f), TerrainLightingMath.NormalizeFogRange(1800f, 300f, 300f, 1800f));
    }

    [Fact]
    public void NormalizeFogRange_ValidSource_ProtectsMinimumSpan()
    {
        (float start, float end) = TerrainLightingMath.NormalizeFogRange(1499.75f, 1500f);

        Assert.Equal(1499f, start);
        Assert.Equal(1500f, end);
    }

    [Fact]
    public void ComputeClientFogRange_ConvertsClassicFixedUnitsBeforeApplyingScalar()
    {
        (float start, float end) = TerrainLightingMath.ComputeClientFogRange(18000f, 0.25f);

        Assert.Equal(375f, start);
        Assert.Equal(500f, end);
    }

    [Fact]
    public void WorldTimeCycle_AdvancesOneFullDayInTwentyFourRealMinutes()
    {
        float advanced = WorldTimeCycle.AdvanceNormalized(0.5f, WorldTimeCycle.RealSecondsPerDay);

        Assert.Equal(0.5f, advanced, 5);
        Assert.Equal(1440, WorldTimeCycle.ToTimeUnits(advanced));
    }

    [Fact]
    public void WorldTimeCycle_ConvertsHalfDayToTheNativeNoonUnit()
    {
        float advanced = WorldTimeCycle.AdvanceNormalized(0f, WorldTimeCycle.RealSecondsPerDay / 2d);

        Assert.Equal(0.5f, advanced, 5);
        Assert.Equal(1440, WorldTimeCycle.ToTimeUnits(advanced));
        Assert.Equal(0.5f, WorldTimeCycle.FromTimeUnits(1440), 5);
    }

    [Fact]
    public void WorldTimeCycle_IgnoresInvalidOrNonPositiveElapsedTime()
    {
        Assert.Equal(0.25f, WorldTimeCycle.AdvanceNormalized(0.25f, 0d));
        Assert.Equal(0.25f, WorldTimeCycle.AdvanceNormalized(0.25f, double.NaN));
        Assert.Equal(0.25f, WorldTimeCycle.AdvanceNormalized(0.25f, -1d));
    }
}
