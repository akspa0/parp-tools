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
    public void ComputeClientFogRange_ConvertsClassicFixedUnitsBeforeApplyingScalar()
    {
        (float start, float end) = TerrainLightingMath.ComputeClientFogRange(18000f, 0.25f);

        Assert.Equal(375f, start);
        Assert.Equal(500f, end);
    }
}
