using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 209 Mechanism A — the MCLQ upsample blended non-present corners into the water surface,
/// dragging it toward a stored value that is not a surface height. Partial-presence quads only occur
/// at the edge of a water body, so the error concentrated on shorelines.
/// </summary>
public sealed class LiquidSurfaceInterpolationTests
{
    private const float WaterHeight = 100f;

    [Fact]
    public void AllCornersPresent_MatchesPlainBilinear()
    {
        // Open water must be untouched by the fix.
        Assert.True(LiquidSurfaceInterpolation.TryInterpolate(
            10f, 20f, 30f, 40f,
            true, true, true, true,
            fx: 0.5f, fy: 0.5f,
            out float height));

        Assert.Equal(25f, height, 4);
    }

    [Fact]
    public void NoCornersPresent_ProducesNoSurface()
    {
        Assert.False(LiquidSurfaceInterpolation.TryInterpolate(
            10f, 20f, 30f, 40f,
            false, false, false, false,
            fx: 0.5f, fy: 0.5f,
            out _));
    }

    [Fact]
    public void AbsentCornerHoldingZero_NoLongerDragsTheSurfaceDown()
    {
        // The defect, stated as a test. Three corners are water at 100; the fourth is land and holds
        // 0. Plain bilinear at the centre returns 75 — a 25-unit sag at the shoreline. Presence
        // weighting returns 100, because every corner that actually carries water is at 100.
        const float plainBilinearResult = 75f;

        Assert.True(LiquidSurfaceInterpolation.TryInterpolate(
            WaterHeight, WaterHeight, WaterHeight, 0f,
            true, true, true, false,
            fx: 0.5f, fy: 0.5f,
            out float height));

        Assert.Equal(WaterHeight, height, 4);
        Assert.NotEqual(plainBilinearResult, height, 4);
    }

    [Fact]
    public void SinglePresentCorner_ReturnsThatCornersHeight()
    {
        Assert.True(LiquidSurfaceInterpolation.TryInterpolate(
            WaterHeight, 0f, 0f, 0f,
            true, false, false, false,
            fx: 0.25f, fy: 0.25f,
            out float height));

        Assert.Equal(WaterHeight, height, 4);
    }

    [Fact]
    public void SampleOnAnAbsentCorner_StillProducesASurface()
    {
        // fx=1, fy=1 puts all bilinear weight on the bottom-right corner, which is absent. Without a
        // fallback the weight sum is zero and the pixel would be left as a one-pixel hole at the
        // quad edge — trading one shoreline artifact for another.
        Assert.True(LiquidSurfaceInterpolation.TryInterpolate(
            WaterHeight, WaterHeight, WaterHeight, 0f,
            true, true, true, false,
            fx: 1f, fy: 1f,
            out float height));

        Assert.Equal(WaterHeight, height, 4);
    }

    [Fact]
    public void PresentCornersAtDifferentHeights_StillInterpolateBetweenThemselves()
    {
        // The fix must not flatten a genuine slope: two present corners at different heights,
        // sampled midway between them, gives their mean.
        Assert.True(LiquidSurfaceInterpolation.TryInterpolate(
            10f, 20f, 0f, 0f,
            true, true, false, false,
            fx: 0.5f, fy: 0f,
            out float height));

        Assert.Equal(15f, height, 4);
    }

    [Theory]
    [InlineData(true, true, true, true, false)]
    [InlineData(false, false, false, false, false)]
    [InlineData(true, false, false, false, true)]
    [InlineData(true, true, true, false, true)]
    public void IsPartiallyPresent_IdentifiesTheEdgePopulation(
        bool a, bool b, bool c, bool d, bool expected)
        => Assert.Equal(expected, LiquidSurfaceInterpolation.IsPartiallyPresent(a, b, c, d));
}
