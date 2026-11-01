using System;
using System.Numerics;
using WowViewer.Core.PM4.Services;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4ColorCodingServiceTests
{
    private readonly Pm4ColorCodingEngine _engine = Pm4ColorCodingEngine.Instance;

    [Fact]
    public void Doodads_WithZeroCk24_ProduceDistinctInstanceColors()
    {
        // Two adjacent doodad instances on tile (35, 38) where Ck24 is 0 (doodad collision hulls)
        Vector3 anchor1 = new(100f, 200f, 50f);
        Vector3 anchor2 = new(115f, 210f, 52f);

        Vector3 color1 = _engine.ComputeInstanceColor(0u, 0, 35, 38, anchor1, 14);
        Vector3 color2 = _engine.ComputeInstanceColor(0u, 1, 35, 38, anchor2, 18);

        // Neither should collapse to zero
        Assert.True(color1.Length() > 0.1f);
        Assert.True(color2.Length() > 0.1f);

        // Crucial test: They must NOT share the identical mono-red hue
        float colorDelta = Vector3.Distance(color1, color2);
        Assert.True(colorDelta > 0.05f, $"Adjacent doodads should have distinct colors, but delta was {colorDelta}");
    }

    [Fact]
    public void ElevationRamp_ProducesContinuousGradientAcrossAltitude()
    {
        float minZ = 0f;
        float maxZ = 200f;

        Vector3 cLow = _engine.ComputeElevationColor(10f, minZ, maxZ);
        Vector3 cMid = _engine.ComputeElevationColor(100f, minZ, maxZ);
        Vector3 cHigh = _engine.ComputeElevationColor(190f, minZ, maxZ);

        Assert.True(Vector3.Distance(cLow, cMid) > 0.1f);
        Assert.True(Vector3.Distance(cMid, cHigh) > 0.1f);
        Assert.True(Vector3.Distance(cLow, cHigh) > 0.2f);
    }

    [Fact]
    public void SemanticCategory_SeparatesDoodadFromWmoFloors()
    {
        Vector3 doodadColor = _engine.ComputeSemanticCategoryColor(0x03, 0);
        Vector3 floorLowestColor = _engine.ComputeSemanticCategoryColor(0x10, 0);
        Vector3 floorUpperColor = _engine.ComputeSemanticCategoryColor(0x13, 0);

        Assert.True(Vector3.Distance(doodadColor, floorLowestColor) > 0.3f);
        Assert.True(Vector3.Distance(floorLowestColor, floorUpperColor) > 0.2f);
    }

    [Fact]
    public void MatchStatus_SeparatesMatchedFromUnmatched()
    {
        Vector3 matched = _engine.ComputeMatchStatusColor(isMatched: true, isDuplicated: false, ck24Type: 0x03);
        Vector3 unmatched = _engine.ComputeMatchStatusColor(isMatched: false, isDuplicated: false, ck24Type: 0x03);
        Vector3 duplicated = _engine.ComputeMatchStatusColor(isMatched: false, isDuplicated: true, ck24Type: 0x03);

        Assert.True(Vector3.Distance(matched, unmatched) > 0.3f);
        Assert.True(Vector3.Distance(matched, duplicated) > 0.3f);
    }
}
