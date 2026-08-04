using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

/// <summary>
/// Detector-power tests for <see cref="Pm4YawEvidenceAnalyzer"/>.
/// </summary>
/// <remarks>
/// This test decides whether to keep a transform that is applied to half the corpus, so before its
/// corpus verdict means anything it has to be shown to (a) catch a rotation when the box can see
/// one, and (b) say so when the box cannot, rather than quietly returning "no difference". Case (b)
/// is the one that matters: an axis-aligned box is blind to rotation for a square footprint and
/// nearly blind for an object much smaller than the box, and both are common.
/// </remarks>
public class Pm4YawEvidenceAnalyzerTests
{
    private static readonly Pm4PlacementBox Box = new(0f, 0f, 100f, 20f, "TEST.WMO", 1);

    /// <summary>A long thin footprint filling the box — the case a box CAN judge.</summary>
    private static List<Vector2> LongThinFootprint()
    {
        List<Vector2> points = [];
        for (int step = 0; step <= 40; step++)
        {
            float x = step * 2.5f;
            points.Add(new Vector2(x, 1f));
            points.Add(new Vector2(x, 19f));
        }

        return points;
    }

    private static Vector2 Centroid(IReadOnlyList<Vector2> points)
    {
        Vector2 sum = Vector2.Zero;
        foreach (Vector2 point in points)
            sum += point;

        return sum / points.Count;
    }

    [Fact]
    public void LongThinObject_BoxSeesARotation_SoTheTestHasPower()
    {
        List<Vector2> points = LongThinFootprint();

        Pm4YawDecision decision = Pm4YawEvidenceAnalyzer.Decide(points, Centroid(points), yawRadians: 0f, Box);

        Assert.Equal(1d, decision.InsideCanonical, 3);
        Assert.True(decision.HasDiscriminatingPower, "a long thin footprint filling its box must be rotation-sensitive");
        Assert.True(decision.InsideControl45 < 0.9d, $"the 45 degree control should eject vertices, got {decision.InsideControl45:P1}");
    }

    [Fact]
    public void LongThinObject_WithAWrongYaw_IsReportedAsHurting()
    {
        List<Vector2> points = LongThinFootprint();

        Pm4YawDecision decision = Pm4YawEvidenceAnalyzer.Decide(
            points, Centroid(points), yawRadians: 15f * MathF.PI / 180f, Box);

        Assert.True(decision.HasDiscriminatingPower);
        Assert.Equal("yaw-hurts", decision.Verdict);
        Assert.True(decision.InsideYawOnly < decision.InsideCanonical);
    }

    [Fact]
    public void SmallObjectInAGenerousBox_IsReportedUndecidableRatherThanTied()
    {
        // A tiny footprint near the middle of the box: rotating it changes nothing, because it never
        // reaches an edge. Calling this a "tie" would silently count it as evidence for the yaw.
        List<Vector2> points =
        [
            new(49f, 9f),
            new(51f, 9f),
            new(51f, 11f),
            new(49f, 11f)
        ];

        Pm4YawDecision decision = Pm4YawEvidenceAnalyzer.Decide(
            points, Centroid(points), yawRadians: 15f * MathF.PI / 180f, Box);

        Assert.False(decision.HasDiscriminatingPower);
        Assert.Equal("undecidable", decision.Verdict);
    }

    [Fact]
    public void ObjectRotatedOffAxis_WithTheCorrectingYaw_IsReportedAsHelping()
    {
        // Build the mirror image of the real situation: geometry that genuinely sits rotated, and a
        // yaw that undoes it. If the test could only ever say "hurts", it would be worthless.
        List<Vector2> aligned = LongThinFootprint();
        Vector2 pivot = Centroid(aligned);

        const float offset = 20f * MathF.PI / 180f;
        List<Vector2> rotated = [];
        foreach (Vector2 point in aligned)
        {
            float dx = point.X - pivot.X;
            float dy = point.Y - pivot.Y;
            rotated.Add(new Vector2(
                pivot.X + dx * MathF.Cos(offset) - dy * MathF.Sin(offset),
                pivot.Y + dx * MathF.Sin(offset) + dy * MathF.Cos(offset)));
        }

        Pm4YawDecision decision = Pm4YawEvidenceAnalyzer.Decide(rotated, pivot, yawRadians: -offset, Box);

        Assert.True(decision.HasDiscriminatingPower);
        Assert.Equal("yaw-helps", decision.Verdict);
        Assert.True(decision.InsideYawOnly > decision.InsideCanonical);
    }

    [Fact]
    public void ZeroYaw_IsNotCountedAsEvidenceEitherWay()
    {
        List<Vector2> points = LongThinFootprint();

        Pm4YawDecision decision = Pm4YawEvidenceAnalyzer.Decide(points, Centroid(points), yawRadians: 0f, Box);

        Assert.Equal("no-yaw-applied", decision.Verdict);
    }

    [Fact]
    public void Decide_IsInvariantToTheOrderOfTheFootprintPoints()
    {
        List<Vector2> points = LongThinFootprint();
        List<Vector2> shuffled = [.. points];
        shuffled.Reverse();

        float yaw = 15f * MathF.PI / 180f;
        Pm4YawDecision a = Pm4YawEvidenceAnalyzer.Decide(points, Centroid(points), yaw, Box);
        Pm4YawDecision b = Pm4YawEvidenceAnalyzer.Decide(shuffled, Centroid(shuffled), yaw, Box);

        Assert.Equal(a.Verdict, b.Verdict);
        Assert.Equal(a.InsideYawOnly, b.InsideYawOnly, 6);
    }
}
