using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4DirectModelMatcherTests
{
    [Fact]
    public void MatchPrism_RanksMatchingProportionsHighestAndSolvesScale()
    {
        // Construct a square prism 20x20 with height 10 (aspectXY = 1.0, aspectZH = 0.5)
        var prism = new Pm4FootprintPrism(
            ConvexHull2D: [new(0, 0), new(20, 0), new(20, 20), new(0, 20)],
            TopZ: 100f,
            BaseZ: 90f,
            Height: 10f,
            Centroid: new Vector3(10f, 10f, 95f),
            YawDegrees: 0f,
            UnrotatedLength: 20f,
            UnrotatedWidth: 20f,
            Walls: [],
            WallTriangles: []);

        var candidates = new List<Pm4ModelCandidate>
        {
            // Candidate 1: Cube ruin (e.g. STRANGLETHORNRUINS05.M2 unscaled 10x10x5, scale = 2.0)
            Pm4ModelCandidate.Create(
                @"WORLD\AZEROTH\STRANGLETHORN\PASSIVEDOODADS\RUINS\STRANGLETHORNRUINS05.M2",
                "M2",
                new Vector3(-5f, -5f, 0f),
                new Vector3(5f, 5f, 5f)),

            // Candidate 2: Tall pole (1x1x20)
            Pm4ModelCandidate.Create(
                @"WORLD\GENERIC\POLE.M2",
                "M2",
                new Vector3(-0.5f, -0.5f, 0f),
                new Vector3(0.5f, 0.5f, 20f)),

            // Candidate 3: Wide wall (40x2x5)
            Pm4ModelCandidate.Create(
                @"WORLD\GENERIC\WALL.WMO",
                "WMO",
                new Vector3(-20f, -1f, 0f),
                new Vector3(20f, 1f, 5f)),
        };

        List<Pm4DirectMatchResult> results = Pm4DirectModelMatcher.MatchPrism(prism, candidates, maxResults: 3);

        Assert.NotEmpty(results);
        // Candidate 1 (cube ruin) should rank #1
        Assert.Equal(@"WORLD\AZEROTH\STRANGLETHORN\PASSIVEDOODADS\RUINS\STRANGLETHORNRUINS05.M2", results[0].Candidate.ModelPath);
        Assert.InRange(results[0].Score, 0.95, 1.0);

        // Scale should be ~2.0
        Assert.InRange(results[0].SolvedScale, 1.95f, 2.05f);

        // Model top should reach prism.TopZ (100)
        // Position.Z + (5 * 2) = 100 => Position.Z = 90
        Assert.InRange(results[0].SolvedPosition.Z, 89.9f, 90.1f);
    }

    [Fact]
    public void MatchPrism_AppliesAsymmetricPivotOffsetForWc3HeritageBuildings()
    {
        // A PM4 footprint prism centered at (500, 600, 50) with yaw = 0
        var prism = new Pm4FootprintPrism(
            ConvexHull2D: [new(490, 590), new(510, 590), new(510, 610), new(490, 610)],
            TopZ: 60f,
            BaseZ: 40f,
            Height: 20f,
            Centroid: new Vector3(500f, 600f, 50f),
            YawDegrees: 0f,
            UnrotatedLength: 20f,
            UnrotatedWidth: 20f,
            Walls: [],
            WallTriangles: []);

        // An asymmetric building where the door is at Y = 0, but the building body extends back to Y = 20
        // Bounds: X [-10, 10], Y [0, 20], Z [0, 20].
        // Local center is (0, 10, 10). Origin offset from center is (0, -10, -10).
        var building = Pm4ModelCandidate.Create(
            @"WORLD\BUILDINGS\HUMANBARRACKS.WMO",
            "WMO",
            new Vector3(-10f, 0f, 0f),
            new Vector3(10f, 20f, 20f));

        List<Pm4DirectMatchResult> results = Pm4DirectModelMatcher.MatchPrism(prism, [building]);

        Assert.Single(results);
        Pm4DirectMatchResult match = results[0];

        // The solved position should NOT be at the PM4 centroid (500, 600),
        // but offset to align the building's entrance doorstep with the foundation.
        // At Yaw = 0, rotated offset shifts Y by -10.
        Assert.InRange(match.SolvedPosition.X, 499.9f, 500.1f);
        Assert.NotEqual(600f, match.SolvedPosition.Y);
    }

    [Fact]
    public void MatchPrism_EnforcesWmoScaleFixedAtOne_AndPenalizesExtremeSizeMismatch()
    {
        // 1. Giant WMO building candidate (60x60x40): e.g. BIG_KEEP.WMO
        var keepCandidate = Pm4ModelCandidate.Create(
            @"WORLD\WMO\AZEROTH\BUILDINGS\BIG_KEEP\BIG_KEEP.WMO",
            "WMO",
            new Vector3(-30f, -30f, 0f),
            new Vector3(30f, 30f, 40f));

        // 2. Small doodad prism (2x2 with height 2)
        var smallPrism = new Pm4FootprintPrism(
            ConvexHull2D: [new(-1, -1), new(1, -1), new(1, 1), new(-1, 1)],
            TopZ: 10f,
            BaseZ: 8f,
            Height: 2f,
            Centroid: new Vector3(0f, 0f, 9f),
            YawDegrees: 0f,
            UnrotatedLength: 2f,
            UnrotatedWidth: 2f,
            Walls: [],
            WallTriangles: []);

        // Matching giant WMO to small prism:
        // Must NEVER scale down WMO to 0.35x or 0.03x! SolvedScale MUST be 1.0f.
        // And score must be penalized to 0 due to extreme size mismatch.
        var keepMatchesSmall = Pm4DirectModelMatcher.MatchPrism(smallPrism, [keepCandidate]);
        Assert.Single(keepMatchesSmall);
        Assert.Equal(1.0f, keepMatchesSmall[0].SolvedScale);
        Assert.Equal("WMO", keepMatchesSmall[0].Candidate.AssetKind);
        Assert.Equal(0.0, keepMatchesSmall[0].Score);

        // 3. Appropriate large building prism (60x60 with height 40)
        var buildingPrism = new Pm4FootprintPrism(
            ConvexHull2D: [new(-30, -30), new(30, -30), new(30, 30), new(-30, 30)],
            TopZ: 40f,
            BaseZ: 0f,
            Height: 40f,
            Centroid: new Vector3(0f, 0f, 20f),
            YawDegrees: 0f,
            UnrotatedLength: 60f,
            UnrotatedWidth: 60f,
            Walls: [],
            WallTriangles: []);

        var keepMatchesBuilding = Pm4DirectModelMatcher.MatchPrism(buildingPrism, [keepCandidate]);
        Assert.Single(keepMatchesBuilding);
        Assert.Equal(1.0f, keepMatchesBuilding[0].SolvedScale);
        Assert.InRange(keepMatchesBuilding[0].Score, 0.95, 1.0);

        // 4. Small M2 candidate (2x1x1) evaluated against the 60x60 building prism
        var benchCandidate = Pm4ModelCandidate.Create(
            @"WORLD\GENERIC\PASSIVEDOODADS\BENCHES\WT_BENCH_01.M2",
            "M2",
            new Vector3(-1f, -0.5f, 0f),
            new Vector3(1f, 0.5f, 1f));

        var benchMatchesBuilding = Pm4DirectModelMatcher.MatchPrism(buildingPrism, [benchCandidate]);
        Assert.Single(benchMatchesBuilding);
        // Bench candidate must NOT receive a high score when matching a massive building prism
        Assert.True(benchMatchesBuilding[0].Score < 0.50, $"Expected bench score < 0.50 against building prism, got {benchMatchesBuilding[0].Score}");
    }
}

