using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Matching;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4DoodadDuplicationTests
{
    [Fact]
    public void FindUnmatchedTargets_RejectsTargetsNearExistingPlacements()
    {
        // Target 1 is at (100, 200, 50)
        var target1 = new Pm4DoodadTarget(
            35, 38, 0u, 1,
            new Vector3(100f, 200f, 50f),
            new Vector3(95f, 195f, 45f),
            new Vector3(105f, 205f, 55f),
            0f, 0x03);

        // Target 2 is at (150, 250, 50)
        var target2 = new Pm4DoodadTarget(
            35, 38, 0u, 2,
            new Vector3(150f, 250f, 50f),
            new Vector3(145f, 245f, 45f),
            new Vector3(155f, 255f, 55f),
            0f, 0x03);

        // Target 3 is at (200, 300, 50)
        var target3 = new Pm4DoodadTarget(
            35, 38, 0u, 3,
            new Vector3(200f, 300f, 50f),
            new Vector3(195f, 295f, 45f),
            new Vector3(205f, 305f, 55f),
            0f, 0x03);

        var candidates = new List<Pm4DoodadTarget> { target1, target2, target3 };

        // Existing placement is right at (101, 201, 50) which is within 8 yds of Target 1
        var existing = new List<Vector3> { new(101f, 201f, 50f) };

        List<Pm4DoodadTarget> unmatched = Pm4DoodadDuplicationEngine.FindUnmatchedTargets(candidates, existing, occupancyRadius: 8.0f);

        Assert.Equal(2, unmatched.Count);
        Assert.Contains(unmatched, t => t.ObjectPartId == 2);
        Assert.Contains(unmatched, t => t.ObjectPartId == 3);
        Assert.DoesNotContain(unmatched, t => t.ObjectPartId == 1);
    }

    [Fact]
    public void FitModelToTarget_ComputesScaleRatioAndCentroidCorrectly()
    {
        // Model base bounds span 10x10 in XY plane
        var template = new ModelTemplateInfo(
            @"WORLD\EXPANSION02\DOODADS\DRAGONBLIGHT\DRAGONBLIGHT_CLIFF_01.M2",
            "dragonblight_cliff_01",
            "DRAGONBLIGHT_CLIFF_01.M2",
            Position: new Vector3(100f, 200f, 50f),
            RotationDegrees: new Vector3(0f, 0f, 45f),
            Scale: 1.0f,
            LocalBoundsMin: new Vector3(-5f, -5f, -2f),
            LocalBoundsMax: new Vector3(5f, 5f, 2f),
            WorldBoundsMin: new Vector3(95f, 195f, 48f),
            WorldBoundsMax: new Vector3(105f, 205f, 52f));

        // PM4 target hull is 20x20 in XY plane (2x the model scale), world-space bounds
        // centered on the centroid: hull top Z = 104, hull floor Z = 96.
        // Base rotation = 90 degrees (PI / 2 radians)
        var target = new Pm4DoodadTarget(
            35, 38, 0u, 10,
            WorldCentroid: new Vector3(500f, 600f, 100f),
            BoundsMin: new Vector3(490f, 590f, 96f),
            BoundsMax: new Vector3(510f, 610f, 104f),
            BaseRotationRadians: MathF.PI * 0.5f,
            DominantGroupKey: 0x03);

        DuplicatedDoodadResult result = Pm4DoodadDuplicationEngine.FitModelToTarget(template, target, uniqueId: 123456);

        // Expected scale is approx 2.0 (since target planar span is 20*sqrt(2) and model is 10*sqrt(2))
        Assert.InRange(result.Scale, 1.95f, 2.05f);

        // Rotation Yaw should match target base rotation (90 degrees); the model is square
        // in XY (10x10), so no long-axis correction applies.
        Assert.InRange(result.RotationDegrees.Z, 89.9f, 90.1f);

        // T001 top-alignment: Position.Z = hull_top - (LocalBoundsMax.Z * scale) = 104 - (2 * 2) = 100
        Assert.InRange(result.Position.Z, 99.9f, 100.1f);
        // The fitted model's top surface reaches the hull top: 100 + (2 * 2) = 104
        Assert.InRange(result.WorldBoundsMax.Z, 103.9f, 104.1f);

        // Unique ID must be preserved
        Assert.Equal(123456, result.UniqueId);

        // Fit score should be high (approx 1.0)
        Assert.InRange(result.FitScore, 0.9f, 1.0f);
    }

    [Fact]
    public void DuplicateAcrossTargets_AllocatesSequentialUniqueIds()
    {
        var template = new ModelTemplateInfo(
            @"WORLD\GENERIC\CLIFF.M2",
            "cliff",
            "CLIFF.M2",
            Position: Vector3.Zero,
            RotationDegrees: Vector3.Zero,
            Scale: 1.0f,
            LocalBoundsMin: new Vector3(-2f),
            LocalBoundsMax: new Vector3(2f),
            WorldBoundsMin: new Vector3(-2f),
            WorldBoundsMax: new Vector3(2f));

        var candidates = new List<Pm4DoodadTarget>
        {
            new(35, 38, 0u, 1, new Vector3(100f, 0f, 0f), new Vector3(-2f), new Vector3(2f), 0f, 0x03),
            new(35, 38, 0u, 2, new Vector3(200f, 0f, 0f), new Vector3(-2f), new Vector3(2f), 0f, 0x03),
            new(35, 38, 0u, 3, new Vector3(300f, 0f, 0f), new Vector3(-2f), new Vector3(2f), 0f, 0x03)
        };

        var existing = new List<Vector3>();

        List<DuplicatedDoodadResult> results = Pm4DoodadDuplicationEngine.DuplicateAcrossTargets(
            template, candidates, existing, startUniqueId: 2000000);

        Assert.Equal(3, results.Count);
        Assert.Equal(2000000, results[0].UniqueId);
        Assert.Equal(2000001, results[1].UniqueId);
        Assert.Equal(2000002, results[2].UniqueId);
    }

    [Fact]
    public void IsTargetOccupied_DetectsPairedHullWhenModelOriginIsOffsetFromCentroid()
    {
        // Hull centroid (100,200,50) with 10x10x10 world bounds.
        var target = new Pm4DoodadTarget(
            35, 38, 0u, 1,
            new Vector3(100f, 200f, 50f),
            new Vector3(95f, 195f, 45f),
            new Vector3(105f, 205f, 55f),
            0f, 0x03);

        // WMO placement origin sits at the model base corner (90,190,45) — ~15 yds from the hull
        // centroid, so the old origin-distance test reported the hull as free and spawned a
        // duplicate. Its world bounds still overlap the hull, which is the correct "already
        // paired" signal.
        var placement = new PlacementBounds(
            new Vector3(90f, 190f, 45f),
            new Vector3(110f, 210f, 55f),
            new Vector3(90f, 190f, 45f),
            BoundsResolved: true);

        Assert.True(Pm4DoodadDuplicationEngine.IsTargetOccupied(target, [placement], originFallbackRadius: 8f));
    }

    [Fact]
    public void FindUnmatchedTargets_BoundsOverload_RejectsOccupiedHullAndKeepsFreeHull()
    {
        var occupied = new Pm4DoodadTarget(
            35, 38, 0u, 1,
            new Vector3(100f, 200f, 50f),
            new Vector3(95f, 195f, 45f),
            new Vector3(105f, 205f, 55f),
            0f, 0x03);

        var free = new Pm4DoodadTarget(
            35, 38, 0u, 2,
            new Vector3(300f, 400f, 50f),
            new Vector3(295f, 395f, 45f),
            new Vector3(305f, 405f, 55f),
            0f, 0x03);

        var placements = new List<PlacementBounds>
        {
            new(new Vector3(90f, 190f, 45f), new Vector3(110f, 210f, 55f), new Vector3(90f, 190f, 45f), true)
        };

        List<Pm4DoodadTarget> unmatched = Pm4DoodadDuplicationEngine.FindUnmatchedTargets(
            new List<Pm4DoodadTarget> { occupied, free }, placements);

        Assert.Single(unmatched);
        Assert.Equal(2, unmatched[0].ObjectPartId);
    }

    [Fact]
    public void DuplicateAcrossTargets_BoundsOverload_DoesNotDuplicateOnPairedHull()
    {
        var template = new ModelTemplateInfo(
            @"WORLD\GENERIC\CLIFF.M2",
            "cliff",
            "CLIFF.M2",
            Position: Vector3.Zero,
            RotationDegrees: Vector3.Zero,
            Scale: 1.0f,
            LocalBoundsMin: new Vector3(-2f),
            LocalBoundsMax: new Vector3(2f),
            WorldBoundsMin: new Vector3(-2f),
            WorldBoundsMax: new Vector3(2f));

        var occupied = new Pm4DoodadTarget(
            35, 38, 0u, 1,
            new Vector3(100f, 200f, 50f),
            new Vector3(95f, 195f, 45f),
            new Vector3(105f, 205f, 55f),
            0f, 0x03);

        var free = new Pm4DoodadTarget(
            35, 38, 0u, 2,
            new Vector3(300f, 400f, 50f),
            new Vector3(295f, 395f, 45f),
            new Vector3(305f, 405f, 55f),
            0f, 0x03);

        var placements = new List<PlacementBounds>
        {
            new(new Vector3(90f, 190f, 45f), new Vector3(110f, 210f, 55f), new Vector3(90f, 190f, 45f), true)
        };

        List<DuplicatedDoodadResult> results = Pm4DoodadDuplicationEngine.DuplicateAcrossTargets(
            template,
            new List<Pm4DoodadTarget> { occupied, free },
            placements,
            startUniqueId: 2000000);

        Assert.Single(results);
        Assert.Equal(2, results[0].ObjectPartId);
        Assert.Equal(2000000, results[0].UniqueId);
    }
}

