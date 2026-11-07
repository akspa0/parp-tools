using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Matching;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4MapWidePropagationTests
{
    [Fact]
    public void ServiceConstants_MatchesSyntheticUniqueIdBase()
    {
        Assert.Equal(0x50000000, Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase);
    }

    [Fact]
    public void TaughtPairing_ResolvesWmoAssetKindProperly()
    {
        var taught = new TaughtPairing(
            ModelName: "TAURENHUT0.WMO",
            ModelPath: @"World\wmo\Kalimdor\Tauren\TaurenHut0.wmo",
            Length: 15.0f,
            Width: 15.0f,
            Height: 11.0f,
            YawDegrees: 45.0f,
            SolvedScale: 1.0f,
            TaughtUtc: System.DateTime.UtcNow,
            Ck24: 0x430000,
            DominantGroupKey: 0x12,
            LocalBoundsMin: new Vector3(-7.5f, -7.5f, 0f),
            LocalBoundsMax: new Vector3(7.5f, 7.5f, 11f),
            AssetKind: "WMO",
            PlacementPosition: new Vector3(100f, 200f, 30f),
            PlacementRotation: new Vector3(0f, 0f, 45f),
            PlacementScale: 1.0f,
            UniqueId: 12345,
            TileX: 30,
            TileY: 40,
            ObjectPartId: 0,
            LinkGroupObjectId: 0x1000,
            HullCentroid: new Vector3(100f, 200f, 30f),
            HullBoundsMin: new Vector3(92.5f, 192.5f, 30f),
            HullBoundsMax: new Vector3(107.5f, 207.5f, 41f));

        Assert.True(Pm4DoodadDuplicationEngine.IsWmoAsset(taught.ModelPath, taught.AssetKind));
        Assert.Equal("WMO", Pm4DoodadDuplicationEngine.ResolveAssetKind(taught.ModelPath, taught.AssetKind));
    }

    [Fact]
    public void IsTargetOccupied_PreventsDuplicateSpawnsViaAABBOverlap()
    {
        var target = new Pm4DoodadTarget(
            30, 40, 0x430000, 0,
            WorldCentroid: new Vector3(100f, 100f, 10f),
            BoundsMin: new Vector3(95f, 95f, 5f),
            BoundsMax: new Vector3(105f, 105f, 15f),
            BaseRotationRadians: 0f,
            DominantGroupKey: 0x12);

        var occupiedBounds = new List<PlacementBounds>
        {
            new(Min: new Vector3(94f, 94f, 4f), Max: new Vector3(106f, 106f, 16f), Origin: new Vector3(100f, 100f, 10f), BoundsResolved: true)
        };

        bool occupied = Pm4DoodadDuplicationEngine.IsTargetOccupied(target, occupiedBounds);
        Assert.True(occupied);

        var nonOverlappingBounds = new List<PlacementBounds>
        {
            new(Min: new Vector3(200f, 200f, 10f), Max: new Vector3(210f, 210f, 20f), Origin: new Vector3(205f, 205f, 15f), BoundsResolved: true)
        };

        bool notOccupied = Pm4DoodadDuplicationEngine.IsTargetOccupied(target, nonOverlappingBounds);
        Assert.False(notOccupied);
    }
}

