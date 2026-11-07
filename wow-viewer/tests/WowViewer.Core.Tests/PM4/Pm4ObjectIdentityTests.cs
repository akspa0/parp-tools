using System.Numerics;
using WoWViewer.Workbench.Services.Pm4Workbench;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using Xunit;

namespace WowViewer.Core.Tests.PM4;

public sealed class Pm4ObjectIdentityTests
{
    private static Pm4GeometricShapeDescriptor SyntheticDescriptor(Vector3 dims, Vector3 centroid)
    {
        return Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(dims) with
        {
            PrismCentroid = centroid
        };
    }

    [Fact]
    public void MakeUniqueHullId_GeneratesDeterministic64BitKey()
    {
        int tileX = 14;
        int tileY = 51;
        int componentId = 12;

        ulong id = Pm4HullShapeInput.MakeUniqueHullId(tileX, tileY, componentId);

        // Verify packing: bits 48..63 = tileX, bits 32..47 = tileY, bits 0..31 = componentId
        ulong expected = ((ulong)(uint)tileX << 48) | ((ulong)(uint)tileY << 32) | (uint)componentId;
        Assert.Equal(expected, id);

        // Different component on same tile produces distinct id
        ulong siblingId = Pm4HullShapeInput.MakeUniqueHullId(tileX, tileY, 15);
        Assert.NotEqual(id, siblingId);

        // Same component on different tile produces distinct id
        ulong otherTileId = Pm4HullShapeInput.MakeUniqueHullId(15, 51, 12);
        Assert.NotEqual(id, otherTileId);
    }

    [Fact]
    public void Pm4ObjectIdentity_FormatTag_IncludesAllCoreDimensions()
    {
        var identity = new Pm4ObjectIdentity(
            TileX: 14,
            TileY: 51,
            ComponentId: 12,
            LinkGroupObjectId: 42,
            PlacementZ: 109.9f,
            Centroid: new Vector3(100f, 200f, 109.9f),
            BoundingSize: new Vector3(12.8f, 12.8f, 28.0f),
            SurfaceClass: 1);

        string tag = identity.FormatTag();
        Assert.Equal("[14_51] Obj #12 (Z=109.9yd, Lnk=42)", tag);
        Assert.Equal(tag, identity.ToString());
        Assert.Equal(Pm4HullShapeInput.MakeUniqueHullId(14, 51, 12), identity.UniqueHullId);
    }

    [Fact]
    public void FindHullConflicts_DisambiguatesDistinctComponents_AtSameElevation()
    {
        // Two distinct components on tile (14, 51) at the same elevation (Z = 109.9 yd) and sharing CK24
        var descA = SyntheticDescriptor(new Vector3(12.8f, 12.8f, 28.0f), new Vector3(100f, 200f, 109.9f));
        var descB = SyntheticDescriptor(new Vector3(12.8f, 12.8f, 28.0f), new Vector3(101f, 200f, 109.9f));

        var hullA = new Pm4HullShapeInput(14, 51, 0x42DBCF, 12, 42, descA);
        var hullB = new Pm4HullShapeInput(14, 51, 0x42DBCF, 15, 42, descB);

        var candidate = new Pm4ModelShapeCandidate(
            "World\\wmo\\Northrend\\Buildings\\Wintertauren_GuardTower.wmo",
            "WMO",
            descA);

        var proposalA = Pm4ShapeMatchAnalysisService.Analyze(hullA, [candidate]);
        var proposalB = Pm4ShapeMatchAnalysisService.Analyze(hullB, [candidate]);

        // Both proposals have non-zero distinct EffectiveUniqueHullId and unique HullKey
        Assert.NotEqual(proposalA.EffectiveUniqueHullId, proposalB.EffectiveUniqueHullId);
        Assert.NotEqual(proposalA.HullKey, proposalB.HullKey);

        // Find conflicts across both proposals
        var conflicts = Pm4ShapeMatchAnalysisService.FindHullConflicts([proposalA, proposalB], conflictDistance: 2.0f);

        // Exactly one conflict reported between the two distinct components
        Assert.Single(conflicts);
        var conflict = conflicts[0];
        Assert.Equal(12, conflict.FirstHull.ObjectPartId);
        Assert.Equal(15, conflict.SecondHull.ObjectPartId);
        Assert.Equal(14, conflict.FirstHull.TileX);
        Assert.Equal(51, conflict.FirstHull.TileY);
        Assert.Equal(0x42DBCF, (int)conflict.FirstHull.Ck24);
        Assert.Equal(0x42DBCF, (int)conflict.SecondHull.Ck24);

        // Self-conflict guard: duplicate proposal for the same component does not conflict with itself
        var selfConflicts = Pm4ShapeMatchAnalysisService.FindHullConflicts([proposalA, proposalA], conflictDistance: 2.0f);
        Assert.Empty(selfConflicts);
    }

    [Fact]
    public void Pm4ShapeMatchProposal_EffectiveUniqueHullId_FallsBackCorrectly()
    {
        // 1. Explicit UniqueHullId
        var propExplicit = new Pm4ShapeMatchProposal(
            HullKey: 99,
            HullCentroid: Vector3.Zero,
            HullBoundsMin: Vector3.Zero,
            HullBoundsMax: Vector3.Zero,
            Candidates: [],
            ConfidenceScore: 0,
            IsLowConfidence: false,
            IsUnmatched: true,
            ConflictCount: 1,
            UniqueHullId: 0x1234567890ABCDEFUL);
        Assert.Equal(0x1234567890ABCDEFUL, propExplicit.EffectiveUniqueHullId);

        // 2. Fallback to Hull.UniqueHullId
        var hull = new Pm4HullShapeInput(10, 20, 0, 5, 0, SyntheticDescriptor(Vector3.One, Vector3.Zero));
        var propWithHull = new Pm4ShapeMatchProposal(
            HullKey: 5,
            HullCentroid: Vector3.Zero,
            HullBoundsMin: Vector3.Zero,
            HullBoundsMax: Vector3.Zero,
            Candidates: [],
            ConfidenceScore: 0,
            IsLowConfidence: false,
            IsUnmatched: true,
            ConflictCount: 1,
            Hull: hull);
        Assert.Equal(hull.UniqueHullId, propWithHull.EffectiveUniqueHullId);

        // 3. Fallback to (ulong)HullKey
        var propWithKey = new Pm4ShapeMatchProposal(
            HullKey: 777,
            HullCentroid: Vector3.Zero,
            HullBoundsMin: Vector3.Zero,
            HullBoundsMax: Vector3.Zero,
            Candidates: [],
            ConfidenceScore: 0,
            IsLowConfidence: false,
            IsUnmatched: true,
            ConflictCount: 1);
        Assert.Equal(777UL, propWithKey.EffectiveUniqueHullId);
    }
}
