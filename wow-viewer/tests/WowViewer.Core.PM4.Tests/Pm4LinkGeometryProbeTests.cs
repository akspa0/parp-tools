using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4LinkGeometryProbeTests
{
    [Fact]
    public void StoredNeighbourSharesBoundaryButWrongTargetsDoNot()
    {
        var doc = Document([
            new(0, 0, 0), new(1, 0, 0), new(0, 1, 0),
            new(1, 0, 0), new(1, 1, 0), new(0, 1, 0),
            new(100, 0, 0), new(101, 0, 0), new(100, 1, 0),
        ]);
        var result = Pm4LinkGeometryProbe.Analyze(doc);
        var stored = Assert.Single(result.Rows, r => r.Relation == "stored-reference");
        Assert.Equal(2, stored.Count);
        Assert.Equal(2, stored.SharedBoundaryEdge);
        Assert.Equal(0, stored.SelfReference);
        var offset = Assert.Single(result.Rows, r => r.Relation == "offset-17-control");
        Assert.True(offset.SharedBoundaryEdge < stored.SharedBoundaryEdge);
    }

    [Fact]
    public void NonAdjacentGeometryDoesNotAcquireAnEdgeMerelyBecauseAReferenceExists()
    {
        var doc = Document([
            new(0, 0, 0), new(1, 0, 0), new(0, 1, 0),
            new(10, 0, 0), new(11, 0, 0), new(10, 1, 0),
            new(100, 0, 0), new(101, 0, 0), new(100, 1, 0),
        ]);
        var stored = Assert.Single(Pm4LinkGeometryProbe.Analyze(doc).Rows, r => r.Relation == "stored-reference");
        Assert.Equal(0, stored.SharedVertex);
        Assert.Equal(0, stored.SharedBoundaryEdge);
    }

    [Fact]
    public void PathGeometryCanTouchSeparateEndpointSurfacesWithoutTheirSharingAnEdge()
    {
        var doc = Document([
            new(0, 0, 0), new(1, 0, 0), new(0, 1, 0),
            new(10, 0, 0), new(11, 0, 0), new(10, 1, 0),
            new(100, 0, 0), new(101, 0, 0), new(100, 1, 0),
        ]);
        doc = doc with { KnownChunks = doc.KnownChunks with
        {
            Mslk = [new(2, 0, 0, 0, 0, 4, 0, 1, 0), new(2, 0, 0, 0, 0, 4, 0, 0, 0)],
            Mspv = [new(0, 0, 0), new(1, 0, 0), new(11, 0, 0), new(10, 0, 0)],
            Mspi = [0, 1, 2, 3],
        } };
        var stored = Assert.Single(Pm4LinkGeometryProbe.Analyze(doc).Rows, r => r.Relation == "stored-reference");
        Assert.Equal(0, stored.SharedBoundaryEdge);
        Assert.Equal(2, stored.ValidPathWindows);
        Assert.Equal(2, stored.PathSharesOwnerEdge);
        Assert.Equal(2, stored.PathSharesReferenceEdge);
    }

    [Fact]
    public void OutOfRangeReferenceRetainsRawRecordInsteadOfBeingDeclaredCorruption()
    {
        var doc = Document([
            new(0, 0, 0), new(1, 0, 0), new(0, 1, 0),
            new(10, 0, 0), new(11, 0, 0), new(10, 1, 0),
            new(100, 0, 0), new(101, 0, 0), new(100, 1, 0),
        ]);
        doc = doc with { KnownChunks = doc.KnownChunks with
        {
            Mslk = [new(2, 0, 0, 123, 0, 4, 456, ushort.MaxValue, 789), doc.KnownChunks.Mslk[1]],
        } };
        var result = Pm4LinkGeometryProbe.Analyze(doc);
        var exception = Assert.Single(result.OutOfRangeReferences);
        Assert.Equal(ushort.MaxValue, exception.Raw.RefIndex);
        Assert.Equal(123u, exception.Raw.GroupObjectId);
        Assert.Equal(0, exception.OwnerSurface);
    }

    private static Pm4ResearchDocument Document(Vector3[] vertices)
    {
        var surfaces = new Pm4MsurEntry[]
        {
            new(0, 3, 1, 0, Vector3.UnitZ, 0, 0, 0, 0),
            new(0, 3, 1, 0, Vector3.UnitZ, 0, 3, 1, 0),
            new(0, 3, 0, 0, Vector3.UnitZ, 0, 6, 2, 0),
        };
        var links = new Pm4MslkEntry[] { new(0, 0, 0, 0, -1, 0, 0, 1, 0), new(0, 0, 0, 0, -1, 0, 0, 0, 0) };
        var known = new Pm4KnownChunkSet(null, links, [], [], vertices, Enumerable.Range(0, 9).Select(i => (uint)i).ToArray(),
            surfaces, [], [], [], null, [], [], [], []);
        return new Pm4ResearchDocument("synthetic", 2, [], known, []);
    }
}
