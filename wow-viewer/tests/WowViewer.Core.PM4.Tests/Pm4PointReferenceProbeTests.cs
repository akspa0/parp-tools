using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4PointReferenceProbeTests
{
    [Fact]
    public void ATruePointOnSharedBoundaryBeatsWrongIndexAndWrongStreamCandidates()
    {
        var input = File();
        var result = Pm4PointReferenceProbe.Analyze([input]);
        var real = Assert.Single(result.PointRows, r => r.Candidate == "source-MSCN");
        Assert.Equal(2, real.BothBoundariesTight);
        Assert.Equal(0, Assert.Single(result.PointRows, r => r.Candidate == "source-MSCN-plus-1-control").BothBoundariesTight);
        Assert.Equal(0, Assert.Single(result.PointRows, r => r.Candidate == "source-MSVT").BothBoundariesTight);
        Assert.Equal(2, Assert.Single(result.SubtypeRows, r => r.Hypothesis == "source-zero-based").MatchesOtherBoundary);
        Assert.Equal(0, Assert.Single(result.SubtypeRows, r => r.Hypothesis == "source-shift-1-control").MatchesOtherBoundary);
    }

    [Fact]
    public void PointOnSegmentInteriorIsRecognizedWithoutBeingMistakenForVertexOrCentroid()
    {
        var real = Assert.Single(Pm4PointReferenceProbe.Analyze([File()]).PointRows, r => r.Candidate == "source-MSCN");
        Assert.Equal(0, real.SourceVertexTight);
        Assert.Equal(0, real.SourceCentroidTight);
        Assert.Equal(2, real.SourceBoundaryTight);
    }

    [Fact]
    public void MissingAddressedTileDoesNotEraseSourcePointMeasurementOrInventTargetContact()
    {
        var input = File();
        input = input with { Document = input.Document with { KnownChunks = input.Document.KnownChunks with
        {
            Mslk = input.Document.KnownChunks.Mslk.Select(l => l with { LinkId = 0xFFFF0909 }).ToArray(),
        } } };
        var result = Pm4PointReferenceProbe.Analyze([input]);
        var real = Assert.Single(result.PointRows, r => r.Candidate == "source-MSCN");
        Assert.Equal(2, result.MissingTargetLinks);
        Assert.Equal(2, real.SourceBoundaryTight);
        Assert.Equal(0, real.TargetAvailable);
        Assert.Equal(0, real.BothBoundariesTight);
    }

    [Fact]
    public void InvalidSubtypeIsCountedRatherThanModuloMappedToAnEdge()
    {
        var input = File();
        input = input with { Document = input.Document with { KnownChunks = input.Document.KnownChunks with
        {
            Mslk = input.Document.KnownChunks.Mslk.Select(l => l with { Subtype = 255 }).ToArray(),
        } } };
        Assert.All(Pm4PointReferenceProbe.Analyze([input]).SubtypeRows, r => Assert.Equal(0, r.ValidOrdinal));
    }

    [Fact]
    public void SelectedEdgeVerticalPlaneSupportsConnectionGeometryWithoutInventingHeightContact()
    {
        var input = File();
        input = input with { Document = input.Document with { KnownChunks = input.Document.KnownChunks with
        {
            Mslk = input.Document.KnownChunks.Mslk.Select(l => l with { TypeFlags = 2, MspiFirstIndex = 0, MspiIndexCount = 4 }).ToArray(),
            Mspv = [new(2, 0, 5), new(0, 2, 5), new(0, 2, 10), new(2, 0, 10)],
            Mspi = [0, 1, 2, 3],
        } } };
        var result = Pm4PointReferenceProbe.Analyze([input], includePointCandidates: false);
        var real = Assert.Single(result.SubtypeRows, r => r.Hypothesis == "source-zero-based");
        Assert.Empty(result.PointRows);
        Assert.Equal(2, real.AuxiliaryInSelectedVerticalPlane);
        Assert.Equal(0, real.EndpointsOnAuxiliaryBoundary);
        Assert.Equal(0, Assert.Single(result.SubtypeRows, r => r.Hypothesis == "source-shift-1-control").AuxiliaryInSelectedVerticalPlane);
    }

    private static Pm4MapAuditInput File()
    {
        Vector3[] points = [new(0, 0, 0), new(2, 0, 0), new(0, 2, 0), new(2, 0, 0), new(2, 2, 0), new(0, 2, 0)];
        var surfaces = new Pm4MsurEntry[] { new(0, 3, 1, 0, Vector3.UnitZ, 0, 0, 0, 0), new(0, 3, 1, 0, Vector3.UnitZ, 0, 3, 1, 0) };
        var links = new Pm4MslkEntry[] { new(1, 1, 0, 0, -1, 0, 0xFFFF0201, 1, 0x8000), new(1, 2, 0, 0, -1, 0, 0xFFFF0201, 0, 0x8000) };
        var chunks = new Pm4KnownChunkSet(null, links, [], [], points, [0, 1, 2, 3, 4, 5], surfaces,
            [new(1, 1, 0), new(100, 100, 100)], [], [], null, [], [], [], []);
        return new Pm4MapAuditInput("development_01_02.pm4", new Pm4ResearchDocument("synthetic", 2, [], chunks, []));
    }
}
