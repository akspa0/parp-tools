using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4EdgeIdentityProbeTests
{
    [Fact]
    public void UniqueReciprocalRecordsWithSameValueAreCountedAsEdgeIdentityEvidence()
    {
        var a = File("development_01_02.pm4", 2, 2, 0x77);
        var b = File("development_02_02.pm4", 1, 2, 0x77);
        var report = Pm4EdgeIdentityProbe.Analyze([a, b]);
        var unique = Assert.Single(report.Rows, r => r.Stratum == "unique-reverse");
        Assert.Equal(2, unique.UniqueReverse);
        Assert.Equal(2, unique.UniqueReverseSameValue);
        Assert.Equal(0, unique.UniqueReverseDifferentValue);
    }

    [Fact]
    public void DifferentValuesOnReciprocalRecordsAreRetainedAsCounterevidence()
    {
        var a = File("development_01_02.pm4", 2, 2, 0x11);
        var b = File("development_02_02.pm4", 1, 2, 0x22);
        var unique = Assert.Single(Pm4EdgeIdentityProbe.Analyze([a, b]).Rows, r => r.Stratum == "unique-reverse");
        Assert.Equal(0, unique.UniqueReverseSameValue);
        Assert.Equal(2, unique.UniqueReverseDifferentValue);
    }

    [Fact]
    public void ReusedValueAcrossDistinctEdgesIsNotPromotedToIdentity()
    {
        // Two different source edges in one file share the same raw value.
        var a = File("development_01_02.pm4", 2, 2, 0x55, extraEdge: true);
        var report = Pm4EdgeIdentityProbe.Analyze([a]);
        var reused = Assert.Single(report.Rows, r => r.Stratum == "reused-value");
        Assert.Equal(2, reused.ReusedValueLinks);
        Assert.Equal(2, reused.ReusedValueDistinctEdges);
    }

    [Fact]
    public void MissingTargetIsNoReverseRatherThanSilentlyLocal()
    {
        var a = File("development_01_02.pm4", 9, 9, 0x33);
        var report = Pm4EdgeIdentityProbe.Analyze([a]);
        var none = Assert.Single(report.Rows, r => r.Stratum == "no-reverse");
        Assert.Equal(1, none.NoReverse);
        Assert.Equal(0, report.UniqueReversePairs);
    }

    [Fact]
    public void DuplicateLinksFromSameSurfaceToSameTargetAreMultipleReverse()
    {
        // b's surface 0 has two links to the same target surface, so a's reverse lookup is ambiguous.
        var a = File("development_01_02.pm4", 2, 2, 0x44);
        var b = File("development_02_02.pm4", 1, 2, 0x44, duplicateLink: true);
        var report = Pm4EdgeIdentityProbe.Analyze([a, b]);
        Assert.True(report.MultipleReverseLinks > 0);
        Assert.True(report.UniqueReversePairs > 0);
    }

    private static Pm4MapAuditInput File(
        string name, byte targetX, byte targetY, uint value,
        bool extraEdge = false, bool duplicateLink = false)
    {
        var surfaces = new List<Pm4MsurEntry>
        {
            new(0, 3, duplicateLink ? (byte)2 : (byte)1, 0, Vector3.UnitZ, 0, 0, 0, 0),
        };
        var links = new List<Pm4MslkEntry>
        {
            new(1, 0, 0, value, -1, 0, 0xFFFF0000u | (uint)targetY << 8 | targetX, 0, 0x8000),
        };
        if (duplicateLink)
        {
            links.Add(new Pm4MslkEntry(1, 0, 0, value, -1, 0, 0xFFFF0000u | (uint)targetY << 8 | targetX, 0, 0x8000));
        }
        if (extraEdge)
        {
            surfaces.Add(new Pm4MsurEntry(0, 3, 1, 0, Vector3.UnitZ, 0, 3, 1, 0));
            links.Add(new Pm4MslkEntry(1, 1, 0, value, -1, 0, 0xFFFF0000u | (uint)targetY << 8 | targetX, 0, 0x8000));
        }
        var points = Enumerable.Range(0, surfaces.Count * 3).Select(i => new Vector3(i, 0, 0)).ToArray();
        var indices = Enumerable.Range(0, points.Length).Select(i => (uint)i).ToArray();
        var chunks = new Pm4KnownChunkSet(null, links, [], [], points, indices, surfaces, [], [], [], null, [], [], [], []);
        return new Pm4MapAuditInput(name, new Pm4ResearchDocument(name, 2, [], chunks, []));
    }
}
