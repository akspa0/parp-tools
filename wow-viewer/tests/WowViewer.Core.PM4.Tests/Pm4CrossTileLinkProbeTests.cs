using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4CrossTileLinkProbeTests
{
    [Fact]
    public void PackedTargetTileAndReferenceResolveReciprocalSeamWhileLocalIndexFitsWrongSurface()
    {
        var a = File("development_01_02.pm4", [new(0, 0, 0), new(1, 0, 0), new(0, 1, 0)], 2, 2);
        var b = File("development_02_02.pm4", [new(1, 0, 0), new(1, 1, 0), new(0, 1, 0)], 1, 2);
        var rows = Pm4CrossTileLinkProbe.Analyze([a, b]);
        var real = Assert.Single(rows, r => r.AddressHypothesis == "low-byte-X-next-byte-Y");
        Assert.True(real.DifferentTile);
        Assert.Equal(2, real.Resolved);
        Assert.Equal(2, real.SharedBoundaryEdge);
        Assert.Equal(2, real.Reciprocal);
        Assert.Equal(2, real.CrossTileWithLocalAlternative);
        Assert.Equal(2, real.PairedAddressReciprocal);
        Assert.Equal(0, real.PairedLocalReciprocal);
        var local = Assert.Single(rows, r => r.AddressHypothesis == "source-file-control");
        Assert.Equal(2, local.SelfReference);
        Assert.Equal(0, local.Reciprocal);
    }

    [Fact]
    public void AbsentAddressedFileIsReportedRatherThanReinterpretedLocally()
    {
        var a = File("development_01_02.pm4", [new(0, 0, 0), new(1, 0, 0), new(0, 1, 0)], 2, 2);
        var row = Assert.Single(Pm4CrossTileLinkProbe.Analyze([a]), r => r.AddressHypothesis == "low-byte-X-next-byte-Y");
        Assert.Equal(1, row.MissingTargetFile);
        Assert.Equal(0, row.Resolved);
    }

    private static Pm4MapAuditInput File(string name, Vector3[] points, byte targetX, byte targetY)
    {
        var surface = new Pm4MsurEntry(0, 3, 1, 0, Vector3.UnitZ, 0, 0, 0, 0);
        var link = new Pm4MslkEntry(1, 0, 0, 0, -1, 0, 0xFFFF0000u | (uint)targetY << 8 | targetX, 0, 0x8000);
        var chunks = new Pm4KnownChunkSet(null, [link], [], [], points, [0, 1, 2], [surface], [], [], [], null, [], [], [], []);
        return new Pm4MapAuditInput(name, new Pm4ResearchDocument(name, 2, [], chunks, []));
    }
}
