using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4FieldDomainProbeTests
{
    [Fact]
    public void TrueAddressedReverseLinkIndexBeatsWrongIndexAndWrongFileControls()
    {
        var a = File("development_01_02.pm4", 2, 2, 0);
        var b = File("development_02_02.pm4", 1, 2, 0);
        var r = Pm4FieldDomainProbe.Analyze([a, b]);
        var real = Assert.Single(r.ReverseFits, r => r.Hypothesis == "addressed-raw-index");
        Assert.Equal(2, real.WindowAndBackReferenceMatches);
        Assert.Equal(0, Assert.Single(r.ReverseFits, r => r.Hypothesis == "addressed-index-plus-1-control").WindowAndBackReferenceMatches);
        Assert.Equal(0, Assert.Single(r.ReverseFits, r => r.Hypothesis == "source-file-control").WindowAndBackReferenceMatches);
    }

    [Fact]
    public void InRangeValueIsNotAutomaticallyAReverseLinkOrAnObjectIdentity()
    {
        var a = File("development_01_02.pm4", 2, 2, 0);
        var b = File("development_02_02.pm4", 2, 2, 0); // target points to itself, not back to a
        var real = Assert.Single(Pm4FieldDomainProbe.Analyze([a, b]).ReverseFits,
            r => r.Hypothesis == "addressed-raw-index");
        Assert.Equal(2, real.CandidateInRange);
        Assert.True(real.WindowAndBackReferenceMatches < real.CandidateInRange);
    }

    [Fact]
    public void MissingAddressedTileDoesNotFallBackToAnInRangeSourceRecord()
    {
        var r = Pm4FieldDomainProbe.Analyze([File("development_01_02.pm4", 2, 2, 0)]);
        var real = Assert.Single(r.ReverseFits, r => r.Hypothesis == "addressed-raw-index");
        Assert.Equal(1, real.MissingTargetTile);
        Assert.Equal(0, real.CandidateInRange);
    }

    private static Pm4MapAuditInput File(string name, byte targetX, byte targetY, uint candidate)
    {
        var s = new Pm4MsurEntry(0, 3, 1, 0, Vector3.UnitZ, 0, 0, 0, 0);
        var l = new Pm4MslkEntry(1, 0, 0, candidate, -1, 0, 0xFFFF0000u | (uint)targetY << 8 | targetX, 0, 0x8000);
        var c = new Pm4KnownChunkSet(null, [l], [], [], [Vector3.Zero, Vector3.UnitX, Vector3.UnitY], [0, 1, 2], [s], [], [], [], null, [], [], [], []);
        return new Pm4MapAuditInput(name, new Pm4ResearchDocument(name, 2, [], c, []));
    }
}
