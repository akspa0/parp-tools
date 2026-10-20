using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4MshdProbeTests
{
    [Fact]
    public void AlwaysZeroWordsAreReportedAsZeroAcrossEveryFile()
    {
        var report = Pm4MshdProbe.Analyze([File("development_01_02.pm4", 10, 20, 30), File("development_01_03.pm4", 11, 21, 31)]);

        foreach (string word in new[] { "Field0C", "Field10", "Field14", "Field18", "Field1C" })
        {
            var stats = Assert.Single(report.Words, w => w.Word == word);
            Assert.Equal(2, stats.Files);
            Assert.Equal(2, stats.ZeroValues);
            Assert.Equal(1, stats.DistinctValues);
        }
    }

    [Fact]
    public void CarryingWordsRetainTheirDistinctValues()
    {
        var report = Pm4MshdProbe.Analyze([File("development_01_02.pm4", 10, 20, 30), File("development_01_03.pm4", 11, 21, 31)]);

        Assert.Equal(2, Assert.Single(report.Words, w => w.Word == "Field00").DistinctValues);
        Assert.Equal(2, Assert.Single(report.Words, w => w.Word == "Field04").DistinctValues);
        Assert.Equal(2, Assert.Single(report.Words, w => w.Word == "Field08").DistinctValues);
    }

    [Fact]
    public void MissingMshdIsTreatedAsZeroRatherThanThrowing()
    {
        var input = File("development_01_02.pm4", 10, 20, 30);
        input = input with { Document = input.Document with { KnownChunks = input.Document.KnownChunks with { Mshd = null } } };

        var report = Pm4MshdProbe.Analyze([input]);

        Assert.All(report.Words, w => Assert.Equal(1, w.ZeroValues));
    }

    private static Pm4MapAuditInput File(string id, uint field00, uint field04, uint field08)
    {
        var chunks = new Pm4KnownChunkSet(
            new Pm4MshdHeader(field00, field04, field08, 0, 0, 0, 0, 0),
            [], [], [], [], [], [], [], [], [], null, [], [], [], []);
        return new Pm4MapAuditInput(id, new Pm4ResearchDocument("synthetic", 2, [], chunks, []));
    }
}
