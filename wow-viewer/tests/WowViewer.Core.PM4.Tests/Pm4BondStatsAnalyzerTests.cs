using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4BondStatsAnalyzerTests
{
    [Fact]
    public void ReportNotes_DescribeCorpus()
    {
        string directory = Pm4TestPaths.DevelopmentDirectoryPath;
        Pm4BondStatsReport report = Pm4BondStatsAnalyzer.AnalyzeDirectory(directory);

        Assert.NotEmpty(report.Notes);
        Assert.Contains(report.Notes, n => n.Contains("PM4 files"));
        Assert.Contains(report.Notes, n => n.Contains("CK24"));
    }
}
