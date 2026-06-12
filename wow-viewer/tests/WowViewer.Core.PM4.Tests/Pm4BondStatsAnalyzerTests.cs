using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4BondStatsAnalyzerTests
{
    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_ProducesNonEmptyReport()
    {
        string directory = Pm4TestPaths.DevelopmentDirectoryPath;
        Assert.True(Directory.Exists(directory));

        Pm4BondStatsReport report = Pm4BondStatsAnalyzer.AnalyzeDirectory(directory);

        Assert.Equal(616, report.FileCount);
        Assert.True(report.TotalSurfaceCount > 0);
        Assert.True(report.ZeroCk24SurfaceCount >= 0);
        Assert.True(report.DistinctCk24Values > 0);
        Assert.True(report.DistinctCk24Types > 0);
        Assert.True(report.CrossTabulation.TotalPairs > 0);
        Assert.Equal(report.FileCount, report.PerFileEntries.Count);
    }

    [Fact]
    public void CrossTabulation_HasConsistentCounts()
    {
        string directory = Pm4TestPaths.DevelopmentDirectoryPath;
        Pm4BondStatsReport report = Pm4BondStatsAnalyzer.AnalyzeDirectory(directory);

        Assert.True(report.CrossTabulation.DistinctHighByteValues > 0);
        Assert.True(report.CrossTabulation.DistinctLowByteValues > 0);
        Assert.NotEmpty(report.CrossTabulation.TopPairsByCount);
        Assert.NotEmpty(report.CrossTabulation.TopPairsByHighByte);
        Assert.NotEmpty(report.CrossTabulation.TopPairsByLowByte);

        foreach (var pair in report.CrossTabulation.TopPairsByCount)
        {
            Assert.True(pair.SurfaceCount > 0);
            Assert.True(pair.FileCount > 0);
            Assert.NotEmpty(pair.AssociatedCk24Types);
        }
    }

    [Fact]
    public void PerFileEntries_ContainTypeBucketBreakdown()
    {
        string directory = Pm4TestPaths.DevelopmentDirectoryPath;
        Pm4BondStatsReport report = Pm4BondStatsAnalyzer.AnalyzeDirectory(directory);

        bool foundBucketEntry = false;
        foreach (var entry in report.PerFileEntries)
        {
            if (entry.TypeBucketBreakdown.Count > 0)
            {
                foundBucketEntry = true;
                foreach (var bucket in entry.TypeBucketBreakdown)
                {
                    Assert.True(bucket.SurfaceCount > 0);
                    Assert.False(string.IsNullOrWhiteSpace(bucket.TypeLabel));
                }
                break;
            }
        }

        Assert.True(foundBucketEntry);
    }

    [Fact]
    public void PerFileEntry_DevelopmentTile_MatchesExpectedCounts()
    {
        string directory = Pm4TestPaths.DevelopmentDirectoryPath;
        Pm4BondStatsReport report = Pm4BondStatsAnalyzer.AnalyzeDirectory(directory);

        var tile = report.PerFileEntries.FirstOrDefault(
            e => e.SourcePath != null
                && e.SourcePath.EndsWith("development_00_00.pm4", StringComparison.OrdinalIgnoreCase));

        Assert.NotNull(tile);
        Assert.Equal(0, tile.TileX);
        Assert.Equal(0, tile.TileY);
        Assert.True(tile.SurfaceCount > 0);
        Assert.True(tile.DistinctCk24Types > 0);
    }

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
