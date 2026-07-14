using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4RegionObjectGrouperTests
{
    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_NonEmptyRegionsHaveObjects()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        foreach (Pm4Region region in report.NonEmptyRegions)
        {
            Assert.True(region.TotalObjectCount > 0, $"Non-empty region {region.RegionId} should have objects.");
            Assert.True(region.TotalSurfaceCount > 0, $"Non-empty region {region.RegionId} should have surfaces.");
        }
    }

}
