using System.Numerics;
using WoWViewer.Workbench.Services.Pm4Workbench;
using WowViewer.Core.PM4.Matching;

namespace WowViewer.Core.Tests.PM4;

public sealed class Pm4ShapeLibraryVisualServiceTests
{
    [Fact]
    public void RebuildFromCorpus_PresentsAtLeast200EntriesWithVisualPreviewsAndProportions()
    {
        var service = new Pm4ShapeLibraryVisualService();
        service.RebuildFromCorpus(Pm4PriorKnowledgeDatabase.Instance);

        // SC-005: Visual library opens with >= 200 entries and visual previews + proportion summaries.
        Assert.True(
            service.AllEntries.Count >= 200,
            $"Expected >= 200 shape library entries for SC-005, got {service.AllEntries.Count}.");

        foreach (var entry in service.AllEntries)
        {
            Assert.False(string.IsNullOrWhiteSpace(entry.DisplayName));
            Assert.False(string.IsNullOrWhiteSpace(entry.PrimaryModelPath));
            Assert.NotEmpty(entry.AssociatedAssetPaths);
            Assert.True(entry.Preview.NormalizedFootprintPolygon.Count >= 3);
            Assert.True(entry.Preview.NormalizedElevationProfile.Count >= 3);
            Assert.False(string.IsNullOrWhiteSpace(entry.Preview.ProportionSummary));
            Assert.True(entry.Preview.PlanarAspectRatio >= 1.0f);
            Assert.True(entry.Preview.HeightToLengthRatio > 0f);
        }
    }

    [Fact]
    public void ApplyFilter_SupportsSearchAndFamilyFilterAndRetainsSelectionAcrossFilterChanges()
    {
        var service = new Pm4ShapeLibraryVisualService();
        service.RebuildFromCorpus(Pm4PriorKnowledgeDatabase.Instance);

        var wgEntry = service.AllEntries.First(e =>
            e.Family.Equals("Wintergrasp", StringComparison.OrdinalIgnoreCase));
        Assert.True(service.SelectEntry(wgEntry.EntryKey));
        Assert.Equal(wgEntry.EntryKey, service.SelectedEntry?.EntryKey);

        // Filter to a completely different family so the selected Wintergrasp entry is hidden from FilteredEntries
        string otherFamily = service.AvailableFamilies.First(f => !f.Equals(wgEntry.Family, StringComparison.OrdinalIgnoreCase));
        service.ApplyFilter(new Pm4ShapeLibraryFilter(FamilyFilter: otherFamily));
        Assert.NotEmpty(service.FilteredEntries);
        Assert.All(service.FilteredEntries, e => Assert.Equal(otherFamily, e.Family, ignoreCase: true));
        Assert.DoesNotContain(service.FilteredEntries, e => e.EntryKey == wgEntry.EntryKey);

        // FR-003: Selection must be retained across filter changes
        Assert.Equal(wgEntry.EntryKey, service.SelectedEntryKey);
        Assert.Equal(wgEntry.PrimaryModelPath, service.SelectedEntry?.PrimaryModelPath);

        // Search filter narrows by substring
        service.ApplyFilter(new Pm4ShapeLibraryFilter(SearchText: "GuardTower"));
        Assert.NotEmpty(service.FilteredEntries);
        Assert.All(service.FilteredEntries, e =>
            Assert.True(
                e.DisplayName.Contains("GuardTower", StringComparison.OrdinalIgnoreCase) ||
                e.PrimaryModelPath.Contains("GuardTower", StringComparison.OrdinalIgnoreCase) ||
                e.AssociatedAssetPaths.Any(p => p.Contains("GuardTower", StringComparison.OrdinalIgnoreCase))));
    }

    [Fact]
    public void Compare_ComputesSideBySideDimensionAndAspectDeltasAndSimilarity()
    {
        var service = new Pm4ShapeLibraryVisualService();
        service.RebuildFromCorpus(Pm4PriorKnowledgeDatabase.Instance);

        var first = service.AllEntries[0];
        var second = service.AllEntries[1];

        Assert.True(service.SelectEntry(first.EntryKey));
        Assert.True(service.SetComparisonEntry(second.EntryKey));

        Pm4ShapeLibraryComparison? comparison = service.GetActiveComparison();
        Assert.NotNull(comparison);
        Assert.Equal(first.EntryKey, comparison.Left.EntryKey);
        Assert.Equal(second.EntryKey, comparison.Right.EntryKey);
        Assert.Equal(first.Dimensions - second.Dimensions, comparison.DimensionDelta);
        Assert.InRange(comparison.SimilarityScore, 0f, 1f);
        Assert.Contains("Similarity=", comparison.Summary);
    }

    [Fact]
    public void RebuildFromCorpus_MarksUnhealthyAssetsAndExcludesThemFromAutomaticProposals()
    {
        var service = new Pm4ShapeLibraryVisualService();
        var badCandidate = new Pm4ModelShapeCandidate(
            "World\\wmo\\Broken\\MissingTower.wmo",
            "WMO",
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24f, 24f, 46f)));

        service.RebuildFromCorpus(
            Pm4PriorKnowledgeDatabase.Instance,
            additionalCandidates: [badCandidate],
            assetProbe: path => path.Contains("MissingTower", StringComparison.OrdinalIgnoreCase)
                ? (false, false, "File missing from archive")
                : (true, true, null));

        var brokenEntry = Assert.Single(service.AllEntries, e =>
            e.PrimaryModelPath.Contains("MissingTower", StringComparison.OrdinalIgnoreCase));
        Assert.False(brokenEntry.IsHealthy);
        Assert.Equal("File missing from archive", brokenEntry.HealthReason);

        var healthyCandidates = service.GetHealthyModelCandidates();
        Assert.DoesNotContain(healthyCandidates, c =>
            c.ModelPath.Contains("MissingTower", StringComparison.OrdinalIgnoreCase));
    }
}

