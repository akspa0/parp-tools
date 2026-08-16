using WowViewer.Core.IO.AssetReferences;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 155. These pin the distinctions that keep a sweep honest: three resolution states rather than
/// two, substitution recorded rather than hidden, and incompleteness visible in the report itself.
/// </summary>
public sealed class AssetReferenceReportTests
{
    private static readonly BuildIdentity Build = new("0.5.3.3368", "<root>");

    [Fact]
    public void UnreadableAsset_IsNotCountedAsAMissingReference()
    {
        // An asset that could not be read contributes no references. Treating that as "found nothing
        // missing" is the failure this whole feature is shaped to avoid.
        SweepReport report = ReportOf(
            new ReferencingAssetResult("a.wmo", ReferencingAssetState.Unreadable, [], "truncated"));

        Assert.Equal(0, report.UnresolvedReferenceCount);
        Assert.Empty(report.DistinctMissingTargets);
        Assert.Equal(1, report.AssetsUnreadable);
        Assert.False(report.Complete);
    }

    [Fact]
    public void UnreadableTarget_IsNotReportedAsMissing()
    {
        SweepReport report = ReportOf(Read("a.wmo",
            new AssetReference("a.wmo", AssetReferenceKind.ModelTexture, "t.blp", AssetResolution.Unreadable)));

        Assert.Equal(0, report.UnresolvedReferenceCount);
        Assert.Empty(report.DistinctMissingTargets);
    }

    [Fact]
    public void SubstitutedReference_ResolvesAndIsRecordedAsSubstituted()
    {
        AssetReference reference = new(
            "a.wmo", AssetReferenceKind.PlacedDoodad, "x.mdl", AssetResolution.Present, "x.mdx");

        SweepReport report = ReportOf(Read("a.wmo", reference));

        Assert.True(reference.ResolvedBySubstitution);
        Assert.Equal(1, report.SubstitutedReferenceCount);
        Assert.Equal(0, report.UnresolvedReferenceCount);
        Assert.Empty(report.DistinctMissingTargets);
    }

    [Fact]
    public void DirectlyResolvedReference_IsNotCountedAsSubstituted()
    {
        SweepReport report = ReportOf(Read("a.wmo",
            new AssetReference("a.wmo", AssetReferenceKind.PlacedDoodad, "x.mdx", AssetResolution.Present)));

        Assert.Equal(0, report.SubstitutedReferenceCount);
    }

    [Fact]
    public void MissingTargets_AreDistinctAcrossManyReferrers()
    {
        // One asset referenced by many objects is one missing asset, not many. References and assets
        // must never be conflated in a count.
        SweepReport report = ReportOf(
            Read("a.wmo", Absent("a.wmo", "shared.blp")),
            Read("b.wmo", Absent("b.wmo", "shared.blp")),
            Read("c.wmo", Absent("c.wmo", "SHARED.BLP")));

        Assert.Equal(3, report.UnresolvedReferenceCount);
        Assert.Single(report.DistinctMissingTargets);
    }

    [Fact]
    public void CleanSweep_IsComplete()
    {
        SweepReport report = ReportOf(Read("a.wmo",
            new AssetReference("a.wmo", AssetReferenceKind.PlacedDoodad, "x.mdx", AssetResolution.Present)));

        Assert.True(report.Complete);
    }

    [Fact]
    public void BlockedRoute_MakesReportIncompleteEvenWithNoFindings()
    {
        // The dangerous case: a route that could not be swept finds nothing, which without the block
        // record is indistinguishable from a healthy build.
        SweepReport report = new()
        {
            Build = Build,
            Results = [],
            BlockedRoutes = [new BlockedRoute("md20-0x107", 17296, "route does not read yet")],
            ReferenceKindsSwept = [AssetReferenceKind.PlacedDoodad],
            WorldObjectsExamined = 9711,
            ModelsExamined = 0,
        };

        Assert.Equal(0, report.UnresolvedReferenceCount);
        Assert.False(report.Complete);
        Assert.Equal(17296, report.BlockedRoutes[0].AssetCount);
    }

    [Fact]
    public void ReadAssetWithNoReferences_IsCompleteAndDistinctFromUnreadable()
    {
        SweepReport read = ReportOf(new ReferencingAssetResult("a.wmo", ReferencingAssetState.Read, []));
        SweepReport unreadable = ReportOf(
            new ReferencingAssetResult("a.wmo", ReferencingAssetState.Unreadable, [], "boom"));

        Assert.True(read.Complete);
        Assert.False(unreadable.Complete);
        Assert.Equal(0, read.AssetsUnreadable);
        Assert.Equal(1, unreadable.AssetsUnreadable);
    }

    private static AssetReference Absent(string source, string target)
        => new(source, AssetReferenceKind.WorldObjectTexture, target, AssetResolution.Absent);

    private static ReferencingAssetResult Read(string path, params AssetReference[] references)
        => new(path, ReferencingAssetState.Read, references);

    private static SweepReport ReportOf(params ReferencingAssetResult[] results) => new()
    {
        Build = Build,
        Results = results,
        BlockedRoutes = [],
        ReferenceKindsSwept = [AssetReferenceKind.PlacedDoodad, AssetReferenceKind.WorldObjectTexture, AssetReferenceKind.ModelTexture],
        WorldObjectsExamined = results.Length,
        ModelsExamined = 0,
    };
}
