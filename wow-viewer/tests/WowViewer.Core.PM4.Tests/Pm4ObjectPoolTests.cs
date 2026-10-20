using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4ObjectPoolTests
{
    [Fact]
    public void WlHintsRetainKnownUnknownAndAmbiguousNamesWithoutAssigningTiles()
    {
        var hints = Pm4DesignKitHints.FromSourceNames([
            "Test - HFjord - Terrace 01.wlw", "Test - Grizzly Pond.wlw", "NotHFjord.wlw", "Unknown.wlm", "HFjord Grizzly.wlw",
        ]);
        Assert.Equal(5, hints.Count);
        Assert.Equal("Howling Fjord", Assert.Single(hints.Single(h => h.SourcePath.Contains("Terrace")).SuggestedZones));
        Assert.Empty(hints.Single(h => h.SourcePath.StartsWith("Not")).SuggestedZones);
        Assert.Empty(hints.Single(h => h.SourcePath.StartsWith("Unknown")).SuggestedZones);
        Assert.Equal(2, hints.Single(h => h.SourcePath.StartsWith("HFjord Grizzly")).SuggestedZones.Count);
        Assert.All(hints, h => Assert.False(h.OperatorConfirmed));
        var custom = Pm4DesignKitHints.FromSourceNames(["Griz - Demo Pond.wlw"], [new("Griz", "Grizzly Hills", "Northrend")]);
        Assert.Equal("Grizzly Hills", Assert.Single(Assert.Single(custom).SuggestedZones));
        var confirmed = hints[0] with { OperatorConfirmed = true, SuggestedZones = ["Operator-confirmed zone"] };
        var merged = Pm4DesignKitHints.Merge([confirmed], hints);
        Assert.Equal("Operator-confirmed zone", Assert.Single(merged.Single(h => h.SourcePath == confirmed.SourcePath).SuggestedZones));
    }

    [Fact]
    public void AssignedTileFiltersByExplicitKindAndDirectoryPrefixWithoutChangingRecordsOrBins()
    {
        var db = Database();
        var selected = Pm4ObjectPoolFilter.Select(Manifest(), db, "development", 10, 20);
        Assert.True(selected.Restricted);
        Assert.Equal(4, selected.InputCandidates);
        Assert.Equal(1, selected.EligibleCandidates);
        Assert.Same(db.Records[0], Assert.Single(selected.Database.Records));
        Assert.Equal(db.EdgeBinSize, selected.Database.EdgeBinSize);
        Assert.Equal(db.NormalAlignmentBinSize, selected.Database.NormalAlignmentBinSize);
        Assert.Equal(0, selected.Database.WmoCount);
        Assert.Equal(1, selected.Database.M2Count);
    }

    [Theory]
    [InlineData("another-map", 10, 20)]
    [InlineData("development", 9, 20)]
    [InlineData("development", 12, 22)]
    public void UnannotatedMapOrTilePreservesTheExactDatabase(string map, int x, int y)
    {
        var db = Database();
        var selected = Pm4ObjectPoolFilter.Select(Manifest(), db, map, x, y);
        Assert.False(selected.Restricted);
        Assert.Same(db, selected.Database);
    }

    [Fact]
    public void OverlappingPoolsFormUnionAndExplicitPathsDoNotChangeGeometryHistograms()
    {
        var m = Manifest();
        var extra = new Pm4ObjectPool("specific", "operator-kit", ["gate"], ["wmo"], ["world/northrend/gate.wmo"], [], "Explicit inclusion");
        m = m with
        {
            Pools = [.. m.Pools, extra],
            Assignments = [.. m.Assignments, new("specific", 10, 20, 10, 20, "Same tile")],
        };
        var db = Database();
        var selection = Pm4ObjectPoolFilter.Select(m, db, "DEVELOPMENT", 10, 20);
        Assert.Equal(2, selection.EligibleCandidates);
        Assert.Equal(new[] { "northrend-test", "specific" }, selection.ActivePools);
        Assert.Same(db.Records[1].TriangleHistogram, selection.Database.Records[1].TriangleHistogram);
    }

    [Fact]
    public void EmptyEligibleSetDoesNotFallBackToUnrelatedAssets()
    {
        var m = Manifest();
        m = m with { Pools = [m.Pools[0] with { PathPrefixes = ["world/no-such-designkit"] }] };
        var selected = Pm4ObjectPoolFilter.Select(m, Database(), "development", 10, 20);
        Assert.True(selected.Restricted);
        Assert.Empty(selected.Database.Records);
    }

    [Fact]
    public void DesignKitAndClassLabelsAreNotUndocumentedPathGuessers()
    {
        var m = Manifest();
        m = m with { Pools = [m.Pools[0] with { DesignKit = "AnyLabel", ObjectClasses = ["SomeClass"] }] };
        var selection = Pm4ObjectPoolFilter.Select(m, Database(), "development", 10, 20);
        Assert.Equal("selected", Assert.Single(selection.Database.Records).AssetId);
    }

    [Fact]
    public void InvalidAssignmentsAndUnsafeOrEmptySelectorsAreRejected()
    {
        var m = Manifest();
        Assert.Throws<ArgumentException>(() => Pm4ObjectPoolFilter.Validate(m with { Assignments = [new("unknown", 0, 0, 0, 0, "")] }));
        Assert.Throws<ArgumentException>(() => Pm4ObjectPoolFilter.Validate(m with { Assignments = [new("northrend-test", 11, 20, 10, 20, "")] }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pm4ObjectPoolFilter.Validate(m with { Assignments = [new("northrend-test", -1, 0, 0, 0, "")] }));
        Assert.Throws<ArgumentException>(() => Pm4ObjectPoolFilter.Validate(m with { Pools = [m.Pools[0] with { PathPrefixes = [] }] }));
        Assert.Throws<ArgumentException>(() => Pm4ObjectPoolFilter.Validate(m with { Pools = [m.Pools[0] with { PathPrefixes = ["../client"] }] }));
    }

    [Fact]
    public void ManifestPersistsOperatorNotesRangesAndSelectorsInsideProjectOutput()
    {
        string directory = Path.Combine(AppContext.BaseDirectory, "output", "object-pool-tests", Guid.NewGuid().ToString("N"));
        string path = Path.Combine(directory, "manifest.json");
        try
        {
            Pm4ObjectPoolManifestStore.Write(path, Manifest());
            var read = Pm4ObjectPoolManifestStore.Read(path);
            Assert.Equal("Operator observation; not confirmed identity.", Assert.Single(read.Assignments).Note);
            Assert.Equal(new[] { "world/northrend" }, Assert.Single(read.Pools).PathPrefixes);
            Assert.Single(Pm4ObjectPoolFilter.Select(read, Database(), "development", 11, 21).Database.Records);
        }
        finally
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }

    private static Pm4ObjectPoolManifest Manifest() => new(1, "development",
        [new("northrend-test", "Northrend", ["test-props"], ["m2"], [], ["world/northrend"], "Synthetic selector, not an actual kit inventory")],
        [new("northrend-test", 10, 20, 11, 21, "Operator observation; not confirmed identity.")]);

    private static SurfaceCorrelationDatabase Database() => new("synthetic", "test", 1, 0.5f, 0.25f, 0.1f, 1,
    [
        Fingerprint("selected", @"WORLD\NORTHREND\prop.m2", "m2"),
        Fingerprint("wrong-kind", "world/northrend/gate.wmo", "wmo"),
        Fingerprint("wrong-directory", "world/northrend-other/prop.m2", "m2"),
        Fingerprint("wrong-kit", "world/kalimdor/prop.m2", "m2"),
    ]) { M2Count = 3 };

    private static SurfaceCorrelationFingerprint Fingerprint(string id, string path, string kind) =>
        new(id, path, kind, 0x40, 1, 1, 3, new Dictionary<string, int> { ["unchanged-geometry"] = 1 }, "synthetic");
}
