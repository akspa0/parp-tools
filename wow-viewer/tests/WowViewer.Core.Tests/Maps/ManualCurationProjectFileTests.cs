using System.Numerics;
using WoWViewer.Workbench.Services.Pm4Workbench;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;

namespace WowViewer.Core.Tests.Maps;

public sealed class ManualCurationProjectFileTests
{
    [Fact]
    public void SaveLoad_RoundTripsPlacementsAndAssociationsWithoutSourceMapWrites()
    {
        string outputDirectory = Path.Combine(AppContext.BaseDirectory, "spec294-test-output", Guid.NewGuid().ToString("N"));
        string path = Path.Combine(outputDirectory, "Azeroth.manual-curation.json");
        string sourceMapSentinel = Path.Combine(outputDirectory, "Azeroth_12_20.adt");
        Directory.CreateDirectory(outputDirectory);
        File.WriteAllText(sourceMapSentinel, "source-map-sentinel");
        string sourceBefore = File.ReadAllText(sourceMapSentinel);

        var project = new ManualCurationProject { MapName = "Azeroth" };
        var placement = new ManualObjectPlacement(
            Guid.Parse("c8a30d0b-6d9d-4c62-a813-b34592b8672d"),
            "World\\Azeroth\\Building.m2",
            "M2",
            "Building",
            TileX: 12,
            TileY: 20,
            PlacementEntryIndex: -1,
            UniqueId: 0x50000001,
            Position: new Vector3(1.25f, 2.5f, 3.75f),
            RotationDegrees: new Vector3(0f, 0f, 17.5f),
            Scale: 1.2f,
            CreatedFromAssetCatalog: true,
            AuthoredUtc: DateTimeOffset.UnixEpoch);
        Assert.True(ManualCurationProjectPolicy.TryAddPlacement(project, placement));
        Pm4HullIdentity hull = new(12, 20, 0x401234, 3, 42);
        Assert.Equal(ManualAssociationResult.Added,
            ManualCurationProjectPolicy.TryAddAssociation(project, placement.PlacementId, hull, DateTimeOffset.UnixEpoch));

        try
        {
            ManualCurationProjectFile.Save(path, project);
            ManualCurationProject loaded = ManualCurationProjectFile.Load(path);

            Assert.Equal(1, loaded.Version);
            Assert.Equal("Azeroth", loaded.MapName);
            ManualObjectPlacement loadedPlacement = Assert.Single(loaded.Placements);
            Assert.Equal(placement.PlacementId, loadedPlacement.PlacementId);
            Assert.Equal(placement.ModelPath, loadedPlacement.ModelPath);
            Assert.Equal(placement.Position, loadedPlacement.Position);
            Assert.Equal(placement.RotationDegrees, loadedPlacement.RotationDegrees);
            Assert.Equal(placement.Scale, loadedPlacement.Scale);
            Assert.True(loadedPlacement.CreatedFromAssetCatalog);
            ManualPm4Association association = Assert.Single(loaded.Associations);
            Assert.Equal(placement.PlacementId, association.PlacementId);
            Assert.Equal(hull, association.Hull);
            Assert.Equal(sourceBefore, File.ReadAllText(sourceMapSentinel));
        }
        finally
        {
            if (Directory.Exists(outputDirectory))
                Directory.Delete(outputDirectory, recursive: true);
        }
    }

    [Fact]
    public void Load_RejectsUnsupportedVersion()
    {
        string directory = Path.Combine(AppContext.BaseDirectory, "spec294-test-output", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "unsupported.json");
        try
        {
            File.WriteAllText(path, "{\"Version\":2,\"MapName\":\"Azeroth\",\"Placements\":[],\"Associations\":[]}");
            Assert.Throws<InvalidDataException>(() => ManualCurationProjectFile.Load(path));
        }
        finally
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void SaveLoad_RoundTripsAcceptRejectReplaceDecisionsAndAuditHistory()
    {
        string outputDirectory = Path.Combine(AppContext.BaseDirectory, "spec295-test-output", Guid.NewGuid().ToString("N"));
        string path = Path.Combine(outputDirectory, "development.manual-curation.json");
        Directory.CreateDirectory(outputDirectory);

        var project = new ManualCurationProject { MapName = "development" };
        var review = new Pm4ShapeMatchReviewService();
        var calibration = new Pm4CentroidCalibration(new Vector3(0.773f, 0f, 0f), "development_00_00 measured +0.773 yd X");

        var hullAccept = new Pm4HullShapeInput(
            0, 0, 0x418D9F, 0, 0,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.5f, 24.4f, 46.1f)) with
            {
                PrismCentroid = new Vector3(16358.5f, 16663.1f, 17.7f)
            });
        var hullRejectReplace = new Pm4HullShapeInput(
            0, 0, 0x41AA27, 1, 0,
            Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.1f, 9.0f, 19.6f)) with
            {
                PrismCentroid = new Vector3(16341.3f, 16614.2f, 21.3f)
            });

        var candidates = new[]
        {
            new Pm4ModelShapeCandidate(
                "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo",
                "WMO",
                Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.5f, 24.4f, 46.1f))),
            new Pm4ModelShapeCandidate(
                "World\\wmo\\Northrend\\Wintergrasp\\WG_Wall01.wmo",
                "WMO",
                Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.1f, 9.0f, 19.6f))),
            new Pm4ModelShapeCandidate(
                "World\\wmo\\Northrend\\Wintergrasp\\WG_WALL01C.wmo",
                "WMO",
                Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(new Vector3(24.3f, 9.2f, 19.6f))),
        };

        var proposal1 = Pm4ShapeMatchAnalysisService.Analyze(hullAccept, candidates, calibration);
        var proposal2 = Pm4ShapeMatchAnalysisService.Analyze(hullRejectReplace, candidates, calibration);

        review.EnqueueProposals(
            [(hullAccept, proposal1), (hullRejectReplace, proposal2)],
            project);

        // SC-006: Enqueuing proposals never commits automatic pairings without an explicit decision
        Assert.Empty(project.ShapeMatchDecisions);
        Assert.Empty(project.ShapeMatchAuditHistory);
        Assert.Equal(0, review.RunBoundedAutoAccept(project));

        Pm4HullIdentity id1 = Pm4ShapeMatchReviewService.ToHullIdentity(hullAccept);
        Pm4HullIdentity id2 = Pm4ShapeMatchReviewService.ToHullIdentity(hullRejectReplace);

        // 1. Accept hull 1
        Assert.True(review.Accept(id1, project, DateTimeOffset.UnixEpoch.AddMinutes(1)));

        // 2. Reject top candidate on hull 2 (WG_Wall01.wmo), verifying it suppresses that pair and promotes WG_WALL01C.wmo (FR-010)
        Assert.True(review.Reject(id2, project, DateTimeOffset.UnixEpoch.AddMinutes(2)));
        Assert.True(ManualCurationProjectPolicy.IsCandidateRejectedForHull(
            project, id2, "World\\wmo\\Northrend\\Wintergrasp\\WG_Wall01.wmo"));
        Assert.Equal(
            "World\\wmo\\Northrend\\Wintergrasp\\WG_WALL01C.wmo",
            review.Queue.Single(q => q.Hull == id2).ActiveTopCandidate?.ModelPath);

        // 3. Replace hull 2 with an explicit library model
        Assert.True(review.Replace(
            id2,
            "World\\wmo\\Northrend\\Wintergrasp\\WG_WALL01D.wmo",
            "WMO",
            project,
            DateTimeOffset.UnixEpoch.AddMinutes(3)));

        try
        {
            ManualCurationProjectFile.Save(path, project);
            ManualCurationProject loaded = ManualCurationProjectFile.Load(path);

            // SC-003: 100% of accept, reject, and replace decisions and audit history survive save and reopen
            Assert.Equal(3, loaded.ShapeMatchDecisions.Count);
            Assert.Equal(3, loaded.ShapeMatchAuditHistory.Count);

            Assert.Contains(loaded.ShapeMatchDecisions, d =>
                d.Hull == id1 &&
                d.Decision == ManualShapeMatchDecisionKind.Accepted &&
                d.ChosenModelPath == "World\\wmo\\Northrend\\Wintergrasp\\GuardTower_intact.wmo");

            Assert.Contains(loaded.ShapeMatchDecisions, d =>
                d.Hull == id2 &&
                d.Decision == ManualShapeMatchDecisionKind.Rejected &&
                d.ProposedModelPath == "World\\wmo\\Northrend\\Wintergrasp\\WG_Wall01.wmo");

            Assert.Contains(loaded.ShapeMatchDecisions, d =>
                d.Hull == id2 &&
                d.Decision == ManualShapeMatchDecisionKind.Replaced &&
                d.ChosenModelPath == "World\\wmo\\Northrend\\Wintergrasp\\WG_WALL01D.wmo");

            Assert.Equal(
                [ManualShapeMatchDecisionKind.Accepted, ManualShapeMatchDecisionKind.Rejected, ManualShapeMatchDecisionKind.Replaced],
                loaded.ShapeMatchAuditHistory.Select(static a => a.Decision).ToArray());
        }
        finally
        {
            if (Directory.Exists(outputDirectory))
                Directory.Delete(outputDirectory, recursive: true);
        }
    }
}
