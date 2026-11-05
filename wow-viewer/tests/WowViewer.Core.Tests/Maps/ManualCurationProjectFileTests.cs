using System.Numerics;
using WowViewer.Core.Maps;

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
}
