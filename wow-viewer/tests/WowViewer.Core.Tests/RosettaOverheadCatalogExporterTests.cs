using System.Numerics;
using WowViewer.Core.IO.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class RosettaOverheadCatalogExporterTests
{
    [Fact]
    public void ExportCatalog_ComputesExtentsAndPlacementBoxesCorrectly()
    {
        // 1. Setup mock corpus data with 2 assets
        var crateMin = new Vector3(-2f, -3f, 0f);
        var crateMax = new Vector3(2f, 3f, 4f);
        var towerMin = new Vector3(-10f, -10f, 0f);
        var towerMax = new Vector3(10f, 10f, 25f);

        var uniqueAssets = new Dictionary<string, RosettaAssetEntry>(StringComparer.OrdinalIgnoreCase)
        {
            ["Doodads/Crate.m2"] = new RosettaAssetEntry(
                "Doodads/Crate.m2",
                RosettaAssetKind.Model,
                crateMin,
                crateMax),
            ["Buildings/Tower.wmo"] = new RosettaAssetEntry(
                "Buildings/Tower.wmo",
                RosettaAssetKind.WorldModel,
                towerMin,
                towerMax),
        };

        string crateId = RosettaDatastoreWriter.ComputeAssetId("Doodads/Crate.m2");
        string towerId = RosettaDatastoreWriter.ComputeAssetId("Buildings/Tower.wmo");

        var placements = new List<RosettaDecodedPlacement>
        {
            new(crateId, "Doodads/Crate.m2", "doodads/crate.m2", RosettaAssetKind.Model,
                32, 32, new Vector3(0f, 0f, 10f), new Vector3(0f, 0f, 10f),
                0.25f, 0.25f, 133.33f, 100f, crateMin, crateMax, "Crate", 1),
            new(towerId, "Buildings/Tower.wmo", "buildings/tower.wmo", RosettaAssetKind.WorldModel,
                32, 32, new Vector3(50f, 50f, 10f), new Vector3(50f, 50f, 10f),
                0.75f, 0.75f, 266.66f, 200f, towerMin, towerMax, "Tower", 2),
        };

        var corpus = new RosettaCorpusData(
            "RosettaTestMap",
            "0_5_3_3368",
            "2026-10-07T00:00:00Z",
            placements,
            new[] { "32_32" },
            uniqueAssets);

        // 2. Export catalog
        RosettaOverheadCatalog catalog = RosettaOverheadCatalogExporter.ExportCatalog(corpus, minimapResolution: 256);

        // 3. Assertions
        Assert.Equal("RosettaTestMap", catalog.MapName);
        Assert.Equal(2, catalog.TotalExhibits);
        Assert.Equal(2, catalog.TotalPlacements);

        // Verify Crate exhibit
        RosettaOverheadExhibit crateExhibit = catalog.Exhibits.First(e => e.AssetId == crateId);
        Assert.Equal("Model", crateExhibit.Kind);
        Assert.Equal(4f, crateExhibit.ExtentX);
        Assert.Equal(6f, crateExhibit.ExtentY);
        Assert.Equal(4f, crateExhibit.ExtentZ);
        Assert.Equal(24f, crateExhibit.FootprintArea);
        Assert.Single(crateExhibit.Placements);

        RosettaOverheadPlacementBox box = crateExhibit.Placements[0];
        Assert.Equal(32, box.TileX);
        Assert.Equal(32, box.TileY);
        Assert.InRange(box.PixelMinX, 0, 256);
        Assert.InRange(box.PixelMaxX, 0, 256);
        Assert.InRange(box.PixelMinY, 0, 256);
        Assert.InRange(box.PixelMaxY, 0, 256);

        // 4. Test JSON serialization
        string json = RosettaOverheadCatalogExporter.ExportToJson(corpus);
        Assert.Contains("\"MapName\": \"RosettaTestMap\"", json);
        Assert.Contains(crateId, json);
        Assert.Contains(towerId, json);
    }
}
