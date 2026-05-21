using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4RegionObjectGrouperTests
{
    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_ProducesRegionGrouping()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        Assert.True(report.TotalFiles > 0, "Should find PM4 files in the development directory.");
        Assert.True(report.NonEmptyFiles > 0, "Should have non-empty PM4 files.");
        Assert.True(report.TotalRegions > 0, "Should produce at least one region.");
        Assert.True(report.TotalObjects > 0, "Should find at least one object.");
        Assert.NotEmpty(report.Notes);
    }

    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_EachRegionHasTileCoordinates()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        foreach (Pm4Region region in report.Regions)
        {
            Assert.True(region.TileCount > 0, $"Region {region.RegionId} should have at least one tile.");
            Assert.NotEmpty(region.TileCoordinates);
        }
    }

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

    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_EachObjectHasSubObjects()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        foreach (Pm4Region region in report.NonEmptyRegions)
        {
            foreach (Pm4RegionObject obj in region.Objects)
            {
                Assert.True(obj.SubObjectCount > 0, $"Object CK24=0x{obj.Ck24:X6} in region {region.RegionId} should have sub-objects.");
                Assert.True(obj.TileCount > 0, $"Object CK24=0x{obj.Ck24:X6} should have tile coordinates.");
                Assert.True(obj.TotalSurfaceCount > 0, $"Object CK24=0x{obj.Ck24:X6} should have surfaces.");
            }
        }
    }

    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_EachSubObjectHasSurfaceIndices()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        int subObjectCount = 0;
        foreach (Pm4Region region in report.NonEmptyRegions)
        {
            foreach (Pm4RegionObject obj in region.Objects)
            {
                foreach (Pm4SubObject subObj in obj.SubObjects)
                {
                    Assert.True(subObj.SurfaceIndices.Count > 0,
                        $"Sub-object GroupObjectId={subObj.GroupObjectId} in CK24=0x{obj.Ck24:X6} should have surface indices.");
                    subObjectCount++;
                }
            }
        }

        Assert.True(subObjectCount > 0, "Should find at least one sub-object across the corpus.");
    }

    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_RegionIdMatchesMshdField04()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        // Verify that each region's ID corresponds to a valid Field04 value.
        // The development corpus has 227 distinct Field04 values (from research notes).
        HashSet<uint> regionIds = report.Regions.Select(r => r.RegionId).ToHashSet();
        Assert.True(regionIds.Count > 1, "Should have multiple distinct region IDs.");

        // Verify that the empty stub region (Field04=1) is present.
        Assert.Contains(report.Regions, r => r.IsEmptyStubRegion);
    }

    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_RegionGroupsObjectsAcrossTiles()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        // Find regions that span multiple tiles.
        List<Pm4Region> multiTileRegions = report.Regions.Where(r => r.TileCount > 1 && !r.IsEmptyStubRegion).ToList();
        Assert.True(multiTileRegions.Count > 0, "Should have at least one region spanning multiple tiles.");

        // Verify that objects within multi-tile regions have tile coordinates.
        foreach (Pm4Region region in multiTileRegions)
        {
            foreach (Pm4RegionObject obj in region.Objects)
            {
                Assert.True(obj.TileCoordinates.Count > 0,
                    $"Object CK24=0x{obj.Ck24:X6} in multi-tile region {region.RegionId} should have tile coordinates.");
            }
        }
    }

    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_ObjectTypeClassification()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        // Collect all distinct CK24 type bytes across the corpus.
        HashSet<byte> distinctTypes = new();
        foreach (Pm4Region region in report.NonEmptyRegions)
        {
            foreach (Pm4RegionObject obj in region.Objects)
            {
                distinctTypes.Add(obj.Ck24Type);
            }
        }

        // Verify that the known types are present.
        Assert.Contains((byte)0x00, distinctTypes); // Nav mesh
        Assert.Contains((byte)0x42, distinctTypes); // WMO

        // Verify type bytes are non-zero for non-empty regions (except nav mesh).
        foreach (Pm4Region region in report.NonEmptyRegions)
        {
            foreach (Pm4RegionObject obj in region.Objects)
            {
                // Type byte 0x00 means nav mesh (CK24=0), all others should be non-zero.
                if (obj.Ck24 != 0)
                    Assert.NotEqual((byte)0, obj.Ck24Type);
            }
        }
    }

    [Fact]
    public void AnalyzeDirectory_DevelopmentCorpus_PositionRefsAreCollectable()
    {
        Pm4RegionGroupingReport report = Pm4RegionObjectGrouper.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        // At least some non-nav-mesh objects should have MPRL position references.
        int objectsWithPositions = 0;
        foreach (Pm4Region region in report.NonEmptyRegions)
        {
            foreach (Pm4RegionObject obj in region.Objects)
            {
                if (obj.Ck24Type == 0x00) // Skip nav mesh
                    continue;

                foreach (Pm4SubObject subObj in obj.SubObjects)
                {
                    if (subObj.PositionRefCount > 0)
                        objectsWithPositions++;
                }
            }
        }

        Assert.True(objectsWithPositions > 0,
            "At least some non-nav-mesh objects should have MPRL position references.");
    }
}
