using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4PlacementGroundTruthProbeTests
{
    [Fact]
    public void ZeroCk24SurfaceCoversM2OriginAndNonZeroSurfaceCoversWmoOrigin()
    {
        var result = Pm4PlacementGroundTruthProbe.Analyze([File()]);

        Assert.Equal(1, result.Coverage.M2Placements);
        Assert.Equal(1, result.Coverage.M2Covered);
        Assert.Equal(1, result.Coverage.M2CoveredBySingleSurface);
        Assert.Equal(1, result.Coverage.WmoPlacements);
        Assert.Equal(1, result.Coverage.WmoCovered);
        Assert.Equal(2, result.Coverage.Surfaces);
        Assert.Equal(1, result.Coverage.SurfacesWithM2);
        Assert.Equal(1, result.Coverage.SurfacesWithWmo);
        Assert.Equal(0, result.Coverage.SurfacesWithBoth);
        Assert.Equal(0, result.Coverage.SurfacesWithNeither);

        var m2 = Assert.Single(result.Classes, c => c.Class == "m2-only");
        Assert.Equal(1, m2.Surfaces);
        Assert.Equal(1, m2.Ck24Zero);
        Assert.Equal(0, m2.Ck24NonZero);

        var wmo = Assert.Single(result.Classes, c => c.Class == "wmo-only");
        Assert.Equal(1, wmo.Surfaces);
        Assert.Equal(0, wmo.Ck24Zero);
        Assert.Equal(1, wmo.Ck24NonZero);
    }

    [Fact]
    public void PuritySeparatesZeroTypeForM2FromNonZeroTypeForWmo()
    {
        var result = Pm4PlacementGroundTruthProbe.Analyze([File()]);

        var zero = Assert.Single(result.Purity, p => p.Feature == "Ck24Type" && p.Value == "0");
        Assert.Equal(1, zero.M2Only);
        Assert.Equal(0, zero.WmoOnly);

        var nonZero = Assert.Single(result.Purity, p => p.Feature == "Ck24Type" && p.Value == "65");
        Assert.Equal(0, nonZero.M2Only);
        Assert.Equal(1, nonZero.WmoOnly);
    }

    [Fact]
    public void PlacementOutsideEverySurfaceIsNotCountedAsCovered()
    {
        var input = File();
        var catalog = input.Placements!;
        var moved = catalog with
        {
            ModelPlacements = [catalog.ModelPlacements[0] with { Position = new Vector3(5000, 5000, 0) }],
        };
        var result = Pm4PlacementGroundTruthProbe.Analyze([input with { Placements = moved }]);

        Assert.Equal(1, result.Coverage.M2Placements);
        Assert.Equal(0, result.Coverage.M2Covered);
        Assert.Equal(0, result.Coverage.SurfacesWithM2);
    }

    [Fact]
    public void MissingCompanionAdtIsSkippedRatherThanGuessed()
    {
        var result = Pm4PlacementGroundTruthProbe.Analyze([File() with { Placements = null }]);

        Assert.Equal(0, result.FilesWithPlacements);
        Assert.Equal(0, result.Coverage.M2Placements);
        Assert.Equal(0, result.Coverage.Surfaces);
    }

    private static Pm4PlacementAuditInput File()
    {
        // The probe converts MSVT to ADT placement space as (MapOrigin - X, MapOrigin - Y, Z).
        // Build raw vertices that land on the intended placement-space squares.
        Vector3 Raw(float placementX, float placementY) =>
            new(Pm4CoordinateService.MapOrigin - placementX, Pm4CoordinateService.MapOrigin - placementY, 0);

        Vector3[] points =
        [
            Raw(99, 199), Raw(101, 199), Raw(101, 201), Raw(99, 201),
            Raw(299, 399), Raw(301, 399), Raw(301, 401), Raw(299, 401),
        ];

        var surfaces = new Pm4MsurEntry[]
        {
            new(3, 4, 0, 0, Vector3.UnitZ, 0, 0, 0, 0),          // GroupKey 3, Ck24Type 0
            new(18, 4, 0, 0, Vector3.UnitZ, 0, 4, 0, 65u << 24), // GroupKey 18, Ck24Type 65
        };

        var chunks = new Pm4KnownChunkSet(
            null, [], [], [], points, [0, 1, 2, 3, 4, 5, 6, 7], surfaces,
            [], [], [], null, [], [], [], []);

        var catalog = new AdtPlacementCatalog(
            "synthetic_obj0.adt",
            MapFileKind.AdtObj,
            ["m2.m2"],
            ["wmo.wmo"],
            [new AdtModelPlacement(0, "m2.m2", 1, new Vector3(100, 200, 0), Vector3.Zero, 1f)],
            [new AdtWorldModelPlacement(0, "wmo.wmo", 2, new Vector3(300, 400, 0), Vector3.Zero,
                new Vector3(299, 399, 0), new Vector3(301, 401, 0), 0)]);

        return new Pm4PlacementAuditInput(
            "development_01_02.pm4",
            new Pm4ResearchDocument("synthetic", 2, [], chunks, []),
            catalog);
    }
}
