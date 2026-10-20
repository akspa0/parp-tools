using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4ObjectGroupCatalogTests
{
    [Fact]
    public void GroupsAreKeyedByUnk04AndCarrySurfaceClassAndNearestModel()
    {
        var entries = Pm4ObjectGroupCatalog.Build(Document(), Catalog());

        Assert.Equal(2, entries.Count);

        var m2 = Assert.Single(entries, e => e.GroupObjectId == 100);
        Assert.Equal(1, m2.PointCount);
        Assert.Equal(1, m2.SurfaceCount);
        Assert.Equal("m2", m2.SurfaceClass);
        Assert.Equal("m2.m2", m2.NearestModel);
        Assert.True(m2.Consistent);

        var wmo = Assert.Single(entries, e => e.GroupObjectId == 200);
        Assert.Equal(1, wmo.PointCount);
        Assert.Equal(1, wmo.SurfaceCount);
        Assert.Equal("wmo", wmo.SurfaceClass);
        Assert.Equal("wmo.wmo", wmo.NearestModel);
        Assert.True(wmo.Consistent);
    }

    [Fact]
    public void ClassMismatchIsNotConsistent()
    {
        var catalog = Catalog() with
        {
            ModelPlacements = [new AdtModelPlacement(0, "m2.m2", 1, new Vector3(300, 400, 0), Vector3.Zero, 1f)],
            WorldModelPlacements = [],
        };
        var entries = Pm4ObjectGroupCatalog.Build(Document(), catalog);

        var wmo = Assert.Single(entries, e => e.GroupObjectId == 200);
        Assert.Equal("wmo", wmo.SurfaceClass);
        Assert.Equal("m2", wmo.NearestModel is null ? "none" : "m2");
        Assert.False(wmo.Consistent);
    }

    [Fact]
    public void MissingPlacementsStillReturnsGroups()
    {
        var entries = Pm4ObjectGroupCatalog.Build(Document(), null);

        Assert.Equal(2, entries.Count);
        Assert.All(entries, e => Assert.Null(e.NearestModel));
        Assert.All(entries, e => Assert.False(e.Consistent));
    }

    private static Pm4ResearchDocument Document()
    {
        Vector3 RawSurface(float placementX, float placementY) =>
            new(Pm4CoordinateService.MapOrigin - placementX, Pm4CoordinateService.MapOrigin - placementY, 0);

        Vector3 RawMprl(float placementX, float placementY) =>
            new(Pm4CoordinateService.MapOrigin - placementY, 0, Pm4CoordinateService.MapOrigin - placementX);

        Vector3[] points =
        [
            RawSurface(99, 199), RawSurface(101, 199), RawSurface(101, 201), RawSurface(99, 201),
            RawSurface(299, 399), RawSurface(301, 399), RawSurface(301, 401), RawSurface(299, 401),
        ];

        var surfaces = new Pm4MsurEntry[]
        {
            new(3, 4, 0, 0, Vector3.UnitZ, 0, 0, 0, 0),          // M2 (Ck24Type 0)
            new(18, 4, 0, 0, Vector3.UnitZ, 0, 4, 0, 65u << 24), // WMO (Ck24Type 65)
        };

        var mprl = new Pm4MprlEntry[]
        {
            new(0, -1, 100, 0x8000, RawMprl(100, 200), 0, 0),
            new(0, -1, 200, 0x8000, RawMprl(300, 400), 0, 0),
        };

        var chunks = new Pm4KnownChunkSet(
            null, [], [], [], points, [0, 1, 2, 3, 4, 5, 6, 7], surfaces,
            [], mprl, [], null, [], [], [], []);

        return new Pm4ResearchDocument("synthetic", 2, [], chunks, []);
    }

    private static AdtPlacementCatalog Catalog() => new(
        "synthetic_obj0.adt",
        MapFileKind.AdtObj,
        ["m2.m2"],
        ["wmo.wmo"],
        [new AdtModelPlacement(0, "m2.m2", 1, new Vector3(100, 200, 0), Vector3.Zero, 1f)],
        [new AdtWorldModelPlacement(0, "wmo.wmo", 2, new Vector3(300, 400, 0), Vector3.Zero,
            new Vector3(299, 399, 0), new Vector3(301, 401, 0), 0)]);
}
