using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Services;
using Xunit;
using Xunit.Abstractions;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4UnauthoredPlacementResolverTests
{
    private readonly ITestOutputHelper _output;
    private const string DevelopmentMapDir = @"I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development";

    public Pm4UnauthoredPlacementResolverTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestCanRead_16_35_obj0()
    {
        string p = Path.Combine(DevelopmentMapDir, "development_16_35_obj0.adt");
        Assert.True(File.Exists(p), $"File {p} must exist");
        AdtPlacementCatalog cat = AdtPlacementReader.Read(p);
        _output.WriteLine($"16_35 WMO count: {cat.WorldModelPlacements.Count}");
        foreach (var w in cat.WorldModelPlacements)
            _output.WriteLine($"  WMO: {w.ModelPath} @ Z={w.Position.Z}");
        Assert.Single(cat.WorldModelPlacements);
    }

    [Fact]
    public void StormwindHarborSpillover_ResolvesOn16_36()
    {
        string pm4Path = Path.Combine(DevelopmentMapDir, "development_16_36.pm4");
        Assert.True(File.Exists(pm4Path), $"PM4 file '{pm4Path}' must exist.");

        Pm4TileResolutionResult result = Pm4UnauthoredPlacementResolver.ResolveTile(
            pm4Path,
            DevelopmentMapDir,
            neighborCatalogReader: path => AdtPlacementReader.Read(path));

        _output.WriteLine($"Resolved {result.ResolvedWmos.Count} WMOs on 16_36:");
        foreach (var wmo in result.ResolvedWmos)
        {
            _output.WriteLine($"  WMO: {wmo.ModelPath} @ Z={wmo.PlacementZ:F2} via {wmo.Method} (Note: {wmo.SourceNote})");
        }

        Assert.NotEmpty(result.ResolvedWmos);
        Pm4ResolvedObjectRecord harbor = result.ResolvedWmos[0];
        Assert.Contains("STORMWINDHARBOR", harbor.ModelPath, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(Pm4PlacementResolutionMethod.AdjacentSpillover, harbor.Method);
        Assert.InRange(harbor.PlacementZ, 101.90f, 102.00f);
    }

    [Fact]
    public void TaurenFortress_ResolvesOn13_50()
    {
        string pm4Path = Path.Combine(DevelopmentMapDir, "development_13_50.pm4");
        Assert.True(File.Exists(pm4Path), $"PM4 file '{pm4Path}' must exist.");

        Pm4TileResolutionResult result = Pm4UnauthoredPlacementResolver.ResolveTile(
            pm4Path,
            DevelopmentMapDir,
            neighborCatalogReader: path => AdtPlacementReader.Read(path));

        _output.WriteLine($"Resolved {result.ResolvedWmos.Count} WMOs on 13_50:");
        foreach (var wmo in result.ResolvedWmos)
        {
            _output.WriteLine($"  WMO: {wmo.ModelPath} @ Z={wmo.PlacementZ:F2} via {wmo.Method} (Note: {wmo.SourceNote})");
        }

        Assert.NotEmpty(result.ResolvedWmos);
        // Should find Tauren structures from 14_51 or shape library
        Assert.Contains(result.ResolvedWmos, w => w.ModelPath.Contains("TAUREN", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void SplitAdtObjWriter_RoundTripsThroughAdtPlacementReader()
    {
        string pm4Path = Path.Combine(DevelopmentMapDir, "development_16_36.pm4");
        Pm4TileResolutionResult result = Pm4UnauthoredPlacementResolver.ResolveTile(
            pm4Path,
            DevelopmentMapDir,
            neighborCatalogReader: path => AdtPlacementReader.Read(path));

        byte[] bytes = SplitAdtObjWriter.Build(result.Catalog);
        Assert.NotEmpty(bytes);

        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, "virtual_16_36_obj0.adt");
        stream.Position = 0;
        AdtPlacementCatalog readBack = AdtPlacementReader.Read(stream, summary);

        Assert.Equal(result.Catalog.WorldModelPlacements.Count, readBack.WorldModelPlacements.Count);
        Assert.Equal(result.Catalog.ModelPlacements.Count, readBack.ModelPlacements.Count);
        Assert.Equal(result.Catalog.WorldModelNames.Count, readBack.WorldModelNames.Count);

        if (result.Catalog.WorldModelPlacements.Count > 0)
        {
            Assert.Equal(result.Catalog.WorldModelPlacements[0].ModelPath, readBack.WorldModelPlacements[0].ModelPath);
            Assert.Equal(result.Catalog.WorldModelPlacements[0].Position.Z, readBack.WorldModelPlacements[0].Position.Z, 0.001f);
        }
    }
}
