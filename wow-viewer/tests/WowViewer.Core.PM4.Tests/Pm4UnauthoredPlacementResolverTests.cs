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

    [Fact]
    public void ComputeFootprintYawDegrees_RecoversOrientation()
    {
        // Elongated points along X axis (yaw ~ 0 deg)
        var horizontalPoints = new List<Vector3>
        {
            new(-10f, 0f, 0f),
            new(10f, 0f, 0f),
            new(-10f, 1f, 0f),
            new(10f, 1f, 0f),
        };
        float yaw0 = Pm4UnauthoredPlacementResolver.ComputeFootprintYawDegrees(horizontalPoints);
        Assert.InRange(MathF.Abs(yaw0), 0f, 1f);

        // Elongated points along Y axis (yaw ~ 90 deg or -90 deg)
        var verticalPoints = new List<Vector3>
        {
            new(0f, -10f, 0f),
            new(0f, 10f, 0f),
            new(1f, -10f, 0f),
            new(1f, 10f, 0f),
        };
        float yaw90 = MathF.Abs(Pm4UnauthoredPlacementResolver.ComputeFootprintYawDegrees(verticalPoints));
        Assert.InRange(yaw90, 89f, 91f);

        // Elongated points along 45 degree line (X == Y)
        var diagPoints = new List<Vector3>
        {
            new(-10f, -10f, 0f),
            new(10f, 10f, 0f),
            new(-10f + 0.5f, -10f - 0.5f, 0f),
            new(10f + 0.5f, 10f - 0.5f, 0f),
        };
        float yaw45 = Pm4UnauthoredPlacementResolver.ComputeFootprintYawDegrees(diagPoints);
        Assert.InRange(yaw45, 44f, 46f);
    }

    [Fact]
    public void Test_63_05_Resolution()
    {
        string pm4Path = Path.Combine(DevelopmentMapDir, "development_63_05.pm4");
        Assert.True(File.Exists(pm4Path), $"PM4 file '{pm4Path}' must exist.");

        Pm4TileResolutionResult result = Pm4UnauthoredPlacementResolver.ResolveTile(
            pm4Path,
            DevelopmentMapDir,
            neighborCatalogReader: path => AdtPlacementReader.Read(path));

        _output.WriteLine($"Resolved {result.ResolvedWmos.Count} WMOs on 63_05:");
        foreach (var wmo in result.ResolvedWmos)
        {
            _output.WriteLine($"  WMO: {wmo.ModelPath} @ Pos={wmo.Position} Rot={wmo.Rotation} via {wmo.Method} (Note: {wmo.SourceNote})");
        }
        _output.WriteLine($"Catalog MODF count: {result.Catalog.WorldModelPlacements.Count}");
        foreach (var modf in result.Catalog.WorldModelPlacements)
        {
            _output.WriteLine($"  MODF: {modf.ModelPath} @ Pos={modf.Position} Rot={modf.Rotation}");
        }
    }
}

