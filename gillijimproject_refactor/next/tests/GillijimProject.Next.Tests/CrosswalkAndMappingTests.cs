using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.Next.Core.Adapters.Dbcd;
using GillijimProject.Next.Core.Services;
using Xunit;

namespace GillijimProject.Next.Tests;

public class CrosswalkAndMappingTests
{
    private static string ProjectDir => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
    private static string TestDataRoot => Path.GetFullPath(Path.Combine(ProjectDir, "..", "test_data"));

    private static bool TryGetDbcPaths(out string alphaArea, out string alphaMap, out string lkArea, out string lkMap)
    {
        var alphaDir = Path.Combine(TestDataRoot, "alpha", "DBC");
        var lkDir = Path.Combine(TestDataRoot, "wotlk", "DBC");
        alphaArea = Path.Combine(alphaDir, "AreaTable.dbc");
        alphaMap = Path.Combine(alphaDir, "Map.dbc");
        lkArea = Path.Combine(lkDir, "AreaTable.dbc");
        lkMap = Path.Combine(lkDir, "Map.dbc");
        return File.Exists(alphaArea) && File.Exists(alphaMap) && File.Exists(lkArea) && File.Exists(lkMap);
    }

    [Fact]
    public void CanLoadDbcdAndCountsOrSkip()
    {
        if (!TryGetDbcPaths(out var alphaArea, out var _, out var lkArea, out var _))
        {
            Console.WriteLine("[skip] test_data not found for Map/AreaTable; skipping DBCD load test.");
            return;
        }

        var provider = new DbcdAreaTableProvider(alphaArea, lkArea);
        provider.EnsureLoaded();

        Assert.True(provider.AlphaRowCount > 0, "Alpha AreaTable row count should be > 0");
        Assert.True(provider.LkRowCount > 0, "LK AreaTable row count should be > 0");
        Assert.True(provider.AlphaMapRowCount > 0, "Alpha Map row count should be > 0");
        Assert.True(provider.LkMapRowCount > 0, "LK Map row count should be > 0");
    }

    [Fact]
    public void MapCrosswalkPartitionsAlphaMapsOrSkip()
    {
        if (!TryGetDbcPaths(out var alphaArea, out var _, out var lkArea, out var _))
        {
            Console.WriteLine("[skip] test_data not found for Map/AreaTable; skipping map crosswalk partition test.");
            return;
        }

        var provider = new DbcdAreaTableProvider(alphaArea, lkArea);
        var translator = new AreaIdTranslator(provider);
        translator.BuildMapping(areaOverridesJsonPath: null, mapOverridesJsonPath: null);

        var alphaMapIds = provider.GetAlphaMapTable().Select(kv => kv.Key).ToHashSet();
        var matched = translator.GetMapCrosswalk().Keys.ToHashSet();
        var amb = translator.GetMapAmbiguousAlpha().ToHashSet();
        var un = translator.GetMapUnmatchedAlpha().ToHashSet();

        // Disjointness
        Assert.Empty(matched.Intersect(amb));
        Assert.Empty(matched.Intersect(un));
        Assert.Empty(amb.Intersect(un));

        // Union equals alpha set
        var union = matched.Union(amb).Union(un).ToHashSet();
        Assert.True(union.SetEquals(alphaMapIds), "Each alpha MapID must be either matched, ambiguous, or unmatched");

        // Counts agree with properties
        Assert.Equal(matched.Count, translator.MapMatchedCount);
        Assert.Equal(amb.Count, translator.MapAmbiguousCount);
        Assert.Equal(un.Count, translator.MapUnmatchedCount);
    }

    [Fact]
    public void AreaMappingPartitionsAlphaAreasOrSkip()
    {
        if (!TryGetDbcPaths(out var alphaArea, out var _, out var lkArea, out var _))
        {
            Console.WriteLine("[skip] test_data not found for Map/AreaTable; skipping area mapping partition test.");
            return;
        }

        var provider = new DbcdAreaTableProvider(alphaArea, lkArea);
        var translator = new AreaIdTranslator(provider);
        translator.BuildMapping(areaOverridesJsonPath: null, mapOverridesJsonPath: null);

        var alphaAreaIds = provider.GetAlphaAreaTable().Select(kv => kv.Key).ToHashSet();
        var matched = translator.GetMapping().Keys.ToHashSet();
        var amb = translator.GetAmbiguousAlpha().ToHashSet();
        var un = translator.GetUnmatchedAlpha().ToHashSet();

        // Disjointness
        Assert.Empty(matched.Intersect(amb));
        Assert.Empty(matched.Intersect(un));
        Assert.Empty(amb.Intersect(un));

        // Union equals alpha set
        var union = matched.Union(amb).Union(un).ToHashSet();
        Assert.True(union.SetEquals(alphaAreaIds), "Each alpha AreaID must be either matched, ambiguous, or unmatched");

        // Counts agree with properties
        Assert.Equal(matched.Count, translator.MatchedCount);
        Assert.Equal(amb.Count, translator.AmbiguousCount);
        Assert.Equal(un.Count, translator.UnmatchedCount);
    }
}
