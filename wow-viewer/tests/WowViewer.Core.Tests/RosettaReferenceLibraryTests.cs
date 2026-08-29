using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class RosettaReferenceLibraryTests
{
    [Fact]
    public void CorpusReader_ReadFromGenerationResult_DecodesAllPlacementsAndUniqueAssets()
    {
        var assets = new List<RosettaAssetEntry>
        {
            new(@"World\Generic\Human\Passive Doodads\Barrels\Barrel01.m2", RosettaAssetKind.Model, new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2f)),
            new(@"World\Generic\Human\Passive Doodads\Chairs\Chair01.m2", RosettaAssetKind.Model, new Vector3(-0.5f, -0.5f, 0f), new Vector3(0.5f, 0.5f, 1.2f)),
            new(@"World\wmo\Azeroth\Buildings\Stormwind\StormwindBarracks.wmo", RosettaAssetKind.WorldModel, new Vector3(-20f, -30f, -5f), new Vector3(20f, 30f, 25f)),
        };

        var options = new RosettaGeneratorOptions("RosettaTestMap", CellChunks: 8, LabelBandChunks: 2);
        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);

        RosettaCorpusData corpus = RosettaCorpusReader.ReadFromGenerationResult(result);

        Assert.Equal("RosettaTestMap", corpus.MapName);
        Assert.Equal(assets.Count, corpus.Placements.Count);
        Assert.Equal(assets.Count, corpus.UniqueAssets.Count);

        foreach (RosettaAssetEntry asset in assets)
        {
            string norm = asset.AssetPath.Replace('\\', '/').ToLowerInvariant();
            Assert.True(corpus.UniqueAssets.ContainsKey(norm));
            Assert.Contains(corpus.Placements, p => p.NormalizedPath == norm);
        }
    }

    [Fact]
    public void CorpusReader_BuildReferenceLibrary_ProducesValidIndexedLibrary()
    {
        var assets = new List<RosettaAssetEntry>
        {
            new(@"World\Generic\Human\Passive Doodads\Barrels\Barrel01.m2", RosettaAssetKind.Model, new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2f)),
            new(@"World\Generic\Human\Passive Doodads\Barrels\Barrel02.m2", RosettaAssetKind.Model, new Vector3(-1.2f, -1.2f, 0f), new Vector3(1.2f, 1.2f, 2.5f)),
            new(@"World\wmo\Azeroth\Buildings\Stormwind\StormwindBarracks.wmo", RosettaAssetKind.WorldModel, new Vector3(-20f, -30f, -5f), new Vector3(20f, 30f, 25f)),
        };

        var options = new RosettaGeneratorOptions("RosettaLibMap", CellChunks: 8);
        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaCorpusData corpus = RosettaCorpusReader.ReadFromGenerationResult(result);

        RosettaReferenceLibrary library = RosettaCorpusReader.BuildReferenceLibrary(corpus, "0.5.3.3368");

        Assert.Equal(3, library.TotalAssets);
        Assert.Equal(2, library.ModelCount);
        Assert.Equal(1, library.WorldModelCount);
        Assert.StartsWith("rosetta_lib_", library.LibraryId);
        Assert.Equal("0.5.3.3368", library.BuildLabel);

        // Verify O(1) Lookups
        string barrel1Path = @"world/generic/human/passive doodads/barrels/barrel01.m2";
        Assert.True(library.TryGetAssetByPath(barrel1Path, out RosettaReferenceAsset? refAsset));
        Assert.NotNull(refAsset);
        Assert.Equal("m2", refAsset.AssetKind);
        Assert.Equal(new Vector3(2f, 2f, 2f), refAsset.Span);
        Assert.True(library.TryGetAssetById(refAsset.AssetId, out _));
    }

    [Fact]
    public void ReferenceLibrary_JsonSerialization_RoundTripsAccurately()
    {
        var assets = new List<RosettaReferenceAsset>
        {
            new(
                "objlib_barrel01",
                @"World\Generic\Human\Passive Doodads\Barrels\Barrel01.m2",
                "world/generic/human/passive doodads/barrels/barrel01.m2",
                "m2",
                "0.5.3.3368",
                ["24_24"],
                new Pm4Bounds3(new Vector3(-1f, -1f, 0f), new Vector3(1f, 1f, 2f)),
                new Vector3(0f, 0f, 1f),
                new Vector3(2f, 2f, 2f),
                MathF.Sqrt(8f),
                8f,
                4f,
                [new Vector2(-1f, -1f), new Vector2(1f, -1f), new Vector2(1f, 1f), new Vector2(-1f, 1f)],
                1.0f,
                1.0f,
                Signals: new Dictionary<string, double> { ["boundsSpanX"] = 2.0, ["boundsVolume"] = 8.0 },
                ValidationTags: ["rosetta-calibration", "ground-truth"]),
            new(
                "objlib_barracks",
                @"World\wmo\Azeroth\Buildings\Stormwind\StormwindBarracks.wmo",
                "world/wmo/azeroth/buildings/stormwind/stormwindbarracks.wmo",
                "wmo",
                "0.5.3.3368",
                ["24_25"],
                new Pm4Bounds3(new Vector3(-20f, -30f, -5f), new Vector3(20f, 30f, 25f)),
                new Vector3(0f, 0f, 10f),
                new Vector3(40f, 60f, 30f),
                MathF.Sqrt(5200f),
                72000f,
                2400f,
                [new Vector2(-20f, -30f), new Vector2(20f, -30f), new Vector2(20f, 30f), new Vector2(-20f, 30f)],
                0.66667f,
                0.5f,
                Signals: new Dictionary<string, double> { ["boundsSpanX"] = 40.0, ["boundsVolume"] = 72000.0 },
                ValidationTags: ["rosetta-calibration", "ground-truth"])
        };

        var original = new RosettaReferenceLibrary("lib_test_123", "0.5.3.3368", assets);
        string tempJson = Path.Combine(Path.GetTempPath(), $"rosetta_lib_test_{Guid.NewGuid():N}.json");

        try
        {
            original.SaveToJson(tempJson);
            Assert.True(File.Exists(tempJson));

            RosettaReferenceLibrary loaded = RosettaReferenceLibrary.LoadFromJson(tempJson);

            Assert.Equal(original.LibraryId, loaded.LibraryId);
            Assert.Equal(original.BuildLabel, loaded.BuildLabel);
            Assert.Equal(original.TotalAssets, loaded.TotalAssets);
            Assert.Equal(original.ModelCount, loaded.ModelCount);
            Assert.Equal(original.WorldModelCount, loaded.WorldModelCount);

            Assert.True(loaded.TryGetAssetById("objlib_barrel01", out var loadedBarrel));
            Assert.NotNull(loadedBarrel);
            Assert.Equal(assets[0].Span, loadedBarrel.Span);
            Assert.Equal(assets[0].Volume, loadedBarrel.Volume);
            Assert.Equal(assets[0].FootprintArea, loadedBarrel.FootprintArea);
            Assert.Equal(assets[0].AspectRatioXY, loadedBarrel.AspectRatioXY, 0.001f);

            Assert.True(loaded.TryGetAssetById("objlib_barracks", out var loadedBarracks));
            Assert.NotNull(loadedBarracks);
            Assert.Equal(assets[1].Span, loadedBarracks.Span);
            Assert.Equal(assets[1].Volume, loadedBarracks.Volume);
        }
        finally
        {
            if (File.Exists(tempJson))
                File.Delete(tempJson);
        }
    }

    [Fact]
    public void ReferenceLibrary_ToAssetReferenceSignalRecords_ConvertsSeamlessly()
    {
        var asset = new RosettaReferenceAsset(
            "objlib_crate01",
            @"World\Generic\Human\Passive Doodads\Crates\Crate01.m2",
            "world/generic/human/passive doodads/crates/crate01.m2",
            "m2",
            "1.12.1",
            ["10_10"],
            new Pm4Bounds3(new Vector3(-0.8f, -0.8f, 0f), new Vector3(0.8f, 0.8f, 1.6f)),
            new Vector3(0f, 0f, 0.8f),
            new Vector3(1.6f, 1.6f, 1.6f),
            MathF.Sqrt(5.12f),
            4.096f,
            2.56f,
            [new Vector2(-0.8f, -0.8f), new Vector2(0.8f, -0.8f), new Vector2(0.8f, 0.8f), new Vector2(-0.8f, 0.8f)],
            1.0f,
            1.0f);

        var library = new RosettaReferenceLibrary("lib_crate", "1.12.1", [asset]);
        IReadOnlyList<Pm4AssetReferenceSignalRecord> records = library.ToAssetReferenceSignalRecords();

        Assert.Single(records);
        Pm4AssetReferenceSignalRecord rec = records[0];
        Assert.Equal("objlib_crate01", rec.AssetId);
        Assert.Equal("m2", rec.AssetKind);
        Assert.NotNull(rec.Bounds);
        Assert.Equal(new Vector3(1.6f, 1.6f, 1.6f), rec.Bounds.Span);
        Assert.Equal(4.096f, (float)rec.RenderOrCollisionSignals["boundsVolume"], 0.001f);
    }

    [Fact]
    public void ReferenceLibrary_SpatialAndBoundsQueries_FindsCorrectCandidates()
    {
        var assets = new List<RosettaReferenceAsset>
        {
            new("small_box", @"World\SmallBox.m2", "world/smallbox.m2", "m2", "1.0", ["0_0"],
                new Pm4Bounds3(new Vector3(-0.5f, -0.5f, 0f), new Vector3(0.5f, 0.5f, 1f)),
                Vector3.Zero, new Vector3(1f, 1f, 1f), MathF.Sqrt(2f), 1f, 1f, [], 1f, 1f),

            new("medium_box", @"World\MediumBox.m2", "world/mediumbox.m2", "m2", "1.0", ["0_0"],
                new Pm4Bounds3(new Vector3(-1.5f, -1.5f, 0f), new Vector3(1.5f, 1.5f, 3f)),
                Vector3.Zero, new Vector3(3f, 3f, 3f), MathF.Sqrt(18f), 27f, 9f, [], 1f, 1f),

            new("large_tower", @"World\LargeTower.wmo", "world/largetower.wmo", "wmo", "1.0", ["0_0"],
                new Pm4Bounds3(new Vector3(-10f, -10f, 0f), new Vector3(10f, 10f, 40f)),
                Vector3.Zero, new Vector3(20f, 20f, 40f), MathF.Sqrt(800f), 16000f, 400f, [], 1f, 2f),
        };

        var library = new RosettaReferenceLibrary("lib_boxes", "1.0", assets);

        // Query for something close to small_box (span ~ 1.05, 0.95, 1.02)
        var queryBoundsSmall = new Pm4Bounds3(new Vector3(-0.525f, -0.475f, 0f), new Vector3(0.525f, 0.475f, 1.02f));
        IReadOnlyList<RosettaCandidateMatch> matchesSmall = library.FindCandidatesByBounds(queryBoundsSmall, assetKind: "m2");

        Assert.NotEmpty(matchesSmall);
        Assert.Equal("small_box", matchesSmall[0].Asset.AssetId);
        Assert.True(matchesSmall[0].OverallScore > 0.90);

        // Query for something close to large_tower
        var queryBoundsTower = new Pm4Bounds3(new Vector3(-10.5f, -9.8f, 0f), new Vector3(10.2f, 10.1f, 39.5f));
        IReadOnlyList<RosettaCandidateMatch> matchesTower = library.FindCandidatesByBounds(queryBoundsTower, assetKind: "wmo");

        Assert.NotEmpty(matchesTower);
        Assert.Equal("large_tower", matchesTower[0].Asset.AssetId);
        Assert.True(matchesTower[0].OverallScore > 0.90);
    }

    [Fact]
    public void ReferenceLibrarySelfTest_SynthesizedCorpus_ExceedsTargetAccuracy()
    {
        // Build a diverse corpus of 30 distinct models and world models
        var assets = new List<RosettaReferenceAsset>();
        for (int i = 1; i <= 20; i++)
        {
            float width = 0.5f + (i * 0.4f);
            float depth = 0.6f + (i * 0.35f);
            float height = 1.0f + (i * 0.8f);

            var bounds = new Pm4Bounds3(new Vector3(-width / 2f, -depth / 2f, 0f), new Vector3(width / 2f, depth / 2f, height));
            Vector3 span = bounds.Span;
            float volume = span.X * span.Y * span.Z;
            float footprint = span.X * span.Y;
            float diag = MathF.Sqrt(span.X * span.X + span.Y * span.Y);

            assets.Add(new RosettaReferenceAsset(
                $"model_{i:D2}",
                $@"World\Generic\Model_{i:D2}.m2",
                $"world/generic/model_{i:d2}.m2",
                "m2",
                "0.5.3",
                ["20_20"],
                bounds,
                (bounds.Min + bounds.Max) * 0.5f,
                span,
                diag,
                volume,
                footprint,
                [],
                span.X / span.Y,
                span.Z / MathF.Max(span.X, span.Y)));
        }

        for (int i = 1; i <= 10; i++)
        {
            float width = 15f + (i * 5f);
            float depth = 20f + (i * 4f);
            float height = 10f + (i * 6f);

            var bounds = new Pm4Bounds3(new Vector3(-width / 2f, -depth / 2f, 0f), new Vector3(width / 2f, depth / 2f, height));
            Vector3 span = bounds.Span;
            float volume = span.X * span.Y * span.Z;
            float footprint = span.X * span.Y;
            float diag = MathF.Sqrt(span.X * span.X + span.Y * span.Y);

            assets.Add(new RosettaReferenceAsset(
                $"wmo_{i:D2}",
                $@"World\wmo\Building_{i:D2}.wmo",
                $"world/wmo/building_{i:d2}.wmo",
                "wmo",
                "0.5.3",
                ["20_21"],
                bounds,
                (bounds.Min + bounds.Max) * 0.5f,
                span,
                diag,
                volume,
                footprint,
                [],
                span.X / span.Y,
                span.Z / MathF.Max(span.X, span.Y)));
        }

        var library = new RosettaReferenceLibrary("lib_selftest_corpus", "0.5.3", assets);

        // Run self-test runner
        RosettaSelfTestResult result = RosettaReferenceLibrarySelfTest.Run(library, new RosettaSelfTestOptions());

        Assert.Equal(30, result.TotalTested);
        Assert.Equal(30, result.Top1Matches);
        Assert.Equal(100.0, result.Top1AccuracyPercent);
        Assert.True(result.Passed);
        Assert.Empty(result.Defects);
    }

    [Fact]
    public void ReferenceLibrarySelfTest_PerturbationResilience_RemainsAccurate()
    {
        // Build 20 distinct assets
        var assets = new List<RosettaReferenceAsset>();
        for (int i = 1; i <= 20; i++)
        {
            float width = 1f + (i * 2.0f);
            float depth = 1.2f + ((i % 4) * 3.0f);
            float height = 2f + ((i % 3) * 4.0f);

            var bounds = new Pm4Bounds3(new Vector3(-width / 2f, -depth / 2f, 0f), new Vector3(width / 2f, depth / 2f, height));
            Vector3 span = bounds.Span;
            float volume = span.X * span.Y * span.Z;
            float footprint = span.X * span.Y;
            float diag = MathF.Sqrt(span.X * span.X + span.Y * span.Y);

            assets.Add(new RosettaReferenceAsset(
                $"asset_{i:D2}",
                $@"World\Object_{i:D2}.m2",
                $"world/object_{i:d2}.m2",
                "m2",
                "0.5.3",
                ["10_10"],
                bounds,
                (bounds.Min + bounds.Max) * 0.5f,
                span,
                diag,
                volume,
                footprint,
                [],
                span.X / span.Y,
                span.Z / MathF.Max(span.X, span.Y)));
        }

        var library = new RosettaReferenceLibrary("lib_perturbed", "0.5.3", assets);

        // Run with 3% random jitter
        var options = new RosettaSelfTestOptions(
            Tolerance: 0.35f,
            IncludePerturbations: true,
            PerturbationJitterPercent: 0.03f);

        RosettaSelfTestResult result = RosettaReferenceLibrarySelfTest.Run(library, options);

        Assert.Equal(20, result.TotalTested);
        Assert.True(result.Top1AccuracyPercent >= 99.0);
        Assert.True(result.Passed);
    }
}
