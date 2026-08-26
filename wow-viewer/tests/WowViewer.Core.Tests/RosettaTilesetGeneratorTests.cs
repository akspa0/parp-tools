using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public class RosettaTilesetGeneratorTests
{
    private static RosettaAssetEntry Model(string path, float size) => new(
        path,
        RosettaAssetKind.Model,
        new Vector3(-size / 2f, -size / 2f, -size / 2f),
        new Vector3(size / 2f, size / 2f, size / 2f));

    [Fact]
    public void Generate_IsDeterministicAcrossRuns()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/azzinoth.mdx", 20f),
            Model("world/stormwind.wmo", 300f),
            Model("world/tree.mdx", 8f),
        };
        var options = new RosettaGeneratorOptions("Development");

        RosettaGenerationResult first = RosettaTilesetGenerator.Generate(assets, options);
        RosettaGenerationResult second = RosettaTilesetGenerator.Generate(assets, options);

        Assert.Equal(first.Placements.Count, second.Placements.Count);
        for (int i = 0; i < first.Placements.Count; i++)
        {
            Assert.Equal(first.Placements[i].Asset.AssetPath, second.Placements[i].Asset.AssetPath);
            Assert.Equal(first.Placements[i].WorldPosition, second.Placements[i].WorldPosition);
            Assert.Equal(first.Placements[i].UniqueId, second.Placements[i].UniqueId);
        }
    }

    [Fact]
    public void Generate_CellsDoNotOverlap()
    {
        var assets = Enumerable.Range(0, 40)
            .Select(i => Model($"asset_{i:0000}.mdx", 10f + i))
            .ToList();

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development"));

        foreach (var byTile in result.Placements.GroupBy(static p => (p.TileX, p.TileY)))
        {
            var rects = byTile.Select(static p => (
                MinU: p.WorldPosition.X - p.CellSize / 2f,
                MaxU: p.WorldPosition.X + p.CellSize / 2f,
                MinV: 17066.666f - p.WorldPosition.Y - p.CellSize / 2f,
                MaxV: 17066.666f - p.WorldPosition.Y + p.CellSize / 2f)).ToList();

            for (int i = 0; i < rects.Count; i++)
            {
                for (int j = i + 1; j < rects.Count; j++)
                {
                    bool overlaps =
                        rects[i].MinU < rects[j].MaxU && rects[j].MinU < rects[i].MaxU &&
                        rects[i].MinV < rects[j].MaxV && rects[j].MinV < rects[i].MaxV;
                    Assert.False(overlaps, $"Cells {i} and {j} overlap in tile {byTile.Key}.");
                }
            }
        }
    }

    [Fact]
    public void Generate_SkipsOccupiedTiles()
    {
        var assets = Enumerable.Range(0, 80).Select(i => Model($"asset_{i:0000}.mdx", 16f)).ToList();
        var occupied = new HashSet<(int X, int Y)> { (24, 25), (24, 26) };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development"), occupied);

        Assert.DoesNotContain(result.Tiles, static t => t.TileX == 24 && t.TileY is 25 or 26);
    }

    [Fact]
    public void GeneratedTile_RoundTripsThroughReader()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("creature/foobar.mdx", 12f),
            new("building/tavern.wmo", RosettaAssetKind.WorldModel,
                new Vector3(-50f, -30f, -20f), new Vector3(50f, 30f, 60f)),
        };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development"));
        RosettaTilePlan tile = Assert.Single(result.Tiles);

        byte[] adtBytes = LkAdtWriter.Build(tile.AdtData);
        LkAdtData readBack = LkAdtReader.Read(adtBytes, null, null, tile.TileX, tile.TileY);

        Assert.Equal(1, readBack.ModelPlacements.Count);
        Assert.Equal(1, readBack.WorldModelPlacements.Count);
        Assert.Contains("creature/foobar.mdx", readBack.ModelNames);
        Assert.Contains("building/tavern.wmo", readBack.WorldModelNames);

        LkMddfEntry mddf = readBack.ModelPlacements[0];
        RosettaPlacementRecord expected = result.Placements.Single(static p => p.Asset.Kind == RosettaAssetKind.Model);
        Assert.Equal(expected.UniqueId, mddf.UniqueId);
        Assert.Equal(expected.WorldPosition, mddf.Position);

        // Every chunk carries MCCV data and at least one vertex was painted bright.
        Assert.All(readBack.Chunks, static c => Assert.NotNull(c.MccvColors));
        Assert.Contains(readBack.Chunks.SelectMany(static c => c.MccvColors!), static b => b == 255);
    }

    [Fact]
    public void SanitizeLabel_MapsUnsupportedCharacters()
    {
        string label = RosettaTilesetGenerator.SanitizeLabel(@"world\kalimdor/azshara (orgrimmar).mdx");
        Assert.Matches("^[A-Z0-9_.\\-]+$", label);
    }
}
