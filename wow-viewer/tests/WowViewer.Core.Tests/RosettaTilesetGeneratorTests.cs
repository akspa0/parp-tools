using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public class RosettaTilesetGeneratorTests
{
    private const float MapOrigin = 17066.666f;
    private const float TileSize = RosettaGeneratorOptions.TileSize;
    private const float ChunkSize = RosettaGeneratorOptions.ChunkSize;

    /// <summary>The single map a test corpus produces. Fails loudly if the layout split it.</summary>
    private static RosettaMapPlan SingleMap(RosettaGenerationResult result) => Assert.Single(result.Maps);

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
            Assert.Equal(first.Placements[i].RawPosition, second.Placements[i].RawPosition);
            Assert.Equal(first.Placements[i].RendererPosition, second.Placements[i].RendererPosition);
            Assert.Equal(first.Placements[i].CellU, second.Placements[i].CellU);
            Assert.Equal(first.Placements[i].CellV, second.Placements[i].CellV);
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
            assets, new RosettaGeneratorOptions("Development", GroupByDesignkit: false));

        foreach (var byTile in result.Placements.GroupBy(static p => (p.TileX, p.TileY)))
        {
            var rects = byTile.Select(static p => (
                MinU: p.CellU,
                MaxU: p.CellU + p.CellSize,
                MinV: p.CellV,
                MaxV: p.CellV + p.CellSize)).ToList();

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
    public void Generate_CellsAreUniformAndChunkAligned()
    {
        // The whole point of the grid rewrite: one tile never mixes cell sizes, and every cell edge
        // lands on a chunk boundary. Assets of wildly different footprints must not perturb it.
        var assets = Enumerable.Range(0, 60)
            .Select(i => Model($"asset_{i:0000}.mdx", 4f + (i % 12) * 12f))
            .ToList();

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", GroupByDesignkit: false));

        Assert.NotEmpty(result.Placements);
        foreach (var byTile in result.Placements.GroupBy(static p => (p.TileX, p.TileY)))
        {
            float cellSize = byTile.First().CellSize;
            int cellsPerAxis = (int)MathF.Round(TileSize / cellSize);
            Assert.True(byTile.Count() <= cellsPerAxis * cellsPerAxis,
                $"Tile {byTile.Key} holds {byTile.Count()} cells but only fits {cellsPerAxis * cellsPerAxis}.");

            foreach (RosettaPlacementRecord p in byTile)
            {
                Assert.Equal(cellSize, p.CellSize, 3);
                AssertChunkAligned(p.CellU, nameof(p.CellU));
                AssertChunkAligned(p.CellV, nameof(p.CellV));
                Assert.InRange(p.CellU, 0f, TileSize - cellSize + 0.01f);
                Assert.InRange(p.CellV, 0f, TileSize - cellSize + 0.01f);
            }
        }

        static void AssertChunkAligned(float value, string name)
        {
            float chunks = value / ChunkSize;
            Assert.True(MathF.Abs(chunks - MathF.Round(chunks)) < 1e-3f,
                $"{name} {value:F3} is not on a chunk boundary ({chunks:F3} chunks).");
        }
    }

    [Fact]
    public void Generate_SkipsOccupiedTiles()
    {
        var assets = Enumerable.Range(0, 80).Select(i => Model($"asset_{i:0000}.mdx", 16f)).ToList();
        var occupied = new HashSet<(int X, int Y)> { (25, 24), (26, 24) };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", GroupByDesignkit: false), occupied);

        Assert.NotEmpty(SingleMap(result).Tiles);
        Assert.DoesNotContain(SingleMap(result).Tiles, static t => t.TileY == 24 && t.TileX is 25 or 26);
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
            assets, new RosettaGeneratorOptions("Development", GroupByDesignkit: false));
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        byte[] adtBytes = LkAdtWriter.Build(RosettaTilesetGenerator.BuildTileAdt(map.MapName, tile));
        LkAdtData readBack = LkAdtReader.Read(adtBytes, null, null, tile.TileX, tile.TileY);

        Assert.Equal(1, readBack.ModelPlacements.Count);
        Assert.Equal(1, readBack.WorldModelPlacements.Count);
        Assert.Contains("creature\\foobar.mdx", readBack.ModelNames);
        Assert.Contains("building\\tavern.wmo", readBack.WorldModelNames);

        // LkAdtWriter/LkAdtReader both speak RENDERER coordinates, so the round trip must return
        // the renderer position, not the raw canvas one.
        LkMddfEntry mddf = readBack.ModelPlacements[0];
        RosettaPlacementRecord model = result.Placements.Single(static p => p.Asset.Kind == RosettaAssetKind.Model);
        Assert.Equal(model.UniqueId, mddf.UniqueId);
        Assert.Equal(model.RendererPosition.X, mddf.Position.X, 2f);
        Assert.Equal(model.RendererPosition.Y, mddf.Position.Y, 2f);

        LkModfEntry modf = readBack.WorldModelPlacements[0];
        RosettaPlacementRecord wmo = result.Placements.Single(static p => p.Asset.Kind == RosettaAssetKind.WorldModel);
        Assert.Equal(wmo.RendererPosition.X, modf.Position.X, 2f);
        Assert.Equal(wmo.RendererPosition.Y, modf.Position.Y, 2f);
        Assert.True(modf.BoundsMin.X <= modf.Position.X && modf.Position.X <= modf.BoundsMax.X,
            "MODF bounds must straddle the placement position on X.");
        Assert.True(modf.BoundsMin.Y <= modf.Position.Y && modf.Position.Y <= modf.BoundsMax.Y,
            "MODF bounds must straddle the placement position on Y.");

        // Painted chunks carry MCCV and at least one vertex was driven bright.
        Assert.Contains(readBack.Chunks, static c => c.MccvColors is not null);
        Assert.Contains(readBack.Chunks.Where(static c => c.MccvColors is not null).SelectMany(static c => c.MccvColors!), static b => b == 255);

        // The chunk containing each placement references it via MCRF.
        int refSum = readBack.Chunks.Sum(static c => c.DoodadRefs.Count) + readBack.Chunks.Sum(static c => c.WorldModelRefs.Count);
        Assert.True(refSum >= 2, $"Expected MCRF refs for both placements, found {refSum}.");
    }

    [Fact]
    public void WrittenBytes_PutEveryObjectOnItsOwnTerrain()
    {
        // The regression guard for the "objects run horizontally, tiles run vertically" defect:
        // LkAdtWriter applies the MapOrigin flip itself, so feeding it raw canvas coordinates
        // double-flips them and throws every object onto the perpendicular axis. This test reads the
        // MDDF/MODF bytes off the built file and decodes them with the VIEWER's exact rule
        // (rendererX = MapOrigin - raw[+16], rendererY = MapOrigin - raw[+8]).
        var assets = new List<RosettaAssetEntry>();
        for (int i = 0; i < 24; i++)
            assets.Add(Model($"model_{i:0000}.mdx", 10f + i));
        for (int i = 0; i < 24; i++)
        {
            float half = 20f + i * 3f;
            assets.Add(new RosettaAssetEntry($"world_{i:0000}.wmo", RosettaAssetKind.WorldModel,
                new Vector3(-half, -half, -5f), new Vector3(half, half, 40f)));
        }

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", StartTileX: 31, StartTileY: 17, GroupByDesignkit: false));

        int checkedEntries = 0;
        RosettaMapPlan map = SingleMap(result);
        foreach (RosettaTilePlan tile in map.Tiles)
        {
            byte[] bytes = LkAdtWriter.Build(RosettaTilesetGenerator.BuildTileAdt(map.MapName, tile));
            checkedEntries += AssertEntriesOnTile(bytes, "MDDF", 36, tile.TileX, tile.TileY);
            checkedEntries += AssertEntriesOnTile(bytes, "MODF", 64, tile.TileX, tile.TileY);
        }

        Assert.Equal(result.Placements.Count, checkedEntries);
    }

    [Fact]
    public void Generate_LabelLinesFitTheirCellAtLegibleFontSize()
    {
        var assets = Enumerable.Range(0, 30)
            .Select(i => Model($"verylongassetname_{i:0000}_withsuffix.mdx", 8f))
            .ToList();

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", GroupByDesignkit: false));

        Assert.NotEmpty(result.Placements);
        float subCell = RosettaTextPainter.SubCellFor(ChunkSize);
        foreach (RosettaPlacementRecord p in result.Placements)
        {
            // A font pixel smaller than one sub-cell falls between terrain vertices and scrambles
            // the glyph; the size must be an exact multiple of it.
            float pixels = p.LabelPixelMeters / subCell;
            Assert.True(pixels >= 1f && MathF.Abs(pixels - MathF.Round(pixels)) < 1e-3f,
                $"Label pixel {p.LabelPixelMeters:F3}m is {pixels:F3} sub-cells; must be a whole multiple >= 1.");

            Assert.NotEmpty(p.LabelLines);
            foreach (string line in p.LabelLines)
            {
                float width = RosettaTextPainter.MeasureWidthMeters(line, p.LabelPixelMeters);
                Assert.True(width <= p.CellSize + 0.01f,
                    $"Label line '{line}' width {width:F1}m exceeds cell {p.CellSize:F1}m.");
            }
        }
    }

    [Fact]
    public void WrapLabel_WrapsInsteadOfTruncatingToNonsense()
    {
        IReadOnlyList<string> lines = RosettaTilesetGenerator.WrapLabel("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10, 3);
        Assert.Equal(["ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZ"], lines);

        // An overflowing name is elided in the MIDDLE so the trailing discriminator survives -
        // a tail cut would make every ..._01 / ..._02 variant paint identically.
        IReadOnlyList<string> clipped = RosettaTilesetGenerator.WrapLabel(
            "ICECROWN_WALL_SEMICIRCLE_PIECE_02_LONG_HOLLOW", 10, 3);
        Assert.Equal(3, clipped.Count);
        string joined = string.Concat(clipped);
        Assert.Equal(30, joined.Length);
        Assert.StartsWith("ICECROWN_", joined);
        Assert.EndsWith("LONG_HOLLOW", joined);
        Assert.Contains('-', joined);
    }

    [Fact]
    public void PaintTile_GlyphPixelsLandOnTerrainVertices()
    {
        // With the run origin on the lattice and a one-sub-cell font pixel, INNER vertices sit at
        // font-pixel centres, so a lit pixel drives exactly one vertex to full ink. 'I' in the 5x7
        // font lights 15 pixels. OUTER vertices sit at pixel corners and must come back as partial
        // coverage - that is the antialiasing, and if it is missing the glyph is a stencil again.
        LkAdtData blank = BlankAdtFactory.CreateBlank("Development", 10, 10);
        float pixel = RosettaTextPainter.SubCellFor(ChunkSize);

        IReadOnlyList<LkMcnkData> painted = RosettaTextPainter.PaintTile(
            blank.Chunks, [], [new RosettaLabel("I", 0f, 0f, pixel)], ChunkSize);

        List<byte> values = painted
            .Where(static c => c.MccvColors is not null)
            .SelectMany(static c => Enumerable.Range(0, c.MccvColors!.Length / 4).Select(i => c.MccvColors[i * 4]))
            .ToList();

        Assert.Equal(15, values.Count(static v => v == 255));
        Assert.True(values.Count(static v => v > RosettaTextPainter.NeutralChannel && v < 255) >= 12,
            "Expected antialiased partial-coverage vertices around the glyph.");
        Assert.Single(painted.Where(static c => c.MccvColors is not null));
    }

    [Fact]
    public void PaintTile_AntialiasedEdgesBlendTowardTheLabelBackground()
    {
        // Partial coverage must blend toward the label's own background, not toward neutral, or the
        // text haloes against its plate.
        LkAdtData blank = BlankAdtFactory.CreateBlank("Development", 10, 10);
        float pixel = RosettaTextPainter.SubCellFor(ChunkSize);
        const byte plate = 28;

        IReadOnlyList<LkMcnkData> painted = RosettaTextPainter.PaintTile(
            blank.Chunks,
            [new RosettaMccvRect(0f, 0f, ChunkSize, ChunkSize, plate, plate, plate)],
            [new RosettaLabel("I", 0f, 0f, pixel, 255, 255, 255, plate, plate, plate)],
            ChunkSize);

        byte[] mccv = Assert.Single(painted.Where(static c => c.MccvColors is not null)).MccvColors!;
        List<byte> values = Enumerable.Range(0, mccv.Length / 4).Select(i => mccv[i * 4]).ToList();

        Assert.Equal(15, values.Count(static v => v == 255));

        // A lightly covered vertex must land between the PLATE and the ink. Blending from neutral
        // instead would floor every antialiased vertex at 127 and halo the text against its plate.
        Assert.Contains(values, v => v > plate && v < RosettaTextPainter.NeutralChannel);
    }

    [Fact]
    public void PaintTile_LeavesUntouchedChunksWithoutMccv()
    {
        LkAdtData blank = BlankAdtFactory.CreateBlank("Development", 10, 10);
        float pixel = RosettaTextPainter.SubCellFor(ChunkSize);

        IReadOnlyList<LkMcnkData> painted = RosettaTextPainter.PaintTile(
            blank.Chunks, [new RosettaMccvRect(0f, 0f, ChunkSize / 2f, ChunkSize / 2f, 30, 30, 30)],
            [new RosettaLabel("A", 0f, 0f, pixel)], ChunkSize);

        Assert.Equal(255, painted.Count(static c => c.MccvColors is null));
    }

    [Fact]
    public void Generate_TilesFormASquareBlockAtTheStartTile()
    {
        var assets = Enumerable.Range(0, 300).Select(i => Model($"asset_{i:0000}.mdx", 8f)).ToList();
        var options = new RosettaGeneratorOptions("Development", StartTileX: 10, StartTileY: 10, GroupByDesignkit: false);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);

        // 300 assets / 4 cells per tile = 75 tiles, so a 9x9 block anchored at (10,10). Predictable
        // from the start tile alone - not a strip that wraps at column 63 and lands wherever the
        // asset count happens to put it.
        Assert.Equal(75, SingleMap(result).Tiles.Count);
        Assert.Equal(9, SingleMap(result).BlockSide);
        Assert.Equal(10, SingleMap(result).BlockOriginX);
        Assert.Equal(10, SingleMap(result).BlockOriginY);

        Assert.All(SingleMap(result).Tiles, static t =>
        {
            Assert.InRange(t.TileX, 10, 18);
            Assert.InRange(t.TileY, 10, 18);
        });

        var expected = Enumerable.Range(0, 75)
            .Select(static i => (X: 10 + (i % 9), Y: 10 + (i / 9)))
            .OrderBy(static t => t.Y).ThenBy(static t => t.X)
            .ToList();
        Assert.Equal(expected, SingleMap(result).Tiles
            .Select(static t => (X: t.TileX, Y: t.TileY))
            .OrderBy(static t => t.Y).ThenBy(static t => t.X)
            .ToList());
    }

    [Fact]
    public void Generate_PullsTheBlockBackWhenItWouldRunOffTheMap()
    {
        var assets = Enumerable.Range(0, 300).Select(i => Model($"asset_{i:0000}.mdx", 8f)).ToList();

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", StartTileX: 60, StartTileY: 60, GroupByDesignkit: false));

        // A 9-wide block cannot start at 60, so the origin clamps to 64-9=55 and the block still fits.
        Assert.Equal(9, SingleMap(result).BlockSide);
        Assert.Equal(55, SingleMap(result).BlockOriginX);
        Assert.Equal(55, SingleMap(result).BlockOriginY);
        Assert.All(SingleMap(result).Tiles, static t =>
        {
            Assert.InRange(t.TileX, 0, 63);
            Assert.InRange(t.TileY, 0, 63);
        });
    }

    [Fact]
    public void Generate_PlacementsDecodeOntoTheirTiles()
    {
        // Mechanical alignment proof: decode every placement with the viewer's exact rule
        // (renderer = (MapOrigin - rawY, MapOrigin - rawX)) and assert it lands inside its
        // tile's renderer-space bounds. No eyeballing required.
        var assets = Enumerable.Range(0, 60).Select(i => Model($"asset_{i:0000}.mdx", 10f + i)).ToList();

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", StartTileX: 10, StartTileY: 10, GroupByDesignkit: false));

        foreach (RosettaPlacementRecord p in result.Placements)
        {
            AssertInsideTile(MapOrigin - p.RawPosition.Y, MapOrigin - p.RawPosition.X, p.TileX, p.TileY, p.Asset.AssetPath);
            AssertInsideTile(p.RendererPosition.X, p.RendererPosition.Y, p.TileX, p.TileY, p.Asset.AssetPath);
        }
    }

    [Fact]
    public void Generate_RejectsCellSizesThatDoNotTileAnAdt()
    {
        var assets = new List<RosettaAssetEntry> { Model("a.mdx", 8f) };

        Assert.Throws<ArgumentOutOfRangeException>(() => RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", CellChunks: 5)));
        Assert.Throws<ArgumentOutOfRangeException>(() => RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", CellChunks: 4, LabelBandChunks: 4)));
    }

    [Fact]
    public void Generate_ExcludesAssetsWithNonFiniteBounds()
    {
        // Camera-path M2s and some effect models carry degenerate bounds. They must be named as
        // such, and must never reach the layout arithmetic.
        var assets = new List<RosettaAssetEntry>
        {
            new("cameras/flyby.m2", RosettaAssetKind.Model,
                new Vector3(float.MaxValue, float.MaxValue, float.MaxValue),
                new Vector3(float.MinValue, float.MinValue, float.MinValue)),
            new("cameras/broken.m2", RosettaAssetKind.Model,
                new Vector3(float.NaN, 0f, 0f), new Vector3(1f, 1f, 1f)),
            Model("world/tree.mdx", 8f),
        };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development"));

        Assert.Single(result.Placements);
        Assert.Equal(2, result.Exclusions.Count);
        Assert.All(result.Exclusions, static e => Assert.Contains("not finite", e.Reason));
    }

    [Fact]
    public void Generate_KeepsEachDesignkitOnItsOwnTiles()
    {
        // The source folder IS the designkit. A tile must never mix two kits, or the index cannot
        // say where a kit lives without listing individual cells.
        var assets = new List<RosettaAssetEntry>();
        foreach (string kit in new[] { "world/azeroth/elwynn/trees", "world/kalimdor/durotar/rocks", "world/generic/human" })
        {
            for (int i = 0; i < 3; i++)
                assets.Add(Model($"{kit}/asset_{i:00}.m2", 10f));
        }

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development"));

        RosettaMapPlan map = SingleMap(result);
        Assert.Equal(3, map.Designkits.Count);
        Assert.Equal(9, map.Placements.Count);

        foreach (var byTile in map.Placements.GroupBy(static p => (p.TileX, p.TileY)))
        {
            var kitsOnTile = byTile
                .Select(static p => RosettaTilesetGenerator.DesignkitOf(p.Asset.AssetPath))
                .Distinct()
                .ToList();
            Assert.Single(kitsOnTile);
        }

        // Every index entry must name exactly the tiles its own placements landed on.
        foreach (RosettaDesignkitPlan kit in map.Designkits)
        {
            var actual = map.Placements
                .Where(p => RosettaTilesetGenerator.DesignkitOf(p.Asset.AssetPath) == kit.Kit)
                .Select(static p => (p.TileX, p.TileY))
                .Distinct()
                .OrderBy(static t => t.TileX).ThenBy(static t => t.TileY)
                .ToList();

            Assert.Equal(actual, kit.Tiles.Distinct().OrderBy(static t => t.TileX).ThenBy(static t => t.TileY).ToList());
            Assert.Equal(map.MapName, kit.MapName);
            Assert.Equal(3, kit.AssetCount);
        }
    }

    [Fact]
    public void Generate_SplitsAcrossMapsAndKeepsEveryKitWhole()
    {
        // One 64x64 map cannot hold a full client's corpus, so kits spill onto numbered maps. A kit
        // must never be split across two of them, which is what makes the index entry single-valued.
        var assets = new List<RosettaAssetEntry>();
        for (int kit = 0; kit < 6; kit++)
        {
            for (int i = 0; i < 5; i++)
                assets.Add(Model($"world/kit_{kit:00}/asset_{i:00}.m2", 10f));
        }

        // Each kit needs 2 tiles (5 assets at 4 cells per tile), so a 3-tile budget holds one kit.
        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", MaxTilesPerMap: 3));

        Assert.Equal(6, result.Maps.Count);
        Assert.Equal(30, result.Placements.Count);
        Assert.Equal(6, result.Designkits.Count);

        // Multi-map runs get numbered names; a single-map run keeps the bare name.
        Assert.Equal(
            Enumerable.Range(0, 6).Select(static i => $"Development{i:00}").ToList(),
            result.Maps.Select(static m => m.MapName).ToList());

        foreach (RosettaDesignkitPlan kit in result.Designkits)
        {
            RosettaMapPlan owner = Assert.Single(result.Maps.Where(m => m.MapName == kit.MapName));
            Assert.All(kit.Tiles, t => Assert.Contains(owner.Tiles, tile => tile.TileX == t.TileX && tile.TileY == t.TileY));
        }

        // Every map re-anchors at the start tile, so uniqueIds are what keep placements distinct.
        Assert.Equal(result.Placements.Count, result.Placements.Select(static p => p.UniqueId).Distinct().Count());
    }

    [Fact]
    public void Generate_SplitsAKitTooBigForOneMapInsteadOfFailing()
    {
        // A few real kits (item\objectcomponents, for one) are larger than a whole map at a small
        // per-map budget. Splitting them beats dead-ending the run; the index then lists the kit
        // once per map it spans.
        var assets = Enumerable.Range(0, 40)
            .Select(i => Model($"item/objectcomponents/piece_{i:000}.m2", 10f))
            .ToList();

        // 40 assets at 4 cells per tile is 10 tiles; a 3-tile budget forces four parts.
        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", MaxTilesPerMap: 3));

        Assert.Equal(40, result.Placements.Count);
        Assert.True(result.Maps.Count >= 4, $"Expected the kit to span several maps, got {result.Maps.Count}.");

        // Every part keeps the kit's name, and every map it spans stays inside its own budget.
        Assert.All(result.Designkits, static k => Assert.Equal(@"item\objectcomponents", k.Kit));
        Assert.All(result.Maps, static m => Assert.True(m.Tiles.Count <= 3, $"Map {m.MapName} holds {m.Tiles.Count} tiles, over budget."));

        // Nothing is lost or duplicated in the split.
        Assert.Equal(40, result.Designkits.Sum(static k => k.AssetCount));
        Assert.Equal(40, result.Placements.Select(static p => p.UniqueId).Distinct().Count());
        Assert.Equal(
            assets.Select(static a => a.AssetPath).OrderBy(static x => x, StringComparer.Ordinal).ToList(),
            result.Placements.Select(static p => p.Asset.AssetPath).OrderBy(static x => x, StringComparer.Ordinal).ToList());
    }

    [Fact]
    public void Generate_WritesTheConfiguredGroundTextureIntoEveryTile()
    {
        // The layer-0 texture must be a path the TARGET client ships. It is era-dependent, so it is
        // a setting: a hard-coded one silently produces tiles naming a texture that does not exist,
        // which is how the corpus ended up rendering flat grey.
        const string texture = @"tileset\generic\black.blp";
        var assets = new List<RosettaAssetEntry> { Model("world/kit/tree.m2", 10f) };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", GroundTexture: texture));

        RosettaMapPlan map = SingleMap(result);
        Assert.Equal(texture, map.GroundTexture);

        RosettaTilePlan tile = Assert.Single(map.Tiles);
        LkAdtData adt = RosettaTilesetGenerator.BuildTileAdt(map.MapName, tile, map.GroundTexture);
        Assert.Equal(texture, adt.TextureNames[0]);

        // And it must survive to the bytes, since that is what a client reads.
        byte[] bytes = LkAdtWriter.Build(adt);
        int mtex = FindChunkPayload(bytes, "MTEX", out int size);
        Assert.True(mtex >= 0, "Written tile must carry an MTEX chunk.");
        string written = System.Text.Encoding.UTF8.GetString(bytes, mtex, size);
        Assert.StartsWith(texture, written);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void GeneratedAlphaWdt_IsNotReportedWmoBased(int modelCount)
    {
        // The alpha MPHD stores the MDX name count where LK stores flags, so reading LK flags out of
        // it made every alpha map with an ODD model count report as WMO-based - which would tell the
        // terrain adapter there is no terrain to draw. Both parities must come back as terrain maps.
        var assets = Enumerable.Range(0, modelCount)
            .Select(i => Model($"world/kit/tree_{i:00}.mdx", 10f))
            .ToList();

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", StartTileX: 20, StartTileY: 20));
        RosettaMapPlan map = SingleMap(result);

        var alphaTiles = new Dictionary<(int tileX, int tileY), AlphaTileData>();
        foreach (RosettaTilePlan tile in map.Tiles)
        {
            alphaTiles[(tile.TileX, tile.TileY)] = LkToAlphaConverter.ConvertTile(
                RosettaTilesetGenerator.BuildTileAdt(map.MapName, tile, map.GroundTexture), tile.TileX, tile.TileY);
        }

        byte[] wdtBytes = AlphaWdtWriter.Build(map.MapName, alphaTiles);

        using var stream = new MemoryStream(wdtBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, $"{map.MapName}.wdt");
        WdtSummary summary = WdtSummaryReader.Read(stream, fileSummary);

        Assert.False(summary.IsWmoBased, $"{modelCount} models must still be a terrain map, not WMO-based.");
        Assert.Equal(map.Tiles.Count, summary.TilesWithData);
        Assert.Equal(modelCount, summary.DoodadNameCount);
    }

    [Fact]
    public void DesignkitOf_NormalizesSeparatorsAndCase()
    {
        // The client mixes CREATURE\ and Creature/ for the same folder; both must be one kit.
        Assert.Equal(
            RosettaTilesetGenerator.DesignkitOf(@"CREATURE\Murloc\Murloc.m2"),
            RosettaTilesetGenerator.DesignkitOf("creature/murloc/MurlocBaby.m2"));
        Assert.Equal(@"creature\murloc", RosettaTilesetGenerator.DesignkitOf(@"CREATURE\Murloc\Murloc.m2"));
        Assert.Equal(string.Empty, RosettaTilesetGenerator.DesignkitOf("loose.m2"));

        // A depth coarsens the grouping so single-asset folders stop taking a whole tile each.
        Assert.Equal("creature", RosettaTilesetGenerator.DesignkitOf(@"CREATURE\Murloc\Murloc.m2", depth: 1));
        Assert.Equal(@"world\azeroth", RosettaTilesetGenerator.DesignkitOf(@"World\Azeroth\Elwynn\Trees\Oak.m2", depth: 2));
    }

    [Fact]
    public void Generate_KitDepthCoarsensGroupingAndSavesTiles()
    {
        var assets = Enumerable.Range(0, 12)
            .Select(i => Model($"creature/beast_{i:00}/beast_{i:00}.m2", 10f))
            .ToList();

        RosettaGenerationResult fine = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development"));
        RosettaGenerationResult coarse = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", KitDepth: 1));

        // One folder per creature: 12 kits, and a kit never shares a tile, so 12 tiles.
        Assert.Equal(12, fine.Designkits.Count);
        Assert.Equal(12, SingleMap(fine).Tiles.Count);

        // Depth 1 collapses them into one "creature" kit that packs 4 to a tile.
        RosettaDesignkitPlan single = Assert.Single(coarse.Designkits);
        Assert.Equal("creature", single.Kit);
        Assert.Equal(3, SingleMap(coarse).Tiles.Count);
        Assert.Equal(12, coarse.Placements.Count);
    }

    [Fact]
    public void GeneratedTiles_SurviveTheAlphaWdtRoundTrip()
    {
        // The alpha container is the 0.5.3 monolith: every tile lives inside one WDT. The object
        // library has to work there too, so placements and their positions must come back intact.
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/kit/tree_a.m2", 10f),
            Model("world/kit/tree_b.m2", 12f),
            new("world/kit/hut.wmo", RosettaAssetKind.WorldModel,
                new Vector3(-40f, -40f, -5f), new Vector3(40f, 40f, 30f)),
        };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", StartTileX: 12, StartTileY: 34));
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        var alphaTiles = new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(tile.TileX, tile.TileY)] = LkToAlphaConverter.ConvertTile(
                RosettaTilesetGenerator.BuildTileAdt(map.MapName, tile), tile.TileX, tile.TileY),
        };

        byte[] wdtBytes = AlphaWdtWriter.Build(map.MapName, alphaTiles);

        Assert.True(AlphaWdtReader.IsAlphaWdt(wdtBytes), "Written container must identify as an alpha WDT.");
        Assert.Equal([(tile.TileX, tile.TileY)], AlphaWdtReader.ReadExistingTiles(wdtBytes).ToList());

        Assert.True(AlphaWdtReader.TryReadTile(wdtBytes, tile.TileX, tile.TileY, out AlphaTileData? readBack));
        Assert.NotNull(readBack);
        Assert.Equal(2, readBack!.ModelPlacements.Count);
        Assert.Equal(1, readBack.WorldModelPlacements.Count);

        // Positions must still decode onto this tile's own terrain after the LK -> alpha hop.
        foreach (Vector3 position in readBack.ModelPlacements.Select(static m => m.Position)
                     .Concat(readBack.WorldModelPlacements.Select(static w => w.Position)))
        {
            AssertInsideTile(position.X, position.Y, tile.TileX, tile.TileY, "alpha placement");
        }
    }

    [Fact]
    public void SanitizeLabel_MapsUnsupportedCharacters()
    {
        string label = RosettaTilesetGenerator.SanitizeLabel(@"world\kalimdor/azshara (orgrimmar).mdx");
        Assert.Matches("^[A-Z0-9_.\\-]+$", label);
    }

    [Fact]
    public void Generate_PedestalHeights_RaiseObjectPlatformWithBevel()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/test_pedestal.mdx", 20f),
        };
        var options = new RosettaGeneratorOptions(
            "PedestalTest",
            PedestalHeightMeters: 4f,
            PedestalBevelMeters: 12.5f);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        Assert.NotEmpty(tile.Pedestals);
        Assert.Equal(4f, tile.Pedestals[0].Height);

        RosettaPlacementRecord placement = Assert.Single(tile.Placements);
        Assert.Equal(4f, placement.RendererPosition.Z);
        Assert.Equal(4f, placement.RawPosition.Z);

        LkAdtData adt = RosettaTilesetGenerator.BuildTileAdt(
            map.MapName, tile, map.GroundTexture, map.InkTexture, options.PedestalBevelMeters);

        // Verify that some chunk heights are raised to the pedestal height
        float maxHeight = 0f;
        foreach (LkMcnkData chunk in adt.Chunks)
        {
            foreach (float h in chunk.Heights)
            {
                if (h > maxHeight)
                    maxHeight = h;
            }
        }

        Assert.Equal(4f, maxHeight, precision: 2);
    }

    [Fact]
    public void PaintTile_AlphaMap_GeneratesValidMcalAndLayers()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/alpha_mcal_test.mdx", 15f),
        };
        var options = new RosettaGeneratorOptions("McalTest");

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        Assert.NotNull(tile.AlphaCanvas);
        Assert.Equal(RosettaAlphaPainter.TexelsPerTile * RosettaAlphaPainter.TexelsPerTile, tile.AlphaCanvas.Length);

        LkAdtData adt = RosettaTilesetGenerator.BuildTileAdt(
            map.MapName, tile, map.GroundTexture, map.InkTexture, options.PedestalBevelMeters);

        Assert.Equal(2, adt.TextureNames.Count);
        Assert.Equal(options.GroundTexture, adt.TextureNames[0]);
        Assert.Equal(options.InkTexture, adt.TextureNames[1]);

        // Find a chunk that contains painted text alpha
        LkMcnkData? alphaChunk = adt.Chunks.FirstOrDefault(static c => c.AlphaMapData is { Length: > 0 });
        Assert.NotNull(alphaChunk);
        Assert.Equal(2, alphaChunk.NLayers);
        Assert.Equal(2, alphaChunk.Layers.Count);
        Assert.Equal(2048, alphaChunk.AlphaMapData!.Length);
    }

    [Fact]
    public void BuildTileAdt_AlphaWdt_RoundTripWithMcalAndPlacements()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/alpha_wdt_mcal_model.mdx", 20f),
        };
        var options = new RosettaGeneratorOptions("AlphaWdtMcalTest");

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        LkAdtData lkAdt = RosettaTilesetGenerator.BuildTileAdt(
            map.MapName, tile, map.GroundTexture, map.InkTexture, options.PedestalBevelMeters);
        AlphaTileData alphaTile = LkToAlphaConverter.ConvertTile(lkAdt, tile.TileX, tile.TileY);

        Assert.Equal(2, alphaTile.TextureNames.Count);
        Assert.NotNull(alphaTile.McalAlphaPack);
        Assert.Equal(1024, alphaTile.McalAlphaPack.GetLength(0));
        Assert.Equal(1024, alphaTile.McalAlphaPack.GetLength(1));

        var tilesDict = new Dictionary<(int, int), AlphaTileData>
        {
            [(tile.TileX, tile.TileY)] = alphaTile,
        };

        byte[] wdtBytes = AlphaWdtWriter.Build(map.MapName, tilesDict);
        Assert.True(wdtBytes.Length > 0);
        Assert.True(AlphaWdtReader.IsAlphaWdt(wdtBytes));

        bool readSuccess = AlphaWdtReader.TryReadTile(wdtBytes, tile.TileX, tile.TileY, out AlphaTileData? readTile);
        Assert.True(readSuccess);
        Assert.NotNull(readTile);
        Assert.Equal(2, readTile.TextureNames.Count);
        Assert.Single(readTile.ModelPlacements);
    }

    [Fact]
    public void Generate_AllowsFull4096Tiles()
    {
        var options = new RosettaGeneratorOptions("FullMapTest", MaxTilesPerMap: 4096);
        Assert.Equal(4096, options.MaxTilesPerMap);
    }

    private static int AssertEntriesOnTile(byte[] bytes, string tag, int stride, int tileX, int tileY)
    {
        int offset = FindChunkPayload(bytes, tag, out int size);
        if (offset < 0 || size < stride)
            return 0;

        int count = size / stride;
        for (int i = 0; i < count; i++)
        {
            int pos = offset + i * stride;
            float rawX = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(pos + 8));
            float rawY = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(pos + 16));

            // On-disk canvas coordinates must be the tile's own canvas range before any decode.
            Assert.InRange(rawX, tileY * TileSize, (tileY + 1) * TileSize);
            Assert.InRange(rawY, tileX * TileSize, (tileX + 1) * TileSize);

            AssertInsideTile(MapOrigin - rawY, MapOrigin - rawX, tileX, tileY, $"{tag}[{i}]");
        }

        return count;
    }

    private static void AssertInsideTile(float rendererX, float rendererY, int tileX, int tileY, string what)
    {
        float minX = MapOrigin - (tileX + 1) * TileSize;
        float maxX = MapOrigin - tileX * TileSize;
        float minY = MapOrigin - (tileY + 1) * TileSize;
        float maxY = MapOrigin - tileY * TileSize;

        Assert.True(rendererX >= minX && rendererX <= maxX,
            $"{what} rendererX {rendererX:F1} outside tile {tileX} range [{minX:F1},{maxX:F1}].");
        Assert.True(rendererY >= minY && rendererY <= maxY,
            $"{what} rendererY {rendererY:F1} outside tile {tileY} range [{minY:F1},{maxY:F1}].");
    }

    /// <summary>Locates a top-level chunk payload. ADT magics are stored reversed on disk.</summary>
    private static int FindChunkPayload(byte[] data, string tag, out int size)
    {
        size = 0;
        Span<byte> needle = [(byte)tag[3], (byte)tag[2], (byte)tag[1], (byte)tag[0]];
        for (int i = 0; i + 8 <= data.Length;)
        {
            int chunkSize = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(i + 4));
            if (chunkSize < 0 || i + 8 + chunkSize > data.Length)
                return -1;

            if (data.AsSpan(i, 4).SequenceEqual(needle))
            {
                size = chunkSize;
                return i + 8;
            }

            i += 8 + chunkSize;
        }

        return -1;
    }
}
