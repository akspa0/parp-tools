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
            assets, new RosettaGeneratorOptions("Development", GroupByDesignkit: false, SplitAssetKinds: false, ObjectZOffsetMeters: 0f));
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
            assets, new RosettaGeneratorOptions("Development", StartTileX: 31, StartTileY: 17, GroupByDesignkit: false, SplitAssetKinds: false));

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
        float texel = RosettaAlphaPainter.TexelSize(ChunkSize);
        foreach (RosettaPlacementRecord p in result.Placements)
        {
            // A font pixel measured in MCAL texels (64 texels per chunk, 0.52m texel pitch)
            float pixels = p.LabelPixelMeters / texel;
            Assert.True(pixels >= 1f && MathF.Abs(pixels - MathF.Round(pixels)) < 1e-3f,
                $"Label pixel {p.LabelPixelMeters:F3}m is {pixels:F3} MCAL texels; must be a whole multiple >= 1.");

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
            assets, new RosettaGeneratorOptions("Development", StartTileX: 12, StartTileY: 34, SplitAssetKinds: false, ObjectZOffsetMeters: 0f));
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
        Assert.Matches("^[a-zA-Z0-9_.\\-]+$", label);
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
            PedestalBevelMeters: 12.5f,
            ObjectZOffsetMeters: 0f);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        Assert.NotEmpty(tile.Pedestals);
        Assert.Equal(4f, tile.Pedestals[0].Height);

        RosettaPlacementRecord placement = Assert.Single(tile.Placements);
        Assert.Equal(14f, placement.RendererPosition.Z);
        Assert.Equal(14f, placement.RawPosition.Z);

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
    public void Generate_PedestalHeights_NegativeSunkenDipWithBevel()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/test_sunken.mdx", 20f),
        };
        var options = new RosettaGeneratorOptions(
            "SunkenTest",
            PedestalHeightMeters: -10f,
            PedestalBevelMeters: 12.5f,
            ObjectZOffsetMeters: 0f);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        Assert.NotEmpty(tile.Pedestals);
        Assert.Equal(-10f, tile.Pedestals[0].Height);

        RosettaPlacementRecord placement = Assert.Single(tile.Placements);
        Assert.Equal(0f, placement.RendererPosition.Z);
        Assert.Equal(0f, placement.RawPosition.Z);

        LkAdtData adt = RosettaTilesetGenerator.BuildTileAdt(
            map.MapName, tile, map.GroundTexture, map.InkTexture, options.PedestalBevelMeters);

        // Verify that some chunk heights are sunken to the negative pedestal height
        float minHeight = 0f;
        foreach (LkMcnkData chunk in adt.Chunks)
        {
            foreach (float h in chunk.Heights)
            {
                if (h < minHeight)
                    minHeight = h;
            }
        }

        Assert.Equal(-10f, minHeight, precision: 2);
    }

    [Fact]
    public void PaintTile_AlphaMap_GeneratesValidMcalAndLayers()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/alpha_mcal_test.mdx", 15f),
        };
        var options = new RosettaGeneratorOptions("McalTest", PedestalHeightMeters: -10f);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        byte[] alphaCanvas = RosettaTilesetGenerator.BuildTileAlphaCanvas(tile);
        Assert.NotNull(alphaCanvas);
        Assert.Equal(RosettaAlphaPainter.TexelsPerTile * RosettaAlphaPainter.TexelsPerTile, alphaCanvas.Length);

        LkAdtData adt = RosettaTilesetGenerator.BuildTileAdt(
            map.MapName, tile, map.GroundTexture, map.InkTexture, options.PedestalBevelMeters, options.CheckersTexture);

        Assert.Equal(3, adt.TextureNames.Count);
        Assert.Equal(options.GroundTexture, adt.TextureNames[0]);
        Assert.Equal(options.CheckersTexture, adt.TextureNames[1]);
        Assert.Equal(options.InkTexture, adt.TextureNames[2]);

        // Find a chunk that contains painted alpha
        LkMcnkData? alphaChunk = adt.Chunks.FirstOrDefault(static c => c.AlphaMapData is { Length: > 0 });
        Assert.NotNull(alphaChunk);
        Assert.True(alphaChunk.NLayers >= 2);
        Assert.True(alphaChunk.Layers.Count >= 2);
        Assert.True(alphaChunk.AlphaMapData!.Length >= 2048);
    }

    [Fact]
    public void BuildTileAdt_AlphaWdt_RoundTripWithMcalAndPlacements()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/alpha_wdt_mcal_model.mdx", 20f),
        };
        var options = new RosettaGeneratorOptions("AlphaWdtMcalTest", PedestalHeightMeters: -10f);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        LkAdtData lkAdt = RosettaTilesetGenerator.BuildTileAdt(
            map.MapName, tile, map.GroundTexture, map.InkTexture, options.PedestalBevelMeters, options.CheckersTexture);
        AlphaTileData alphaTile = LkToAlphaConverter.ConvertTile(lkAdt, tile.TileX, tile.TileY);

        Assert.Equal(3, alphaTile.TextureNames.Count);
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
        Assert.Equal(3, readTile.TextureNames.Count);
        Assert.Single(readTile.ModelPlacements);
    }

    [Fact]
    public void PaintTile_CheckersAlpha_CoversPedestalIndentationPad()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("doodads/chest.mdx", 5f),
        };
        var options = new RosettaGeneratorOptions("CheckersPadTest", PedestalHeightMeters: -10f, PedestalBevelMeters: 10f);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        byte[] checkersCanvas = RosettaTilesetGenerator.BuildTileCheckersCanvas(tile, options.PedestalBevelMeters);
        Assert.NotNull(checkersCanvas);
        int litTexels = checkersCanvas.Count(static b => b == 255);
        Assert.True(litTexels > 0, "Expected non-zero filled texels on the checkers alpha pad.");
    }

    [Fact]
    public void PaintTile_AlphaMap_PaintsCellBordersAndGridLines()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("doodads/bench.mdx", 5f),
        };
        var options = new RosettaGeneratorOptions("GridLinesTest", PaintCellBorders: true);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        // 1. With PaintCellBorders = true
        byte[] canvasWithBorders = RosettaTilesetGenerator.BuildTileAlphaCanvas(tile, paintCellBorders: true);
        int litWithBorders = canvasWithBorders.Count(static b => b > 0);

        // 2. With PaintCellBorders = false
        byte[] canvasWithoutBorders = RosettaTilesetGenerator.BuildTileAlphaCanvas(tile, paintCellBorders: false);
        int litWithoutBorders = canvasWithoutBorders.Count(static b => b > 0);

        Assert.True(litWithBorders > litWithoutBorders, "Expected cell borders to paint additional grid line texels onto the terrain canvas.");

        // 3. Verify corner and edge texels of the cell are painted
        RosettaPlacementRecord placement = Assert.Single(tile.Placements);
        float texelSize = RosettaAlphaPainter.TexelSize(RosettaGeneratorOptions.ChunkSize);
        int borderX = Math.Clamp((int)MathF.Floor(placement.CellU / texelSize), 0, RosettaAlphaPainter.TexelsPerTile - 1);
        int borderY = Math.Clamp((int)MathF.Floor(placement.CellV / texelSize), 0, RosettaAlphaPainter.TexelsPerTile - 1);
        Assert.True(canvasWithBorders[(borderY * RosettaAlphaPainter.TexelsPerTile) + borderX] > 0, "Expected cell perimeter border line to be lit.");
    }

    [Fact]
    public void Generate_AllowsFull4096Tiles()
    {
        var options = new RosettaGeneratorOptions("FullMapTest", MaxTilesPerMap: 4096);
        Assert.Equal(4096, options.MaxTilesPerMap);
    }

    [Fact]
    public void RosettaMinimapPainter_RendersValid256x256ImageAndBlp()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/minimap_test_model.mdx", 20f),
            new("world/minimap_test_wmo.wmo", RosettaAssetKind.WorldModel,
                new Vector3(-25f, -25f, 0f), new Vector3(25f, 25f, 15f)),
        };
        var options = new RosettaGeneratorOptions("MinimapTest", PedestalHeightMeters: 4f, SplitAssetKinds: false);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> img =
            RosettaMinimapPainter.RenderTileImage(tile, tile.Pedestals, tile.AlphaCanvas);

        Assert.Equal(256, img.Width);
        Assert.Equal(256, img.Height);

        byte[] blpBytes = RosettaMinimapPainter.RenderTileBlp(tile, tile.Pedestals, tile.AlphaCanvas);
        Assert.NotNull(blpBytes);
        Assert.True(blpBytes.Length > 148);

        // Verify SereniaBLPLib decodes the written BLP2 file cleanly
        using var ms = new MemoryStream(blpBytes);
        using var blp = new SereniaBLPLib.BlpFile(ms);
        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> decoded = blp.GetImage(0);
        Assert.Equal(256, decoded.Width);
        Assert.Equal(256, decoded.Height);
    }

    [Fact]
    public void RosettaMinimapPainter_RendersPinAtObjectBandCenter()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/minimap_pin_test.mdx", 20f),
        };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("MinimapPin", PedestalHeightMeters: 4f));
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);
        RosettaPlacementRecord placement = Assert.Single(tile.Placements);

        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> img =
            RosettaMinimapPainter.RenderTileImage(tile, tile.Pedestals, tile.AlphaCanvas);

        (float centerU, float centerV) = RosettaTilesetGenerator.GetObjectBandCenter(placement);
        int markerX = ToMinimapPixel(centerU);
        int markerY = ToMinimapPixel(centerV);
        var groundColor = new SixLabors.ImageSharp.PixelFormats.Rgba32(208, 192, 160, 255);

        // Center has rendered object footprint (distinct from background ground)
        Assert.NotEqual(groundColor, img[markerX, markerY]);

        int staleWholeCellCenterY = ToMinimapPixel(placement.CellV + (placement.CellSize / 2f));
        Assert.NotEqual(img[markerX, markerY], img[markerX, staleWholeCellCenterY]);
    }

    [Fact]
    public void Blp2Writer_EncodeDxt1_ProducesParsableBlp2File()
    {
        using var testImage = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(256, 256);
        testImage.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < 256; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < 256; x++)
                    row[x] = new SixLabors.ImageSharp.PixelFormats.Rgba32((byte)x, (byte)y, 128, 255);
            }
        });

        byte[] blpBytes = WowViewer.Core.IO.Blp.Blp2Writer.EncodeDxt1(testImage);
        Assert.NotNull(blpBytes);
        Assert.Equal(148 + 32768, blpBytes.Length);

        using var ms = new MemoryStream(blpBytes);
        using var blp = new SereniaBLPLib.BlpFile(ms);
        using SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32> decoded = blp.GetImage(0);
        Assert.Equal(256, decoded.Width);
        Assert.Equal(256, decoded.Height);
    }

    [Fact]
    public void Generate_MuseumExhibitOrdering_SortsModelsBeforeWorldModelsAndSmallToLarge()
    {
        var assets = new List<RosettaAssetEntry>
        {
            new("doodads/large_building.wmo", RosettaAssetKind.WorldModel, new Vector3(-60f, -60f, 0f), new Vector3(60f, 60f, 40f)),
            Model("doodads/small_potion.mdx", 2f),
            Model("doodads/huge_dragon.mdx", 40f),
            Model("doodads/medium_human.mdx", 6f),
            new("doodads/small_hut.wmo", RosettaAssetKind.WorldModel, new Vector3(-20f, -20f, 0f), new Vector3(20f, 20f, 15f)),
        };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("MuseumTest", GroupByDesignkit: false, CellChunks: 4, LabelBandChunks: 1));

        // Museum ordering must place Models first (potion -> human -> dragon), then WorldModels (hut -> building)
        Assert.Equal(5, result.Placements.Count);
        Assert.Equal("doodads/small_potion.mdx", result.Placements[0].Asset.AssetPath);
        Assert.Equal("doodads/medium_human.mdx", result.Placements[1].Asset.AssetPath);
        Assert.Equal("doodads/huge_dragon.mdx", result.Placements[2].Asset.AssetPath);
        Assert.Equal("doodads/small_hut.wmo", result.Placements[3].Asset.AssetPath);
        Assert.Equal("doodads/large_building.wmo", result.Placements[4].Asset.AssetPath);
    }

    [Fact]
    public void Generate_CompactCellLayout_PacksSixteenCellsPerTile()
    {
        var assets = Enumerable.Range(0, 16)
            .Select(i => Model($"items/gem_{i:D2}.mdx", 4f))
            .ToList();

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("CompactTest", GroupByDesignkit: false, CellChunks: 4, LabelBandChunks: 1));

        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);
        Assert.Equal(16, tile.Placements.Count);
        foreach (RosettaPlacementRecord placement in tile.Placements)
        {
            Assert.Equal(TileSize / 4f, placement.CellSize, 2);
            Assert.Equal((TileSize / 4f) * (3f / 4f), placement.ObjectBandSize, 2);
        }
    }

    [Fact]
    public void RosettaTextPainter_HandwritingFont_RendersBothUpperAndLowercaseGlyphs()
    {
        string sampleText = "Creature/Murloc_01.mdx";
        List<int> masks = RosettaTextPainter.BuildColumnMasks(sampleText);
        Assert.NotEmpty(masks);
        Assert.Equal(sampleText.Length * RosettaTextPainter.CharAdvanceColumns, masks.Count);

        // Every character must contribute lit ink columns
        for (int i = 0; i < sampleText.Length; i++)
        {
            int charStart = i * RosettaTextPainter.CharAdvanceColumns;
            bool hasInk = false;
            for (int col = 0; col < RosettaTextPainter.GlyphColumns; col++)
            {
                if (masks[charStart + col] != 0)
                    hasInk = true;
            }
            Assert.True(hasInk, $"Character '{sampleText[i]}' at index {i} produced no ink mask.");
        }
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

    private static int ToMinimapPixel(float meters)
        => Math.Clamp(
            (int)(meters / TileSize * RosettaMinimapPainter.MinimapResolution),
            0,
            RosettaMinimapPainter.MinimapResolution - 1);

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

    [Fact]
    public void AlphaPainter_DrawBullseyePattern_FillsExpectedRegions()
    {
        byte[] canvas = RosettaAlphaPainter.CreateCanvas();
        RosettaAlphaPainter.DrawBullseyePattern(
            canvas,
            RosettaGeneratorOptions.TileSize / 2f,
            RosettaGeneratorOptions.TileSize / 2f,
            RosettaGeneratorOptions.ChunkSize,
            ink: 255);

        int nonZero = 0;
        for (int i = 0; i < canvas.Length; i++)
        {
            if (canvas[i] > 0) nonZero++;
        }

        Assert.True(nonZero > 1000, $"Expected bullseye canvas to have substantial painted pixels, got {nonZero}.");

        byte[][] chunks = RosettaAlphaPainter.SliceToChunks(canvas);
        Assert.Equal(256, chunks.Length);
        int activeChunks = chunks.Count(static c => c.Length > 0);
        Assert.True(activeChunks > 0, "Bullseye should slice into active MCAL chunks.");
    }

    [Fact]
    public void AlphaWdt_AsymmetricTileCoordinates_MaintainRowMajorIntegrity()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("world/kit/portal.m2", 10f),
        };

        // Asymmetric coordinates: TileX = 41, TileY = 39 (not equal)
        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Development", StartTileX: 41, StartTileY: 39));
        RosettaMapPlan map = SingleMap(result);
        RosettaTilePlan tile = Assert.Single(map.Tiles);

        Assert.Equal(41, tile.TileX);
        Assert.Equal(39, tile.TileY);

        var alphaTiles = new Dictionary<(int tileX, int tileY), AlphaTileData>
        {
            [(tile.TileX, tile.TileY)] = LkToAlphaConverter.ConvertTile(
                RosettaTilesetGenerator.BuildTileAdt(map.MapName, tile), tile.TileX, tile.TileY),
        };

        byte[] wdtBytes = AlphaWdtWriter.Build(map.MapName, alphaTiles);

        // Tile (41, 39) must exist
        Assert.True(AlphaWdtReader.TryReadTile(wdtBytes, 41, 39, out AlphaTileData? readBack));
        Assert.NotNull(readBack);
        Assert.Single(readBack!.ModelPlacements);

        // Transposed tile (39, 41) must NOT exist
        Assert.False(AlphaWdtReader.TryReadTile(wdtBytes, 39, 41, out _));
    }

    [Fact]
    public void RosettaDbcGenerator_BuildAlphaMapDbc_ProducesValidWDBC()
    {
        var entries = new List<WowViewer.Core.IO.Dbc.RosettaMapDbcEntry>
        {
            new(500, "Rosetta053_MDX", InstanceType: 0, Pvp: 0, MapName: "Rosetta Exhibit (MDX)"),
            new(501, "Rosetta053_WMO", InstanceType: 0, Pvp: 0, MapName: "Rosetta Exhibit (WMO)"),
        };

        byte[] dbcBytes = WowViewer.Core.IO.Dbc.RosettaDbcGenerator.BuildAlphaMapDbc(entries);
        Assert.NotNull(dbcBytes);
        Assert.True(dbcBytes.Length > 20);

        using var ms = new MemoryStream(dbcBytes);
        using var reader = new BinaryReader(ms);

        uint magic = reader.ReadUInt32();
        uint records = reader.ReadUInt32();
        uint fields = reader.ReadUInt32();
        uint recordSize = reader.ReadUInt32();
        uint stringBlockSize = reader.ReadUInt32();

        Assert.Equal(0x43424457u, magic); // "WDBC"
        Assert.Equal(2u, records);
        Assert.Equal(5u, fields);
        Assert.Equal(20u, recordSize);
        Assert.True(stringBlockSize > 0);

        uint firstId = reader.ReadUInt32();
        Assert.Equal(500u, firstId);
    }

    [Fact]
    public void RosettaDbcGenerator_BuildAlphaAreaTableDbc_ProducesValidWDBC()
    {
        var entries = new List<WowViewer.Core.IO.Dbc.RosettaAreaTableDbcEntry>
        {
            new(5000, 500, ParentAreaId: 0, AreaBit: 0, Flags: 0, AreaName: "Rosetta: Creature"),
            new(5001, 500, ParentAreaId: 0, AreaBit: 0, Flags: 0, AreaName: "Rosetta: Doodads"),
        };

        byte[] dbcBytes = WowViewer.Core.IO.Dbc.RosettaDbcGenerator.BuildAlphaAreaTableDbc(entries);
        Assert.NotNull(dbcBytes);

        using var ms = new MemoryStream(dbcBytes);
        using var reader = new BinaryReader(ms);

        uint magic = reader.ReadUInt32();
        uint records = reader.ReadUInt32();
        uint fields = reader.ReadUInt32();
        uint recordSize = reader.ReadUInt32();

        Assert.Equal(0x43424457u, magic);
        Assert.Equal(2u, records);
        Assert.Equal(14u, fields);
        Assert.Equal(56u, recordSize);
    }

    [Fact]
    public void RosettaMinimapPainter_GenerateMinimapTrs_ProducesValidBlocks()
    {
        var assets = new List<RosettaAssetEntry> { Model("world/test.mdx", 10f) };
        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("Rosetta053_MDX", StartTileX: 20, StartTileY: 25));

        string trs = RosettaMinimapPainter.GenerateMinimapTrs(result.Maps, extraAliases: ["Azeroth"]);
        Assert.NotNull(trs);
        Assert.Contains("dir: Rosetta053_MDX", trs);
        Assert.Contains("dir: Azeroth", trs);
        Assert.Contains(@"Rosetta053_MDX\map20_25.blp", trs);
        Assert.Contains(@"Azeroth\map20_25.blp", trs);
    }

    [Fact]
    public void Generate_SplitAssetKinds_ProducesSeparateMdxAndWmoMaps()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("creature/murloc.mdx", 10f),
            new("building/inn.wmo", RosettaAssetKind.WorldModel, new Vector3(-20f, -20f, 0f), new Vector3(20f, 20f, 15f)),
        };

        var options = new RosettaGeneratorOptions("Rosetta053", SplitAssetKinds: true);
        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(assets, options);

        Assert.Equal(2, result.Maps.Count);
        Assert.Equal("Rosetta053_MDX", result.Maps[0].MapName);
        Assert.Equal("Rosetta053_WMO", result.Maps[1].MapName);

        Assert.All(result.Maps[0].Placements, static p => Assert.Equal(RosettaAssetKind.Model, p.Asset.Kind));
        Assert.All(result.Maps[1].Placements, static p => Assert.Equal(RosettaAssetKind.WorldModel, p.Asset.Kind));
    }

    [Fact]
    public void Generate_ObjectZOffset_ElevatesPlacementAboveTerrain()
    {
        var asset = new RosettaAssetEntry(
            "creature/giant.mdx",
            RosettaAssetKind.Model,
            new Vector3(-10f, -10f, -15f), // Lowest point is -15m underground
            new Vector3(10f, 10f, 25f));

        var options = new RosettaGeneratorOptions(
            "ZOffsetTest",
            PedestalHeightMeters: 0f,
            ObjectZOffsetMeters: 20f);

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate([asset], options);
        RosettaPlacementRecord placement = Assert.Single(result.Placements);

        // RawPosition.Z must be groundZ (0) + (-minZ: 15) + ObjectZOffset (20) = 35m
        Assert.Equal(35f, placement.RawPosition.Z);
        Assert.Equal(35f, placement.RendererPosition.Z);
    }

    [Fact]
    public void RosettaDbcGenerator_InferClientBuild_IdentifiesCorrectBuildVersions()
    {
        string lkPath = @"H:\CLIENTS\Wrath\3.X_Retail_OSX_enUS_3.3.5.12340\World of Warcraft\";
        Assert.Equal("3.3.5.12340", WowViewer.Core.IO.Dbc.RosettaDbcGenerator.InferClientBuild(lkPath));

        string alphaPath = @"C:\Games\WoW_0.5.3_Client";
        Assert.Equal("0.5.3.3368", WowViewer.Core.IO.Dbc.RosettaDbcGenerator.InferClientBuild(alphaPath));

        string vanillaPath = @"D:\Games\1.12.1.5875";
        Assert.Equal("1.12.1.5875", WowViewer.Core.IO.Dbc.RosettaDbcGenerator.InferClientBuild(vanillaPath));

        Assert.Equal("0.5.3.3368", WowViewer.Core.IO.Dbc.RosettaDbcGenerator.InferClientBuild(null, "alpha"));
        Assert.Equal("3.3.5.12340", WowViewer.Core.IO.Dbc.RosettaDbcGenerator.InferClientBuild(null, "lk"));
    }

    [Fact]
    public void RosettaDbcGenerator_TryFindDefinitionsDirectory_LocatesMapDbd()
    {
        string? dbdDir = WowViewer.Core.IO.Dbc.RosettaDbcGenerator.TryFindDefinitionsDirectory();
        Assert.NotNull(dbdDir);
        Assert.True(Directory.Exists(dbdDir));
        Assert.True(File.Exists(Path.Combine(dbdDir, "Map.dbd")));
        Assert.True(File.Exists(Path.Combine(dbdDir, "AreaTable.dbd")));
    }

    [Fact]
    public void RosettaDbcGenerator_PatchAndSaveClientDbcs_PatchesLkDbcPreservingOriginals()
    {
        string tempOutDir = Path.Combine(Path.GetTempPath(), $"rosetta_dbc_lk_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempOutDir);

        try
        {
            // Build authentic 3.3.5 format Map and AreaTable DBCs
            byte[] originalMapDbc = BuildLk335MapDbc();
            byte[] originalAreaDbc = BuildLk335AreaTableDbc();

            var fakeProvider = new TestMemoryDbcProvider(
                ("Map", originalMapDbc),
                ("AreaTable", originalAreaDbc));

            var newMaps = new List<WowViewer.Core.IO.Dbc.RosettaMapDbcEntry>
            {
                new(500, "Rosetta_MDX", InstanceType: 0, Pvp: 0, MapName: "Rosetta Exhibit (MDX)", AreaTableId: 5000),
                new(501, "Rosetta_WMO", InstanceType: 0, Pvp: 0, MapName: "Rosetta Exhibit (WMO)", AreaTableId: 5001),
            };
            var newAreas = new List<WowViewer.Core.IO.Dbc.RosettaAreaTableDbcEntry>
            {
                new(5000, 500, ParentAreaId: 0, AreaBit: 0, Flags: 0, AreaName: "Rosetta: Creature"),
                new(5001, 501, ParentAreaId: 0, AreaBit: 0, Flags: 0, AreaName: "Rosetta: Doodad"),
            };

            var (patchedMaps, patchedAreas) = WowViewer.Core.IO.Dbc.RosettaDbcGenerator.PatchAndSaveClientDbcs(
                fakeProvider,
                "3.3.5.12340",
                newMaps,
                newAreas,
                tempOutDir);

            Assert.Equal(3, patchedMaps); // 1 original + 2 added
            Assert.Equal(3, patchedAreas); // 1 original + 2 added

            string patchedMapPath = Path.Combine(tempOutDir, "DBFilesClient", "Map.dbc");
            string patchedAreaPath = Path.Combine(tempOutDir, "DBFilesClient", "AreaTable.dbc");
            Assert.True(File.Exists(patchedMapPath));
            Assert.True(File.Exists(patchedAreaPath));

            // Verify with DBCD that the patched files read back properly
            string? dbdDir = WowViewer.Core.IO.Dbc.RosettaDbcGenerator.TryFindDefinitionsDirectory();
            Assert.NotNull(dbdDir);

            var dbFilesProvider = new DBCD.Providers.FilesystemDBCProvider(Path.Combine(tempOutDir, "DBFilesClient"));
            var dbdProvider = new DBCD.Providers.FilesystemDBDProvider(dbdDir);
            var dbcd = new DBCD.DBCD(dbFilesProvider, dbdProvider);

            var loadedMaps = dbcd.Load("Map", "3.3.5.12340");
            Assert.Equal(3, loadedMaps.Count);
            var mapIds = loadedMaps.Values.Select(static r => Convert.ToInt32(r["ID"])).ToList();
            Assert.Contains(0, mapIds);
            Assert.Contains(500, mapIds);
            Assert.Contains(501, mapIds);

            var loadedAreas = dbcd.Load("AreaTable", "3.3.5.12340");
            Assert.Equal(3, loadedAreas.Count);
            var areaIds = loadedAreas.Values.Select(static r => Convert.ToInt32(r["ID"])).ToList();
            Assert.Contains(1, areaIds);
            Assert.Contains(5000, areaIds);
            Assert.Contains(5001, areaIds);
        }
        finally
        {
            if (Directory.Exists(tempOutDir))
                Directory.Delete(tempOutDir, true);
        }
    }

    private static byte[] BuildLk335MapDbc()
    {
        const uint fieldCount = 66;
        const uint recordSize = fieldCount * 4;
        using MemoryStream stringStream = new();
        stringStream.WriteByte(0);
        uint dirOffset = WriteTestDbcString(stringStream, "Azeroth");
        uint nameOffset = WriteTestDbcString(stringStream, "Eastern Kingdoms");

        uint[] row = new uint[fieldCount];
        row[0] = 0; // ID
        row[1] = dirOffset; // Directory
        row[2] = 0; // InstanceType
        row[3] = 0; // Flags
        row[4] = 0; // PVP
        row[5] = nameOffset; // MapName_lang[0]
        row[21] = 0xFFu; // MapName_lang_mask
        row[22] = 0; // AreaTableID
        row[57] = FloatBits(1.0f); // MinimapIconScale
        row[58] = 0; // CorpseMapID

        return AssembleTestDbc(fieldCount, recordSize, [row], stringStream);
    }

    private static byte[] BuildLk335AreaTableDbc()
    {
        const uint fieldCount = 36;
        const uint recordSize = fieldCount * 4;
        using MemoryStream stringStream = new();
        stringStream.WriteByte(0);
        uint nameOffset = WriteTestDbcString(stringStream, "Echo Isles");

        uint[] row = new uint[fieldCount];
        row[0] = 1; // ID
        row[1] = 0; // ContinentID
        row[2] = 0; // ParentAreaID
        row[11] = nameOffset; // AreaName_lang[0]
        row[27] = 0xFFu; // AreaName_lang_mask

        return AssembleTestDbc(fieldCount, recordSize, [row], stringStream);
    }

    private static uint WriteTestDbcString(MemoryStream stringStream, string value)
    {
        if (string.IsNullOrEmpty(value))
            return 0;
        uint offset = checked((uint)stringStream.Position);
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(value);
        stringStream.Write(bytes, 0, bytes.Length);
        stringStream.WriteByte(0);
        return offset;
    }

    private static byte[] AssembleTestDbc(uint fieldCount, uint recordSize, List<uint[]> rows, MemoryStream stringStream)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(0x43424457u); // "WDBC"
        writer.Write(checked((uint)rows.Count));
        writer.Write(fieldCount);
        writer.Write(recordSize);
        writer.Write(checked((uint)stringStream.Length));
        foreach (uint[] row in rows)
        {
            foreach (uint val in row)
                writer.Write(val);
        }
        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();
        return stream.ToArray();
    }

    private static uint FloatBits(float value) => BitConverter.ToUInt32(BitConverter.GetBytes(value), 0);

    private static RosettaAssetEntry WmoAsset(string path, float size) => new(
        path,
        RosettaAssetKind.WorldModel,
        new Vector3(-size / 2f, -size / 2f, -size / 2f),
        new Vector3(size / 2f, size / 2f, size / 2f));

    [Fact]
    public void RosettaMinimapPainter_RenderAndSaveMapOverview_ProducesStitchedPng()
    {
        var assets = new List<RosettaAssetEntry>
        {
            Model("creature/dragon/dragon.mdx", 25f),
            WmoAsset("wmo/dungeon/keep.wmo", 60f)
        };

        RosettaGenerationResult result = RosettaTilesetGenerator.Generate(
            assets, new RosettaGeneratorOptions("OverviewTest", PedestalHeightMeters: 4f));
        Assert.NotEmpty(result.Maps);

        foreach (RosettaMapPlan map in result.Maps)
        {
            string tempPath = Path.Combine(Path.GetTempPath(), $"rosetta_overview_{map.MapName}_{Guid.NewGuid():N}.png");
            try
            {
                RosettaMinimapPainter.RenderAndSaveMapOverview(map, tempPath, tileResolution: 128);
                Assert.True(File.Exists(tempPath));

                using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(tempPath);
                Assert.Equal(128, img.Width);
                Assert.Equal(128, img.Height);
            }
            finally
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
        }
    }

    private sealed class TestMemoryDbcProvider(params (string TableName, byte[] Data)[] tables) : DBCD.Providers.IDBCProvider
    {
        private readonly Dictionary<string, byte[]> _tables = tables.ToDictionary(static t => t.TableName, static t => t.Data, StringComparer.OrdinalIgnoreCase);

        public Stream StreamForTableName(string tableName, string build)
        {
            if (_tables.TryGetValue(tableName, out byte[]? bytes))
                return new MemoryStream(bytes, writable: false);
            throw new FileNotFoundException($"Table not found: {tableName}");
        }
    }
}


