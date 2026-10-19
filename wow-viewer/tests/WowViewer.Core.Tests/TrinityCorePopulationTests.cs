using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Population;
using WowViewer.Core.Population;
using Xunit;

namespace WowViewer.Core.Tests;

public class TrinityCorePopulationTests
{
    [Fact]
    public void SqlTupleParser_ExtractsQuotedAndEscapedTokens()
    {
        string line = "INSERT INTO `creature` VALUES (53940, 14337, 451, 0, 0, 1, 1, 0, 0, 16397.2, -16136.0, 41.0866, 4.31874);";
        var tuples = TrinityCoreSqlReader.ParseTuples(line);

        Assert.Single(tuples);
        var t = tuples[0];
        Assert.Equal("53940", t[0]);
        Assert.Equal("14337", t[1]);
        Assert.Equal("451", t[2]);
        Assert.Equal("16397.2", t[9]);
        Assert.Equal("-16136.0", t[10]);
        Assert.Equal("41.0866", t[11]);
        Assert.Equal("4.31874", t[12]);
    }

    [Fact]
    public void SqlTupleParser_HandlesEscapedStringsAndCommas()
    {
        string line = "INSERT INTO `creature_template` VALUES (14337, 0, 0, 0, 0, 0, 14379, 0, 0, 0, 'Field Repair Bot 74A', 'Super, Repair \\'Bot\\'', 0, 0, 60, 60, 0, 35, 1, 1.0, 1.0, 1.25);";
        var tuples = TrinityCoreSqlReader.ParseTuples(line);

        Assert.Single(tuples);
        var t = tuples[0];
        Assert.Equal("14337", t[0]);
        Assert.Equal("14379", t[6]);
        Assert.Equal("Field Repair Bot 74A", t[10]);
        Assert.Equal("Super, Repair 'Bot'", t[11]);
        Assert.Equal("1.25", t[21]);
    }

    [Fact]
    public async Task ExtractAllMapsAsync_ParsesCorrectScalesForBothTrinityCoreAndAlphaCore()
    {
        string tcSql = Path.GetTempFileName();
        try
        {
            await File.WriteAllLinesAsync(tcSql, new[]
            {
                // TC creature: level 80, scale 1.25
                "INSERT INTO `creature` VALUES (1001, 14337, 451, 0, 0, 1, 1, 0, 0, 100.0, 200.0, 30.0, 1.5);",
                "INSERT INTO `creature_template` VALUES (14337, 0, 0, 0, 0, 0, 14379, 0, 0, 0, 'Field Repair Bot 74A', NULL, 0, 0, 80, 80, 0, 35, 1, 1.0, 1.0, 1.25);",
                // TC gameobject: size 0.63 (col 7)
                "INSERT INTO `gameobject` VALUES (2001, 31, 451, 0, 0, 1, 1, 150.0, 250.0, 35.0, 3.14, 0, 0, 0, 1);",
                "INSERT INTO `gameobject_template` VALUES (31, 2, 6, 'Old Lion Statue', '', '', '', 0.63, 43, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '', '', NULL, 12340);"
            });

            var reader = new TrinityCoreSqlReader();
            var result = await reader.ExtractAllMapsAsync(tcSql);

            Assert.True(result.SpawnsByMap.ContainsKey(451));
            var spawns = result.SpawnsByMap[451];
            var creature = Assert.Single(spawns, s => s.SpawnType == PopulationSpawnType.Creature);
            Assert.Equal(1.25f, creature.Scale); // Must NOT be level 80!

            var go = Assert.Single(spawns, s => s.SpawnType == PopulationSpawnType.GameObject);
            Assert.Equal(0.63f, go.Scale); // Must NOT be 1.0f or col 4!
        }
        finally
        {
            if (File.Exists(tcSql)) File.Delete(tcSql);
        }

        string alphaSql = Path.GetTempFileName();
        try
        {
            await File.WriteAllLinesAsync(alphaSql, new[]
            {
                // AlphaCore creature: level 60, scale 1.5
                "INSERT INTO `spawns_creatures` VALUES (1002, 2051, 0, 0, 0, 451, 100.0, 200.0, 30.0, 1.5);",
                "INSERT INTO `creature_template` VALUES (2051, 907, 0, 0, 0, 0, 'Twain The Tester', NULL, 0, 0, 60, 60, 35, 1, 1.0, 1.0, 1.5);",
                // AlphaCore gameobject: faction 84 (col 4), size 0.75 (col 6)
                "INSERT INTO `spawns_gameobjects` VALUES (2002, 1719, 451, 150.0, 250.0, 35.0, 3.14);",
                "INSERT INTO `gameobject_template` VALUES (1719, 6, 273, 'Anvil', 84, 0, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '');"
            });

            var reader = new TrinityCoreSqlReader();
            var result = await reader.ExtractAllMapsAsync(alphaSql);

            Assert.True(result.SpawnsByMap.ContainsKey(451));
            var spawns = result.SpawnsByMap[451];
            var creature = Assert.Single(spawns, s => s.SpawnType == PopulationSpawnType.Creature);
            Assert.Equal(1.5f, creature.Scale); // Must NOT be level 60!

            var go = Assert.Single(spawns, s => s.SpawnType == PopulationSpawnType.GameObject);
            Assert.Equal(0.75f, go.Scale); // Must NOT be faction 84!
        }
        finally
        {
            if (File.Exists(alphaSql)) File.Delete(alphaSql);
        }
    }

    [Fact]
    public void AlphaCoreCornerTransposer_CorrectlyIdentifiesDevCornerFixtures()
    {
        // Tile (0, 0) - Upper left corner test fixture (e.g. Twain The Tester, Jesse The Tester)
        Assert.True(AlphaCoreCornerTransposer.IsCornerTestFixture(0, 16771.8f, 16819.2f));

        // Tile (1, 1) - Upper left corner test fixture (e.g. CHOW Guard)
        Assert.True(AlphaCoreCornerTransposer.IsCornerTestFixture(0, 16302.8f, 16286.2f));

        // Tile (1, 62) - Upper right corner test fixture (e.g. Forge, Anvil, TEST BRONZE)
        Assert.True(AlphaCoreCornerTransposer.IsCornerTestFixture(0, 16414.1f, -16126.0f));

        // Center of Azeroth (Stormwind / Dun Morogh / etc.) - NOT corner fixture
        Assert.False(AlphaCoreCornerTransposer.IsCornerTestFixture(0, -8949.95f, -132.493f));

        // Map 1 (Kalimdor) at high coords - only Map 0 is transposed
        Assert.False(AlphaCoreCornerTransposer.IsCornerTestFixture(1, 16771.8f, 16819.2f));
    }

    [Fact]
    public void AlphaCoreCornerTransposer_TransposesToMap451WithLineage()
    {
        var original = new PopulationRecord
        {
            SpawnId = 400104,
            EntryId = 2051,
            MapId = 0,
            SpawnType = PopulationSpawnType.Creature,
            Name = "Twain The Tester",
            ModelPath = @"Creature\Human\HumanMale.m2",
            Position = new Vector3(16771.8f, 16819.2f, 35.787f),
            Orientation = 4.536f,
            Scale = 1.0f,
            DisplayScale = 1.0f,
            EffectiveScale = 1.0f
        };

        var transposed = AlphaCoreCornerTransposer.TransposeToMap451(original);

        Assert.Equal(451, transposed.MapId);
        Assert.True(transposed.IsTransposed);
        Assert.Equal(original.SpawnId, transposed.SpawnId);
        Assert.Equal(original.EntryId, transposed.EntryId);
        Assert.Equal(original.Position, transposed.Position);
        Assert.Equal(original.Orientation, transposed.Orientation);
        Assert.NotNull(transposed.SourceNote);
        Assert.Contains("Alpha-Core Map 0", transposed.SourceNote);
        Assert.Contains("(0, 0)", transposed.SourceNote);
    }

    [Fact]
    public void DbcModelResolver_NormalizesAndResolvesPaths()
    {
        var resolver = new DbcModelResolver();
        resolver.RegisterCreature(14379, "FieldRepairBot", 1.25f);
        resolver.RegisterGameObject(273, @"World/Generic/Active/Anvil/Anvil", 1.0f);

        Assert.True(resolver.TryResolveCreature(14379, out string? creaturePath, out float creatureScale));
        Assert.Equal(@"Creature\FieldRepairBot\FieldRepairBot.m2", creaturePath);
        Assert.Equal(1.25f, creatureScale);

        Assert.True(resolver.TryResolveGameObject(273, out string? goPath, out float goScale));
        Assert.Equal(@"World\Generic\Active\Anvil\Anvil.mdx", goPath);
        Assert.Equal(1.0f, goScale);
    }

    [Fact]
    public void ManifestIndexing_BothDatasetsExistAndCoverAllMaps()
    {
        string baseDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "population"));
        if (!Directory.Exists(baseDir))
            baseDir = @"I:\parp\parp-tools\wow-viewer\data\population";

        if (!Directory.Exists(baseDir))
            return; // Skip if run in an environment without the local data directory

        string manifest335 = Path.Combine(baseDir, "3.3.5a", "manifest.json");
        string manifest053 = Path.Combine(baseDir, "0.5.3", "manifest.json");

        if (File.Exists(manifest335))
        {
            string json = File.ReadAllText(manifest335);
            var manifest = System.Text.Json.JsonSerializer.Deserialize<PopulationMapManifest>(json, new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            Assert.NotNull(manifest);
            Assert.Equal("3.3.5a", manifest.Version);
            Assert.True(manifest.Maps.Count >= 100, $"Expected >= 100 maps in 3.3.5a manifest, found {manifest.Maps.Count}");
            Assert.Contains(manifest.Maps, m => m.MapId == 0);   // Eastern Kingdoms
            Assert.Contains(manifest.Maps, m => m.MapId == 1);   // Kalimdor
            Assert.Contains(manifest.Maps, m => m.MapId == 530); // Outland
            Assert.Contains(manifest.Maps, m => m.MapId == 571); // Northrend
            Assert.Contains(manifest.Maps, m => m.MapId == 451); // Development

            // Verify canonical map names are populated instead of placeholder Map_XX
            var map546 = Assert.Single(manifest.Maps, m => m.MapId == 546);
            Assert.Equal("Coilfang: The Underbog", map546.MapName);

            var map33 = Assert.Single(manifest.Maps, m => m.MapId == 33);
            Assert.Equal("Shadowfang Keep", map33.MapName);
        }

        if (File.Exists(manifest053))
        {
            string json = File.ReadAllText(manifest053);
            var manifest = System.Text.Json.JsonSerializer.Deserialize<PopulationMapManifest>(json, new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            Assert.NotNull(manifest);
            Assert.Equal("0.5.3", manifest.Version);
            Assert.True(manifest.Maps.Count >= 30, $"Expected >= 30 maps in 0.5.3 manifest, found {manifest.Maps.Count}");
            Assert.Contains(manifest.Maps, m => m.MapId == 0);   // Eastern Kingdoms
            Assert.Contains(manifest.Maps, m => m.MapId == 1);   // Kalimdor
            Assert.Contains(manifest.Maps, m => m.MapId == 30);  // Alterac Valley
            Assert.Contains(manifest.Maps, m => m.MapId == 451); // Development

            var map33Alpha = Assert.Single(manifest.Maps, m => m.MapId == 33);
            Assert.Equal("Shadowfang Keep", map33Alpha.MapName);
        }
    }

    [Theory]
    [InlineData(0, "Eastern Kingdoms")]
    [InlineData(1, "Kalimdor")]
    [InlineData(30, "Alterac Valley")]
    [InlineData(33, "Shadowfang Keep")]
    [InlineData(451, "development")]
    [InlineData(530, "Outland")]
    [InlineData(546, "Coilfang: The Underbog")]
    [InlineData(571, "Northrend")]
    [InlineData(9999, "Map_9999")]
    public void PopulationMapNames_ResolvesCanonicalNames(int mapId, string expected)
    {
        Assert.Equal(expected, PopulationMapNames.GetMapName(mapId));
    }

    [Fact]
    public void RealClientDbcLoading_Loads053And335Correctly()
    {
        string dir053 = @"H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft\Data";
        if (Directory.Exists(dir053))
        {
            using var catalog053 = new MpqArchiveCatalog();
            catalog053.LoadArchives([dir053]);
            var resolver053 = DbcModelResolver.LoadFromArchive(catalog053);
            Assert.True(resolver053.CreatureDisplayCount > 1000, $"Expected >1000 creatures in 0.5.3, got {resolver053.CreatureDisplayCount}");
            Assert.True(resolver053.GameObjectDisplayCount > 300, $"Expected >300 gameobjects in 0.5.3, got {resolver053.GameObjectDisplayCount}");

            // Verify DisplayID 251 (Pat Nagle's test dude)
            Assert.True(resolver053.TryResolveCreature(251, out string? path251, out float scale251));
            Assert.Equal(@"Creature\HumanMaleKid\HumanMaleKid.mdx", path251);

            // Verify DisplayID 907 (Twain)
            Assert.True(resolver053.TryResolveCreature(907, out string? path907, out float scale907));
            Assert.Equal(@"Character\Human\Male\HumanMale.mdx", path907);
        }

        string dir335 = @"H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft";
        if (Directory.Exists(dir335))
        {
            using var catalog335 = new MpqArchiveCatalog();
            catalog335.LoadArchives([dir335]);
            var resolver335 = DbcModelResolver.LoadFromArchive(catalog335);
            Assert.True(resolver335.CreatureDisplayCount > 10000, $"Expected >10000 creatures in 3.3.5, got {resolver335.CreatureDisplayCount}");
            Assert.True(resolver335.GameObjectDisplayCount > 3000, $"Expected >3000 gameobjects in 3.3.5, got {resolver335.GameObjectDisplayCount}");
        }
    }

    [Fact]
    public void WmoOnlyMap_MapRootWmoPreservation_NeverRemovedByPopulationClearing()
    {
        // Simulate a WMO-only map root placement (e.g. Map 229 Blackrock Spire)
        var rootWmo = new WowViewer.Core.Runtime.World.WorldObjectInstance
        {
            ModelPath = @"world\wmo\dungeon\blackrock\blackrock.wmo",
            ModelName = "blackrock.wmo",
            ModelKey = @"world\wmo\dungeon\blackrock\blackrock.wmo",
            UniqueId = 1,
            PlacementEntryIndex = -1,
            TileX = -1,
            TileY = -1,
            HasTileCoordinate = false,
            SpawnEntryId = null,
            AssetKind = null
        };

        // Simulate an external spawn WMO
        var spawnWmo = new WowViewer.Core.Runtime.World.WorldObjectInstance
        {
            ModelPath = @"world\wmo\dungeon\some_prop.wmo",
            ModelName = "some_prop.wmo",
            ModelKey = @"world\wmo\dungeon\some_prop.wmo",
            UniqueId = 5001,
            PlacementEntryIndex = -1,
            TileX = 32,
            TileY = 48,
            HasTileCoordinate = true,
            SpawnEntryId = 12345,
            AssetKind = "GameObject Spawn"
        };

        var instances = new List<WowViewer.Core.Runtime.World.WorldObjectInstance> { rootWmo, spawnWmo };

        // Predicate used by ExternalSpawnLayer
        static bool IsMapRootWmo(in WowViewer.Core.Runtime.World.WorldObjectInstance inst) =>
            !inst.SpawnEntryId.HasValue && string.IsNullOrEmpty(inst.AssetKind);

        // Clearing external spawns must remove spawn WMO but preserve map root WMO
        instances.RemoveAll(static inst => !IsMapRootWmo(in inst));

        Assert.Single(instances);
        Assert.Equal("blackrock.wmo", instances[0].ModelName);
        Assert.Null(instances[0].SpawnEntryId);
        Assert.True(string.IsNullOrEmpty(instances[0].AssetKind));
    }
}
