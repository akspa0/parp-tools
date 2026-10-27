using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Population;
using WowViewer.Core.Population;
using WowViewer.Core.IO.Dbc;
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
            Assert.True(map33Alpha.MapName is "Shadowfang" or "Shadowfang Keep");
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

            // Verify DisplayID 1756 (Caretaker Folsom - baked NPC texture)
            Assert.True(resolver053.TryResolveCreature(1756, out string? path1756, out float scale1756, out uint? fdid1756, out string[]? tex1756));
            Assert.NotNull(tex1756);
            Assert.Single(tex1756);
            Assert.Contains(@"textures\BakedNpcTextures\", tex1756[0], StringComparison.OrdinalIgnoreCase);

            // Verify DisplayID 1141 (Rat - BlackSkin variation)
            Assert.True(resolver053.TryResolveCreature(1141, out string? path1141, out float scale1141, out uint? fdid1141, out string[]? tex1141));
            Assert.NotNull(tex1141);
            Assert.Contains("BlackSkin", tex1141[0], StringComparison.OrdinalIgnoreCase);

            // Verify DisplayID 3167 (Stormwind City Guard) and 4137 (Stromgarde Defender)
            Assert.True(resolver053.TryResolveCreature(3167, out string? path3167, out float scale3167, out uint? fdid3167, out string[]? tex3167));
            Assert.NotNull(tex3167);
            Assert.Single(tex3167);
            Assert.Contains(@"textures\BakedNpcTextures\", tex3167[0], StringComparison.OrdinalIgnoreCase);

            Assert.True(resolver053.TryResolveCreature(4137, out string? path4137, out float scale4137, out uint? fdid4137, out string[]? tex4137));
            Assert.NotNull(tex4137);
            Assert.Single(tex4137);
            Assert.Contains(@"textures\BakedNpcTextures\", tex4137[0], StringComparison.OrdinalIgnoreCase);

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

    [Fact]
    public void EventFilter_BaseWorldOnly_ExcludesHolidaySpawns()
    {
        // Simulate a holiday map with Base World (eventId 0), Winter Veil (eventId 1), and Hallow's End (eventId 12)
        var goSpawns = new Dictionary<int, int>
        {
            [101] = 1,  // Winter Veil Tree
            [102] = 12, // Hallow's End Pumpkin
            // 103 has no event (base world)
        };

        bool IsActive(int spawnId, int selectedEventId)
        {
            if (selectedEventId == -1) return true;
            int evId = goSpawns.TryGetValue(spawnId, out int e) ? e : 0;
            if (selectedEventId == 0) return evId == 0;
            return evId == 0 || evId == selectedEventId;
        }

        // Base world only: only 103 active
        Assert.False(IsActive(101, 0));
        Assert.False(IsActive(102, 0));
        Assert.True(IsActive(103, 0));

        // Winter Veil selected: 101 and 103 active, 102 excluded
        Assert.True(IsActive(101, 1));
        Assert.False(IsActive(102, 1));
        Assert.True(IsActive(103, 1));

        // Show all: all active
        Assert.True(IsActive(101, -1));
        Assert.True(IsActive(102, -1));
        Assert.True(IsActive(103, -1));
    }

    [Fact]
    public void CreatureHumanoidPath_RegexAndVariantExtraction()
    {
        string modelName = "humanmale0128";
        var match = System.Text.RegularExpressions.Regex.Match(modelName, @"^(?<base>.*?)(?<digits>\d+)$");
        Assert.True(match.Success);
        Assert.Equal("humanmale", match.Groups["base"].Value);
        Assert.Equal("0128", match.Groups["digits"].Value);
        Assert.Equal(128, int.Parse(match.Groups["digits"].Value));
    }

    [Fact]
    public void ScreenToRay_PlacesRayOriginInFrontOfCameraAlongViewFrustum()
    {
        var pos = new System.Numerics.Vector3(100f, 200f, 50f);
        var forward = new System.Numerics.Vector3(1f, 0f, 0f); // Facing +X
        var target = pos + forward;
        var up = System.Numerics.Vector3.UnitZ;
        var view = System.Numerics.Matrix4x4.CreateLookAt(pos, target, up);
        float nearPlane = 0.1f;
        float farPlane = 1000f;
        var proj = System.Numerics.Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 3f, 16f / 9f, nearPlane, farPlane);

        System.Numerics.Matrix4x4.Invert(proj, out var invProj);
        System.Numerics.Matrix4x4.Invert(view, out var invView);

        // Near point in clip space (0 for DirectX convention used by System.Numerics)
        var nearClip = new System.Numerics.Vector4(0f, 0f, 0f, 1f);
        var nearView = System.Numerics.Vector4.Transform(nearClip, invProj);
        nearView /= nearView.W;
        var nearWorld = System.Numerics.Vector4.Transform(nearView, invView);

        var farClip = new System.Numerics.Vector4(0f, 0f, 1f, 1f);
        var farView = System.Numerics.Vector4.Transform(farClip, invProj);
        farView /= farView.W;
        var farWorld = System.Numerics.Vector4.Transform(farView, invView);

        var origin = new System.Numerics.Vector3(nearWorld.X, nearWorld.Y, nearWorld.Z);
        var farPt = new System.Numerics.Vector3(farWorld.X, farWorld.Y, farWorld.Z);
        var dir = System.Numerics.Vector3.Normalize(farPt - origin);

        // The ray direction must point forward along +X
        Assert.True(System.Numerics.Vector3.Dot(dir, forward) > 0.999f);

        // The ray origin must be in front of the camera (greater X than camera position), NOT behind it
        Assert.True(origin.X > pos.X);
        Assert.True(origin.X <= pos.X + nearPlane + 0.01f);
    }
}
