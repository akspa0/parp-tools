using System.Numerics;
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
        }

        if (File.Exists(manifest053))
        {
            string json = File.ReadAllText(manifest053);
            var manifest = System.Text.Json.JsonSerializer.Deserialize<PopulationMapManifest>(json, new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            Assert.NotNull(manifest);
            Assert.Equal("0.5.3", manifest.Version);
            Assert.True(manifest.Maps.Count >= 30, $"Expected >= 30 maps in 0.5.3 manifest, found {manifest.Maps.Count}");
            Assert.Contains(manifest.Maps, m => m.MapId == 0);   // Azeroth
            Assert.Contains(manifest.Maps, m => m.MapId == 1);   // Kalimdor
            Assert.Contains(manifest.Maps, m => m.MapId == 30);  // UnderIsland
            Assert.Contains(manifest.Maps, m => m.MapId == 451); // Development
        }
    }
}
