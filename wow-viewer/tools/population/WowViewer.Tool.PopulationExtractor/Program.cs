using System.Text.Json;
using WowViewer.Core.IO.Population;
using WowViewer.Core.Population;

namespace WowViewer.Tool.PopulationExtractor;

internal static class Program
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        IncludeFields = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public static async Task<int> Main(string[] args)
    {
        Console.WriteLine("===================================================================");
        Console.WriteLine("  WoWViewer Population Extractor (TrinityCore & Alpha-Core)");
        Console.WriteLine("===================================================================");

        string? sqlPath = null;
        string? alphaSqlPath = null;
        string? dbcDir = null;
        string version = "3.3.5a";
        string outputDir = Path.Combine(Directory.GetCurrentDirectory(), "output", "population", version);
        var targetMaps = new HashSet<int>();
        bool transposeCorners = true;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i].ToLowerInvariant())
            {
                case "--sql":
                    if (i + 1 < args.Length) sqlPath = args[++i];
                    break;
                case "--alpha-sql":
                    if (i + 1 < args.Length) alphaSqlPath = args[++i];
                    break;
                case "--dbc":
                    if (i + 1 < args.Length) dbcDir = args[++i];
                    break;
                case "--version":
                    if (i + 1 < args.Length) version = args[++i];
                    break;
                case "--output":
                    if (i + 1 < args.Length) outputDir = args[++i];
                    break;
                case "--maps":
                    if (i + 1 < args.Length)
                    {
                        var parts = args[++i].Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        foreach (var p in parts)
                            if (int.TryParse(p, out int mid)) targetMaps.Add(mid);
                    }
                    break;
                case "--no-transpose-corners":
                    transposeCorners = false;
                    break;
            }
        }

        // Auto-detect default TrinityCore SQL if not specified
        if (string.IsNullOrWhiteSpace(sqlPath))
        {
            string defaultTc = @"H:\trinitycore\TDB_full_world_335.26091_2026_09_09.sql";
            if (File.Exists(defaultTc))
                sqlPath = defaultTc;
        }

        // Auto-detect alpha-core SQL if not specified
        if (string.IsNullOrWhiteSpace(alphaSqlPath))
        {
            string[] alphaCandidates =
            {
                @"I:\parp\parp-tools\wow-viewer\libs\alpha-core\etc\databases\world\world.sql",
                Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\..\..\..\..\libs\alpha-core\etc\databases\world\world.sql"))
            };
            foreach (var c in alphaCandidates)
            {
                if (File.Exists(c))
                {
                    alphaSqlPath = c;
                    break;
                }
            }
        }

        if (string.IsNullOrWhiteSpace(sqlPath) || !File.Exists(sqlPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Error: SQL file not found: {sqlPath}");
            Console.ResetColor();
            Console.WriteLine("Usage: dotnet run --project WowViewer.Tool.PopulationExtractor -- --sql <path> [--alpha-sql <path>] [--dbc <path>] [--maps <csv>] [--output <dir>]");
            return 1;
        }

        Directory.CreateDirectory(outputDir);

        Console.WriteLine($"Source SQL:       {sqlPath}");
        Console.WriteLine($"Alpha-Core SQL:   {alphaSqlPath ?? "(none)"}");
        Console.WriteLine($"DBC Directory:    {dbcDir ?? "(fallback definitions)"}");
        Console.WriteLine($"Target Version:   {version}");
        Console.WriteLine($"Output Directory: {outputDir}");
        Console.WriteLine($"Target Maps:      {(targetMaps.Count > 0 ? string.Join(", ", targetMaps) : "All / Auto-detected (including Map 451)")}");
        Console.WriteLine($"Transpose Corners:{transposeCorners}");
        Console.WriteLine();

        // Step 1: Initialize DBC Resolver with built-in fallbacks and client DBCs if provided
        var dbcResolver = new DbcModelResolver();
        RegisterBuiltInDevModelFallbacks(dbcResolver);

        if (!string.IsNullOrWhiteSpace(dbcDir))
        {
            Console.WriteLine($"Loading client DBCs from {dbcDir}...");
            var loadedResolver = DbcModelResolver.LoadFromPath(dbcDir);
            dbcResolver.MergeFrom(loadedResolver);
            Console.WriteLine($"Loaded {loadedResolver.CreatureDisplayCount} creature displays, {loadedResolver.GameObjectDisplayCount} gameobject displays.");
        }

        // Step 2: Read SQL and extract map populations in a single high-throughput pass
        var reader = new TrinityCoreSqlReader();
        Console.WriteLine($"Streaming SQL dump and extracting all maps...");
        var extraction = await reader.ExtractAllMapsAsync(sqlPath, dbcResolver, targetMaps.Count > 0 ? targetMaps : null);
        Console.WriteLine($"Extracted {extraction.TotalCreatureTemplates} creature templates, {extraction.TotalGameObjectTemplates} gameobject templates across {extraction.SpawnsByMap.Count} maps.");

        // Step 3: Handle Prototype Corner Dev Transposition for Map 451
        if (transposeCorners && !string.IsNullOrWhiteSpace(alphaSqlPath) && File.Exists(alphaSqlPath) &&
            (targetMaps.Count == 0 || targetMaps.Contains(451)))
        {
            Console.WriteLine("  Transposing Alpha-Core Map 0 corner test fixtures into Map 451...");
            var alphaCornerSpawns = await reader.ReadCreatureSpawnsAsync(alphaSqlPath, targetMapId: 0);
            var alphaCreatureTemplates = await reader.ReadCreatureTemplatesAsync(alphaSqlPath);

            var cornerRecords = new List<PopulationRecord>();
            foreach (var c in alphaCornerSpawns)
            {
                if (AlphaCoreCornerTransposer.IsCornerTestFixture(c.MapId, c.Position.X, c.Position.Y))
                {
                    alphaCreatureTemplates.TryGetValue(c.Entry, out var tmpl);
                    string name = tmpl?.Name ?? $"Prototype #{c.Entry}";
                    int displayId = (tmpl?.ModelIds != null && tmpl.ModelIds.Length > 0 && tmpl.ModelIds[0] > 0)
                        ? tmpl.ModelIds[0] : 0;

                    dbcResolver.TryResolveCreature(displayId, out string? modelPath, out float displayScale);

                    var orig = new PopulationRecord
                    {
                        SpawnId = c.Guid,
                        EntryId = c.Entry,
                        MapId = c.MapId,
                        SpawnType = PopulationSpawnType.Creature,
                        Name = name,
                        Subname = tmpl?.Subname,
                        ModelPath = modelPath,
                        Scale = tmpl?.Scale ?? 1.0f,
                        DisplayScale = displayScale,
                        EffectiveScale = (tmpl?.Scale ?? 1.0f) * displayScale,
                        Position = c.Position,
                        Orientation = c.Orientation,
                        Rotation = null,
                        Faction = tmpl?.Faction ?? 0,
                        Flags = tmpl?.NpcFlags ?? 0,
                        GameObjectType = 0,
                        IsTransposed = false,
                        SourceNote = null
                    };

                    cornerRecords.Add(AlphaCoreCornerTransposer.TransposeToMap451(orig));
                }
            }

            var alphaCornerGoSpawns = await reader.ReadGameObjectSpawnsAsync(alphaSqlPath, targetMapId: 0);
            var alphaGoTemplates = await reader.ReadGameObjectTemplatesAsync(alphaSqlPath);

            foreach (var g in alphaCornerGoSpawns)
            {
                if (AlphaCoreCornerTransposer.IsCornerTestFixture(g.MapId, g.Position.X, g.Position.Y))
                {
                    alphaGoTemplates.TryGetValue(g.Entry, out var tmpl);
                    string name = tmpl?.Name ?? $"GameObject #{g.Entry}";
                    float scale = tmpl?.Size ?? 1.0f;
                    int displayId = tmpl?.DisplayId ?? 0;
                    int goType = tmpl?.Type ?? 0;

                    dbcResolver.TryResolveGameObject(displayId, out string? modelPath, out float displayScale);

                    var orig = new PopulationRecord
                    {
                        SpawnId = g.Guid,
                        EntryId = g.Entry,
                        MapId = g.MapId,
                        SpawnType = PopulationSpawnType.GameObject,
                        Name = name,
                        Subname = null,
                        ModelPath = modelPath,
                        Scale = scale,
                        DisplayScale = displayScale,
                        EffectiveScale = scale * displayScale,
                        Position = g.Position,
                        Orientation = g.Orientation,
                        Rotation = g.Rotation,
                        Faction = 0,
                        Flags = 0,
                        GameObjectType = goType,
                        IsTransposed = false,
                        SourceNote = null
                    };

                    cornerRecords.Add(AlphaCoreCornerTransposer.TransposeToMap451(orig));
                }
            }

            Console.WriteLine($"  Transposed {cornerRecords.Count} authentic corner test fixtures (Twain, Jesse, pnagle, CHOW Guard, Forges, Anvils, Signposts).");
            if (!extraction.SpawnsByMap.TryGetValue(451, out var map451List))
            {
                map451List = new List<PopulationRecord>();
                extraction.SpawnsByMap[451] = map451List;
            }
            map451List.AddRange(cornerRecords);
        }

        var mapSummaries = new List<PopulationMapSummary>();
        int grandTotalCreatures = 0;
        int grandTotalGameObjects = 0;

        foreach (var (mapId, mapRecords) in extraction.SpawnsByMap.OrderBy(kv => kv.Key))
        {
            if (mapRecords.Count == 0) continue;

            int creatureCount = mapRecords.Count(s => s.SpawnType == PopulationSpawnType.Creature);
            int goCount = mapRecords.Count(s => s.SpawnType == PopulationSpawnType.GameObject);

            string mapFileName = $"map_{mapId}.json";
            string mapFilePath = Path.Combine(outputDir, mapFileName);

            var mapData = new PopulationMapData
            {
                MapId = mapId,
                MapName = GetMapName(mapId),
                ClientVersion = version,
                ExportedUtc = DateTime.UtcNow,
                Spawns = mapRecords
            };

            await File.WriteAllTextAsync(mapFilePath, JsonSerializer.Serialize(mapData, JsonOptions));

            mapSummaries.Add(new PopulationMapSummary
            {
                MapId = mapId,
                MapName = mapData.MapName,
                CreatureCount = creatureCount,
                GameObjectCount = goCount,
                RelativeFilePath = mapFileName
            });

            grandTotalCreatures += creatureCount;
            grandTotalGameObjects += goCount;

            Console.WriteLine($"  Wrote {mapFilePath} ({creatureCount} creatures, {goCount} gameobjects).");
        }

        // Step 4: Write Manifest
        var manifest = new PopulationMapManifest
        {
            Version = version,
            GeneratedUtc = DateTime.UtcNow,
            TotalCreatures = grandTotalCreatures,
            TotalGameObjects = grandTotalGameObjects,
            Maps = mapSummaries
        };

        string manifestPath = Path.Combine(outputDir, "manifest.json");
        await File.WriteAllTextAsync(manifestPath, JsonSerializer.Serialize(manifest, JsonOptions));
        Console.WriteLine();
        Console.WriteLine($"Successfully generated manifest: {manifestPath}");
        Console.WriteLine($"Total across {mapSummaries.Count} maps: {grandTotalCreatures} creatures, {grandTotalGameObjects} gameobjects.");
        return 0;
    }

    private static string GetMapName(int mapId) => mapId switch
    {
        0 => "Eastern Kingdoms",
        1 => "Kalimdor",
        451 => "development",
        530 => "Outland",
        571 => "Northrend",
        _ => $"Map_{mapId}"
    };

    private static void RegisterBuiltInDevModelFallbacks(DbcModelResolver resolver)
    {
        // 3.3.5a Map 451 Creature Models
        resolver.RegisterCreature(14379, @"Creature\FieldRepairBot\FieldRepairBot.m2", 1.25f);

        // Alpha-Core Corner Prototype Test Fixtures (authentic 0.5.3 models)
        resolver.RegisterCreature(907, @"Character\Human\Male\HumanMale.mdx", 1.0f);   // Twain The Tester & Jesse The Tester
        resolver.RegisterCreature(251, @"Creature\HumanMaleKid\HumanMaleKid.mdx", 1.0f); // pnagle's test dude
        resolver.RegisterCreature(2688, @"Character\Human\Male\HumanMale.mdx", 1.0f);  // CHOW Guard

        // 3.3.5a Map 451 GameObject Models
        resolver.RegisterGameObject(399, @"World\Generic\Active\Anvil\Anvil.mdx", 1.0f);          // TEST BRONZE/MARBLE/STONE/SILVER
        resolver.RegisterGameObject(233, @"World\Generic\Active\Forge\Forge.mdx", 1.0f);          // Forge
        resolver.RegisterGameObject(245, @"World\Generic\Passive\Signposts\ScottMarker.mdx", 1.0f);// Scott's Marker
        resolver.RegisterGameObject(273, @"World\Generic\Active\Anvil\Anvil.mdx", 1.0f);          // Anvil
        resolver.RegisterGameObject(153, @"World\Generic\Active\Chests\OrcLock.mdx", 1.0f);       // Orc Lock
        resolver.RegisterGameObject(307, @"World\Generic\Active\FirePit\FirePit.mdx", 1.0f);      // Firepit of Death
    }
}
