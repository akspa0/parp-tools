using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using MdxViewer.Logging;

namespace MdxViewer.Catalog;

/// <summary>
/// Reads creature_template, gameobject_template, spawn tables, and DBC display info
/// from alpha-core SQL dump files (no MySQL required).
/// 
/// Data chain for creatures:
///   creature_template.display_id1 → CreatureDisplayInfo.ModelID → mdx_models_data.ModelName
/// Data chain for gameobjects:
///   gameobject_template.displayId → GameObjectDisplayInfo.ModelName
/// 
/// Expected directory layout:
///   {alphaCoreRoot}/etc/databases/world/world.sql
///   {alphaCoreRoot}/etc/databases/dbc/dbc.sql
/// </summary>
public class AlphaCoreDbReader : IDisposable
{
    private readonly string _worldSqlPath;
    private readonly string _dbcSqlPath;
    private readonly string _worldUpdatesSqlPath;
    private SpawnUpdatesOverlay? _spawnUpdatesOverlay;

    /// <summary>
    /// Create a reader pointing at the alpha-core repository root.
    /// </summary>
    public AlphaCoreDbReader(string alphaCoreRoot)
    {
        _worldSqlPath = Path.Combine(alphaCoreRoot, "etc", "databases", "world", "world.sql");
        _dbcSqlPath = Path.Combine(alphaCoreRoot, "etc", "databases", "dbc", "dbc.sql");
        _worldUpdatesSqlPath = Path.Combine(alphaCoreRoot, "etc", "databases", "world", "updates", "updates.sql");
    }

    /// <summary>
    /// Validate that the SQL dump files exist.
    /// </summary>
    public (bool success, string message) Validate()
    {
        if (!File.Exists(_worldSqlPath))
            return (false, $"world.sql not found: {_worldSqlPath}");
        if (!File.Exists(_dbcSqlPath))
            return (false, $"dbc.sql not found: {_dbcSqlPath}");
        return (true, "SQL dump files found.");
    }

    /// <summary>
    /// Load all creature templates with resolved model paths.
    /// </summary>
    public async Task<List<AssetCatalogEntry>> LoadCreaturesAsync()
    {
        var entries = new List<AssetCatalogEntry>();

        // Step 1: Load CreatureDisplayInfo → ModelID mapping from DBC
        // Columns: ID(0), ModelID(1), SoundID(2), ExtendedDisplayInfoID(3),
        //          CreatureModelScale(4), CreatureModelAlpha(5),
        //          TextureVariation_1(6), TextureVariation_2(7), TextureVariation_3(8), BloodID(9)
        var displayToModel = new Dictionary<int, (int modelId, float scale, string? tex1, string? tex2, string? tex3)>();
        await foreach (var row in ParseInsertRowsAsync(_dbcSqlPath, "CreatureDisplayInfo"))
        {
            if (row.Count < 9) continue;
            int id = ParseInt(row[0]);
            int modelId = ParseInt(row[1]);
            float scale = ParseFloat(row[4]);
            string? tex1 = NullIfEmpty(row[6]);
            string? tex2 = NullIfEmpty(row[7]);
            string? tex3 = NullIfEmpty(row[8]);
            if (scale <= 0) scale = 1.0f;
            displayToModel[id] = (modelId, scale, tex1, tex2, tex3);
        }
        ViewerLog.Trace($"[AssetCatalog] Loaded {displayToModel.Count} CreatureDisplayInfo entries");

        // Step 2: Load mdx_models_data → ModelName mapping from DBC
        // Columns: ID(0), ModelName(1), ModelScale(2), BoundingRadius(3), Height(4)
        // ModelName is often a bare name like "Basilisk" — resolve to "Creature\Basilisk\Basilisk.mdx"
        var modelIdToPath = new Dictionary<int, string>();
        await foreach (var row in ParseInsertRowsAsync(_dbcSqlPath, "mdx_models_data"))
        {
            if (row.Count < 2) continue;
            int id = ParseInt(row[0]);
            string? name = NullIfEmpty(row[1]);
            if (!string.IsNullOrEmpty(name))
                modelIdToPath[id] = ResolveCreatureModelPath(name);
        }
        ViewerLog.Trace($"[AssetCatalog] Loaded {modelIdToPath.Count} mdx_models_data entries");

        // Step 3: Load creature_template
        // Columns: entry(0), display_id1(1), display_id2(2), display_id3(3), display_id4(4),
        //          mount_display_id(5), name(6), subname(7), static_flags(8), gossip_menu_id(9),
        //          level_min(10), level_max(11), faction(12), npc_flags(13), speed_walk(14),
        //          speed_run(15), scale(16), detection_range(17), call_for_help_range(18),
        //          leash_range(19), rank(20), ... type(34), ...
        await foreach (var row in ParseInsertRowsAsync(_worldSqlPath, "creature_template"))
        {
            if (row.Count < 35) continue;
            int entryId = ParseInt(row[0]);
            int did1 = ParseInt(row[1]);
            int did2 = ParseInt(row[2]);
            int did3 = ParseInt(row[3]);
            int did4 = ParseInt(row[4]);
            string name = UnescapeSql(row[6]).Trim('\0');
            string? subname = NullIfEmpty(row[7])?.Trim('\0');
            int levelMin = ParseInt(row[10]);
            int levelMax = ParseInt(row[11]);
            int faction = ParseInt(row[12]);
            int npcFlags = ParseInt(row[13]);
            float scale = ParseFloat(row[16]);
            int rank = ParseInt(row[20]);
            int creatureType = ParseInt(row[34]);
            if (scale <= 0) scale = 1.0f;

            // Resolve primary display ID → model path
            int primaryDid = did1 > 0 ? did1 : did2 > 0 ? did2 : did3 > 0 ? did3 : did4;
            string? modelPath = null;
            float displayScale = 1.0f;
            var texVariations = new List<string>();

            if (primaryDid > 0 && displayToModel.TryGetValue(primaryDid, out var displayInfo))
            {
                displayScale = displayInfo.scale;
                if (displayInfo.modelId > 0 && modelIdToPath.TryGetValue(displayInfo.modelId, out var path))
                    modelPath = path;
                if (!string.IsNullOrEmpty(displayInfo.tex1)) texVariations.Add(displayInfo.tex1);
                if (!string.IsNullOrEmpty(displayInfo.tex2)) texVariations.Add(displayInfo.tex2);
                if (!string.IsNullOrEmpty(displayInfo.tex3)) texVariations.Add(displayInfo.tex3);
            }

            var allDids = new[] { did1, did2, did3, did4 }.Where(d => d > 0).ToArray();

            entries.Add(new AssetCatalogEntry
            {
                EntryId = entryId,
                Type = AssetType.Creature,
                Name = name,
                Subname = string.IsNullOrEmpty(subname) ? null : subname,
                DisplayId = primaryDid,
                AllDisplayIds = allDids,
                ModelPath = modelPath,
                Scale = scale,
                DisplayScale = displayScale,
                TextureVariations = texVariations.ToArray(),
                LevelMin = levelMin,
                LevelMax = levelMax,
                Rank = rank,
                CreatureType = creatureType,
                Faction = faction,
                NpcFlags = npcFlags
            });
        }
        ViewerLog.Trace($"[AssetCatalog] Loaded {entries.Count} creature templates");

        return entries;
    }

    /// <summary>
    /// Load all gameobject templates with resolved model paths.
    /// </summary>
    public async Task<List<AssetCatalogEntry>> LoadGameObjectsAsync()
    {
        var entries = new List<AssetCatalogEntry>();

        // Step 1: Load GameObjectDisplayInfo → ModelName from DBC
        // Columns: ID(0), ModelName(1), Sound_1..Sound_10(2..11)
        var goDisplayToModel = new Dictionary<int, string>();
        await foreach (var row in ParseInsertRowsAsync(_dbcSqlPath, "GameObjectDisplayInfo"))
        {
            if (row.Count < 2) continue;
            int id = ParseInt(row[0]);
            string? name = NullIfEmpty(row[1]);
            if (!string.IsNullOrEmpty(name))
                goDisplayToModel[id] = name;
        }
        ViewerLog.Trace($"[AssetCatalog] Loaded {goDisplayToModel.Count} GameObjectDisplayInfo entries");

        // Step 2: Load gameobject_template
        // Columns: entry(0), type(1), displayId(2), name(3), faction(4), flags(5), size(6), ...
        await foreach (var row in ParseInsertRowsAsync(_worldSqlPath, "gameobject_template"))
        {
            if (row.Count < 7) continue;
            int entryId = ParseInt(row[0]);
            int goType = ParseInt(row[1]);
            int displayId = ParseInt(row[2]);
            string name = UnescapeSql(row[3]).Trim('\0');
            int faction = ParseInt(row[4]);
            int flags = ParseInt(row[5]);
            float scale = ParseFloat(row[6]);
            if (scale <= 0) scale = 1.0f;

            string? modelPath = null;
            if (displayId > 0 && goDisplayToModel.TryGetValue(displayId, out var path))
                modelPath = path;

            entries.Add(new AssetCatalogEntry
            {
                EntryId = entryId,
                Type = AssetType.GameObject,
                Name = name,
                DisplayId = displayId,
                AllDisplayIds = displayId > 0 ? new[] { displayId } : Array.Empty<int>(),
                ModelPath = modelPath,
                Scale = scale,
                GameObjectType = goType,
                Faction = faction,
                Flags = flags
            });
        }
        ViewerLog.Trace($"[AssetCatalog] Loaded {entries.Count} gameobject templates");

        return entries;
    }

    /// <summary>
    /// Load spawn locations for a set of creature entries.
    /// </summary>
    public async Task LoadCreatureSpawnsAsync(IEnumerable<AssetCatalogEntry> creatures)
    {
        var byEntry = creatures.Where(c => c.Type == AssetType.Creature)
            .ToDictionary(c => c.EntryId);
        if (byEntry.Count == 0) return;
        var updates = GetSpawnUpdatesOverlay();
        var seenSpawnIds = new HashSet<int>();

        // Columns: spawn_id(0), spawn_entry1(1), spawn_entry2(2), spawn_entry3(3), spawn_entry4(4),
        //          map(5), position_x(6), position_y(7), position_z(8), orientation(9), ...
        int count = 0;
        int totalRows = 0;
        int ignoredRows = 0;
        int variantRows = 0;
        int unmatchedEntries = 0;
        await foreach (var row in ParseInsertRowsAsync(_worldSqlPath, "spawns_creatures"))
        {
            if (row.Count < 10) continue;
            totalRows++;

            // Schema: ignored at index 18
            if (row.Count > 18 && ParseInt(row[18]) != 0)
            {
                ignoredRows++;
                continue;
            }

            int spawnId = ParseInt(row[0]);
            seenSpawnIds.Add(spawnId);

            int entry1 = ParseInt(row[1]);
            int entry2 = row.Count > 2 ? ParseInt(row[2]) : 0;
            int entry3 = row.Count > 3 ? ParseInt(row[3]) : 0;
            int entry4 = row.Count > 4 ? ParseInt(row[4]) : 0;
            int mapId = ParseInt(row[5]);
            float x = ParseFloat(row[6]);
            float y = ParseFloat(row[7]);
            float z = ParseFloat(row[8]);
            float o = ParseFloat(row[9]);
            int ignored = row.Count > 18 ? ParseInt(row[18]) : 0;

            if (updates.CreatureBySpawnId.TryGetValue(spawnId, out var patch))
            {
                if (patch.SpawnEntry1.HasValue) entry1 = patch.SpawnEntry1.Value;
                if (patch.SpawnEntry2.HasValue) entry2 = patch.SpawnEntry2.Value;
                if (patch.SpawnEntry3.HasValue) entry3 = patch.SpawnEntry3.Value;
                if (patch.SpawnEntry4.HasValue) entry4 = patch.SpawnEntry4.Value;
                if (patch.MapId.HasValue) mapId = patch.MapId.Value;
                if (patch.X.HasValue) x = patch.X.Value;
                if (patch.Y.HasValue) y = patch.Y.Value;
                if (patch.Z.HasValue) z = patch.Z.Value;
                if (patch.Orientation.HasValue) o = patch.Orientation.Value;
                if (patch.Ignored.HasValue) ignored = patch.Ignored.Value;
            }

            if (ignored != 0)
            {
                ignoredRows++;
                continue;
            }

            int[] entries =
            {
                entry1,
                entry2,
                entry3,
                entry4
            };

            var uniqueEntries = new HashSet<int>(entries.Where(e => e > 0));
            if (uniqueEntries.Count > 1)
                variantRows++;

            foreach (int entry in uniqueEntries)
            {
                if (!byEntry.TryGetValue(entry, out var creature))
                {
                    unmatchedEntries++;
                    continue;
                }

                creature.Spawns.Add(new SpawnLocation
                {
                    SpawnId = spawnId, MapId = mapId,
                    X = x, Y = y, Z = z, Orientation = o
                });
                count++;
            }
        }

        // Apply inserted creature spawns from updates.sql
        foreach (var inserted in updates.CreatureInserts)
        {
            if (inserted.Ignored != 0) continue;
            if (!seenSpawnIds.Add(inserted.SpawnId)) continue;

            int[] entries = { inserted.SpawnEntry1, inserted.SpawnEntry2, inserted.SpawnEntry3, inserted.SpawnEntry4 };
            var uniqueEntries = new HashSet<int>(entries.Where(e => e > 0));

            foreach (int entry in uniqueEntries)
            {
                if (!byEntry.TryGetValue(entry, out var creature))
                {
                    unmatchedEntries++;
                    continue;
                }

                creature.Spawns.Add(new SpawnLocation
                {
                    SpawnId = inserted.SpawnId,
                    MapId = inserted.MapId,
                    X = inserted.X,
                    Y = inserted.Y,
                    Z = inserted.Z,
                    Orientation = inserted.Orientation
                });
                count++;
            }
        }

        ViewerLog.Trace($"[AssetCatalog] Loaded {count} creature spawn links from {totalRows} rows ({ignoredRows} ignored, {variantRows} variant rows, {unmatchedEntries} unmatched entries)");
    }

    /// <summary>
    /// Load spawn locations for a set of gameobject entries.
    /// </summary>
    public async Task LoadGameObjectSpawnsAsync(IEnumerable<AssetCatalogEntry> gameObjects)
    {
        var byEntry = gameObjects.Where(g => g.Type == AssetType.GameObject)
            .ToDictionary(g => g.EntryId);
        if (byEntry.Count == 0) return;
        var updates = GetSpawnUpdatesOverlay();
        var seenSpawnIds = new HashSet<int>();

        // Columns: spawn_id(0), spawn_entry(1), spawn_map(2),
        //          spawn_positionX(3), spawn_positionY(4), spawn_positionZ(5), spawn_orientation(6), ...
        int count = 0;
        int totalRows = 0;
        int ignoredRows = 0;
        int unmatchedEntries = 0;
        await foreach (var row in ParseInsertRowsAsync(_worldSqlPath, "spawns_gameobjects"))
        {
            if (row.Count < 7) continue;
            totalRows++;

            // Schema: ignored at index 17
            if (row.Count > 17 && ParseInt(row[17]) != 0)
            {
                ignoredRows++;
                continue;
            }

            int spawnId = ParseInt(row[0]);
            seenSpawnIds.Add(spawnId);

            int entry = ParseInt(row[1]);
            int mapId = ParseInt(row[2]);
            float x = ParseFloat(row[3]);
            float y = ParseFloat(row[4]);
            float z = ParseFloat(row[5]);
            float o = ParseFloat(row[6]);
            int ignored = row.Count > 17 ? ParseInt(row[17]) : 0;

            if (updates.GameObjectBySpawnId.TryGetValue(spawnId, out var patch))
            {
                if (patch.SpawnEntry.HasValue) entry = patch.SpawnEntry.Value;
                if (patch.MapId.HasValue) mapId = patch.MapId.Value;
                if (patch.X.HasValue) x = patch.X.Value;
                if (patch.Y.HasValue) y = patch.Y.Value;
                if (patch.Z.HasValue) z = patch.Z.Value;
                if (patch.Orientation.HasValue) o = patch.Orientation.Value;
                if (patch.Ignored.HasValue) ignored = patch.Ignored.Value;
            }

            if (updates.GameObjectIgnoredByEntry.TryGetValue(entry, out int ignoredByEntry) && ignoredByEntry != 0)
                ignored = ignoredByEntry;

            if (ignored != 0)
            {
                ignoredRows++;
                continue;
            }

            if (byEntry.TryGetValue(entry, out var go))
            {
                go.Spawns.Add(new SpawnLocation
                {
                    SpawnId = spawnId, MapId = mapId,
                    X = x, Y = y, Z = z, Orientation = o
                });
                count++;
            }
            else
            {
                unmatchedEntries++;
            }
        }

        foreach (var inserted in updates.GameObjectInserts)
        {
            if (inserted.Ignored != 0) continue;
            if (!seenSpawnIds.Add(inserted.SpawnId)) continue;

            if (updates.GameObjectIgnoredByEntry.TryGetValue(inserted.SpawnEntry, out int ignoredByEntry) && ignoredByEntry != 0)
                continue;

            if (byEntry.TryGetValue(inserted.SpawnEntry, out var go))
            {
                go.Spawns.Add(new SpawnLocation
                {
                    SpawnId = inserted.SpawnId,
                    MapId = inserted.MapId,
                    X = inserted.X,
                    Y = inserted.Y,
                    Z = inserted.Z,
                    Orientation = inserted.Orientation
                });
                count++;
            }
            else
            {
                unmatchedEntries++;
            }
        }

        ViewerLog.Trace($"[AssetCatalog] Loaded {count} gameobject spawns from {totalRows} rows ({ignoredRows} ignored, {unmatchedEntries} unmatched entries)");
    }

    /// <summary>
    /// Load everything: all creatures + gameobjects with resolved models and spawns.
    /// </summary>
    public async Task<List<AssetCatalogEntry>> LoadAllAsync()
    {
        var creatures = await LoadCreaturesAsync();
        var gameObjects = await LoadGameObjectsAsync();

        await LoadCreatureSpawnsAsync(creatures);
        await LoadGameObjectSpawnsAsync(gameObjects);

        var all = new List<AssetCatalogEntry>(creatures.Count + gameObjects.Count);
        all.AddRange(creatures);
        all.AddRange(gameObjects);

        int withModel = all.Count(e => e.ModelPath != null);
        int withSpawns = all.Count(e => e.Spawns.Count > 0);
        ViewerLog.Trace($"[AssetCatalog] Total: {all.Count} entries ({withModel} with models, {withSpawns} with spawns)");

        return all;
    }

    public void Dispose() { }

    private SpawnUpdatesOverlay GetSpawnUpdatesOverlay()
    {
        if (_spawnUpdatesOverlay != null)
            return _spawnUpdatesOverlay;

        var overlay = new SpawnUpdatesOverlay();
        if (!File.Exists(_worldUpdatesSqlPath))
        {
            _spawnUpdatesOverlay = overlay;
            return overlay;
        }

        string sql = File.ReadAllText(_worldUpdatesSqlPath, Encoding.UTF8);

        ParseCreatureSpawnUpdates(sql, overlay);
        ParseGameObjectSpawnUpdates(sql, overlay);
        ParseCreatureSpawnInserts(sql, overlay);
        ParseGameObjectSpawnInserts(sql, overlay);

        ViewerLog.Trace($"[AssetCatalog] updates.sql overlay: {overlay.CreatureBySpawnId.Count} creature updates, {overlay.GameObjectBySpawnId.Count} gameobject updates, {overlay.CreatureInserts.Count} creature inserts, {overlay.GameObjectInserts.Count} gameobject inserts, {overlay.GameObjectIgnoredByEntry.Count} entry-level GO ignores");

        _spawnUpdatesOverlay = overlay;
        return overlay;
    }

    private static void ParseCreatureSpawnUpdates(string sql, SpawnUpdatesOverlay overlay)
    {
        var re = new Regex(@"UPDATE\s+`spawns_creatures`\s+SET\s+(?<set>.*?)\s+WHERE\s+(?<where>.*?);", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        foreach (Match match in re.Matches(sql))
        {
            var setValues = ParseSetClause(match.Groups["set"].Value);
            var patch = new CreatureSpawnPatch
            {
                SpawnEntry1 = GetInt(setValues, "spawn_entry1"),
                SpawnEntry2 = GetInt(setValues, "spawn_entry2"),
                SpawnEntry3 = GetInt(setValues, "spawn_entry3"),
                SpawnEntry4 = GetInt(setValues, "spawn_entry4"),
                MapId = GetInt(setValues, "map"),
                X = GetFloat(setValues, "position_x"),
                Y = GetFloat(setValues, "position_y"),
                Z = GetFloat(setValues, "position_z"),
                Orientation = GetFloat(setValues, "orientation"),
                Ignored = GetInt(setValues, "ignored")
            };

            foreach (int spawnId in ParseSpawnIdsFromWhere(match.Groups["where"].Value))
                overlay.CreatureBySpawnId[spawnId] = patch;
        }
    }

    private static void ParseGameObjectSpawnUpdates(string sql, SpawnUpdatesOverlay overlay)
    {
        var re = new Regex(@"UPDATE\s+`spawns_gameobjects`\s+SET\s+(?<set>.*?)\s+WHERE\s+(?<where>.*?);", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        foreach (Match match in re.Matches(sql))
        {
            var setValues = ParseSetClause(match.Groups["set"].Value);
            var patch = new GameObjectSpawnPatch
            {
                SpawnEntry = GetInt(setValues, "spawn_entry"),
                MapId = GetInt(setValues, "spawn_map"),
                X = GetFloat(setValues, "spawn_positionx"),
                Y = GetFloat(setValues, "spawn_positiony"),
                Z = GetFloat(setValues, "spawn_positionz"),
                Orientation = GetFloat(setValues, "spawn_orientation"),
                Ignored = GetInt(setValues, "ignored")
            };

            string where = match.Groups["where"].Value;
            foreach (int spawnId in ParseSpawnIdsFromWhere(where))
                overlay.GameObjectBySpawnId[spawnId] = patch;

            foreach (int entry in ParseEntryIdsFromWhere(where, "spawn_entry"))
            {
                if (patch.Ignored.HasValue)
                    overlay.GameObjectIgnoredByEntry[entry] = patch.Ignored.Value;
            }
        }
    }

    private static void ParseCreatureSpawnInserts(string sql, SpawnUpdatesOverlay overlay)
    {
        var re = new Regex(@"INSERT\s+INTO\s+`spawns_creatures`\s*\((?<cols>[^)]*)\)\s*VALUES\s*(?<vals>.*?);", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        foreach (Match match in re.Matches(sql))
        {
            var cols = SplitCsv(match.Groups["cols"].Value).Select(NormalizeSqlIdentifier).ToList();
            foreach (var tuple in ExtractTuples(match.Groups["vals"].Value))
            {
                var values = ParseTupleValues(tuple);
                if (values.Count == 0 || values.Count != cols.Count) continue;

                var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                for (int i = 0; i < cols.Count; i++)
                    map[cols[i]] = values[i];

                var inserted = new CreatureSpawnInserted
                {
                    SpawnId = GetInt(map, "spawn_id") ?? 0,
                    SpawnEntry1 = GetInt(map, "spawn_entry1") ?? 0,
                    SpawnEntry2 = GetInt(map, "spawn_entry2") ?? 0,
                    SpawnEntry3 = GetInt(map, "spawn_entry3") ?? 0,
                    SpawnEntry4 = GetInt(map, "spawn_entry4") ?? 0,
                    MapId = GetInt(map, "map") ?? 0,
                    X = GetFloat(map, "position_x") ?? 0,
                    Y = GetFloat(map, "position_y") ?? 0,
                    Z = GetFloat(map, "position_z") ?? 0,
                    Orientation = GetFloat(map, "orientation") ?? 0,
                    Ignored = GetInt(map, "ignored") ?? 0
                };

                if (inserted.SpawnId > 0)
                    overlay.CreatureInserts.Add(inserted);
            }
        }
    }

    private static void ParseGameObjectSpawnInserts(string sql, SpawnUpdatesOverlay overlay)
    {
        var re = new Regex(@"INSERT\s+INTO\s+`spawns_gameobjects`\s*\((?<cols>[^)]*)\)\s*VALUES\s*(?<vals>.*?);", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        foreach (Match match in re.Matches(sql))
        {
            var cols = SplitCsv(match.Groups["cols"].Value).Select(NormalizeSqlIdentifier).ToList();
            foreach (var tuple in ExtractTuples(match.Groups["vals"].Value))
            {
                var values = ParseTupleValues(tuple);
                if (values.Count == 0 || values.Count != cols.Count) continue;

                var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                for (int i = 0; i < cols.Count; i++)
                    map[cols[i]] = values[i];

                var inserted = new GameObjectSpawnInserted
                {
                    SpawnId = GetInt(map, "spawn_id") ?? 0,
                    SpawnEntry = GetInt(map, "spawn_entry") ?? 0,
                    MapId = GetInt(map, "spawn_map") ?? 0,
                    X = GetFloat(map, "spawn_positionx") ?? 0,
                    Y = GetFloat(map, "spawn_positiony") ?? 0,
                    Z = GetFloat(map, "spawn_positionz") ?? 0,
                    Orientation = GetFloat(map, "spawn_orientation") ?? 0,
                    Ignored = GetInt(map, "ignored") ?? 0
                };

                if (inserted.SpawnId > 0)
                    overlay.GameObjectInserts.Add(inserted);
            }
        }
    }

    private static Dictionary<string, string> ParseSetClause(string setClause)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string part in SplitCsv(setClause))
        {
            int eq = part.IndexOf('=');
            if (eq <= 0) continue;
            string key = NormalizeSqlIdentifier(part[..eq]);
            string val = part[(eq + 1)..].Trim();
            val = TrimSqlLiteral(val);
            result[key] = val;
        }
        return result;
    }

    private static IEnumerable<int> ParseSpawnIdsFromWhere(string whereClause)
    {
        var ids = new HashSet<int>();
        foreach (Match m in Regex.Matches(whereClause, @"spawn_id`?\s*=\s*'?(?<id>-?\d+)'?", RegexOptions.IgnoreCase))
            ids.Add(ParseInt(m.Groups["id"].Value));

        foreach (Match m in Regex.Matches(whereClause, @"spawn_id`?\s+in\s*\((?<list>[^)]*)\)", RegexOptions.IgnoreCase))
        {
            foreach (Match n in Regex.Matches(m.Groups["list"].Value, @"-?\d+"))
                ids.Add(ParseInt(n.Value));
        }
        return ids;
    }

    private static IEnumerable<int> ParseEntryIdsFromWhere(string whereClause, string columnName)
    {
        var ids = new HashSet<int>();
        string esc = Regex.Escape(columnName);

        foreach (Match m in Regex.Matches(whereClause, $@"{esc}`?\s*=\s*'?(?<id>-?\d+)'?", RegexOptions.IgnoreCase))
            ids.Add(ParseInt(m.Groups["id"].Value));

        foreach (Match m in Regex.Matches(whereClause, $@"{esc}`?\s+in\s*\((?<list>[^)]*)\)", RegexOptions.IgnoreCase))
        {
            foreach (Match n in Regex.Matches(m.Groups["list"].Value, @"-?\d+"))
                ids.Add(ParseInt(n.Value));
        }

        return ids;
    }

    private static List<string> SplitCsv(string input)
    {
        var result = new List<string>();
        var current = new StringBuilder();
        bool inQuote = false;

        for (int i = 0; i < input.Length; i++)
        {
            char c = input[i];
            if (c == '\'' && (i == 0 || input[i - 1] != '\\'))
                inQuote = !inQuote;

            if (c == ',' && !inQuote)
            {
                result.Add(current.ToString().Trim());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }

        if (current.Length > 0)
            result.Add(current.ToString().Trim());

        return result;
    }

    private static List<string> ExtractTuples(string valuesClause)
    {
        var tuples = new List<string>();
        int depth = 0;
        int start = -1;
        bool inQuote = false;

        for (int i = 0; i < valuesClause.Length; i++)
        {
            char c = valuesClause[i];
            if (c == '\'' && (i == 0 || valuesClause[i - 1] != '\\'))
                inQuote = !inQuote;

            if (inQuote) continue;

            if (c == '(')
            {
                if (depth == 0) start = i;
                depth++;
            }
            else if (c == ')')
            {
                depth--;
                if (depth == 0 && start >= 0)
                {
                    tuples.Add(valuesClause[start..(i + 1)]);
                    start = -1;
                }
            }
        }

        return tuples;
    }

    private static List<string> ParseTupleValues(string tuple)
    {
        var sb = new StringBuilder(tuple.Trim());
        var row = ParseRowTuple(sb, out _);
        return row ?? new List<string>();
    }

    private static string NormalizeSqlIdentifier(string raw)
    {
        return raw.Trim().Trim('`').Trim().ToLowerInvariant();
    }

    private static string TrimSqlLiteral(string raw)
    {
        string value = raw.Trim();
        if (value.StartsWith("(") && value.EndsWith(")"))
            value = value[1..^1].Trim();
        if (value.StartsWith("'") && value.EndsWith("'"))
            value = value[1..^1];
        return value.Trim();
    }

    private static int? GetInt(Dictionary<string, string> values, string key)
    {
        if (!values.TryGetValue(key, out string? raw)) return null;
        return ParseInt(raw);
    }

    private static float? GetFloat(Dictionary<string, string> values, string key)
    {
        if (!values.TryGetValue(key, out string? raw)) return null;
        return ParseFloat(raw);
    }

    private sealed class SpawnUpdatesOverlay
    {
        public Dictionary<int, CreatureSpawnPatch> CreatureBySpawnId { get; } = new();
        public Dictionary<int, GameObjectSpawnPatch> GameObjectBySpawnId { get; } = new();
        public Dictionary<int, int> GameObjectIgnoredByEntry { get; } = new();
        public List<CreatureSpawnInserted> CreatureInserts { get; } = new();
        public List<GameObjectSpawnInserted> GameObjectInserts { get; } = new();
    }

    private sealed class CreatureSpawnPatch
    {
        public int? SpawnEntry1 { get; init; }
        public int? SpawnEntry2 { get; init; }
        public int? SpawnEntry3 { get; init; }
        public int? SpawnEntry4 { get; init; }
        public int? MapId { get; init; }
        public float? X { get; init; }
        public float? Y { get; init; }
        public float? Z { get; init; }
        public float? Orientation { get; init; }
        public int? Ignored { get; init; }
    }

    private sealed class GameObjectSpawnPatch
    {
        public int? SpawnEntry { get; init; }
        public int? MapId { get; init; }
        public float? X { get; init; }
        public float? Y { get; init; }
        public float? Z { get; init; }
        public float? Orientation { get; init; }
        public int? Ignored { get; init; }
    }

    private sealed class CreatureSpawnInserted
    {
        public int SpawnId { get; init; }
        public int SpawnEntry1 { get; init; }
        public int SpawnEntry2 { get; init; }
        public int SpawnEntry3 { get; init; }
        public int SpawnEntry4 { get; init; }
        public int MapId { get; init; }
        public float X { get; init; }
        public float Y { get; init; }
        public float Z { get; init; }
        public float Orientation { get; init; }
        public int Ignored { get; init; }
    }

    private sealed class GameObjectSpawnInserted
    {
        public int SpawnId { get; init; }
        public int SpawnEntry { get; init; }
        public int MapId { get; init; }
        public float X { get; init; }
        public float Y { get; init; }
        public float Z { get; init; }
        public float Orientation { get; init; }
        public int Ignored { get; init; }
    }

    // ─── SQL Dump Parser ───────────────────────────────────────────────

    /// <summary>
    /// Stream-parse a MySQL dump file, yielding each value-tuple row for the given table.
    /// Handles: INSERT INTO `tableName` VALUES (...),(...); across multiple lines.
    /// </summary>
    private async IAsyncEnumerable<List<string>> ParseInsertRowsAsync(string sqlPath, string tableName)
    {
        string insertPrefix = $"INSERT INTO `{tableName}` VALUES";
        bool inInsert = false;
        var buffer = new StringBuilder();

        using var reader = new StreamReader(sqlPath, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 1 << 16);
        string? line;
        while ((line = await reader.ReadLineAsync()) != null)
        {
            if (!inInsert)
            {
                if (line.StartsWith(insertPrefix, StringComparison.OrdinalIgnoreCase))
                {
                    inInsert = true;
                    // The data starts after "INSERT INTO `tableName` VALUES\n" or on the same line
                    string remainder = line.Substring(insertPrefix.Length).TrimStart();
                    if (remainder.Length > 0)
                        buffer.Append(remainder);
                }
                continue;
            }

            // We're inside an INSERT block — accumulate lines
            buffer.Append(line);

            // Process complete rows from the buffer
            while (buffer.Length > 0)
            {
                // Skip whitespace
                int pos = 0;
                while (pos < buffer.Length && char.IsWhiteSpace(buffer[pos])) pos++;
                if (pos > 0) buffer.Remove(0, pos);
                if (buffer.Length == 0) break;

                char ch = buffer[0];

                if (ch == '(')
                {
                    // Parse one row tuple
                    var row = ParseRowTuple(buffer, out int consumed);
                    if (row == null)
                        break; // incomplete row, need more data
                    buffer.Remove(0, consumed);
                    yield return row;

                    // After a row: expect ',' (more rows), ';' (end of INSERT), or newline
                    while (buffer.Length > 0 && char.IsWhiteSpace(buffer[0]))
                        buffer.Remove(0, 1);
                    if (buffer.Length > 0 && buffer[0] == ',')
                    {
                        buffer.Remove(0, 1); // consume comma, continue to next row
                    }
                    else if (buffer.Length > 0 && buffer[0] == ';')
                    {
                        buffer.Remove(0, 1);
                        inInsert = false;
                        buffer.Clear();
                        break;
                    }
                }
                else if (ch == ';')
                {
                    inInsert = false;
                    buffer.Clear();
                    break;
                }
                else
                {
                    // Unexpected character — skip this INSERT block
                    inInsert = false;
                    buffer.Clear();
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Parse a single parenthesized value tuple from the buffer: (val1,val2,...).
    /// Returns null if the buffer doesn't contain a complete tuple yet.
    /// </summary>
    private static List<string>? ParseRowTuple(StringBuilder buffer, out int consumed)
    {
        consumed = 0;
        if (buffer.Length == 0 || buffer[0] != '(') return null;

        var values = new List<string>();
        int pos = 1; // skip opening '('
        var field = new StringBuilder();

        while (pos < buffer.Length)
        {
            char c = buffer[pos];

            if (c == '\'')
            {
                // Quoted string — read until unescaped closing quote
                pos++; // skip opening quote
                field.Clear();
                while (pos < buffer.Length)
                {
                    c = buffer[pos];
                    if (c == '\\' && pos + 1 < buffer.Length)
                    {
                        // Escaped character
                        char next = buffer[pos + 1];
                        field.Append(next switch
                        {
                            'n' => '\n',
                            'r' => '\r',
                            't' => '\t',
                            '0' => '\0',
                            _ => next // \', \\, etc.
                        });
                        pos += 2;
                    }
                    else if (c == '\'')
                    {
                        pos++; // skip closing quote
                        break;
                    }
                    else
                    {
                        field.Append(c);
                        pos++;
                    }
                }
                values.Add(field.ToString());
            }
            else if (c == ',' )
            {
                // End of unquoted field (already accumulated in field)
                values.Add(field.ToString());
                field.Clear();
                pos++;
            }
            else if (c == ')')
            {
                // End of tuple
                values.Add(field.ToString());
                field.Clear();
                pos++; // skip closing ')'
                consumed = pos;
                return values;
            }
            else if (c == 'N' && pos + 3 < buffer.Length &&
                     buffer[pos + 1] == 'U' && buffer[pos + 2] == 'L' && buffer[pos + 3] == 'L')
            {
                field.Append("NULL");
                pos += 4;
            }
            else
            {
                // Part of an unquoted value (number, etc.)
                field.Append(c);
                pos++;
            }
        }

        // Incomplete tuple
        return null;
    }

    private static int ParseInt(string s)
    {
        if (string.IsNullOrEmpty(s) || s == "NULL") return 0;
        return int.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out int v) ? v : 0;
    }

    private static float ParseFloat(string s)
    {
        if (string.IsNullOrEmpty(s) || s == "NULL") return 0f;
        return float.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out float v) ? v : 0f;
    }

    private static string? NullIfEmpty(string s)
    {
        if (string.IsNullOrEmpty(s) || s == "NULL") return null;
        return s;
    }

    private static string UnescapeSql(string s)
    {
        if (string.IsNullOrEmpty(s) || s == "NULL") return "";
        return s;
    }

    /// <summary>
    /// Resolve a creature model name from mdx_models_data to a full virtual path.
    /// 
    /// mdx_models_data stores bare names like "Basilisk", "kobold", "Wolf".
    /// The actual MPQ paths follow the convention: Creature\{Name}\{Name}.mdx
    /// 
    /// If the name already contains a path separator or .mdx extension, it's returned as-is.
    /// </summary>
    private static string ResolveCreatureModelPath(string modelName)
    {
        // Already a full path (contains backslash or forward slash)?
        if (modelName.Contains('\\') || modelName.Contains('/'))
            return modelName;

        // Already has .mdx extension?
        if (modelName.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            string nameNoExt = Path.GetFileNameWithoutExtension(modelName);
            return $"Creature\\{nameNoExt}\\{modelName}";
        }

        // Bare name — construct Creature\Name\Name.mdx
        return $"Creature\\{modelName}\\{modelName}.mdx";
    }
}
