using System.Globalization;
using System.Text;

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

    /// <summary>
    /// Create a reader pointing at the alpha-core repository root.
    /// </summary>
    public AlphaCoreDbReader(string alphaCoreRoot)
    {
        _worldSqlPath = Path.Combine(alphaCoreRoot, "etc", "databases", "world", "world.sql");
        _dbcSqlPath = Path.Combine(alphaCoreRoot, "etc", "databases", "dbc", "dbc.sql");
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
        Console.WriteLine($"[AssetCatalog] Loaded {displayToModel.Count} CreatureDisplayInfo entries");

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
        Console.WriteLine($"[AssetCatalog] Loaded {modelIdToPath.Count} mdx_models_data entries");

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
        Console.WriteLine($"[AssetCatalog] Loaded {entries.Count} creature templates");

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
        Console.WriteLine($"[AssetCatalog] Loaded {goDisplayToModel.Count} GameObjectDisplayInfo entries");

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
        Console.WriteLine($"[AssetCatalog] Loaded {entries.Count} gameobject templates");

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

        // Columns: spawn_id(0), spawn_entry1(1), spawn_entry2(2), spawn_entry3(3), spawn_entry4(4),
        //          map(5), position_x(6), position_y(7), position_z(8), orientation(9), ...
        int count = 0;
        await foreach (var row in ParseInsertRowsAsync(_worldSqlPath, "spawns_creatures"))
        {
            if (row.Count < 10) continue;
            int spawnId = ParseInt(row[0]);
            int entry1 = ParseInt(row[1]);
            int mapId = ParseInt(row[5]);
            float x = ParseFloat(row[6]);
            float y = ParseFloat(row[7]);
            float z = ParseFloat(row[8]);
            float o = ParseFloat(row[9]);

            if (byEntry.TryGetValue(entry1, out var creature))
            {
                creature.Spawns.Add(new SpawnLocation
                {
                    SpawnId = spawnId, MapId = mapId,
                    X = x, Y = y, Z = z, Orientation = o
                });
                count++;
            }
        }
        Console.WriteLine($"[AssetCatalog] Loaded {count} creature spawns");
    }

    /// <summary>
    /// Load spawn locations for a set of gameobject entries.
    /// </summary>
    public async Task LoadGameObjectSpawnsAsync(IEnumerable<AssetCatalogEntry> gameObjects)
    {
        var byEntry = gameObjects.Where(g => g.Type == AssetType.GameObject)
            .ToDictionary(g => g.EntryId);
        if (byEntry.Count == 0) return;

        // Columns: spawn_id(0), spawn_entry(1), spawn_map(2),
        //          spawn_positionX(3), spawn_positionY(4), spawn_positionZ(5), spawn_orientation(6), ...
        int count = 0;
        await foreach (var row in ParseInsertRowsAsync(_worldSqlPath, "spawns_gameobjects"))
        {
            if (row.Count < 7) continue;
            int spawnId = ParseInt(row[0]);
            int entry = ParseInt(row[1]);
            int mapId = ParseInt(row[2]);
            float x = ParseFloat(row[3]);
            float y = ParseFloat(row[4]);
            float z = ParseFloat(row[5]);
            float o = ParseFloat(row[6]);

            if (byEntry.TryGetValue(entry, out var go))
            {
                go.Spawns.Add(new SpawnLocation
                {
                    SpawnId = spawnId, MapId = mapId,
                    X = x, Y = y, Z = z, Orientation = o
                });
                count++;
            }
        }
        Console.WriteLine($"[AssetCatalog] Loaded {count} gameobject spawns");
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
        Console.WriteLine($"[AssetCatalog] Total: {all.Count} entries ({withModel} with models, {withSpawns} with spawns)");

        return all;
    }

    public void Dispose() { }

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
