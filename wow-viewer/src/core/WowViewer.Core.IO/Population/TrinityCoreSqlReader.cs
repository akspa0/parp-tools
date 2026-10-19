using System.Globalization;
using System.Numerics;
using System.Text;
using WowViewer.Core.Population;

namespace WowViewer.Core.IO.Population;

/// <summary>
/// High-throughput streaming parser for TrinityCore (and Alpha-Core) world database SQL dumps.
/// Efficiently processes multi-gigabyte or hundreds-of-megabytes .sql files line-by-line
/// without loading entire dumps into memory.
/// </summary>
public sealed class TrinityCoreSqlReader
{
    public sealed record RawCreatureSpawn(
        int Guid,
        int Entry,
        int MapId,
        int ModelIdOverride,
        Vector3 Position,
        float Orientation);

    public sealed record RawCreatureTemplate(
        int Entry,
        string Name,
        string? Subname,
        int[] ModelIds,
        float Scale,
        int Faction,
        int NpcFlags);

    public sealed record RawGameObjectSpawn(
        int Guid,
        int Entry,
        int MapId,
        Vector3 Position,
        float Orientation,
        Quaternion? Rotation);

    public sealed record RawGameObjectTemplate(
        int Entry,
        int DisplayId,
        string Name,
        float Size,
        int Type);

    /// <summary>
    /// Reads creature spawns from a SQL dump file, optionally filtering by target map ID.
    /// </summary>
    public async Task<List<RawCreatureSpawn>> ReadCreatureSpawnsAsync(
        string sqlPath,
        int? targetMapId = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<RawCreatureSpawn>();
        using var stream = File.OpenRead(sqlPath);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 65536);

        string? line;
        bool inCreatureBlock = false;
        bool isAlphaCoreSchema = false;

        while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string trimmed = line.Trim();

            if (trimmed.StartsWith("INSERT INTO `creature`", StringComparison.OrdinalIgnoreCase))
            {
                inCreatureBlock = true;
                isAlphaCoreSchema = false;
            }
            else if (trimmed.StartsWith("INSERT INTO `spawns_creatures`", StringComparison.OrdinalIgnoreCase))
            {
                inCreatureBlock = true;
                isAlphaCoreSchema = true;
            }
            else if (trimmed.StartsWith("INSERT INTO", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("CREATE TABLE", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("LOCK TABLES", StringComparison.OrdinalIgnoreCase))
            {
                if (!trimmed.Contains("`creature`", StringComparison.OrdinalIgnoreCase) &&
                    !trimmed.Contains("`spawns_creatures`", StringComparison.OrdinalIgnoreCase))
                {
                    inCreatureBlock = false;
                }
            }

            if (!inCreatureBlock || !trimmed.Contains('('))
                continue;

            foreach (var tuple in ParseTuples(trimmed))
            {
                if (isAlphaCoreSchema)
                {
                    // spawns_creatures: spawn_id(0), e1(1), e2(2), e3(3), e4(4), map(5), x(6), y(7), z(8), o(9)...
                    if (tuple.Count < 10) continue;
                    int guid = ParseInt(tuple[0]);
                    int entry = ParseInt(tuple[1]);
                    int map = ParseInt(tuple[5]);
                    if (targetMapId.HasValue && map != targetMapId.Value) continue;

                    float x = ParseFloat(tuple[6]);
                    float y = ParseFloat(tuple[7]);
                    float z = ParseFloat(tuple[8]);
                    float o = ParseFloat(tuple[9]);
                    results.Add(new RawCreatureSpawn(guid, entry, map, 0, new Vector3(x, y, z), o));
                }
                else
                {
                    // TrinityCore creature: guid(0), id(1), map(2), zoneId(3), areaId(4), spawnMask(5), phaseMask(6),
                    //                       equipment_id(7), modelid(8), position_x(9), position_y(10), position_z(11), orientation(12)...
                    if (tuple.Count < 13) continue;
                    int guid = ParseInt(tuple[0]);
                    int entry = ParseInt(tuple[1]);
                    int map = ParseInt(tuple[2]);
                    if (targetMapId.HasValue && map != targetMapId.Value) continue;

                    int modelOverride = ParseInt(tuple[8]);
                    float x = ParseFloat(tuple[9]);
                    float y = ParseFloat(tuple[10]);
                    float z = ParseFloat(tuple[11]);
                    float o = ParseFloat(tuple[12]);
                    results.Add(new RawCreatureSpawn(guid, entry, map, modelOverride, new Vector3(x, y, z), o));
                }
            }
        }

        return results;
    }

    /// <summary>
    /// Reads creature templates from a SQL dump file.
    /// </summary>
    public async Task<Dictionary<int, RawCreatureTemplate>> ReadCreatureTemplatesAsync(
        string sqlPath,
        CancellationToken cancellationToken = default)
    {
        var templates = new Dictionary<int, RawCreatureTemplate>();
        using var stream = File.OpenRead(sqlPath);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 65536);

        string? line;
        bool inBlock = false;

        while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string trimmed = line.Trim();

            if (trimmed.StartsWith("INSERT INTO `creature_template`", StringComparison.OrdinalIgnoreCase))
            {
                inBlock = true;
            }
            else if (trimmed.StartsWith("INSERT INTO", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("CREATE TABLE", StringComparison.OrdinalIgnoreCase))
            {
                if (!trimmed.Contains("`creature_template`", StringComparison.OrdinalIgnoreCase))
                    inBlock = false;
            }

            if (!inBlock || !trimmed.Contains('('))
                continue;

            foreach (var tuple in ParseTuples(trimmed))
            {
                if (tuple.Count < 8) continue;
                int entry = ParseInt(tuple[0]);

                // Auto-detect Alpha-Core vs TrinityCore:
                // In Alpha-Core, column 6 is name (string), and columns 1..4 are display IDs.
                // In TrinityCore, column 6 is modelid1 (int), and column 10 is name.
                bool isAlphaCore = tuple.Count >= 7 && !int.TryParse(CleanToken(tuple[6]), NumberStyles.Integer, CultureInfo.InvariantCulture, out _);

                int m1, m2, m3, m4;
                string name;
                string? subname;
                int faction;
                int npcFlags;
                float scale;

                if (isAlphaCore)
                {
                    m1 = ParseInt(tuple[1]);
                    m2 = tuple.Count > 2 ? ParseInt(tuple[2]) : 0;
                    m3 = tuple.Count > 3 ? ParseInt(tuple[3]) : 0;
                    m4 = tuple.Count > 4 ? ParseInt(tuple[4]) : 0;
                    name = tuple[6];
                    subname = tuple.Count > 7 && !string.IsNullOrWhiteSpace(tuple[7]) ? tuple[7] : null;
                    faction = tuple.Count > 12 ? ParseInt(tuple[12]) : 0;
                    npcFlags = tuple.Count > 13 ? ParseInt(tuple[13]) : 0;
                    scale = tuple.Count > 16 ? ParseFloat(tuple[16]) : 1.0f;
                }
                else
                {
                    if (tuple.Count < 12) continue;
                    m1 = ParseInt(tuple[6]);
                    m2 = ParseInt(tuple[7]);
                    m3 = ParseInt(tuple[8]);
                    m4 = ParseInt(tuple[9]);
                    name = tuple[10];
                    subname = string.IsNullOrWhiteSpace(tuple[11]) ? null : tuple[11];
                    faction = tuple.Count > 17 ? ParseInt(tuple[17]) : 0;
                    npcFlags = tuple.Count > 18 ? ParseInt(tuple[18]) : 0;
                    scale = tuple.Count > 21 ? ParseFloat(tuple[21]) : 1.0f;
                }

                if (scale <= 0) scale = 1.0f;

                templates[entry] = new RawCreatureTemplate(
                    entry,
                    name,
                    subname,
                    new[] { m1, m2, m3, m4 },
                    scale,
                    faction,
                    npcFlags);
            }
        }

        return templates;
    }

    /// <summary>
    /// Reads gameobject spawns from a SQL dump file, optionally filtering by target map ID.
    /// </summary>
    public async Task<List<RawGameObjectSpawn>> ReadGameObjectSpawnsAsync(
        string sqlPath,
        int? targetMapId = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<RawGameObjectSpawn>();
        using var stream = File.OpenRead(sqlPath);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 65536);

        string? line;
        bool inBlock = false;
        bool isAlphaCore = false;

        while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string trimmed = line.Trim();

            if (trimmed.StartsWith("INSERT INTO `gameobject`", StringComparison.OrdinalIgnoreCase))
            {
                inBlock = true;
                isAlphaCore = false;
            }
            else if (trimmed.StartsWith("INSERT INTO `spawns_gameobjects`", StringComparison.OrdinalIgnoreCase))
            {
                inBlock = true;
                isAlphaCore = true;
            }
            else if (trimmed.StartsWith("INSERT INTO", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("CREATE TABLE", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("LOCK TABLES", StringComparison.OrdinalIgnoreCase))
            {
                if (!trimmed.Contains("`gameobject`", StringComparison.OrdinalIgnoreCase) &&
                    !trimmed.Contains("`spawns_gameobjects`", StringComparison.OrdinalIgnoreCase))
                {
                    inBlock = false;
                }
            }

            if (!inBlock || !trimmed.Contains('('))
                continue;

            foreach (var tuple in ParseTuples(trimmed))
            {
                if (isAlphaCore)
                {
                    // spawns_gameobjects: guid(0), entry(1), map(2), x(3), y(4), z(5), o(6)...
                    if (tuple.Count < 7) continue;
                    int guid = ParseInt(tuple[0]);
                    int entry = ParseInt(tuple[1]);
                    int map = ParseInt(tuple[2]);
                    if (targetMapId.HasValue && map != targetMapId.Value) continue;

                    float x = ParseFloat(tuple[3]);
                    float y = ParseFloat(tuple[4]);
                    float z = ParseFloat(tuple[5]);
                    float o = ParseFloat(tuple[6]);
                    results.Add(new RawGameObjectSpawn(guid, entry, map, new Vector3(x, y, z), o, null));
                }
                else
                {
                    // TrinityCore: guid(0), id(1), map(2), zoneId(3), areaId(4), spawnMask(5), phaseMask(6),
                    //              position_x(7), position_y(8), position_z(9), orientation(10),
                    //              rotation0(11), rotation1(12), rotation2(13), rotation3(14)...
                    if (tuple.Count < 11) continue;
                    int guid = ParseInt(tuple[0]);
                    int entry = ParseInt(tuple[1]);
                    int map = ParseInt(tuple[2]);
                    if (targetMapId.HasValue && map != targetMapId.Value) continue;

                    float x = ParseFloat(tuple[7]);
                    float y = ParseFloat(tuple[8]);
                    float z = ParseFloat(tuple[9]);
                    float o = ParseFloat(tuple[10]);

                    Quaternion? rot = null;
                    if (tuple.Count >= 15)
                    {
                        float r0 = ParseFloat(tuple[11]);
                        float r1 = ParseFloat(tuple[12]);
                        float r2 = ParseFloat(tuple[13]);
                        float r3 = ParseFloat(tuple[14]);
                        rot = new Quaternion(r0, r1, r2, r3);
                    }

                    results.Add(new RawGameObjectSpawn(guid, entry, map, new Vector3(x, y, z), o, rot));
                }
            }
        }

        return results;
    }

    /// <summary>
    /// Reads gameobject templates from a SQL dump file.
    /// </summary>
    public async Task<Dictionary<int, RawGameObjectTemplate>> ReadGameObjectTemplatesAsync(
        string sqlPath,
        CancellationToken cancellationToken = default)
    {
        var templates = new Dictionary<int, RawGameObjectTemplate>();
        using var stream = File.OpenRead(sqlPath);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 65536);

        string? line;
        bool inBlock = false;

        while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string trimmed = line.Trim();

            if (trimmed.StartsWith("INSERT INTO `gameobject_template`", StringComparison.OrdinalIgnoreCase))
            {
                inBlock = true;
            }
            else if (trimmed.StartsWith("INSERT INTO", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("CREATE TABLE", StringComparison.OrdinalIgnoreCase))
            {
                if (!trimmed.Contains("`gameobject_template`", StringComparison.OrdinalIgnoreCase))
                    inBlock = false;
            }

            if (!inBlock || !trimmed.Contains('('))
                continue;

            foreach (var tuple in ParseTuples(trimmed))
            {
                // TrinityCore: entry(0), type(1), displayId(2), name(3), IconName(4), castBarCaption(5), unk1(6), size(7), data0..23...
                // Alpha-Core:  entry(0), type(1), displayId(2), name(3), faction(4), flags(5), size(6), data0..9...
                if (tuple.Count < 4) continue;
                int entry = ParseInt(tuple[0]);
                int type = ParseInt(tuple[1]);
                int displayId = ParseInt(tuple[2]);
                string name = tuple[3];
                bool isTrinityCore = tuple.Count >= 25 || (tuple.Count > 4 && !int.TryParse(CleanToken(tuple[4]), NumberStyles.Integer, CultureInfo.InvariantCulture, out _));
                float size = isTrinityCore
                    ? (tuple.Count > 7 ? ParseFloat(tuple[7]) : 1.0f)
                    : (tuple.Count > 6 ? ParseFloat(tuple[6]) : 1.0f);
                if (size <= 0) size = 1.0f;

                templates[entry] = new RawGameObjectTemplate(entry, displayId, name, size, type);
            }
        }

        return templates;
    }

    /// <summary>
    /// Builds full PopulationRecords for a specified map, combining spawns with resolved templates and DBC paths.
    /// </summary>
    public async Task<List<PopulationRecord>> LoadMapPopulationAsync(
        string sqlPath,
        int mapId,
        IDbcModelResolver? dbcResolver = null,
        CancellationToken cancellationToken = default)
    {
        var creatures = await ReadCreatureSpawnsAsync(sqlPath, mapId, cancellationToken);
        var gameObjects = await ReadGameObjectSpawnsAsync(sqlPath, mapId, cancellationToken);

        var creatureTemplates = await ReadCreatureTemplatesAsync(sqlPath, cancellationToken);
        var goTemplates = await ReadGameObjectTemplatesAsync(sqlPath, cancellationToken);

        var records = new List<PopulationRecord>(creatures.Count + gameObjects.Count);

        // Map creatures
        foreach (var c in creatures)
        {
            creatureTemplates.TryGetValue(c.Entry, out var tmpl);
            string name = tmpl?.Name ?? $"Creature #{c.Entry}";
            string? subname = tmpl?.Subname;
            float scale = tmpl?.Scale ?? 1.0f;
            int faction = tmpl?.Faction ?? 0;
            int flags = tmpl?.NpcFlags ?? 0;

            // Display ID resolution
            int displayId = c.ModelIdOverride > 0 ? c.ModelIdOverride : 0;
            if (displayId == 0 && tmpl?.ModelIds != null)
            {
                foreach (int m in tmpl.ModelIds)
                {
                    if (m > 0)
                    {
                        displayId = m;
                        break;
                    }
                }
            }

            string? modelPath = null;
            float displayScale = 1.0f;
            if (displayId > 0 && dbcResolver != null)
            {
                dbcResolver.TryResolveCreature(displayId, out modelPath, out displayScale);
            }

            records.Add(new PopulationRecord
            {
                SpawnId = c.Guid,
                EntryId = c.Entry,
                MapId = c.MapId,
                SpawnType = PopulationSpawnType.Creature,
                Name = name,
                Subname = subname,
                ModelPath = modelPath,
                Scale = scale,
                DisplayScale = displayScale,
                EffectiveScale = scale * displayScale,
                Position = c.Position,
                Orientation = c.Orientation,
                Rotation = null,
                Faction = faction,
                Flags = flags,
                GameObjectType = 0,
                IsTransposed = false,
                SourceNote = null
            });
        }

        // Map gameobjects
        foreach (var g in gameObjects)
        {
            goTemplates.TryGetValue(g.Entry, out var tmpl);
            string name = tmpl?.Name ?? $"GameObject #{g.Entry}";
            float scale = tmpl?.Size ?? 1.0f;
            int displayId = tmpl?.DisplayId ?? 0;
            int goType = tmpl?.Type ?? 0;

            string? modelPath = null;
            float displayScale = 1.0f;
            if (displayId > 0 && dbcResolver != null)
            {
                dbcResolver.TryResolveGameObject(displayId, out modelPath, out displayScale);
            }

            records.Add(new PopulationRecord
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
            });
        }

        return records;
    }

    public sealed record AllMapsExtractionResult(
        Dictionary<int, List<PopulationRecord>> SpawnsByMap,
        int TotalCreatureTemplates,
        int TotalGameObjectTemplates);

    /// <summary>
    /// Performs a high-throughput single-pass streaming extraction of all maps in a TrinityCore or Alpha-Core SQL file.
    /// </summary>
    public async Task<AllMapsExtractionResult> ExtractAllMapsAsync(
        string sqlPath,
        IDbcModelResolver? dbcResolver = null,
        ISet<int>? targetMaps = null,
        CancellationToken cancellationToken = default)
    {
        var creatureTemplates = new Dictionary<int, RawCreatureTemplate>();
        var goTemplates = new Dictionary<int, RawGameObjectTemplate>();
        var creatureSpawnsByMap = new Dictionary<int, List<RawCreatureSpawn>>();
        var goSpawnsByMap = new Dictionary<int, List<RawGameObjectSpawn>>();

        using var stream = File.OpenRead(sqlPath);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 131072);

        string? line;
        int activeTableType = 0; // 0: None, 1: creature, 2: spawns_creatures, 3: creature_template, 4: gameobject, 5: spawns_gameobjects, 6: gameobject_template

        while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string trimmed = line.Trim();

            if (trimmed.StartsWith("INSERT INTO `creature_template`", StringComparison.OrdinalIgnoreCase))
            {
                activeTableType = 3;
            }
            else if (trimmed.StartsWith("INSERT INTO `creature`", StringComparison.OrdinalIgnoreCase))
            {
                activeTableType = 1;
            }
            else if (trimmed.StartsWith("INSERT INTO `spawns_creatures`", StringComparison.OrdinalIgnoreCase))
            {
                activeTableType = 2;
            }
            else if (trimmed.StartsWith("INSERT INTO `gameobject_template`", StringComparison.OrdinalIgnoreCase))
            {
                activeTableType = 6;
            }
            else if (trimmed.StartsWith("INSERT INTO `gameobject`", StringComparison.OrdinalIgnoreCase))
            {
                activeTableType = 4;
            }
            else if (trimmed.StartsWith("INSERT INTO `spawns_gameobjects`", StringComparison.OrdinalIgnoreCase))
            {
                activeTableType = 5;
            }
            else if (trimmed.StartsWith("INSERT INTO", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("CREATE TABLE", StringComparison.OrdinalIgnoreCase) ||
                     trimmed.StartsWith("LOCK TABLES", StringComparison.OrdinalIgnoreCase))
            {
                activeTableType = 0;
            }

            if (activeTableType == 0 || !trimmed.Contains('('))
                continue;

            var tuples = ParseTuples(trimmed);
            foreach (var tuple in tuples)
            {
                switch (activeTableType)
                {
                    case 1: // TrinityCore creature
                        if (tuple.Count >= 13)
                        {
                            int guid = ParseInt(tuple[0]);
                            int entry = ParseInt(tuple[1]);
                            int map = ParseInt(tuple[2]);
                            if (targetMaps != null && targetMaps.Count > 0 && !targetMaps.Contains(map))
                                continue;
                            int modelOverride = ParseInt(tuple[8]);
                            float x = ParseFloat(tuple[9]);
                            float y = ParseFloat(tuple[10]);
                            float z = ParseFloat(tuple[11]);
                            float o = ParseFloat(tuple[12]);
                            if (!creatureSpawnsByMap.TryGetValue(map, out var list))
                            {
                                list = new List<RawCreatureSpawn>();
                                creatureSpawnsByMap[map] = list;
                            }
                            list.Add(new RawCreatureSpawn(guid, entry, map, modelOverride, new Vector3(x, y, z), o));
                        }
                        break;

                    case 2: // Alpha-Core spawns_creatures
                        if (tuple.Count >= 10)
                        {
                            int guid = ParseInt(tuple[0]);
                            int entry = ParseInt(tuple[1]);
                            int map = ParseInt(tuple[5]);
                            if (targetMaps != null && targetMaps.Count > 0 && !targetMaps.Contains(map))
                                continue;
                            float x = ParseFloat(tuple[6]);
                            float y = ParseFloat(tuple[7]);
                            float z = ParseFloat(tuple[8]);
                            float o = ParseFloat(tuple[9]);
                            if (!creatureSpawnsByMap.TryGetValue(map, out var list))
                            {
                                list = new List<RawCreatureSpawn>();
                                creatureSpawnsByMap[map] = list;
                            }
                            list.Add(new RawCreatureSpawn(guid, entry, map, 0, new Vector3(x, y, z), o));
                        }
                        break;

                    case 3: // creature_template
                        if (tuple.Count >= 8)
                        {
                            int entry = ParseInt(tuple[0]);
                            bool isAlpha = tuple.Count >= 7 && !int.TryParse(CleanToken(tuple[6]), NumberStyles.Integer, CultureInfo.InvariantCulture, out _);
                            int m1, m2, m3, m4;
                            string name;
                            string? subname;
                            int faction;
                            int npcFlags;
                            float scale;

                            if (isAlpha)
                            {
                                m1 = ParseInt(tuple[1]);
                                m2 = tuple.Count > 2 ? ParseInt(tuple[2]) : 0;
                                m3 = tuple.Count > 3 ? ParseInt(tuple[3]) : 0;
                                m4 = tuple.Count > 4 ? ParseInt(tuple[4]) : 0;
                                name = CleanToken(tuple[6]);
                                subname = tuple.Count > 7 ? CleanToken(tuple[7]) : null;
                                faction = tuple.Count > 12 ? ParseInt(tuple[12]) : 0;
                                npcFlags = tuple.Count > 13 ? ParseInt(tuple[13]) : 0;
                                scale = tuple.Count > 16 ? ParseFloat(tuple[16]) : 1.0f;
                            }
                            else
                            {
                                m1 = ParseInt(tuple[6]);
                                m2 = tuple.Count > 7 ? ParseInt(tuple[7]) : 0;
                                m3 = tuple.Count > 8 ? ParseInt(tuple[8]) : 0;
                                m4 = tuple.Count > 9 ? ParseInt(tuple[9]) : 0;
                                name = tuple.Count > 10 ? CleanToken(tuple[10]) : $"Creature #{entry}";
                                subname = tuple.Count > 11 ? CleanToken(tuple[11]) : null;
                                faction = tuple.Count > 17 ? ParseInt(tuple[17]) : 0;
                                npcFlags = tuple.Count > 18 ? ParseInt(tuple[18]) : 0;
                                scale = tuple.Count > 21 ? ParseFloat(tuple[21]) : 1.0f;
                            }

                            if (scale <= 0f) scale = 1.0f;
                            if (string.IsNullOrWhiteSpace(subname)) subname = null;
                            creatureTemplates[entry] = new RawCreatureTemplate(entry, name, subname, new[] { m1, m2, m3, m4 }, scale, faction, npcFlags);
                        }
                        break;

                    case 4: // TrinityCore gameobject
                        if (tuple.Count >= 11)
                        {
                            int guid = ParseInt(tuple[0]);
                            int entry = ParseInt(tuple[1]);
                            int map = ParseInt(tuple[2]);
                            if (targetMaps != null && targetMaps.Count > 0 && !targetMaps.Contains(map))
                                continue;
                            float x = ParseFloat(tuple[7]);
                            float y = ParseFloat(tuple[8]);
                            float z = ParseFloat(tuple[9]);
                            float o = ParseFloat(tuple[10]);
                            Quaternion? rot = null;
                            if (tuple.Count >= 15)
                            {
                                float r0 = ParseFloat(tuple[11]);
                                float r1 = ParseFloat(tuple[12]);
                                float r2 = ParseFloat(tuple[13]);
                                float r3 = ParseFloat(tuple[14]);
                                if (r0 != 0 || r1 != 0 || r2 != 0 || r3 != 0)
                                    rot = new Quaternion(r0, r1, r2, r3);
                            }
                            if (!goSpawnsByMap.TryGetValue(map, out var list))
                            {
                                list = new List<RawGameObjectSpawn>();
                                goSpawnsByMap[map] = list;
                            }
                            list.Add(new RawGameObjectSpawn(guid, entry, map, new Vector3(x, y, z), o, rot));
                        }
                        break;

                    case 5: // Alpha-Core spawns_gameobjects
                        if (tuple.Count >= 7)
                        {
                            int guid = ParseInt(tuple[0]);
                            int entry = ParseInt(tuple[1]);
                            int map = ParseInt(tuple[2]);
                            if (targetMaps != null && targetMaps.Count > 0 && !targetMaps.Contains(map))
                                continue;
                            float x = ParseFloat(tuple[3]);
                            float y = ParseFloat(tuple[4]);
                            float z = ParseFloat(tuple[5]);
                            float o = ParseFloat(tuple[6]);
                            if (!goSpawnsByMap.TryGetValue(map, out var list))
                            {
                                list = new List<RawGameObjectSpawn>();
                                goSpawnsByMap[map] = list;
                            }
                            list.Add(new RawGameObjectSpawn(guid, entry, map, new Vector3(x, y, z), o, null));
                        }
                        break;

                    case 6: // gameobject_template
                        if (tuple.Count >= 4)
                        {
                            int entry = ParseInt(tuple[0]);
                            int type = ParseInt(tuple[1]);
                            int displayId = ParseInt(tuple[2]);
                            string name = CleanToken(tuple[3]);
                            bool isTrinityCore = tuple.Count >= 25 || (tuple.Count > 4 && !int.TryParse(CleanToken(tuple[4]), NumberStyles.Integer, CultureInfo.InvariantCulture, out _));
                            float size = isTrinityCore
                                ? (tuple.Count > 7 ? ParseFloat(tuple[7]) : 1.0f)
                                : (tuple.Count > 6 ? ParseFloat(tuple[6]) : 1.0f);
                            if (size <= 0f) size = 1.0f;
                            goTemplates[entry] = new RawGameObjectTemplate(entry, displayId, name, size, type);
                        }
                        break;
                }
            }
        }

        var allMapIds = new HashSet<int>(creatureSpawnsByMap.Keys);
        allMapIds.UnionWith(goSpawnsByMap.Keys);

        var spawnsByMap = new Dictionary<int, List<PopulationRecord>>();
        foreach (int mapId in allMapIds)
        {
            if (targetMaps != null && targetMaps.Count > 0 && !targetMaps.Contains(mapId))
                continue;

            var records = new List<PopulationRecord>();

            if (creatureSpawnsByMap.TryGetValue(mapId, out var creatures))
            {
                foreach (var c in creatures)
                {
                    creatureTemplates.TryGetValue(c.Entry, out var tmpl);
                    string name = tmpl?.Name ?? $"Creature #{c.Entry}";
                    string? subname = tmpl?.Subname;
                    float scale = tmpl?.Scale ?? 1.0f;
                    int faction = tmpl?.Faction ?? 0;
                    int flags = tmpl?.NpcFlags ?? 0;

                    int displayId = c.ModelIdOverride > 0 ? c.ModelIdOverride : 0;
                    if (displayId == 0 && tmpl?.ModelIds != null)
                    {
                        foreach (int m in tmpl.ModelIds)
                        {
                            if (m > 0)
                            {
                                displayId = m;
                                break;
                            }
                        }
                    }

                    string? modelPath = null;
                    float displayScale = 1.0f;
                    if (displayId > 0 && dbcResolver != null)
                        dbcResolver.TryResolveCreature(displayId, out modelPath, out displayScale);

                    records.Add(new PopulationRecord
                    {
                        SpawnId = c.Guid,
                        EntryId = c.Entry,
                        MapId = c.MapId,
                        SpawnType = PopulationSpawnType.Creature,
                        Name = name,
                        Subname = subname,
                        ModelPath = modelPath,
                        Scale = scale,
                        DisplayScale = displayScale,
                        EffectiveScale = scale * displayScale,
                        Position = c.Position,
                        Orientation = c.Orientation,
                        Rotation = null,
                        Faction = faction,
                        Flags = flags,
                        GameObjectType = 0,
                        IsTransposed = false,
                        SourceNote = null
                    });
                }
            }

            if (goSpawnsByMap.TryGetValue(mapId, out var gameObjects))
            {
                foreach (var g in gameObjects)
                {
                    goTemplates.TryGetValue(g.Entry, out var tmpl);
                    string name = tmpl?.Name ?? $"GameObject #{g.Entry}";
                    float scale = tmpl?.Size ?? 1.0f;
                    int displayId = tmpl?.DisplayId ?? 0;
                    int goType = tmpl?.Type ?? 0;

                    string? modelPath = null;
                    float displayScale = 1.0f;
                    if (displayId > 0 && dbcResolver != null)
                        dbcResolver.TryResolveGameObject(displayId, out modelPath, out displayScale);

                    records.Add(new PopulationRecord
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
                    });
                }
            }

            spawnsByMap[mapId] = records;
        }

        return new AllMapsExtractionResult(spawnsByMap, creatureTemplates.Count, goTemplates.Count);
    }

    /// <summary>
    /// Parses SQL row value tuples '(...)' from a line, handling string literals, escapes, and commas.
    /// </summary>
    public static List<List<string>> ParseTuples(string line)
    {
        var tuples = new List<List<string>>();
        int len = line.Length;
        int i = 0;

        while (i < len)
        {
            // Find start of tuple
            while (i < len && line[i] != '(') i++;
            if (i >= len) break;
            i++; // skip '('

            var fields = new List<string>();
            var currentField = new StringBuilder();
            bool inString = false;

            while (i < len)
            {
                char c = line[i];

                if (inString)
                {
                    if (c == '\\' && i + 1 < len)
                    {
                        // SQL escape: \' or \\ etc.
                        i++;
                        currentField.Append(line[i]);
                    }
                    else if (c == '\'')
                    {
                        if (i + 1 < len && line[i + 1] == '\'')
                        {
                            // Doubled quote '' escape
                            currentField.Append('\'');
                            i++;
                        }
                        else
                        {
                            inString = false;
                        }
                    }
                    else
                    {
                        currentField.Append(c);
                    }
                }
                else
                {
                    if (c == '\'')
                    {
                        inString = true;
                    }
                    else if (c == ',')
                    {
                        fields.Add(currentField.ToString().Trim());
                        currentField.Clear();
                    }
                    else if (c == ')')
                    {
                        fields.Add(currentField.ToString().Trim());
                        currentField.Clear();
                        i++;
                        break;
                    }
                    else
                    {
                        currentField.Append(c);
                    }
                }

                i++;
            }

            if (fields.Count > 0)
                tuples.Add(fields);
        }

        return tuples;
    }

    private static int ParseInt(string str)
    {
        string cleaned = CleanToken(str);
        return int.TryParse(cleaned, NumberStyles.Integer, CultureInfo.InvariantCulture, out int val) ? val : 0;
    }

    private static float ParseFloat(string str)
    {
        string cleaned = CleanToken(str);
        return float.TryParse(cleaned, NumberStyles.Float, CultureInfo.InvariantCulture, out float val) ? val : 0.0f;
    }

    private static string CleanToken(string str)
    {
        string trimmed = str.Trim();
        if (trimmed.StartsWith('\'') && trimmed.EndsWith('\'') && trimmed.Length >= 2)
            trimmed = trimmed[1..^1];
        return trimmed;
    }
}
