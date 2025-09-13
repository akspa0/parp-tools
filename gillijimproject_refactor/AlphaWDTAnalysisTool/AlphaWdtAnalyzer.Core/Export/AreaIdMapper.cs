using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Text.Json;

#if USE_DBCD
using DBCD;
using DBCD.Providers;
#endif

namespace AlphaWdtAnalyzer.Core.Export;

public sealed class AreaIdMapper
{
    private readonly Dictionary<int, string> _alphaIdToName;
    private readonly Dictionary<string, int> _lkNameToId;
    private readonly Dictionary<int, string> _lkIdToName;
    private readonly HashSet<int> _lkIds;
    private readonly int? _lkFallbackOnMapDungeonId;
    // Remap support
    private readonly Dictionary<int, int> _explicitMap = new();
    // Map-aware explicit mapping: key = "{mapId}:{areaNumber}"
    private readonly Dictionary<string, int> _explicitMapByMap = new(StringComparer.Ordinal);
    private readonly HashSet<int> _ignoreAlphaIds = new();
    private readonly Dictionary<string, string[]> _aliases = new(StringComparer.OrdinalIgnoreCase);
    private readonly bool _disallowDoNotUseTargets = true;

    public int AlphaNameColumnIndex { get; }
    public int LkNameColumnIndex { get; }

    private AreaIdMapper(
        Dictionary<int, string> alphaIdToName,
        Dictionary<string, int> lkNameToId,
        Dictionary<int, string> lkIdToName,
        int alphaNameCol,
        int lkNameCol,
        Dictionary<int, int>? explicitMap = null,
        HashSet<int>? ignoreAlphaIds = null,
        Dictionary<string, string[]>? aliases = null,
        bool disallowDoNotUseTargets = true)
    {
        _alphaIdToName = alphaIdToName;
        _lkNameToId = lkNameToId;
        _lkIdToName = lkIdToName;
        _lkIds = _lkIdToName.Keys.ToHashSet();
        AlphaNameColumnIndex = alphaNameCol;
        LkNameColumnIndex = lkNameCol;
        _lkFallbackOnMapDungeonId = ResolveOnMapDungeonId(lkNameToId);
        if (explicitMap is not null)
        {
            foreach (var kv in explicitMap) _explicitMap[kv.Key] = kv.Value;
        }
        if (ignoreAlphaIds is not null)
        {
            foreach (var id in ignoreAlphaIds) _ignoreAlphaIds.Add(id);
        }
        if (aliases is not null)
        {
            foreach (var kv in aliases) _aliases[kv.Key] = kv.Value;
        }
        _disallowDoNotUseTargets = disallowDoNotUseTargets;
    }

    // Deprecated signature retained for compatibility; returns null to force DBCD path
    public static AreaIdMapper? TryCreate(string? alphaDbcPath, string? lkDbcPath)
    {
        return null; // RawDBC is removed; use the DBCD overload below
    }

    public static AreaIdMapper? TryCreate(string? alphaDbcPath, string? lkDbcPath, string? dbdDefinitionsDir)
    {
        if (string.IsNullOrWhiteSpace(alphaDbcPath) || string.IsNullOrWhiteSpace(lkDbcPath)) return null;
        if (!File.Exists(alphaDbcPath) || !File.Exists(lkDbcPath)) return null;
        if (string.IsNullOrWhiteSpace(dbdDefinitionsDir) || !Directory.Exists(dbdDefinitionsDir)) return null;

#if USE_DBCD
        try
        {
            var alphaIdToName = LoadIdToNameDbcd(alphaDbcPath!, dbdDefinitionsDir!, TryGetAlphaBuilds());
            var lkIdToName = LoadIdToNameDbcd(lkDbcPath!, dbdDefinitionsDir!, new[] { "3.3.5.12340" });

            // Build LK name->id map using normalized names
            var lkNameToId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in lkIdToName)
            {
                var norm = NormalizeName(kv.Value);
                if (!lkNameToId.ContainsKey(norm)) lkNameToId[norm] = kv.Key;
            }

            // AlphaNameColumnIndex/LkNameColumnIndex are not meaningful with DBCD; set to 0
            return new AreaIdMapper(alphaIdToName, lkNameToId, lkIdToName, 0, 0);
        }
        catch
        {
            return null;
        }
#else
        // DBCD not available in this build
        return null;
#endif
    }

    // New overload that can operate with remap-only if DBCD is unavailable.
    public static AreaIdMapper? TryCreate(string? alphaDbcPath, string? lkDbcPath, string? dbdDefinitionsDir, string? remapPath)
    {
        // Load DBCD name dictionaries when available
        Dictionary<int, string> alphaIdToName = new();
        Dictionary<int, string> lkIdToName = new();
        Dictionary<string, int> lkNameToId = new(StringComparer.OrdinalIgnoreCase);

#if USE_DBCD
        bool hasDbcd = !string.IsNullOrWhiteSpace(alphaDbcPath)
                       && !string.IsNullOrWhiteSpace(lkDbcPath)
                       && !string.IsNullOrWhiteSpace(dbdDefinitionsDir)
                       && File.Exists(alphaDbcPath!)
                       && File.Exists(lkDbcPath!)
                       && Directory.Exists(dbdDefinitionsDir!);
        if (hasDbcd)
        {
            try
            {
                alphaIdToName = LoadIdToNameDbcd(alphaDbcPath!, dbdDefinitionsDir!, TryGetAlphaBuilds());
                lkIdToName = LoadIdToNameDbcd(lkDbcPath!, dbdDefinitionsDir!, new[] { "3.3.5.12340" });
                foreach (var kv in lkIdToName)
                {
                    var norm = NormalizeName(kv.Value);
                    if (!lkNameToId.ContainsKey(norm)) lkNameToId[norm] = kv.Key;
                }
            }
            catch
            {
                // fall back to remap-only if provided
            }
        }
#endif

        // If neither DBCD names nor remap are available, bail out
        bool hasRemap = !string.IsNullOrWhiteSpace(remapPath) && File.Exists(remapPath!);
        if (!hasRemap && alphaIdToName.Count == 0 && lkIdToName.Count == 0)
        {
            return null;
        }

        // Parse remap if present
        var (explicitMap, ignore, aliases, disallow, explicitByMap) = ParseRemapOrDefaults(remapPath);

        return new AreaIdMapper(
            alphaIdToName,
            lkNameToId,
            lkIdToName,
            0,
            0,
            explicitMap,
            ignore,
            aliases,
            disallow);
    }

    public bool TryResolveById(int alphaAreaId, out int lkAreaId, out string reason)
    {
        if (_lkIds.Contains(alphaAreaId))
        {
            lkAreaId = alphaAreaId;
            reason = "direct";
            return true;
        }
        if (_lkFallbackOnMapDungeonId.HasValue)
        {
            lkAreaId = _lkFallbackOnMapDungeonId.Value;
            reason = "fallback_on_map_dungeon";
            return true;
        }
        lkAreaId = 0;
        reason = "unmapped";
        return false;
    }

    public bool TryMap(int alphaAreaId, out int lkAreaId, out string? alphaName, out string? lkName)
    {
        return TryMapDetailed(alphaAreaId, out lkAreaId, out alphaName, out lkName, out var _);
    }

    // Detailed mapping that returns a reason string for diagnostics/CSV
    public bool TryMapDetailed(int alphaAreaId, out int lkAreaId, out string? alphaName, out string? lkName, out string reason)
    {
        return TryMapDetailed(alphaAreaId, currentMapId: null, out lkAreaId, out alphaName, out lkName, out reason);
    }

    // Map-aware overload: prefer explicit mappings that match the current map
    public bool TryMapDetailed(int alphaAreaId, int? currentMapId, out int lkAreaId, out string? alphaName, out string? lkName, out string reason)
    {
        lkAreaId = 0;
        alphaName = null;
        lkName = null;
        reason = "unmapped";

        // Ignore list short-circuit
        if (_ignoreAlphaIds.Contains(alphaAreaId))
        {
            reason = "ignored";
            return false;
        }

        // Prefer (mapId, areaNumber) exact match when available
        if (currentMapId.HasValue)
        {
            var key = currentMapId.Value.ToString() + ":" + alphaAreaId.ToString();
            if (_explicitMapByMap.TryGetValue(key, out var mapped))
            {
                lkAreaId = mapped;
                _alphaIdToName.TryGetValue(alphaAreaId, out alphaName);
                lkName = GetLkNameById(lkAreaId);
                reason = "remap_explicit";
                return true;
            }
        }

        // Explicit map wins
        if (_explicitMap.TryGetValue(alphaAreaId, out var explicitTarget))
        {
            lkAreaId = explicitTarget;
            _alphaIdToName.TryGetValue(alphaAreaId, out alphaName);
            lkName = GetLkNameById(lkAreaId);
            reason = "remap_explicit";
            return true;
        }

        // Name-based when names are available
        if (_alphaIdToName.TryGetValue(alphaAreaId, out var aName) && !string.IsNullOrWhiteSpace(aName))
        {
            alphaName = aName;
            var norm = NormalizeName(aName);
            if (TryResolveLkByName(norm, out lkAreaId))
            {
                lkName = GetLkNameById(lkAreaId) ?? aName;
                if (_disallowDoNotUseTargets && ContainsDoNotUse(lkName))
                {
                    // treat as unmapped if disallowed
                    lkAreaId = 0;
                    lkName = null;
                    reason = "disallowed_target";
                    return false;
                }
                reason = "name";
                return true;
            }

            // Try alias variants
            if (_aliases.TryGetValue(norm, out var variants))
            {
                foreach (var v in variants)
                {
                    var vn = NormalizeName(v);
                    if (TryResolveLkByName(vn, out lkAreaId))
                    {
                        lkName = GetLkNameById(lkAreaId) ?? v;
                        if (_disallowDoNotUseTargets && ContainsDoNotUse(lkName))
                        {
                            lkAreaId = 0; lkName = null; reason = "disallowed_target"; return false;
                        }
                        reason = "name_alias";
                        return true;
                    }
                }
            }
        }

        return false;
    }

#if USE_DBCD
    private static Dictionary<int, string> LoadIdToNameDbcd(string dbcPath, string dbdDefinitionsDir, IEnumerable<string> buildCandidates)
    {
        var result = new Dictionary<int, string>();

        var dbcDir = Path.GetDirectoryName(dbcPath)!;
        var tableName = Path.GetFileNameWithoutExtension(dbcPath)!;

        var dbcProvider = new FilesystemDBCProvider(dbcDir);
        // DBCD expects the "definitions" subfolder
        var defDir = Directory.Exists(Path.Combine(dbdDefinitionsDir, "definitions"))
            ? Path.Combine(dbdDefinitionsDir, "definitions")
            : dbdDefinitionsDir;
        var dbdProvider = new FilesystemDBDProvider(defDir);

        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        Exception? last = null;
        foreach (var build in buildCandidates)
        {
            try
            {
                var storage = dbcd.Load(tableName, build);
                foreach (KeyValuePair<int, DBCDRow> entry in (IEnumerable<KeyValuePair<int, DBCDRow>>)storage)
                {
                    int id = entry.Key;
                    DBCDRow row = entry.Value;
                    string? name = TryGetName(row);
                    if (!string.IsNullOrWhiteSpace(name))
                    {
                        result[id] = name!;
                    }
                }
                return result;
            }
            catch (Exception ex)
            {
                last = ex;
                continue;
            }
        }
        throw last ?? new InvalidOperationException("Failed to load DBC via DBCD");
    }

    private static string? TryGetName(DBCDRow row)
    {
        // Try common field names across eras
        string[] fields = new[] { "AreaName_lang", "Name_lang", "AreaName_Lang", "Name_Lang", "AreaName", "Name" };
        foreach (var f in fields)
        {
            try
            {
                object val = row[f];
                if (val is string s && !string.IsNullOrWhiteSpace(s)) return s;
            }
            catch { /* field may not exist; continue */ }
        }
        return null;
    }

    private static IEnumerable<string> TryGetAlphaBuilds()
    {
        // Reasonable 0.5.x candidates; will try in order until one matches available DBD
        yield return "0.5.5.3494";
        yield return "0.5.3.3368";
    }
#endif

    private static int? ResolveOnMapDungeonId(Dictionary<string, int> lkNameToId)
    {
        var key = NormalizeName("On Map Dungeon");
        if (lkNameToId.TryGetValue(key, out var id)) return id;
        var candidate = lkNameToId.FirstOrDefault(kv => kv.Key.Contains("on map dungeon", StringComparison.OrdinalIgnoreCase));
        if (!candidate.Equals(default(KeyValuePair<string,int>))) return candidate.Value;
        return null;
    }

    public string? GetAlphaName(int id) => _alphaIdToName.TryGetValue(id, out var n) ? n : null;
    public string? GetLkNameById(int id) => _lkIdToName.TryGetValue(id, out var n) ? n : null;

    private static string NormalizeName(string s)
    {
        var norm = s.Trim();
        norm = Regex.Replace(norm, "[\'\"]", "");
        norm = Regex.Replace(norm, "\\s+", " ");
        return norm.ToLowerInvariant();
    }

    private static (Dictionary<int,int> explicitMap, HashSet<int> ignore, Dictionary<string,string[]> aliases, bool disallow, Dictionary<string,int> explicitMapByMap) ParseRemapOrDefaults(string? remapPath)
    {
        var explicitMap = new Dictionary<int, int>();
        var ignore = new HashSet<int>();
        var aliases = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
        bool disallow = true;
        var explicitByMap = new Dictionary<string, int>(StringComparer.Ordinal);

        if (string.IsNullOrWhiteSpace(remapPath) || !File.Exists(remapPath!))
        {
            return (explicitMap, ignore, aliases, disallow, explicitByMap);
        }

        try
        {
            var json = File.ReadAllText(remapPath!);
            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var doc = JsonSerializer.Deserialize<RemapRoot>(json, options);
            if (doc is null) return (explicitMap, ignore, aliases, disallow, explicitByMap);

            if (doc.aliases is not null)
            {
                foreach (var kv in doc.aliases)
                {
                    if (kv.Value is null) continue;
                    var key = NormalizeName(kv.Key);
                    var vals = kv.Value.Where(v => !string.IsNullOrWhiteSpace(v)).ToArray();
                    aliases[key] = vals;
                }
            }
            if (doc.explicit_map is not null)
            {
                foreach (var e in doc.explicit_map)
                {
                    explicitMap[e.src_areaNumber] = e.tgt_areaID;
                    if (e.src_mapID.HasValue)
                    {
                        var key = e.src_mapID.Value.ToString() + ":" + e.src_areaNumber.ToString();
                        explicitByMap[key] = e.tgt_areaID;
                    }
                }
            }
            if (doc.ignore_area_numbers is not null)
            {
                foreach (var n in doc.ignore_area_numbers) ignore.Add(n);
            }
            if (doc.options is not null)
            {
                disallow = doc.options.disallow_do_not_use_targets;
            }
        }
        catch
        {
            // ignore parse errors; return defaults
        }

        return (explicitMap, ignore, aliases, disallow, explicitByMap);
    }

    private bool TryResolveLkByName(string normalizedName, out int lkAreaId)
    {
        lkAreaId = 0;
        if (_lkNameToId.TryGetValue(normalizedName, out lkAreaId)) return true;
        return false;
    }

    private static bool ContainsDoNotUse(string? name)
    {
        if (string.IsNullOrWhiteSpace(name)) return false;
        return name.IndexOf("do not use", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private sealed class RemapRoot
    {
        public RemapMeta? meta { get; set; }
        public Dictionary<string, string[]>? aliases { get; set; }
        public List<ExplicitMapEntry>? explicit_map { get; set; }
        public List<int>? ignore_area_numbers { get; set; }
        public RemapOptions? options { get; set; }
    }
    private sealed class RemapMeta { public string? src_alias { get; set; } public string? src_build { get; set; } public string? tgt_build { get; set; } public string? generated_at { get; set; } }
    private sealed class ExplicitMapEntry { public int src_areaNumber { get; set; } public int tgt_areaID { get; set; } public int? src_mapID { get; set; } public string? note { get; set; } }
    private sealed class RemapOptions { public bool disallow_do_not_use_targets { get; set; } = true; }
}
