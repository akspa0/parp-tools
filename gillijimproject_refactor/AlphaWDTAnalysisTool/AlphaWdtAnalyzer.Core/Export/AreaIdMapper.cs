using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

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

    public int AlphaNameColumnIndex { get; }
    public int LkNameColumnIndex { get; }

    private AreaIdMapper(
        Dictionary<int, string> alphaIdToName,
        Dictionary<string, int> lkNameToId,
        Dictionary<int, string> lkIdToName,
        int alphaNameCol,
        int lkNameCol)
    {
        _alphaIdToName = alphaIdToName;
        _lkNameToId = lkNameToId;
        _lkIdToName = lkIdToName;
        _lkIds = _lkIdToName.Keys.ToHashSet();
        AlphaNameColumnIndex = alphaNameCol;
        LkNameColumnIndex = lkNameCol;
        _lkFallbackOnMapDungeonId = ResolveOnMapDungeonId(lkNameToId);
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
        lkAreaId = 0;
        alphaName = null;
        lkName = null;

        if (_alphaIdToName.TryGetValue(alphaAreaId, out var aName) && !string.IsNullOrWhiteSpace(aName))
        {
            alphaName = aName;
            var norm = NormalizeName(aName);
            if (_lkNameToId.TryGetValue(norm, out lkAreaId))
            {
                lkName = GetLkNameById(lkAreaId) ?? aName;
                return true;
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
}
