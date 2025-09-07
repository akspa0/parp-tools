using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AlphaWdtAnalyzer.Core.Dbc;

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

    public static AreaIdMapper? TryCreate(string? alphaDbcPath, string? lkDbcPath)
    {
        if (string.IsNullOrWhiteSpace(alphaDbcPath) || string.IsNullOrWhiteSpace(lkDbcPath)) return null;
        if (!File.Exists(alphaDbcPath) || !File.Exists(lkDbcPath)) return null;

        var alpha = LoadTable(alphaDbcPath);
        var lk = LoadTable(lkDbcPath);
        if (alpha is null || lk is null) return null;

        // Pick a single, robust name column per table
        var alphaNameCol = GuessNameColumn(alpha);
        var lkNameCol = GuessNameColumn(lk);

        var alphaIdToName = BuildIdToName(alpha, alphaNameCol);
        var (lkNameToId, lkIdToName) = BuildNameToId(lk, lkNameCol);

        return new AreaIdMapper(alphaIdToName, lkNameToId, lkIdToName, alphaNameCol, lkNameCol);
    }

    public bool TryResolveById(int alphaAreaId, out int lkAreaId, out string reason)
    {
        // Keep for diagnostics but not used in strict name-only flows
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

        // Strict name-only mapping (exact normalized match)
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

    private static RawDbcTable? LoadTable(string path)
    {
        try
        {
            return RawDbcParser.Parse(path);
        }
        catch
        {
            return null;
        }
    }

    private static Dictionary<int, string> BuildIdToName(RawDbcTable table, int nameCol)
    {
        var result = new Dictionary<int, string>();
        foreach (var row in table.Rows)
        {
            int id = unchecked((int)row.Fields[0]);
            string? s = (nameCol >= 0 && nameCol < row.GuessedStrings.Length) ? row.GuessedStrings[nameCol] : null;
            if (!string.IsNullOrWhiteSpace(s))
            {
                result[id] = s!;
            }
        }
        return result;
    }

    private static (Dictionary<string, int> nameToId, Dictionary<int, string> idToName) BuildNameToId(RawDbcTable table, int nameCol)
    {
        var nameToId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var idToName = new Dictionary<int, string>();
        foreach (var row in table.Rows)
        {
            int id = unchecked((int)row.Fields[0]);
            string? s = (nameCol >= 0 && nameCol < row.GuessedStrings.Length) ? row.GuessedStrings[nameCol] : null;
            if (string.IsNullOrWhiteSpace(s)) continue;
            var norm = NormalizeName(s);
            if (!nameToId.ContainsKey(norm)) nameToId[norm] = id;
            if (!idToName.ContainsKey(id)) idToName[id] = s;
        }
        return (nameToId, idToName);
    }

    private static int GuessNameColumn(RawDbcTable table)
    {
        // Heuristic: choose the column index with the highest count of plausible name strings
        // Plausible name: non-empty, ASCII printable, contains letters, no path separators
        int bestCol = -1;
        int bestScore = -1;
        for (int f = 0; f < table.FieldCount; f++)
        {
            int score = 0;
            foreach (var row in table.Rows)
            {
                var s = row.GuessedStrings.Length > f ? row.GuessedStrings[f] : null;
                if (IsPlausibleName(s)) score++;
            }
            if (score > bestScore)
            {
                bestScore = score;
                bestCol = f;
            }
        }
        return bestCol;
    }

    private static bool IsPlausibleName(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return false;
        if (s.Length < 2) return false;
        if (s.IndexOf('/') >= 0 || s.IndexOf('\\') >= 0) return false;
        foreach (var ch in s)
        {
            if (ch < 0x20 || ch > 0x7E) return false; // ASCII printable only
        }
        return Regex.IsMatch(s, "[A-Za-z]");
    }

    private static string NormalizeName(string s)
    {
        var norm = s.Trim();
        norm = Regex.Replace(norm, "[\'\"]", ""); // remove quotes/possessives
        norm = Regex.Replace(norm, "\\s+", " ");
        return norm;
    }

    private static double FuzzyScore(string a, string b)
    {
        a = a.ToLowerInvariant();
        b = b.ToLowerInvariant();
        if (a == b) return 1.0;
        if (a.Length == 0 || b.Length == 0) return 0.0;
        if (a.StartsWith(b) || b.StartsWith(a)) return 0.9;
        if (a.Contains(b) || b.Contains(a)) return 0.8;
        // fallback Jaccard on words
        var wa = a.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var wb = b.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var inter = wa.Intersect(wb).Count();
        var union = wa.Union(wb).Count();
        return union == 0 ? 0.0 : (double)inter / union;
    }

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
}
