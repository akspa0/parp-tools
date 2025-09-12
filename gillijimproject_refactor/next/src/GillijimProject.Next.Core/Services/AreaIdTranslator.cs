using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using DBCD;
using GillijimProject.Next.Core.Adapters.Dbcd;

namespace GillijimProject.Next.Core.Services;

/// <summary>
/// Translates Alpha AreaIDs to LK AreaIDs using DBCD data and optional overrides.
/// Includes a Map.dbc crosswalk to account for renamed/reused map IDs across versions.
/// </summary>
public sealed class AreaIdTranslator
{
    private readonly DbcdAreaTableProvider _provider;
    private bool _built;

    // Area mapping
    private readonly Dictionary<int, int> _map = new();
    private readonly HashSet<int> _ambiguousAlpha = new();
    private readonly HashSet<int> _unmatchedAlpha = new();

    // Map crosswalk
    private readonly Dictionary<int, int> _mapCrosswalk = new();
    private readonly HashSet<int> _mapAmbiguousAlpha = new();
    private readonly HashSet<int> _mapUnmatchedAlpha = new();

    private static readonly string[] NameFields = new[] { "Name_lang", "AreaName_lang", "ZoneName_lang", "Name", "AreaName", "ZoneName" };
    private static readonly string[] ParentIdFields = new[] { "ParentAreaID", "ParentArea" };
    private static readonly string[] MapIdFields = new[] { "MapID", "ContinentID" };

    private static readonly string[] MapNameFields = new[] { "MapName_lang", "Name_lang", "MapName", "Name" };
    private static readonly string[] MapDirFields = new[] { "Directory", "MapDirectory", "InternalName" };

    public AreaIdTranslator(DbcdAreaTableProvider provider)
    {
        _provider = provider;
    }

    /// <summary>
    /// Build alpha->lk mapping using names, parent context, map crosswalk, fuzziness, and optional overrides.
    /// </summary>
    public void BuildMapping(string? areaOverridesJsonPath = null, string? mapOverridesJsonPath = null)
    {
        _provider.EnsureLoaded();

        // Build Map crosswalk first
        BuildMapCrosswalk(mapOverridesJsonPath);

        // Build indices for AreaTable
        var alpha = _provider.GetAlphaAreaTable();
        var lk = _provider.GetLkAreaTable();
        var alphaIndex = BuildAreaIndex(alpha);
        var lkIndex = BuildAreaIndex(lk);

        // Load area overrides
        var overrides = LoadOverrides(areaOverridesJsonPath);
        foreach (var kv in overrides)
        {
            _map[kv.Key] = kv.Value;
        }

        // Compute area mapping
        foreach (var (alphaId, aRec) in alphaIndex.ById)
        {
            if (_map.ContainsKey(alphaId))
                continue; // user override already set

            // 1) Combined key (name+parent) exact
            if (aRec.CombinedKey is not null && lkIndex.CombinedKeyToIds.TryGetValue(aRec.CombinedKey, out var lkIdsByCombo))
            {
                var chosen = ChooseByCrosswalk(aRec, lkIdsByCombo.Select(id => lkIndex.ById[id]).ToList());
                if (chosen is not null)
                {
                    _map[alphaId] = chosen.Id;
                    continue;
                }
                _ambiguousAlpha.Add(alphaId);
                continue;
            }

            // 2) Name-only exact
            if (aRec.NameKey is not null && lkIndex.NameToIds.TryGetValue(aRec.NameKey, out var lkIdsByName))
            {
                var chosen = ChooseByCrosswalk(aRec, lkIdsByName.Select(id => lkIndex.ById[id]).ToList());
                if (chosen is not null)
                {
                    _map[alphaId] = chosen.Id;
                    continue;
                }
                _ambiguousAlpha.Add(alphaId);
                continue;
            }

            // 3) Fuzzy similarity on names
            var best = FuzzyMatchArea(aRec, lkIndex);
            if (best is not null)
            {
                // Crosswalk preference if possible
                if (CrosswalkPrefers(aRec, best))
                {
                    _map[alphaId] = best.Id;
                    continue;
                }
            }

            _unmatchedAlpha.Add(alphaId);
        }

        _built = true;
    }

    /// <summary>
    /// Attempts to translate an Alpha AreaID to an LK AreaID. Returns true if a mapping was found.
    /// If not found, returns false and sets lkAreaId = alphaAreaId.
    /// </summary>
    public bool TryTranslate(int alphaAreaId, out int lkAreaId)
    {
        if (!_built)
            BuildMapping(null, null);

        if (_map.TryGetValue(alphaAreaId, out lkAreaId))
            return true;

        lkAreaId = alphaAreaId;
        return false;
    }

    // Area report
    public int MatchedCount => _map.Count;
    public int AmbiguousCount => _ambiguousAlpha.Count;
    public int UnmatchedCount => _unmatchedAlpha.Count;
    public IReadOnlyDictionary<int, int> GetMapping() => _map;
    public IReadOnlyCollection<int> GetAmbiguousAlpha() => _ambiguousAlpha.ToArray();
    public IReadOnlyCollection<int> GetUnmatchedAlpha() => _unmatchedAlpha.ToArray();

    // Map crosswalk report
    public int MapMatchedCount => _mapCrosswalk.Count;
    public int MapAmbiguousCount => _mapAmbiguousAlpha.Count;
    public int MapUnmatchedCount => _mapUnmatchedAlpha.Count;
    public IReadOnlyDictionary<int, int> GetMapCrosswalk() => _mapCrosswalk;
    public IReadOnlyCollection<int> GetMapAmbiguousAlpha() => _mapAmbiguousAlpha.ToArray();
    public IReadOnlyCollection<int> GetMapUnmatchedAlpha() => _mapUnmatchedAlpha.ToArray();

    // ===== Area indexing =====
    private sealed class AreaRec
    {
        public required int Id { get; init; }
        public string? NameKey { get; init; }
        public string? ParentNameKey { get; init; }
        public string? CombinedKey { get; init; }
        public int MapId { get; init; }
    }

    private sealed class AreaIndex
    {
        public Dictionary<int, AreaRec> ById { get; } = new();
        public Dictionary<string, List<int>> NameToIds { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, List<int>> CombinedKeyToIds { get; } = new(StringComparer.Ordinal);
    }

    private static AreaIndex BuildAreaIndex(IDBCDStorage storage)
    {
        var idx = new AreaIndex();
        foreach (var pair in (IEnumerable<KeyValuePair<int, DBCDRow>>)storage)
        {
            var id = pair.Key;
            var row = (DBCDRow)pair.Value;
            var name = GetFirstString(storage, row, NameFields);
            var nameKey = Normalize(name);

            var parentId = GetFirstInt(storage, row, ParentIdFields);
            string? parentNameKey = null;
            if (parentId > 0 && storage.ContainsKey(parentId))
            {
                var parentRow = storage[parentId];
                var parentName = GetFirstString(storage, parentRow, NameFields);
                parentNameKey = Normalize(parentName);
            }

            string? combinedKey = (nameKey is not null && parentNameKey is not null) ? $"{nameKey}|{parentNameKey}" : null;
            var mapId = GetFirstInt(storage, row, MapIdFields);

            var rec = new AreaRec
            {
                Id = id,
                NameKey = nameKey,
                ParentNameKey = parentNameKey,
                CombinedKey = combinedKey,
                MapId = mapId
            };

            idx.ById[id] = rec;

            if (nameKey is not null)
                AddToMulti(idx.NameToIds, nameKey, id);

            if (combinedKey is not null)
                AddToMulti(idx.CombinedKeyToIds, combinedKey, id);
        }
        return idx;
    }

    // ===== Map indexing and crosswalk =====
    private sealed class MapRec
    {
        public required int Id { get; init; }
        public string? DirKey { get; init; }
        public string? NameKey { get; init; }
    }

    private sealed class MapIndex
    {
        public Dictionary<int, MapRec> ById { get; } = new();
        public Dictionary<string, List<int>> DirToIds { get; } = new(StringComparer.Ordinal);
        public Dictionary<string, List<int>> NameToIds { get; } = new(StringComparer.Ordinal);
    }

    private static MapIndex BuildMapIndex(IDBCDStorage storage)
    {
        var idx = new MapIndex();
        foreach (var pair in (IEnumerable<KeyValuePair<int, DBCDRow>>)storage)
        {
            var id = pair.Key;
            var row = (DBCDRow)pair.Value;
            var dir = GetFirstString(storage, row, MapDirFields);
            var name = GetFirstString(storage, row, MapNameFields);
            var dirKey = Normalize(dir);
            var nameKey = Normalize(name);

            var rec = new MapRec { Id = id, DirKey = dirKey, NameKey = nameKey };
            idx.ById[id] = rec;

            if (dirKey is not null) AddToMulti(idx.DirToIds, dirKey, id);
            if (nameKey is not null) AddToMulti(idx.NameToIds, nameKey, id);
        }
        return idx;
    }

    private void BuildMapCrosswalk(string? mapOverridesJsonPath)
    {
        // Reset prior state if called multiple times
        _mapCrosswalk.Clear();
        _mapAmbiguousAlpha.Clear();
        _mapUnmatchedAlpha.Clear();

        var alpha = _provider.GetAlphaMapTable();
        var lk = _provider.GetLkMapTable();
        var aIdx = BuildMapIndex(alpha);
        var lIdx = BuildMapIndex(lk);

        // Apply overrides first
        var overrides = LoadOverrides(mapOverridesJsonPath);
        foreach (var kv in overrides)
        {
            _mapCrosswalk[kv.Key] = kv.Value;
        }

        foreach (var (alphaId, aRec) in aIdx.ById)
        {
            if (_mapCrosswalk.ContainsKey(alphaId))
                continue;

            // 1) Exact directory
            if (aRec.DirKey is not null && lIdx.DirToIds.TryGetValue(aRec.DirKey, out var lkIdsByDir))
            {
                if (lkIdsByDir.Count == 1)
                {
                    _mapCrosswalk[alphaId] = lkIdsByDir[0];
                    continue;
                }
                _mapAmbiguousAlpha.Add(alphaId);
                continue;
            }

            // 2) Exact name
            if (aRec.NameKey is not null && lIdx.NameToIds.TryGetValue(aRec.NameKey, out var lkIdsByName))
            {
                if (lkIdsByName.Count == 1)
                {
                    _mapCrosswalk[alphaId] = lkIdsByName[0];
                    continue;
                }
                _mapAmbiguousAlpha.Add(alphaId);
                continue;
            }

            // 3) Fuzzy by name
            if (!string.IsNullOrEmpty(aRec.NameKey))
            {
                double best = 0; int? bestId = null;
                foreach (var kv in lIdx.NameToIds)
                {
                    var score = Similarity(aRec.NameKey!, kv.Key);
                    if (score > best)
                    {
                        best = score;
                        bestId = kv.Value.Count == 1 ? kv.Value[0] : null;
                    }
                }
                if (bestId.HasValue && best >= 0.96)
                {
                    _mapCrosswalk[alphaId] = bestId.Value;
                    continue;
                }
            }

            _mapUnmatchedAlpha.Add(alphaId);
        }
    }

    // ===== Helpers =====
    private static void AddToMulti(Dictionary<string, List<int>> dict, string key, int id)
    {
        if (!dict.TryGetValue(key, out var list))
        {
            list = new List<int>();
            dict[key] = list;
        }
        if (!list.Contains(id)) list.Add(id);
    }

    private static string? GetFirstString(IDBCDStorage storage, DBCDRow row, string[] candidates)
    {
        foreach (var f in candidates)
        {
            if (storage.AvailableColumns.Contains(f))
            {
                try
                {
                    var val = row.FieldAs<string>(f);
                    if (!string.IsNullOrWhiteSpace(val)) return val;
                }
                catch { /* ignore */ }
            }
        }
        return null;
    }

    private static int GetFirstInt(IDBCDStorage storage, DBCDRow row, string[] candidates)
    {
        foreach (var f in candidates)
        {
            if (storage.AvailableColumns.Contains(f))
            {
                try
                {
                    return row.FieldAs<int>(f);
                }
                catch { /* ignore */ }
            }
        }
        return 0;
    }

    private static readonly Regex NonWord = new("[^a-z0-9]+", RegexOptions.Compiled);
    private static string? Normalize(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return null;
        var v = s.Trim().ToLowerInvariant();
        v = NonWord.Replace(v, " ");
        v = Regex.Replace(v, "\\s+", " ");
        v = v.Trim();
        return string.IsNullOrEmpty(v) ? null : v;
    }

    private AreaRec? ChooseByCrosswalk(AreaRec alpha, List<AreaRec> candidates)
    {
        if (candidates.Count == 0) return null;
        if (candidates.Count == 1) return candidates[0];

        // Prefer candidates that match the mapped LK MapID
        if (_mapCrosswalk.TryGetValue(alpha.MapId, out var lkMapId))
        {
            var prefer = candidates.Where(c => c.MapId == lkMapId).ToList();
            if (prefer.Count == 1) return prefer[0];
            if (prefer.Count > 1) return null; // ambiguous within mapped map
            // If none match the mapped map, avoid guessing; fallthrough to ambiguity
            return null;
        }

        // If crosswalk doesn't know, try same MapID (works when IDs did not change)
        var sameMap = candidates.Where(c => c.MapId != 0 && c.MapId == alpha.MapId).ToList();
        if (sameMap.Count == 1) return sameMap[0];

        return null;
    }

    private static AreaRec? FuzzyMatchArea(AreaRec alpha, AreaIndex lk)
    {
        var baseKey = alpha.CombinedKey ?? alpha.NameKey;
        if (string.IsNullOrEmpty(baseKey)) return null;

        double best = 0;
        int? bestId = null;
        foreach (var kv in lk.NameToIds)
        {
            var score = Similarity(baseKey, kv.Key);
            if (score > best)
            {
                best = score;
                bestId = kv.Value.Count == 1 ? kv.Value[0] : null;
            }
        }

        if (bestId.HasValue && best >= 0.92)
            return lk.ById[bestId.Value];

        return null;
    }

    private bool CrosswalkPrefers(AreaRec alpha, AreaRec candidate)
    {
        if (_mapCrosswalk.TryGetValue(alpha.MapId, out var lkMapId))
            return candidate.MapId == lkMapId;
        // If unknown, allow candidate (caller will still treat as mapping)
        return true;
    }

    private static int Levenshtein(ReadOnlySpan<char> a, ReadOnlySpan<char> b)
    {
        int n = a.Length, m = b.Length;
        if (n == 0) return m;
        if (m == 0) return n;
        var d = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) d[i, 0] = i;
        for (int j = 0; j <= m; j++) d[0, j] = j;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + cost);
            }
        }
        return d[n, m];
    }

    private static double Similarity(string a, string b)
    {
        int maxLen = Math.Max(a.Length, b.Length);
        if (maxLen == 0) return 1.0;
        int dist = Levenshtein(a.AsSpan(), b.AsSpan());
        return 1.0 - (double)dist / maxLen;
    }

    private static Dictionary<int, int> LoadOverrides(string? overridesJsonPath)
    {
        if (string.IsNullOrEmpty(overridesJsonPath)) return new Dictionary<int, int>();

        try
        {
            var json = File.ReadAllText(overridesJsonPath);
            return JsonSerializer.Deserialize<Dictionary<int, int>>(json) ?? new Dictionary<int, int>();
        }
        catch
        {
            return new Dictionary<int, int>();
        }
    }
}
