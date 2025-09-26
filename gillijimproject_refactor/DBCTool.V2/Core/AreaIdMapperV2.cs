using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using DBCD;
using DBCD.Providers;
using DBCTool.V2.Crosswalk;
using DBCTool.V2.Mapping;
using static DBCTool.V2.IO.DbdcHelper;

namespace DBCTool.V2.Core;

/// <summary>
/// [PORT] Minimal programmatic API for V2 mapping suitable for integration in other tools.
/// Construct once, then call TryMapArea for repeated (contRaw, areaNumber) lookups.
/// </summary>
public sealed class AreaIdMapperV2
{
    private readonly IDBCDStorage _storSrcArea;
    private readonly IDBCDStorage _storSrcMap;
    private readonly IDBCDStorage _storTgtArea;
    private readonly IDBCDStorage _storTgtMap;

    private readonly string _srcAlias;

    // Indices
    private readonly Dictionary<(int cont, int zoneBase), DBCDRow> _idxSrcZoneByCont = new();
    private readonly string _areaNameColSrc;
    private readonly string _areaNameColTgt;
    private readonly string _keyColSrc;
    private readonly string _idColSrc;

    private readonly Dictionary<int, Dictionary<string, int>> _idxTgtTopZonesByMap = new();
    private readonly Dictionary<int, Dictionary<string, int>> _idxTgtChildrenByZone = new();
    private readonly Dictionary<int, DBCDRow> _tgtIdToRow = new();

    private readonly Dictionary<string, (int id, int map)> _idxTgtTopGlobal = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, (int id, int map)> _idxTgtChildGlobal = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<(string key, int id, int map)> _lkTopList = new();
    private readonly List<(string key, int id, int map)> _lkChildList = new();

    private readonly Dictionary<int, int> _cw053To335;
    private readonly Dictionary<int, string> _mapSrcNames;
    private readonly Dictionary<int, string> _mapTgtNames;

    private readonly AreaMatcher _matcher = new();
    private readonly Dictionary<string, List<AreaRenameOverride>> _renameOverrides = new(StringComparer.OrdinalIgnoreCase);

    private AreaIdMapperV2(
        IDBCDStorage storSrcArea,
        IDBCDStorage storSrcMap,
        IDBCDStorage storTgtArea,
        IDBCDStorage storTgtMap,
        string srcAlias,
        IEnumerable<AreaRenameOverride>? renameOverrides)
    {
        _storSrcArea = storSrcArea;
        _storSrcMap = storSrcMap;
        _storTgtArea = storTgtArea;
        _storTgtMap = storTgtMap;
        _srcAlias = srcAlias;

        _keyColSrc = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID";
        _idColSrc = DetectIdColumn(_storSrcArea);
        _areaNameColSrc = DetectColumn(_storSrcArea, "AreaName_lang", "AreaName", "Name");
        _areaNameColTgt = DetectColumn(_storTgtArea, "AreaName_lang", "AreaName", "Name");

        // Build source zone-by-cont index (zone rows only)
        foreach (var sid in _storSrcArea.Keys)
        {
            var srow = _storSrcArea[sid];
            int areaNumIdx = SafeField<int>(srow, _keyColSrc);
            if (areaNumIdx <= 0) continue;
            int contIdx = SafeField<int>(srow, "ContinentID");
            int hi = (areaNumIdx >> 16) & 0xFFFF;
            int lo = areaNumIdx & 0xFFFF;
            int zb = hi << 16;
            if (lo == 0)
            {
                var k = (contIdx, zb);
                if (!_idxSrcZoneByCont.ContainsKey(k)) _idxSrcZoneByCont[k] = srow;
            }
        }

        // Build target indices
        string idColAreaTgt = DetectIdColumn(_storTgtArea);
        foreach (var key in _storTgtArea.Keys)
        {
            var row = _storTgtArea[key];
            int id = !string.IsNullOrWhiteSpace(idColAreaTgt) ? SafeField<int>(row, idColAreaTgt) : key;
            string name = FirstNonEmpty(SafeField<string>(row, _areaNameColTgt)) ?? string.Empty;
            int parentId = SafeField<int>(row, "ParentAreaID");
            if (parentId <= 0) parentId = id;
            int mapId = SafeField<int>(row, "ContinentID");
            _tgtIdToRow[id] = row;
            var nk = NormKey(name);
            if (parentId == id)
            {
                if (!_idxTgtTopZonesByMap.TryGetValue(mapId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); _idxTgtTopZonesByMap[mapId] = dict; }
                dict[nk] = id;
                _lkTopList.Add((nk, id, mapId));
                if (!_idxTgtTopGlobal.ContainsKey(nk)) _idxTgtTopGlobal[nk] = (id, mapId);
                else if (_idxTgtTopGlobal[nk].id != id) _idxTgtTopGlobal.Remove(nk); // enforce uniqueness only
            }
            else
            {
                if (!_idxTgtChildrenByZone.TryGetValue(parentId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); _idxTgtChildrenByZone[parentId] = dict; }
                dict[nk] = id;
                _lkChildList.Add((nk, id, mapId));
                if (!_idxTgtChildGlobal.ContainsKey(nk)) _idxTgtChildGlobal[nk] = (id, mapId);
                else if (_idxTgtChildGlobal[nk].id != id) _idxTgtChildGlobal.Remove(nk);
            }
        }

        // Crosswalk and map names
        var crosswalk = new MapCrosswalkService();
        _cw053To335 = crosswalk.Build053To335(_storSrcMap, _storTgtMap);
        _mapSrcNames = BuildMapNames(_storSrcMap);
        _mapTgtNames = BuildMapNames(_storTgtMap);

        if (renameOverrides is not null)
        {
            foreach (var ov in renameOverrides)
            {
                var key = NormKey(ov.From);
                if (string.IsNullOrEmpty(key)) continue;
                if (!_renameOverrides.TryGetValue(key, out var list))
                {
                    list = new List<AreaRenameOverride>();
                    _renameOverrides[key] = list;
                }
                list.Add(ov);
            }
        }
    }

    public static AreaIdMapperV2? TryCreate(string dbdDir, string srcAlias, string srcDir, string dir335, IEnumerable<AreaRenameOverride>? renameOverrides = null)
    {
        if (string.IsNullOrWhiteSpace(dbdDir) || string.IsNullOrWhiteSpace(srcAlias) || string.IsNullOrWhiteSpace(srcDir) || string.IsNullOrWhiteSpace(dir335)) return null;
        var provider = new FilesystemDBDProvider(dbdDir);
        var storSrcArea = LoadTable("AreaTable", CanonicalizeBuild(srcAlias), srcDir, provider, DBCD.Locale.EnUS);
        var storSrcMap  = LoadTable("Map",       CanonicalizeBuild(srcAlias), srcDir, provider, DBCD.Locale.EnUS);
        var storTgtArea = LoadTable("AreaTable", CanonicalizeBuild("3.3.5"),  dir335, provider, DBCD.Locale.EnUS);
        var storTgtMap  = LoadTable("Map",       CanonicalizeBuild("3.3.5"),  dir335, provider, DBCD.Locale.EnUS);
        return new AreaIdMapperV2(storSrcArea, storSrcMap, storTgtArea, storTgtMap, srcAlias, renameOverrides);
    }

    public bool TryMapArea(int contRaw, int areaNum, out int targetId, out string method)
    {
        targetId = 0;
        method = "unmatched";

        // Accept Alpha AreaTable.ID inputs by resolving to AreaNumber first
        if (areaNum > 0 && (areaNum >> 16) == 0 && areaNum != 0xEEEE)
        {
            foreach (var key in _storSrcArea.Keys)
            {
                var row = _storSrcArea[key];
                int idVal = string.IsNullOrWhiteSpace(_idColSrc) ? key : SafeField<int>(row, _idColSrc);
                if (idVal == areaNum)
                {
                    areaNum = SafeField<int>(row, _keyColSrc);
                    break;
                }
            }
        }

        int area_hi16 = (areaNum >> 16) & 0xFFFF;
        int area_lo16 = areaNum & 0xFFFF;
        int zoneBase = area_hi16 << 16;

        // Special-case On Map Dungeon
        if (string.Equals(NormKey(GetSrcName(areaNum)), "onmapdungeon", StringComparison.OrdinalIgnoreCase))
        {
            method = "onmapdungeon_0";
            targetId = 0;
            return true;
        }

        // Determine LK map via crosswalk from the source row's continent
        int mapIdX = -1; bool hasMapX = false;
        if (_cw053To335.TryGetValue(contRaw, out var mx)) { mapIdX = mx; hasMapX = true; }
        else if (_mapTgtNames.ContainsKey(contRaw)) { mapIdX = contRaw; hasMapX = true; }

        string nm = GetSrcName(areaNum);
        // Map-locked rename overrides (CompareArea V3)
        var normSourceName = string.IsNullOrWhiteSpace(nm) ? string.Empty : NormKey(nm);
        if (!string.IsNullOrEmpty(normSourceName) && _renameOverrides.TryGetValue(normSourceName, out var overrides) && overrides.Count > 0)
        {
            AreaRenameOverride? selected = null;
            if (hasMapX)
            {
                selected = overrides.FirstOrDefault(o => o.TargetMapId == mapIdX);
            }
            else if (overrides.Count == 1)
            {
                selected = overrides[0];
            }

            if (selected is not null && _tgtIdToRow.TryGetValue(selected.TargetAreaId, out var overrideRow))
            {
                int overrideMap = SafeField<int>(overrideRow, "ContinentID");
                if (!hasMapX || MapsAreEquivalent(overrideMap, mapIdX))
                {
                    targetId = selected.TargetAreaId;
                    method = "rename_override";
                    return true;
                }
            }
        }

        // Build zone-only chain
        var chain = new List<string>();
        if (area_lo16 == 0)
        {
            if (!string.IsNullOrWhiteSpace(nm)) chain.Add(NormKey(nm));
        }
        else
        {
            if (_idxSrcZoneByCont.TryGetValue((contRaw, zoneBase), out var zrow))
            {
                string zname = FirstNonEmpty(SafeField<string>(zrow, _areaNameColSrc)) ?? string.Empty;
                if (!string.IsNullOrWhiteSpace(zname)) chain.Add(NormKey(zname));
            }
            if (chain.Count == 0 && !string.IsNullOrWhiteSpace(nm)) chain.Add(NormKey(nm));
        }

        int depth;
        int chosen = -1;
        if (hasMapX && chain.Count > 0)
        {
            chosen = _matcher.TryMatchChainExact(mapIdX, chain, _idxTgtTopZonesByMap, _idxTgtChildrenByZone, out depth);
            if (chosen >= 0) { targetId = chosen; method = (depth == chain.Count) ? "exact" : "zone_only"; return true; }
        }

        // Global rename (unique) across LK
        if (chain.Count > 0)
        {
            var k = chain[0];
            if (_idxTgtTopGlobal.TryGetValue(k, out var rec))
            { targetId = rec.id; method = "rename_global"; return true; }
        }
        if (!string.IsNullOrWhiteSpace(nm))
        {
            var k2 = NormKey(nm);
            if (_idxTgtChildGlobal.TryGetValue(k2, out var rec2))
            { targetId = rec2.id; method = "rename_global_child"; return true; }
        }

        // Fuzzy rename top-level then child
        if (chain.Count > 0)
        {
            var k = chain[0];
            var (fid, ok) = FindBestFuzzy(k, _lkTopList, 2);
            if (ok) { targetId = fid; method = "rename_fuzzy"; return true; }
        }
        if (!string.IsNullOrWhiteSpace(nm))
        {
            var k = NormKey(nm);
            var (fid, ok) = FindBestFuzzy(k, _lkChildList, 2);
            if (ok) { targetId = fid; method = "rename_fuzzy_child"; return true; }
        }

        return false;
    }

    private string GetSrcName(int areaNum)
    {
        // Look up by value of AreaNumber/ID, not by row key
        foreach (var key in _storSrcArea.Keys)
        {
            var row = _storSrcArea[key];
            int v = SafeField<int>(row, _keyColSrc);
            if (v == areaNum)
            {
                return FirstNonEmpty(SafeField<string>(row, _areaNameColSrc)) ?? string.Empty;
            }
        }
        return string.Empty;
    }

    private static (int id, bool ok) FindBestFuzzy(string keyNorm, List<(string key, int id, int map)> candidates, int threshold)
    {
        int best = int.MaxValue; int second = int.MaxValue; int bid = -1;
        foreach (var c in candidates)
        {
            int d = EditDistance(keyNorm, c.key);
            if (d < best) { second = best; best = d; bid = c.id; }
            else if (d < second) { second = d; }
        }
        bool unique = best <= threshold && best < second;
        return unique ? (bid, true) : (-1, false);
    }

    private static int EditDistance(string a, string b)
    {
        a ??= string.Empty; b ??= string.Empty;
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) dp[i, 0] = i;
        for (int j = 0; j <= m; j++) dp[0, j] = j;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1), dp[i - 1, j - 1] + cost);
            }
        }
        return dp[n, m];
    }

    private static string CanonicalizeBuild(string alias)
    {
        return alias switch
        {
            "0.5.3" => "0.5.3.3368",
            "0.5.5" => "0.5.5.3494",
            "0.6.0" => "0.6.0.3592",
            "3.3.5" => "3.3.5.12340",
            _ => alias
        };
    }

    private static bool MapsAreEquivalent(int a, int b)
    {
        if (a == b) return true;
        return (a == 451 && b == 0) || (a == 0 && b == 451);
    }

    public static List<AreaRenameOverride> LoadRenameOverridesCsv(string path)
    {
        var list = new List<AreaRenameOverride>();
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return list;

        foreach (var line in File.ReadLines(path))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed)) continue;
            if (trimmed.StartsWith('#')) continue;

            var parts = SplitCsvLine(trimmed);
            if (parts.Count < 4) continue;

            var from = parts[0].Trim().Trim('"');
            var to = parts[1].Trim().Trim('"');
            if (string.IsNullOrEmpty(from) || string.IsNullOrEmpty(to)) continue;

            if (!int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var mapId)) continue;
            if (!int.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out var areaId)) continue;

            list.Add(new AreaRenameOverride(from, to, mapId, areaId));
        }

        return list;
    }

    private static List<string> SplitCsvLine(string line)
    {
        var values = new List<string>();
        var sb = new StringBuilder();
        bool inQuotes = false;

        for (int i = 0; i < line.Length; i++)
        {
            char ch = line[i];
            if (ch == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    sb.Append('"');
                    i++;
                    continue;
                }

                inQuotes = !inQuotes;
                continue;
            }

            if (ch == ',' && !inQuotes)
            {
                values.Add(sb.ToString());
                sb.Clear();
                continue;
            }

            sb.Append(ch);
        }

        values.Add(sb.ToString());
        return values;
    }

    public sealed record AreaRenameOverride(string From, string TargetName, int TargetMapId, int TargetAreaId);
}
