using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core.Export;

/// <summary>
/// Loads DBCTool.V2 patch CSVs (Area_patch_crosswalk_*.csv) and provides fast lookup
/// from (src_mapId, src_areaNumber) to target 3.3.5 AreaID. Also captures tgt_parentID and tgt_name when available.
/// </summary>
public sealed class DbcPatchMapping
{
    private readonly Dictionary<int, Dictionary<int, int>> _bySrcMap = new(); // src_mapId -> (src_areaNumber -> tgt_areaID)
    private readonly Dictionary<int, int> _global = new(); // src_areaNumber -> tgt_areaID (from global files)
    private readonly Dictionary<int, string> _tgtNameById = new(); // tgt_areaID -> tgt_name (best-effort)
    private readonly Dictionary<int, int> _tgtParentById = new(); // tgt_areaID -> tgt_parentID (best-effort)
    private readonly Dictionary<string, Dictionary<int, int>> _bySrcName = new(StringComparer.OrdinalIgnoreCase); // src_mapName -> (src_areaNumber -> tgt_areaID)
    private readonly Dictionary<int, Dictionary<int, int>> _byTgtMapX = new(); // tgt_mapId_xwalk -> (src_areaNumber -> tgt_areaID)
    private readonly Dictionary<string, Dictionary<int, int>> _byTgtNameX = new(StringComparer.OrdinalIgnoreCase); // tgt_mapName_xwalk -> (src_areaNumber -> tgt_areaID)
    // via060-preferred target-locked dictionaries
    private readonly Dictionary<int, Dictionary<int, int>> _byTgtMapXVia = new(); // via060 only
    private readonly Dictionary<string, Dictionary<int, int>> _byTgtNameXVia = new(StringComparer.OrdinalIgnoreCase); // via060 only
    private readonly Dictionary<int, (int target, bool via)> _bySrcAreaBest = new(); // src_areaNumber -> best tgt_areaID, prefer via060
    private readonly Dictionary<string, Dictionary<int, (int target, bool via)>> _bestBySrcNameArea = new(StringComparer.OrdinalIgnoreCase); // src_mapName -> (src_areaNumber -> best tgt)
    private readonly Dictionary<int, ZoneEntry> _zoneByBase = new(); // zoneBase -> entry
    private readonly Dictionary<int, Dictionary<int, ZoneEntry>> _subByZoneBase = new(); // zoneBase -> (subLo -> entry)
    private readonly Dictionary<(int srcMapId, int srcAreaNumber), MidEntry> _midBySrcArea = new(); // (src_mapId, src_areaNumber) -> mid info
    private readonly Dictionary<(int mapId, int midAreaId), (int targetId, bool via)> _tgtByMidArea = new(); // (mid_mapID, mid_areaID) -> target areaID
    private readonly Dictionary<int, IReadOnlyList<int>> _childIdsByTarget = new(); // tgt_zoneId -> child areaIDs
    private readonly Dictionary<int, IReadOnlyList<string>> _childNamesByTarget = new(); // tgt_zoneId -> child names
    private readonly Dictionary<string, int> _srcMapIdByName = new(StringComparer.OrdinalIgnoreCase);

    private readonly struct ZoneEntry
    {
        public ZoneEntry(int targetId, int targetParentId, int targetMapId, bool via060)
        {
            TargetId = targetId;
            TargetParentId = targetParentId;
            TargetMapId = targetMapId;
            Via060 = via060;
        }

        public int TargetId { get; }
        public int TargetParentId { get; }
        public int TargetMapId { get; }
        public bool Via060 { get; }
        public bool IsChild => TargetParentId > 0 && TargetParentId != TargetId;
    }

    private readonly struct MidEntry
    {
        public MidEntry(int midAreaId, int midMapId, int midParentId, string midChain, bool via060)
        {
            MidAreaId = midAreaId;
            MidMapId = midMapId;
            MidParentId = midParentId;
            MidChain = midChain;
            Via060 = via060;
        }

        public int MidAreaId { get; }
        public int MidMapId { get; }
        public int MidParentId { get; }
        public string MidChain { get; }
        public bool Via060 { get; }
    }

    // Strict mode: no inferred targets, no LK dump usage in numeric mapping.

    public void LoadFile(string csvPath)
    {
        if (!File.Exists(csvPath)) return;
        using var sr = new StreamReader(csvPath);
        string? header = sr.ReadLine();
        int srcMapIdx = -1, srcMapNameIdx = -1, srcAreaIdx = -1, tgtIdIdx = -1, tgtNameIdx = -1, tgtParentIdx = -1, tgtMapXIdx = -1, tgtMapNameXIdx = -1;
        bool hasHeader = false;
        bool isVia060 = csvPath.IndexOf("via060", StringComparison.OrdinalIgnoreCase) >= 0;
        string[]? headerCols = null;
        int midMapIdx = -1, midAreaIdx = -1, midParentIdx = -1, midChainIdx = -1;
        int childIdsIdx = -1, childNamesIdx = -1;
        if (header is not null)
        {
            var cols = SplitCsv(header);
            headerCols = cols;
            for (int i = 0; i < cols.Length; i++)
            {
                var c = cols[i].Trim().ToLowerInvariant();
                if (c == "src_mapid") srcMapIdx = i;
                else if (c == "src_mapname") srcMapNameIdx = i;
                else if (c == "src_areanumber") srcAreaIdx = i;
                else if (c == "tgt_areaid") tgtIdIdx = i;
                else if (c == "tgt_mapid_xwalk") tgtMapXIdx = i;
                else if (c == "tgt_mapname_xwalk") tgtMapNameXIdx = i;
                else if (c == "tgt_name") tgtNameIdx = i;
                else if (c == "tgt_parentid") tgtParentIdx = i;
                else if (c == "mid060_mapid") midMapIdx = i;
                else if (c == "mid060_areaid") midAreaIdx = i;
                else if (c == "mid060_parentid") midParentIdx = i;
                else if (c == "mid060_chain") midChainIdx = i;
                else if (c == "tgt_child_ids") childIdsIdx = i;
                else if (c == "tgt_child_names") childNamesIdx = i;
                // keep src_name if present
                else if (c == "src_name") { /* captured later via index lookup */ }
            }

    // No fallback scanning in strict mode
            hasHeader = srcAreaIdx >= 0 && tgtIdIdx >= 0; // src_mapId optional (global file)
            if (!hasHeader)
            {
                // If the first line isn't a header, treat it as a data line
                // Reset stream to start so we can read it as data
                sr.DiscardBufferedData();
                sr.BaseStream.Seek(0, SeekOrigin.Begin);
            }
        }

        string? line;
        while ((line = sr.ReadLine()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var cells = SplitCsv(line);
            // Fast path for known column order
            if (hasHeader)
            {
                int sMap = -1;
                if (srcMapIdx >= 0 && srcMapIdx < cells.Length)
                    int.TryParse(cells[srcMapIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out sMap);
                string sMapName = string.Empty;
                if (srcMapNameIdx >= 0 && srcMapNameIdx < cells.Length)
                    sMapName = (cells[srcMapNameIdx] ?? string.Empty).Trim();

                if (srcAreaIdx < 0 || srcAreaIdx >= cells.Length) continue;
                if (tgtIdIdx < 0 || tgtIdIdx >= cells.Length) continue;
                if (!int.TryParse(cells[srcAreaIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out var sArea)) continue;
                if (!int.TryParse(cells[tgtIdIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tId)) continue;
                int tMapX = -1;
                if (tgtMapXIdx >= 0 && tgtMapXIdx < cells.Length)
                    int.TryParse(cells[tgtMapXIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out tMapX);
                string tMapNameX = string.Empty;
                if (tgtMapNameXIdx >= 0 && tgtMapNameXIdx < cells.Length)
                    tMapNameX = (cells[tgtMapNameXIdx] ?? string.Empty).Trim();
                int? tParOpt = null;
                if (tgtParentIdx >= 0 && tgtParentIdx < cells.Length && int.TryParse(cells[tgtParentIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tParHdr))
                    tParOpt = tParHdr;
                Insert(sMap, sArea, tId, sMapName, tMapX, tMapNameX, isVia060, tParOpt);
                if (!string.IsNullOrWhiteSpace(sMapName) && sMap >= 0)
                {
                    if (!_srcMapIdByName.ContainsKey(sMapName))
                        _srcMapIdByName[sMapName] = sMap;
                }
                if (tgtNameIdx >= 0 && tgtNameIdx < cells.Length)
                {
                    var nm = cells[tgtNameIdx]?.Trim();
                    if (!string.IsNullOrEmpty(nm) && !_tgtNameById.ContainsKey(tId)) _tgtNameById[tId] = nm;
                }
                if (tParOpt.HasValue && tParOpt.Value > 0) _tgtParentById[tId] = tParOpt.Value;

                // Capture mid/pivot metadata when available
                int midMapId = -1;
                if (midMapIdx >= 0 && midMapIdx < cells.Length)
                    int.TryParse(cells[midMapIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out midMapId);
                if (midMapId < 0 && sMap >= 0)
                    midMapId = sMap;
                int midAreaId = -1;
                if (midAreaIdx >= 0 && midAreaIdx < cells.Length)
                    int.TryParse(cells[midAreaIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out midAreaId);
                int midParentId = -1;
                if (midParentIdx >= 0 && midParentIdx < cells.Length)
                    int.TryParse(cells[midParentIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out midParentId);
                string midChain = string.Empty;
                if (midChainIdx >= 0 && midChainIdx < cells.Length)
                    midChain = (cells[midChainIdx] ?? string.Empty).Trim();

                bool hasPivotData = midAreaIdx >= 0 || midParentIdx >= 0 || midMapIdx >= 0 || !string.IsNullOrWhiteSpace(midChain);
                int midAreaForEntry = midAreaId;
                if (midAreaForEntry <= 0 && midParentId > 0)
                {
                    midAreaForEntry = midParentId;
                }

                if (midAreaForEntry > 0 || hasPivotData)
                {
                    var midMapForEntry = midMapId >= 0 ? midMapId : sMap;
                    int srcMapKey = sMap >= 0 ? sMap : -1;
                    _midBySrcArea[(srcMapKey, sArea)] = new MidEntry(midAreaForEntry, midMapForEntry, midParentId > 0 ? midParentId : midAreaForEntry, midChain, isVia060);
                    if (midAreaForEntry > 0 && tId > 0)
                    {
                        var midMapKey = midMapForEntry >= 0 ? midMapForEntry : -1;
                        var key = (midMapKey, midAreaForEntry);
                        if (!_tgtByMidArea.TryGetValue(key, out var existing) || (!existing.via && isVia060))
                        {
                            _tgtByMidArea[key] = (tId, isVia060);
                        }
                    }
                }

                if (childIdsIdx >= 0 && childIdsIdx < cells.Length && tId > 0)
                {
                    var rawIds = cells[childIdsIdx]?.Trim();
                    if (!string.IsNullOrEmpty(rawIds))
                    {
                        var splitIds = rawIds.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                        var parsedIds = new List<int>(splitIds.Length);
                        foreach (var part in splitIds)
                        {
                            if (int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out var cid) && cid > 0)
                            {
                                parsedIds.Add(cid);
                            }
                        }
                        if (parsedIds.Count > 0)
                        {
                            var distinct = parsedIds.Distinct().OrderBy(x => x).ToArray();
                            _childIdsByTarget[tId] = distinct;
                        }
                    }
                }
                if (childNamesIdx >= 0 && childNamesIdx < cells.Length && tId > 0)
                {
                    var rawNames = cells[childNamesIdx]?.Trim();
                    if (!string.IsNullOrEmpty(rawNames))
                    {
                        var splitNames = rawNames.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                        if (splitNames.Length > 0)
                        {
                            _childNamesByTarget[tId] = splitNames.ToArray();
                        }
                    }
                }
            }
            else
            {
                // Fallback: try to parse as: src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name
                var cells2 = cells;
                if (cells2.Length >= 9)
                {
                    int.TryParse(cells2[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var sMap);
                    string sMapName = (cells2.Length > 1 ? (cells2[1] ?? string.Empty).Trim() : string.Empty);
                    if (!int.TryParse(cells2[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var sArea)) continue;
                    if (!int.TryParse(cells2[7], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tId)) continue;
                    int tMapX = -1;
                    if (cells2.Length > 5)
                        int.TryParse(cells2[5], NumberStyles.Integer, CultureInfo.InvariantCulture, out tMapX);
                    string tMapNameX = (cells2.Length > 6 ? (cells2[6] ?? string.Empty).Trim() : string.Empty);
                    int? tParOpt = null;
                    if (cells2.Length >= 9 && int.TryParse(cells2[8], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tPar2))
                        tParOpt = tPar2;
                    Insert(sMap, sArea, tId, sMapName, tMapX, tMapNameX, isVia060, tParOpt);
                    // Name at index 9 if present
                    if (cells2.Length >= 10)
                    {
                        var nm = cells2[9]?.Trim();
                        if (!string.IsNullOrEmpty(nm) && !_tgtNameById.ContainsKey(tId)) _tgtNameById[tId] = nm;
                    }
                    // Parent at index 8 if present
                    if (tParOpt.HasValue && tParOpt.Value > 0) _tgtParentById[tId] = tParOpt.Value;
                }
            }
        }
    }

    public bool TryMap(int srcMapId, int srcAreaNumber, out int targetId)
    {
        // Strict: only return when explicit per-map entry exists
        if (_bySrcMap.TryGetValue(srcMapId, out var dict) && dict.TryGetValue(srcAreaNumber, out targetId))
            return true;
        targetId = 0;
        return false;
    }

    public bool TryMapByName(string srcMapName, int srcAreaNumber, out int targetId)
    {
        if (!string.IsNullOrWhiteSpace(srcMapName) && _bySrcName.TryGetValue(srcMapName, out var dict) && dict.TryGetValue(srcAreaNumber, out targetId))
            return true;
        targetId = 0;
        return false;
    }

    public bool TryMapByTarget(int tgtMapIdX, int srcAreaNumber, out int targetId)
    {
        if (_byTgtMapX.TryGetValue(tgtMapIdX, out var dict) && dict.TryGetValue(srcAreaNumber, out targetId)) return true;
        targetId = 0;
        return false;
    }

    // Prefer via060 mapping when present; expose whether the hit came from via060
    public bool TryMapByTargetViaFirst(int tgtMapIdX, int srcAreaNumber, out int targetId, out bool via060)
    {
        if (_byTgtMapXVia.TryGetValue(tgtMapIdX, out var dVia) && dVia.TryGetValue(srcAreaNumber, out targetId)) { via060 = true; return true; }
        if (_byTgtMapX.TryGetValue(tgtMapIdX, out var dAny) && dAny.TryGetValue(srcAreaNumber, out targetId)) { via060 = false; return true; }
        targetId = 0; via060 = false; return false;
    }

    // Simple numeric mapping scoped by src_mapName: (mapName, src_areaNumber) -> best tgt_areaID
    // Prefer via060 when available, but accept non-via if via is absent.
    public bool TryMapBySrcAreaSimple(string srcMapName, int srcAreaNumber, out int targetId)
    {
        if (!string.IsNullOrWhiteSpace(srcMapName)
            && _bestBySrcNameArea.TryGetValue(srcMapName, out var dict)
            && dict.TryGetValue(srcAreaNumber, out var rec))
        {
            targetId = rec.target;
            return targetId > 0;
        }
        targetId = 0;
        return false;
    }

    public bool TryMapByTargetName(string tgtMapName, int srcAreaNumber, out int targetId)
    {
        if (!string.IsNullOrWhiteSpace(tgtMapName) && _byTgtNameX.TryGetValue(tgtMapName, out var dict) && dict.TryGetValue(srcAreaNumber, out targetId)) return true;
        targetId = 0;
        return false;
    }

    public bool TryMapByTargetNameViaFirst(string tgtMapName, int srcAreaNumber, out int targetId, out bool via060)
    {
        if (!string.IsNullOrWhiteSpace(tgtMapName) && _byTgtNameXVia.TryGetValue(tgtMapName, out var dVia) && dVia.TryGetValue(srcAreaNumber, out targetId)) { via060 = true; return true; }
        if (!string.IsNullOrWhiteSpace(tgtMapName) && _byTgtNameX.TryGetValue(tgtMapName, out var dAny) && dAny.TryGetValue(srcAreaNumber, out targetId)) { via060 = false; return true; }
        targetId = 0; via060 = false; return false;
    }

    public bool TryMapByAnyTarget(int srcAreaNumber, out int targetId)
    {
        int hits = 0; int cand = 0;
        foreach (var kv in _byTgtMapX)
        {
            if (kv.Value.TryGetValue(srcAreaNumber, out var tid))
            {
                hits++; cand = tid;
                if (hits > 1) break;
            }
        }
        if (hits == 1) { targetId = cand; return true; }
        targetId = 0; return false;
    }

    public bool TryMapSubZone(int zoneBase, int subLo, int? mapIdHint, out int targetId, out bool via060)
    {
        targetId = 0; via060 = false;
        if (zoneBase == 0 || subLo <= 0) return false;
        if (_subByZoneBase.TryGetValue(zoneBase, out var dict) && dict.TryGetValue(subLo, out var entry))
        {
            if (!MatchesMap(entry, mapIdHint)) return false;
            if (entry.TargetId <= 0) return false;
            targetId = entry.TargetId;
            via060 = entry.Via060;
            return true;
        }
        return false;
    }

    public bool TryMapZone(int zoneBase, int? mapIdHint, out int targetId, out bool via060)
    {
        targetId = 0; via060 = false;
        if (zoneBase == 0) return false;
        if (_zoneByBase.TryGetValue(zoneBase, out var entry))
        {
            if (!MatchesMap(entry, mapIdHint)) return false;
            if (entry.TargetId <= 0) return false;
            targetId = entry.TargetId;
            via060 = entry.Via060;
            return true;
        }
        return false;
    }

    public bool TryMapBySrcAreaNumber(int srcAreaNumber, out int targetId, out bool via060)
    {
        targetId = 0; via060 = false;
        if (_bySrcAreaBest.TryGetValue(srcAreaNumber, out var rec))
        {
            if (rec.target > 0)
            {
                targetId = rec.target;
                via060 = rec.via;
                return true;
            }
        }
        return false;
    }

    public bool TryResolveSourceMapId(string? srcMapName, out int mapId)
    {
        mapId = -1;
        if (string.IsNullOrWhiteSpace(srcMapName))
        {
            return false;
        }
        return _srcMapIdByName.TryGetValue(srcMapName, out mapId);
    }

    public bool TryMapViaMid(int? srcMapId, int srcAreaNumber, out int targetId, out int midAreaId, out bool via060)
    {
        targetId = 0;
        midAreaId = 0;
        via060 = false;
        if (TryGetMidEntry(srcMapId, srcAreaNumber, out var mid))
        {
            midAreaId = mid.MidAreaId;
            via060 = mid.Via060;
            if (mid.MidAreaId > 0)
            {
                var midMapKey = mid.MidMapId >= 0 ? mid.MidMapId : -1;
                var key = (midMapKey, mid.MidAreaId);
                if (_tgtByMidArea.TryGetValue(key, out var pair))
                {
                    targetId = pair.targetId;
                    if (pair.via) via060 = true;
                    return targetId > 0;
                }
            }
        }
        return false;
    }

    public bool TryGetMidInfo(int? srcMapId, int srcAreaNumber, out int midAreaId, out int midMapId, out int midParentId, out string midChain, out bool via060)
    {
        if (TryGetMidEntry(srcMapId, srcAreaNumber, out var mid))
        {
            midAreaId = mid.MidAreaId;
            midMapId = mid.MidMapId;
            midParentId = mid.MidParentId;
            midChain = mid.MidChain;
            via060 = mid.Via060;
            return true;
        }
        midAreaId = 0;
        midMapId = -1;
        midParentId = -1;
        midChain = string.Empty;
        via060 = false;
        return false;
    }

    public bool TryGetMidInfo(int srcAreaNumber, out int midAreaId, out int midMapId, out int midParentId, out string midChain, out bool via060)
    {
        return TryGetMidInfo(null, srcAreaNumber, out midAreaId, out midMapId, out midParentId, out midChain, out via060);
    }

    public bool TryGetMidInfo(int srcAreaNumber, out int midAreaId)
    {
        if (TryGetMidInfo(null, srcAreaNumber, out midAreaId, out _, out _, out _, out _))
        {
            return true;
        }
        midAreaId = 0;
        return false;
    }

    public IReadOnlyList<int> GetChildCandidateIds(int targetZoneId)
    {
        return _childIdsByTarget.TryGetValue(targetZoneId, out var list) ? list : Array.Empty<int>();
    }

    public IReadOnlyList<string> GetChildCandidateNames(int targetZoneId)
    {
        return _childNamesByTarget.TryGetValue(targetZoneId, out var list) ? list : Array.Empty<string>();
    }

    private bool TryGetMidEntry(int? srcMapId, int srcAreaNumber, out MidEntry entry)
    {
        entry = default;
        int mapKey = srcMapId.GetValueOrDefault(-1);
        if (_midBySrcArea.TryGetValue((mapKey, srcAreaNumber), out entry))
        {
            return true;
        }

        if (_midBySrcArea.TryGetValue((-1, srcAreaNumber), out entry))
        {
            return true;
        }

        bool hasCandidate = false;
        MidEntry candidate = default;
        foreach (var kv in _midBySrcArea)
        {
            if (kv.Key.srcAreaNumber != srcAreaNumber) continue;

            if (srcMapId.HasValue && srcMapId.Value >= 0)
            {
                if (kv.Key.srcMapId == srcMapId.Value)
                {
                    entry = kv.Value;
                    return true;
                }
            }

            if (!hasCandidate)
            {
                candidate = kv.Value;
                hasCandidate = true;
            }
            else
            {
                // Multiple candidates without a map hint; treat as ambiguous
                if (!srcMapId.HasValue || srcMapId.Value < 0)
                {
                    return false;
                }
            }
        }

        if (hasCandidate)
        {
            entry = candidate;
            return true;
        }

        return false;
    }

    private void Insert(int srcMapId, int srcAreaNumber, int targetId, string? srcMapName, int tgtMapIdX, string? tgtMapNameX, bool isVia060, int? tgtParentId)
    {
        int zoneBase = (srcAreaNumber & unchecked((int)0xFFFF0000));
        int subLo = srcAreaNumber & 0xFFFF;

        if (srcMapId >= 0)
        {
            if (!_bySrcMap.TryGetValue(srcMapId, out var dict)) { dict = new Dictionary<int, int>(); _bySrcMap[srcMapId] = dict; }
            dict[srcAreaNumber] = targetId;
        }
        else
        {
            _global[srcAreaNumber] = targetId;
        }
        // Maintain best numeric mapping per src_areaNumber
        if (_bySrcAreaBest.TryGetValue(srcAreaNumber, out var cur))
        {
            // Prefer via060 over non-via; otherwise keep first seen
            if (!cur.via && isVia060)
                _bySrcAreaBest[srcAreaNumber] = (targetId, true);
        }
        else
        {
            _bySrcAreaBest[srcAreaNumber] = (targetId, isVia060);
        }
        if (!string.IsNullOrWhiteSpace(srcMapName))
        {
            if (!_bySrcName.TryGetValue(srcMapName, out var byArea))
            {
                byArea = new Dictionary<int, int>();
                _bySrcName[srcMapName] = byArea;
            }
            byArea[srcAreaNumber] = targetId;

            if (targetId > 0)
            {
                if (!_bestBySrcNameArea.TryGetValue(srcMapName, out var bestDict))
                {
                    bestDict = new Dictionary<int, (int target, bool via)>();
                    _bestBySrcNameArea[srcMapName] = bestDict;
                }

                if (bestDict.TryGetValue(srcAreaNumber, out var cur2))
                {
                    // Prefer via060 over non-via
                    if (!cur2.via && isVia060)
                    {
                        bestDict[srcAreaNumber] = (targetId, true);
                    }
                    else if (cur2.via == isVia060 && tgtParentId.HasValue)
                    {
                        // Prefer child matches when both have same via flag
                        var curIsChild = _tgtParentById.TryGetValue(cur2.target, out var p1) && p1 > 0;
                        var newIsChild = tgtParentId.Value > 0;
                        if (newIsChild && !curIsChild)
                        {
                            bestDict[srcAreaNumber] = (targetId, isVia060);
                        }
                    }
                }
                else
                {
                    bestDict[srcAreaNumber] = (targetId, isVia060);
                }
            }
        }
        // Populate target-locked numeric mapping from all rows with non-zero target
        if (/*isVia060 &&*/ targetId > 0 && tgtMapIdX >= 0)
        {
            if (!_byTgtMapX.TryGetValue(tgtMapIdX, out var d2)) { d2 = new Dictionary<int, int>(); _byTgtMapX[tgtMapIdX] = d2; }
            if (d2.TryGetValue(srcAreaNumber, out var existingTid))
            {
                // Prefer child (has parent) over parent (no parent)
                var existingIsChild = _tgtParentById.TryGetValue(existingTid, out var p1) && p1 > 0;
                var newIsChild = tgtParentId.HasValue && tgtParentId.Value > 0;
                if (newIsChild && !existingIsChild)
                {
                    d2[srcAreaNumber] = targetId;
                }
                // else keep existing mapping
            }
            else
            {
                d2[srcAreaNumber] = targetId;
            }
        }
        if (isVia060 && targetId > 0 && tgtMapIdX >= 0)
        {
            if (!_byTgtMapXVia.TryGetValue(tgtMapIdX, out var d2v)) { d2v = new Dictionary<int, int>(); _byTgtMapXVia[tgtMapIdX] = d2v; }
            if (d2v.TryGetValue(srcAreaNumber, out var existingTidV))
            {
                var existingIsChildV = _tgtParentById.TryGetValue(existingTidV, out var pv) && pv > 0;
                var newIsChildV = tgtParentId.HasValue && tgtParentId.Value > 0;
                if (newIsChildV && !existingIsChildV)
                {
                    d2v[srcAreaNumber] = targetId;
                }
            }
            else
            {
                d2v[srcAreaNumber] = targetId;
            }
        }
        if (!string.IsNullOrWhiteSpace(tgtMapNameX))
        {
            if (!_byTgtNameX.TryGetValue(tgtMapNameX, out var d3)) { d3 = new Dictionary<int, int>(); _byTgtNameX[tgtMapNameX] = d3; }
            if (d3.TryGetValue(srcAreaNumber, out var existingTid2))
            {
                var existingIsChild2 = _tgtParentById.TryGetValue(existingTid2, out var p2) && p2 > 0;
                var newIsChild2 = tgtParentId.HasValue && tgtParentId.Value > 0;
                if (newIsChild2 && !existingIsChild2)
                {
                    d3[srcAreaNumber] = targetId;
                }
                // else keep existing mapping
            }
            else
            {
                d3[srcAreaNumber] = targetId;
            }
        }
        if (isVia060 && !string.IsNullOrWhiteSpace(tgtMapNameX))
        {
            if (!_byTgtNameXVia.TryGetValue(tgtMapNameX, out var d3v)) { d3v = new Dictionary<int, int>(); _byTgtNameXVia[tgtMapNameX] = d3v; }
            if (d3v.TryGetValue(srcAreaNumber, out var existingTid2v))
            {
                var existingIsChild2v = _tgtParentById.TryGetValue(existingTid2v, out var p2v) && p2v > 0;
                var newIsChild2v = tgtParentId.HasValue && tgtParentId.Value > 0;
                if (newIsChild2v && !existingIsChild2v)
                {
                    d3v[srcAreaNumber] = targetId;
                }
            }
            else
            {
                d3v[srcAreaNumber] = targetId;
            }
        }
        // Keep parent cache up to date if provided
        if (tgtParentId.HasValue && tgtParentId.Value > 0)
        {
            _tgtParentById[targetId] = tgtParentId.Value;
        }

        var entry = new ZoneEntry(targetId, tgtParentId ?? 0, tgtMapIdX, isVia060);
        if (subLo == 0)
        {
            // Zone-level entry
            if (!_zoneByBase.TryGetValue(zoneBase, out var existing))
            {
                _zoneByBase[zoneBase] = entry;
            }
            else if (ShouldReplace(existing, entry))
            {
                _zoneByBase[zoneBase] = entry;
            }
        }
        else
        {
            if (!_subByZoneBase.TryGetValue(zoneBase, out var dict))
            {
                dict = new Dictionary<int, ZoneEntry>();
                _subByZoneBase[zoneBase] = dict;
            }
            if (!dict.TryGetValue(subLo, out var existingSub) || ShouldReplace(existingSub, entry))
            {
                dict[subLo] = entry;
            }
        }
    }

    private static bool MatchesMap(ZoneEntry entry, int? mapIdHint)
    {
        if (!mapIdHint.HasValue || mapIdHint.Value < 0) return true;
        if (entry.TargetMapId < 0) return false;
        return entry.TargetMapId == mapIdHint.Value;
    }

    private static bool ShouldReplace(ZoneEntry existing, ZoneEntry candidate)
    {
        if (candidate.TargetId <= 0) return false;
        if (existing.TargetId <= 0) return true;
        if (!existing.Via060 && candidate.Via060) return true;
        if (existing.Via060 == candidate.Via060)
        {
            // Prefer child over parent
            if (!existing.IsChild && candidate.IsChild) return true;
        }
        return false;
    }

    // No LK dump or heuristic resolution in strict mode

    public bool TryGetTargetName(int targetId, out string name)
    {
        if (_tgtNameById.TryGetValue(targetId, out name!)) return true;
        name = string.Empty;
        return false;
    }

    public bool TryGetTargetParentId(int targetId, out int parentId)
    {
        if (_tgtParentById.TryGetValue(targetId, out parentId)) return true;
        parentId = 0; return false;
    }

    // Diagnostics / introspection
    public int GlobalCount => _global.Count;
    public int PerMapCount => _bySrcMap.Sum(kv => kv.Value.Count);
    public IReadOnlyCollection<int> MapIds => _bySrcMap.Keys.ToList().AsReadOnly();
    public int CountByName(string mapName) => string.IsNullOrWhiteSpace(mapName) ? 0 : (_bySrcName.TryGetValue(mapName, out var d) ? d.Count : 0);
    public int CountByTargetMap(int mapId) => _byTgtMapX.TryGetValue(mapId, out var d) ? d.Count : 0;

    private static string[] SplitCsv(string line)
    {
        // Simple CSV splitter handling quotes
        var list = new List<string>();
        bool inQuotes = false;
        var cur = new System.Text.StringBuilder();
        for (int i = 0; i < line.Length; i++)
        {
            char ch = line[i];
            if (inQuotes)
            {
                if (ch == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { cur.Append('"'); i++; }
                    else { inQuotes = false; }
                }
                else cur.Append(ch);
            }
            else
            {
                if (ch == ',') { list.Add(cur.ToString()); cur.Clear(); }
                else if (ch == '"') inQuotes = true;
                else cur.Append(ch);
            }
        }
        list.Add(cur.ToString());
        return list.ToArray();
    }
}
