using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace AlphaWdtAnalyzer.Core.Export;

/// <summary>
/// Loads DBCTool.V2 patch CSVs (Area_patch_crosswalk_*.csv) and provides fast lookup
/// from (src_mapId, src_areaNumber) to target 3.3.5 AreaID.
/// </summary>
public sealed class DbcPatchMapping
{
    private readonly Dictionary<int, Dictionary<int, int>> _bySrcMap = new(); // src_mapId -> (src_areaNumber -> tgt_areaID)
    private readonly Dictionary<int, int> _global = new(); // src_areaNumber -> tgt_areaID (from global files)

    public void LoadFile(string csvPath)
    {
        if (!File.Exists(csvPath)) return;
        using var sr = new StreamReader(csvPath);
        string? header = sr.ReadLine();
        int srcMapIdx = -1, srcAreaIdx = -1, tgtIdIdx = -1;
        bool hasHeader = false;
        if (header is not null)
        {
            var cols = SplitCsv(header);
            for (int i = 0; i < cols.Length; i++)
            {
                var c = cols[i].Trim().ToLowerInvariant();
                if (c == "src_mapid") srcMapIdx = i;
                else if (c == "src_areanumber") srcAreaIdx = i;
                else if (c == "tgt_areaid") tgtIdIdx = i;
            }
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

                if (srcAreaIdx < 0 || srcAreaIdx >= cells.Length) continue;
                if (tgtIdIdx < 0 || tgtIdIdx >= cells.Length) continue;
                if (!int.TryParse(cells[srcAreaIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out var sArea)) continue;
                if (!int.TryParse(cells[tgtIdIdx], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tId)) continue;
                Insert(sMap, sArea, tId);
            }
            else
            {
                // Fallback: try to parse as: src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name
                var cells2 = cells;
                if (cells2.Length >= 9)
                {
                    int.TryParse(cells2[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var sMap);
                    if (!int.TryParse(cells2[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var sArea)) continue;
                    if (!int.TryParse(cells2[7], NumberStyles.Integer, CultureInfo.InvariantCulture, out var tId)) continue;
                    Insert(sMap, sArea, tId);
                }
            }
        }
    }

    public bool TryMap(int srcMapId, int srcAreaNumber, out int targetId)
    {
        // Prefer map-specific
        if (_bySrcMap.TryGetValue(srcMapId, out var dict) && dict.TryGetValue(srcAreaNumber, out targetId))
            return true;
        // Fallback to global mapping
        if (_global.TryGetValue(srcAreaNumber, out targetId)) return true;
        targetId = 0;
        return false;
    }

    private void Insert(int srcMapId, int srcAreaNumber, int targetId)
    {
        if (srcMapId >= 0)
        {
            if (!_bySrcMap.TryGetValue(srcMapId, out var dict)) { dict = new Dictionary<int, int>(); _bySrcMap[srcMapId] = dict; }
            dict[srcAreaNumber] = targetId;
        }
        else
        {
            _global[srcAreaNumber] = targetId;
        }
    }

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
