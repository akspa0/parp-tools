using DBCD;
using DBCTool.V2.Domain;
using static DBCTool.V2.IO.DbdcHelper;

namespace DBCTool.V2.Crosswalk;

public sealed class MapCrosswalkService : IMapCrosswalk
{
    public Dictionary<int, int> Build053To335(IDBCDStorage srcMap, IDBCDStorage tgtMap)
    {
        var result = new Dictionary<int, int>();

        string idColSrc = DetectIdColumn(srcMap);
        string idColTgt = DetectIdColumn(tgtMap);

        // Build target indices
        var dirIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var nameIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var tgtIds = new HashSet<int>();
        foreach (var k in tgtMap.Keys)
        {
            var row = tgtMap[k];
            int id = !string.IsNullOrWhiteSpace(idColTgt) ? SafeField<int>(row, idColTgt) : k;
            var dirTok = DirToken(SafeField<string>(row, "Directory"));
            if (!string.IsNullOrWhiteSpace(dirTok) && !dirIndex.ContainsKey(dirTok)) dirIndex[dirTok] = id;
            var name = NormKey(FirstNonEmpty(SafeField<string>(row, "MapName_lang"), SafeField<string>(row, "MapName"), SafeField<string>(row, "InternalName"), dirTok));
            if (!string.IsNullOrWhiteSpace(name) && !nameIndex.ContainsKey(name)) nameIndex[name] = id;
            tgtIds.Add(id);
        }

        // Walk sources and match
        var srcIds = new HashSet<int>();
        foreach (var k in srcMap.Keys)
        {
            var row = srcMap[k];
            int id = !string.IsNullOrWhiteSpace(idColSrc) ? SafeField<int>(row, idColSrc) : k;
            srcIds.Add(id);
            var dirTok = DirToken(SafeField<string>(row, "Directory"));
            var name = FirstNonEmpty(SafeField<string>(row, "MapName_lang"), SafeField<string>(row, "MapName"), SafeField<string>(row, "InternalName"), dirTok) ?? string.Empty;

            int tgt = -1;
            if (!string.IsNullOrWhiteSpace(dirTok) && dirIndex.TryGetValue(dirTok, out var byDir)) tgt = byDir;
            else
            {
                var key = NormKey(name);
                if (!string.IsNullOrWhiteSpace(key) && nameIndex.TryGetValue(key, out var byName)) tgt = byName;
            }
            if (tgt >= 0) result[id] = tgt;
        }

        // Preseed continent mappings when present
        if (!result.ContainsKey(0) && srcIds.Contains(0) && tgtIds.Contains(0)) result[0] = 0;
        if (!result.ContainsKey(1) && srcIds.Contains(1) && tgtIds.Contains(1)) result[1] = 1;

        return result;
    }
}
