using System.Globalization;
using System.Text;
using DBCD;
using DBCTool.V2.Domain;
using static DBCTool.V2.IO.DbdcHelper;

namespace DBCTool.V2.AlphaDecode;

public sealed class AlphaDecoderV2 : IAlphaDecoder
{
    public (Dictionary<(int cont, int zoneBase), ZoneRec> ZoneIndex,
            Dictionary<(int cont, int zoneBase, int subLo), SubRec> SubIndex,
            Dictionary<int, int> ZoneOwner)
        BuildIndices(IDBCDStorage storSrcArea, string srcAlias)
    {
        string parentCol = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "ParentAreaNum" : "ParentAreaID";
        string keyCol = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID";
        string nameCol = FirstNonEmpty("AreaName_lang", "AreaName", "Name");

        var zoneNameCounts = new Dictionary<(int cont, int zoneBase), Dictionary<string, (int validated, int total)>>( );
        var subNameCounts  = new Dictionary<(int cont, int zoneBase, int subLo), Dictionary<string, int>>( );

        foreach (var key in storSrcArea.Keys)
        {
            var row = storSrcArea[key];
            int areaNum = SafeField<int>(row, keyCol);
            if (areaNum <= 0) continue;
            int parentNum = SafeField<int>(row, parentCol);
            int cont = SafeField<int>(row, "ContinentID");
            string name = FirstNonEmpty(SafeField<string>(row, nameCol)) ?? string.Empty;

            int area_hi16 = (areaNum >> 16) & 0xFFFF;
            int area_lo16 = areaNum & 0xFFFF;
            int zoneBase = area_hi16 << 16;
            bool parent_ok = (area_lo16 == 0) || (parentNum == zoneBase);

            if (area_lo16 == 0)
            {
                var k = (cont, zoneBase);
                if (!zoneNameCounts.TryGetValue(k, out var dict)) { dict = new Dictionary<string, (int validated, int total)>(StringComparer.OrdinalIgnoreCase); zoneNameCounts[k] = dict; }
                if (!dict.TryGetValue(name, out var c)) c = (0, 0);
                c.total++;
                if (parent_ok) c.validated++;
                dict[name] = c;
            }
            else if (parent_ok)
            {
                var k = (cont, zoneBase, area_lo16);
                if (!subNameCounts.TryGetValue(k, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); subNameCounts[k] = dict; }
                dict[name] = dict.TryGetValue(name, out var cnt) ? (cnt + 1) : 1;
            }
        }

        var ZoneIndex = new Dictionary<(int cont, int zoneBase), ZoneRec>();
        foreach (var kv in zoneNameCounts)
        {
            string bestName = string.Empty; int bestVal = -1; int bestTot = -1;
            foreach (var nkv in kv.Value)
            {
                var (v, t) = nkv.Value;
                if (v > bestVal || (v == bestVal && t > bestTot)) { bestName = nkv.Key; bestVal = v; bestTot = t; }
            }
            if (string.IsNullOrWhiteSpace(bestName) && kv.Value.Count > 0) bestName = kv.Value.Keys.First();
            ZoneIndex[kv.Key] = new ZoneRec(bestName, bestVal < 0 ? 0 : bestVal, bestTot < 0 ? 0 : bestTot);
        }

        var SubIndex = new Dictionary<(int cont, int zoneBase, int subLo), SubRec>();
        foreach (var kv in subNameCounts)
        {
            string bestName = string.Empty; int bestCnt = -1;
            foreach (var nkv in kv.Value)
            {
                if (nkv.Value > bestCnt) { bestCnt = nkv.Value; bestName = nkv.Key; }
            }
            if (!string.IsNullOrWhiteSpace(bestName)) SubIndex[kv.Key] = new SubRec(bestName);
        }

        // Zone ownership per zoneBase across continents: choose highest validated, tie-break by total
        var ZoneOwner = new Dictionary<int, int>();
        var ownerTallies = new Dictionary<int, List<(int cont, int validated, int total)>>();
        foreach (var kv in ZoneIndex)
        {
            int zoneBase = kv.Key.zoneBase; int cont = kv.Key.cont;
            if (!ownerTallies.TryGetValue(zoneBase, out var list)) { list = new List<(int, int, int)>(); ownerTallies[zoneBase] = list; }
            list.Add((cont, kv.Value.ValidatedCount, kv.Value.TotalCount));
        }
        foreach (var kv in ownerTallies)
        {
            int bestCont = -1; int bestVal = -1; int bestTot = -1;
            foreach (var rec in kv.Value)
            {
                if (rec.validated > bestVal || (rec.validated == bestVal && rec.total > bestTot))
                { bestCont = rec.cont; bestVal = rec.validated; bestTot = rec.total; }
            }
            if (bestCont >= 0) ZoneOwner[kv.Key] = bestCont;
        }

        return (ZoneIndex, SubIndex, ZoneOwner);
    }

    public void WriteAuditCSVs(IDBCDStorage storSrcArea, string compareDir, string srcAlias)
    {
        Directory.CreateDirectory(compareDir);
        var decodePath = Path.Combine(compareDir, "alpha_areaid_decode_v2.csv");
        var anomaliesPath = Path.Combine(compareDir, "alpha_areaid_anomalies.csv");

        var (ZoneIndex, SubIndex, ZoneOwner) = BuildIndices(storSrcArea, srcAlias);

        var sb = new StringBuilder();
        sb.AppendLine("alpha_raw,alpha_raw_hex,cont,area_hi16,area_lo16,parent_hi16,parent_lo16,zone_base_hex,zone_name,sub_lo16,sub_name,parent_ok");
        var emitted = new HashSet<(int cont, int areaNum)>();

        string parentCol = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "ParentAreaNum" : "ParentAreaID";
        string keyCol = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID";
        foreach (var key in storSrcArea.Keys)
        {
            var row = storSrcArea[key];
            int areaNum = SafeField<int>(row, keyCol);
            if (areaNum <= 0) continue;
            int parentNum = SafeField<int>(row, parentCol);
            int cont = SafeField<int>(row, "ContinentID");
            if (!emitted.Add((cont, areaNum))) continue;

            int area_hi16 = (areaNum >> 16) & 0xFFFF;
            int area_lo16 = areaNum & 0xFFFF;
            int parent_hi16 = (parentNum >> 16) & 0xFFFF;
            int parent_lo16 = parentNum & 0xFFFF;
            int zoneBase = area_hi16 << 16;
            bool parent_ok = (area_lo16 == 0) || (parentNum == zoneBase);

            string zoneName = ZoneIndex.TryGetValue((cont, zoneBase), out var zr) ? zr.ZoneName ?? string.Empty : string.Empty;
            string subName = (area_lo16 > 0 && SubIndex.TryGetValue((cont, zoneBase, area_lo16), out var sr)) ? sr.SubName ?? string.Empty : string.Empty;

            sb.AppendLine(string.Join(',', new[]
            {
                areaNum.ToString(CultureInfo.InvariantCulture),
                $"0x{areaNum:X8}",
                cont.ToString(CultureInfo.InvariantCulture),
                area_hi16.ToString(CultureInfo.InvariantCulture),
                area_lo16.ToString(CultureInfo.InvariantCulture),
                parent_hi16.ToString(CultureInfo.InvariantCulture),
                parent_lo16.ToString(CultureInfo.InvariantCulture),
                $"0x{zoneBase:X8}",
                Csv(zoneName ?? string.Empty),
                area_lo16.ToString(CultureInfo.InvariantCulture),
                Csv(subName ?? string.Empty),
                parent_ok ? "1" : "0"
            }));
        }
        File.WriteAllText(decodePath, sb.ToString(), new UTF8Encoding(true));

        var an = new StringBuilder();
        an.AppendLine("zone_base_hex,owner_cont,other_cont");
        var seen = new HashSet<string>();
        foreach (var z in ZoneIndex.Keys)
        {
            int zb = z.zoneBase;
            if (!ZoneOwner.TryGetValue(zb, out var owner)) continue;
            foreach (var alt in ZoneIndex.Keys.Where(k => k.zoneBase == zb && k.cont != owner))
            {
                // Skip expected churn between continents 0 and 1
                if ((owner == 0 && alt.cont == 1) || (owner == 1 && alt.cont == 0)) continue;
                var keyStr = $"{zb}:{owner}:{alt.cont}";
                if (seen.Add(keyStr))
                    an.AppendLine(string.Join(',', new[] { $"0x{zb:X8}", owner.ToString(CultureInfo.InvariantCulture), alt.cont.ToString(CultureInfo.InvariantCulture) }));
            }
        }
        File.WriteAllText(anomaliesPath, an.ToString(), new UTF8Encoding(true));
    }
}
