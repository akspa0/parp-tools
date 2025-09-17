using DBCTool.V2.Domain;

namespace DBCTool.V2.Mapping;

public sealed class AreaMatcher : IAreaMatcher
{
    public int TryMatchChainExact(
        int mapIdOnTgt,
        List<string> srcChain,
        Dictionary<int, Dictionary<string, int>> idxTgtTopZonesByMap,
        Dictionary<int, Dictionary<string, int>> idxTgtChildrenByZone,
        out int matchedDepth)
    {
        matchedDepth = 0;
        if (srcChain is null || srcChain.Count == 0) return -1;
        if (!idxTgtTopZonesByMap.TryGetValue(mapIdOnTgt, out var zones)) return -1;

        string first = DBCTool.V2.IO.DbdcHelper.NormKey(srcChain[0]);
        if (!zones.TryGetValue(first, out var cur)) return -1;
        matchedDepth = 1;

        for (int i = 1; i < srcChain.Count; i++)
        {
            var name = DBCTool.V2.IO.DbdcHelper.NormKey(srcChain[i]);
            if (idxTgtChildrenByZone.TryGetValue(cur, out var kids) && kids.TryGetValue(name, out var next))
            {
                cur = next;
                matchedDepth++;
            }
            else
            {
                break;
            }
        }
        return cur;
    }
}
