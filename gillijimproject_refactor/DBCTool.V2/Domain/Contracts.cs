using DBCD;

namespace DBCTool.V2.Domain;

public interface IAlphaDecoder
{
    (Dictionary<(int cont, int zoneBase), ZoneRec> ZoneIndex,
     Dictionary<(int cont, int zoneBase, int subLo), SubRec> SubIndex,
     Dictionary<int, int> ZoneOwner) BuildIndices(IDBCDStorage storSrcArea, string srcAlias);

    void WriteAuditCSVs(IDBCDStorage storSrcArea, string compareDir, string srcAlias);
}

public interface IMapCrosswalk
{
    Dictionary<int, int> Build053To335(IDBCDStorage srcMap, IDBCDStorage tgtMap);
}

public interface IAreaMatcher
{
    // Map-locked exact chain match; returns target leaf ID or -1
    int TryMatchChainExact(int mapIdOnTgt, List<string> srcChain,
        Dictionary<int, Dictionary<string, int>> idxTgtTopZonesByMap,
        Dictionary<int, Dictionary<string, int>> idxTgtChildrenByZone,
        out int matchedDepth);
}

public interface IDbcdProvider
{
    IDBCDStorage Load(string table, string build, string dir, DBCD.Locale locale);
}
