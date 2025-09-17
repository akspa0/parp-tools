using DBCD;
using DBCD.Providers;
using DBCTool.V2.Domain;
using DBCTool.V2.IO;
using DBCTool.V2.AlphaDecode;
using DBCTool.V2.Crosswalk;
using DBCTool.V2.Mapping;
using System;
using System.Text;
using System.Globalization;
using System.Linq;
using static DBCTool.V2.IO.DbdcHelper;

namespace DBCTool.V2.Cli;

internal sealed class CompareAreaV2Command
{
    public int Run(string dbdDir, string outBase, string localeStr, List<(string build, string dir)> inputs)
    {
        // Resolve required inputs
        string? dir053 = null, dir055 = null, dir060 = null, dir335 = null;
        foreach (var (build, dir) in inputs)
        {
            var alias = ResolveAliasOrInfer(build, dir);
            if (alias == "0.5.3") dir053 = Normalize(dir);
            else if (alias == "0.5.5") dir055 = Normalize(dir);
            else if (alias == "0.6.0") dir060 = Normalize(dir);
            else if (alias == "3.3.5") dir335 = Normalize(dir);
        }
        if (string.IsNullOrEmpty(dir335))
        {
            Console.Error.WriteLine("[V2] ERROR: 3.3.5 input is required.");
            return 2;
        }
        var srcDir = dir053 ?? dir055 ?? dir060;
        var srcAlias = dir053 != null ? "0.5.3" : (dir055 != null ? "0.5.5" : (dir060 != null ? "0.6.0" : ""));
        if (string.IsNullOrEmpty(srcDir))
        {
            Console.Error.WriteLine("[V2] ERROR: One of 0.5.3/0.5.5/0.6.0 inputs is required.");
            return 2;
        }

        // Output folders
        var compareDir = Path.Combine(outBase, "compare");
        var compareV2Dir = Path.Combine(compareDir, "v2");
        Directory.CreateDirectory(compareDir);
        Directory.CreateDirectory(compareV2Dir);

        // Load storages
        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var locale = ParseLocale(localeStr);

        var storSrc_Area = DbdcHelper.LoadTable("AreaTable", CanonicalizeBuild(srcAlias), srcDir!, dbdProvider, locale);
        var storSrc_Map  = DbdcHelper.LoadTable("Map",       CanonicalizeBuild(srcAlias), srcDir!, dbdProvider, locale);
        var storTgt_Area = DbdcHelper.LoadTable("AreaTable", CanonicalizeBuild("3.3.5"),  dir335!, dbdProvider, locale);
        var storTgt_Map  = DbdcHelper.LoadTable("Map",       CanonicalizeBuild("3.3.5"),  dir335!, dbdProvider, locale);

        // Build source indices for zone-only resolution
        // (contRaw, zoneBase) -> zone row (lo16==0)
        string keyColSrcIdx = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID";
        var idxSrcZoneByCont = new Dictionary<(int cont, int zoneBase), DBCDRow>();
        foreach (var sid in storSrc_Area.Keys)
        {
            var srow = storSrc_Area[sid];
            int areaNumIdx = SafeField<int>(srow, keyColSrcIdx);
            if (areaNumIdx <= 0) continue;
            int contIdx = SafeField<int>(srow, "ContinentID");
            int hi = (areaNumIdx >> 16) & 0xFFFF;
            int lo = areaNumIdx & 0xFFFF;
            int zb = hi << 16;
            if (lo == 0)
            {
                var k = (contIdx, zb);
                if (!idxSrcZoneByCont.ContainsKey(k)) idxSrcZoneByCont[k] = srow;
            }
        }

        // Crosswalk service (needed for optional pivot)
        var crosswalk = new MapCrosswalkService();

        // Optional 0.6.0 pivot
        bool has060 = !string.IsNullOrWhiteSpace(dir060) && srcAlias != "0.6.0";
        IDBCDStorage? stor060_Area = null;
        IDBCDStorage? stor060_Map = null;
        var idx060TopZonesByMap = new Dictionary<int, Dictionary<string, int>>();
        var idx060ChildrenByZone = new Dictionary<int, Dictionary<string, int>>();
        var id060ToRow = new Dictionary<int, DBCDRow>();
        string areaNameCol060 = string.Empty;
        string parentCol060 = string.Empty;
        var map060Names = new Dictionary<int, string>();
        var cwSrcTo060 = new Dictionary<int, int>();
        var cw060To335 = new Dictionary<int, int>();
        if (has060)
        {
            stor060_Area = DbdcHelper.LoadTable("AreaTable", CanonicalizeBuild("0.6.0"),  dir060!, dbdProvider, locale);
            stor060_Map  = DbdcHelper.LoadTable("Map",       CanonicalizeBuild("0.6.0"),  dir060!, dbdProvider, locale);
            string idColArea060 = DetectIdColumn(stor060_Area);
            areaNameCol060 = DetectColumn(stor060_Area, "AreaName_lang", "AreaName", "Name");
            parentCol060 = DetectColumn(stor060_Area, "ParentAreaID", "ParentAreaNum");
            foreach (var key in stor060_Area.Keys)
            {
                var row = stor060_Area[key];
                int id = !string.IsNullOrWhiteSpace(idColArea060) ? SafeField<int>(row, idColArea060) : key;
                string name = FirstNonEmpty(SafeField<string>(row, areaNameCol060)) ?? string.Empty;
                int parentId = SafeField<int>(row, parentCol060);
                if (parentId <= 0) parentId = id;
                int mapId = SafeField<int>(row, "ContinentID");
                id060ToRow[id] = row;
                if (parentId == id)
                {
                    if (!idx060TopZonesByMap.TryGetValue(mapId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idx060TopZonesByMap[mapId] = dict; }
                    dict[NormKey(name)] = id;
                }
                else
                {
                    if (!idx060ChildrenByZone.TryGetValue(parentId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idx060ChildrenByZone[parentId] = dict; }
                    dict[NormKey(name)] = id;
                }
            }
            cwSrcTo060 = crosswalk.Build053To335(storSrc_Map, stor060_Map);
            cw060To335 = crosswalk.Build053To335(stor060_Map, storTgt_Map);
            map060Names = BuildMapNames(stor060_Map);
        }

        // Alpha decode V2: build indices and write audit CSVs
        var decoder = new AlphaDecoderV2();
        var (ZoneIndex, SubIndex, ZoneOwner) = decoder.BuildIndices(storSrc_Area, srcAlias);
        decoder.WriteAuditCSVs(storSrc_Area, compareDir, srcAlias);

        // Crosswalk and matcher
        var matcher = new AreaMatcher();

        // Build LK indices (top-level zones per map, children per zone)
        string idColAreaTgt = DetectIdColumn(storTgt_Area);
        var idxTgtTopZonesByMap = new Dictionary<int, Dictionary<string, int>>();
        var idxTgtChildrenByZone = new Dictionary<int, Dictionary<string, int>>();
        var tgtIdToRow = new Dictionary<int, DBCDRow>();
        string areaNameColTgt = DetectColumn(storTgt_Area, "AreaName_lang", "AreaName", "Name");
        foreach (var key in storTgt_Area.Keys)
        {
            var row = storTgt_Area[key];
            int id = !string.IsNullOrWhiteSpace(idColAreaTgt) ? SafeField<int>(row, idColAreaTgt) : key;
            string name = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
            int parentId = SafeField<int>(row, "ParentAreaID");
            if (parentId <= 0) parentId = id;
            int mapId = SafeField<int>(row, "ContinentID");
            tgtIdToRow[id] = row;
            if (parentId == id)
            {
                if (!idxTgtTopZonesByMap.TryGetValue(mapId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idxTgtTopZonesByMap[mapId] = dict; }
                dict[NormKey(name)] = id;
            }
            else
            {
                if (!idxTgtChildrenByZone.TryGetValue(parentId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idxTgtChildrenByZone[parentId] = dict; }
                dict[NormKey(name)] = id;
            }
        }

        // Build global unique indices across LK for rename matching
        var idxTgtTopGlobal = new Dictionary<string, (int areaId, int mapId)>(StringComparer.OrdinalIgnoreCase);
        var ambTop = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var idxTgtChildGlobal = new Dictionary<string, (int areaId, int mapId)>(StringComparer.OrdinalIgnoreCase);
        var ambChild = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        // Also collect full lists for fuzzy rename
        var lkTopList = new List<(string key, int id, int map)>();
        var lkChildList = new List<(string key, int id, int map)>();
        foreach (var key in storTgt_Area.Keys)
        {
            var row = storTgt_Area[key];
            int id = !string.IsNullOrWhiteSpace(idColAreaTgt) ? SafeField<int>(row, idColAreaTgt) : key;
            string name = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
            int parentId = SafeField<int>(row, "ParentAreaID");
            if (parentId <= 0) parentId = id;
            int mapId = SafeField<int>(row, "ContinentID");
            var nk = NormKey(name);
            if (string.IsNullOrWhiteSpace(nk)) continue;
            if (parentId == id)
            {
                if (!ambTop.Contains(nk))
                {
                    if (!idxTgtTopGlobal.TryGetValue(nk, out var ex)) idxTgtTopGlobal[nk] = (id, mapId);
                    else if (ex.areaId != id) { idxTgtTopGlobal.Remove(nk); ambTop.Add(nk); }
                }
                lkTopList.Add((nk, id, mapId));
            }
            else
            {
                if (!ambChild.Contains(nk))
                {
                    if (!idxTgtChildGlobal.TryGetValue(nk, out var ex)) idxTgtChildGlobal[nk] = (id, mapId);
                    else if (ex.areaId != id) { idxTgtChildGlobal.Remove(nk); ambChild.Add(nk); }
                }
                lkChildList.Add((nk, id, mapId));
            }
        }

        // Crosswalk 0.5.x → 3.3.5
        var cw053To335 = crosswalk.Build053To335(storSrc_Map, storTgt_Map);

        // Map names
        var mapSrcNames = BuildMapNames(storSrc_Map);
        var mapTgtNames = BuildMapNames(storTgt_Map);

        // Prepare CSV builders
        var mapping = new StringBuilder();
        var unmatched = new StringBuilder();
        var patch = new StringBuilder();
        var header = string.Join(',', new[]
        {
            "src_row_id","src_areaNumber","src_parentNumber","src_name","src_mapId","src_mapName","src_mapId_xwalk","src_mapName_xwalk","src_path",
            "tgt_id_335","tgt_name","tgt_parent_id","tgt_parent_name","tgt_mapId","tgt_mapName","tgt_path","match_method"
        });
        mapping.AppendLine(header);
        unmatched.AppendLine(header);
        var patchHeader = "src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name";
        patch.AppendLine(patchHeader);
        var patchFallback = new StringBuilder();
        patchFallback.AppendLine(patchHeader);
        var perMap = new Dictionary<int, (StringBuilder map, StringBuilder un, StringBuilder patch)>();
        var perMapFallback = new Dictionary<int, StringBuilder>();

        // Source columns
        string parentColSrc = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "ParentAreaNum" : "ParentAreaID";
        string keyColSrc = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID";
        string areaNameColSrc = DetectColumn(storSrc_Area, "AreaName_lang", "AreaName", "Name");

        // Diagnostics: track missing zones per target map
        var zoneMissing = new Dictionary<int, HashSet<string>>();

        foreach (var key in storSrc_Area.Keys)
        {
            var row = storSrc_Area[key];
            string nm = FirstNonEmpty(SafeField<string>(row, areaNameColSrc)) ?? string.Empty;
            int areaNum = SafeField<int>(row, keyColSrc);
            int parentNum = SafeField<int>(row, parentColSrc);
            if (parentNum <= 0) parentNum = areaNum;
            int contRaw = SafeField<int>(row, "ContinentID");

            int area_hi16 = (areaNum >> 16) & 0xFFFF;
            int area_lo16 = areaNum & 0xFFFF;
            int zoneBase = area_hi16 << 16;

            // Lock to the source row's continent
            int contResolved = contRaw;
            int mapIdX = -1; bool hasMapX = false;
            if (cw053To335.TryGetValue(contResolved, out var mx)) { mapIdX = mx; hasMapX = true; }
            else if (mapTgtNames.ContainsKey(contResolved)) { mapIdX = contResolved; hasMapX = true; }

            // Compose zone-only chain: [zoneName] using the zone row on the same continent
            var chain = new List<string>();
            if (area_lo16 == 0)
            {
                if (!string.IsNullOrWhiteSpace(nm)) chain.Add(NormKey(nm));
            }
            else
            {
                if (idxSrcZoneByCont.TryGetValue((contResolved, zoneBase), out var zrow))
                {
                    string zname = FirstNonEmpty(SafeField<string>(zrow, areaNameColSrc)) ?? string.Empty;
                    if (!string.IsNullOrWhiteSpace(zname)) chain.Add(NormKey(zname));
                }
                if (chain.Count == 0 && !string.IsNullOrWhiteSpace(nm)) chain.Add(NormKey(nm));
            }

            string mapName = mapSrcNames.TryGetValue(contResolved, out var mnSrc) ? mnSrc : string.Empty;
            string mapNameX = hasMapX && mapTgtNames.TryGetValue(mapIdX, out var mnTgt) ? mnTgt : string.Empty;
            var chainPath = string.Join('/', chain);
            string path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{chainPath}" : chainPath;

            int chosen = -1; int depth = 0; string method = string.Empty;
            if (hasMapX && chain.Count > 0)
            {
                chosen = matcher.TryMatchChainExact(mapIdX, chain, idxTgtTopZonesByMap, idxTgtChildrenByZone, out depth);
            }

            // Pivot via 0.6.0 if direct match is missing or partial
            if ((chosen < 0 || depth < chain.Count) && has060 && chain.Count > 0)
            {
                int pivotMapIdX = -1; bool hasPivot = false;
                if (cwSrcTo060.TryGetValue(contResolved, out var mx060)) { pivotMapIdX = mx060; hasPivot = true; }
                else if (map060Names.ContainsKey(contResolved)) { pivotMapIdX = contResolved; hasPivot = true; }
                if (hasPivot)
                {
                    int chosen060 = matcher.TryMatchChainExact(pivotMapIdX, chain, idx060TopZonesByMap, idx060ChildrenByZone, out var depth060);
                    if (chosen060 >= 0)
                    {
                        var pivotRow = id060ToRow[chosen060];
                        int pParent = SafeField<int>(pivotRow, parentCol060);
                        if (pParent <= 0) pParent = chosen060;
                        int zoneId060 = pParent == chosen060 ? chosen060 : pParent;
                        var zoneRow060 = id060ToRow[zoneId060];
                        string zoneName060 = FirstNonEmpty(SafeField<string>(zoneRow060, areaNameCol060)) ?? string.Empty;
                        string subName060 = (pParent != chosen060) ? (FirstNonEmpty(SafeField<string>(pivotRow, areaNameCol060)) ?? string.Empty) : string.Empty;
                        var pivotChain = new List<string>();
                        if (!string.IsNullOrWhiteSpace(zoneName060)) pivotChain.Add(NormKey(zoneName060));
                        if (!string.IsNullOrWhiteSpace(subName060)) pivotChain.Add(NormKey(subName060));

                        int lkMapIdX = -1; bool hasLk = false;
                        if (cw060To335.TryGetValue(pivotMapIdX, out var m335)) { lkMapIdX = m335; hasLk = true; }
                        else if (mapTgtNames.ContainsKey(pivotMapIdX)) { lkMapIdX = pivotMapIdX; hasLk = true; }
                        if (hasLk && pivotChain.Count > 0)
                        {
                            int chosenLK = matcher.TryMatchChainExact(lkMapIdX, pivotChain, idxTgtTopZonesByMap, idxTgtChildrenByZone, out var depthLK);
                            if (chosenLK >= 0)
                            {
                                chosen = chosenLK; depth = depthLK; mapIdX = lkMapIdX; hasMapX = true;
                                mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn2) ? mn2 : string.Empty;
                                path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{chainPath}" : chainPath;
                                method = (depth == pivotChain.Count) ? "pivot_060" : "pivot_060_zone_only";
                            }
                        }
                    }
                }
            }

            // Global rename match: unique top-level zone across LK (cross-map allowed)
            if (chosen < 0 && chain.Count > 0)
            {
                var keyNorm = chain[0];
                if (idxTgtTopGlobal.TryGetValue(keyNorm, out var rec))
                {
                    chosen = rec.areaId; depth = 1; mapIdX = rec.mapId; hasMapX = true;
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn2) ? mn2 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm}" : keyNorm;
                    method = "rename_global";
                }
            }

            // Global rename match for child names (use the current row's own name)
            if (chosen < 0 && !string.IsNullOrWhiteSpace(nm))
            {
                var keyNorm2 = NormKey(nm);
                if (idxTgtChildGlobal.TryGetValue(keyNorm2, out var rec2))
                {
                    chosen = rec2.areaId; depth = 1; mapIdX = rec2.mapId; hasMapX = true;
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn3) ? mn3 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm2}" : keyNorm2;
                    method = "rename_global_child";
                }
            }

            // Fuzzy rename (top-level) if still unmatched
            if (chosen < 0 && chain.Count > 0)
            {
                var keyNorm = chain[0];
                var (fid, fmap, ok) = FindBestFuzzy(keyNorm, lkTopList, 2);
                if (ok)
                {
                    chosen = fid; depth = 1; mapIdX = fmap; hasMapX = true; method = "rename_fuzzy";
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn4) ? mn4 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm}" : keyNorm;
                }
            }

            // Fuzzy rename (child) using row name if still unmatched
            if (chosen < 0 && !string.IsNullOrWhiteSpace(nm))
            {
                var keyNorm = NormKey(nm);
                var (fid, fmap, ok) = FindBestFuzzy(keyNorm, lkChildList, 2);
                if (ok)
                {
                    chosen = fid; depth = 1; mapIdX = fmap; hasMapX = true; method = "rename_fuzzy_child";
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn5) ? mn5 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm}" : keyNorm;
                }
            }

            // Exact-key fallback across LK with map preference (prefer map 0, then 1)
            if (chosen < 0)
            {
                int pref(int m) => (m == 0 ? 0 : (m == 1 ? 1 : 2));
                if (chain.Count > 0)
                {
                    var k = chain[0];
                    var hits = lkTopList.Where(x => x.key.Equals(k, StringComparison.OrdinalIgnoreCase)).OrderBy(x => pref(x.map)).ToList();
                    if (hits.Count > 0)
                    {
                        chosen = hits[0].id; mapIdX = hits[0].map; hasMapX = true; depth = 1; method = "rename_exact_global";
                        mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn6) ? mn6 : string.Empty;
                        path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{k}" : k;
                    }
                }
                if (chosen < 0 && !string.IsNullOrWhiteSpace(nm))
                {
                    var k2 = NormKey(nm);
                    var hitsChild = lkChildList.Where(x => x.key.Equals(k2, StringComparison.OrdinalIgnoreCase)).OrderBy(x => pref(x.map)).ToList();
                    if (hitsChild.Count > 0)
                    {
                        chosen = hitsChild[0].id; mapIdX = hitsChild[0].map; hasMapX = true; depth = 1; method = "rename_exact_global_child";
                        mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn7) ? mn7 : string.Empty;
                        path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{k2}" : k2;
                    }
                }
            }

            // Special-case: ***On Map Dungeon*** -> force target 0 in fallback
            bool isOnMapDungeon = string.Equals(NormKey(nm), "onmapdungeon", StringComparison.OrdinalIgnoreCase);

            if (!isOnMapDungeon && chosen >= 0 && tgtIdToRow.TryGetValue(chosen, out var tRow))
            {
                string tgtName = FirstNonEmpty(SafeField<string>(tRow, areaNameColTgt)) ?? string.Empty;
                int tgtParentId = SafeField<int>(tRow, "ParentAreaID");
                int tgtMap = SafeField<int>(tRow, "ContinentID");
                string tgtMapName = mapTgtNames.TryGetValue(tgtMap, out var mn) ? mn : string.Empty;
                // For patch emission, ignore parent and self-parent the target
                int tgtParentIdOut = chosen;
                string tgtParentName = FirstNonEmpty(SafeField<string>(tRow, areaNameColTgt)) ?? tgtName;
                string tgtPath = $"{Norm(tgtMapName)}/{Norm(tgtName)}";
                if (string.IsNullOrEmpty(method)) method = (depth == chain.Count) ? "exact" : "zone_only";

                var line = string.Join(',', new[]
                {
                    key.ToString(CultureInfo.InvariantCulture),
                    areaNum.ToString(CultureInfo.InvariantCulture),
                    parentNum.ToString(CultureInfo.InvariantCulture),
                    Csv(nm),
                    contResolved.ToString(CultureInfo.InvariantCulture),
                    Csv(mapName),
                    (hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(mapNameX),
                    Csv(path),
                    chosen.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtName),
                    tgtParentId.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtParentName),
                    tgtMap.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtMapName),
                    Csv(tgtPath),
                    method
                });
                mapping.AppendLine(line);
                if (hasMapX)
                {
                    if (!perMap.TryGetValue(mapIdX, out var tuple)) { tuple = (new StringBuilder(), new StringBuilder(), new StringBuilder()); tuple.map.AppendLine(header); tuple.un.AppendLine(header); tuple.patch.AppendLine("src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name"); perMap[mapIdX] = tuple; }
                    tuple.map.AppendLine(line);
                    tuple.patch.AppendLine(string.Join(',', new[]
                    {
                        contResolved.ToString(CultureInfo.InvariantCulture),
                        Csv(mapName),
                        areaNum.ToString(CultureInfo.InvariantCulture),
                        parentNum.ToString(CultureInfo.InvariantCulture),
                        Csv(nm),
                        hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                        Csv(mapNameX),
                        chosen.ToString(CultureInfo.InvariantCulture),
                        tgtParentIdOut.ToString(CultureInfo.InvariantCulture),
                        Csv(tgtName)
                    }));
                }
                // Do not include matched rows in fallback0 CSVs
            }
            else
            {
                var line = string.Join(',', new[]
                {
                    key.ToString(CultureInfo.InvariantCulture),
                    areaNum.ToString(CultureInfo.InvariantCulture),
                    parentNum.ToString(CultureInfo.InvariantCulture),
                    Csv(nm),
                    contResolved.ToString(CultureInfo.InvariantCulture),
                    Csv(mapName),
                    (hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(mapNameX),
                    Csv(path),
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    string.Empty,
                    "unmatched"
                });
                unmatched.AppendLine(line);
                if (hasMapX)
                {
                    if (!perMap.TryGetValue(mapIdX, out var tuple)) { tuple = (new StringBuilder(), new StringBuilder(), new StringBuilder()); tuple.map.AppendLine(header); tuple.un.AppendLine(header); tuple.patch.AppendLine("src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name"); perMap[mapIdX] = tuple; }
                    tuple.un.AppendLine(line);
                    // Fallback0: add a patch row with target areaID=0, parentID=0
                    if (!perMapFallback.TryGetValue(mapIdX, out var pfb)) { pfb = new StringBuilder(); pfb.AppendLine(patchHeader); perMapFallback[mapIdX] = pfb; }
                    pfb.AppendLine(string.Join(',', new[]
                    {
                        contResolved.ToString(CultureInfo.InvariantCulture),
                        Csv(mapName),
                        areaNum.ToString(CultureInfo.InvariantCulture),
                        parentNum.ToString(CultureInfo.InvariantCulture),
                        Csv(nm),
                        hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                        Csv(mapNameX),
                        "0",
                        "0",
                        string.Empty
                    }));
                    // Global fallback builder as well
                    patchFallback.AppendLine(string.Join(',', new[]
                    {
                        contResolved.ToString(CultureInfo.InvariantCulture),
                        Csv(mapName),
                        areaNum.ToString(CultureInfo.InvariantCulture),
                        parentNum.ToString(CultureInfo.InvariantCulture),
                        Csv(nm),
                        hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                        Csv(mapNameX),
                        "0",
                        "0",
                        string.Empty
                    }));
                    if (chain.Count > 0)
                    {
                        var keyNorm = NormKey(chain[0]);
                        if (!zoneMissing.TryGetValue(mapIdX, out var set)) { set = new HashSet<string>(StringComparer.OrdinalIgnoreCase); zoneMissing[mapIdX] = set; }
                        set.Add(keyNorm);
                    }
                }
            }
        }

        // Write outputs
        var mappingPath = Path.Combine(compareV2Dir, $"AreaTable_mapping_{srcAlias}_to_335.csv");
        var unmatchedPath = Path.Combine(compareV2Dir, $"AreaTable_unmatched_{srcAlias}_to_335.csv");
        var patchPath = Path.Combine(compareV2Dir, $"Area_patch_crosswalk_{srcAlias}_to_335.csv");
        var patchFallbackPath = Path.Combine(compareV2Dir, $"Area_patch_with_fallback0_{srcAlias}_to_335.csv");
        File.WriteAllText(mappingPath, mapping.ToString(), new UTF8Encoding(true));
        File.WriteAllText(unmatchedPath, unmatched.ToString(), new UTF8Encoding(true));
        File.WriteAllText(patchPath, patch.ToString(), new UTF8Encoding(true));
        File.WriteAllText(patchFallbackPath, patchFallback.ToString(), new UTF8Encoding(true));

        // Dump raw AreaTable rows for source and target into compare/v2 for inspection
        DumpAreaTable(storSrc_Area, compareV2Dir, srcAlias);
        DumpAreaTable(storTgt_Area, compareV2Dir, "3.3.5");

        foreach (var kv in perMap)
        {
            File.WriteAllText(Path.Combine(compareV2Dir, $"AreaTable_mapping_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.map.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(compareV2Dir, $"AreaTable_unmatched_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.un.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(compareV2Dir, $"Area_patch_crosswalk_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.patch.ToString(), new UTF8Encoding(true));
            if (perMapFallback.TryGetValue(kv.Key, out var pfb))
                File.WriteAllText(Path.Combine(compareV2Dir, $"Area_patch_with_fallback0_map{kv.Key}_{srcAlias}_to_335.csv"), pfb.ToString(), new UTF8Encoding(true));
        }

        // Write diagnostics: zones missing in target map
        if (zoneMissing.Count > 0)
        {
            var diag = new StringBuilder();
            diag.AppendLine("tgt_mapId,tgt_mapName,zone_key_normalized");
            foreach (var kv in zoneMissing)
            {
                string mname = mapTgtNames.TryGetValue(kv.Key, out var n) ? n : string.Empty;
                foreach (var z in kv.Value)
                {
                    diag.AppendLine(string.Join(',', new[]
                    {
                        kv.Key.ToString(CultureInfo.InvariantCulture),
                        Csv(mname),
                        Csv(z)
                    }));
                }
            }
            File.WriteAllText(Path.Combine(compareV2Dir, $"zone_missing_in_target_map_{srcAlias}_to_335.csv"), diag.ToString(), new UTF8Encoding(true));
        }

        Console.WriteLine("[V2] Wrote mapping/unmatched/patch CSVs under compare/v2/ and audit CSVs under compare/.");
        return 0;
    }

    // Simple edit distance for fuzzy rename (Levenshtein)
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

    private static (int id, int map, bool ok) FindBestFuzzy(string keyNorm, List<(string key, int id, int map)> candidates, int threshold)
    {
        int best = int.MaxValue; int second = int.MaxValue; int bid = -1; int bmap = -1;
        foreach (var c in candidates)
        {
            int d = EditDistance(keyNorm, c.key);
            if (d < best)
            {
                second = best; best = d; bid = c.id; bmap = c.map;
            }
            else if (d < second)
            {
                second = d;
            }
        }
        bool unique = best <= threshold && best < second;
        return unique ? (bid, bmap, true) : (-1, -1, false);
    }

    private static void DumpAreaTable(IDBCDStorage stor, string outDir, string alias)
    {
        string idCol = DetectIdColumn(stor);
        string parentCol = DetectColumn(stor, "ParentAreaID", "ParentAreaNum");
        string nameCol = DetectColumn(stor, "AreaName_lang", "AreaName", "Name");
        var sb = new StringBuilder();
        sb.AppendLine("row_key,id,parent,continentId,name");
        foreach (var k in stor.Keys)
        {
            var r = stor[k];
            int id = !string.IsNullOrWhiteSpace(idCol) ? SafeField<int>(r, idCol) : k;
            int parent = SafeField<int>(r, parentCol);
            if (parent <= 0) parent = id;
            int map = SafeField<int>(r, "ContinentID");
            string name = FirstNonEmpty(SafeField<string>(r, nameCol)) ?? string.Empty;
            sb.AppendLine(string.Join(',', new[]
            {
                k.ToString(CultureInfo.InvariantCulture),
                id.ToString(CultureInfo.InvariantCulture),
                parent.ToString(CultureInfo.InvariantCulture),
                map.ToString(CultureInfo.InvariantCulture),
                Csv(name)
            }));
        }
        File.WriteAllText(Path.Combine(outDir, $"AreaTable_dump_{alias}.csv"), sb.ToString(), new UTF8Encoding(true));
    }

    

    private static string Normalize(string p) => Path.GetFullPath(p);

    private static string ResolveAliasOrInfer(string build, string dir)
    {
        if (!string.IsNullOrWhiteSpace(build)) return build.Trim();
        var tok = (dir ?? "").ToLowerInvariant();
        if (tok.Contains("0.5.3")) return "0.5.3";
        if (tok.Contains("0.5.5")) return "0.5.5";
        if (tok.Contains("0.6.0")) return "0.6.0";
        if (tok.Contains("3.3.5")) return "3.3.5";
        return build?.Trim() ?? "";
    }

    private static DBCD.Locale ParseLocale(string s)
    {
        if (Enum.TryParse<DBCD.Locale>(s, ignoreCase: true, out var loc)) return loc;
        return DBCD.Locale.EnUS;
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
}
