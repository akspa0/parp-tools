using DBCD;
using DBCD.Providers;
using DBCTool.V2.Domain;
using DBCTool.V2.IO;
using DBCTool.V2.AlphaDecode;
using DBCTool.V2.Crosswalk;
using DBCTool.V2.Mapping;
using System.Text;
using System.Globalization;
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

        // Build source AreaNumber/ID -> row index for zoneBase lookups
        string keyColSrcIdx = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID";
        var idxSrcNumToRow = new Dictionary<int, DBCDRow>();
        foreach (var sid in storSrc_Area.Keys)
        {
            var srow = storSrc_Area[sid];
            int areaKey = SafeField<int>(srow, keyColSrcIdx);
            if (areaKey > 0 && !idxSrcNumToRow.ContainsKey(areaKey)) idxSrcNumToRow[areaKey] = srow;
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
        patch.AppendLine("src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name");
        var perMap = new Dictionary<int, (StringBuilder map, StringBuilder un, StringBuilder patch)>();

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

            int contResolved = contRaw;
            if (idxSrcNumToRow.TryGetValue(zoneBase, out var zrSrc))
            {
                contResolved = SafeField<int>(zrSrc, "ContinentID");
            }
            else if (ZoneOwner.TryGetValue(zoneBase, out var owner))
            {
                contResolved = owner;
            }
            int mapIdX = -1; bool hasMapX = false;
            if (cw053To335.TryGetValue(contResolved, out var mx)) { mapIdX = mx; hasMapX = true; }
            else if (mapTgtNames.ContainsKey(contResolved)) { mapIdX = contResolved; hasMapX = true; }

            // Compose chain from decode indices
            var chain = new List<string>();
            if (ZoneIndex.TryGetValue((contResolved, zoneBase), out var zr) && !string.IsNullOrWhiteSpace(zr.ZoneName)) chain.Add(Norm(zr.ZoneName));
            if (area_lo16 > 0 && SubIndex.TryGetValue((contResolved, zoneBase, area_lo16), out var sr) && !string.IsNullOrWhiteSpace(sr.SubName)) chain.Add(Norm(sr.SubName));
            if (chain.Count == 0 && !string.IsNullOrWhiteSpace(nm)) chain.Add(Norm(nm));

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
                        if (!string.IsNullOrWhiteSpace(zoneName060)) pivotChain.Add(Norm(zoneName060));
                        if (!string.IsNullOrWhiteSpace(subName060)) pivotChain.Add(Norm(subName060));

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

            if (chosen >= 0 && tgtIdToRow.TryGetValue(chosen, out var tRow))
            {
                string tgtName = FirstNonEmpty(SafeField<string>(tRow, areaNameColTgt)) ?? string.Empty;
                int tgtParentId = SafeField<int>(tRow, "ParentAreaID");
                int tgtMap = SafeField<int>(tRow, "ContinentID");
                string tgtMapName = mapTgtNames.TryGetValue(tgtMap, out var mn) ? mn : string.Empty;
                string tgtParentName = FirstNonEmpty(SafeField<string>(tgtIdToRow.TryGetValue(tgtParentId, out var prowT) ? prowT : tRow, areaNameColTgt)) ?? tgtName;
                string tgtPath = $"{Norm(tgtMapName)}/{Norm(tgtParentName)}/{Norm(tgtName)}";
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
                        tgtParentId.ToString(CultureInfo.InvariantCulture),
                        Csv(tgtName)
                    }));
                }
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
        File.WriteAllText(mappingPath, mapping.ToString(), new UTF8Encoding(true));
        File.WriteAllText(unmatchedPath, unmatched.ToString(), new UTF8Encoding(true));
        File.WriteAllText(patchPath, patch.ToString(), new UTF8Encoding(true));

        foreach (var kv in perMap)
        {
            File.WriteAllText(Path.Combine(compareV2Dir, $"AreaTable_mapping_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.map.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(compareV2Dir, $"AreaTable_unmatched_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.un.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(compareV2Dir, $"Area_patch_crosswalk_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.patch.ToString(), new UTF8Encoding(true));
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
