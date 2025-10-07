using DBCD;
using DBCD.Providers;
using DBCTool.V2.Domain;
using DBCTool.V2.IO;
using DBCTool.V2.AlphaDecode;
using DBCTool.V2.Crosswalk;
using DBCTool.V2.Mapping;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using static DBCTool.V2.IO.DbdcHelper;

namespace DBCTool.V2.Cli;

public sealed class CompareAreaV2Command
{
    public int Run(string dbdDir, string outBase, string localeStr, List<(string build, string dir)> inputs, bool chainVia060)
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

        // Output folders (stable, version-specific): outBase/<srcAlias>/compare/v2
        var outAliasRoot = Path.Combine(outBase, srcAlias);
        var compareDir = Path.Combine(outAliasRoot, "compare");
        var compareV2Dir = Path.Combine(compareDir, "v2");
        var compareV3Dir = Path.Combine(compareDir, "v3");
        Directory.CreateDirectory(compareDir);
        Directory.CreateDirectory(compareV2Dir);
        Directory.CreateDirectory(compareV3Dir);

        string aliasSlug = srcAlias.Replace('.', '_');

        // Optional hierarchy graphs (YAML emitted by prior runs)
        AreaHierarchyGraph? srcHierarchy = null;
        AreaHierarchyGraph? tgtHierarchy = null;
        Dictionary<(int mapId, int areaId), HierarchyPairingCandidate>? hierarchyPairings = null;
        try
        {
            var srcHierarchyPath = Path.Combine(compareV3Dir, $"Area_hierarchy_src_{srcAlias}.yaml");
            if (File.Exists(srcHierarchyPath))
            {
                srcHierarchy = AreaHierarchyLoader.LoadFromFile(srcHierarchyPath);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[V2] WARN: Failed to load source hierarchy YAML: {ex.Message}");
        }
        try
        {
            var tgtHierarchyPath = Path.Combine(compareV3Dir, "Area_hierarchy_335.yaml");
            if (File.Exists(tgtHierarchyPath))
            {
                tgtHierarchy = AreaHierarchyLoader.LoadFromFile(tgtHierarchyPath);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[V2] WARN: Failed to load target hierarchy YAML: {ex.Message}");
        }

        // Load storages
        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var locale = ParseLocale(localeStr);

        var storSrc_Area = DbdcHelper.LoadTable("AreaTable", CanonicalizeBuild(srcAlias), srcDir!, dbdProvider, locale);
        var storSrc_Map  = DbdcHelper.LoadTable("Map",       CanonicalizeBuild(srcAlias), srcDir!, dbdProvider, locale);
        var storTgt_Area = DbdcHelper.LoadTable("AreaTable", CanonicalizeBuild("3.3.5"),  dir335!, dbdProvider, locale);
        var storTgt_Map  = DbdcHelper.LoadTable("Map",       CanonicalizeBuild("3.3.5"),  dir335!, dbdProvider, locale);

        string parentColSrc = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "ParentAreaNum" : "ParentAreaID";
        string keyColSrc = (srcAlias == "0.5.3" || srcAlias == "0.5.5") ? "AreaNumber" : "ID";
        string areaNameColSrc = DetectColumn(storSrc_Area, "AreaName_lang", "AreaName", "Name");
        string areaNameColTgt = DetectColumn(storTgt_Area, "AreaName_lang", "AreaName", "Name");

        var renameAliasConfigs = LoadRenameAliasConfigs(compareV3Dir);

        foreach (var alias in renameAliasConfigs)
        {
            DbdcHelper.AddNameAlias(alias.From, alias.To);
        }

        if (srcHierarchy is null)
        {
            srcHierarchy = HierarchyPairingGenerator.BuildGraphFromRows(BuildSourceGraph(storSrc_Area, storSrc_Map, areaNameColSrc, parentColSrc, keyColSrc));
        }
        if (tgtHierarchy is null)
        {
            tgtHierarchy = HierarchyPairingGenerator.BuildGraphFromRows(BuildTargetGraph(storTgt_Area, storTgt_Map, areaNameColTgt));
        }

        Dictionary<string, (int mapId, int areaId)> renameOverrides = new(StringComparer.OrdinalIgnoreCase);

        if (hierarchyPairings is null)
        {
            try
            {
                var resolvedOverrides = ResolveRenameOverrides(renameAliasConfigs, tgtHierarchy);
                foreach (var kvp in resolvedOverrides)
                {
                    renameOverrides[kvp.Key] = (kvp.Value.MapId, kvp.Value.AreaId);
                }

                var supplemental = BuildSupplementalMatches(srcHierarchy, resolvedOverrides);
                var pairings = HierarchyPairingGenerator.GenerateCandidates(srcHierarchy, tgtHierarchy, supplemental);
                var pairingPath = Path.Combine(compareV3Dir, $"Area_pairings_{aliasSlug}_to_335.csv");
                WriteHierarchyPairings(pairingPath, pairings);
                Console.WriteLine($"[V3] Wrote hierarchy pairing candidates -> {pairingPath}");

                var pairingsUnmatchedPath = Path.Combine(compareV3Dir, $"Area_pairings_unmatched_{aliasSlug}_to_335.csv");
                WriteHierarchyPairingsUnmatched(pairingsUnmatchedPath, pairings);
                Console.WriteLine($"[V3] Wrote hierarchy unmatched report -> {pairingsUnmatchedPath}");

                hierarchyPairings = pairings.ToDictionary(p => (p.SrcMapId, p.SrcAreaId));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[V3] WARN: Failed to generate hierarchy pairings: {ex.Message}");
            }
        }
        else
        {
            var resolvedOverrides = ResolveRenameOverrides(renameAliasConfigs, tgtHierarchy);
            foreach (var kvp in resolvedOverrides)
            {
                renameOverrides[kvp.Key] = (kvp.Value.MapId, kvp.Value.AreaId);
            }
        }

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
        var midNodeInfo = new Dictionary<(int mapId, int areaNum), AreaNodeInfo>();
        var midZonesByMap = new Dictionary<int, HashSet<int>>();
        var midChildrenByParent = new Dictionary<(int mapId, int zoneBase), List<int>>();
        var midAreaCanonical = new Dictionary<(int mapId, int areaNum), (string name, int priority)>();
        var midZoneCanonical = new Dictionary<(int mapId, int zoneBase), (string name, int priority)>();
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
                bool isZone = parentId == id;
                int zoneId = isZone ? id : parentId;
                midNodeInfo[(mapId, id)] = new AreaNodeInfo(id, parentId, mapId, name);
                PromoteAreaName(midAreaCanonical, (mapId, id), name, 1);
                id060ToRow[id] = row;
                if (isZone)
                {
                    if (!idx060TopZonesByMap.TryGetValue(mapId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idx060TopZonesByMap[mapId] = dict; }
                    dict[NormKey(name)] = id;
                    if (!midZonesByMap.TryGetValue(mapId, out var zoneSet)) { zoneSet = new HashSet<int>(); midZonesByMap[mapId] = zoneSet; }
                    zoneSet.Add(zoneId);
                    PromoteZoneName(midZoneCanonical, (mapId, zoneId), name, 1);
                }
                else
                {
                    if (!idx060ChildrenByZone.TryGetValue(parentId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idx060ChildrenByZone[parentId] = dict; }
                    dict[NormKey(name)] = id;
                    var childKey = (mapId, zoneId);
                    if (!midChildrenByParent.TryGetValue(childKey, out var kids)) { kids = new List<int>(); midChildrenByParent[childKey] = kids; }
                    kids.Add(id);
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
        var tgtNodeInfo = new Dictionary<int, AreaNodeInfo>();
        var tgtZonesByMap = new Dictionary<int, List<int>>();
        var tgtChildrenByParent = new Dictionary<int, List<int>>();
        foreach (var key in storTgt_Area.Keys)
        {
            var row = storTgt_Area[key];
            int id = !string.IsNullOrWhiteSpace(idColAreaTgt) ? SafeField<int>(row, idColAreaTgt) : key;
            string name = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
            int parentId = SafeField<int>(row, "ParentAreaID");
            if (parentId <= 0) parentId = id;
            int mapId = SafeField<int>(row, "ContinentID");
            tgtIdToRow[id] = row;
            tgtNodeInfo[id] = new AreaNodeInfo(id, parentId, mapId, name);
            if (parentId == id)
            {
                if (!idxTgtTopZonesByMap.TryGetValue(mapId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idxTgtTopZonesByMap[mapId] = dict; }
                dict[NormKey(name)] = id;
                if (!tgtZonesByMap.TryGetValue(mapId, out var zones)) { zones = new List<int>(); tgtZonesByMap[mapId] = zones; }
                zones.Add(id);
            }
            else
            {
                if (!idxTgtChildrenByZone.TryGetValue(parentId, out var dict)) { dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase); idxTgtChildrenByZone[parentId] = dict; }
                dict[NormKey(name)] = id;
                if (!tgtChildrenByParent.TryGetValue(parentId, out var list)) { list = new List<int>(); tgtChildrenByParent[parentId] = list; }
                list.Add(id);
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
        var srcNodeInfo = new Dictionary<(int mapId, int areaNum), AreaNodeInfo>();
        var srcZonesByMap = new Dictionary<int, HashSet<int>>();
        var srcChildrenByParent = new Dictionary<(int mapId, int zoneBase), List<int>>();
        var srcAreaCanonical = new Dictionary<(int mapId, int areaNum), (string name, int priority)>();
        var srcZoneCanonical = new Dictionary<(int mapId, int zoneBase), (string name, int priority)>();

        // Special-case overrides and forced parents for strict chain via 0.6.0
        // Keys are normalized with NormKey(s)
        var overrideTargetMapBySrcKey = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            // Dev/test content relocated to Development (map 451) in later builds
            ["programmerisle"] = 451,
            ["designerisland"] = 451,
            ["jeffnequadrant"] = 451,
            ["jeffnwquadrant"] = 451,
            ["jeffsequadrant"] = 451,
            ["jeffswquadrant"] = 451,
            ["deadmanshole"] = 451,
        };

        var forcePivotParentBySrcKey = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            // Known oddities: source listed as zone in 0.5.x but should be sub under these parents later
            ["twilightgrove"] = "duskwood",
            ["caerdarrow"] = "western plaguelands",
            ["darrowmerelake"] = "western plaguelands",
            ["stonewroughtpass"] = "searing gorge",
        };

        var fuzzyOddities = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "twilightgrove","caerdarrow","darrowmerelake"
        };

        AddNameAliases(new[]
        {
            new KeyValuePair<string, string>("Lik'ash Tar Pits", "Lakkari Tar Pits"),
            new KeyValuePair<string, string>("Likkari Tar Pits", "Lakkari Tar Pits")
        });

        static IEnumerable<KeyValuePair<string, string>> ColonAliases(IEnumerable<string> names)
        {
            foreach (var raw in names)
            {
                if (string.IsNullOrWhiteSpace(raw)) continue;
                var parts = raw.Split(':', 2, StringSplitOptions.TrimEntries);
                if (parts.Length != 2) continue;
                var join = $"{parts[0]} {parts[1]}";
                yield return new KeyValuePair<string, string>(raw, join);
                yield return new KeyValuePair<string, string>(join, raw);
            }
        }

        AddNameAliases(ColonAliases(lkChildList.Select(t => t.key)));
        AddNameAliases(ColonAliases(lkTopList.Select(t => t.key)));

        // Some 0.6.0 top-level zones become subzones under different parents in 3.3.5.
        // Re-parent the LK chain accordingly before trying to match against 3.3.5.
        var lkParentByPivotZone = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["darrowmerelake"] = "western plaguelands",
            ["stonewroughtpass"] = "searing gorge",
        };

        // Prepare CSV builders
        var mapping = new StringBuilder();
        var unmatched = new StringBuilder();
        var patch = new StringBuilder();
        var patchVia060 = new StringBuilder();
        var trace = new StringBuilder();
        var crosswalkV3 = new StringBuilder();
        var crosswalkV3Header = string.Join(',', new[]
        {
            "src_row_id","src_areaNumber","src_parentNumber","src_name","src_mapId","src_mapName","src_mapId_resolved","src_zone_base","src_zone_name","src_area_name","src_path",
            "mid060_areaID","mid060_name","mid060_parentID","mid060_parent_name","mid060_mapId","mid060_mapName","mid060_chain",
            "tgt_areaID","tgt_name","tgt_parentID","tgt_parent_name","tgt_mapId","tgt_mapName","tgt_chain","match_method","override_notes"
        });
        crosswalkV3.AppendLine(crosswalkV3Header);
        var perMapCrosswalk = new Dictionary<int, StringBuilder>();
        var perMapCrosswalkMid = new Dictionary<int, StringBuilder>();
        var perMapUnified = new Dictionary<int, StringBuilder>();
        var unifiedHeader = string.Join(',', new[]
        {
            "src_mapId_resolved","src_zone_base","src_areaNumber","src_zone_name","src_area_name",
            "mid_mapId","mid_zone_id","mid_areaID","mid_zone_name","mid_area_name",
            "tgt_mapId","tgt_zone_id","tgt_areaID","tgt_zone_name","tgt_area_name","match_method","override_notes"
        });
        var header = string.Join(',', new[]
        {
            "src_row_id","src_areaNumber","src_parentNumber","src_zone_hi16","src_sub_lo16","src_parent_hi16","src_parent_lo16",
            "src_name","src_mapId","src_mapName","src_mapId_xwalk","src_mapName_xwalk","src_path",
            "mid060_mapId","mid060_mapName","mid060_areaID","mid060_parentID","mid060_chain",
            "tgt_id_335","tgt_name","tgt_parent_id","tgt_parent_name","tgt_mapId","tgt_mapName","tgt_path","tgt_child_ids","tgt_child_names","match_method","override_notes"
        });
        mapping.AppendLine(header);
        unmatched.AppendLine(header);
        var patchHeader = "src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,mid060_mapId,mid060_mapName,mid060_areaID,mid060_parentID,mid060_chain,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name,tgt_child_ids,tgt_child_names,match_method,override_notes";
        patch.AppendLine(patchHeader);
        var patchHeaderVia060 = "src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,mid060_mapId,mid060_mapName,mid060_areaID,mid060_parentID,mid060_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name,tgt_child_ids,tgt_child_names,match_method,override_notes";
        patchVia060.AppendLine(patchHeaderVia060);
        var traceHeader = "src_row_id,src_areaNumber,src_parentNumber,src_name,src_mapId,src_chain,pivot_mapId,pivot_chain,lk_mapId,lk_chain,chosen_tgt_id,matched_depth,method,override_notes";
        trace.AppendLine(traceHeader);
        var patchFallback = new StringBuilder();
        patchFallback.AppendLine(patchHeader);
        var perMap = new Dictionary<int, (StringBuilder map, StringBuilder un, StringBuilder patch)>();
        var perMapFallback = new Dictionary<int, StringBuilder>();
        var perMapVia060 = new Dictionary<int, StringBuilder>();
        var perMapTrace = new Dictionary<int, StringBuilder>();

        // Source columns
        // Source columns already resolved above

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
            int mapResolved = hasMapX ? mapIdX : contResolved;

            var nodeKey = (mapResolved, areaNum);
            srcNodeInfo[nodeKey] = new AreaNodeInfo(areaNum, parentNum, mapResolved, nm);
            PromoteAreaName(srcAreaCanonical, nodeKey, nm, 1);
            if (area_lo16 == 0)
            {
                if (!srcZonesByMap.TryGetValue(mapResolved, out var set)) { set = new HashSet<int>(); srcZonesByMap[mapResolved] = set; }
                set.Add(zoneBase);
            }
            else
            {
                var childKey = (mapResolved, zoneBase);
                if (!srcChildrenByParent.TryGetValue(childKey, out var kids)) { kids = new List<int>(); srcChildrenByParent[childKey] = kids; }
                kids.Add(areaNum);
            }

            // Compose source chain strictly by zone name (ignore sub components in 0.5.x packed values)
            var chain = new List<string>();
            string zoneName = string.Empty;
            string subName = area_lo16 > 0 ? nm : string.Empty;

            if (idxSrcZoneByCont.TryGetValue((contResolved, zoneBase), out var zoneRowForChain))
            {
                zoneName = FirstNonEmpty(SafeField<string>(zoneRowForChain, areaNameColSrc)) ?? string.Empty;
            }

            if (string.IsNullOrWhiteSpace(zoneName))
            {
                // Fallback for zones missing explicit name entries: use the current row when it is the zone itself
                if (area_lo16 == 0 && !string.IsNullOrWhiteSpace(nm))
                {
                    zoneName = nm;
                }
            }

            if (!string.IsNullOrWhiteSpace(zoneName))
            {
                PromoteZoneName(srcZoneCanonical, (mapResolved, zoneBase), zoneName, 1);
            }
            if (!string.IsNullOrWhiteSpace(zoneName))
            {
                chain.Add(NormKey(zoneName));
            }
            if (area_lo16 > 0 && !string.IsNullOrWhiteSpace(subName))
            {
                chain.Add(NormKey(subName));
            }

            string mapName = mapSrcNames.TryGetValue(contResolved, out var mnSrc) ? mnSrc : string.Empty;
            string mapNameX = hasMapX && mapTgtNames.TryGetValue(mapIdX, out var mnTgt) ? mnTgt : string.Empty;
            var chainPath = string.Join('/', chain);
            string path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{chainPath}" : chainPath;

            int chosen = -1; int depth = 0; string method = string.Empty;
            var overrideNotes = new List<string>();
            string lkChainDisplay = string.Empty;
            string pivotChainDisplay = string.Empty;
            bool matchedByHierarchy = false;
            var srcKeyNorm = NormKey(nm);
            if (renameOverrides.TryGetValue(srcKeyNorm, out var forced))
            {
                chosen = forced.areaId;
                mapIdX = forced.mapId;
                hasMapX = true;
                mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mnRename) ? mnRename : string.Empty;
                method = "rename_override";
                overrideNotes.Add($"rename_override={nm}");
                depth = chain.Count;
                matchedByHierarchy = true;
                lkChainDisplay = BuildLkChainDisplay(chosen, tgtIdToRow, areaNameColTgt);
            }
            else if (chain.Count > 0 && renameOverrides.TryGetValue(chain.Last(), out var forcedTarget))
            {
                chosen = forcedTarget.areaId;
                mapIdX = forcedTarget.mapId;
                hasMapX = true;
                mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mnRename) ? mnRename : string.Empty;
                method = "rename_override";
                overrideNotes.Add($"rename_override={nm}");
                depth = chain.Count;
                matchedByHierarchy = true;
                lkChainDisplay = BuildLkChainDisplay(chosen, tgtIdToRow, areaNameColTgt);
            }

            if (!matchedByHierarchy && !chainVia060 && hasMapX && chain.Count > 0)
            {
                // Direct 0.5.x → 3.3.5 chain when not forcing via 0.6.0
                chosen = matcher.TryMatchChainExact(mapIdX, chain, idxTgtTopZonesByMap, idxTgtChildrenByZone, out depth);
                if (chosen >= 0)
                {
                    lkChainDisplay = BuildLkChainDisplay(chosen, tgtIdToRow, areaNameColTgt);
                }
            }

            // Pivot via 0.6.0 (chain) when requested, or as fallback when direct is missing/partial
            int chosen060 = -1; int depth060 = 0; int pivotMapIdX = -1; bool hasPivot = false;
            int midParentId = -1; string midName = string.Empty; string midParentName = string.Empty; int midMapOut = -1; string midMapName = string.Empty;
            int midAreaResolved = -1;
            string pivotChainDesc = string.Empty; string lkChainDesc = string.Empty;
            if (((chainVia060) || (chosen < 0 || depth < chain.Count)) && has060 && chain.Count > 0)
            {
                if (cwSrcTo060.TryGetValue(contResolved, out var mx060)) { pivotMapIdX = mx060; hasPivot = true; }
                else if (map060Names.ContainsKey(contResolved)) { pivotMapIdX = contResolved; hasPivot = true; }
                if (hasPivot)
                {
                    // Attempt strict child resolution on pivot map first when source chain is zone-only or oddity
                    bool isZoneOnlyChain = chain.Count == 1;
                    int localChosen060 = -1; int localParent060 = -1;
                    string zoneName060 = string.Empty; string subName060 = string.Empty;

                    int forcedParent060 = -1; // track even if child not present in 0.6.0
                    string forcedParentNameRaw = string.Empty; // allow name-only fallback when 0.6.0 lacks the zone row
                    if (isZoneOnlyChain)
                    {
                        // Forced parent rule for known oddities
                        if (forcePivotParentBySrcKey.TryGetValue(srcKeyNorm, out var forceParentRaw))
                        {
                            forcedParentNameRaw = forceParentRaw;
                            var parentKey = NormKey(forceParentRaw);
                            if (idx060TopZonesByMap.TryGetValue(pivotMapIdX, out var tops) && tops.TryGetValue(parentKey, out var zoneId060fp))
                            {
                                forcedParent060 = zoneId060fp;
                                if (idx060ChildrenByZone.TryGetValue(zoneId060fp, out var kids) && kids.TryGetValue(srcKeyNorm, out var childId060))
                                {
                                    localChosen060 = childId060; localParent060 = zoneId060fp;
                                }
                            }
                        }

                        // Exact child scan across pivot map if still unresolved
                        if (localChosen060 < 0)
                        {
                            int foundParent = -1, foundChild = -1, matches = 0;
                            foreach (var kv in idx060ChildrenByZone)
                            {
                                int parentId = kv.Key;
                                if (!id060ToRow.TryGetValue(parentId, out var pRow0)) continue;
                                int pMap = SafeField<int>(pRow0, "ContinentID");
                                if (pMap != pivotMapIdX) continue;
                                var kids = kv.Value;
                                if (kids.TryGetValue(srcKeyNorm, out var child))
                                {
                                    matches++; foundParent = parentId; foundChild = child;
                                    if (matches > 1) break;
                                }
                            }
                            if (matches == 1) { localChosen060 = foundChild; localParent060 = foundParent; }
                        }

                        // Minimal fuzzy only for oddities (EditDistance <= 1 and unique)
                        if (localChosen060 < 0 && fuzzyOddities.Contains(srcKeyNorm))
                        {
                            int foundParent = -1, foundChild = -1, matches = 0;
                            foreach (var kv in idx060ChildrenByZone)
                            {
                                int parentId = kv.Key;
                                if (!id060ToRow.TryGetValue(parentId, out var pRow0)) continue;
                                int pMap = SafeField<int>(pRow0, "ContinentID");
                                if (pMap != pivotMapIdX) continue;
                                foreach (var kid in kv.Value)
                                {
                                    int dist = EditDistance(srcKeyNorm, kid.Key);
                                    if (dist <= 1)
                                    {
                                        matches++; foundParent = parentId; foundChild = kid.Value;
                                        if (matches > 1) break;
                                    }
                                }
                                if (matches > 1) break;
                            }
                            if (matches == 1) { localChosen060 = foundChild; localParent060 = foundParent; }
                        }
                    }

                    // Fallback: original chain match on pivot map
                    if (localChosen060 < 0)
                    {
                        localChosen060 = matcher.TryMatchChainExact(pivotMapIdX, chain, idx060TopZonesByMap, idx060ChildrenByZone, out depth060);
                    }

                    if (localChosen060 >= 0 || forcedParent060 > 0 || (!string.IsNullOrWhiteSpace(forcedParentNameRaw) && isZoneOnlyChain))
                    {
                        if (localChosen060 >= 0)
                        {
                            chosen060 = localChosen060;
                            midAreaResolved = chosen060;
                            var pivotRow = id060ToRow[chosen060];
                            int pParent = SafeField<int>(pivotRow, parentCol060);
                            if (pParent <= 0) pParent = chosen060;
                            int zoneId060 = pParent == chosen060 ? (localParent060 > 0 ? localParent060 : chosen060) : pParent;
                            var zoneRow060 = id060ToRow[zoneId060];
                            zoneName060 = FirstNonEmpty(SafeField<string>(zoneRow060, areaNameCol060)) ?? string.Empty;
                            subName060 = (pParent != chosen060 || localParent060 > 0) ? (FirstNonEmpty(SafeField<string>(pivotRow, areaNameCol060)) ?? string.Empty) : string.Empty;

                            // capture mid outputs
                            midParentId = (localParent060 > 0) ? localParent060 : pParent;
                            midName = FirstNonEmpty(SafeField<string>(pivotRow, areaNameCol060)) ?? string.Empty;
                            midMapOut = pivotMapIdX;
                            midMapName = map060Names.TryGetValue(midMapOut, out var mmn) ? mmn : string.Empty;
                            midParentName = (midParentId == chosen060) ? (FirstNonEmpty(SafeField<string>(zoneRow060, areaNameCol060)) ?? midName) : (FirstNonEmpty(SafeField<string>(id060ToRow[midParentId], areaNameCol060)) ?? midName);
                        }
                        else if (forcedParent060 > 0) // forced parent with 0.6.0 zone row; build pivot via parent row + src name
                        {
                            var zoneRow060 = id060ToRow[forcedParent060];
                            zoneName060 = FirstNonEmpty(SafeField<string>(zoneRow060, areaNameCol060)) ?? string.Empty;
                            subName060 = nm;
                            chosen060 = -1; // no concrete 0.6.0 child
                            // capture mid outputs (parent only)
                            midParentId = forcedParent060;
                            midName = nm;
                            midMapOut = pivotMapIdX;
                            midMapName = map060Names.TryGetValue(midMapOut, out var mmn2) ? mmn2 : string.Empty;
                            midParentName = FirstNonEmpty(SafeField<string>(zoneRow060, areaNameCol060)) ?? string.Empty;
                            midAreaResolved = forcedParent060;
                        }
                        else // name-only forced parent (0.6.0 lacks the zone row); build LK chain using names directly
                        {
                            zoneName060 = forcedParentNameRaw;
                            subName060 = nm;
                            chosen060 = -1;
                            midParentId = -1;
                            midName = nm;
                            midMapOut = pivotMapIdX;
                            midMapName = map060Names.TryGetValue(midMapOut, out var mmn2) ? mmn2 : string.Empty;
                            midParentName = forcedParentNameRaw;
                        }

                        pivotChainDisplay = BuildDisplayChain(zoneName060, subName060);
                        var pivotChain = new List<string>();
                        if (!string.IsNullOrWhiteSpace(zoneName060)) pivotChain.Add(NormKey(zoneName060));
                        if (!string.IsNullOrWhiteSpace(subName060)) pivotChain.Add(NormKey(subName060));

                        // Build the LK chain (may be the same as pivot chain or re-parented)
                        var lkChain = new List<string>(pivotChain);
                        if (lkChain.Count > 0)
                        {
                            var pz = lkChain[0];
                            if (lkParentByPivotZone.TryGetValue(pz, out var lkParentName))
                            {
                                var parentKey = NormKey(lkParentName);
                                if (lkChain.Count >= 2)
                                {
                                    lkChain = new List<string> { parentKey, lkChain[1] };
                                }
                                else
                                {
                                    // Zone-only case: the original pivot zone becomes a child under the mapped parent
                                    lkChain = new List<string> { parentKey, pz };
                                }
                            }
                        }

                        int lkMapIdX = -1; bool hasLk = false;
                        if (overrideTargetMapBySrcKey.TryGetValue(srcKeyNorm, out var oMap)) { lkMapIdX = oMap; hasLk = true; }
                        else if (cw060To335.TryGetValue(pivotMapIdX, out var m335)) { lkMapIdX = m335; hasLk = true; }
                        else if (mapTgtNames.ContainsKey(pivotMapIdX)) { lkMapIdX = pivotMapIdX; hasLk = true; }
                        pivotChainDesc = string.Join('/', pivotChain);
                        lkChainDesc = string.Join('/', lkChain);

                        if (hasLk && lkChain.Count > 0)
                        {
                            int chosenLK = matcher.TryMatchChainExact(lkMapIdX, lkChain, idxTgtTopZonesByMap, idxTgtChildrenByZone, out var depthLK);
                            if (chosenLK >= 0)
                            {
                                chosen = chosenLK; depth = depthLK; mapIdX = lkMapIdX; hasMapX = true;
                                mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn2) ? mn2 : string.Empty;
                                path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{lkChainDesc}" : lkChainDesc;
                                method = (lkChain.Count >= 2 && pivotChain.Count == 1 && lkChain[1] == pivotChain[0]) ?
                                    "pivot_060_reparent" : ((depth == lkChain.Count) ? "pivot_060" : "pivot_060_zone_only");
                                overrideNotes.Add($"pivot_chain={pivotChainDisplay}");
                            }
                            else
                            {
                                // Fallback: treat sub name as a top-level on the same LK map (handles cases like Caer Darrow, Darrowmere Lake, Stonewrought Pass)
                                var subKey = (lkChain.Count >= 2 ? lkChain[1] : NormKey(subName060));
                                if (!string.IsNullOrWhiteSpace(subKey) && idxTgtTopZonesByMap.TryGetValue(lkMapIdX, out var topsLk) && topsLk.TryGetValue(subKey, out var topId))
                                {
                                    chosen = topId; depth = 1; mapIdX = lkMapIdX; hasMapX = true;
                                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn2b) ? mn2b : string.Empty;
                                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{subKey}" : subKey;
                                    method = "pivot_060_top_on_map";
                                    overrideNotes.Add($"pivot_top_on_map={subKey}");
                                }
                            }
                        }
                    }
                }
            }

            if (has060 && midMapOut < 0)
            {
                string zoneNameForMid = ResolveCanonicalName(srcZoneCanonical, (mapResolved, zoneBase));
                if (string.IsNullOrWhiteSpace(zoneNameForMid)) zoneNameForMid = zoneName;
                string areaNameForMid = string.Empty;
                if (area_lo16 > 0)
                {
                    areaNameForMid = ResolveCanonicalName(srcAreaCanonical, (mapResolved, areaNum));
                    if (string.IsNullOrWhiteSpace(areaNameForMid)) areaNameForMid = subName;
                }
                else
                {
                    areaNameForMid = subName;
                }

                if (!string.IsNullOrWhiteSpace(zoneNameForMid)
                    && TryInferMidFromCanonical(zoneNameForMid, areaNameForMid, midZonesByMap, midChildrenByParent, midZoneCanonical, midAreaCanonical,
                        out var inferredMapId, out var inferredZoneId, out var inferredAreaId))
                {
                    midMapOut = inferredMapId;
                    midParentId = inferredZoneId;
                    midMapName = map060Names.TryGetValue(inferredMapId, out var inferredMapName) ? inferredMapName : string.Empty;
                    midParentName = ResolveCanonicalName(midZoneCanonical, (inferredMapId, inferredZoneId));
                    if (string.IsNullOrWhiteSpace(midParentName)) midParentName = zoneNameForMid;
                    if (inferredAreaId > 0)
                    {
                        midAreaResolved = inferredAreaId;
                        midName = ResolveCanonicalName(midAreaCanonical, (inferredMapId, inferredAreaId));
                        if (string.IsNullOrWhiteSpace(midName)) midName = areaNameForMid;
                    }
                    else
                    {
                        midAreaResolved = -1;
                        if (string.IsNullOrWhiteSpace(midName)) midName = areaNameForMid;
                    }
                }
            }

            if (midAreaResolved < 0 && midParentId > 0)
            {
                midAreaResolved = midParentId;
                if (string.IsNullOrWhiteSpace(midName)) midName = midParentName;
            }

            // Global rename match: unique top-level zone across LK (cross-map allowed) - disabled in chainVia060 mode
            if (!matchedByHierarchy && !chainVia060 && chosen < 0 && chain.Count > 0)
            {
                var keyNorm = chain[0];
                if (idxTgtTopGlobal.TryGetValue(keyNorm, out var rec))
                {
                    chosen = rec.areaId; depth = 1; mapIdX = rec.mapId; hasMapX = true;
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn2) ? mn2 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm}" : keyNorm;
                    method = "rename_global";
                    overrideNotes.Add($"rename_global={keyNorm}");
                }
            }

            // Global rename match for child names (use the current row's own name) - disabled in chainVia060 mode
            if (!matchedByHierarchy && !chainVia060 && chosen < 0 && !string.IsNullOrWhiteSpace(nm))
            {
                var keyNorm2 = NormKey(nm);
                if (idxTgtChildGlobal.TryGetValue(keyNorm2, out var rec2))
                {
                    chosen = rec2.areaId; depth = 1; mapIdX = rec2.mapId; hasMapX = true;
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn3) ? mn3 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm2}" : keyNorm2;
                    method = "rename_global_child";
                    overrideNotes.Add($"rename_global_child={keyNorm2}");
                }
            }

            // Fuzzy rename (top-level) if still unmatched
            if (!matchedByHierarchy && !chainVia060 && chosen < 0 && chain.Count > 0)
            {
                var keyNorm = chain[0];
                var (fid, fmap, ok) = FindBestFuzzy(keyNorm, lkTopList, 2);
                if (ok)
                {
                    chosen = fid; depth = 1; mapIdX = fmap; hasMapX = true; method = "rename_fuzzy";
                    overrideNotes.Add($"rename_fuzzy={keyNorm}");
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn4) ? mn4 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm}" : keyNorm;
                }
            }

            // Fuzzy rename (child) using row name if still unmatched - disabled in chainVia060 mode
            if (!matchedByHierarchy && !chainVia060 && chosen < 0 && !string.IsNullOrWhiteSpace(nm))
            {
                var keyNorm = NormKey(nm);
                var (fid, fmap, ok) = FindBestFuzzy(keyNorm, lkChildList, 2);
                if (ok)
                {
                    chosen = fid; depth = 1; mapIdX = fmap; hasMapX = true; method = "rename_fuzzy_child";
                    overrideNotes.Add($"rename_fuzzy_child={keyNorm}");
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn5) ? mn5 : string.Empty;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{keyNorm}" : keyNorm;
                }
            }

            // Exact-key fallback across LK with map preference (prefer map 0, then 1)
            if (!matchedByHierarchy && !chainVia060 && chosen < 0)
            {
                int pref(int m) => (m == 0 ? 0 : (m == 1 ? 1 : 2));
                if (chain.Count > 0)
                {
                    var k = chain[0];
                    var hits = lkTopList.Where(x => x.key.Equals(k, StringComparison.OrdinalIgnoreCase)).OrderBy(x => pref(x.map)).ToList();
                    if (hits.Count > 0)
                    {
                        chosen = hits[0].id; mapIdX = hits[0].map; hasMapX = true; depth = 1; method = "rename_exact_global";
                        overrideNotes.Add($"rename_exact_global={k}");
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
                        overrideNotes.Add($"rename_exact_global_child={k2}");
                        mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mn7) ? mn7 : string.Empty;
                        path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{k2}" : k2;
                    }
                }
            }

            // Special-case: ***On Map Dungeon*** -> force target 0 in fallback
            bool isOnMapDungeon = string.Equals(NormKey(nm), "onmapdungeon", StringComparison.OrdinalIgnoreCase);

            var originalMethod = method;
            bool matchedChildTarget = originalMethod.IndexOf("_child", StringComparison.OrdinalIgnoreCase) >= 0;

            if (!isOnMapDungeon && chosen >= 0 && tgtIdToRow.TryGetValue(chosen, out var tRow))
            {
                string tgtName = FirstNonEmpty(SafeField<string>(tRow, areaNameColTgt)) ?? string.Empty;
                int tgtParentId = SafeField<int>(tRow, "ParentAreaID");
                int tgtMap = SafeField<int>(tRow, "ContinentID");
                string tgtMapName = mapTgtNames.TryGetValue(tgtMap, out var mn) ? mn : string.Empty;
                string tgtParentNameResolved = ResolveTargetAreaName(tgtIdToRow, tgtParentId, areaNameColTgt);
                // Promote to top-level zone when the matched record is a child
                bool promotedToParent = false;
                if (tgtParentId <= 0) tgtParentId = chosen;
                string tgtParentName = tgtName;
                bool keepChildForZone = matchedChildTarget && area_lo16 == 0 && NormKey(tgtName) == NormKey(nm);
                if (area_lo16 == 0 && tgtParentId != chosen && tgtParentId > 0 && !keepChildForZone && tgtIdToRow.TryGetValue(tgtParentId, out var pRow))
                {
                    promotedToParent = true;
                    chosen = tgtParentId;
                    tgtName = FirstNonEmpty(SafeField<string>(pRow, areaNameColTgt)) ?? tgtParentName;
                    tgtParentName = tgtName;
                    tgtParentId = chosen;
                    tgtMap = SafeField<int>(pRow, "ContinentID");
                    tgtMapName = mapTgtNames.TryGetValue(tgtMap, out var mnParent) ? mnParent : tgtMapName;
                    mapIdX = tgtMap;
                    hasMapX = true;
                    mapNameX = mapTgtNames.TryGetValue(mapIdX, out var mnX) ? mnX : mapNameX;
                    path = !string.IsNullOrWhiteSpace(mapNameX) ? $"{Norm(mapNameX)}/{NormKey(tgtName)}" : NormKey(tgtName);
                }
                else if (!keepChildForZone)
                {
                    tgtParentName = tgtName;
                    tgtParentId = chosen;
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(tgtParentNameResolved))
                    {
                        tgtParentName = tgtParentNameResolved;
                    }
                    else if (tgtParentId > 0 && tgtIdToRow.TryGetValue(tgtParentId, out var parentRow))
                    {
                        tgtParentName = FirstNonEmpty(SafeField<string>(parentRow, areaNameColTgt)) ?? tgtParentName;
                    }
                    overrideNotes.Add("retain_child_for_zone");
                }

                string tgtPath = $"{Norm(tgtMapName)}/{Norm(tgtName)}";
                if (string.IsNullOrEmpty(method) || string.Equals(method, "exact", StringComparison.OrdinalIgnoreCase) || string.Equals(method, "zone_only", StringComparison.OrdinalIgnoreCase))
                {
                    method = "exact_zone";
                }
                if (string.Equals(method, "rename_global_child", StringComparison.OrdinalIgnoreCase)) method = "rename_global";
                if (string.Equals(method, "rename_fuzzy_child", StringComparison.OrdinalIgnoreCase)) method = "rename_fuzzy";
                if (string.Equals(method, "rename_exact_global_child", StringComparison.OrdinalIgnoreCase)) method = "rename_exact_global";
                if (promotedToParent)
                {
                    method = method.Contains("_parent", StringComparison.OrdinalIgnoreCase) ? method : $"{method}_parent";
                    overrideNotes.Add("promoted_to_parent");
                }
                else if ((area_lo16 > 0 || keepChildForZone) && !method.Contains("_sub", StringComparison.OrdinalIgnoreCase))
                {
                    method = $"{method}_sub";
                }

                var overrideNotesStr = overrideNotes.Count == 0 ? string.Empty : string.Join('|', overrideNotes);

                lkChainDisplay = BuildLkChainDisplay(chosen, tgtIdToRow, areaNameColTgt);

                (string childIdsStr, string childNamesStr) = BuildTargetChildrenInfo(tgtIdToRow, idxTgtChildrenByZone, chosen, tgtParentId, areaNameColTgt);

                string midMapNameOut = string.Empty;
                if (midMapOut >= 0 && map060Names.TryGetValue(midMapOut, out var mmOut))
                {
                    midMapNameOut = mmOut;
                }
                else if (hasPivot && pivotMapIdX >= 0 && map060Names.TryGetValue(pivotMapIdX, out var mmFallback))
                {
                    midMapNameOut = mmFallback;
                }
                string midChainOut = string.Empty;
                int midChosenId = midAreaResolved >= 0 ? midAreaResolved : chosen060;
                int midParentOut = midParentId > 0 ? midParentId : (midChosenId > 0 ? midChosenId : midParentId);
                if (!string.IsNullOrWhiteSpace(pivotChainDisplay)) midChainOut = pivotChainDisplay;
                else if (!string.IsNullOrWhiteSpace(pivotChainDesc)) midChainOut = pivotChainDesc;
                else if (midMapOut >= 0)
                {
                    var midZoneDisplay = midParentOut > 0 ? ResolveCanonicalName(midZoneCanonical, (midMapOut, midParentOut)) : string.Empty;
                    if (string.IsNullOrWhiteSpace(midZoneDisplay)) midZoneDisplay = midParentName;
                    var midSubDisplay = string.Empty;
                    if (midChosenId > 0 && midParentOut > 0 && midChosenId != midParentOut)
                    {
                        midSubDisplay = ResolveCanonicalName(midAreaCanonical, (midMapOut, midChosenId));
                        if (string.IsNullOrWhiteSpace(midSubDisplay)) midSubDisplay = midName;
                    }
                    else if (!string.IsNullOrWhiteSpace(midName) && !string.Equals(midName, midZoneDisplay, StringComparison.OrdinalIgnoreCase))
                    {
                        midSubDisplay = midName;
                    }
                    midChainOut = BuildDisplayChain(midZoneDisplay, midSubDisplay);
                    if (string.IsNullOrWhiteSpace(midChainOut))
                    {
                        midChainOut = midZoneDisplay;
                    }
                }

                var line = string.Join(',', new[]
                {
                    key.ToString(CultureInfo.InvariantCulture),
                    areaNum.ToString(CultureInfo.InvariantCulture),
                    parentNum.ToString(CultureInfo.InvariantCulture),
                    area_hi16.ToString(CultureInfo.InvariantCulture),
                    area_lo16.ToString(CultureInfo.InvariantCulture),
                    ((parentNum >> 16) & 0xFFFF).ToString(CultureInfo.InvariantCulture),
                    (parentNum & 0xFFFF).ToString(CultureInfo.InvariantCulture),
                    Csv(nm),
                    contResolved.ToString(CultureInfo.InvariantCulture),
                    Csv(mapName),
                    (hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(mapNameX),
                    Csv(path),
                    (hasPivot ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midMapNameOut),
                    (midChosenId >= 0 ? midChosenId.ToString(CultureInfo.InvariantCulture) : "-1"),
                    (midParentOut >= 0 ? midParentOut.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midChainOut),
                    chosen.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtName),
                    tgtParentId.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtParentName),
                    tgtMap.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtMapName),
                    Csv(tgtPath),
                    Csv(childIdsStr),
                    Csv(childNamesStr),
                    method,
                    Csv(overrideNotesStr)
                });
                mapping.AppendLine(line);
                string canonicalZoneName = !string.IsNullOrWhiteSpace(lkChainDisplay)
                    ? lkChainDisplay.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? string.Empty
                    : ResolveCanonicalName(srcZoneCanonical, (mapResolved, zoneBase));
                if (string.IsNullOrWhiteSpace(canonicalZoneName) && !string.IsNullOrWhiteSpace(zoneName))
                {
                    canonicalZoneName = zoneName;
                }
                string canonicalAreaName = !string.IsNullOrWhiteSpace(lkChainDisplay) && lkChainDisplay.Contains('/')
                    ? lkChainDisplay.Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault() ?? string.Empty
                    : ResolveCanonicalName(srcAreaCanonical, (mapResolved, areaNum));
                if (string.IsNullOrWhiteSpace(canonicalAreaName) && !string.IsNullOrWhiteSpace(subName))
                {
                    canonicalAreaName = subName;
                }

                var crosswalkLine = string.Join(',', new[]
                {
                    key.ToString(CultureInfo.InvariantCulture),
                    areaNum.ToString(CultureInfo.InvariantCulture),
                    parentNum.ToString(CultureInfo.InvariantCulture),
                    Csv(nm),
                    contResolved.ToString(CultureInfo.InvariantCulture),
                    Csv(mapName),
                    mapResolved.ToString(CultureInfo.InvariantCulture),
                    zoneBase.ToString(CultureInfo.InvariantCulture),
                    Csv(canonicalZoneName),
                    Csv(canonicalAreaName),
                    Csv(path),
                    (midChosenId >= 0 ? midChosenId.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midName),
                    (midParentOut >= 0 ? midParentOut.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midParentName),
                    (hasPivot ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midMapNameOut),
                    Csv(midChainOut),
                    chosen.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtName),
                    tgtParentId.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtParentName),
                    tgtMap.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtMapName),
                    Csv(lkChainDisplay),
                    method,
                    Csv(overrideNotesStr)
                });
                crosswalkV3.AppendLine(crosswalkLine);
                if (!perMapCrosswalk.TryGetValue(mapResolved, out var mapCrosswalk))
                {
                    mapCrosswalk = new StringBuilder();
                    mapCrosswalk.AppendLine(crosswalkV3Header);
                    perMapCrosswalk[mapResolved] = mapCrosswalk;
                }
                mapCrosswalk.AppendLine(crosswalkLine);
                if (midMapOut >= 0)
                {
                    if (!perMapCrosswalkMid.TryGetValue(midMapOut, out var midCrosswalk))
                    {
                        midCrosswalk = new StringBuilder();
                        midCrosswalk.AppendLine(crosswalkV3Header);
                        perMapCrosswalkMid[midMapOut] = midCrosswalk;
                    }
                    midCrosswalk.AppendLine(crosswalkLine);
                }
                if (!perMapUnified.TryGetValue(mapResolved, out var unifiedBuilder))
                {
                    unifiedBuilder = new StringBuilder();
                    unifiedBuilder.AppendLine(unifiedHeader);
                    perMapUnified[mapResolved] = unifiedBuilder;
                }
                int midZoneId = midParentOut > 0 ? midParentOut : (midChosenId >= 0 ? midChosenId : -1);
                string midZoneName = (midMapOut >= 0 && midZoneId > 0) ? ResolveCanonicalName(midZoneCanonical, (midMapOut, midZoneId)) : string.Empty;
                if (string.IsNullOrWhiteSpace(midZoneName) && !string.IsNullOrWhiteSpace(midParentName)) midZoneName = midParentName;
                string midAreaName = (midMapOut >= 0 && midChosenId >= 0) ? ResolveCanonicalName(midAreaCanonical, (midMapOut, midChosenId)) : string.Empty;
                if (string.IsNullOrWhiteSpace(midAreaName) && !string.IsNullOrWhiteSpace(midName)) midAreaName = midName;
                string tgtZoneName = ResolveTargetAreaName(tgtIdToRow, tgtParentId, areaNameColTgt);
                string tgtAreaName = ResolveTargetAreaName(tgtIdToRow, chosen, areaNameColTgt);
                unifiedBuilder.AppendLine(string.Join(',', new[]
                {
                    mapResolved.ToString(CultureInfo.InvariantCulture),
                    zoneBase.ToString(CultureInfo.InvariantCulture),
                    areaNum.ToString(CultureInfo.InvariantCulture),
                    Csv(canonicalZoneName),
                    Csv(canonicalAreaName),
                    (midMapOut >= 0 ? midMapOut.ToString(CultureInfo.InvariantCulture) : "-1"),
                    (midZoneId > 0 ? midZoneId.ToString(CultureInfo.InvariantCulture) : "-1"),
                    (midChosenId >= 0 ? midChosenId.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midZoneName),
                    Csv(midAreaName),
                    tgtMap.ToString(CultureInfo.InvariantCulture),
                    tgtParentId.ToString(CultureInfo.InvariantCulture),
                    chosen.ToString(CultureInfo.InvariantCulture),
                    Csv(tgtZoneName),
                    Csv(tgtAreaName),
                    method,
                    Csv(overrideNotesStr)
                }));
                if (hasMapX)
                {
                    if (!perMap.TryGetValue(mapIdX, out var tuple)) { tuple = (new StringBuilder(), new StringBuilder(), new StringBuilder()); tuple.map.AppendLine(header); tuple.un.AppendLine(header); tuple.patch.AppendLine(patchHeader); perMap[mapIdX] = tuple; }
                    tuple.map.AppendLine(line);
                    tuple.patch.AppendLine(string.Join(',', new[]
                    {
                        contResolved.ToString(CultureInfo.InvariantCulture),
                        Csv(mapName),
                        areaNum.ToString(CultureInfo.InvariantCulture),
                        parentNum.ToString(CultureInfo.InvariantCulture),
                        Csv(nm),
                        (hasPivot ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                        Csv(midMapNameOut),
                        (midChosenId >= 0 ? midChosenId.ToString(CultureInfo.InvariantCulture) : "-1"),
                        (midParentOut >= 0 ? midParentOut.ToString(CultureInfo.InvariantCulture) : "-1"),
                        Csv(midChainOut),
                        hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                        Csv(mapNameX),
                        chosen.ToString(CultureInfo.InvariantCulture),
                        tgtParentId.ToString(CultureInfo.InvariantCulture),
                        Csv(tgtName),
                        Csv(childIdsStr),
                        Csv(childNamesStr),
                        method,
                        Csv(overrideNotesStr)
                    }));
                    // Also emit via060 patch row when chain was used or requested (mid columns)
                    if (chainVia060)
                    {
                        if (!perMapVia060.TryGetValue(mapIdX, out var via)) { via = new StringBuilder(); via.AppendLine(patchHeaderVia060); perMapVia060[mapIdX] = via; }
                        via.AppendLine(string.Join(',', new[]
                        {
                            contResolved.ToString(CultureInfo.InvariantCulture),
                            Csv(mapName),
                            areaNum.ToString(CultureInfo.InvariantCulture),
                            parentNum.ToString(CultureInfo.InvariantCulture),
                            Csv(nm),
                            (midMapOut >= 0 ? midMapOut.ToString(CultureInfo.InvariantCulture) : "-1"),
                            Csv(midMapName),
                            (chosen060 >= 0 ? chosen060.ToString(CultureInfo.InvariantCulture) : "-1"),
                            (midParentId > 0 ? midParentId.ToString(CultureInfo.InvariantCulture) : "-1"),
                            Csv(midName),
                            hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                            Csv(mapNameX),
                            chosen.ToString(CultureInfo.InvariantCulture),
                            tgtParentId.ToString(CultureInfo.InvariantCulture),
                            Csv(tgtName),
                            Csv(childIdsStr),
                            Csv(childNamesStr),
                            method,
                            Csv(overrideNotesStr)
                        }));
                        // Trace for matched rows
                        var pivotOut = !string.IsNullOrWhiteSpace(pivotChainDisplay) ? pivotChainDisplay : pivotChainDesc;
                        var lkOut = !string.IsNullOrWhiteSpace(lkChainDisplay) ? lkChainDisplay : lkChainDesc;

                        var tr = string.Join(',', new[]
                        {
                            key.ToString(CultureInfo.InvariantCulture),
                            areaNum.ToString(CultureInfo.InvariantCulture),
                            parentNum.ToString(CultureInfo.InvariantCulture),
                            Csv(nm),
                            contResolved.ToString(CultureInfo.InvariantCulture),
                            Csv(string.Join('/', chain)),
                            (midMapOut >= 0 ? midMapOut.ToString(CultureInfo.InvariantCulture) : "-1"),
                            Csv(pivotOut),
                            (hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                            Csv(lkOut),
                            chosen.ToString(CultureInfo.InvariantCulture),
                            depth.ToString(CultureInfo.InvariantCulture),
                            method
                        });
                        trace.AppendLine(tr);
                        if (!perMapTrace.TryGetValue(mapIdX, out var tmap)) { tmap = new StringBuilder(); tmap.AppendLine(traceHeader); perMapTrace[mapIdX] = tmap; }
                        perMapTrace[mapIdX].AppendLine(tr);
                    }
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
                    area_hi16.ToString(CultureInfo.InvariantCulture),
                    area_lo16.ToString(CultureInfo.InvariantCulture),
                    ((parentNum >> 16) & 0xFFFF).ToString(CultureInfo.InvariantCulture),
                    (parentNum & 0xFFFF).ToString(CultureInfo.InvariantCulture),
                    Csv(nm),
                    contResolved.ToString(CultureInfo.InvariantCulture),
                    Csv(mapName),
                    (hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(mapNameX),
                    Csv(path),
                    (hasPivot ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(hasPivot && pivotMapIdX >= 0 && map060Names.TryGetValue(pivotMapIdX, out var mmUn) ? mmUn : string.Empty),
                    "-1",
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    string.Empty,
                    string.Empty,
                    string.Empty,
                    "unmatched_zone"
                });
                unmatched.AppendLine(line);
                int mapResolvedUnmatched = hasMapX ? mapIdX : contResolved;
                string canonicalZoneNameUn = ResolveCanonicalName(srcZoneCanonical, (mapResolvedUnmatched, zoneBase));
                if (string.IsNullOrWhiteSpace(canonicalZoneNameUn) && !string.IsNullOrWhiteSpace(zoneName))
                {
                    canonicalZoneNameUn = zoneName;
                }
                string canonicalAreaNameUn = ResolveCanonicalName(srcAreaCanonical, (mapResolvedUnmatched, areaNum));
                if (string.IsNullOrWhiteSpace(canonicalAreaNameUn) && !string.IsNullOrWhiteSpace(subName))
                {
                    canonicalAreaNameUn = subName;
                }
                var crosswalkUnmatched = string.Join(',', new[]
                {
                    key.ToString(CultureInfo.InvariantCulture),
                    areaNum.ToString(CultureInfo.InvariantCulture),
                    parentNum.ToString(CultureInfo.InvariantCulture),
                    Csv(nm),
                    contResolved.ToString(CultureInfo.InvariantCulture),
                    Csv(mapName),
                    mapResolvedUnmatched.ToString(CultureInfo.InvariantCulture),
                    zoneBase.ToString(CultureInfo.InvariantCulture),
                    Csv(canonicalZoneNameUn),
                    Csv(canonicalAreaNameUn),
                    Csv(path),
                    (chosen060 >= 0 ? chosen060.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midName),
                    (midParentId > 0 ? midParentId.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(midParentName),
                    (hasPivot ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(hasPivot && pivotMapIdX >= 0 && map060Names.TryGetValue(pivotMapIdX, out var mmMid) ? mmMid : string.Empty),
                    Csv(pivotChainDesc),
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    (hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                    Csv(mapNameX),
                    string.Empty,
                    "unmatched_zone"
                });
                crosswalkV3.AppendLine(crosswalkUnmatched);
                if (!perMapCrosswalk.TryGetValue(mapResolvedUnmatched, out var mapCrosswalkUn))
                {
                    mapCrosswalkUn = new StringBuilder();
                    mapCrosswalkUn.AppendLine(crosswalkV3Header);
                    perMapCrosswalk[mapResolvedUnmatched] = mapCrosswalkUn;
                }
                mapCrosswalkUn.AppendLine(crosswalkUnmatched);
                // Append trace row for unmatched when via-060 is on
                if (chainVia060)
                {
                    var pivotOut = !string.IsNullOrWhiteSpace(pivotChainDisplay) ? pivotChainDisplay : pivotChainDesc;
                    var lkOut = !string.IsNullOrWhiteSpace(lkChainDisplay) ? lkChainDisplay : lkChainDesc;

                    var tr = string.Join(',', new[]
                    {
                        key.ToString(CultureInfo.InvariantCulture),
                        areaNum.ToString(CultureInfo.InvariantCulture),
                        parentNum.ToString(CultureInfo.InvariantCulture),
                        Csv(nm),
                        contResolved.ToString(CultureInfo.InvariantCulture),
                        Csv(string.Join('/', chain)),
                        (pivotMapIdX >= 0 ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                        Csv(pivotOut),
                        (hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                        Csv(lkOut),
                        "-1",
                        "0",
                        "unmatched"
                    });
                    trace.AppendLine(tr);
                    if (hasMapX)
                    {
                        if (!perMapTrace.TryGetValue(mapIdX, out var tmap)) { tmap = new StringBuilder(); tmap.AppendLine(traceHeader); perMapTrace[mapIdX] = tmap; }
                        perMapTrace[mapIdX].AppendLine(tr);
                    }
                }
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
                        (hasPivot ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                        Csv(hasPivot && pivotMapIdX >= 0 && map060Names.TryGetValue(pivotMapIdX, out var mmUnFallback) ? mmUnFallback : string.Empty),
                        "-1",
                        "-1",
                        string.Empty,
                        hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                        Csv(mapNameX),
                        "0",
                        "0",
                        string.Empty,
                        string.Empty,
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
                        (hasPivot ? pivotMapIdX.ToString(CultureInfo.InvariantCulture) : "-1"),
                        Csv(hasPivot && pivotMapIdX >= 0 && map060Names.TryGetValue(pivotMapIdX, out var mmUnFallbackGlobal) ? mmUnFallbackGlobal : string.Empty),
                        "-1",
                        "-1",
                        string.Empty,
                        hasMapX ? mapIdX.ToString(CultureInfo.InvariantCulture) : "-1",
                        Csv(mapNameX),
                        "0",
                        "0",
                        string.Empty,
                        string.Empty,
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
        var crosswalkV3Path = Path.Combine(compareV3Dir, $"Area_crosswalk_v3_{srcAlias}_to_335.csv");
        File.WriteAllText(crosswalkV3Path, crosswalkV3.ToString(), new UTF8Encoding(true));
        foreach (var kv in perMapCrosswalk)
        {
            var perMapPath = Path.Combine(compareV3Dir, $"Area_crosswalk_v3_map{kv.Key}_{srcAlias}_to_335.csv");
            File.WriteAllText(perMapPath, kv.Value.ToString(), new UTF8Encoding(true));
        }
        var hierarchyYamlPath = Path.Combine(compareV3Dir, "Area_hierarchy_335.yaml");
        var hierarchyYaml = BuildHierarchyYaml(tgtZonesByMap, tgtChildrenByParent, tgtNodeInfo, mapTgtNames);
        File.WriteAllText(hierarchyYamlPath, hierarchyYaml, new UTF8Encoding(true));
        var srcHierarchyYamlPath = Path.Combine(compareV3Dir, $"Area_hierarchy_src_{srcAlias}.yaml");
        var srcHierarchyYaml = BuildSourceHierarchyYaml(srcZonesByMap, srcChildrenByParent, srcNodeInfo, srcZoneCanonical, srcAreaCanonical, mapSrcNames);
        File.WriteAllText(srcHierarchyYamlPath, srcHierarchyYaml, new UTF8Encoding(true));
        if (has060)
        {
            var midYamlPath = Path.Combine(compareV3Dir, "Area_hierarchy_mid_0.6.0.yaml");
            var midYaml = BuildSourceHierarchyYaml(midZonesByMap, midChildrenByParent, midNodeInfo, midZoneCanonical, midAreaCanonical, map060Names);
            File.WriteAllText(midYamlPath, midYaml, new UTF8Encoding(true));
            foreach (var kv in perMapCrosswalkMid)
            {
                var midPerMapPath = Path.Combine(compareV3Dir, $"Area_crosswalk_via060_map{kv.Key}_{srcAlias}_to_335.csv");
                File.WriteAllText(midPerMapPath, kv.Value.ToString(), new UTF8Encoding(true));
            }
        }
        if (patchVia060.Length > 0 && chainVia060)
        {
            var patchPathVia060 = Path.Combine(compareV2Dir, $"Area_patch_crosswalk_via060_{srcAlias}_to_335.csv");
            File.WriteAllText(patchPathVia060, patchVia060.ToString(), new UTF8Encoding(true));
        }
        File.WriteAllText(patchFallbackPath, patchFallback.ToString(), new UTF8Encoding(true));
        if (chainVia060)
        {
            File.WriteAllText(Path.Combine(compareV2Dir, $"Area_chain_trace_{srcAlias}_to_335.csv"), trace.ToString(), new UTF8Encoding(true));
        }

        // Dump raw AreaTable rows for source, mid (0.6.0 pivot when present), and target into compare/v2 for inspection
        DumpAreaTable(storSrc_Area, compareV2Dir, srcAlias);
        if (has060 && stor060_Area is not null)
            DumpAreaTable(stor060_Area, compareV2Dir, "0.6.0");
        DumpAreaTable(storTgt_Area, compareV2Dir, "3.3.5");

        foreach (var kv in perMap)
        {
            File.WriteAllText(Path.Combine(compareV2Dir, $"AreaTable_mapping_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.map.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(compareV2Dir, $"AreaTable_unmatched_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.un.ToString(), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(compareV2Dir, $"Area_patch_crosswalk_map{kv.Key}_{srcAlias}_to_335.csv"), kv.Value.patch.ToString(), new UTF8Encoding(true));
            if (perMapFallback.TryGetValue(kv.Key, out var pfb))
                File.WriteAllText(Path.Combine(compareV2Dir, $"Area_patch_with_fallback0_map{kv.Key}_{srcAlias}_to_335.csv"), pfb.ToString(), new UTF8Encoding(true));
            if (chainVia060 && perMapVia060.TryGetValue(kv.Key, out var via))
                File.WriteAllText(Path.Combine(compareV2Dir, $"Area_patch_crosswalk_via060_map{kv.Key}_{srcAlias}_to_335.csv"), via.ToString(), new UTF8Encoding(true));
            if (chainVia060 && perMapTrace.TryGetValue(kv.Key, out var tmap))
                File.WriteAllText(Path.Combine(compareV2Dir, $"Area_chain_trace_map{kv.Key}_{srcAlias}_to_335.csv"), tmap.ToString(), new UTF8Encoding(true));
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

        // Generate Map.dbc metadata for source version
        try
        {
            var mapMetadata = Core.MapDbcReader.ReadMapDbc(
                srcDir!,
                dbdDir,
                CanonicalizeBuild(srcAlias),
                srcAlias,
                locale
            );
            var mapsJsonPath = Path.Combine(outAliasRoot, "maps.json");
            Core.MapDbcReader.WriteMapMetadata(mapMetadata, mapsJsonPath);
            Console.WriteLine($"[V2] Wrote Map.dbc metadata ({mapMetadata.Maps.Count} maps) to {mapsJsonPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[V2] WARN: Failed to generate Map.dbc metadata: {ex.Message}");
        }

        Console.WriteLine("[V2] Wrote mapping/unmatched/patch CSVs under compare/v2/ and audit CSVs under compare/.");
        return 0;
    }

    private static string BuildDisplayChain(string? zoneName, string? subName)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(zoneName)) parts.Add(zoneName.Trim());
        if (!string.IsNullOrWhiteSpace(subName)) parts.Add(subName.Trim());
        return string.Join('/', parts);
    }

    private readonly struct AreaNodeInfo
    {
        public AreaNodeInfo(int id, int parentId, int mapId, string name)
        {
            Id = id;
            ParentId = parentId;
            MapId = mapId;
            Name = name;
        }

        public int Id { get; }
        public int ParentId { get; }
        public int MapId { get; }
        public string Name { get; }
    }

    private static string BuildLkChainDisplay(int lkAreaId, Dictionary<int, DBCDRow> tgtIdToRow, string areaNameColTgt)
    {
        if (!tgtIdToRow.TryGetValue(lkAreaId, out var row))
        {
            return string.Empty;
        }

        int parentId = SafeField<int>(row, "ParentAreaID");
        if (parentId <= 0 || parentId == lkAreaId)
        {
            string name = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
            return BuildDisplayChain(name, null);
        }

        string zoneName = string.Empty;
        if (tgtIdToRow.TryGetValue(parentId, out var pRow))
        {
            zoneName = FirstNonEmpty(SafeField<string>(pRow, areaNameColTgt)) ?? string.Empty;
        }
        string subName = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
        return BuildDisplayChain(zoneName, subName);
    }

    private static (string childIds, string childNames) BuildTargetChildrenInfo(
        Dictionary<int, DBCDRow> tgtIdToRow,
        Dictionary<int, Dictionary<string, int>> idxTgtChildrenByZone,
        int chosenId,
        int tgtParentId,
        string areaNameColTgt)
    {
        IEnumerable<int> EnumerateChildrenForZone(int zoneId)
        {
            if (idxTgtChildrenByZone.TryGetValue(zoneId, out var dict))
            {
                foreach (var id in dict.Values.Distinct())
                {
                    yield return id;
                }
            }
        }

        List<int> childIds = new();
        if (idxTgtChildrenByZone.ContainsKey(chosenId))
        {
            childIds.AddRange(EnumerateChildrenForZone(chosenId));
        }
        else if (tgtParentId > 0 && tgtParentId != chosenId && idxTgtChildrenByZone.ContainsKey(tgtParentId))
        {
            childIds.AddRange(EnumerateChildrenForZone(tgtParentId));
        }

        if (childIds.Count == 0)
        {
            return (string.Empty, string.Empty);
        }

        childIds = childIds.Distinct().OrderBy(id => id).ToList();
        var childNames = new List<string>(childIds.Count);
        foreach (var cid in childIds)
        {
            string name = string.Empty;
            if (tgtIdToRow.TryGetValue(cid, out var row))
            {
                name = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
            }
            childNames.Add(name);
        }

        string idsStr = string.Join(';', childIds.Select(id => id.ToString(CultureInfo.InvariantCulture)));
        string namesStr = string.Join(';', childNames.Select(n => n ?? string.Empty));
        return (idsStr, namesStr);
    }

    private static string BuildHierarchyYaml(
        Dictionary<int, List<int>> zonesByMap,
        Dictionary<int, List<int>> childrenByParent,
        Dictionary<int, AreaNodeInfo> nodeInfo,
        Dictionary<int, string> mapNames)
    {
        var sb = new StringBuilder();
        sb.AppendLine("maps:");
        foreach (var mapId in zonesByMap.Keys.OrderBy(id => id))
        {
            mapNames.TryGetValue(mapId, out var mapName);
            sb.AppendLine($"  - mapId: {mapId}");
            sb.AppendLine($"    mapName: {YamlQuote(mapName)}");
            sb.AppendLine("    zones:");
            foreach (var zoneId in zonesByMap[mapId].Distinct().OrderBy(id => id))
            {
                if (!nodeInfo.TryGetValue(zoneId, out var zone)) continue;
                sb.AppendLine($"      - areaId: {zone.Id}");
                sb.AppendLine($"        name: {YamlQuote(zone.Name)}");
                sb.AppendLine($"        parentId: {zone.ParentId}");
                sb.AppendLine("        children:");
                if (childrenByParent.TryGetValue(zone.Id, out var children) && children.Count > 0)
                {
                    foreach (var childId in children.Distinct().OrderBy(id => id))
                    {
                        if (!nodeInfo.TryGetValue(childId, out var child)) continue;
                        sb.AppendLine($"          - areaId: {child.Id}");
                        sb.AppendLine($"            name: {YamlQuote(child.Name)}");
                        sb.AppendLine($"            parentId: {child.ParentId}");
                        sb.AppendLine($"            mapId: {child.MapId}");
                    }
                }
                else
                {
                    sb.AppendLine("          []");
                }
            }
        }
        return sb.ToString();
    }

    private static string YamlQuote(string? value)
    {
        value ??= string.Empty;
        var escaped = value.Replace("\\", "\\\\").Replace("\"", "\\\"");
        return $"\"{escaped}\"";
    }

    private static void PromoteAreaName(
        Dictionary<(int mapId, int areaNum), (string name, int priority)> dict,
        (int mapId, int areaNum) key,
        string candidate,
        int priority)
    {
        if (string.IsNullOrWhiteSpace(candidate)) return;
        candidate = candidate.Trim();
        if (!dict.TryGetValue(key, out var existing)
            || priority > existing.priority
            || (priority == existing.priority && string.IsNullOrWhiteSpace(existing.name)))
        {
            dict[key] = (candidate, priority);
        }
    }

    private static void PromoteZoneName(
        Dictionary<(int mapId, int zoneBase), (string name, int priority)> dict,
        (int mapId, int zoneBase) key,
        string candidate,
        int priority)
    {
        if (string.IsNullOrWhiteSpace(candidate)) return;
        candidate = candidate.Trim();
        if (!dict.TryGetValue(key, out var existing)
            || priority > existing.priority
            || (priority == existing.priority && string.IsNullOrWhiteSpace(existing.name)))
        {
            dict[key] = (candidate, priority);
        }
    }

    private static string ResolveCanonicalName<TKey>(Dictionary<TKey, (string name, int priority)> dict, TKey key)
        where TKey : notnull
    {
        if (dict.TryGetValue(key, out var entry) && !string.IsNullOrWhiteSpace(entry.name))
        {
            return entry.name;
        }
        return string.Empty;
    }

    private static bool TryInferMidFromCanonical(
        string zoneName,
        string areaName,
        Dictionary<int, HashSet<int>> zonesByMap,
        Dictionary<(int mapId, int zoneId), List<int>> childrenByParent,
        Dictionary<(int mapId, int zoneId), (string name, int priority)> zoneCanonical,
        Dictionary<(int mapId, int areaNum), (string name, int priority)> areaCanonical,
        out int mapId,
        out int zoneId,
        out int areaId)
    {
        mapId = -1;
        zoneId = -1;
        areaId = -1;
        if (string.IsNullOrWhiteSpace(zoneName)) return false;
        var zoneNorm = NormKey(zoneName);
        foreach (var kv in zonesByMap)
        {
            foreach (var z in kv.Value)
            {
                var canonical = ResolveCanonicalName(zoneCanonical, (kv.Key, z));
                if (string.IsNullOrWhiteSpace(canonical)) continue;
                if (NormKey(canonical) != zoneNorm) continue;
                mapId = kv.Key;
                zoneId = z;
                if (!string.IsNullOrWhiteSpace(areaName)
                    && childrenByParent.TryGetValue((mapId, zoneId), out var children))
                {
                    var areaNorm = NormKey(areaName);
                    foreach (var child in children)
                    {
                        var childName = ResolveCanonicalName(areaCanonical, (mapId, child));
                        if (!string.IsNullOrWhiteSpace(childName) && NormKey(childName) == areaNorm)
                        {
                            areaId = child;
                            return true;
                        }
                    }
                }
                areaId = -1;
                return true;
            }
        }
        return false;
    }

    private static string ResolveTargetAreaName(Dictionary<int, DBCDRow> tgtIdToRow, int areaId, string areaNameColTgt)
    {
        if (areaId <= 0) return string.Empty;
        if (tgtIdToRow.TryGetValue(areaId, out var row))
        {
            return FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;
        }
        return string.Empty;
    }

    private static string BuildSourceHierarchyYaml(
        Dictionary<int, HashSet<int>> zonesByMap,
        Dictionary<(int mapId, int zoneBase), List<int>> childrenByParent,
        Dictionary<(int mapId, int areaNum), AreaNodeInfo> nodeInfo,
        Dictionary<(int mapId, int zoneBase), (string name, int priority)> zoneCanonical,
        Dictionary<(int mapId, int areaNum), (string name, int priority)> areaCanonical,
        Dictionary<int, string> mapNames)
    {
        var sb = new StringBuilder();
        sb.AppendLine("maps:");
        foreach (var mapId in zonesByMap.Keys.OrderBy(id => id))
        {
            mapNames.TryGetValue(mapId, out var mapName);
            sb.AppendLine($"  - mapId: {mapId}");
            sb.AppendLine($"    mapName: {YamlQuote(mapName)}");

            if (!zonesByMap.TryGetValue(mapId, out var zoneSet) || zoneSet.Count == 0)
            {
                sb.AppendLine("    zones:");
                sb.AppendLine("      []");
                continue;
            }

            sb.AppendLine("    zones:");
            foreach (var zoneBase in zoneSet.OrderBy(z => z))
            {
                var zoneKey = (mapId, zoneBase);
                string zoneName = ResolveCanonicalName(zoneCanonical, zoneKey);
                int zoneParentId = zoneBase;
                if (nodeInfo.TryGetValue(zoneKey, out var zoneInfo))
                {
                    if (!string.IsNullOrWhiteSpace(zoneInfo.Name) && string.IsNullOrWhiteSpace(zoneName))
                    {
                        zoneName = zoneInfo.Name;
                    }
                    zoneParentId = zoneInfo.ParentId;
                }

                sb.AppendLine($"      - areaId: {zoneBase}");
                sb.AppendLine($"        name: {YamlQuote(zoneName)}");
                sb.AppendLine($"        parentId: {zoneParentId}");
                sb.AppendLine("        children:");

                var childKey = (mapId, zoneBase);
                if (childrenByParent.TryGetValue(childKey, out var children) && children.Count > 0)
                {
                    foreach (var childArea in children.Distinct().OrderBy(a => a))
                    {
                        var childNodeKey = (mapId, childArea);
                        string childName = ResolveCanonicalName(areaCanonical, childNodeKey);
                        int childParentId = zoneBase;
                        int childMapId = mapId;
                        if (nodeInfo.TryGetValue(childNodeKey, out var childInfo))
                        {
                            if (!string.IsNullOrWhiteSpace(childInfo.Name) && string.IsNullOrWhiteSpace(childName))
                            {
                                childName = childInfo.Name;
                            }
                            childParentId = childInfo.ParentId;
                            childMapId = childInfo.MapId;
                        }

                        sb.AppendLine($"          - areaId: {childArea}");
                        sb.AppendLine($"            name: {YamlQuote(childName)}");
                        sb.AppendLine($"            parentId: {childParentId}");
                        sb.AppendLine($"            mapId: {childMapId}");
                    }
                }
                else
                {
                    sb.AppendLine("          []");
                }
            }
        }

        return sb.ToString();
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

    private static void WriteHierarchyPairings(string path, IReadOnlyList<HierarchyPairingCandidate> pairings)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.WriteLine("src_mapId,src_areaId,src_parentId,src_isZone,src_name,match_count,matches");
        foreach (var candidate in pairings)
        {
            var matches = BuildMatchSummary(candidate.Matches);
            writer.WriteLine(string.Join(',', new[]
            {
                candidate.SrcMapId.ToString(CultureInfo.InvariantCulture),
                candidate.SrcAreaId.ToString(CultureInfo.InvariantCulture),
                candidate.SrcParentId.ToString(CultureInfo.InvariantCulture),
                candidate.SrcIsZone ? "1" : "0",
                Csv(candidate.SrcName),
                candidate.Matches.Count.ToString(CultureInfo.InvariantCulture),
                Csv(matches)
            }));
        }
    }

    private static void WriteHierarchyPairingsUnmatched(string path, IReadOnlyList<HierarchyPairingCandidate> pairings)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.WriteLine("src_mapId,src_areaId,src_parentId,src_isZone,src_name");
        foreach (var candidate in pairings)
        {
            if (candidate.Matches.Count > 0) continue;
            writer.WriteLine(string.Join(',', new[]
            {
                candidate.SrcMapId.ToString(CultureInfo.InvariantCulture),
                candidate.SrcAreaId.ToString(CultureInfo.InvariantCulture),
                candidate.SrcParentId.ToString(CultureInfo.InvariantCulture),
                candidate.SrcIsZone ? "1" : "0",
                Csv(candidate.SrcName)
            }));
        }
    }

    private static string BuildMatchSummary(IReadOnlyList<HierarchyPairingMatch> matches)
    {
        if (matches.Count == 0) return string.Empty;
        var parts = new string[matches.Count];
        for (int i = 0; i < matches.Count; i++)
        {
            var match = matches[i];
            var state = match.IsUnused ? "unused" : "active";
            parts[i] = $"{match.MapId}:{match.AreaId}:{match.ParentAreaId}:{match.Reason}:{state}:{match.Name}";
        }
        return string.Join('|', parts);
    }

    private static (int MapId, int AreaId, string Method, string Note, bool Ambiguous)? SelectHierarchyMatch(HierarchyPairingCandidate candidate, int preferredMapId, int fallbackMapId, bool hasPreferred)
    {
        if (candidate.Matches.Count == 0) return null;

        var ordered = candidate.Matches
            .Select(m => new
            {
                Match = m,
                Priority = ScoreMatch(m, preferredMapId, fallbackMapId, hasPreferred)
            })
            .Where(x => x.Priority >= 0)
            .OrderBy(x => x.Priority)
            .ThenBy(x => x.Match.IsUnused ? 1 : 0)
            .ToList();

        if (ordered.Count == 0) return null;

        var best = ordered[0];
        bool ambiguous = ordered.Count > 1 && ordered[1].Priority == best.Priority;
        string note = $"hierarchy={best.Match.Reason}";
        if (best.Match.IsUnused) note += "(unused)";
        return (best.Match.MapId, best.Match.AreaId, best.Match.Reason, note, ambiguous);
    }

    private static int ScoreMatch(HierarchyPairingMatch match, int preferredMapId, int fallbackMapId, bool hasPreferred)
    {
        int baseScore = match.Reason.StartsWith("map", StringComparison.OrdinalIgnoreCase) ? 0 : 10;
        if (hasPreferred)
        {
            if (match.MapId == preferredMapId) return baseScore;
            return baseScore + 5;
        }

        if (match.MapId == fallbackMapId) return baseScore + 1;
        if (match.MapId == preferredMapId) return baseScore;
        if (fallbackMapId >= 0) return baseScore + 4;
        return baseScore + 6;
    }

    private static IEnumerable<HierarchyPairingGenerator.HierarchySourceRecord> BuildSourceGraph(IDBCDStorage storSrc_Area, IDBCDStorage storSrc_Map, string areaNameColSrc, string parentColSrc, string keyColSrc)
    {
        var mapNames = BuildMapNames(storSrc_Map);
        var nodesByMap = new Dictionary<int, List<HierarchyPairingGenerator.HierarchyNodeRecord>>();

        foreach (var key in storSrc_Area.Keys)
        {
            var row = storSrc_Area[key];
            int areaNum = SafeField<int>(row, keyColSrc);
            int parentNum = SafeField<int>(row, parentColSrc);
            if (parentNum <= 0) parentNum = areaNum;
            int mapId = SafeField<int>(row, "ContinentID");
            string name = FirstNonEmpty(SafeField<string>(row, areaNameColSrc)) ?? string.Empty;

            if (!nodesByMap.TryGetValue(mapId, out var list))
            {
                list = new List<HierarchyPairingGenerator.HierarchyNodeRecord>();
                nodesByMap[mapId] = list;
            }
            list.Add(new HierarchyPairingGenerator.HierarchyNodeRecord(areaNum, parentNum, name));
        }

        foreach (var kvp in nodesByMap)
        {
            mapNames.TryGetValue(kvp.Key, out var mapName);
            yield return new HierarchyPairingGenerator.HierarchySourceRecord(kvp.Key, mapName, kvp.Value);
        }
    }

    private static IEnumerable<HierarchyPairingGenerator.HierarchySourceRecord> BuildTargetGraph(IDBCDStorage storTgt_Area, IDBCDStorage storTgt_Map, string areaNameColTgt)
    {
        var mapNames = BuildMapNames(storTgt_Map);
        var nodesByMap = new Dictionary<int, List<HierarchyPairingGenerator.HierarchyNodeRecord>>();

        foreach (var key in storTgt_Area.Keys)
        {
            var row = storTgt_Area[key];
            int id = SafeField<int>(row, "ID");
            int parentId = SafeField<int>(row, "ParentAreaID");
            if (parentId <= 0) parentId = id;
            int mapId = SafeField<int>(row, "ContinentID");
            string name = FirstNonEmpty(SafeField<string>(row, areaNameColTgt)) ?? string.Empty;

            if (!nodesByMap.TryGetValue(mapId, out var list))
            {
                list = new List<HierarchyPairingGenerator.HierarchyNodeRecord>();
                nodesByMap[mapId] = list;
            }
            list.Add(new HierarchyPairingGenerator.HierarchyNodeRecord(id, parentId, name));
        }

        foreach (var kvp in nodesByMap)
        {
            mapNames.TryGetValue(kvp.Key, out var mapName);
            yield return new HierarchyPairingGenerator.HierarchySourceRecord(kvp.Key, mapName, kvp.Value);
        }
    }

    private static IEnumerable<HierarchyPairingMatch> BuildSupplementalMatches(AreaHierarchyGraph srcGraph, IReadOnlyDictionary<string, ResolvedRenameOverride> overrides)
    {
        var matches = new List<HierarchyPairingMatch>();
        if (srcGraph is null || overrides.Count == 0) return matches;

        var srcByKey = new Dictionary<string, List<AreaHierarchyNode>>(StringComparer.OrdinalIgnoreCase);
        foreach (var node in srcGraph.EnumerateNodes())
        {
            var key = NormKey(node.Name);
            if (string.IsNullOrEmpty(key)) continue;
            if (!srcByKey.TryGetValue(key, out var list))
            {
                list = new List<AreaHierarchyNode>();
                srcByKey[key] = list;
            }
            list.Add(node);
        }

        foreach (var kvp in overrides)
        {
            if (!srcByKey.TryGetValue(kvp.Key, out var sources)) continue;
            var resolved = kvp.Value;
            foreach (var srcNode in sources)
            {
                matches.Add(new HierarchyPairingMatch(
                    resolved.MapId,
                    resolved.AreaId,
                    resolved.TargetName,
                    resolved.ParentAreaId,
                    $"alias:{resolved.From}->{resolved.TargetName}",
                    resolved.IsTargetUnused));
            }
        }

        return matches;
    }

    private static List<RenameAliasConfig> LoadRenameAliasConfigs(string compareV3Dir)
    {
        var configs = new Dictionary<string, RenameAliasConfig>(StringComparer.OrdinalIgnoreCase)
        {
            ["Lik'ash Tar Pits"] = new("Lik'ash Tar Pits", "Lakkari Tar Pits", 269, 3706),
            ["Lik'kari Tar Pits"] = new("Lik'kari Tar Pits", "Lakkari Tar Pits", 269, 3706),
            ["Likkari Tar Pits"] = new("Likkari Tar Pits", "Lakkari Tar Pits", 269, 3706),
            ["Tranquil Gardens Cemetary"] = new("Tranquil Gardens Cemetary", "Tranquil Gardens Cemetery", 0, 121)
        };

        var candidatePaths = new List<string>();
        if (!string.IsNullOrWhiteSpace(compareV3Dir))
        {
            candidatePaths.Add(Path.Combine(compareV3Dir, "Area_rename_overrides.csv"));
        }

        var baseDir = AppContext.BaseDirectory;
        if (!string.IsNullOrWhiteSpace(baseDir))
        {
            candidatePaths.Add(Path.Combine(baseDir, "Area_rename_overrides.csv"));
            candidatePaths.Add(Path.Combine(baseDir, "Config", "Area_rename_overrides.csv"));
        }

        foreach (var pathCandidate in candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!File.Exists(pathCandidate)) continue;
            try
            {
                LoadRenameAliasCsv(pathCandidate, configs);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[V3] WARN: Failed to load rename overrides from '{pathCandidate}': {ex.Message}");
            }
        }

        return configs.Values.ToList();
    }

    private static void LoadRenameAliasCsv(string path, Dictionary<string, RenameAliasConfig> configs)
    {
        using var reader = new StreamReader(path);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#')) continue;

            var parts = SplitCsvLine(trimmed);
            if (parts.Count < 2) continue;

            var from = parts[0].Trim().Trim('"');
            var to = parts[1].Trim().Trim('"');
            if (string.IsNullOrEmpty(from) || string.IsNullOrEmpty(to)) continue;

            int? targetMap = null;
            int? targetArea = null;

            if (parts.Count > 2 && int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var mapId))
            {
                targetMap = mapId;
            }

            if (parts.Count > 3 && int.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out var areaId))
            {
                targetArea = areaId;
            }

            configs[from] = new RenameAliasConfig(from, to, targetMap, targetArea);
        }
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

    private static IReadOnlyDictionary<string, ResolvedRenameOverride> ResolveRenameOverrides(IEnumerable<RenameAliasConfig> configs, AreaHierarchyGraph tgtGraph)
    {
        var result = new Dictionary<string, ResolvedRenameOverride>(StringComparer.OrdinalIgnoreCase);
        if (configs is null || tgtGraph is null) return result;

        var mapLookup = new Dictionary<int, Dictionary<string, AreaHierarchyNode>>();
        var globalLookup = new Dictionary<string, AreaHierarchyNode>(StringComparer.OrdinalIgnoreCase);
        var ambiguous = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var map in tgtGraph.Maps)
        {
            var dict = new Dictionary<string, AreaHierarchyNode>(StringComparer.OrdinalIgnoreCase);
            foreach (var node in map.Zones.SelectMany(z => z.SelfAndDescendants()))
            {
                var key = NormKey(node.Name);
                if (string.IsNullOrEmpty(key)) continue;

                if (!dict.ContainsKey(key))
                {
                    dict[key] = node;
                }

                if (!ambiguous.Contains(key))
                {
                    if (!globalLookup.TryGetValue(key, out var existing))
                    {
                        globalLookup[key] = node;
                    }
                    else if (existing.AreaId != node.AreaId)
                    {
                        globalLookup.Remove(key);
                        ambiguous.Add(key);
                    }
                }
            }
            mapLookup[map.MapId] = dict;
        }

        foreach (var config in configs)
        {
            var fromKey = NormKey(config.From);
            if (string.IsNullOrEmpty(fromKey)) continue;

            AreaHierarchyNode? target = null;
            if (config.TargetAreaId.HasValue)
            {
                target = tgtGraph.FindArea(config.TargetAreaId.Value);
            }

            if (target is null)
            {
                var toKey = NormKey(config.To);
                if (!string.IsNullOrEmpty(toKey))
                {
                    if (config.TargetMapId.HasValue && mapLookup.TryGetValue(config.TargetMapId.Value, out var perMap) && perMap.TryGetValue(toKey, out var mapNode))
                    {
                        target = mapNode;
                    }
                    else if (globalLookup.TryGetValue(toKey, out var globalNode))
                    {
                        target = globalNode;
                    }
                }
            }

            if (target is null)
            {
                Console.WriteLine($"[V3] WARN: rename alias '{config.From}' -> '{config.To}' could not resolve target area.");
                continue;
            }

            result[fromKey] = new ResolvedRenameOverride(
                config.From,
                target.Name,
                fromKey,
                target.MapId,
                target.AreaId,
                target.Parent?.AreaId ?? target.AreaId,
                target.IsUnused);
        }

        return result;
    }

    private sealed record RenameAliasConfig(string From, string To, int? TargetMapId, int? TargetAreaId);

    private sealed record ResolvedRenameOverride(string From, string TargetName, string NormalizedFrom, int MapId, int AreaId, int ParentAreaId, bool IsTargetUnused);
}
