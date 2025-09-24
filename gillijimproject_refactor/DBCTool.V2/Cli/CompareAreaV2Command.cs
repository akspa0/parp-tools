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
        var header = string.Join(',', new[]
        {
            "src_row_id","src_areaNumber","src_parentNumber","src_zone_hi16","src_sub_lo16","src_parent_hi16","src_parent_lo16",
            "src_name","src_mapId","src_mapName","src_mapId_xwalk","src_mapName_xwalk","src_path",
            "tgt_id_335","tgt_name","tgt_parent_id","tgt_parent_name","tgt_mapId","tgt_mapName","tgt_path","match_method"
        });
        mapping.AppendLine(header);
        unmatched.AppendLine(header);
        var patchHeader = "src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name";
        patch.AppendLine(patchHeader);
        var patchHeaderVia060 = "src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name,mid060_mapId,mid060_mapName,mid060_areaID,mid060_parentID,mid060_name,tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name";
        patchVia060.AppendLine(patchHeaderVia060);
        var traceHeader = "src_row_id,src_areaNumber,src_parentNumber,src_name,src_mapId,src_chain,pivot_mapId,pivot_chain,lk_mapId,lk_chain,chosen_tgt_id,matched_depth,method";
        trace.AppendLine(traceHeader);
        var patchFallback = new StringBuilder();
        patchFallback.AppendLine(patchHeader);
        var perMap = new Dictionary<int, (StringBuilder map, StringBuilder un, StringBuilder patch)>();
        var perMapFallback = new Dictionary<int, StringBuilder>();
        var perMapVia060 = new Dictionary<int, StringBuilder>();
        var perMapTrace = new Dictionary<int, StringBuilder>();

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
            string lkChainDisplay = string.Empty;
            string pivotChainDisplay = string.Empty;
            if (!chainVia060 && hasMapX && chain.Count > 0)
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
            string pivotChainDesc = string.Empty; string lkChainDesc = string.Empty;
            if (((chainVia060) || (chosen < 0 || depth < chain.Count)) && has060 && chain.Count > 0)
            {
                if (cwSrcTo060.TryGetValue(contResolved, out var mx060)) { pivotMapIdX = mx060; hasPivot = true; }
                else if (map060Names.ContainsKey(contResolved)) { pivotMapIdX = contResolved; hasPivot = true; }
                if (hasPivot)
                {
                    // Attempt strict child resolution on pivot map first when source chain is zone-only or oddity
                    string srcKeyNorm = NormKey(nm);
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
                                }
                            }
                        }
                    }
                }
            }

            // Global rename match: unique top-level zone across LK (cross-map allowed) - disabled in chainVia060 mode
            if (!chainVia060 && chosen < 0 && chain.Count > 0)
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

            // Global rename match for child names (use the current row's own name) - disabled in chainVia060 mode
            if (!chainVia060 && chosen < 0 && !string.IsNullOrWhiteSpace(nm))
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
            if (!chainVia060 && chosen < 0 && chain.Count > 0)
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

            // Fuzzy rename (child) using row name if still unmatched - disabled in chainVia060 mode
            if (!chainVia060 && chosen < 0 && !string.IsNullOrWhiteSpace(nm))
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
            if (!chainVia060 && chosen < 0)
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
                // Promote to top-level zone when the matched record is a child
                bool promotedToParent = false;
                if (tgtParentId <= 0) tgtParentId = chosen;
                string tgtParentName = tgtName;
                if (area_lo16 == 0 && tgtParentId != chosen && tgtParentId > 0 && tgtIdToRow.TryGetValue(tgtParentId, out var pRow))
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
                else
                {
                    tgtParentName = tgtName;
                    tgtParentId = chosen;
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
                }
                else if (area_lo16 > 0 && !method.Contains("_sub", StringComparison.OrdinalIgnoreCase))
                {
                    method = $"{method}_sub";
                }

                lkChainDisplay = BuildLkChainDisplay(chosen, tgtIdToRow, areaNameColTgt);

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
                            Csv(tgtName)
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
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    "-1",
                    string.Empty,
                    string.Empty,
                    "unmatched_zone"
                });
                unmatched.AppendLine(line);
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
