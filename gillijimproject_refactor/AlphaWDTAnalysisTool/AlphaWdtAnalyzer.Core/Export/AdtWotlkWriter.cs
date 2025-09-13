using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GillijimProject.WowFiles.Alpha;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtWotlkWriter
{
    private static readonly HashSet<string> InitializedCsv = new(StringComparer.OrdinalIgnoreCase);
    private static readonly HashSet<string> EmittedWdtForMap = new(StringComparer.OrdinalIgnoreCase);

    public sealed class WriteContext
    {
        public required string ExportDir { get; init; }
        public required string MapName { get; init; }
        public required int TileX { get; init; }
        public required int TileY { get; init; }
        public required IEnumerable<PlacementRecord> Placements { get; init; }
        public required AssetFixupPolicy Fixup { get; init; }
        public bool ConvertToMh2o { get; init; }
        public AreaIdMapper? AreaMapper { get; init; }
        public IReadOnlyList<int>? AlphaAreaIds { get; init; }
        public required string WdtPath { get; init; }
        public required int AdtNumber { get; init; }
        public required int AdtOffset { get; init; }
        public required IReadOnlyList<string> MdnmFiles { get; init; }
        public required IReadOnlyList<string> MonmFiles { get; init; }
        public bool Verbose { get; init; } = false;
        public bool TrackAssets { get; init; } = false;
        public int? CurrentMapId { get; init; }
        // Optional version metadata for CSV dumps
        public string? SrcAlias { get; init; }
        public string? SrcBuild { get; init; }
        public string? TgtBuild { get; init; }
        // Remap-only write guard: when false (no --remap), AreaIDs are NOT written
        public bool AllowAreaIdWrites { get; init; } = false;
        // Optional decode dictionaries (Alpha/LK AreaTable-backed)
        public IReadOnlyDictionary<int, string>? AlphaAreaNamesByNumber { get; init; }
        public IReadOnlyDictionary<int, int>? AlphaParentByNumber { get; init; }
        public IReadOnlyDictionary<int, int>? AlphaContinentByNumber { get; init; }
        public IReadOnlyDictionary<int, string>? LkAreaNamesById { get; init; }
        public IReadOnlyDictionary<int, int>? LkParentById { get; init; }
        public IReadOnlyDictionary<int, int>? LkContinentById { get; init; }
        // Alpha per-continent disambiguation (areaNumber -> continentID -> name/parent)
        public IReadOnlyDictionary<int, IReadOnlyDictionary<int, string>>? AlphaAreaNamesByNumberByCont { get; init; }
        public IReadOnlyDictionary<int, IReadOnlyDictionary<int, int>>? AlphaParentByNumberByCont { get; init; }
        // Diagnostics-only mode: skip ADT file writing/patching for speed
        public bool DiagnosticsOnly { get; init; } = false;
        // Per-tile sample size
        public int SampleCount { get; init; } = 16;
    }

    public static void WritePlaceholder(WriteContext ctx)
    {
        Directory.CreateDirectory(ctx.ExportDir);
        var file = Path.Combine(ctx.ExportDir, $"{ctx.MapName}_{ctx.TileX}_{ctx.TileY}.adt.placeholder.txt");
        using var sw = new StreamWriter(file);
        sw.WriteLine($"Map={ctx.MapName} Tile=({ctx.TileX},{ctx.TileY})");
        sw.WriteLine($"ConvertToMh2o={ctx.ConvertToMh2o}");

        // Area ID mapping summary and CSV (emit whenever we have AlphaAreaIds)
        if (ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
        {
            int mapped = 0, unmapped = 0, present = 0;
            for (int i = 0; i < 256; i++)
            {
                var aId = ctx.AlphaAreaIds[i];
                if (aId < 0) continue;
                present++;
                bool rowMapped = false;
                if (ctx.AreaMapper is not null)
                {
                    if (ctx.AreaMapper.TryMapDetailed(aId, out _, out _, out _, out _))
                    {
                        rowMapped = true;
                    }
                }
                if (rowMapped) mapped++; else unmapped++;
            }
            sw.WriteLine($"AreaIds: present={present} mapped={mapped} unmapped={unmapped}");
        }

        sw.WriteLine("Placements (with potential fixups):");
        foreach (var p in ctx.Placements.OrderBy(p => p.Type).ThenBy(p => p.AssetPath))
        {
            var type = p.Type;
            var path = p.AssetPath;
            var fixedPath = type switch
            {
                AssetType.Wmo => ctx.Fixup.Resolve(AssetType.Wmo, path),
                AssetType.MdxOrM2 => ctx.Fixup.Resolve(AssetType.MdxOrM2, path),
                _ => path
            };
            var flag = (fixedPath.Equals(path, StringComparison.OrdinalIgnoreCase)) ? "ok" : $"fixed -> {fixedPath}";
            sw.WriteLine($"  {type}: {path} [{flag}] UniqueId={p.UniqueId?.ToString() ?? ""}");
        }
        sw.WriteLine();
        sw.WriteLine("NOTE: This is a placeholder. Binary WotLK ADT writing will be implemented next.");
    }

    public static void WriteBinary(WriteContext ctx)
    {
        // Output to World/Maps/<map>/
        var mapsDir = Path.Combine(ctx.ExportDir, "World", "Maps", ctx.MapName);
        Directory.CreateDirectory(mapsDir);
        var outFile = Path.Combine(mapsDir, $"{ctx.MapName}_{ctx.TileX}_{ctx.TileY}.adt");

        // Build Alpha ADT handle once (also used to enumerate MTEX textures)
        var alpha = new AdtAlpha(ctx.WdtPath, ctx.AdtOffset, ctx.AdtNumber);

        // Before conversion, record any placements or textures we could not resolve at all
        if (ctx.TrackAssets)
        {
            var missingPath = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName, "missing_assets.csv");
            Directory.CreateDirectory(Path.GetDirectoryName(missingPath)!);
            using (var missing = new MissingAssetsLogger(missingPath))
            {
                // placements (WMO/M2)
                foreach (var p in ctx.Placements)
                {
                    var _ = ctx.Fixup.ResolveWithMethod(p.Type, p.AssetPath, out var method);
                    if (string.Equals(method, "preserve_missing", StringComparison.OrdinalIgnoreCase))
                    {
                        missing.Write(new MissingAssetRecord
                        {
                            Type = p.Type.ToString(),
                            Original = p.AssetPath,
                            MapName = p.MapName,
                            TileX = p.TileX,
                            TileY = p.TileY,
                            UniqueId = p.UniqueId
                        });
                    }
                }

                // textures (BLP via MTEX)
                foreach (var tex in alpha.GetMtexTextureNames())
                {
                    var norm = ListfileLoader.NormalizePath(tex);
                    if (string.IsNullOrWhiteSpace(norm)) continue;
                    var _ = ctx.Fixup.ResolveTextureWithMethod(norm, out var method);
                    if (string.Equals(method, "preserve_missing", StringComparison.OrdinalIgnoreCase))
                    {
                        missing.Write(new MissingAssetRecord
                        {
                            Type = AssetType.Blp.ToString(),
                            Original = norm,
                            MapName = ctx.MapName,
                            TileX = ctx.TileX,
                            TileY = ctx.TileY,
                            UniqueId = null
                        });
                    }
                }
            }
        }

        if (!ctx.DiagnosticsOnly)
        {
            // Build LK ADT from Alpha using WDT MDNM/MONM tables
            var fixedM2 = ctx.MdnmFiles.Select(n => ctx.Fixup.Resolve(AssetType.MdxOrM2, n)).ToList();
            var fixedWmo = ctx.MonmFiles.Select(n => ctx.Fixup.Resolve(AssetType.Wmo, n)).ToList();
            var adtLk = alpha.ToAdtLk(fixedM2, fixedWmo);
            adtLk.ToFile(outFile);

            // Patch MTEX in-place with capacity-aware replacements (do not change file size)
            try { PatchMtexOnDiskInternal(outFile, ctx.Fixup); }
            catch (Exception ex) { Console.Error.WriteLine($"[MTEX] Failed to patch MTEX for {outFile}: {ex.Message}"); }

            // Patch MMDX (M2/MDX) and MWMO (WMO) name tables in-place
            try { PatchStringTableInPlace(outFile, "MMDX", AssetType.MdxOrM2, ctx.Fixup, (orig) => ctx.Fixup.ResolveWithMethod(AssetType.MdxOrM2, orig, out _)); }
            catch (Exception ex) { Console.Error.WriteLine($"[MMDX] Failed to patch M2/MDX names for {outFile}: {ex.Message}"); }
            try { PatchStringTableInPlace(outFile, "MWMO", AssetType.Wmo, ctx.Fixup, (orig) => ctx.Fixup.ResolveWithMethod(AssetType.Wmo, orig, out _)); }
            catch (Exception ex) { Console.Error.WriteLine($"[MWMO] Failed to patch WMO names for {outFile}: {ex.Message}"); }

            // Emit/refresh WDT once per map in the same folder, rename *_new to <map>.wdt
            if (!EmittedWdtForMap.Contains(ctx.MapName))
            {
                try
                {
                    var wdtAlpha = new WdtAlpha(ctx.WdtPath);
                    var wdt = wdtAlpha.ToWdt();
                    wdt.ToFile(mapsDir); // writes <basename>.wdt_new
                    var newFile = Path.Combine(mapsDir, Path.GetFileName(ctx.WdtPath) + "_new");
                    var finalFile = Path.Combine(mapsDir, ctx.MapName + ".wdt");
                    if (File.Exists(finalFile)) File.Delete(finalFile);
                    if (File.Exists(newFile)) File.Move(newFile, finalFile, overwrite: true);
                    EmittedWdtForMap.Add(ctx.MapName);
                }
                catch (Exception ex) { Console.Error.WriteLine($"[WDT] Failed to emit WDT for {ctx.MapName}: {ex.Message}"); }
            }
        }

        // Patch per-MCNK AreaId in-place using mapper when available (alpha-driven via MCIN)
        if (ctx.AllowAreaIdWrites && ctx.AreaMapper is not null && ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
        {
            try
            {
                var (present, patched, ignored) = PatchMcnkAreaIdsOnDisk(outFile, ctx.AlphaAreaIds, ctx.AreaMapper, ctx.Verbose, ctx.CurrentMapId);
                if (ctx.Verbose)
                {
                    Console.WriteLine($"[{ctx.MapName} {ctx.TileX},{ctx.TileY}] AreaIds: present={present} patched={patched} ignored={ignored}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[AreaPatch] Failed to patch AreaIDs for {outFile}: {ex.Message}");
            }

            // Emit verification CSVs: per-map remap dump(s) and per-tile sample with readbacks
            try
            {
                WriteRemapDumpCsv(ctx, ctx.AlphaAreaIds, ctx.AreaMapper);
                WriteRemapExplicitUsedCsv(ctx, ctx.AlphaAreaIds, ctx.AreaMapper);
                WriteAreaIdSampleCsv(ctx, outFile, ctx.AlphaAreaIds, ctx.AreaMapper);
            }
            catch (Exception ex)
            {
                if (ctx.Verbose)
                {
                    Console.Error.WriteLine($"[AreaCSV] Failed to write areaid verification CSVs for {outFile}: {ex.Message}");
                }
            }
        }
        else if (ctx.Verbose)
        {
            Console.WriteLine($"[{ctx.MapName} {ctx.TileX},{ctx.TileY}] Diagnostics-only: AreaIDs not written (no --remap). Remap CSVs suppressed.");
        }

        // Decode-first diagnostics: emit alpha decode proof and LK suggestions when dictionaries are available
        if (ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256 &&
            ctx.AlphaAreaNamesByNumber is not null && ctx.AlphaParentByNumber is not null && ctx.AlphaContinentByNumber is not null &&
            ctx.LkAreaNamesById is not null && ctx.LkParentById is not null && ctx.LkContinentById is not null)
        {
            try { WriteAlphaAreaDecodeCsv(ctx, ctx.AlphaAreaIds); }
            catch (Exception ex) { if (ctx.Verbose) Console.Error.WriteLine($"[DecodeCSV] {ctx.MapName} {ctx.TileX},{ctx.TileY}: {ex.Message}"); }
            try { WriteAlphaTo335SuggestionsCsv(ctx, ctx.AlphaAreaIds); }
            catch (Exception ex) { if (ctx.Verbose) Console.Error.WriteLine($"[SuggestCSV] {ctx.MapName} {ctx.TileX},{ctx.TileY}: {ex.Message}"); }
        }

        // Always emit per-tile sample (enriched) for quick inspection
        if (ctx.AlphaAreaIds is not null && ctx.AlphaAreaIds.Count == 256)
        {
            try { WriteAreaIdSampleCsv(ctx, outFile, ctx.AlphaAreaIds, ctx.AreaMapper); }
            catch (Exception ex) { if (ctx.Verbose) Console.Error.WriteLine($"[SampleCSV] {ctx.MapName} {ctx.TileX},{ctx.TileY}: {ex.Message}"); }
        }
    }

    private static (int zone, int subzone) DecodeZoneSubzone(int alphaRaw)
    {
        int zone = (alphaRaw >> 16) & 0xFFFF;
        int sub = alphaRaw & 0xFFFF;
        return (zone, sub);
    }

    private static string Norm(string s) => (s ?? string.Empty).Trim().ToLowerInvariant();

    private static readonly HashSet<string> AlphaDecodeKeys = new(StringComparer.OrdinalIgnoreCase);

    // Continent-aware alpha helpers
    private static bool TryGetAlphaNameByCont(WriteContext ctx, int areaNum, int? mapId, bool strict, out string name)
    {
        name = string.Empty;
        // Prefer per-continent dictionary if available and mapId is known
        if (mapId.HasValue && ctx.AlphaAreaNamesByNumberByCont is not null &&
            ctx.AlphaAreaNamesByNumberByCont.TryGetValue(areaNum, out var perCont) &&
            perCont.TryGetValue(mapId.Value, out var nm))
        {
            name = nm ?? string.Empty;
            return name.Length > 0;
        }
        // Fallback: global dictionary (first-encountered) only when not strict or mapId is unknown
        if (!strict || !mapId.HasValue)
        {
            if (ctx.AlphaAreaNamesByNumber is not null && ctx.AlphaAreaNamesByNumber.TryGetValue(areaNum, out var nm2))
            {
                name = nm2 ?? string.Empty;
                return name.Length > 0;
            }
        }
        return false;
    }

    private static bool TryGetAlphaParentByCont(WriteContext ctx, int areaNum, int? mapId, bool strict, out int parentAreaNum)
    {
        parentAreaNum = 0;
        if (mapId.HasValue && ctx.AlphaParentByNumberByCont is not null &&
            ctx.AlphaParentByNumberByCont.TryGetValue(areaNum, out var perCont) &&
            perCont.TryGetValue(mapId.Value, out var p))
        {
            parentAreaNum = p;
            return true;
        }
        if (!strict || !mapId.HasValue)
        {
            if (ctx.AlphaParentByNumber is not null && ctx.AlphaParentByNumber.TryGetValue(areaNum, out var p2))
            {
                parentAreaNum = p2;
                return true;
            }
        }
        return false;
    }

    private static int GetAlphaContForArea(WriteContext ctx, int areaNum, int? mapId)
    {
        // If we have a per-continent entry for this area and a current mapId, prefer it
        if (mapId.HasValue && ctx.AlphaAreaNamesByNumberByCont is not null &&
            ctx.AlphaAreaNamesByNumberByCont.TryGetValue(areaNum, out var perCont) &&
            perCont.ContainsKey(mapId.Value))
        {
            return mapId.Value;
        }
        // Fallback to the raw AlphaContinentByNumber table
        if (ctx.AlphaContinentByNumber is not null && ctx.AlphaContinentByNumber.TryGetValue(areaNum, out var c))
        {
            return c;
        }
        return -1;
    }

    private static void WriteAlphaAreaDecodeCsv(WriteContext ctx, IReadOnlyList<int> alphaAreaIds)
    {
        var dir = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName);
        Directory.CreateDirectory(dir);
        var file = Path.Combine(dir, "alpha_areaid_decode.csv");

        bool appendDecode = InitializedCsv.Contains(file);
        using var sw = new StreamWriter(file, append: appendDecode);
        if (!appendDecode)
        {
            sw.WriteLine("alpha_raw,alpha_raw_hex,zone_num,zone_name_alpha,sub_num,sub_name_alpha,parent_ok,alpha_continent");
            InitializedCsv.Add(file);
        }

        var alphaNames = ctx.AlphaAreaNamesByNumber!;
        var alphaParent = ctx.AlphaParentByNumber!;
        var alphaCont = ctx.AlphaContinentByNumber!;

        // Use only unique alpha_raw values per map
        for (int i = 0; i < 256; i++)
        {
            int raw = alphaAreaIds[i];
            if (raw <= 0) continue;
            string key = ctx.MapName + ":" + raw.ToString();
            if (AlphaDecodeKeys.Contains(key)) continue;

            var (z, s) = DecodeZoneSubzone(raw);
            int zoneBase_hi = (z << 16);
            int zoneBase_lo = (s << 16);

            // Strict hi16->zone, lo16->sub (no swap heuristics)
            int zoneNumOut = z;
            string zoneNameOut = string.Empty;
            int subNumOut = s;
            string subNameOut = string.Empty;
            if (TryGetAlphaNameByCont(ctx, zoneBase_hi, ctx.CurrentMapId, false, out var znm)) zoneNameOut = znm;
            if (s != 0 && TryGetAlphaNameByCont(ctx, raw, ctx.CurrentMapId, false, out var snm)) subNameOut = snm;
            bool parentOkOut = s == 0 ? true : (TryGetAlphaParentByCont(ctx, raw, ctx.CurrentMapId, true, out var p) && p == zoneBase_hi);

            int cont = GetAlphaContForArea(ctx, zoneBase_hi, ctx.CurrentMapId);
            sw.WriteLine(string.Join(",",
                raw.ToString(),
                $"0x{raw:X}",
                zoneNumOut.ToString(),
                EscapeCsv(zoneNameOut),
                subNumOut.ToString(),
                EscapeCsv(subNameOut),
                parentOkOut ? "1" : "0",
                cont.ToString()
            ));

            AlphaDecodeKeys.Add(key);
        }
    }

    private static void WriteAlphaTo335SuggestionsCsv(WriteContext ctx, IReadOnlyList<int> alphaAreaIds)
    {
        var dir = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName);
        Directory.CreateDirectory(dir);
        var file = Path.Combine(dir, "alpha_to_335_suggestions.csv");

        bool appendSugg = InitializedCsv.Contains(file);
        using var sw = new StreamWriter(file, append: appendSugg);
        if (!appendSugg)
        {
            sw.WriteLine("alpha_raw,alpha_raw_hex,zone_num,zone_name_alpha,sub_num,sub_name_alpha,alpha_continent,lk_zone_id_suggested,lk_zone_name,lk_sub_id_suggested,lk_sub_name,method");
            InitializedCsv.Add(file);
        }

        var alphaNames = ctx.AlphaAreaNamesByNumber!;
        var alphaParent = ctx.AlphaParentByNumber!;
        var alphaCont = ctx.AlphaContinentByNumber!;
        var lkNames = ctx.LkAreaNamesById!;
        var lkParent = ctx.LkParentById!;
        var lkCont = ctx.LkContinentById!;

        // Build LK indices by normalized name, per-continent and global
        var idxByContName = new Dictionary<int, Dictionary<string, List<int>>>();
        var idxByName = new Dictionary<string, List<int>>();
        foreach (var kv in lkNames)
        {
            int id = kv.Key;
            string nm = kv.Value ?? string.Empty;
            string n = Norm(nm);
            int cont = lkCont.TryGetValue(id, out var m) ? m : -1;
            if (!idxByContName.TryGetValue(cont, out var dict)) { dict = new Dictionary<string, List<int>>(); idxByContName[cont] = dict; }
            if (!dict.TryGetValue(n, out var lst)) { lst = new List<int>(); dict[n] = lst; }
            lst.Add(id);
            if (!idxByName.TryGetValue(n, out var lg)) { lg = new List<int>(); idxByName[n] = lg; }
            lg.Add(id);
        }

        // Unique per map to avoid duplicates
        var seen = new HashSet<int>();
        for (int i = 0; i < 256; i++)
        {
            int raw = alphaAreaIds[i];
            if (raw <= 0 || !seen.Add(raw)) continue;

            var (z, s) = DecodeZoneSubzone(raw);
            int zoneBase = (z << 16);

            // Strict hi16->zone, lo16->sub for suggestions as well (no swap)
            string zoneNameAlpha = string.Empty;
            string subNameAlpha = string.Empty;
            TryGetAlphaNameByCont(ctx, zoneBase, ctx.CurrentMapId, false, out zoneNameAlpha);
            if (s != 0) TryGetAlphaNameByCont(ctx, raw, ctx.CurrentMapId, false, out subNameAlpha);

            string nZone = Norm(zoneNameAlpha);
            int lkZone = -1;
            string method = "global";
            if (idxByContName.TryGetValue(GetAlphaContForArea(ctx, zoneBase, ctx.CurrentMapId), out var dict) && dict.TryGetValue(nZone, out var lm) && lm.Count > 0)
            {
                lkZone = lm.Min();
                method = "map_biased";
            }
            else if (idxByName.TryGetValue(nZone, out var lg) && lg.Count > 0)
            {
                lkZone = lg.Min();
            }

            int lkSub = -1;
            if (s != 0 && lkZone > 0)
            {
                string nSub = Norm(subNameAlpha);
                // Filter LK by sub name and parent
                if (idxByName.TryGetValue(nSub, out var lsub) && lsub.Count > 0)
                {
                    lkSub = lsub.Where(id => lkParent.TryGetValue(id, out var pr) && pr == lkZone).DefaultIfEmpty(-1).Min();
                }
                if (lkSub <= 0)
                {
                    method = method + ":fallback_to_zone";
                }
            }

            sw.WriteLine(string.Join(",",
                raw.ToString(),
                $"0x{raw:X}",
                z.ToString(),
                EscapeCsv(zoneNameAlpha),
                s.ToString(),
                EscapeCsv(subNameAlpha),
                GetAlphaContForArea(ctx, zoneBase, ctx.CurrentMapId).ToString(),
                lkZone > 0 ? lkZone.ToString() : string.Empty,
                lkZone > 0 && lkNames.TryGetValue(lkZone, out var lzn) ? EscapeCsv(lzn) : string.Empty,
                lkSub > 0 ? lkSub.ToString() : string.Empty,
                lkSub > 0 && lkNames.TryGetValue(lkSub, out var lsn) ? EscapeCsv(lsn) : string.Empty,
                method
            ));
        }
    }

    private static void WriteRemapDumpCsv(WriteContext ctx, IReadOnlyList<int> alphaAreaIds, AreaIdMapper mapper)
    {
        var dir = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName);
        Directory.CreateDirectory(dir);
        var file = Path.Combine(dir, "remap_dump.csv");

        bool exists = File.Exists(file);
        using var sw = new StreamWriter(file, append: true);
        if (!exists)
        {
            sw.WriteLine("src_alias,src_build,tgt_build,map_name,map_id,idx,alpha_raw,alpha_raw_hex,zone,subzone,alpha_name,lk_areaid,lk_name,reason");
        }

        string srcAlias = ctx.SrcAlias ?? string.Empty;
        string srcBuild = ctx.SrcBuild ?? string.Empty;
        string tgtBuild = ctx.TgtBuild ?? string.Empty;

        for (int i = 0; i < 256; i++)
        {
            int aId = alphaAreaIds[i];
            if (aId < 0) continue;
            var (zone, subzone) = DecodeZoneSubzone(aId);

            int lkId;
            string? alphaName;
            string? lkName;
            string reason;
            mapper.TryMapDetailed(aId, ctx.CurrentMapId, out lkId, out alphaName, out lkName, out reason);

            sw.WriteLine(string.Join(",",
                EscapeCsv(srcAlias),
                EscapeCsv(srcBuild),
                EscapeCsv(tgtBuild),
                EscapeCsv(ctx.MapName),
                ctx.CurrentMapId?.ToString() ?? string.Empty,
                i.ToString(),
                aId.ToString(),
                $"0x{aId:X}",
                zone.ToString(),
                subzone.ToString(),
                EscapeCsv(alphaName),
                lkId > 0 ? lkId.ToString() : string.Empty,
                EscapeCsv(lkName),
                EscapeCsv(reason)
            ));
        }
    }

    private static void WriteRemapExplicitUsedCsv(WriteContext ctx, IReadOnlyList<int> alphaAreaIds, AreaIdMapper mapper)
    {
        var dir = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName);
        Directory.CreateDirectory(dir);
        var file = Path.Combine(dir, "remap_explicit_used.csv");

        bool exists = File.Exists(file);
        using var sw = new StreamWriter(file, append: true);
        if (!exists)
        {
            sw.WriteLine("src_alias,src_build,tgt_build,map_name,map_id,alpha_areaNumber,alpha_areaNumber_hex,zone,subzone,tgt_areaID,lk_name");
        }

        string srcAlias = ctx.SrcAlias ?? string.Empty;
        string srcBuild = ctx.SrcBuild ?? string.Empty;
        string tgtBuild = ctx.TgtBuild ?? string.Empty;

        for (int i = 0; i < 256; i++)
        {
            int aId = alphaAreaIds[i];
            if (aId < 0) continue;
            var (zone, subzone) = DecodeZoneSubzone(aId);

            if (mapper.TryMapDetailed(aId, ctx.CurrentMapId, out var lkId, out _, out var lkName, out var reason))
            {
                if (!string.Equals(reason, "remap_explicit", StringComparison.OrdinalIgnoreCase)) continue;
                sw.WriteLine(string.Join(",",
                    EscapeCsv(srcAlias),
                    EscapeCsv(srcBuild),
                    EscapeCsv(tgtBuild),
                    EscapeCsv(ctx.MapName),
                    ctx.CurrentMapId?.ToString() ?? string.Empty,
                    aId.ToString(),
                    $"0x{aId:X}",
                    zone.ToString(),
                    subzone.ToString(),
                    lkId.ToString(),
                    EscapeCsv(lkName)
                ));
            }
        }
    }

    private static void WriteAreaIdSampleCsv(WriteContext ctx, string outFile, IReadOnlyList<int> alphaAreaIds, AreaIdMapper? mapper)
    {
        // Per-tile sample (first N present rows) including readbacks from written LK ADT
        int sampleMax = ctx.SampleCount > 0 ? ctx.SampleCount : 16;
        var dir = Path.Combine(ctx.ExportDir, "csv", "maps", ctx.MapName);
        Directory.CreateDirectory(dir);
        var file = Path.Combine(dir, $"{ctx.MapName}_{ctx.TileX}_{ctx.TileY}_areaid_sample.csv");

        // Diagnostics-only or missing ADT file: write sample rows without readbacks
        if (ctx.DiagnosticsOnly || !File.Exists(outFile))
        {
            using var swLite = new StreamWriter(file, append: false);
            swLite.WriteLine("idx,alpha_raw_hex,zone,subzone,zone_name_alpha,sub_name_alpha,alpha_continent,reason,lk_write,lk_zone_id_suggested,lk_zone_name,lk_sub_id_suggested,lk_sub_name,lk_readback_areaid,lk_readback_holes_hex");

            int writtenLite = 0;
            for (int i = 0; i < 256 && writtenLite < sampleMax; i++)
            {
                int aId = alphaAreaIds[i];
                if (aId < 0) continue;
                var (zone, subzone) = DecodeZoneSubzone(aId);

                int lkId = -1;
                string reason = string.Empty;
                if (mapper is not null)
                {
                    mapper.TryMapDetailed(aId, ctx.CurrentMapId, out lkId, out _, out _, out reason);
                }

                string lkWrite = string.Equals(reason, "remap_explicit", StringComparison.OrdinalIgnoreCase) ? lkId.ToString() : string.Empty;

                // Enrich with Alpha names and LK suggestions if possible
                string zoneNameAlpha = string.Empty;
                string subNameAlpha = string.Empty;
                int aCont = GetAlphaContForArea(ctx, (zone << 16), ctx.CurrentMapId);
                int lkZoneSugg = -1;
                string lkZoneName = string.Empty;
                int lkSubSugg = -1;
                string lkSubName = string.Empty;
                if (ctx.AlphaAreaNamesByNumber is not null && ctx.AlphaParentByNumber is not null &&
                    ctx.LkAreaNamesById is not null && ctx.LkParentById is not null && ctx.LkContinentById is not null)
                {
                    var (idxByContName, idxByName) = EnsureLkNameIndexes(ctx);
                    int zoneBase = (zone << 16);
                    TryGetAlphaNameByCont(ctx, zoneBase, ctx.CurrentMapId, true, out zoneNameAlpha);
                    if (subzone != 0) TryGetAlphaNameByCont(ctx, aId, ctx.CurrentMapId, true, out subNameAlpha);

                    string nZone = Norm(zoneNameAlpha);
                    if (idxByContName.TryGetValue(aCont, out var dict) && dict.TryGetValue(nZone, out var lm) && lm.Count > 0)
                    {
                        lkZoneSugg = lm.Min();
                    }
                    else if (idxByName.TryGetValue(nZone, out var lg) && lg.Count > 0)
                    {
                        lkZoneSugg = lg.Min();
                    }
                    if (lkZoneSugg > 0)
                    {
                        if (ctx.LkAreaNamesById.TryGetValue(lkZoneSugg, out var lzn)) lkZoneName = lzn;
                        if (subzone != 0)
                        {
                            string nSub = Norm(subNameAlpha);
                            if (idxByName.TryGetValue(nSub, out var lsub) && lsub.Count > 0)
                            {
                                lkSubSugg = lsub.Where(id => ctx.LkParentById!.TryGetValue(id, out var pr) && pr == lkZoneSugg).DefaultIfEmpty(-1).Min();
                                if (lkSubSugg > 0 && ctx.LkAreaNamesById.TryGetValue(lkSubSugg, out var lsn)) lkSubName = lsn;
                            }
                        }
                    }
                }

                swLite.WriteLine(string.Join(",",
                    i.ToString(),
                    $"0x{aId:X}",
                    zone.ToString(),
                    subzone.ToString(),
                    EscapeCsv(zoneNameAlpha),
                    EscapeCsv(subNameAlpha),
                    aCont.ToString(),
                    EscapeCsv(reason),
                    lkWrite,
                    lkZoneSugg > 0 ? lkZoneSugg.ToString() : string.Empty,
                    EscapeCsv(lkZoneName),
                    lkSubSugg > 0 ? lkSubSugg.ToString() : string.Empty,
                    EscapeCsv(lkSubName),
                    string.Empty,
                    string.Empty
                ));

                writtenLite++;
            }
            return;
        }

        using var fs = new FileStream(outFile, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs);

        // Locate MCIN
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mcinDataPos = -1;
        int mcinSize = 0;
        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == "MCIN") { mcinDataPos = dpos; mcinSize = sz; break; }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }
        if (mcinDataPos < 0 || mcinSize < 16) return;

        using var sw = new StreamWriter(file, append: false);
        sw.WriteLine("idx,alpha_raw_hex,zone,subzone,zone_name_alpha,sub_name_alpha,alpha_continent,reason,lk_write,lk_zone_id_suggested,lk_zone_name,lk_sub_id_suggested,lk_sub_name,lk_readback_areaid,lk_readback_holes_hex");

        int written = 0;
        for (int i = 0; i < 256 && written < sampleMax; i++)
        {
            int aId = alphaAreaIds[i];
            if (aId < 0) continue;
            var (zone, subzone) = DecodeZoneSubzone(aId);

            int lkId = -1;
            string reason = string.Empty;
            if (mapper is not null)
            {
                mapper.TryMapDetailed(aId, ctx.CurrentMapId, out lkId, out _, out _, out reason);
            }

            // MCNK offset
            fs.Position = mcinDataPos + (i * 16);
            int mcnkOffset = br.ReadInt32();
            if (mcnkOffset <= 0) continue;

            // Read back LK AreaID and Holes from on-disk file
            long areaFieldPos = (long)mcnkOffset + 8 + 0x34; // LK MCNK header AreaId (0x34); 0x3C is Holes
            long holesPos = (long)mcnkOffset + 8 + 0x3C;
            if (areaFieldPos + 4 > fileLen || holesPos + 4 > fileLen) continue;

            fs.Position = areaFieldPos;
            uint onDiskArea = br.ReadUInt32();

            fs.Position = holesPos;
            ushort holesLow = br.ReadUInt16();
            ushort holesUnknown = br.ReadUInt16();
            string holesHex = $"0x{holesLow:X4}{holesUnknown:X4}";

            string lkWrite = string.Equals(reason, "remap_explicit", StringComparison.OrdinalIgnoreCase) ? lkId.ToString() : string.Empty;

            // Enrich with Alpha names and LK suggestions if possible
            string zoneNameAlpha = string.Empty;
            string subNameAlpha = string.Empty;
            int aCont = GetAlphaContForArea(ctx, (zone << 16), ctx.CurrentMapId);
            int lkZoneSugg = -1;
            string lkZoneName = string.Empty;
            int lkSubSugg = -1;
            string lkSubName = string.Empty;
            if (ctx.AlphaAreaNamesByNumber is not null && ctx.AlphaParentByNumber is not null &&
                ctx.LkAreaNamesById is not null && ctx.LkParentById is not null && ctx.LkContinentById is not null)
            {
                var (idxByContName, idxByName) = EnsureLkNameIndexes(ctx);
                int zoneBase = (zone << 16);
                TryGetAlphaNameByCont(ctx, zoneBase, ctx.CurrentMapId, true, out zoneNameAlpha);
                if (subzone != 0) TryGetAlphaNameByCont(ctx, aId, ctx.CurrentMapId, true, out subNameAlpha);

                string nZone = Norm(zoneNameAlpha);
                if (idxByContName.TryGetValue(aCont, out var dict) && dict.TryGetValue(nZone, out var lm) && lm.Count > 0)
                {
                    lkZoneSugg = lm.Min();
                }
                else if (idxByName.TryGetValue(nZone, out var lg) && lg.Count > 0)
                {
                    lkZoneSugg = lg.Min();
                }
                if (lkZoneSugg > 0)
                {
                    if (ctx.LkAreaNamesById.TryGetValue(lkZoneSugg, out var lzn)) lkZoneName = lzn;
                    if (subzone != 0)
                    {
                        string nSub = Norm(subNameAlpha);
                        if (idxByName.TryGetValue(nSub, out var lsub) && lsub.Count > 0)
                        {
                            lkSubSugg = lsub.Where(id => ctx.LkParentById!.TryGetValue(id, out var pr) && pr == lkZoneSugg).DefaultIfEmpty(-1).Min();
                            if (lkSubSugg > 0 && ctx.LkAreaNamesById.TryGetValue(lkSubSugg, out var lsn)) lkSubName = lsn;
                        }
                    }
                }
            }

            sw.WriteLine(string.Join(",",
                i.ToString(),
                $"0x{aId:X}",
                zone.ToString(),
                subzone.ToString(),
                EscapeCsv(zoneNameAlpha),
                EscapeCsv(subNameAlpha),
                aCont.ToString(),
                EscapeCsv(reason),
                lkWrite,
                lkZoneSugg > 0 ? lkZoneSugg.ToString() : string.Empty,
                EscapeCsv(lkZoneName),
                lkSubSugg > 0 ? lkSubSugg.ToString() : string.Empty,
                EscapeCsv(lkSubName),
                onDiskArea.ToString(),
                holesHex
            ));

            written++;
        }
    }

    private static (int present, int patched, int ignored) PatchMcnkAreaIdsOnDisk(string filePath, IReadOnlyList<int> alphaAreaIds, AreaIdMapper mapper, bool verbose, int? currentMapId)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        // Locate MCIN to get MCNK offsets
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long mcinDataPos = -1;
        int mcinSize = 0;

        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == "MCIN") { mcinDataPos = dpos; mcinSize = sz; break; }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }

        int present = 0, patched = 0, ignored = 0;
        int debugPrinted = 0;
        if (mcinDataPos >= 0 && mcinSize >= 16)
        {
            for (int i2 = 0; i2 < 256; i2++)
            {
                int aId = alphaAreaIds[i2];
                if (aId < 0) continue; // no MCNK present
                present++;

                fs.Position = mcinDataPos + (i2 * 16);
                int mcnkOffset = br.ReadInt32();
                if (mcnkOffset <= 0) continue;

                // Map strictly by explicit remap, with map-awareness
                int lkAreaId;
                string reason;
                if (mapper.TryMapDetailed(aId, currentMapId, out lkAreaId, out _, out _, out reason))
                {
                    if (string.Equals(reason, "ignored", StringComparison.OrdinalIgnoreCase))
                    { ignored++; if (verbose && debugPrinted < 8) { Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) reason=ignored"); debugPrinted++; } continue; }
                    if (!string.Equals(reason, "remap_explicit", StringComparison.OrdinalIgnoreCase))
                    { if (verbose && debugPrinted < 8) { Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) reason={reason} (no write)"); debugPrinted++; } continue; }
                }
                else
                { if (verbose && debugPrinted < 8) { Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) reason=unmapped (no write)"); debugPrinted++; } continue; }

                long areaFieldPos = (long)mcnkOffset + 8 + 0x34; // LK MCNK header AreaId (0x34); 0x3C is Holes
                if (areaFieldPos + 4 > fileLen) continue;

                long save = fs.Position;
                fs.Position = areaFieldPos;
                bw.Write((uint)lkAreaId); // write full 32-bit LE
                fs.Position = areaFieldPos;
                uint onDisk = br.ReadUInt32();
                fs.Position = save;
                patched++;
                if (verbose && debugPrinted < 8)
                {
                    Console.WriteLine($"  idx={i2:D3} alpha={aId} (0x{aId:X}) -> write={lkAreaId} (0x{lkAreaId:X}) onDisk={onDisk} (0x{onDisk:X})");
                    debugPrinted++;
                }
            }
        }
        return (present, patched, ignored);
    }

    private static string ReverseFourCC(string s)
    {
        if (string.IsNullOrEmpty(s) || s.Length != 4) return s ?? string.Empty;
        return new string(new[] { s[3], s[2], s[1], s[0] });
    }

    private static string EscapeCsv(string? s)
    {
        if (string.IsNullOrEmpty(s)) return string.Empty;
        if (s.Contains(',') || s.Contains('"'))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static void PatchStringTableInPlace(string filePath, string chunkFourCC, AssetType type, AssetFixupPolicy fixup, Func<string, string> resolve)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.Read);
        using var br = new BinaryReader(fs);
        using var bw = new BinaryWriter(fs);

        // Locate the string table chunk by scanning top-level chunks
        fs.Seek(0, SeekOrigin.Begin);
        long fileLen = fs.Length;
        long dataPos = -1;
        int size = 0;
        while (fs.Position + 8 <= fileLen)
        {
            var fourccRevBytes = br.ReadBytes(4);
            if (fourccRevBytes.Length < 4) break;
            var fourcc = ReverseFourCC(Encoding.ASCII.GetString(fourccRevBytes));
            int sz = br.ReadInt32();
            long dpos = fs.Position;
            if (fourcc == chunkFourCC)
            {
                dataPos = dpos;
                size = sz;
                break;
            }
            fs.Position = dpos + sz + ((sz & 1) == 1 ? 1 : 0);
        }
        if (dataPos < 0 || size <= 0) return;

        // Read chunk payload
        fs.Position = dataPos;
        var data = br.ReadBytes(size);

        // Iterate null-terminated strings and patch when replacement fits
        int i = 0;
        while (i < data.Length)
        {
            int start = i;
            while (i < data.Length && data[i] != 0) i++;
            int end = i;
            int capacity = end - start;
            if (capacity > 0)
            {
                var original = Encoding.ASCII.GetString(data, start, capacity);
                var norm = ListfileLoader.NormalizePath(original);
                var resolved = resolve(norm);
                var bytes = Encoding.ASCII.GetBytes(resolved);
                if (bytes.Length <= capacity && !norm.Equals(resolved, StringComparison.OrdinalIgnoreCase))
                {
                    Array.Copy(bytes, 0, data, start, bytes.Length);
                    for (int k = start + bytes.Length; k < end; k++) data[k] = 0;
                }
                else if (bytes.Length > capacity)
                {
                    fixup.LogDiagnostic(type, norm, resolved, "overflow_skip:" + chunkFourCC.ToLowerInvariant());
                }
            }
            i = end + 1;
        }

        fs.Position = dataPos;
        bw.Write(data);
    }

    // Cache LK name indices per map for performance
    private static readonly Dictionary<string, (Dictionary<int, Dictionary<string, List<int>>> ByContName, Dictionary<string, List<int>> ByName)> s_LkIdxCache
        = new(StringComparer.OrdinalIgnoreCase);

    private static (Dictionary<int, Dictionary<string, List<int>>> ByContName, Dictionary<string, List<int>> ByName) EnsureLkNameIndexes(WriteContext ctx)
    {
        if (s_LkIdxCache.TryGetValue(ctx.MapName, out var cached)) return cached;
        var byCont = new Dictionary<int, Dictionary<string, List<int>>>();
        var byName = new Dictionary<string, List<int>>();
        foreach (var kv in ctx.LkAreaNamesById!)
        {
            int id = kv.Key; string nm = kv.Value ?? string.Empty; string n = Norm(nm);
            int cont = ctx.LkContinentById!.TryGetValue(id, out var m) ? m : -1;
            if (!byCont.TryGetValue(cont, out var dict)) { dict = new Dictionary<string, List<int>>(); byCont[cont] = dict; }
            if (!dict.TryGetValue(n, out var lst)) { lst = new List<int>(); dict[n] = lst; }
            lst.Add(id);
            if (!byName.TryGetValue(n, out var lg)) { lg = new List<int>(); byName[n] = lg; }
            lg.Add(id);
        }
        s_LkIdxCache[ctx.MapName] = (byCont, byName);
        return (byCont, byName);
    }

    private static void PatchMtexOnDiskInternal(string filePath, AssetFixupPolicy fixup)
    {
        // implementation of PatchMtexOnDiskInternal
    }
}
