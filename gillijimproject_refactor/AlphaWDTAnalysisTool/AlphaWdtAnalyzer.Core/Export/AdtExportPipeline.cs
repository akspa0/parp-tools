using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using GillijimProject.WowFiles.Alpha;
using AlphaWdtAnalyzer.Core.Dbc;
using AlphaWdtAnalyzer.Core.Assets;
using DBCD;
using DBCD.Providers;

namespace AlphaWdtAnalyzer.Core.Export;

public static class AdtExportPipeline
{
    public sealed class Options
    {
        public string? SingleWdtPath { get; init; }
        public string? InputRoot { get; init; }
        public string? CommunityListfilePath { get; init; }
        public string? LkListfilePath { get; init; }
        public required string ExportDir { get; init; }
        public required string FallbackTileset { get; init; }
        public required string FallbackNonTilesetBlp { get; init; }
        public required string FallbackWmo { get; init; }
        public required string FallbackM2 { get; init; }
        public bool ConvertToMh2o { get; init; } = true;
        public bool AssetFuzzy { get; init; } = true;
        public bool UseFallbacks { get; init; } = true;
        public bool EnableFixups { get; init; } = true;
        public string?[]? AssetRoots { get; init; }
        public bool LogExact { get; init; } = false;
        public string? RemapPath { get; init; }
        public bool Verbose { get; init; } = false;
        public bool TrackAssets { get; init; } = false;
        // DBCD mapping fallback (when RemapPath is not provided)
        public string? DbdDir { get; init; }
        public string? DbcSrcDir { get; init; }
        public string? DbcTgtDir { get; init; }
        public string? SrcAlias { get; init; }
        public string? SrcBuild { get; init; }
        public string? TgtBuild { get; init; }
        public bool AllowDoNotUse { get; init; } = false;
        public int SampleCount { get; init; } = 8;
        public bool DiagnosticsOnly { get; init; } = false;
    }

    public static void ExportSingle(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.SingleWdtPath)) throw new ArgumentException("SingleWdtPath required", nameof(opts.SingleWdtPath));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);
        var mapName = Path.GetFileNameWithoutExtension(opts.SingleWdtPath!);
        var logDir = Path.Combine(opts.ExportDir, "csv", "maps", mapName);
        Directory.CreateDirectory(logDir);
        using var fixupLogger = new FixupLogger(Path.Combine(logDir, "asset_fixups.csv"));
        var roots1 = (opts.AssetRoots ?? Array.Empty<string>()).Where(s => !string.IsNullOrWhiteSpace(s)).Select(s => s!);
        var inventory = new AssetInventory(roots1);
        var fixup = new AssetFixupPolicy(
            resolver,
            opts.FallbackTileset,
            opts.FallbackNonTilesetBlp,
            opts.FallbackWmo,
            opts.FallbackM2,
            opts.AssetFuzzy,
            opts.UseFallbacks,
            opts.EnableFixups,
            fixupLogger,
            inventory,
            opts.LogExact);

        // Prefer explicit remap.json; otherwise attempt DBCD-based mapping if DBC flags are provided
        AreaIdMapper? areaMapper = null;
        if (!string.IsNullOrWhiteSpace(opts.RemapPath))
        {
            areaMapper = AreaIdMapper.TryCreate(null, null, null, opts.RemapPath);
        }
        else if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbcSrcDir) && !string.IsNullOrWhiteSpace(opts.DbcTgtDir))
        {
            areaMapper = BuildMapperFromDbc(opts);
        }

        // Build decode dictionaries (Alpha/LK AreaTable) when DBCD resources are available
        (Dictionary<int, string> AlphaNames,
         Dictionary<int, int> AlphaParent,
         Dictionary<int, int> AlphaCont,
         Dictionary<int, string> LkNames,
         Dictionary<int, int> LkParent,
         Dictionary<int, int> LkCont,
         Dictionary<int, Dictionary<int, string>> AlphaNamesByCont,
         Dictionary<int, Dictionary<int, int>> AlphaParentByCont)? decodeInfo = null;
        if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbcSrcDir) && !string.IsNullOrWhiteSpace(opts.DbcTgtDir))
        {
            decodeInfo = TryBuildAreaDecodeFromDbc(opts);
        }

        var wdt = new WdtAlphaScanner(opts.SingleWdtPath!);
        var adtScanner = new AdtScanner();
        var result = adtScanner.Scan(wdt);

        // Build union of tiles from placements and from WDT offsets (non-zero)
        var placementsByTile = result.Placements
            .GroupBy(p => (p.TileX, p.TileY))
            .ToDictionary(g => g.Key, g => (IEnumerable<PlacementRecord>)g);

        var candidateTiles = new HashSet<(int tx, int ty)>(placementsByTile.Keys);
        for (int adtNum = 0; adtNum < wdt.AdtMhdrOffsets.Count; adtNum++)
        {
            if (wdt.AdtMhdrOffsets[adtNum] > 0)
            {
                int x = adtNum % 64;
                int y = adtNum / 64;
                candidateTiles.Add((x, y));
            }
        }

        foreach (var (x, y) in candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty))
        {
            var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
            var g = hasGroup ? group! : Array.Empty<PlacementRecord>();

            IReadOnlyList<int>? alphaAreaIds = null;
            int adtNum = (y * 64) + x;
            int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (offset > 0)
            {
                var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
                alphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                int currentMapId = ResolveMapIdByName(wdt.MapName);

                var ctx = new AdtWotlkWriter.WriteContext
                {
                    ExportDir = opts.ExportDir,
                    MapName = wdt.MapName,
                    TileX = x,
                    TileY = y,
                    Placements = g,
                    Fixup = fixup,
                    ConvertToMh2o = opts.ConvertToMh2o,
                    AreaMapper = areaMapper,
                    AlphaAreaIds = alphaAreaIds,
                    WdtPath = wdt.WdtPath,
                    AdtNumber = adtNum,
                    AdtOffset = offset,
                    MdnmFiles = wdt.MdnmFiles,
                    MonmFiles = wdt.MonmFiles,
                    Verbose = opts.Verbose,
                    TrackAssets = opts.TrackAssets,
                    CurrentMapId = currentMapId,
                    SrcAlias = opts.SrcAlias,
                    SrcBuild = opts.SrcBuild,
                    TgtBuild = opts.TgtBuild,
                    AllowAreaIdWrites = !string.IsNullOrWhiteSpace(opts.RemapPath),
                    AlphaAreaNamesByNumber = decodeInfo?.AlphaNames,
                    AlphaParentByNumber = decodeInfo?.AlphaParent,
                    AlphaContinentByNumber = decodeInfo?.AlphaCont,
                    AlphaAreaNamesByNumberByCont = decodeInfo == null ? null : decodeInfo.Value.AlphaNamesByCont.ToDictionary(
                        kvp => kvp.Key,
                        kvp => (IReadOnlyDictionary<int, string>)new ReadOnlyDictionary<int, string>(kvp.Value)
                    ),
                    AlphaParentByNumberByCont = decodeInfo == null ? null : decodeInfo.Value.AlphaParentByCont.ToDictionary(
                        kvp => kvp.Key,
                        kvp => (IReadOnlyDictionary<int, int>)new ReadOnlyDictionary<int, int>(kvp.Value)
                    ),
                    LkAreaNamesById = decodeInfo?.LkNames,
                    LkParentById = decodeInfo?.LkParent,
                    LkContinentById = decodeInfo?.LkCont,
                    SampleCount = opts.SampleCount,
                    DiagnosticsOnly = opts.DiagnosticsOnly
                };
                AdtWotlkWriter.WriteBinary(ctx);
            }
        }
    }

    public static void ExportBatch(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.InputRoot)) throw new ArgumentException("InputRoot required", nameof(opts.InputRoot));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);

        var wdts = Directory.EnumerateFiles(opts.InputRoot!, "*.wdt", SearchOption.AllDirectories)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

        var roots2 = (opts.AssetRoots ?? Array.Empty<string>()).Where(s => !string.IsNullOrWhiteSpace(s)).Select(s => s!);
        var inventory = new AssetInventory(roots2);

        foreach (var wdtPath in wdts)
        {
            try
            {
                var wdt = new WdtAlphaScanner(wdtPath);

                var mapName = Path.GetFileNameWithoutExtension(wdtPath);
                var logDir = Path.Combine(opts.ExportDir, "csv", "maps", mapName);
                Directory.CreateDirectory(logDir);
                using var fixupLogger = new FixupLogger(Path.Combine(logDir, "asset_fixups.csv"));

                var fixup = new AssetFixupPolicy(
                    resolver,
                    opts.FallbackTileset,
                    opts.FallbackNonTilesetBlp,
                    opts.FallbackWmo,
                    opts.FallbackM2,
                    opts.AssetFuzzy,
                    opts.UseFallbacks,
                    opts.EnableFixups,
                    fixupLogger,
                    inventory,
                    opts.LogExact);

                // Prefer explicit remap.json; otherwise attempt DBCD-based mapping if DBC flags are provided
                AreaIdMapper? areaMapper = null;
                if (!string.IsNullOrWhiteSpace(opts.RemapPath))
                {
                    areaMapper = AreaIdMapper.TryCreate(null, null, null, opts.RemapPath);
                }
                else if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbcSrcDir) && !string.IsNullOrWhiteSpace(opts.DbcTgtDir))
                {
                    areaMapper = BuildMapperFromDbc(opts);
                }

                // Build decode dictionaries (Alpha/LK AreaTable) when DBCD resources are available
                (Dictionary<int, string> AlphaNames,
                 Dictionary<int, int> AlphaParent,
                 Dictionary<int, int> AlphaCont,
                 Dictionary<int, string> LkNames,
                 Dictionary<int, int> LkParent,
                 Dictionary<int, int> LkCont,
                 Dictionary<int, Dictionary<int, string>> AlphaNamesByCont,
                 Dictionary<int, Dictionary<int, int>> AlphaParentByCont)? decodeInfo = null;
                if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbcSrcDir) && !string.IsNullOrWhiteSpace(opts.DbcTgtDir))
                {
                    decodeInfo = TryBuildAreaDecodeFromDbc(opts);
                }

                var adtScanner = new AdtScanner();
                var result = adtScanner.Scan(wdt);

                // Union of placements and WDT-present tiles
                var placementsByTile = result.Placements
                    .GroupBy(p => (p.TileX, p.TileY))
                    .ToDictionary(g => g.Key, g => (IEnumerable<PlacementRecord>)g);

                var candidateTiles = new HashSet<(int tx, int ty)>(placementsByTile.Keys);
                for (int adtNum = 0; adtNum < wdt.AdtMhdrOffsets.Count; adtNum++)
                {
                    if (wdt.AdtMhdrOffsets[adtNum] > 0)
                    {
                        int x = adtNum % 64;
                        int y = adtNum / 64;
                        candidateTiles.Add((x, y));
                    }
                }

                foreach (var (x, y) in candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty))
                {
                    var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
                    var g = hasGroup ? group! : Array.Empty<PlacementRecord>();

                    IReadOnlyList<int>? alphaAreaIds = null;
                    int adtNum = (y * 64) + x;
                    int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
                    if (offset > 0)
                    {
                        var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
                        alphaAreaIds = alpha.GetAlphaMcnkAreaIds();

                        int currentMapId = ResolveMapIdByName(wdt.MapName);

                        var ctx = new AdtWotlkWriter.WriteContext
                        {
                            ExportDir = opts.ExportDir,
                            MapName = wdt.MapName,
                            TileX = x,
                            TileY = y,
                            Placements = g,
                            Fixup = fixup,
                            ConvertToMh2o = opts.ConvertToMh2o,
                            AreaMapper = areaMapper,
                            AlphaAreaIds = alphaAreaIds,
                            WdtPath = wdt.WdtPath,
                            AdtNumber = adtNum,
                            AdtOffset = offset,
                            MdnmFiles = wdt.MdnmFiles,
                            MonmFiles = wdt.MonmFiles,
                            Verbose = opts.Verbose,
                            TrackAssets = opts.TrackAssets,
                            CurrentMapId = currentMapId,
                            SrcAlias = opts.SrcAlias,
                            SrcBuild = opts.SrcBuild,
                            TgtBuild = opts.TgtBuild,
                            AllowAreaIdWrites = !string.IsNullOrWhiteSpace(opts.RemapPath),
                            AlphaAreaNamesByNumber = decodeInfo?.AlphaNames,
                            AlphaParentByNumber = decodeInfo?.AlphaParent,
                            AlphaContinentByNumber = decodeInfo?.AlphaCont,
                            AlphaAreaNamesByNumberByCont = decodeInfo == null ? null : decodeInfo.Value.AlphaNamesByCont.ToDictionary(
                                kvp => kvp.Key,
                                kvp => (IReadOnlyDictionary<int, string>)new ReadOnlyDictionary<int, string>(kvp.Value)
                            ),
                            AlphaParentByNumberByCont = decodeInfo == null ? null : decodeInfo.Value.AlphaParentByCont.ToDictionary(
                                kvp => kvp.Key,
                                kvp => (IReadOnlyDictionary<int, int>)new ReadOnlyDictionary<int, int>(kvp.Value)
                            ),
                            LkAreaNamesById = decodeInfo?.LkNames,
                            LkParentById = decodeInfo?.LkParent,
                            LkContinentById = decodeInfo?.LkCont,
                            SampleCount = opts.SampleCount,
                            DiagnosticsOnly = opts.DiagnosticsOnly
                        };
                        AdtWotlkWriter.WriteBinary(ctx);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Export failed for {wdtPath}: {ex.Message}");
            }
        }
    }

    private static int ResolveMapIdByName(string mapName)
    {
        // Minimal classic mapping; extend as needed
        if (mapName.Equals("Azeroth", StringComparison.OrdinalIgnoreCase)) return 0;
        if (mapName.Equals("Kalimdor", StringComparison.OrdinalIgnoreCase)) return 1;
        // fallback unknown
        return -1;
    }

    // Build an AreaIdMapper from raw DBCs using DBCD (map-aware explicit mappings only)
    private static AreaIdMapper? BuildMapperFromDbc(Options opts)
    {
        try
        {
            var srcAlias = string.IsNullOrWhiteSpace(opts.SrcAlias) ? InferAliasFromPath(opts.DbcSrcDir!) : opts.SrcAlias!;
            string canonicalSrc = !string.IsNullOrWhiteSpace(opts.SrcBuild) ? opts.SrcBuild! : CanonicalizeBuild(srcAlias);
            string canonicalTgt = !string.IsNullOrWhiteSpace(opts.TgtBuild) ? opts.TgtBuild! : CanonicalizeBuild("3.3.5");

            var srcProv = new FilesystemDBCProvider(opts.DbcSrcDir!, useCache: true);
            var tgtProv = new FilesystemDBCProvider(opts.DbcTgtDir!, useCache: true);
            var dbdProv = new FilesystemDBDProvider(opts.DbdDir!);

            var dbcdSrc = new DBCD.DBCD(srcProv, dbdProv);
            var dbcdTgt = new DBCD.DBCD(tgtProv, dbdProv);

            // Load storages (AreaTable, Map) with Locale fallback
            IDBCDStorage Load(DBCD.DBCD d, string table, string build)
            {
                try { return d.Load(table, build, DBCD.Locale.EnUS); } catch { return d.Load(table, build, DBCD.Locale.None); }
            }

            var sArea = Load(dbcdSrc, "AreaTable", canonicalSrc);
            var sMap  = Load(dbcdSrc, "Map",       canonicalSrc);
            var tArea = Load(dbcdTgt, "AreaTable", canonicalTgt);
            var tMap  = Load(dbcdTgt, "Map",       canonicalTgt);

            // Helper: detect ID column (ID/MapID variants)
            #pragma warning disable CS8321 // local function declared but never used
            string DetectIdColumn(IDBCDStorage storage)
            {
                try
                {
                    var cols = storage.AvailableColumns ?? Array.Empty<string>();
                    string[] prefers = new[] { "ID", "Id", "MapID", "MapId", "m_ID" };
                    foreach (var p in prefers)
                    {
                        var match = cols.FirstOrDefault(x => string.Equals(x, p, StringComparison.OrdinalIgnoreCase));
                        if (!string.IsNullOrEmpty(match)) return match;
                    }
                    var anyId = cols.FirstOrDefault(x => x.EndsWith("ID", StringComparison.OrdinalIgnoreCase));
                    return anyId ?? string.Empty;
                }
                catch { return string.Empty; }
            }
            #pragma warning restore CS8321

            // Build target indices: by mapId + normalized name, and global by name
            string tAreaNameCol = DetectColumn(tArea, "AreaName_lang", "AreaName", "Name");
            var idxByMapName = new Dictionary<int, Dictionary<string, List<int>>>();
            var idxByName = new Dictionary<string, List<int>>();
            foreach (var tid in tArea.Keys)
            {
                var row = tArea[tid];
                string nm = SafeField<string>(row, tAreaNameCol) ?? string.Empty;
                string n = Norm(nm);
                int mapId = SafeField<int>(row, "ContinentID");
                if (!idxByMapName.TryGetValue(mapId, out var dict)) { dict = new Dictionary<string, List<int>>(); idxByMapName[mapId] = dict; }
                if (!dict.TryGetValue(n, out var lst)) { lst = new List<int>(); dict[n] = lst; }
                lst.Add(tid);
                if (!idxByName.TryGetValue(n, out var lg)) { lg = new List<int>(); idxByName[n] = lg; }
                lg.Add(tid);
            }

            // Source name column
            string sAreaNameCol = DetectColumn(sArea, "AreaName_lang", "AreaName", "Name");
            bool isClassic = srcAlias == "0.5.3" || srcAlias == "0.5.5" || srcAlias == "0.6.0";

            var explicitByMap = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var sid in sArea.Keys)
            {
                var row = sArea[sid];
                int srcAreaNumber = SafeField<int>(row, isClassic ? "AreaNumber" : "ID");
                if (srcAreaNumber <= 0) continue;
                string nm = SafeField<string>(row, sAreaNameCol) ?? string.Empty;
                string n = Norm(nm);
                int srcMapId = SafeField<int>(row, "ContinentID"); // 0=Azeroth,1=Kalimdor on 0.5.x

                int chosen = -1;
                // Prefer exact name match within same mapId in target
                if (idxByMapName.TryGetValue(srcMapId, out var dict) && dict.TryGetValue(n, out var lm) && lm.Count > 0)
                {
                    chosen = lm.Min();
                }
                else if (idxByName.TryGetValue(n, out var lg) && lg.Count > 0)
                {
                    chosen = lg.Min();
                }
                if (chosen > 0)
                {
                    var key = srcMapId.ToString() + ":" + srcAreaNumber.ToString();
                    if (!explicitByMap.ContainsKey(key)) explicitByMap[key] = chosen;
                }
            }

            if (explicitByMap.Count == 0) return null;
            // Create mapper with map-aware explicit only
            return AreaIdMapperFactory.CreateFromExplicit(explicitByMap, disallowDoNotUse: !opts.AllowDoNotUse);

            // Local helpers
            static string DetectColumn(IDBCDStorage storage, params string[] preferred)
            {
                var cols = storage.AvailableColumns ?? Array.Empty<string>();
                foreach (var c in preferred)
                    if (cols.Any(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase))) return cols.First(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase));
                var any = cols.FirstOrDefault(x => x.IndexOf("name", StringComparison.OrdinalIgnoreCase) >= 0);
                return any ?? (preferred.Length > 0 ? preferred[0] : string.Empty);
            }
            static T SafeField<T>(DBCDRow row, string col) { try { return row.Field<T>(col); } catch { return default!; } }
            static string Norm(string s) => (s ?? string.Empty).Trim().ToLowerInvariant();
            static string CanonicalizeBuild(string alias) => alias switch { "0.5.3" => "0.5.3.3368", "0.5.5" => "0.5.5.3494", "0.6.0" => "0.6.0.3592", "3.3.5" => "3.3.5.12340", _ => alias };
            static string InferAliasFromPath(string p)
            {
                var s = (p ?? string.Empty).Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar).ToLowerInvariant();
                if (s.Contains("0.5.3")) return "0.5.3";
                if (s.Contains("0.5.5")) return "0.5.5";
                if (s.Contains("0.6.0")) return "0.6.0";
                if (s.Contains("3.3.5")) return "3.3.5";
                return "0.5.3";
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[DBCD] Failed to build on-the-fly mapping: {ex.Message}");
            return null;
        }
    }

    // Build dictionaries needed to validate alpha decode and suggest LK targets
    private static (Dictionary<int, string> AlphaNames,
                    Dictionary<int, int> AlphaParent,
                    Dictionary<int, int> AlphaCont,
                    Dictionary<int, string> LkNames,
                    Dictionary<int, int> LkParent,
                    Dictionary<int, int> LkCont,
                    Dictionary<int, Dictionary<int, string>> AlphaNamesByCont,
                    Dictionary<int, Dictionary<int, int>> AlphaParentByCont)? TryBuildAreaDecodeFromDbc(Options opts)
    {
        try
        {
            var srcAlias = string.IsNullOrWhiteSpace(opts.SrcAlias) ? InferAliasFromPath(opts.DbcSrcDir!) : opts.SrcAlias!;
            string canonicalSrc = !string.IsNullOrWhiteSpace(opts.SrcBuild) ? opts.SrcBuild! : CanonicalizeBuild(srcAlias);
            string canonicalTgt = !string.IsNullOrWhiteSpace(opts.TgtBuild) ? opts.TgtBuild! : CanonicalizeBuild("3.3.5");

            var srcProv = new FilesystemDBCProvider(opts.DbcSrcDir!, useCache: true);
            var tgtProv = new FilesystemDBCProvider(opts.DbcTgtDir!, useCache: true);
            var dbdProv = new FilesystemDBDProvider(opts.DbdDir!);

            var dbcdSrc = new DBCD.DBCD(srcProv, dbdProv);
            var dbcdTgt = new DBCD.DBCD(tgtProv, dbdProv);

            IDBCDStorage Load(DBCD.DBCD d, string table, string build)
            {
                try { return d.Load(table, build, DBCD.Locale.EnUS); } catch { return d.Load(table, build, DBCD.Locale.None); }
            }

            var sArea = Load(dbcdSrc, "AreaTable", canonicalSrc);
            var tArea = Load(dbcdTgt, "AreaTable", canonicalTgt);

            string DetectColumn(IDBCDStorage storage, params string[] preferred)
            {
                var cols = storage.AvailableColumns ?? Array.Empty<string>();
                foreach (var c in preferred)
                    if (cols.Any(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase))) return cols.First(x => string.Equals(x, c, StringComparison.OrdinalIgnoreCase));
                var any = cols.FirstOrDefault(x => x.IndexOf("name", StringComparison.OrdinalIgnoreCase) >= 0);
                return any ?? (preferred.Length > 0 ? preferred[0] : string.Empty);
            }
            static T SafeField<T>(DBCDRow row, string col) { try { return row.Field<T>(col); } catch { return default!; } }

            // Source (alpha) dictionaries keyed by AreaNumber
            string sAreaNameCol = DetectColumn(sArea, "AreaName_lang", "AreaName", "Name");
            var alphaNames = new Dictionary<int, string>();
            var alphaParent = new Dictionary<int, int>();
            var alphaCont = new Dictionary<int, int>();
            var alphaNamesByCont = new Dictionary<int, Dictionary<int, string>>();
            var alphaParentByCont = new Dictionary<int, Dictionary<int, int>>();
            foreach (var sid in sArea.Keys)
            {
                var row = sArea[sid];
                int areaNum = SafeField<int>(row, "AreaNumber");
                if (areaNum <= 0) continue;
                string nm = SafeField<string>(row, sAreaNameCol) ?? string.Empty;
                int parent = SafeField<int>(row, "ParentAreaNum");
                int cont = SafeField<int>(row, "ContinentID");
                if (!alphaNames.ContainsKey(areaNum)) alphaNames[areaNum] = nm;
                if (!alphaParent.ContainsKey(areaNum)) alphaParent[areaNum] = parent;
                if (!alphaCont.ContainsKey(areaNum)) alphaCont[areaNum] = cont;
                if (!alphaNamesByCont.TryGetValue(areaNum, out var namesByMap)) { namesByMap = new Dictionary<int, string>(); alphaNamesByCont[areaNum] = namesByMap; }
                namesByMap[cont] = nm;
                if (!alphaParentByCont.TryGetValue(areaNum, out var parentByMap)) { parentByMap = new Dictionary<int, int>(); alphaParentByCont[areaNum] = parentByMap; }
                parentByMap[cont] = parent;
            }

            // Target (LK) dictionaries keyed by ID
            string tAreaNameCol = DetectColumn(tArea, "AreaName_lang", "AreaName", "Name");
            var lkNames = new Dictionary<int, string>();
            var lkParent = new Dictionary<int, int>();
            var lkCont = new Dictionary<int, int>();
            foreach (var tid in tArea.Keys)
            {
                var row = tArea[tid];
                string nm = SafeField<string>(row, tAreaNameCol) ?? string.Empty;
                int parent = SafeField<int>(row, "ParentAreaID");
                int cont = SafeField<int>(row, "ContinentID");
                lkNames[tid] = nm;
                lkParent[tid] = parent;
                lkCont[tid] = cont;
            }

            return (alphaNames, alphaParent, alphaCont, lkNames, lkParent, lkCont, alphaNamesByCont, alphaParentByCont);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[DBCD] Failed to build decode dictionaries: {ex.Message}");
            return null;
        }

        static string CanonicalizeBuild(string alias) => alias switch { "0.5.3" => "0.5.3.3368", "0.5.5" => "0.5.5.3494", "0.6.0" => "0.6.0.3592", "3.3.5" => "3.3.5.12340", _ => alias };
        static string InferAliasFromPath(string p)
        {
            var s = (p ?? string.Empty).Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar).ToLowerInvariant();
            if (s.Contains("0.5.3")) return "0.5.3";
            if (s.Contains("0.5.5")) return "0.5.5";
            if (s.Contains("0.6.0")) return "0.6.0";
            if (s.Contains("3.3.5")) return "3.3.5";
            return "0.5.3";
        }
    }
}
