using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using GillijimProject.WowFiles.Alpha;
using DBCTool.V2.Core;
using AlphaWdtAnalyzer.Core.Dbc;
using AlphaWdtAnalyzer.Core.Assets;
using System.Text;

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
        // Optional DBCTool.V2 integration (when provided, enables AreaIdMapperV2)
        public string? DbdDir { get; init; }
        public string? DbctoolOutRoot { get; init; } // preferred: root folder containing <alias>/compare/v2
        public string? DbctoolSrcAlias { get; init; } // e.g., 0.5.3 | 0.5.5 | 0.6.0
        public string? DbctoolSrcDir { get; init; }   // folder with source DBCs
        public string? DbctoolLkDir { get; init; }    // folder with 3.3.5 DBCs
        // Optional: Use precomputed DBCTool.V2 patch CSV(s)
        public string? DbctoolPatchDir { get; init; } // directory containing Area_patch_crosswalk_*.csv
        public string? DbctoolPatchFile { get; init; } // specific patch CSV file
        // Visualization
        public bool VizSvg { get; init; } = false;
        public string? VizDir { get; init; }
        public bool VizHtml { get; init; } = false;
        public bool PatchOnly { get; init; } = false;
        public int? MaxDegreeOfParallelism { get; init; }
        public bool NoZoneFallback { get; init; } = false;
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
        var inventory = new AssetInventory(opts.AssetRoots?.Where(r => !string.IsNullOrWhiteSpace(r)).Select(r => r!));
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

        var areaMapper = AreaIdMapper.TryCreate(null, null, null, opts.RemapPath);
        AreaIdMapperV2? areaMapperV2 = null;
        if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbctoolSrcDir) && !string.IsNullOrWhiteSpace(opts.DbctoolLkDir))
        {
            var alias = string.IsNullOrWhiteSpace(opts.DbctoolSrcAlias) ? "0.5.3" : opts.DbctoolSrcAlias!;
            areaMapperV2 = AreaIdMapperV2.TryCreate(opts.DbdDir!, alias, opts.DbctoolSrcDir!, opts.DbctoolLkDir!);
        }

        var wdt = new WdtAlphaScanner(opts.SingleWdtPath!);
        // Load DBCTool.V2 patch mapping (prefer stable out root -> <alias>/compare/v2)
        DbcPatchMapping? patchMap = null;
        string aliasUsed = ResolveSrcAlias(opts.DbctoolSrcAlias, opts.SingleWdtPath, null);
        string patchDirUsed = string.Empty;
        if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) || !string.IsNullOrWhiteSpace(opts.DbctoolPatchDir))
        {
            patchMap = new DbcPatchMapping();
            if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) && File.Exists(opts.DbctoolPatchFile!))
                patchMap.LoadFile(opts.DbctoolPatchFile!);
            if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchDir) && Directory.Exists(opts.DbctoolPatchDir!))
            {
                patchDirUsed = opts.DbctoolPatchDir!;
                foreach (var f in Directory.EnumerateFiles(opts.DbctoolPatchDir!, "Area_patch_crosswalk_*.csv", SearchOption.AllDirectories))
                    patchMap.LoadFile(f);
            }
        }
        else if (!string.IsNullOrWhiteSpace(opts.DbctoolOutRoot))
        {
            var v2Dir = Path.Combine(opts.DbctoolOutRoot!, aliasUsed, "compare", "v2");
            if (Directory.Exists(v2Dir))
            {
                patchDirUsed = v2Dir;
                patchMap = new DbcPatchMapping();
                foreach (var f in Directory.EnumerateFiles(v2Dir, "Area_patch_crosswalk_*.csv", SearchOption.AllDirectories))
                    patchMap.LoadFile(f);
            }
            else if (opts.PatchOnly)
            {
                Console.Error.WriteLine($"[PatchOnly] No CSVs found under --dbctool-out-root: {v2Dir}");
                return;
            }
        }
        else if (opts.PatchOnly)
        {
            Console.Error.WriteLine("[PatchOnly] Missing --dbctool-out-root or --dbctool-patch-dir/--dbctool-patch-file. Patch-only mode requires CSV crosswalks. Aborting.");
            return;
        }
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

        int processed = 0;
        int totalTiles = 0;
        void ProcessTile(int x, int y)
        {
            if (totalTiles == 0) totalTiles = candidateTiles.Count;
            var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
            var g = hasGroup ? group! : Array.Empty<PlacementRecord>();
            int adtNum = (y * 64) + x;
            int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (offset <= 0) return;
            var n = Interlocked.Increment(ref processed);
            if (!opts.Verbose)
                Console.WriteLine($"[Tile] {wdt.MapName} {x},{y} ({n}/{totalTiles})");
            var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
            var alphaAreaIds = (IReadOnlyList<int>)alpha.GetAlphaMcnkAreaIds();
            int currentMapId = ResolveMapIdFromDbc(wdt.MapName, opts.DbctoolLkDir, opts.Verbose);
            if (opts.Verbose)
                Console.WriteLine($"[MapId] {wdt.MapName} -> {currentMapId}");

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
                AreaMapperV2 = areaMapperV2,
                PatchMapping = patchMap,
                AlphaAreaIds = alphaAreaIds,
                WdtPath = wdt.WdtPath,
                AdtNumber = adtNum,
                AdtOffset = offset,
                MdnmFiles = wdt.MdnmFiles,
                MonmFiles = wdt.MonmFiles,
                Verbose = opts.Verbose,
                TrackAssets = opts.TrackAssets,
                CurrentMapId = currentMapId,
                VizSvg = opts.VizSvg,
                VizDir = opts.VizDir,
                LkDbcDir = opts.DbctoolLkDir,
                VizHtml = opts.VizHtml,
                PatchOnly = opts.PatchOnly,
                NoZoneFallback = opts.NoZoneFallback,
            };
            if (opts.Verbose && patchMap is not null)
                Console.WriteLine($"[PatchMap] alias={aliasUsed} dir={patchDirUsed} per-map={patchMap.PerMapCount} global={patchMap.GlobalCount} by-name[{wdt.MapName}]={patchMap.CountByName(wdt.MapName)} by-tgt-map[{currentMapId}]={patchMap.CountByTargetMap(currentMapId)}");
            // Track asset fixups per tile
            fixupLogger.BeginTile(wdt.MapName, x, y);
            AdtWotlkWriter.WriteBinary(ctx);
            var (fuzzy, capacity, overflow) = fixupLogger.EndTile();
            Console.WriteLine($"[Fixups] {wdt.MapName} {x},{y}: fuzzy={fuzzy} capacity={capacity} overflow={overflow}");
        }

        var tileList = candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty).ToList();
        bool canParallel = !opts.EnableFixups && !opts.TrackAssets; // Fixup logging isn't thread-safe
        int mdp = opts.MaxDegreeOfParallelism ?? Environment.ProcessorCount;
        if (canParallel && tileList.Count > 1)
        {
            Parallel.ForEach(tileList, new ParallelOptions { MaxDegreeOfParallelism = mdp }, t => ProcessTile(t.tx, t.ty));
        }
        else
        {
            foreach (var (x, y) in tileList) ProcessTile(x, y);
        }

        if (opts.VizHtml)
        {
            try { AdtWotlkWriter.WriteMapVisualizationHtml(opts.ExportDir, wdt.MapName, patchMap, opts.VizDir); }
            catch (Exception ex) { Console.Error.WriteLine($"[VizMap] Failed for {wdt.MapName}: {ex.Message}"); }
        }
    }

    public static void ExportBatch(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.InputRoot)) throw new ArgumentException("InputRoot required", nameof(opts.InputRoot));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);

        var wdts = Directory.EnumerateFiles(opts.InputRoot!, "*.wdt", SearchOption.AllDirectories)
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase);

        var inventory = new AssetInventory(opts.AssetRoots?.Where(r => !string.IsNullOrWhiteSpace(r)).Select(r => r!));

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

                var areaMapper = AreaIdMapper.TryCreate(null, null, null, opts.RemapPath);
                AreaIdMapperV2? areaMapperV2 = null;
                if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbctoolSrcDir) && !string.IsNullOrWhiteSpace(opts.DbctoolLkDir))
                {
                    var alias = string.IsNullOrWhiteSpace(opts.DbctoolSrcAlias) ? "0.5.3" : opts.DbctoolSrcAlias!;
                    areaMapperV2 = AreaIdMapperV2.TryCreate(opts.DbdDir!, alias, opts.DbctoolSrcDir!, opts.DbctoolLkDir!);
                }
                // Load DBCTool.V2 patch mapping (prefer stable out root -> <alias>/compare/v2)
                DbcPatchMapping? patchMap = null;
                string aliasUsed = ResolveSrcAlias(opts.DbctoolSrcAlias, wdt.WdtPath, opts.InputRoot);
                string patchDirUsed = string.Empty;
                if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) || !string.IsNullOrWhiteSpace(opts.DbctoolPatchDir))
                {
                    patchMap = new DbcPatchMapping();
                    if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) && File.Exists(opts.DbctoolPatchFile!))
                        patchMap.LoadFile(opts.DbctoolPatchFile!);
                    if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchDir) && Directory.Exists(opts.DbctoolPatchDir!))
                    {
                        patchDirUsed = opts.DbctoolPatchDir!;
                        foreach (var f in Directory.EnumerateFiles(opts.DbctoolPatchDir!, "Area_patch_crosswalk_*.csv", SearchOption.AllDirectories))
                            patchMap.LoadFile(f);
                    }
                }
                else if (!string.IsNullOrWhiteSpace(opts.DbctoolOutRoot))
                {
                    var v2Dir = Path.Combine(opts.DbctoolOutRoot!, aliasUsed, "compare", "v2");
                    if (Directory.Exists(v2Dir))
                    {
                        patchDirUsed = v2Dir;
                        patchMap = new DbcPatchMapping();
                        foreach (var f in Directory.EnumerateFiles(v2Dir, "Area_patch_crosswalk_*.csv", SearchOption.AllDirectories))
                            patchMap.LoadFile(f);
                    }
                    else if (opts.PatchOnly)
                    {
                        Console.Error.WriteLine($"[PatchOnly] No CSVs found under --dbctool-out-root: {v2Dir} for map {mapName}. Skipping map.");
                        continue;
                    }
                }
                else if (opts.PatchOnly)
                {
                    Console.Error.WriteLine($"[PatchOnly] Missing --dbctool-out-root or --dbctool-patch-dir/--dbctool-patch-file for map {mapName}. Skipping map.");
                    continue;
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

                int processedB = 0; int totalTilesB = 0;
        void ProcessTileB(int x, int y)
        {
            if (totalTilesB == 0) totalTilesB = candidateTiles.Count;
            var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
            var g = hasGroup ? group! : Array.Empty<PlacementRecord>();

            int adtNum = (y * 64) + x;
            int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (offset <= 0) return;
            var n = Interlocked.Increment(ref processedB);
            if (!opts.Verbose)
                Console.WriteLine($"[Tile] {wdt.MapName} {x},{y} ({n}/{totalTilesB})");
            var alpha = new AdtAlpha(wdt.WdtPath, offset, adtNum);
            var alphaAreaIds = (IReadOnlyList<int>)alpha.GetAlphaMcnkAreaIds();

            int currentMapId = ResolveMapIdFromDbc(wdt.MapName, opts.DbctoolLkDir, opts.Verbose);
            if (opts.Verbose)
                Console.WriteLine($"[MapId] {wdt.MapName} -> {currentMapId}");

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
                        AreaMapperV2 = areaMapperV2,
                        PatchMapping = patchMap,
                        AlphaAreaIds = alphaAreaIds,
                        WdtPath = wdt.WdtPath,
                        AdtNumber = adtNum,
                        AdtOffset = offset,
                        MdnmFiles = wdt.MdnmFiles,
                        MonmFiles = wdt.MonmFiles,
                        Verbose = opts.Verbose,
                        TrackAssets = opts.TrackAssets,
                        CurrentMapId = currentMapId,
                        VizSvg = opts.VizSvg,
                        VizDir = opts.VizDir,
                        LkDbcDir = opts.DbctoolLkDir,
                        VizHtml = opts.VizHtml,
                        PatchOnly = opts.PatchOnly
                    };
                    if (opts.Verbose && patchMap is not null)
                        Console.WriteLine($"[PatchMap] alias={aliasUsed} dir={patchDirUsed} per-map={patchMap.PerMapCount} global={patchMap.GlobalCount} by-name[{wdt.MapName}]={patchMap.CountByName(wdt.MapName)} by-tgt-map[{currentMapId}]={patchMap.CountByTargetMap(currentMapId)}");
                    fixupLogger.BeginTile(wdt.MapName, x, y);
                    AdtWotlkWriter.WriteBinary(ctx);
                    var (fuzzy, capacity, overflow) = fixupLogger.EndTile();
                    Console.WriteLine($"[Fixups] {wdt.MapName} {x},{y}: fuzzy={fuzzy} capacity={capacity} overflow={overflow}");
                }

                var tileList = candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty).ToList();
                bool canParallel = !opts.EnableFixups && !opts.TrackAssets;
                int mdp = opts.MaxDegreeOfParallelism ?? Environment.ProcessorCount;
                if (canParallel && tileList.Count > 1)
                {
                    Parallel.ForEach(tileList, new ParallelOptions { MaxDegreeOfParallelism = mdp }, t => ProcessTileB(t.tx, t.ty));
                }
                else
                {
                    foreach (var (x, y) in tileList) ProcessTileB(x, y);
                }

                if (opts.VizHtml)
                {
                    try { AdtWotlkWriter.WriteMapVisualizationHtml(opts.ExportDir, mapName, patchMap, opts.VizDir); }
                    catch (Exception ex) { Console.Error.WriteLine($"[VizMap] Failed for {mapName}: {ex.Message}"); }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Export failed for {wdtPath}: {ex}");
            }
        }
    }

    private static int ResolveMapIdFromDbc(string mapName, string? lkDir, bool verbose)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(lkDir))
            {
                var mapDbc = Path.Combine(lkDir!, "Map.dbc");
                if (File.Exists(mapDbc))
                {
                    return ReadMapDbcIdByDirectory(mapDbc, mapName);
                }
                else if (verbose)
                {
                    Console.Error.WriteLine($"[MapId] Map.dbc not found under --dbctool-lk-dir: {mapDbc}");
                }
            }
        }
        catch (Exception ex)
        {
            if (verbose) Console.Error.WriteLine($"[MapId] Error resolving map id for {mapName}: {ex.Message}");
        }
        return -1;
    }

    private static int ReadMapDbcIdByDirectory(string dbcPath, string targetName)
    {
        using var fs = new FileStream(dbcPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);
        // Read header
        var magic = br.ReadBytes(4);
        if (magic.Length != 4) return -1;
        int recordCount = br.ReadInt32();
        int fieldCount = br.ReadInt32();
        int recordSize = br.ReadInt32();
        int stringBlockSize = br.ReadInt32();
        var records = br.ReadBytes(recordCount * recordSize);
        var stringBlock = br.ReadBytes(stringBlockSize);
        for (int i = 0; i < recordCount; i++)
        {
            int baseOff = i * recordSize;
            var ints = new int[fieldCount];
            for (int f = 0; f < fieldCount; f++)
            {
                int off = baseOff + (f * 4);
                if (off + 4 <= records.Length) ints[f] = BitConverter.ToInt32(records, off);
            }
            int id = (fieldCount > 0) ? ints[0] : -1;
            if (id < 0) continue;
            // Compare against all string fields in the row; Map.dbc directory/name position varies across builds
            for (int f = 0; f < fieldCount; f++)
            {
                int sOff = ints[f];
                if (sOff > 0 && sOff < stringBlock.Length)
                {
                    int end = sOff;
                    while (end < stringBlock.Length && stringBlock[end] != 0) end++;
                    if (end > sOff)
                    {
                        var s = Encoding.UTF8.GetString(stringBlock, sOff, end - sOff);
                        if (!string.IsNullOrWhiteSpace(s) && s.Equals(targetName, StringComparison.OrdinalIgnoreCase))
                            return id;
                    }
                }
            }
        }
        return -1;
    }

    private static string ResolveSrcAlias(string? explicitAlias, string? singleWdtPath, string? inputRoot)
    {
        static string Normalize(string s)
        {
            var t = (s ?? string.Empty).Trim().ToLowerInvariant();
            if (t is "053" or "0.5.3" or "5.3") return "0.5.3";
            if (t is "055" or "0.5.5" or "5.5") return "0.5.5";
            if (t is "060" or "0.6.0" or "6.0" or "0.6") return "0.6.0";
            return s ?? string.Empty;
        }
        if (!string.IsNullOrWhiteSpace(explicitAlias)) return Normalize(explicitAlias!);
        var corpus = ($"{singleWdtPath}|{inputRoot}" ?? string.Empty).ToLowerInvariant();
        if (corpus.Contains("0.6.0") || corpus.Contains("\\060\\") || corpus.Contains("/060/") || corpus.Contains("0_6_0")) return "0.6.0";
        if (corpus.Contains("0.5.5") || corpus.Contains("\\055\\") || corpus.Contains("/055/") || corpus.Contains("0_5_5")) return "0.5.5";
        if (corpus.Contains("0.5.3") || corpus.Contains("\\053\\") || corpus.Contains("/053/") || corpus.Contains("0_5_3")) return "0.5.3";
        return "0.5.3"; // default
    }
}
