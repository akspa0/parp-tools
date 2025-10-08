using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AlphaWdtAnalyzer.Core;
using AlphaWdtAnalyzer.Core.Assets;
using AlphaWdtAnalyzer.Core.Dbc;
using AlphaWdtAnalyzer.Core.Export;
using DBCTool.V2.Core;

namespace WoWRollback.AdtModule.Convert;

internal static class ConvertPipelineMT
{
    internal sealed class Options
    {
        public required string WdtPath { get; init; }
        public required string ExportDir { get; init; }
        public string? CommunityListfilePath { get; init; }
        public string? LkListfilePath { get; init; }
        public string? DbdDir { get; init; }
        public string? DbctoolOutRoot { get; init; }
        public string? DbctoolSrcAlias { get; init; }
        public string? DbctoolSrcDir { get; init; }
        public string? DbctoolLkDir { get; init; }
        public string? DbctoolPatchDir { get; init; }
        public string? DbctoolPatchFile { get; init; }
        public bool ConvertToMh2o { get; init; } = true;
        public bool AssetFuzzy { get; init; } = true;
        public bool UseFallbacks { get; init; } = true;
        public bool EnableFixups { get; init; } = true;
        public bool TrackAssets { get; init; } = false;
        public bool Verbose { get; init; } = false;
        public string FallbackTileset { get; init; } = "Tileset\\Generic\\Checkers.blp";
        public string FallbackNonTilesetBlp { get; init; } = "Dungeons\\Textures\\temp\\64.blp";
        public string FallbackWmo { get; init; } = "wmo\\Dungeon\\test\\missingwmo.wmo";
        public string FallbackM2 { get; init; } = "World\\Scale\\HumanMaleScale.mdx";
        public int MaxDegreeOfParallelism { get; init; } = 0;
    }

    public static void Run(Options opts)
    {
        if (string.IsNullOrWhiteSpace(opts.WdtPath)) throw new ArgumentException("WdtPath required", nameof(opts.WdtPath));
        Directory.CreateDirectory(opts.ExportDir);

        var resolver = MultiListfileResolver.FromFiles(opts.LkListfilePath, opts.CommunityListfilePath);
        var mapName = Path.GetFileNameWithoutExtension(opts.WdtPath);
        var logDir = Path.Combine(opts.ExportDir, "csv", "maps", mapName);
        Directory.CreateDirectory(logDir);
        var inventory = new AssetInventory(null);

        var areaMapper = AreaIdMapper.TryCreate(null, null, null, null);
        AreaIdMapperV2? areaMapperV2 = null;
        var aliasUsed = ResolveSrcAlias(opts.DbctoolSrcAlias, opts.WdtPath, null);
        if (!string.IsNullOrWhiteSpace(opts.DbdDir) && !string.IsNullOrWhiteSpace(opts.DbctoolSrcDir) && !string.IsNullOrWhiteSpace(opts.DbctoolLkDir))
        {
            areaMapperV2 = AreaIdMapperV2.TryCreate(opts.DbdDir!, aliasUsed, opts.DbctoolSrcDir!, opts.DbctoolLkDir!);
        }

        // Patch mapping
        DbcPatchMapping? patchMap = null;
        string patchDirUsed = string.Empty;
        if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile) || !string.IsNullOrWhiteSpace(opts.DbctoolPatchDir))
        {
            patchMap = new DbcPatchMapping();
            bool loadedAny = false;
            if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchFile))
            {
                var resolved = ResolvePatchFilePath(opts.DbctoolPatchFile!, opts.DbctoolPatchDir, opts.DbctoolOutRoot, aliasUsed);
                if (File.Exists(resolved)) { patchMap.LoadFile(resolved); loadedAny = true; }
            }
            if (!string.IsNullOrWhiteSpace(opts.DbctoolPatchDir) && Directory.Exists(opts.DbctoolPatchDir!))
            {
                patchDirUsed = opts.DbctoolPatchDir!;
                foreach (var f in EnumerateCrosswalkCsvs(opts.DbctoolPatchDir!)) { patchMap.LoadFile(f); loadedAny = true; }
            }
            if (!loadedAny) patchMap = null;
        }
        else if (!string.IsNullOrWhiteSpace(opts.DbctoolOutRoot))
        {
            var v2Dir = Path.Combine(opts.DbctoolOutRoot!, aliasUsed, "compare", "v2");
            var v3Dir = Path.Combine(opts.DbctoolOutRoot!, aliasUsed, "compare", "v3");
            var search = new List<string>();
            if (Directory.Exists(v3Dir)) search.Add(v3Dir);
            if (Directory.Exists(v2Dir)) search.Add(v2Dir);
            if (search.Count > 0)
            {
                patchDirUsed = search[0];
                patchMap = new DbcPatchMapping();
                bool loadedAny = false;
                foreach (var dir in search)
                {
                    foreach (var f in EnumerateCrosswalkCsvs(dir)) { patchMap.LoadFile(f); loadedAny = true; }
                }
                if (!loadedAny) patchMap = null;
            }
        }

        var wdt = new WdtAlphaScanner(opts.WdtPath);

        // MapIdResolver for LK map guard
        MapIdResolver? mapIdResolver = null;
        if (!string.IsNullOrWhiteSpace(opts.DbctoolOutRoot))
        {
            mapIdResolver = MapIdResolver.LoadFromDbcToolOutput(opts.DbctoolOutRoot!, aliasUsed);
        }

        // Candidate tiles (union of placements + present ADTs)
        var adtScanner = new AdtScanner();
        var scanned = adtScanner.Scan(wdt);
        var placementsByTile = scanned.Placements.GroupBy(p => (p.TileX, p.TileY)).ToDictionary(g => g.Key, g => (IEnumerable<PlacementRecord>)g);
        var candidateTiles = new HashSet<(int tx, int ty)>(placementsByTile.Keys);
        for (int adtNum = 0; adtNum < wdt.AdtMhdrOffsets.Count; adtNum++)
        {
            if (wdt.AdtMhdrOffsets[adtNum] > 0) { candidateTiles.Add((adtNum % 64, adtNum / 64)); }
        }

        int processed = 0; int totalTiles = candidateTiles.Count;
        int dop = opts.MaxDegreeOfParallelism > 0 ? opts.MaxDegreeOfParallelism : Math.Max(1, Environment.ProcessorCount - 1);
        var tileList = candidateTiles.OrderBy(t => t.tx).ThenBy(t => t.ty).ToList();
        Parallel.ForEach(tileList, new ParallelOptions { MaxDegreeOfParallelism = dop }, t =>
        {
            var (x, y) = t;
            var hasGroup = placementsByTile.TryGetValue((x, y), out var group);
            var g = hasGroup ? group! : Array.Empty<PlacementRecord>();
            int adtNum = (y * 64) + x;
            int offset = (adtNum < wdt.AdtMhdrOffsets.Count) ? wdt.AdtMhdrOffsets[adtNum] : 0;
            if (offset <= 0) return;
            var n = Interlocked.Increment(ref processed);
            if (!opts.Verbose) Console.WriteLine($"[Tile] {wdt.MapName} {x},{y} ({n}/{totalTiles})");

            var alpha = new GillijimProject.WowFiles.Alpha.AdtAlpha(wdt.WdtPath, offset, adtNum);
            var alphaAreaIds = (IReadOnlyList<int>)alpha.GetAlphaMcnkAreaIds();

            int currentMapId = ResolveMapIdFromDbc(wdt.MapName, mapIdResolver, opts.DbctoolLkDir, opts.Verbose);

            var tileLogPath = Path.Combine(logDir, $"asset_fixups_{x}_{y}.csv");
            using var tileLogger = new FixupLogger(tileLogPath);
            var tileFixup = new AssetFixupPolicy(
                resolver,
                opts.FallbackTileset,
                opts.FallbackNonTilesetBlp,
                opts.FallbackWmo,
                opts.FallbackM2,
                opts.AssetFuzzy,
                opts.UseFallbacks,
                opts.EnableFixups,
                tileLogger,
                inventory,
                logExact: false);

            var ctx = new AdtWotlkWriter.WriteContext
            {
                ExportDir = opts.ExportDir,
                MapName = wdt.MapName,
                TileX = x,
                TileY = y,
                Placements = g,
                Fixup = tileFixup,
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
                VizSvg = false,
                VizDir = null,
                LkDbcDir = opts.DbctoolLkDir,
                VizHtml = false,
                PatchOnly = false,
                NoZoneFallback = false,
            };

            tileLogger.BeginTile(wdt.MapName, x, y);
            AdtWotlkWriter.WriteBinary(ctx);
            var (fuzzy, capacity, overflow) = tileLogger.EndTile();
            Console.WriteLine($"[Fixups] {wdt.MapName} {x},{y}: fuzzy={fuzzy} capacity={capacity} overflow={overflow}");
        });

        // Merge per-tile fixups
        try
        {
            var mergedPath = Path.Combine(logDir, "asset_fixups.csv");
            using var writer = new StreamWriter(mergedPath, append: false);
            writer.WriteLine("type,original,resolved,method,map,tile_x,tile_y");
            foreach (var file in Directory.EnumerateFiles(logDir, "asset_fixups_*.csv", SearchOption.TopDirectoryOnly))
            {
                foreach (var line in File.ReadLines(file).Skip(1)) { writer.WriteLine(line); }
            }
        }
        catch { }
    }

    private static IEnumerable<string> EnumerateCrosswalkCsvs(string directory)
    {
        var patterns = new[] { "Area_patch_crosswalk_*.csv", "Area_crosswalk_v*.csv" };
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var pattern in patterns)
        {
            foreach (var file in Directory.EnumerateFiles(directory, pattern, SearchOption.AllDirectories))
            {
                if (seen.Add(file)) yield return file;
            }
        }
    }

    private static string ResolvePatchFilePath(string patchFile, string? patchDir, string? outRoot, string alias)
    {
        if (Path.IsPathFullyQualified(patchFile)) return patchFile;
        if (!string.IsNullOrWhiteSpace(patchDir) && Directory.Exists(patchDir))
        {
            var combined = Path.Combine(patchDir, patchFile);
            if (File.Exists(combined)) return combined;
        }
        if (!string.IsNullOrWhiteSpace(outRoot))
        {
            var v3 = Path.Combine(outRoot, alias, "compare", "v3", patchFile);
            if (File.Exists(v3)) return v3;
            var v2 = Path.Combine(outRoot, alias, "compare", "v2", patchFile);
            if (File.Exists(v2)) return v2;
        }
        var cwd = Path.Combine(Directory.GetCurrentDirectory(), patchFile);
        return cwd;
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
        return "0.5.3";
    }

    private static int ResolveMapIdFromDbc(string mapName, MapIdResolver? resolver, string? lkDir, bool verbose)
    {
        if (resolver != null)
        {
            var mapId = resolver.GetMapIdByDirectory(mapName);
            if (mapId.HasValue) return mapId.Value;
        }
        return -1;
    }
}
