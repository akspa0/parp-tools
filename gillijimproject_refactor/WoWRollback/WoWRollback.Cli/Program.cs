using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services;
using WoWRollback.Core.Services.Config;
using WoWRollback.Core.Services.Viewer;
using WoWRollback.WDLtoGLB;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.FileProviders;
using WoWFormatLib.FileReaders;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace WoWRollback.Cli;

internal static class Program
{
    private static readonly Dictionary<(string Version, string Map), string?> AlphaWdtCache = new(new TupleComparer());

    private static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help")
        {
            PrintHelp();
            return 0;
        }

        var cmd = args[0].ToLowerInvariant();
        var opts = ParseArgs(args.Skip(1).ToArray());

        try
        {
            switch (cmd)
            {
                case "analyze-alpha-wdt":
                    return RunAnalyzeAlphaWdt(opts);
                case "analyze-lk-adt":
                    return RunAnalyzeLkAdt(opts);
                case "analyze-ranges": // Legacy alias for analyze-lk-adt
                    return RunAnalyzeLkAdt(opts);
                case "wdl-to-glb":
                    return RunWdlToGlb(opts);
                case "viewer-pack":
                    return RunViewerPack(opts);
                case "viewer-serve":
                    return RunViewerServe(opts);
                case "dry-run":
                    return RunDryRun(opts);
                case "compare-versions":
                    return RunCompareVersions(opts);
                default:
                    Console.Error.WriteLine($"Unknown command: {cmd}");
                    PrintHelp();
                    return 2;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[error] {ex.Message}");
            return 1;
        }
    }

    private static int RunAnalyzeAlphaWdt(Dictionary<string, string> opts)
    {
        Require(opts, "wdt-file");
        var wdtFile = opts["wdt-file"];
        var outRoot = opts.GetValueOrDefault("out", "");
        var mapName = Path.GetFileNameWithoutExtension(wdtFile);

        var buildTag = BuildTagResolver.ResolveForPath(Path.GetDirectoryName(Path.GetFullPath(wdtFile)) ?? wdtFile);
        var sessionDir = OutputSession.Create(outRoot, mapName, buildTag);
        Console.WriteLine($"[info] Archaeological analysis session: {sessionDir}");
        Console.WriteLine($"[info] Excavating Alpha WDT: {wdtFile}");
        Console.WriteLine($"[info] Using raw Alpha coordinates (no transforms)");

        var analysis = WoWRollback.Core.Services.AlphaWdtAnalyzer.AnalyzeAlphaWdt(wdtFile);
        var csvResult = RangeCsvWriter.WritePerMapCsv(sessionDir, $"alpha_{mapName}", analysis.Ranges, analysis.Assets);

        Console.WriteLine($"[ok] Extracted {analysis.Ranges.Count} archaeological placement layers");
        Console.WriteLine($"[ok] Alpha UniqueID ranges written to: {csvResult.PerMapPath}");
        if (!string.IsNullOrEmpty(csvResult.TimelinePath))
        {
            Console.WriteLine($"[ok] Timeline CSV: {csvResult.TimelinePath}");
        }
        if (!string.IsNullOrEmpty(csvResult.AssetLedgerPath))
        {
            Console.WriteLine($"[ok] Asset ledger CSV: {csvResult.AssetLedgerPath}");
        }
        if (!string.IsNullOrEmpty(csvResult.TimelineAssetsPath))
        {
            Console.WriteLine($"[ok] Timeline asset summary CSV: {csvResult.TimelineAssetsPath}");
        }

        return 0;
    }

    private static int RunWdlToGlb(Dictionary<string, string> opts)
    {
        Require(opts, "map");
        Require(opts, "wdl");
        Require(opts, "out-glb");

        var mapName = opts["map"]; 
        var wdlPath = opts["wdl"]; 
        var outGlb = opts["out-glb"]; 

        var textureOverride = GetOption(opts, "texture");
        var invertX = string.Equals(GetOption(opts, "invert-x"), "true", StringComparison.OrdinalIgnoreCase);
        var scale = TryParseDouble(opts, "scale") ?? 1.0;
        var minimapFolder = GetOption(opts, "minimap-folder");
        var minimapRoot = GetOption(opts, "minimap-root");
        var trsPath = GetOption(opts, "trs");
        var perTileFlag = string.Equals(GetOption(opts, "per-tile"), "true", StringComparison.OrdinalIgnoreCase);

        var options = new WdlToGlbOptions
        {
            MapName = mapName,
            TextureOverridePath = textureOverride,
            MinimapFolder = minimapFolder,
            MinimapRoot = minimapRoot,
            TrsPath = trsPath,
            InvertX = invertX,
            Scale = (float)scale,
            SRGB = true,
            AnisotropyHint = null
        };

        Console.WriteLine($"[info] Converting WDL → GLB: map={mapName}");
        Console.WriteLine($"[info]  WDL:  {wdlPath}");
        if (!string.IsNullOrWhiteSpace(textureOverride)) Console.WriteLine($"[info]  Texture override: {textureOverride}");
        if (!string.IsNullOrWhiteSpace(minimapFolder)) Console.WriteLine($"[info]  Minimap folder: {minimapFolder}");
        if (!string.IsNullOrWhiteSpace(minimapRoot)) Console.WriteLine($"[info]  Minimap root: {minimapRoot}");
        if (!string.IsNullOrWhiteSpace(trsPath)) Console.WriteLine($"[info]  TRS: {trsPath}");
        Console.WriteLine($"[info]  Out:  {outGlb}");

        var code = WdlToGlbConverter.Convert(mapName, wdlPath, outGlb, options);
        if (code == 0)
        {
            Console.WriteLine("[ok] WDLtoGLB completed.");
        }
        return code;
    }

    private static bool ShouldGenerateViewer(Dictionary<string, string> opts)
    {
        if (!opts.TryGetValue("viewer-report", out var value))
            return false;

        return !string.Equals(value, "false", StringComparison.OrdinalIgnoreCase);
    }

    private static ViewerOptions BuildViewerOptions(Dictionary<string, string> opts)
    {
        var defaults = ViewerOptions.CreateDefault();

        var defaultVersion = opts.TryGetValue("default-version", out var requestedDefault)
            ? requestedDefault
            : defaults.DefaultVersion;

        var minimapWidth = TryParseInt(opts, "minimap-width") ?? defaults.MinimapWidth;
        var minimapHeight = TryParseInt(opts, "minimap-height") ?? defaults.MinimapHeight;
        var distanceThreshold = TryParseDouble(opts, "distance-threshold") ?? defaults.DiffDistanceThreshold;
        var moveEpsilon = TryParseDouble(opts, "move-epsilon") ?? defaults.MoveEpsilonRatio;

        return new ViewerOptions(
            defaultVersion,
            DiffPair: null,
            MinimapWidth: minimapWidth,
            MinimapHeight: minimapHeight,
            DiffDistanceThreshold: distanceThreshold,
            MoveEpsilonRatio: moveEpsilon);
    }

    private static (string Baseline, string Comparison)? ParseDiffPair(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        var parts = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
            return null;

        return (parts[0], parts[1]);
    }

    private static int? TryParseInt(Dictionary<string, string> opts, string key)
    {
        if (!opts.TryGetValue(key, out var value))
            return null;

        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : null;
    }

    private static double? TryParseDouble(Dictionary<string, string> opts, string key)
    {
        if (!opts.TryGetValue(key, out var value))
            return null;

        return double.TryParse(value, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : null;
    }

    private static int RunCompareVersions(Dictionary<string, string> opts)
    {
        Require(opts, "versions");
        var rootCandidate = GetOption(opts, "root");
        var outCandidate = GetOption(opts, "out");
        var root = string.IsNullOrWhiteSpace(rootCandidate) ? (outCandidate ?? string.Empty) : rootCandidate!;
        var outputRoot = string.IsNullOrWhiteSpace(root)
            ? Path.Combine(Directory.GetCurrentDirectory(), "rollback_outputs")
            : Path.GetFullPath(root);

        var versions = opts["versions"].Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();

        IReadOnlyList<string>? maps = null;
        if (opts.TryGetValue("maps", out var mapsSpec) && !string.IsNullOrWhiteSpace(mapsSpec))
        {
            maps = mapsSpec.Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim())
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .ToList();
        }

        EnsureComparisonPrerequisites(outputRoot, versions, maps, opts);

        Console.WriteLine($"[info] Comparing versions: {string.Join(", ", versions)}");
        if (!string.IsNullOrWhiteSpace(root)) Console.WriteLine($"[info] Root directory: {root}");
        if (maps is not null && maps.Count > 0) Console.WriteLine($"[info] Map filter: {string.Join(", ", maps)}");

        var result = VersionComparisonService.CompareVersions(root, versions, maps);
        var paths = VersionComparisonWriter.WriteOutputs(root, result);
        var wantYaml = opts.TryGetValue("yaml-report", out var yamlOpt) && !string.Equals(yamlOpt, "false", StringComparison.OrdinalIgnoreCase);
        if (wantYaml)
        {
            var yamlRoot = VersionComparisonWriter.WriteYamlReports(root, result);
            Console.WriteLine($"[ok] YAML reports written to: {yamlRoot}");
        }

        if (ShouldGenerateViewer(opts))
        {
            var options = BuildViewerOptions(opts);
            var diffPair = ParseDiffPair(opts.GetValueOrDefault("diff"));
            var viewerWriter = new ViewerReportWriter();
            var viewerRoot = viewerWriter.Generate(paths.ComparisonDirectory, result, options, diffPair);
            if (string.IsNullOrEmpty(viewerRoot))
            {
                Console.WriteLine("[info] Viewer assets skipped (no placement data).");
            }
            else
            {
                Console.WriteLine($"[ok] Viewer assets written to: {viewerRoot}");
            }
        }

        Console.WriteLine($"[ok] Comparison key: {result.ComparisonKey}");
        Console.WriteLine($"[ok] Outputs written to: {paths.ComparisonDirectory}");
        if (result.Warnings.Count > 0)
        {
            Console.WriteLine($"[warn] {result.Warnings.Count} warnings emitted. See warnings_{result.ComparisonKey}.txt");
        }
        return 0;
    }

    private static int RunAnalyzeLkAdt(Dictionary<string, string> opts)
    {
        Require(opts, "input-dir");
        var map = opts["map"]; 
        var inputDir = opts["input-dir"]; 
        var outRoot = opts.GetValueOrDefault("out", "");
        
        var buildTag = BuildTagResolver.ResolveForPath(inputDir);
        var sessionDir = OutputSession.Create(outRoot, map, buildTag);
        Console.WriteLine($"[info] LK ADT analysis session: {sessionDir}");
        Console.WriteLine($"[info] Analyzing converted LK ADT files for map: {map}");

        var ranges = RangeScanner.AnalyzeRangesForMap(inputDir, map);
        var csvResult = RangeCsvWriter.WritePerMapCsv(sessionDir, $"lk_{map}", ranges);

        Console.WriteLine($"[ok] Extracted {ranges.Count} preserved placement ranges from LK ADTs");
        Console.WriteLine($"[ok] LK UniqueID ranges written to: {csvResult.PerMapPath}");
        if (!string.IsNullOrEmpty(csvResult.TimelinePath))
        {
            Console.WriteLine($"[ok] LK timeline CSV: {csvResult.TimelinePath}");
        }
        if (!string.IsNullOrEmpty(csvResult.AssetLedgerPath))
        {
            Console.WriteLine($"[ok] LK asset ledger CSV: {csvResult.AssetLedgerPath}");
        }
        if (!string.IsNullOrEmpty(csvResult.TimelineAssetsPath))
        {
            Console.WriteLine($"[ok] LK timeline asset summary CSV: {csvResult.TimelineAssetsPath}");
        }

        return 0;
    }

    private static int RunDryRun(Dictionary<string, string> opts)
    {
        Require(opts, "input-dir");
        var map = opts["map"]; var inputDir = opts["input-dir"]; var outRoot = opts.GetValueOrDefault("out", "");

        RangeConfig config = new RangeConfig { Map = map, Mode = opts.GetValueOrDefault("mode", "keep") };
        if (opts.TryGetValue("config", out var cfgPath) && File.Exists(cfgPath))
        {
            config = RangeConfigLoader.LoadFromJson(cfgPath);
        }

        var keepRanges = opts.TryGetValue("keep-range", out var keepSpec) ? new[] { keepSpec } : Array.Empty<string>();
        var dropRanges = opts.TryGetValue("drop-range", out var dropSpec) ? new[] { dropSpec } : Array.Empty<string>();
        RangeConfigLoader.ApplyCliOverrides(config, keepRanges, dropRanges);

        var adts = Directory.EnumerateFiles(inputDir, "*.adt", SearchOption.AllDirectories)
            .Where(p => Path.GetFileNameWithoutExtension(p).StartsWith(map + "_", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();

        Console.WriteLine($"[info] Dry-run for map={map}, mode={config.Mode}, include={config.IncludeRanges.Count}, exclude={config.ExcludeRanges.Count}");
        int total = 0, removed = 0;
        foreach (var adt in adts)
        {
            int localTotal = 0, localRemoved = 0;
            foreach (var entry in AdtPlacementAnalyzer.EnumeratePlacements(adt))
            {
                localTotal++;
                if (RangeSelector.ShouldRemove(config, entry.UniqueId)) localRemoved++;
            }
            total += localTotal; removed += localRemoved;
            Console.WriteLine($"{Path.GetFileName(adt)}, total={localTotal}, would_remove={localRemoved}");
        }
        Console.WriteLine($"[summary] total={total}, would_remove={removed}");
        return 0;
    }

    private static void Require(Dictionary<string, string> opts, string key)
    {
        if (!opts.ContainsKey(key) || string.IsNullOrWhiteSpace(opts[key]))
            throw new ArgumentException($"Missing required --{key}");
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a.StartsWith("--"))
            {
                var key = a.Substring(2);
                string val = "true";
                if (i + 1 < args.Length && !args[i + 1].StartsWith("--")) { val = args[++i]; }
                dict[key] = val;
            }
        }
        return dict;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("WoWRollback CLI - Digital Archaeology of World of Warcraft Development");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  analyze-alpha-wdt --wdt-file <path> [--out <dir>]");
        Console.WriteLine("    Extract UniqueID ranges from Alpha WDT files (archaeological excavation)");
        Console.WriteLine();
        Console.WriteLine("  analyze-lk-adt    --map <name> --input-dir <dir> [--out <dir>]");
        Console.WriteLine("    Extract UniqueID ranges from converted LK ADT files (preservation analysis)");
        Console.WriteLine();
        Console.WriteLine("  wdl-to-glb        --map <name> --wdl <file> --out-glb <file> [--texture <image>] [--invert-x] [--scale <float>] [--minimap-folder <dir>] [--minimap-root <dir>] [--trs <file>] [--per-tile]");
        Console.WriteLine("    Export GLB terrain from WDL. --texture bakes a single image for merged output.");
        Console.WriteLine("    Per-tile textures prefer md5translate under --minimap-root (auto-detects md5translate.trs/txt). Fallback to --minimap-folder name index.");
        Console.WriteLine();
        Console.WriteLine("  viewer-pack build --session-root <dir> --minimap-root <dir> --out <dir> [--maps m1,m2] [--label <text>]");
        Console.WriteLine("    Build a self-contained 'viewer-pack' from harvested data and TRS tiles (PNG). No parsing at view time.");
        Console.WriteLine();
        Console.WriteLine("  viewer-serve      --pack <dir>");
        Console.WriteLine("    Serve the 2D viewer UI and a prebuilt viewer-pack at /data. Open / to use the viewer.");
        Console.WriteLine();
        Console.WriteLine("  dry-run           --map <name> --input-dir <dir> [--config <file>] [--keep-range min:max] [--drop-range min:max] [--mode keep|drop]");
        Console.WriteLine("    Preview rollback effects without modifying files");
        Console.WriteLine();
        Console.WriteLine("  compare-versions  --versions v1,v2[,v3...] [--maps m1,m2,...] [--root <dir>] [--yaml-report]");
        Console.WriteLine("                    [--viewer-report] [--default-version <ver>] [--diff base,comp]");
        Console.WriteLine("    Compare placement ranges across versions; outputs CSVs under rollback_outputs/comparisons/<key>");
        Console.WriteLine("    If --yaml-report is present, also writes YAML exploration reports under .../<key>/yaml/.");
        Console.WriteLine("    --viewer-report additionally emits minimaps, overlays, and diffs for the static viewer.");
        Console.WriteLine();
        Console.WriteLine("Archaeological Perspective:");
        Console.WriteLine("  Each UniqueID range represents a 'volume of work' by ancient developers.");
        Console.WriteLine("  Singleton IDs and outliers are precious artifacts showing experiments and tests.");
        Console.WriteLine("  We're uncovering sedimentary layers of 20+ years of WoW development history.");
    }

    // --- 2D Viewer Host ---
    private static int RunViewerServe(Dictionary<string, string> opts)
    {
        // Require viewer-pack path
        var packRootOpt = GetOption(opts, "pack");
        if (string.IsNullOrWhiteSpace(packRootOpt) || !Directory.Exists(packRootOpt))
        {
            Console.Error.WriteLine("[error] --pack <dir> is required and must point to a built viewer-pack (contains index.json, tiles/).");
            return 2;
        }
        var packRoot = Path.GetFullPath(packRootOpt!);

        // Build web app (static UI + static pack mounted at /data)
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://localhost:5000");
        var app = builder.Build();
        // Static files: serve UI from assets2d and pack under /data
        var staticRoot = FindAssets2DRoot();
        if (Directory.Exists(staticRoot))
        {
            Console.WriteLine($"[info] Serving static UI from: {staticRoot}");
            app.UseDefaultFiles(new DefaultFilesOptions
            {
                FileProvider = new PhysicalFileProvider(staticRoot)
            });
            app.UseStaticFiles(new StaticFileOptions
            {
                FileProvider = new PhysicalFileProvider(staticRoot)
            });
        }
        else
        {
            Console.WriteLine($"[warn] Static root not found: {staticRoot}");
        }

        if (Directory.Exists(packRoot))
        {
            Console.WriteLine($"[info] Serving viewer-pack from: {packRoot} at /data");
            app.UseStaticFiles(new StaticFileOptions
            {
                FileProvider = new PhysicalFileProvider(packRoot),
                RequestPath = "/data"
            });
        }
        else
        {
            Console.WriteLine($"[error] Pack directory not found: {packRoot}");
            return 2;
        }

        Console.WriteLine("[ok] Viewer server running on http://localhost:5000");
        app.Run();
        return 0;
    }

    private static string FindAssets2DRoot()
    {
        // bin/Debug/net9.0 -> up 4 levels to WoWRollback folder
        var baseDir = AppContext.BaseDirectory; // .../WoWRollback.Cli/bin/Debug/net9.0/
        var repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));
        var assets2d = Path.Combine(repoRoot, "WoWRollback.Viewer", "assets2d");
        return assets2d;
    }

    private static Dictionary<(string map,int x,int y), string>? TryBuildTrsIndex(string minimapRoot, string? trsPath)
    {
        if (string.IsNullOrWhiteSpace(minimapRoot) || !Directory.Exists(minimapRoot)) return null;
        string? trs = trsPath;
        if (string.IsNullOrWhiteSpace(trs))
        {
            var p1 = Path.Combine(minimapRoot, "md5translate.trs");
            var p2 = Path.Combine(minimapRoot, "md5translate.txt");
            trs = File.Exists(p1) ? p1 : (File.Exists(p2) ? p2 : null);
        }
        if (string.IsNullOrWhiteSpace(trs) || !File.Exists(trs)) return null;

        var dict = new Dictionary<(string,int,int), string>();
        var baseDir = Path.GetDirectoryName(trs)!;
        string? currentMap = null;
        foreach (var raw in File.ReadAllLines(trs))
        {
            var line = raw.Trim();
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
            if (line.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
            {
                currentMap = line.Substring(4).Trim();
                continue;
            }
            if (currentMap == null) continue;
            var parts = line.Split('\t');
            if (parts.Length != 2) continue;
            string a = parts[0].Trim();
            string b = parts[1].Trim();
            string mapSide = (a.Contains("map") && a.Contains(".blp", StringComparison.OrdinalIgnoreCase)) ? a : b;
            string actual = mapSide == a ? b : a;

            var stem = Path.GetFileNameWithoutExtension(mapSide);
            if (!stem.StartsWith("map", StringComparison.OrdinalIgnoreCase)) continue;
            var xy = stem.Substring(3).Split('_');
            if (xy.Length != 2 || !int.TryParse(xy[0], out var tx) || !int.TryParse(xy[1], out var ty)) continue;
            var fullPath = Path.Combine(baseDir, actual.Replace('/', Path.DirectorySeparatorChar));
            dict[(currentMap.ToLowerInvariant(), tx, ty)] = fullPath;
        }
        return dict;
    }

    private sealed class ViewerHostConfig
    {
        public string? MinimapRoot { get; set; }
        public string? TrsPath { get; set; }
        public string? DefaultMap { get; set; }
        public string? VersionLabel { get; set; }
    }

    // --- Build a viewer-pack from harvested data + TRS tiles ---
    private static int RunViewerPack(Dictionary<string, string> opts)
    {

        Require(opts, "session-root");
        Require(opts, "minimap-root");
        Require(opts, "out");

        var sessionRoot = Path.GetFullPath(opts["session-root"]);
        var minimapRoot = Path.GetFullPath(opts["minimap-root"]);
        var outRoot = Path.GetFullPath(opts["out"]);
        var label = GetOption(opts, "label", "dev");
        var mapsFilter = opts.TryGetValue("maps", out var m) && !string.IsNullOrWhiteSpace(m)
            ? m.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet(StringComparer.OrdinalIgnoreCase)
            : null;

        if (!Directory.Exists(sessionRoot)) throw new DirectoryNotFoundException(sessionRoot);
        if (!Directory.Exists(minimapRoot)) throw new DirectoryNotFoundException(minimapRoot);
        Directory.CreateDirectory(outRoot);

        // Build versions from session-root and use existing MinimapLocator to resolve TRS and files
        static bool ContainsMinimapIndicators(string dir)
        {
            try
            {
                if (Directory.Exists(Path.Combine(dir, "tree"))) return true;
                if (Directory.EnumerateFiles(dir, "md5translate.*", SearchOption.AllDirectories).Any()) return true;
                var candidates = new[]
                {
                    Path.Combine(dir, "tree", "World", "Textures", "Minimap"),
                    Path.Combine(dir, "tree", "world", "textures", "minimap"),
                    Path.Combine(dir, "World", "Textures", "Minimap"),
                    Path.Combine(dir, "world", "textures", "minimap"),
                    Path.Combine(dir, "Textures", "Minimap"),
                    Path.Combine(dir, "textures", "minimap")
                };
                return candidates.Any(Directory.Exists);
            }
            catch { return false; }
        }

        var versions = new List<string>();
        string rootBase;
        if (ContainsMinimapIndicators(sessionRoot))
        {
            // sessionRoot itself is a version directory
            rootBase = Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(sessionRoot)) ?? sessionRoot;
            versions.Add(Path.GetFileName(Path.TrimEndingDirectorySeparator(sessionRoot)) ?? "dev");
        }
        else
        {
            rootBase = sessionRoot;
            foreach (var sub in Directory.EnumerateDirectories(sessionRoot, "*", SearchOption.TopDirectoryOnly))
            {
                if (ContainsMinimapIndicators(sub))
                {
                    var name = Path.GetFileName(sub);
                    if (!string.IsNullOrWhiteSpace(name)) versions.Add(name);
                }
            }
        }

        if (versions.Count == 0)
        {
            Console.Error.WriteLine("[error] No version directories with minimap data found under session-root.");
            return 2;
        }

        var locator = MinimapLocator.Build(rootBase, versions);

        // Collect maps across versions
        var allMaps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var ver in versions)
        {
            foreach (var mname in locator.EnumerateMaps(ver)) allMaps.Add(mname);
        }
        var maps = allMaps.OrderBy(s => s).ToList();
        if (mapsFilter is not null) maps = maps.Where(mn => mapsFilter.Contains(mn)).ToList();
        var defaultMap = maps.FirstOrDefault() ?? "";

        // Write tiles using locator streams → WebP tiles at out/tiles/<map>/<x>_<y>.webp
        int written = 0;
        var composer = new MinimapComposer();
        foreach (var ver in versions)
        {
            foreach (var map in maps)
            {
                var mapDir = Path.Combine(outRoot, "tiles", map);
                Directory.CreateDirectory(mapDir);
                foreach (var (row, col) in locator.EnumerateTiles(ver, map))
                {
                    if (!locator.TryGetTile(ver, map, row, col, out var tile)) continue;
                    var x = col; var y = row;
                    var dst = Path.Combine(mapDir, $"{x}_{y}.webp");
                    if (File.Exists(dst)) { written++; continue; }
                    try
                    {
                        using var fsSrc = tile.Open();
                        // Synchronously compose to keep method non-async
                        System.Threading.Tasks.Task.Run(async () => await composer.ComposeAsync(fsSrc, dst, ViewerOptions.CreateDefault())).Wait();
                        written++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[warn] Failed tile {map} {x}_{y}: {ex.Message}");
                    }
                }
                // Minimal overlays scaffold
                var overlaysDir = Path.Combine(outRoot, "overlays", map);
                Directory.CreateDirectory(overlaysDir);
                var manifestPath = Path.Combine(overlaysDir, "manifest.json");
                File.WriteAllText(manifestPath, JsonSerializer.Serialize(new { layers = Array.Empty<object>() }));
            }
        }

        // index.json
        var indexPath = Path.Combine(outRoot, "index.json");
        var indexObj = new { version = label, defaultMap, maps = maps.Select(n => new { name = n, size = 64 }).ToArray() };
        File.WriteAllText(indexPath, JsonSerializer.Serialize(indexObj));

        Console.WriteLine($"[ok] Viewer-pack written to: {outRoot} (tiles: {written}, maps: {maps.Count})");
        return 0;
    }

    private static void EnsureComparisonPrerequisites(
        string outputRoot,
        IReadOnlyList<string> versions,
        IReadOnlyList<string>? maps,
        Dictionary<string, string> opts)
    {
        if (!Directory.Exists(outputRoot))
        {
            Directory.CreateDirectory(outputRoot);
        }

        if (maps is null || maps.Count == 0)
        {
            return;
        }

        var alphaRoot = GetOption(opts, "alpha-root");
        var convertedAdtRoot = GetOption(opts, "converted-adt-root");
        var convertedAdtCacheRoot = GetOption(opts, "converted-adt-cache");

        string? normalizedAlphaRoot = null;
        if (!string.IsNullOrWhiteSpace(alphaRoot))
        {
            normalizedAlphaRoot = Path.GetFullPath(alphaRoot);
            if (!Directory.Exists(normalizedAlphaRoot))
            {
                throw new DirectoryNotFoundException($"Alpha test data root not found: {normalizedAlphaRoot}");
            }
        }

        string? normalizedConvertedRoot = null;
        if (!string.IsNullOrWhiteSpace(convertedAdtRoot))
        {
            var candidate = Path.GetFullPath(convertedAdtRoot);
            if (Directory.Exists(candidate))
            {
                normalizedConvertedRoot = candidate;
            }
            else
            {
                Console.WriteLine($"[warn] Converted ADT root not found: {candidate}. Coordinates will fall back to raw Alpha data.");
            }
        }

        string? normalizedCacheRoot = null;
        if (!string.IsNullOrWhiteSpace(convertedAdtCacheRoot))
        {
            var candidate = Path.GetFullPath(convertedAdtCacheRoot);
            if (Directory.Exists(candidate))
            {
                normalizedCacheRoot = candidate;
            }
            else
            {
                Console.WriteLine($"[warn] Converted ADT cache not found: {candidate}. Rebuilding may be necessary.");
            }
        }

        foreach (var version in versions)
        {
            foreach (var map in maps)
            {
                if (HasAlphaOutputs(outputRoot, version, map))
                {
                    continue;
                }

                if (normalizedAlphaRoot is null)
                {
                    throw new InvalidOperationException(
                        $"Required comparison data for version '{version}', map '{map}' is missing under '{outputRoot}'. " +
                        "Supply --alpha-root so the CLI can auto-generate the placement ranges.");
                }

                var wdtPath = FindAlphaWdt(normalizedAlphaRoot, version, map);
                if (wdtPath is null)
                {
                    throw new InvalidOperationException(
                        $"Could not locate Alpha WDT for version '{version}', map '{map}' beneath '{normalizedAlphaRoot}'.");
                }

                var buildTag = BuildTagResolver.ResolveForPath(Path.GetDirectoryName(Path.GetFullPath(wdtPath)) ?? wdtPath);
                var sessionDir = OutputSession.Create(outputRoot, map, buildTag);
                var convertedDir = ResolveConvertedAdtDirectory(normalizedConvertedRoot, version, map);
                if (string.IsNullOrWhiteSpace(convertedDir) && normalizedCacheRoot is not null)
                {
                    foreach (var candidate in EnumerateCacheCandidates(normalizedCacheRoot, version, map))
                    {
                        if (Directory.Exists(candidate))
                        {
                            convertedDir = candidate;
                            break;
                        }
                    }
                }

                Console.WriteLine($"[auto] Generating placement ranges for {version} / {map}");
                Console.WriteLine($"[auto]  WDT: {wdtPath}");
                Console.WriteLine($"[auto]  Using raw Alpha coordinates (no transforms)");

                var analysis = WoWRollback.Core.Services.AlphaWdtAnalyzer.AnalyzeAlphaWdt(wdtPath);
                RangeCsvWriter.WritePerMapCsv(sessionDir, $"alpha_{map}", analysis.Ranges, analysis.Assets);
            }
        }
    }

    private static bool HasAlphaOutputs(string outputRoot, string version, string map)
    {
        var versionDirectory = Path.Combine(outputRoot, version);
        if (!Directory.Exists(versionDirectory))
        {
            return false;
        }

        var mapDirectory = Path.Combine(versionDirectory, map);
        if (!Directory.Exists(mapDirectory))
        {
            return false;
        }

        var idRanges = Path.Combine(mapDirectory, $"id_ranges_by_map_alpha_{map}.csv");
        return File.Exists(idRanges);
    }

    private static string? FindAlphaWdt(string alphaRoot, string version, string map)
    {
        var key = (version, map);
        if (AlphaWdtCache.TryGetValue(key, out var cached))
        {
            return cached;
        }

        try
        {
            var matches = Directory.EnumerateFiles(alphaRoot, map + ".wdt", SearchOption.AllDirectories)
                .Where(path => path.EndsWith(Path.DirectorySeparatorChar + map + ".wdt", StringComparison.OrdinalIgnoreCase) ||
                               path.EndsWith(Path.AltDirectorySeparatorChar + map + ".wdt", StringComparison.OrdinalIgnoreCase))
                .Where(path => path.IndexOf($"{Path.DirectorySeparatorChar}World{Path.DirectorySeparatorChar}Maps{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) >= 0 ||
                               path.IndexOf($"{Path.AltDirectorySeparatorChar}World{Path.AltDirectorySeparatorChar}Maps{Path.AltDirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) >= 0)
                .OrderBy(path => ScoreVersionMatch(path, version))
                .ThenBy(path => path.Length)
                .ToList();

            var resolved = matches.FirstOrDefault();
            AlphaWdtCache[key] = resolved;
            return resolved;
        }
        catch
        {
            AlphaWdtCache[key] = null;
            return null;
        }
    }

    private static int ScoreVersionMatch(string path, string version)
    {
        if (path.IndexOf(version, StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return 0;
        }

        var prefixLength = Math.Min(5, version.Length);
        var prefix = version[..prefixLength];
        if (path.IndexOf(prefix, StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return 1;
        }

        var majorMinor = version.Split('.', StringSplitOptions.RemoveEmptyEntries);
        if (majorMinor.Length >= 2)
        {
            var partial = string.Join('.', majorMinor.Take(2));
            if (path.IndexOf(partial, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return 2;
            }
        }

        return 3;
    }

    private static IEnumerable<string> EnumerateCacheCandidates(string root, string version, string map)
    {
        yield return Path.Combine(root, "World", "Maps", map, version);
        yield return Path.Combine(root, "World", "Maps", map);
        yield return Path.Combine(root, version, "World", "Maps", map);
        yield return Path.Combine(root, version, map);
        yield return Path.Combine(root, map);
    }

    private static string? ResolveConvertedAdtDirectory(string? convertedRoot, string version, string map)
    {
        if (string.IsNullOrWhiteSpace(convertedRoot))
        {
            return null;
        }

        foreach (var candidate in EnumerateCacheCandidates(convertedRoot, version, map))
        {
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private sealed class TupleComparer : IEqualityComparer<(string Version, string Map)>
    {
        public bool Equals((string Version, string Map) x, (string Version, string Map) y) =>
            string.Equals(x.Version, y.Version, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(x.Map, y.Map, StringComparison.OrdinalIgnoreCase);

        public int GetHashCode((string Version, string Map) obj) =>
            HashCode.Combine(obj.Version.ToUpperInvariant(), obj.Map.ToUpperInvariant());
    }

    private static string? GetOption(Dictionary<string, string> opts, string key, string? fallback = null) =>
        opts.TryGetValue(key, out var value) ? value : fallback;
}
