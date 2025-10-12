using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using WoWRollback.AnalysisModule;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services;
using WoWRollback.Core.Services.Config;
using WoWRollback.Core.Services.Viewer;

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
                case "analyze-map-adts":
                    return RunAnalyzeMapAdts(opts);
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

    private static int RunAnalyzeMapAdts(Dictionary<string, string> opts)
    {
        Require(opts, "map-dir");
        Require(opts, "map");
        
        var mapDir = opts["map-dir"];
        var mapName = opts["map"];
        var outDir = opts.GetValueOrDefault("out", Path.Combine("analysis_output", mapName));

        Console.WriteLine($"[info] Analyzing ADT files for map: {mapName}");
        Console.WriteLine($"[info] Map directory: {mapDir}");
        Console.WriteLine($"[info] Output directory: {outDir}");

        // Step 1: Extract placements from ADT files
        Console.WriteLine("\n=== Step 1: Extracting placements from ADT files ===");
        var extractor = new AdtPlacementsExtractor();
        var placementsCsvPath = Path.Combine(outDir, $"{mapName}_placements.csv");
        var extractResult = extractor.Extract(mapDir, mapName, placementsCsvPath);

        if (!extractResult.Success)
        {
            Console.Error.WriteLine($"[error] Placement extraction failed: {extractResult.ErrorMessage}");
            return 1;
        }

        Console.WriteLine($"[ok] Extracted {extractResult.M2Count} M2 placements");
        Console.WriteLine($"[ok] Extracted {extractResult.WmoCount} WMO placements");
        Console.WriteLine($"[ok] Processed {extractResult.TilesProcessed} tiles");
        Console.WriteLine($"[ok] Placements CSV: {placementsCsvPath}");

        // Step 1.5: Process minimap tiles
        Console.WriteLine("\n=== Processing minimap tiles ===");
        var minimapHandler = new MinimapHandler();
        var minimapResult = minimapHandler.ProcessMinimaps(mapDir, mapName, outDir);

        if (minimapResult.Success && minimapResult.TilesCopied > 0)
        {
            Console.WriteLine($"[ok] Copied {minimapResult.TilesCopied} minimap tiles");
            Console.WriteLine($"[ok] Minimap directory: {minimapResult.MinimapDir}");
        }
        else if (minimapResult.Success)
        {
            Console.WriteLine($"[info] No minimap PNG files found");
            Console.WriteLine($"[info] Expected location: World\\Textures\\Minimap\\ or World\\Textures\\Minimap\\{mapName}\\");
            Console.WriteLine($"[info] Place PNGs named {mapName}_X_Y.png or mapX_Y.png for minimap support");
        }
        else
        {
            Console.WriteLine($"[warn] Minimap processing failed: {minimapResult.ErrorMessage}");
        }

        // Step 1.75: Extract terrain data (MCNK chunks) for terrain overlays
        Console.WriteLine("\n=== Extracting terrain data (MCNK chunks) ===");
        var terrainExtractor = new AdtTerrainExtractor();
        var terrainResult = terrainExtractor.ExtractTerrainForMap(mapDir, mapName, outDir);
        
        if (!terrainResult.Success)
        {
            Console.WriteLine($"[warn] Terrain extraction failed - terrain overlays will not be available");
        }
        else
        {
            Console.WriteLine($"[ok] Extracted {terrainResult.ChunksExtracted} MCNK chunks from {terrainResult.TilesProcessed} tiles");
            Console.WriteLine($"[ok] Terrain CSV: {terrainResult.CsvPath}");
        }

        // Step 2: Analyze UniqueIDs and detect layers
        Console.WriteLine("\n=== Step 2: Analyzing UniqueIDs and detecting layers ===");
        var analyzer = new UniqueIdAnalyzer(gapThreshold: 100);
        var analysisResult = analyzer.AnalyzeFromPlacementsCsv(placementsCsvPath, mapName, outDir);

        if (!analysisResult.Success)
        {
            Console.Error.WriteLine($"[error] UniqueID analysis failed: {analysisResult.ErrorMessage}");
            return 1;
        }

        Console.WriteLine($"[ok] Analyzed {analysisResult.TileCount} tiles");
        Console.WriteLine($"[ok] UniqueID analysis CSV: {analysisResult.CsvPath}");
        Console.WriteLine($"[ok] Layers JSON: {analysisResult.LayersJsonPath}");

        // Step 3: Detect spatial clusters and patterns (prefabs/brushes)
        Console.WriteLine("\n=== Step 3: Detecting spatial clusters and patterns ===");
        var clusterAnalyzer = new ClusterAnalyzer(proximityThreshold: 50.0f, minClusterSize: 3);
        var clusterResult = clusterAnalyzer.Analyze(placementsCsvPath, mapName, outDir);

        if (!clusterResult.Success)
        {
            Console.WriteLine($"[warn] Cluster analysis failed: {clusterResult.ErrorMessage}");
        }
        else
        {
            Console.WriteLine($"[ok] Detected {clusterResult.TotalClusters} spatial clusters");
            Console.WriteLine($"[ok] Identified {clusterResult.TotalPatterns} recurring patterns (potential prefabs)");
            Console.WriteLine($"[ok] Clusters JSON: {clusterResult.ClustersJsonPath}");
            Console.WriteLine($"[ok] Patterns JSON: {clusterResult.PatternsJsonPath}");
            Console.WriteLine($"[ok] Summary CSV: {clusterResult.SummaryCsvPath}");
        }

        // Step 4: Generate viewer using existing infrastructure
        Console.WriteLine("\n=== Step 4: Generating viewer ===");
        var viewerAdapter = new AnalysisViewerAdapter();
        var viewerRoot = viewerAdapter.GenerateViewer(placementsCsvPath, mapName, outDir, minimapResult.MinimapDir);

        if (!string.IsNullOrEmpty(viewerRoot))
        {
            Console.WriteLine($"[ok] Viewer generated: {viewerRoot}");
            Console.WriteLine($"[info] Open: {Path.Combine(viewerRoot, "index.html")}");
        }
        else
        {
            Console.WriteLine($"[warn] Viewer generation skipped (no placements)");
        }

        Console.WriteLine("\n=== Analysis Complete ===");
        Console.WriteLine($"All outputs written to: {outDir}");
        Console.WriteLine("\nℹ️  Spatial clusters reveal object groups placed together - likely prefabs or brushes");
        Console.WriteLine("ℹ️  Recurring patterns show reused object compositions across the map");
        Console.WriteLine("ℹ️  Open the viewer in a web browser to explore your map interactively");
        
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
        Console.WriteLine("  analyze-map-adts  --map <name> --map-dir <dir> [--out <dir>]");
        Console.WriteLine("    Analyze ADT files (pre-Cata or Cata+ split) and extract UniqueID data");
        Console.WriteLine("    Supports 0.6.0 through 4.0.0+ ADT formats");
        Console.WriteLine();
        Console.WriteLine("  dry-run           --map <name> --input-dir <dir> [--config <file>] [--keep-range min:max] [--drop-range min:max] [--mode keep|drop]");
        Console.WriteLine("    Preview rollback effects without modifying files");
        Console.WriteLine();
        Console.WriteLine("  compare-versions  --versions v1,v2[,v3...] [--maps m1,m2,...] [--root <dir>] [--yaml-report]");
        Console.WriteLine("                    [--viewer-report] [--default-version <ver>] [--diff base,comp]");
        Console.WriteLine("                    [--mpq-path <dir>]");
        Console.WriteLine("    Compare placement ranges across versions; outputs CSVs under rollback_outputs/comparisons/<key>");
        Console.WriteLine("    If --yaml-report is present, also writes YAML exploration reports under .../<key>/yaml/.");
        Console.WriteLine("    --viewer-report additionally emits minimaps, overlays, and diffs for the static viewer.");
        Console.WriteLine("    --mpq-path specifies MPQ archive directory for minimap extraction (default: loose files)");
        Console.WriteLine();
        Console.WriteLine("Archaeological Perspective:");
        Console.WriteLine("  Each UniqueID range represents a 'volume of work' by ancient developers.");
        Console.WriteLine("  Singleton IDs and outliers are precious artifacts showing experiments and tests.");
        Console.WriteLine("  We're uncovering sedimentary layers of 20+ years of WoW development history.");
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
