using System.Globalization;
using System.Linq;
using WoWRollback.Core.Models;
using WoWRollback.Core.Services;
using WoWRollback.Core.Services.Config;

namespace WoWRollback.Cli;

internal static class Program
{
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

    private static int RunCompareVersions(Dictionary<string, string> opts)
    {
        Require(opts, "versions");
        var root = opts.GetValueOrDefault("root", opts.GetValueOrDefault("out", ""));

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

        Console.WriteLine($"[info] Comparing versions: {string.Join(", ", versions)}");
        if (!string.IsNullOrWhiteSpace(root)) Console.WriteLine($"[info] Root directory: {root}");
        if (maps is not null && maps.Count > 0) Console.WriteLine($"[info] Map filter: {string.Join(", ", maps)}");

        var result = VersionComparisonService.CompareVersions(root, versions, maps);
        var paths = VersionComparisonWriter.WriteOutputs(root, result);

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
        Console.WriteLine("  dry-run           --map <name> --input-dir <dir> [--config <file>] [--keep-range min:max] [--drop-range min:max] [--mode keep|drop]");
        Console.WriteLine("    Preview rollback effects without modifying files");
        Console.WriteLine();
        Console.WriteLine("  compare-versions  --versions v1,v2[,v3...] [--maps m1,m2,...] [--root <dir>]");
        Console.WriteLine("    Compare placement ranges across versions; outputs CSVs under rollback_outputs/comparisons/<key>");
        Console.WriteLine();
        Console.WriteLine("Archaeological Perspective:");
        Console.WriteLine("  Each UniqueID range represents a 'volume of work' by ancient developers.");
        Console.WriteLine("  Singleton IDs and outliers are precious artifacts showing experiments and tests.");
        Console.WriteLine("  We're uncovering sedimentary layers of 20+ years of WoW development history.");
    }
}
