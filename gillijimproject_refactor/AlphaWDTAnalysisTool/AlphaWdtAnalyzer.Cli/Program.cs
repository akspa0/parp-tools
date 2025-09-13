using System;
using System.IO;
using System.Linq;
using AlphaWdtAnalyzer.Core;
using AlphaWdtAnalyzer.Core.Export;

namespace AlphaWdtAnalyzer.Cli;

public static class Program
{
    private static int Usage()
    {
        Console.WriteLine("AlphaWdtAnalyzer");
        Console.WriteLine("Usage:");
        Console.WriteLine("  Single map: AlphaWdtAnalyzer --input <path/to/map.wdt> --listfile <community_listfile.csv> [--lk-listfile <3x.txt>] --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--web] [--export-adt --export-dir <dir> [--fallback-tileset <blp>] [--fallback-wmo <wmo>] [--fallback-m2 <m2>] [--fallback-blp <blp>] [--no-mh2o] [--asset-fuzzy on|off] [--profile preserve|modified] [--no-fallbacks] [--no-fixups] [--remap <remap.json>] [--verbose] [--track-assets]]");
        Console.WriteLine("  Batch maps:  AlphaWdtAnalyzer --input-dir <root_of_wdts> --listfile <community_listfile.csv> [--lk-listfile <3x.txt>] --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--web] [--export-adt --export-dir <dir> [--fallback-tileset <blp>] [--fallback-wmo <wmo>] [--fallback-m2 <m2>] [--fallback-blp <blp>] [--no-mh2o] [--asset-fuzzy on|off] [--profile preserve|modified] [--no-fallbacks] [--no-fixups] [--remap <remap.json>] [--verbose] [--track-assets]]");
        Console.WriteLine("  DBCD mapping (no remap.json): add --dbd-dir <dir> --dbc-src <src DBFilesClient> --dbc-tgt <tgt DBFilesClient> [--src-alias 0.5.3|0.5.5|0.6.0] [--src-build <build>] [--tgt-build 3.3.5.12340] [--allow-do-not-use]");
        Console.WriteLine("  Lazy mode: AlphaWdtAnalyzer --root <test_data> --out <out_dir> --export-adt --export-dir <out_dir> [--src-alias 0.5.3|0.5.5|0.6.0] [--verbose]");
        return 2;
    }

    public static int Main(string[] args)
    {
        string? wdt = null;
        string? inputDir = null;
        string? listfile = null;
        string? lkListfile = null;
        string? outDir = null;
        bool web = false; // default off
        int? clusterThreshold = null;
        int? clusterGap = null;
        // export flags
        bool exportAdt = false;
        string? exportDir = null;
        string fallbackTileset = @"Tileset\Generic\Checkers.blp";
        string fallbackWmo = @"wmo\Dungeon\test\missingwmo.wmo";
        string fallbackM2 = @"World\Scale\HumanMaleScale.mdx";
        string fallbackNonTilesetBlp = @"Dungeons\Textures\temp\64.blp";
        bool mh2o = true;
        bool assetFuzzy = true;
        // profiles/toggles
        string profile = "modified"; // modified|preserve
        bool useFallbacks = true;
        bool enableFixups = true;
        string? remap = null;
        bool verbose = false;
        bool trackAssets = false;
        string? dbdDir = null;
        string? dbcSrc = null;
        string? dbcTgt = null;
        string? srcAlias = null;
        string? srcBuild = null;
        string? tgtBuild = null;
        bool allowDoNotUse = false;
        string? rootDir = null;
        int sampleCount = 8; // limits per-tile sample rows in CSV
        bool diagnosticsOnly = false; // skip ADT writing/patching for speed

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            switch (a)
            {
                case "--root":
                    if (i + 1 >= args.Length) return Usage();
                    rootDir = args[++i];
                    break;
                case "--input":
                    if (i + 1 >= args.Length) return Usage();
                    wdt = args[++i];
                    break;
                case "--input-dir":
                    if (i + 1 >= args.Length) return Usage();
                    inputDir = args[++i];
                    break;
                case "--listfile":
                    if (i + 1 >= args.Length) return Usage();
                    listfile = args[++i];
                    break;
                case "--lk-listfile":
                    if (i + 1 >= args.Length) return Usage();
                    lkListfile = args[++i];
                    break;
                case "--out":
                    if (i + 1 >= args.Length) return Usage();
                    outDir = args[++i];
                    break;
                case "--cluster-threshold":
                    if (i + 1 >= args.Length) return Usage();
                    if (!int.TryParse(args[++i], out var ct)) return Usage();
                    clusterThreshold = ct;
                    break;
                case "--cluster-gap":
                    if (i + 1 >= args.Length) return Usage();
                    if (!int.TryParse(args[++i], out var cg)) return Usage();
                    clusterGap = cg;
                    break;
                case "--web":
                    web = true;
                    break;
                case "--no-web":
                    web = false;
                    break;
                case "--export-adt":
                    exportAdt = true;
                    break;
                case "--export-dir":
                    if (i + 1 >= args.Length) return Usage();
                    exportDir = args[++i];
                    break;
                case "--fallback-tileset":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackTileset = args[++i];
                    break;
                case "--fallback-wmo":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackWmo = args[++i];
                    break;
                case "--fallback-m2":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackM2 = args[++i];
                    break;
                case "--fallback-blp":
                    if (i + 1 >= args.Length) return Usage();
                    fallbackNonTilesetBlp = args[++i];
                    break;
                case "--no-mh2o":
                    mh2o = false;
                    break;
                case "--asset-fuzzy":
                    if (i + 1 >= args.Length) return Usage();
                    var v = args[++i];
                    assetFuzzy = !string.Equals(v, "off", StringComparison.OrdinalIgnoreCase);
                    break;
                case "--profile":
                    if (i + 1 >= args.Length) return Usage();
                    profile = args[++i];
                    break;
                case "--no-fallbacks":
                    useFallbacks = false;
                    break;
                case "--no-fixups":
                    enableFixups = false;
                    break;
                case "--remap":
                    if (i + 1 >= args.Length) return Usage();
                    remap = args[++i];
                    break;
                case "--dbd-dir":
                    if (i + 1 >= args.Length) return Usage();
                    dbdDir = args[++i];
                    break;
                case "--dbc-src":
                    if (i + 1 >= args.Length) return Usage();
                    dbcSrc = args[++i];
                    break;
                case "--dbc-tgt":
                    if (i + 1 >= args.Length) return Usage();
                    dbcTgt = args[++i];
                    break;
                case "--src-alias":
                    if (i + 1 >= args.Length) return Usage();
                    srcAlias = args[++i];
                    break;
                case "--src-build":
                    if (i + 1 >= args.Length) return Usage();
                    srcBuild = args[++i];
                    break;
                case "--tgt-build":
                    if (i + 1 >= args.Length) return Usage();
                    tgtBuild = args[++i];
                    break;
                case "--allow-do-not-use":
                    allowDoNotUse = true;
                    break;
                case "--verbose":
                    verbose = true;
                    break;
                case "--track-assets":
                    trackAssets = true;
                    break;
                case "--sample-count":
                    if (i + 1 >= args.Length) return Usage();
                    if (!int.TryParse(args[++i], out sampleCount)) return Usage();
                    if (sampleCount < 0) sampleCount = 0;
                    break;
                case "--diagnostics-only":
                    diagnosticsOnly = true;
                    break;
                case "-h":
                case "--help":
                    return Usage();
            }
        }

        // --- Simplified defaults & auto-detection ---
        // Find repo root that contains test_data to resolve defaults
        static string? FindRepoRootWithTestData()
        {
            // Try from BaseDirectory upwards
            string? dir = AppContext.BaseDirectory;
            for (int i = 0; i < 8 && !string.IsNullOrEmpty(dir); i++)
            {
                if (Directory.Exists(Path.Combine(dir, "test_data"))) return dir;
                dir = Directory.GetParent(dir)?.FullName;
            }
            // Try from current working directory
            dir = Directory.GetCurrentDirectory();
            for (int i = 0; i < 8 && !string.IsNullOrEmpty(dir); i++)
            {
                if (Directory.Exists(Path.Combine(dir, "test_data"))) return dir;
                dir = Directory.GetParent(dir)?.FullName;
            }
            return null;
        }

        static string InferAliasFromAnyPath(string p)
        {
            var s = (p ?? string.Empty).Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar).ToLowerInvariant();
            if (s.Contains("0.5.3")) return "0.5.3";
            if (s.Contains("0.5.5")) return "0.5.5";
            if (s.Contains("0.6.0")) return "0.6.0";
            return "0.5.3"; // sensible default for alpha
        }

        // Default listfiles from test_data root if not provided
        var repoRoot = FindRepoRootWithTestData();
        var testDataRoot = repoRoot is null ? null : Path.Combine(repoRoot, "test_data");
        if (string.IsNullOrWhiteSpace(listfile) || !File.Exists(listfile))
        {
            var cand = testDataRoot is null ? null : Path.Combine(testDataRoot, "community-listfile-withcapitals.csv");
            if (!string.IsNullOrWhiteSpace(cand) && File.Exists(cand)) listfile = cand;
        }
        if (string.IsNullOrWhiteSpace(lkListfile) || !File.Exists(lkListfile))
        {
            var cand = testDataRoot is null ? null : Path.Combine(testDataRoot, "World of Warcraft 3x.txt");
            if (!string.IsNullOrWhiteSpace(cand) && File.Exists(cand)) lkListfile = cand;
        }

        // If exporting ADTs, default export-dir to --out (single output folder)
        if (exportAdt && string.IsNullOrWhiteSpace(exportDir))
        {
            exportDir = outDir;
        }

        // Infer source alias/build from input path when not specified
        string anyInputPath = !string.IsNullOrWhiteSpace(wdt) ? wdt! : (inputDir ?? string.Empty);
        if (string.IsNullOrWhiteSpace(srcAlias) && !string.IsNullOrWhiteSpace(anyInputPath)) srcAlias = InferAliasFromAnyPath(anyInputPath);
        if (string.IsNullOrWhiteSpace(srcBuild) && !string.IsNullOrWhiteSpace(srcAlias))
        {
            srcBuild = srcAlias switch { "0.5.3" => "0.5.3.3368", "0.5.5" => "0.5.5.3494", "0.6.0" => "0.6.0.3592", _ => srcBuild };
        }
        // Default target build to LK 3.3.5
        tgtBuild ??= "3.3.5.12340";

        // Auto-guess DBCD directories if available in repo
        if (string.IsNullOrWhiteSpace(dbdDir) && repoRoot is not null)
        {
            var cand = Path.Combine(repoRoot, "lib", "WoWDBDefs", "definitions");
            if (Directory.Exists(cand)) dbdDir = cand;
        }
        if (string.IsNullOrWhiteSpace(dbcSrc) && !string.IsNullOrWhiteSpace(srcAlias) && testDataRoot is not null)
        {
            var cand = Path.Combine(testDataRoot, srcAlias, "tree", "DBFilesClient");
            if (Directory.Exists(cand)) dbcSrc = cand;
        }
        if (string.IsNullOrWhiteSpace(dbcTgt) && testDataRoot is not null)
        {
            var cand = Path.Combine(testDataRoot, "3.3.5", "tree", "DBFilesClient");
            if (Directory.Exists(cand)) dbcTgt = cand;
        }

        // Apply profile
        if (string.Equals(profile, "preserve", StringComparison.OrdinalIgnoreCase))
        {
            assetFuzzy = false;
            useFallbacks = false;
            enableFixups = false;
        }

        // Allow directory passed via --input
        if (!string.IsNullOrWhiteSpace(wdt) && Directory.Exists(wdt))
        {
            inputDir = wdt;
            wdt = null;
        }

        var isRootMode = !string.IsNullOrWhiteSpace(rootDir);
        var isBatch = isRootMode || !string.IsNullOrWhiteSpace(inputDir);

        // Validation: in root mode, only require --out; in other modes require input + out; listfiles default from test_data if omitted
        if ((isBatch && !isRootMode && string.IsNullOrWhiteSpace(outDir)) ||
            (!isBatch && (string.IsNullOrWhiteSpace(wdt) || string.IsNullOrWhiteSpace(outDir))) ||
            (isRootMode && string.IsNullOrWhiteSpace(outDir)))
        {
            return Usage();
        }

        if (!string.IsNullOrWhiteSpace(listfile) && !File.Exists(listfile))
        {
            Console.Error.WriteLine($"Listfile not found: {listfile}");
            return 1;
        }
        if (!string.IsNullOrWhiteSpace(lkListfile) && !File.Exists(lkListfile))
        {
            Console.Error.WriteLine($"LK listfile not found: {lkListfile}");
            return 1;
        }

        try
        {
            // Lazy auto-discovery from --root
            if (isRootMode)
            {
                var auto = AutoDiscoverFromRoot(rootDir!, srcAlias);
                // Fill DBCD fields if not explicitly provided
                dbdDir ??= Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "lib", "WoWDBDefs", "definitions");
                if (!Directory.Exists(dbdDir)) dbdDir = Path.Combine("lib", "WoWDBDefs", "definitions");
                dbcSrc ??= auto.DbcSrcDir;
                dbcTgt ??= auto.DbcTgtDir;
                if (string.IsNullOrWhiteSpace(srcAlias)) srcAlias = auto.SrcAlias; // honor preferred alias if provided
                else if (!string.Equals(srcAlias, auto.SrcAlias, StringComparison.OrdinalIgnoreCase))
                    Console.WriteLine($"[Auto] Requested src-alias '{srcAlias}' not found; using '{auto.SrcAlias}'");
                // WDT search root defaults to the overall root; ExportBatch will scan recursively
                inputDir ??= rootDir;
                // Turn on ADT export by default if not set
                if (exportAdt && string.IsNullOrWhiteSpace(exportDir)) exportDir = outDir;
                if (!exportAdt)
                {
                    exportAdt = true;
                    exportDir ??= outDir;
                }
            }

            if (isBatch)
            {
                if (!Directory.Exists(inputDir!))
                {
                    Console.Error.WriteLine($"Input directory not found: {inputDir}");
                    return 1;
                }

                BatchAnalysis.Run(new BatchAnalysis.Options
                {
                    InputRoot = inputDir!,
                    ListfilePath = listfile!,
                    OutDir = outDir!,
                    ClusterThreshold = clusterThreshold ?? 10,
                    ClusterGap = clusterGap ?? 1000,
                    Web = web
                });

                if (exportAdt)
                {
                    AdtExportPipeline.ExportBatch(new AdtExportPipeline.Options
                    {
                        InputRoot = inputDir!,
                        CommunityListfilePath = listfile,
                        LkListfilePath = lkListfile,
                        ExportDir = exportDir!,
                        FallbackTileset = fallbackTileset,
                        FallbackNonTilesetBlp = fallbackNonTilesetBlp,
                        FallbackWmo = fallbackWmo,
                        FallbackM2 = fallbackM2,
                        ConvertToMh2o = mh2o,
                        AssetFuzzy = assetFuzzy,
                        UseFallbacks = useFallbacks,
                        EnableFixups = enableFixups,
                        RemapPath = remap,
                        Verbose = verbose,
                        TrackAssets = trackAssets,
                        DbdDir = dbdDir,
                        DbcSrcDir = dbcSrc,
                        DbcTgtDir = dbcTgt,
                        SrcAlias = srcAlias,
                        SrcBuild = srcBuild,
                        TgtBuild = tgtBuild,
                        AllowDoNotUse = allowDoNotUse,
                        SampleCount = sampleCount,
                        DiagnosticsOnly = diagnosticsOnly,
                    });
                }
            }
            else
            {
                if (!File.Exists(wdt!))
                {
                    Console.Error.WriteLine($"WDT not found: {wdt}");
                    return 1;
                }

                AnalysisPipeline.Run(new AnalysisPipeline.Options
                {
                    WdtPath = wdt!,
                    ListfilePath = listfile!,
                    OutDir = outDir!,
                    ClusterThreshold = clusterThreshold ?? 10,
                    ClusterGap = clusterGap ?? 1000,
                });

                if (web)
                {
                    WebAssetsWriter.Write(outDir!);
                    Console.WriteLine($"Web UI written to {Path.Combine(outDir!, "web")}. Open index.html in a browser.");
                }

                if (exportAdt)
                {
                    AdtExportPipeline.ExportSingle(new AdtExportPipeline.Options
                    {
                        SingleWdtPath = wdt!,
                        CommunityListfilePath = listfile!,
                        LkListfilePath = lkListfile,
                        ExportDir = exportDir!,
                        FallbackTileset = fallbackTileset,
                        FallbackNonTilesetBlp = fallbackNonTilesetBlp,
                        FallbackWmo = fallbackWmo,
                        FallbackM2 = fallbackM2,
                        ConvertToMh2o = mh2o,
                        AssetFuzzy = assetFuzzy,
                        UseFallbacks = useFallbacks,
                        EnableFixups = enableFixups,
                        RemapPath = remap,
                        Verbose = verbose,
                        TrackAssets = trackAssets,
                        DbdDir = dbdDir,
                        DbcSrcDir = dbcSrc,
                        DbcTgtDir = dbcTgt,
                        SrcAlias = srcAlias,
                        SrcBuild = srcBuild,
                        TgtBuild = tgtBuild,
                        AllowDoNotUse = allowDoNotUse,
                        SampleCount = sampleCount,
                        DiagnosticsOnly = diagnosticsOnly,
                    });
                }
            }

            Console.WriteLine("Analysis complete.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static (string SrcAlias, string DbcSrcDir, string DbcTgtDir) AutoDiscoverFromRoot(string root, string? preferredAlias)
    {
        // Find DBFilesClient directories for classic and 3.3.5
        string FindDbcDir(string token)
        {
            try
            {
                var dirs = Directory.EnumerateDirectories(root, "DBFilesClient", SearchOption.AllDirectories)
                    .Where(d => d.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0)
                    .OrderBy(d => d.Length)
                    .ToList();
                return dirs.FirstOrDefault() ?? string.Empty;
            }
            catch { return string.Empty; }
        }

        string srcAlias = "0.5.3";
        string srcDir = string.Empty;
        // If user requested a preferred alias, try it first exclusively
        if (!string.IsNullOrWhiteSpace(preferredAlias))
        {
            var pref = preferredAlias.Trim();
            srcAlias = pref;
            srcDir = FindDbcDir(pref);
        }
        // Fallback to auto-discovery order
        if (string.IsNullOrEmpty(srcDir)) { srcDir = FindDbcDir("0.5.3"); srcAlias = string.IsNullOrEmpty(srcDir) ? srcAlias : "0.5.3"; }
        if (string.IsNullOrEmpty(srcDir)) { srcDir = FindDbcDir("0.5.5"); srcAlias = string.IsNullOrEmpty(srcDir) ? srcAlias : "0.5.5"; }
        if (string.IsNullOrEmpty(srcDir)) { srcDir = FindDbcDir("0.6.0"); srcAlias = string.IsNullOrEmpty(srcDir) ? srcAlias : "0.6.0"; }
        if (string.IsNullOrEmpty(srcDir))
        {
            // fallback: first DBFilesClient under root
            try { srcDir = Directory.EnumerateDirectories(root, "DBFilesClient", SearchOption.AllDirectories).FirstOrDefault() ?? string.Empty; } catch { }
        }

        string tgtDir = FindDbcDir("3.3.5");
        if (string.IsNullOrEmpty(tgtDir))
        {
            // fallback: any DBFilesClient under root containing 3.3.5 token elsewhere
            try {
                tgtDir = Directory.EnumerateDirectories(root, "DBFilesClient", SearchOption.AllDirectories)
                    .FirstOrDefault(d => d.IndexOf("3.3.5", StringComparison.OrdinalIgnoreCase) >= 0) ?? string.Empty;
            } catch { }
        }

        return (srcAlias, srcDir, tgtDir);
    }
}
