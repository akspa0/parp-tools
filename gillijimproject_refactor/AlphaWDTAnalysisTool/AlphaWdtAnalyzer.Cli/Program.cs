using System;
using System.IO;
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

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            switch (a)
            {
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
                case "--verbose":
                    verbose = true;
                    break;
                case "--track-assets":
                    trackAssets = true;
                    break;
                case "-h":
                case "--help":
                    return Usage();
            }
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

        var isBatch = !string.IsNullOrWhiteSpace(inputDir);

        if ((isBatch && (string.IsNullOrWhiteSpace(listfile) || string.IsNullOrWhiteSpace(outDir))) ||
            (!isBatch && (string.IsNullOrWhiteSpace(wdt) || string.IsNullOrWhiteSpace(listfile) || string.IsNullOrWhiteSpace(outDir))))
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
        if (exportAdt)
        {
            if (string.IsNullOrWhiteSpace(exportDir))
            {
                Console.Error.WriteLine("--export-adt requires --export-dir <dir>");
                return 1;
            }
        }

        try
        {
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
}
