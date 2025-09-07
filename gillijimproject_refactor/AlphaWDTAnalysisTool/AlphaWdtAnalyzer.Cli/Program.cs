using System;
using System.IO;
using AlphaWdtAnalyzer.Core;

namespace AlphaWdtAnalyzer.Cli;

public static class Program
{
    private static int Usage()
    {
        Console.WriteLine("AlphaWdtAnalyzer");
        Console.WriteLine("Usage:");
        Console.WriteLine("  Single map: AlphaWdtAnalyzer --input <path/to/map.wdt> --listfile <path/to/listfile.csv> --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--dbc-dir <dir>] [--area-alpha <AreaTable.dbc>] [--area-lk <AreaTable.dbc>] [--web]");
        Console.WriteLine("  Batch maps:  AlphaWdtAnalyzer --input-dir <root_of_wdts> --listfile <path/to/listfile.csv> --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--dbc-dir <dir>] [--web]");
        return 2;
    }

    public static int Main(string[] args)
    {
        string? wdt = null;
        string? inputDir = null;
        string? listfile = null;
        string? outDir = null;
        bool web = false; // default off
        int? clusterThreshold = null;
        int? clusterGap = null;
        string? dbcDir = null;
        string? areaAlpha = null;
        string? areaLk = null;

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
                case "--dbc-dir":
                    if (i + 1 >= args.Length) return Usage();
                    dbcDir = args[++i];
                    break;
                case "--area-alpha":
                    if (i + 1 >= args.Length) return Usage();
                    areaAlpha = args[++i];
                    break;
                case "--area-lk":
                    if (i + 1 >= args.Length) return Usage();
                    areaLk = args[++i];
                    break;
                case "--web":
                    web = true;
                    break;
                case "--no-web":
                    web = false;
                    break;
                case "-h":
                case "--help":
                    return Usage();
            }
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
        if (!string.IsNullOrWhiteSpace(dbcDir) && !Directory.Exists(dbcDir))
        {
            Console.Error.WriteLine($"DBC dir not found: {dbcDir}");
            return 1;
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
                    DbcDir = dbcDir,
                    Web = web
                });
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
                    DbcDir = dbcDir,
                    AreaAlphaPath = areaAlpha,
                    AreaLkPath = areaLk,
                });

                if (web)
                {
                    WebAssetsWriter.Write(outDir!);
                    Console.WriteLine($"Web UI written to {Path.Combine(outDir!, "web")}. Open index.html in a browser.");
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
