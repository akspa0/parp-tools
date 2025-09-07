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
        Console.WriteLine("  AlphaWdtAnalyzer --input <path/to/map.wdt> --listfile <path/to/listfile.csv> --out <output_dir> [--cluster-threshold N] [--cluster-gap N] [--dbc-dir <dir>] [--area-alpha <AreaTable.dbc>] [--area-lk <AreaTable.dbc>] [--no-web]");
        return 2;
    }

    public static int Main(string[] args)
    {
        string? wdt = null;
        string? listfile = null;
        string? outDir = null;
        bool noWeb = false;
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
                case "--no-web":
                    noWeb = true;
                    break;
                case "-h":
                case "--help":
                    return Usage();
            }
        }

        if (string.IsNullOrWhiteSpace(wdt) || string.IsNullOrWhiteSpace(listfile) || string.IsNullOrWhiteSpace(outDir))
        {
            return Usage();
        }

        if (!File.Exists(wdt))
        {
            Console.Error.WriteLine($"WDT not found: {wdt}");
            return 1;
        }
        if (!File.Exists(listfile))
        {
            Console.Error.WriteLine($"Listfile not found: {listfile}");
            return 1;
        }
        if (!string.IsNullOrWhiteSpace(areaAlpha) && !File.Exists(areaAlpha))
        {
            Console.Error.WriteLine($"AreaTable alpha not found: {areaAlpha}");
            return 1;
        }
        if (!string.IsNullOrWhiteSpace(areaLk) && !File.Exists(areaLk))
        {
            Console.Error.WriteLine($"AreaTable LK not found: {areaLk}");
            return 1;
        }

        try
        {
            AnalysisPipeline.Run(new AnalysisPipeline.Options
            {
                WdtPath = wdt,
                ListfilePath = listfile,
                OutDir = outDir!,
                ClusterThreshold = clusterThreshold ?? 10,
                ClusterGap = clusterGap ?? 1000,
                DbcDir = dbcDir,
                AreaAlphaPath = areaAlpha,
                AreaLkPath = areaLk,
            });

            if (!noWeb)
            {
                WebAssetsWriter.Write(outDir!);
                Console.WriteLine($"Web UI written to {Path.Combine(outDir!, "web")}. Open index.html in a browser.");
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
