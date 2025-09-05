using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Next.Core.IO;

namespace GillijimProject.Next.Cli.Commands;

public static class GenAlphaWdtCommand
{
    public static int Run(string[] args)
    {
        // Options:
        //   --map <MapName>
        //   --in <LK ADT dir>
        //   --out <Alpha .wdt path>
        // Flags:
        //   --no-empty-mdnm (default MDNM included)
        //   --include-empty-monm (default false)
        //   --wmo-based (default false)

        var opts = ParseOptions(args);

        if (!opts.TryGetValue("--map", out var mapName) || string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("[gen-alpha-wdt] Missing --map <MapName>.");
            return 2;
        }
        if (!opts.TryGetValue("--in", out var inDir) || string.IsNullOrWhiteSpace(inDir) || !Directory.Exists(inDir))
        {
            Console.Error.WriteLine("[gen-alpha-wdt] Missing or invalid --in <dir>.");
            return 2;
        }
        if (!opts.TryGetValue("--out", out var outWdt) || string.IsNullOrWhiteSpace(outWdt))
        {
            Console.Error.WriteLine("[gen-alpha-wdt] Missing --out <Alpha .wdt path>.");
            return 2;
        }

        bool includeEmptyMdnm = !opts.ContainsKey("--no-empty-mdnm");
        bool includeEmptyMonm = opts.ContainsKey("--include-empty-monm");
        bool wmoBased = opts.ContainsKey("--wmo-based");

        try
        {
            var result = AlphaWdtWriter.GenerateFromLkAdts(
                mapName,
                inDir,
                outWdt,
                new AlphaWdtWriter.Options(
                    IncludeEmptyMdnm: includeEmptyMdnm,
                    IncludeEmptyMonm: includeEmptyMonm,
                    WmoBased: wmoBased
                )
            );

            Console.WriteLine($"[gen-alpha-wdt] Wrote: {result.OutWdtPath}");
            Console.WriteLine($"[gen-alpha-wdt] tiles_total={result.TilesTotal} embedded={result.TilesEmbedded} missing={result.TilesMissing}");
            if (result.Warnings.Count > 0)
            {
                Console.WriteLine($"[gen-alpha-wdt] warnings={result.Warnings.Count}");
                foreach (var w in result.Warnings) Console.WriteLine($"  - {w}");
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[gen-alpha-wdt] Failed: {ex.Message}");
            return 3;
        }
    }

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (key.StartsWith("--", StringComparison.Ordinal))
            {
                if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    dict[key] = args[++i];
                }
                else
                {
                    // treat as flag presence
                    dict[key] = string.Empty;
                }
            }
        }
        return dict;
    }
}
