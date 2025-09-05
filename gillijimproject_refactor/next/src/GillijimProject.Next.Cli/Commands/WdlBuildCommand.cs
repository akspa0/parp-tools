using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Next.Core.IO;

namespace GillijimProject.Next.Cli.Commands;

public static class WdlBuildCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);
        if (!opts.TryGetValue("--in", out var inPath) || string.IsNullOrWhiteSpace(inPath) || !File.Exists(inPath))
        {
            Console.Error.WriteLine("[wdl-build] Missing or invalid --in <path-to.wdl>.");
            Console.Error.WriteLine("Usage: wdl-build --in <path-to.wdl> --out <path-to-out.wdl> [--no-mver]");
            return 2;
        }
        if (!opts.TryGetValue("--out", out var outPath) || string.IsNullOrWhiteSpace(outPath))
        {
            Console.Error.WriteLine("[wdl-build] Missing --out <path-to-out.wdl>.");
            return 2;
        }

        bool includeMver = !opts.ContainsKey("--no-mver");

        try
        {
            var model = AlphaReader.ParseWdl(inPath);
            WdlWriter.Write(model, outPath, includeMver: includeMver);
            Console.WriteLine($"[wdl-build] Wrote: {Path.GetFullPath(outPath)}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[wdl-build] Failed: {ex.Message}");
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
                    dict[key] = string.Empty; // flag
                }
            }
        }
        return dict;
    }
}
