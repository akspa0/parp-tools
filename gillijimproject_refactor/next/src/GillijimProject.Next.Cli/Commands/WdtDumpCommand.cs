using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Next.Core.WowFiles.Alpha;

namespace GillijimProject.Next.Cli.Commands;

public static class WdtDumpCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);
        if (!opts.TryGetValue("--in", out var inPath) || string.IsNullOrWhiteSpace(inPath) || !File.Exists(inPath))
        {
            Console.Error.WriteLine("[wdt-dump] Missing or invalid --in <path-to.wdt>.");
            Console.Error.WriteLine("Usage: wdt-dump --in <path-to.wdt>");
            return 2;
        }

        try
        {
            var wdt = AlphaWdtReader.Read(inPath);
            int present = 0;
            for (int i = 0; i < wdt.AdtOffsets.Count; i++) if (wdt.AdtOffsets[i] != 0) present++;
            Console.WriteLine($"[wdt-dump] file={Path.GetFileName(inPath)} wmoBased={wdt.WmoBased} present_tiles={present}");
            Console.WriteLine($"[wdt-dump] mdnm={wdt.MdnmFiles.Count} monm={wdt.MonmFiles.Count}");
            if (wdt.MdnmFiles.Count > 0)
                Console.WriteLine($"[wdt-dump] MDNM sample: {string.Join(" | ", Take(wdt.MdnmFiles, 3))}");
            if (wdt.MonmFiles.Count > 0)
                Console.WriteLine($"[wdt-dump] MONM sample: {string.Join(" | ", Take(wdt.MonmFiles, 3))}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[wdt-dump] Failed: {ex.Message}");
            return 3;
        }
    }

    private static IEnumerable<string> Take(IReadOnlyList<string> list, int n)
    {
        int count = Math.Min(n, list.Count);
        for (int i = 0; i < count; i++) yield return list[i];
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
                    dict[key] = string.Empty;
                }
            }
        }
        return dict;
    }
}
