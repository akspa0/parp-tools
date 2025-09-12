using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Next.Core.WowFiles.Alpha;

namespace GillijimProject.Next.Cli.Commands;

public static class AdtDumpCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);
        if (!opts.TryGetValue("--in", out var inPath) || string.IsNullOrWhiteSpace(inPath) || !File.Exists(inPath))
        {
            Console.Error.WriteLine("[adt-dump] Missing or invalid --in <path-to.adt>.");
            Console.Error.WriteLine("Usage: adt-dump --in <path-to.adt>");
            return 2;
        }

        try
        {
            var adt = AlphaAdtReader.Read(inPath);
            int mcnk = adt.PresentChunks;
            int withMcvt = 0, withMclq = 0;
            foreach (var c in adt.Chunks)
            {
                if (c.HasMcvt) withMcvt++;
                if (c.HasMclq) withMclq++;
            }

            Console.WriteLine($"[adt-dump] file={Path.GetFileName(inPath)} mcnk={mcnk} mcvt={withMcvt} mclq={withMclq}");
            var sample = FindFirst(adt.Chunks, c => c.HasMcvt) ?? FindFirst(adt.Chunks, _ => true);
            if (sample is not null)
            {
                Console.WriteLine($"[adt-dump] sample index={sample.Index} off={sample.McnkOffset} size={sample.McnkSize} hasMcvt={sample.HasMcvt} hasMclq={sample.HasMclq}");
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[adt-dump] Failed: {ex.Message}");
            return 3;
        }
    }

    private static T? FindFirst<T>(IReadOnlyList<T> list, Func<T, bool> pred) where T : class
    {
        for (int i = 0; i < list.Count; i++)
        {
            var item = list[i];
            if (pred(item)) return item;
        }
        return null;
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
