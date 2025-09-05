using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.IO;

namespace GillijimProject.Next.Cli.Commands;

public static class WdlDumpCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);
        if (!opts.TryGetValue("--in", out var inPath) || string.IsNullOrWhiteSpace(inPath) || !File.Exists(inPath))
        {
            Console.Error.WriteLine("[wdl-dump] Missing or invalid --in <path-to.wdl>.");
            Console.Error.WriteLine("Usage: wdl-dump --in <path-to.wdl>");
            return 2;
        }

        try
        {
            var wdl = AlphaReader.ParseWdl(inPath);
            int present = 0;
            int tilesWithHoles = 0;
            WdlTile? sample = null;
            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    var t = wdl.Tiles[y, x];
                    if (t is null) continue;
                    present++;
                    if (sample is null) sample = t;
                    bool anyHole = false;
                    var rows = t.HoleMask16;
                    for (int r = 0; r < rows.Length; r++)
                    {
                        if (rows[r] != 0) { anyHole = true; break; }
                    }
                    if (anyHole) tilesWithHoles++;
                }
            }

            Console.WriteLine($"[wdl-dump] file={Path.GetFileName(inPath)} present_tiles={present} of 4096 tiles_with_holes={tilesWithHoles}");
            if (sample is not null)
            {
                Console.WriteLine($"[wdl-dump] sample height17[0,0]={sample.Height17[0,0]} height16[0,0]={sample.Height16[0,0]}");
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[wdl-dump] Failed: {ex.Message}");
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
                    dict[key] = string.Empty;
                }
            }
        }
        return dict;
    }
}
