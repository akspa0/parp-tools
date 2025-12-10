using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.Next.Core.Export;
using GillijimProject.Next.Core.IO;

namespace GillijimProject.Next.Cli.Commands;

/// <summary>
/// CLI command to generate ADT terrain files from WDL low-resolution height data.
/// </summary>
public static class WdlToAdtCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);
        
        if (!opts.TryGetValue("--in", out var inPath) || string.IsNullOrWhiteSpace(inPath) || !File.Exists(inPath))
        {
            Console.Error.WriteLine("[wdl-to-adt] Missing or invalid --in <path-to.wdl>.");
            PrintHelp();
            return 2;
        }
        
        if (!opts.TryGetValue("--out", out var outDir) || string.IsNullOrWhiteSpace(outDir))
        {
            outDir = Path.Combine(Path.GetDirectoryName(inPath) ?? ".", "adt_from_wdl");
        }

        string mapName = opts.TryGetValue("--map", out var m) && !string.IsNullOrWhiteSpace(m) 
            ? m 
            : Path.GetFileNameWithoutExtension(inPath);

        try
        {
            Console.WriteLine($"[wdl-to-adt] Reading WDL: {inPath}");
            var wdl = AlphaReader.ParseWdl(inPath);

            Directory.CreateDirectory(outDir);
            Console.WriteLine($"[wdl-to-adt] Output directory: {outDir}");

            bool fillGapsOnly = opts.ContainsKey("--fill-gaps");
            string? existingAdtDir = opts.TryGetValue("--existing", out var e) ? e : null;

            int generated = 0;
            int skipped = 0;
            int existingSkipped = 0;

            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    var tile = wdl.Tiles[y, x];
                    if (tile is null)
                    {
                        skipped++;
                        continue;
                    }

                    var adtPath = Path.Combine(outDir, $"{mapName}_{x}_{y}.adt");
                    
                    // Skip if --fill-gaps and ADT already exists (in output or existing dir)
                    if (fillGapsOnly)
                    {
                        bool exists = File.Exists(adtPath);
                        if (!exists && !string.IsNullOrEmpty(existingAdtDir))
                        {
                            var existingPath = Path.Combine(existingAdtDir, $"{mapName}_{x}_{y}.adt");
                            exists = File.Exists(existingPath);
                        }
                        if (exists)
                        {
                            existingSkipped++;
                            continue;
                        }
                    }

                    var adtData = WdlToAdtGenerator.GenerateAdt(tile, x, y);
                    File.WriteAllBytes(adtPath, adtData);
                    generated++;

                    if (generated % 50 == 0 || generated == 1)
                    {
                        Console.WriteLine($"[wdl-to-adt] Generated {generated} ADTs...");
                    }
                }
            }

            Console.WriteLine($"[wdl-to-adt] Complete: {generated} ADTs generated, {skipped} tiles skipped (no WDL data), {existingSkipped} tiles skipped (existing ADT)");
            Console.WriteLine($"[wdl-to-adt] Output: {outDir}");

            // Generate WDT file
            var wdtPath = Path.Combine(outDir, $"{mapName}.wdt");
            GenerateWdt(wdl, wdtPath, mapName);
            Console.WriteLine($"[wdl-to-adt] WDT written: {wdtPath}");

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[wdl-to-adt] Failed: {ex.Message}");
            return 3;
        }
    }

    private static void GenerateWdt(Core.Domain.Wdl wdl, string wdtPath, string mapName)
    {
        using var fs = new FileStream(wdtPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // MVER
        bw.Write(System.Text.Encoding.ASCII.GetBytes("REVM"));
        bw.Write(4);
        bw.Write(18);

        // MPHD (32 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("DHPM"));
        bw.Write(32);
        bw.Write(0x0E); // flags: MCCV | BigAlpha | DoodadRefsSorted
        for (int i = 0; i < 28; i++) bw.Write((byte)0);

        // MAIN (64*64*8 = 32768 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("NIAM"));
        bw.Write(64 * 64 * 8);
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                uint flags = wdl.Tiles[y, x] != null ? 1u : 0u; // HasAdt flag
                bw.Write(flags);
                bw.Write(0u); // asyncId
            }
        }

        // MWMO (empty)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("OMWM"));
        bw.Write(0);

        // MODF (empty)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("FDOM"));
        bw.Write(0);
    }

    private static void PrintHelp()
    {
        Console.WriteLine("wdl-to-adt: Generate 3.3.5 ADT terrain files from WDL low-resolution heights");
        Console.WriteLine("Usage: wdl-to-adt --in <path-to.wdl> [--out <output-dir>] [--map <map-name>] [--fill-gaps] [--existing <dir>]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --in <path>       Input WDL file (required)");
        Console.WriteLine("  --out <dir>       Output directory (default: adt_from_wdl next to input)");
        Console.WriteLine("  --map <name>      Map name for output files (default: WDL filename)");
        Console.WriteLine("  --fill-gaps       Only generate ADTs for tiles that don't already exist");
        Console.WriteLine("  --existing <dir>  Directory with existing ADTs to check (used with --fill-gaps)");
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
