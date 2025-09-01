using System;
using System.IO;
using System.Linq;
using U = GillijimProject.Utilities.Utilities;
using GillijimProject.WowFiles.Alpha;

namespace GillijimProject;

/// <summary>
/// [PORT] Minimal CLI stub. Mirrors lib/gillijimproject/main.cpp flow incrementally.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: gillijimproject-csharp <WDT_ALPHA_PATH> [-o|--out <OUTPUT_DIR>]");
            return 1;
        }

        string? path = null;
        string? outputDir = null;
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a == "-o" || a == "--out")
            {
                if (i + 1 >= args.Length)
                {
                    Console.Error.WriteLine("Missing value for -o|--out");
                    return 1;
                }
                outputDir = args[++i];
            }
            else if (!a.StartsWith("-"))
            {
                if (path == null) path = a;
            }
        }

        if (path == null || !File.Exists(path))
        {
            Console.Error.WriteLine($"File not found: {path}");
            return 2;
        }

        if (string.IsNullOrWhiteSpace(outputDir))
        {
            var baseName = Path.GetFileNameWithoutExtension(path);
            var dirName = baseName + "_out";
            var parent = Path.GetDirectoryName(path) ?? string.Empty;
            outputDir = Path.Combine(parent, dirName);
        }
        Directory.CreateDirectory(outputDir);
        Console.WriteLine($"[INFO] Output directory: {Path.GetFullPath(outputDir)}");

        try
        {
            var bytes = U.GetWholeFile(path);
            Console.WriteLine($"[INFO] Read {bytes.Length} bytes from {path}");
            var wdtVersion = U.GetWdtVersion(path);
            Console.WriteLine($"[INFO] WDT version guess: {wdtVersion}");

            var wdtAlpha = new WdtAlpha(path);
            var existing = wdtAlpha.GetExistingAdtsNumbers();
            Console.WriteLine($"[INFO] ADT tiles present: {existing.Count}");
            Console.WriteLine("[INFO] First 16 tile indices: " + string.Join(", ", existing.Take(16)));

            var mdnm = wdtAlpha.GetMdnmFileNames();
            var monm = wdtAlpha.GetMonmFileNames();
            Console.WriteLine($"[INFO] MDNM count: {mdnm.Count}, MONM count: {monm.Count}");
            Console.WriteLine("[INFO] MDNM sample: " + string.Join(" | ", mdnm.Take(3)));
            Console.WriteLine("[INFO] MONM sample: " + string.Join(" | ", monm.Take(3)));

            // Write LK WDT
            var wdtLk = wdtAlpha.ToWdt();
            wdtLk.ToFile(outputDir);
            Console.WriteLine($"[INFO] Wrote WDT: {Path.Combine(outputDir, Path.GetFileName(path) + "_new")}");

            // Write minimal LK ADTs for existing tiles
            var adtOffsets = wdtAlpha.GetAdtOffsetsInMain();
            int written = 0;
            foreach (var idx in existing)
            {
                var adtAlphaTile = new AdtAlpha(path, adtOffsets[idx], idx);
                var lkAdt = adtAlphaTile.ToAdtLk(mdnm, monm);
                lkAdt.ToFile(outputDir);
                written++;
            }
            Console.WriteLine($"[INFO] Wrote {written} ADT files (minimal MVER+MHDR stubs).\nNOTE: Full content mapping to LK is pending.");

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 3;
        }
    }
}
