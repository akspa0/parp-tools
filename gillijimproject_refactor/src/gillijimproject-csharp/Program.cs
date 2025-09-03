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

        string path = Path.GetFullPath(args[0]);
        string? outputDir = null;
        for (int i = 1; i < args.Length; i++)
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
        }

        string mapName = Path.GetFileNameWithoutExtension(path);
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        
        if (outputDir == null)
        {
            // Use project_output/<mapname>_TIMESTAMP directory
            outputDir = Path.Combine(
                Path.GetDirectoryName(Path.GetDirectoryName(path) ?? string.Empty) ?? string.Empty, 
                "project_output", 
                $"{mapName}_{timestamp}");
        }
        Directory.CreateDirectory(outputDir);
        Console.WriteLine($"[INFO] Output directory: {Path.GetFullPath(outputDir)}");

        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"File not found: {path}");
            return 2;
        }

        try
        {
            var bytes = File.ReadAllBytes(path);
            var wdtAlpha = new WdtAlpha(bytes);
            var adtOffsets = wdtAlpha.GetAdtOffsets();
            var existing = adtOffsets.Keys.ToList();
            existing.Sort();

            var mdnm = wdtAlpha.GetMphdAndMdnm();
            var monm = wdtAlpha.GetMphdAndMonm();

            Console.WriteLine($"[INFO] ADT tiles present: {existing.Count}");
            Console.WriteLine($"[INFO] First 16 tile indices: {string.Join(", ", existing.Take(16))}");
            Console.WriteLine($"[INFO] MDNM count: {mdnm.Count}, MONM count: {monm.Count}");
            Console.WriteLine($"[INFO] MDNM sample: {string.Join(" | ", mdnm.Take(3))}");
            Console.WriteLine($"[INFO] MONM sample: {string.Join(" | ", monm.Take(3))}");

            var newWdt = wdtAlpha.ToWdtLk(mdnm, monm);
            newWdt.Write(Path.Combine(outputDir, Path.GetFileName(path) + "_new"));
            Console.WriteLine($"[INFO] Wrote WDT: {Path.Combine(outputDir, Path.GetFileName(path) + "_new")}");

            int written = 0;
            foreach (var idx in existing)
            {
                // Generate ADT filename from index (row,col) format
                int row = idx / 64;
                int col = idx % 64;
                string adtFileName = $"{Path.GetFileNameWithoutExtension(path)}_{row}_{col}.adt";

                var adtFullPath = Path.Combine(Path.GetDirectoryName(path) ?? string.Empty, adtFileName);
                
                // Check if file exists before trying to process it
                if (!File.Exists(adtFullPath))
                {
                    Console.WriteLine($"[WARNING] ADT file not found: {adtFullPath}");
                    continue;
                }
                
                var adtAlphaTile = new AdtAlpha(adtFullPath, idx);
                var lkAdt = adtAlphaTile.ToAdtLk(mdnm, monm);
                lkAdt.ToFile(outputDir);
                written++;
                if (written % 50 == 0) Console.WriteLine($"[INFO] Wrote {written}/{existing.Count} ADTs");
            }
            Console.WriteLine($"[INFO] Wrote {written}/{existing.Count} ADTs");
        }
        catch (Exception e)
        {
            Console.Error.WriteLine(e.ToString());
            return 3;
        }
        return 0;
    }
}
