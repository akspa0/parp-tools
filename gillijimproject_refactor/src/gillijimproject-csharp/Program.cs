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
            Console.WriteLine("Usage: gillijimproject-csharp <WDT_ALPHA_PATH>");
            return 1;
        }

        var path = args[0];
        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"File not found: {path}");
            return 2;
        }

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

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 3;
        }
    }
}
