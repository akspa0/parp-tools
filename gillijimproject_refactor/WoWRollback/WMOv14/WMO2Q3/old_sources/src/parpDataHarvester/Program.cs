using System;
using System.IO;

namespace ParpDataHarvester
{
    internal static class Program
    {
        static int Main(string[] args)
        {
            if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
            {
                PrintHelp();
                return 0;
            }

            var cmd = args[0];
            switch (cmd)
            {
                case "export-glb-raw":
                    return Commands.ExportGlbRawCommand.Run(args.AsSpan(1));
                default:
                    Console.Error.WriteLine($"Unknown command: {cmd}");
                    PrintHelp();
                    return 2;
            }
        }

        private static void PrintHelp()
        {
            Console.WriteLine("parpDataHarvester - PM4 GLB-RAW Exporter");
            Console.WriteLine();
            Console.WriteLine("Usage:");
            Console.WriteLine("  parpDataHarvester export-glb-raw --in <path> --out <dir> [--per-region] [--mode objects|surfaces]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --in            Path to a PM4 file or a directory containing tiles");
            Console.WriteLine("  --out           Output directory (will be created if missing)");
            Console.WriteLine("  --per-region    When --in is a directory, aggregate tiles as a region (default: off)");
            Console.WriteLine("  --mode          Geometry grouping: 'objects' (default) or 'surfaces'");
        }
    }
}
