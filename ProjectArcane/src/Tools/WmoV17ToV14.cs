using System;
using System.IO;
using ArcaneFileParser.Core.Formats.WMO.Conversion;

namespace ArcaneFileParser.Tools
{
    /// <summary>
    /// Command-line tool for converting WMO v17+ files to v14 format.
    /// </summary>
    public class WmoV17ToV14
    {
        public static int Main(string[] args)
        {
            if (args.Length < 1 || args.Length > 2)
            {
                PrintUsage();
                return 1;
            }

            var sourcePath = args[0];
            var targetPath = args.Length == 2 ? args[1] : GetDefaultTargetPath(sourcePath);

            if (!File.Exists(sourcePath))
            {
                Console.WriteLine($"Error: Source file '{sourcePath}' does not exist.");
                return 1;
            }

            Console.WriteLine($"Converting {sourcePath} to v14 format...");
            Console.WriteLine($"Output will be written to {targetPath}");

            try
            {
                var converter = new WmoV17ToV14Converter();
                if (converter.ConvertFile(sourcePath, targetPath))
                {
                    Console.WriteLine("Conversion completed successfully.");
                    return 0;
                }
                else
                {
                    Console.WriteLine("Conversion failed.");
                    return 1;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during conversion: {ex.Message}");
                return 1;
            }
        }

        private static string GetDefaultTargetPath(string sourcePath)
        {
            var directory = Path.GetDirectoryName(sourcePath);
            var fileName = Path.GetFileNameWithoutExtension(sourcePath);
            var extension = Path.GetExtension(sourcePath);
            return Path.Combine(directory ?? "", $"{fileName}_v14{extension}");
        }

        private static void PrintUsage()
        {
            Console.WriteLine("WMO v17 to v14 Converter");
            Console.WriteLine("Usage: WmoV17ToV14 <source_file> [target_file]");
            Console.WriteLine();
            Console.WriteLine("Arguments:");
            Console.WriteLine("  source_file    Path to the source WMO file (v17+)");
            Console.WriteLine("  target_file    Optional path for the converted v14 file");
            Console.WriteLine("                 If not specified, '_v14' will be appended to the source filename");
            Console.WriteLine();
            Console.WriteLine("Example:");
            Console.WriteLine("  WmoV17ToV14 MyWMO.wmo");
            Console.WriteLine("  WmoV17ToV14 MyWMO.wmo MyWMO_Classic.wmo");
        }
    }
} 