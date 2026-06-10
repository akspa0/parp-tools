using System;
using System.IO;

namespace WoWToolbox.WmoV14Converter
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: dotnet run -- <input.wmo> [output_prefix] [--merge]");
                Console.WriteLine("Example (per-group): dotnet run -- test_data/053_wmo/Ironforge_053.wmo output/053");
                Console.WriteLine("Example (merged):    dotnet run -- test_data/053_wmo/Ironforge_053.wmo output/053 --merge");
                return;
            }
            string inputWmo = args[0];
            string outputPrefix = args.Length > 1 && !args[1].StartsWith("--") ? args[1] : "output/053";
            bool merge = Array.Exists(args, a => a == "--merge");
            string outputDir = Path.GetDirectoryName(outputPrefix);
            if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
                Console.WriteLine($"[INFO] Created output directory: {outputDir}");
            }
            if (merge)
            {
                string mergedObj = outputPrefix + "-merged.obj";
                Console.WriteLine($"[INFO] Converting all groups in {inputWmo} to a single merged OBJ: {mergedObj}");
                WmoV14ToV17Converter.ExportMergedGroupsAsObj(inputWmo, mergedObj);
            }
            else
            {
                Console.WriteLine($"[INFO] Converting all groups in {inputWmo} to OBJ files with prefix {outputPrefix}-group-XXX.obj");
                WmoV14ToV17Converter.ExportAllGroupsAsObj(inputWmo, outputPrefix);
            }
        }
    }
}
