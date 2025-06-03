using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.MSCNExporter
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 4)
            {
                Console.WriteLine("Usage: dotnet WoWToolbox.MSCNExporter.dll -i <input_file_or_dir> -o <output_dir>");
                return;
            }

            string inputPath = null;
            string outputDir = null;
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-i" && i + 1 < args.Length)
                    inputPath = args[++i];
                else if (args[i] == "-o" && i + 1 < args.Length)
                    outputDir = args[++i];
            }

            if (string.IsNullOrEmpty(inputPath) || string.IsNullOrEmpty(outputDir))
            {
                Console.WriteLine("ERROR: Input path and output directory are required.");
                return;
            }

            Directory.CreateDirectory(outputDir);

            var pm4Files = File.Exists(inputPath)
                ? new[] { inputPath }
                : Directory.Exists(inputPath)
                    ? Directory.EnumerateFiles(inputPath, "*.pm4", SearchOption.TopDirectoryOnly).ToArray()
                    : Array.Empty<string>();

            if (pm4Files.Length == 0)
            {
                Console.WriteLine($"No PM4 files found at {inputPath}");
                return;
            }

            foreach (var pm4File in pm4Files)
            {
                try
                {
                    var pm4 = PM4File.FromFile(pm4File);
                    if (pm4.MSCN == null || pm4.MSCN.ExteriorVertices.Count == 0)
                    {
                        Console.WriteLine($"No MSCN data in {Path.GetFileName(pm4File)}");
                        continue;
                    }
                    var outPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(pm4File) + "_mscn.obj");
                    using (var writer = new StreamWriter(outPath))
                    {
                        writer.WriteLine($"# MSCN Points OBJ Export for {Path.GetFileName(pm4File)}");
                        writer.WriteLine($"# Generated: {DateTime.Now}");
                        writer.WriteLine("o MSCN_Boundary");
                        foreach (var v in pm4.MSCN.ExteriorVertices)
                            writer.WriteLine($"v {v.X} {v.Y} {v.Z}");
                    }
                    Console.WriteLine($"Exported {pm4.MSCN.ExteriorVertices.Count} MSCN points to {outPath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR processing {pm4File}: {ex.Message}");
                }
            }
        }
    }
} 