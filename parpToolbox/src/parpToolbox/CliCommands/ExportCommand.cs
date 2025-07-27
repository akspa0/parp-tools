using System;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Handles the PM4 export command. This has been consolidated to use a single,
    /// authoritative exporter (`Pm4PerObjectExporter`) to prevent implementation drift.
    /// </summary>
    internal static class ExportCommand
    {
        /// <summary>
        /// Executes a PM4 export run using the single, correct exporter.
        /// </summary>
        /// <param name="args">Full CLI args array (including the command token at index 0).</param>
        /// <param name="inputPath">Resolved absolute path to the PM4 file specified by the user.</param>
        /// <returns>Exit code (0 = success, non-zero = failure).</returns>
        public static int Run(string[] args, string inputPath)
        {
            // --- Flag Parsing ---
            // All legacy/experimental flags have been removed. The exporter now works correctly by default.
            // The only supported flag is --single-tile for performance testing or isolated analysis.
            bool useSingleTile = args.Contains("--single-tile");

            // --- Scene Loading ---
            Pm4Scene scene;
            var adapter = new Pm4Adapter();
            if (useSingleTile)
            {
                ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
                scene = adapter.Load(inputPath);
            }
            else
            {
                ConsoleLogger.WriteLine("Region mode active (default) – loading cross-tile references...");
                scene = adapter.LoadRegion(inputPath);
            }

            // --- Export ---
            var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputPath));
            
            ConsoleLogger.WriteLine("Using PoC spatial grouping exporter...");
            var exporter = new Pm4PerObjectExporter();
            exporter.ExportObjects(scene, outputDir);

            ConsoleLogger.WriteLine($"Export complete. Check the output directory: {outputDir}");
            return 0;
        }
    }
}
