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
        public static async Task<int> Run(string[] args, string inputPath)
        {
            // --- Flag Parsing ---
            // All legacy/experimental flags have been removed. The exporter now works correctly by default.
            // The only supported flag is --single-tile for performance testing or isolated analysis.
            bool useSingleTile = args.Contains("--single-tile");

            // --- Scene Loading ---
            // --- Load with Raw Data Capture ---
            var adapter = new Pm4Adapter();
            var loadOptions = new Pm4LoadOptions
            {
                CaptureRawData = true,  // Enable raw chunk capture for database storage
                VerboseLogging = false
            };
            
            Pm4Scene scene;
            // Trigger region loading for coordinate-based files (e.g., development_00_00.pm4)
            if (inputPath.Contains("_00_00") || inputPath.Contains("_000"))
            {
                scene = adapter.LoadRegion(inputPath, loadOptions);
            }
            else
            {
                scene = adapter.Load(inputPath, loadOptions);
            }

            // --- Export ---
            var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputPath));
            
            ConsoleLogger.WriteLine("Using JSON export pipeline...");
            var jsonExporter = new Pm4JsonExportPipeline();
            var outputPath = await jsonExporter.ExportSceneAsync(
                scene, 
                Path.GetFileName(inputPath),
                inputPath,
                adapter.CapturedRawData,
                outputDir
            );

            ConsoleLogger.WriteLine($"JSON export complete! Output file: {outputPath}");
            return 0;
        }
    }
}
