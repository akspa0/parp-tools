using System;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using ParpToolbox.Services.PM4.Core;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Handles the PM4 chunk combination testing command.
    /// </summary>
    internal static class ChunkCombinationTestCommand
    {
        /// <summary>
        /// Executes a PM4 chunk combination test run.
        /// </summary>
        /// <param name="args">Full CLI args array (including the command token at index 0).</param>
        /// <param name="inputPath">Resolved absolute path to the PM4 file specified by the user.</param>
        /// <returns>Exit code (0 = success, non-zero = failure).</returns>
        public static async Task<int> Run(string[] args, string inputPath)
        {
            try
            {
                // --- Flag Parsing ---
                bool useSingleTile = args.Contains("--single-tile");

                // --- Service Initialization ---
                var chunkAccessService = new Pm4ChunkAccessService();
                var fieldMappingService = new Pm4FieldMappingService(chunkAccessService);
                var exportService = new Pm4ExportService(chunkAccessService, fieldMappingService);

                // --- Scene Loading ---
                var adapter = new Pm4Adapter();
                var loadOptions = new Pm4LoadOptions { CaptureRawData = true };
                Pm4Scene scene;
                
                if (useSingleTile || inputPath.Contains("_00_00") || inputPath.Contains("_000"))
                {
                    ConsoleLogger.WriteLine($"Loading single tile: {inputPath}");
                    scene = adapter.Load(inputPath, loadOptions);
                }
                else
                {
                    ConsoleLogger.WriteLine($"Loading region from: {inputPath}");
                    scene = adapter.LoadRegion(inputPath, loadOptions);
                }

                ConsoleLogger.WriteLine($"Scene loaded successfully:");
                ConsoleLogger.WriteLine($"  Vertices: {scene.Vertices.Count}");
                ConsoleLogger.WriteLine($"  Indices: {scene.Indices.Count}");
                ConsoleLogger.WriteLine($"  Surfaces: {scene.Surfaces.Count}");
                ConsoleLogger.WriteLine($"  Links: {scene.Links.Count}");
                ConsoleLogger.WriteLine($"  Placements: {scene.Placements.Count}");
                ConsoleLogger.WriteLine($"  Properties: {scene.Properties.Count}");

                // --- Run Chunk Combination Tests ---
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"chunk_test_{timestamp}");
                Directory.CreateDirectory(outputDir);

                ConsoleLogger.WriteLine($"Running chunk combination tests, output to: {outputDir}");
                var results = Pm4ChunkCombinationTester.RunExhaustiveChunkTests(scene, outputDir);

                ConsoleLogger.WriteLine($"Chunk combination tests completed. Results exported to: {outputDir}");
                return 0;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during chunk combination testing: {ex.Message}");
                ConsoleLogger.WriteLine(ex.StackTrace);
                return 1;
            }
        }
    }
}
