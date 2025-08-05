using System;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Services.PM4.Core;

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
            // Supported flags:
            // --single-tile for performance testing or isolated analysis
            // --per-object to export each surface group as a separate OBJ file
            bool useSingleTile = args.Contains("--single-tile");
            bool usePerObject = args.Contains("--per-object");

            // --- Service Initialization ---
            var chunkAccessService = new Pm4ChunkAccessService();
            var fieldMappingService = new Pm4FieldMappingService(chunkAccessService);
            var exportService = new Pm4ExportService(chunkAccessService, fieldMappingService);

            // --- Scene Loading ---
            var adapter = new Pm4Adapter();
            var loadOptions = new Pm4LoadOptions { CaptureRawData = true };
            Pm4Scene scene;
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
            
            if (usePerObject)
            {
                // Use the refined hierarchical assembler for multiple object output
                ConsoleLogger.WriteLine("Executing refined hierarchical PM4 export pipeline...");
                Pm4RefinedHierarchicalObjectAssembler.StreamRefinedHierarchicalObjects(scene, outputDir);
                ConsoleLogger.WriteLine($"Export complete! Multiple OBJ files written to: {outputDir}");
            }
            else
            {
                // Use the original single-file exporter for raw surface output
                var outputPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputPath) + "_raw.obj");
                var exportOptions = new Pm4ExportOptions
                {
                    OutputPath = outputPath,
                    Strategy = ExportStrategy.RawSurfaces
                };
                
                ConsoleLogger.WriteLine("Executing raw PM4 export pipeline...");
                await exportService.ExportAsync(scene, exportOptions);
                ConsoleLogger.WriteLine($"Export complete! Output file: {outputPath}");
            }
            return 0;
        }
    }
}
