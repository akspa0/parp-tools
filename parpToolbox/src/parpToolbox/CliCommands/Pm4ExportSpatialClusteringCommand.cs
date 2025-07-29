using System;
using System.IO;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command to export PM4 objects using the WORKING spatial clustering approach from POC
    /// </summary>
    public class Pm4ExportSpatialClusteringCommand
    {
        public void Execute(string inputPath, string outputPath)
        {
            ConsoleLogger.WriteLine("=== PM4 Spatial Clustering Export (POC Working Implementation) ===");
            ConsoleLogger.WriteLine($"Input: {inputPath}");
            ConsoleLogger.WriteLine($"Output: {outputPath}");

            try
            {
                // Validate input file
                if (!File.Exists(inputPath))
                {
                    ConsoleLogger.WriteLine($"ERROR: Input file not found: {inputPath}");
                    return;
                }

                if (!inputPath.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase))
                {
                    ConsoleLogger.WriteLine($"ERROR: Input file must be a PM4 file: {inputPath}");
                    return;
                }

                // Create output directory
                Directory.CreateDirectory(outputPath);

                // Load PM4 region (resolves cross-tile vertex references)
                ConsoleLogger.WriteLine("Loading PM4 region to resolve cross-tile references...");
                var adapter = new Pm4Adapter();
                var scene = adapter.LoadRegion(inputPath);

                if (scene == null)
                {
                    ConsoleLogger.WriteLine("ERROR: Failed to load PM4 scene");
                    return;
                }

                ConsoleLogger.WriteLine($"Scene loaded: {scene.Placements.Count} placements, {scene.Links.Count} links, {scene.Surfaces.Count} surfaces");

                // Extract base filename for output naming
                var baseFileName = Path.GetFileNameWithoutExtension(inputPath);

                // Create and run spatial clustering assembler
                var assembler = new Pm4SpatialClusteringAssembler();
                var summary = assembler.ExportBuildingsUsingSpatialClustering(scene, outputPath, baseFileName);

                // Report results
                ConsoleLogger.WriteLine("=== EXPORT SUMMARY ===");
                ConsoleLogger.WriteLine($"Total Buildings: {summary.TotalBuildings}");
                ConsoleLogger.WriteLine($"Successful Exports: {summary.SuccessfulExports}");
                ConsoleLogger.WriteLine($"Export Duration: {summary.ExportDuration.TotalMilliseconds:F0}ms");
                ConsoleLogger.WriteLine($"Output Directory: {summary.OutputDirectory}");

                if (summary.SuccessfulExports > 0)
                {
                    ConsoleLogger.WriteLine("\n=== EXPORTED BUILDINGS ===");
                    foreach (var building in summary.Buildings)
                    {
                        if (building.HasGeometry)
                        {
                            ConsoleLogger.WriteLine($"  {building.FileName}: {building.TriangleCount} triangles, {building.VertexCount} vertices");
                            if (building.Metadata.ContainsKey("StructuralElements") && building.Metadata.ContainsKey("RenderSurfaces"))
                            {
                                ConsoleLogger.WriteLine($"    Structural Elements: {building.Metadata["StructuralElements"]}, Render Surfaces: {building.Metadata["RenderSurfaces"]}");
                            }
                        }
                    }
                }

                if (summary.SuccessfulExports == 0)
                {
                    ConsoleLogger.WriteLine("WARNING: No buildings were exported. Check if the PM4 file contains valid MSLK hierarchy data.");
                }
                else
                {
                    ConsoleLogger.WriteLine($"\n✅ SUCCESS: Exported {summary.SuccessfulExports} buildings using spatial clustering approach");
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR: Failed to export PM4 objects: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}
