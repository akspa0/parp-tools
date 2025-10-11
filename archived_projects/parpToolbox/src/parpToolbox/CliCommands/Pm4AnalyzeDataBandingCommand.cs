using System;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command to analyze PM4 data for banding/layering patterns
    /// Investigates the hypothesis that object geometry is subdivided across "bands" or "layers"
    /// </summary>
    public class Pm4AnalyzeDataBandingCommand
    {
        public async Task RunAsync(string inputPath, string outputDir)
        {
            ConsoleLogger.WriteLine("=== PM4 Data Banding Analysis Command ===");
            
            if (!File.Exists(inputPath))
            {
                ConsoleLogger.WriteLine($"Input file not found: {inputPath}");
                return;
            }
            
            Directory.CreateDirectory(outputDir);
            
            try
            {
                ConsoleLogger.WriteLine($"Loading PM4 scene from: {inputPath}");
                
                // Load the PM4 scene
                var adapter = new Pm4Adapter();
                var scene = await Task.Run(() => adapter.LoadRegion(inputPath)); // Load full region to get cross-tile data
                
                ConsoleLogger.WriteLine($"Scene loaded: {scene.Vertices.Count} vertices, {scene.Links?.Count ?? 0} links, {scene.Surfaces?.Count ?? 0} surfaces");
                
                // Perform banding analysis
                var report = Pm4DataBandingAnalyzer.AnalyzeDataBanding(scene);
                
                // Export detailed report
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var reportPath = Path.Combine(outputDir, $"pm4_banding_analysis_{timestamp}.txt");
                
                Pm4DataBandingAnalyzer.ExportBandingReport(report, reportPath);
                
                ConsoleLogger.WriteLine($"Banding analysis report exported to: {reportPath}");
                
                // Summary to console
                ConsoleLogger.WriteLine();
                ConsoleLogger.WriteLine("=== ANALYSIS SUMMARY ===");
                ConsoleLogger.WriteLine($"Tiles found: {report.TileData.Count}");
                
                var crossTileParentIds = report.ParentIdToTiles.Count(kvp => kvp.Value.Count > 1);
                var crossTileSurfaceKeys = report.SurfaceKeyToTiles.Count(kvp => kvp.Value.Count > 1);
                
                ConsoleLogger.WriteLine($"ParentIds spanning multiple tiles: {crossTileParentIds}");
                ConsoleLogger.WriteLine($"SurfaceKeys spanning multiple tiles: {crossTileSurfaceKeys}");
                ConsoleLogger.WriteLine($"Unexplored chunk types: {report.UnexploredChunks.Distinct().Count()}");
                
                if (crossTileParentIds > 0 || crossTileSurfaceKeys > 0)
                {
                    ConsoleLogger.WriteLine();
                    ConsoleLogger.WriteLine("*** HYPOTHESIS CONFIRMED: Cross-tile object banding detected! ***");
                    ConsoleLogger.WriteLine("Objects are indeed subdivided across multiple tiles.");
                }
                
                // Recommend next steps
                ConsoleLogger.WriteLine();
                ConsoleLogger.WriteLine("=== RECOMMENDED NEXT STEPS ===");
                ConsoleLogger.WriteLine("1. Examine cross-tile ParentIds and SurfaceKeys for object reconstruction patterns");
                ConsoleLogger.WriteLine("2. Investigate unexplored chunks for hidden linking data");
                ConsoleLogger.WriteLine("3. Compare whole-map vs per-tile processing results");
                
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during banding analysis: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}
