using System;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command to test the revolutionary 4D cross-tile object assembler
    /// Implements the USER's breakthrough theory: SurfaceKeys as depth selectors in 4D construct
    /// </summary>
    public class Pm4Export4DObjectsCommand
    {
        public async Task RunAsync(string inputPath, string outputDir)
        {
            ConsoleLogger.WriteLine("=== PM4 4D Cross-Tile Object Export ===");
            ConsoleLogger.WriteLine("Testing revolutionary 4D theory: ParentId + Cross-tile + Cross-depth assembly");
            
            if (!File.Exists(inputPath))
            {
                ConsoleLogger.WriteLine($"Input file not found: {inputPath}");
                return;
            }
            
            Directory.CreateDirectory(outputDir);
            
            try
            {
                ConsoleLogger.WriteLine($"Loading PM4 scene from: {inputPath}");
                
                // Load the PM4 scene with full region data
                var adapter = new Pm4Adapter();
                var scene = await Task.Run(() => adapter.LoadRegion(inputPath));
                
                ConsoleLogger.WriteLine($"Scene loaded: {scene.Vertices.Count} vertices, {scene.Links?.Count ?? 0} links, {scene.Surfaces?.Count ?? 0} surfaces");
                
                // Create 4D cross-tile assembler
                var assembler = new Pm4CrossTileObjectAssembler();
                
                // Generate base filename from input
                var inputFileName = Path.GetFileNameWithoutExtension(inputPath);
                var baseFileName = Path.Combine(outputDir, inputFileName);
                
                // Execute working spatial clustering assembly and export
                assembler.Export4DObjects(scene, outputDir, inputFileName);
                
                ConsoleLogger.WriteLine("=== 4D OBJECT EXPORT COMPLETED ===");
                ConsoleLogger.WriteLine($"Output directory: {outputDir}");
                ConsoleLogger.WriteLine("Check OBJ files for complete cross-tile + cross-depth objects!");
                
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during 4D object export: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}
