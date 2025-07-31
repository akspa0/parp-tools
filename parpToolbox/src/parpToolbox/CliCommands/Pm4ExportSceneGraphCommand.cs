using System;
using System.CommandLine;
using System.IO;
using ParpToolbox.Utils;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

namespace parpToolbox.CliCommands
{
    /// <summary>
    /// CLI Command for PM4 Scene Graph Traversal Export
    /// 
    /// Implements the breakthrough PM4 scene graph architecture approach with:
    /// - Hierarchical scene graph traversal (MPRL -> MSLK -> geometry)
    /// - Nested coordinate system transforms (MSCN 1/4096 scale -> full resolution)
    /// - Proper spatial anchoring and single-object building exports
    /// </summary>
    public static class Pm4ExportSceneGraphCommand
    {
        public static Command CreateCommand()
        {
            var command = new Command("pm4-export-scene-graph", "Export PM4 using scene graph traversal approach");
            
            var inputFileOption = new Option<FileInfo>(
                "--input",
                description: "Input PM4 file path")
            {
                IsRequired = true
            };
            inputFileOption.AddAlias("-i");
            
            var outputDirOption = new Option<DirectoryInfo>(
                "--output-dir",
                description: "Output directory for OBJ files (optional - uses timestamped project output)")
            {
                IsRequired = false
            };
            outputDirOption.AddAlias("-o");
            
            command.AddOption(inputFileOption);
            command.AddOption(outputDirOption);
            
            command.SetHandler((FileInfo inputFile, DirectoryInfo outputDir) =>
            {
                try
                {
                    ExecuteSceneGraphExport(inputFile, outputDir);
                }
                catch (Exception ex)
                {
                    ConsoleLogger.WriteLine($"ERROR: Scene graph export failed: {ex.Message}");
                    if (ex.InnerException != null)
                    {
                        ConsoleLogger.WriteLine($"ERROR: Inner exception: {ex.InnerException.Message}");
                    }
                    Environment.Exit(1);
                }
            }, inputFileOption, outputDirOption);
            
            return command;
        }
        
        private static void ExecuteSceneGraphExport(FileInfo inputFile, DirectoryInfo outputDir)
        {
            ConsoleLogger.WriteLine("=== PM4 Scene Graph Traversal Export ===");
            ConsoleLogger.WriteLine($"BREAKTHROUGH: Using PM4 as scene graph with nested coordinate systems");
            
            if (!inputFile.Exists)
            {
                throw new FileNotFoundException($"PM4 file not found: {inputFile.FullName}");
            }
            
            // Use project output directory if not specified
            string finalOutputDir;
            if (outputDir != null && outputDir.Exists)
            {
                finalOutputDir = outputDir.FullName;
                ConsoleLogger.WriteLine($"Using specified output directory: {finalOutputDir}");
            }
            else
            {
                finalOutputDir = Path.Combine(Directory.GetCurrentDirectory(), "output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                ConsoleLogger.WriteLine($"Using timestamped project output: {finalOutputDir}");
            }
            
            // Ensure output directory exists
            Directory.CreateDirectory(finalOutputDir);
            
            ConsoleLogger.WriteLine($"Input: {inputFile.FullName}");
            ConsoleLogger.WriteLine($"Output: {finalOutputDir}");
            
            // Load PM4 file
            ConsoleLogger.WriteLine("Loading PM4 scene...");
            var adapter = new Pm4Adapter();
            var pm4Scene = adapter.LoadRegion(inputFile.FullName);
            
            if (pm4Scene == null)
            {
                throw new InvalidOperationException("Failed to load PM4 scene");
            }
            
            ConsoleLogger.WriteLine($"PM4 scene loaded successfully");
            
            // Log scene graph structure
            LogSceneGraphStructure(pm4Scene);
            
            // Execute scene graph export
            ConsoleLogger.WriteLine("Executing scene graph traversal export...");
            var sceneGraphExporter = new Pm4SceneGraphExporter();
            sceneGraphExporter.ExportPm4SceneGraph(pm4Scene, finalOutputDir);
            
            ConsoleLogger.WriteLine($"Scene graph export completed!");
            ConsoleLogger.WriteLine($"Output files written to: {finalOutputDir}");
        }
        
        private static void LogSceneGraphStructure(ParpToolbox.Formats.PM4.Pm4Scene pm4Scene)
        {
            try
            {
                // Get data counts for scene graph structure logging
                var mprlCount = pm4Scene.Placements?.Count ?? 0;
                var mslkCount = pm4Scene.Links?.Count ?? 0;
                var msurCount = pm4Scene.Surfaces?.Count ?? 0;
                var msvtCount = pm4Scene.Vertices?.Count ?? 0;
                var indicesCount = pm4Scene.Indices?.Count ?? 0;
                var trianglesCount = pm4Scene.Triangles?.Count ?? 0;
                
                ConsoleLogger.WriteLine("=== Scene Graph Structure ===");
                ConsoleLogger.WriteLine($"MPRL (Root Nodes):      {mprlCount:N0}");
                ConsoleLogger.WriteLine($"MSLK (Child Nodes):     {mslkCount:N0}");
                ConsoleLogger.WriteLine($"MSUR (Surface Geometry): {msurCount:N0}");
                ConsoleLogger.WriteLine($"MSVT (Render Vertices):  {msvtCount:N0} (full scale)");
                ConsoleLogger.WriteLine($"Indices (Flat Buffer):   {indicesCount:N0}");
                ConsoleLogger.WriteLine($"Triangles (Processed):   {trianglesCount:N0}");
                
                // Calculate expected building count
                var uniqueBuildings = pm4Scene.Placements?.Select(m => m.Unknown4).Distinct().Count() ?? 0;
                
                ConsoleLogger.WriteLine($"Expected Buildings:     {uniqueBuildings} (grouped by MPRL.Unknown4)");
                
                if (mslkCount > 0 && uniqueBuildings > 0)
                {
                    var avgChildNodes = (double)mslkCount / uniqueBuildings;
                    ConsoleLogger.WriteLine($"Avg Child Nodes/Building: {avgChildNodes:F1}");
                }
                
                ConsoleLogger.WriteLine("=== Coordinate Systems ===");
                ConsoleLogger.WriteLine("MSCN: 1/4096 scale spatial anchors");
                ConsoleLogger.WriteLine("MSUR/MSVT: Full ADT resolution geometry");
                ConsoleLogger.WriteLine("Transform: MSCN * 4096 -> World Space");
                ConsoleLogger.WriteLine("=====================================");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"WARNING: Could not log scene graph structure: {ex.Message}");
            }
        }
    }
}
