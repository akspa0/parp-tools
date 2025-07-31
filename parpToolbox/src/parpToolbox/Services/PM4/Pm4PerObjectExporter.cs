using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;
using ParpToolbox.Services.PM4.Database;
using System.Globalization;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Exports PM4 objects using the correct PoC MSUR-based building extraction logic.
    /// This implementation directly ports the working ExportCompleteBuildings_FromMSUR_RenderGeometry
    /// approach to produce coherent building-scale OBJ exports.
    /// </summary>
    public class Pm4PerObjectExporter
    {
        public Pm4PerObjectExporter()
        {
        }

        public async Task ExportObjectsAsync(Pm4Scene scene, string outputDirectory, string sourceFileName = "PM4_Scene", IReadOnlyDictionary<string, byte[]>? capturedRawData = null)
        {
            ConsoleLogger.WriteLine("Starting PM4 database-first object extraction...");
            
            if (scene?.Vertices == null || scene.Vertices.Count == 0)
            {
                ConsoleLogger.WriteLine("No vertices found in scene.");
                return;
            }

            ConsoleLogger.WriteLine($"Global mesh loaded: {scene.Vertices.Count:N0} vertices, {scene.Triangles.Count:N0} triangles");
            ConsoleLogger.WriteLine($"Available data: {scene.Links.Count} MSLK links, {scene.Placements.Count} MPRL placements");
            
            // Create database path
            var databasePath = Path.Combine(outputDirectory, $"{sourceFileName}_analysis.db");
            ConsoleLogger.WriteLine($"Using database: {databasePath}");
            
            try
            {
                // Export scene to database (Phase 1: Pure data import)
                var dbExporter = new Pm4DatabaseExporter(databasePath);
                var pm4FileId = await dbExporter.ExportSceneAsync(scene, sourceFileName, databasePath, capturedRawData);
                
                ConsoleLogger.WriteLine($"Scene exported to database with ID: {pm4FileId}");
                ConsoleLogger.WriteLine($"Database path: {databasePath}");
                
                // Phase 1 Complete: Database import successful
                // Phase 2: Extract buildings from database and generate OBJs
                ConsoleLogger.WriteLine("Extracting buildings from database...");
                
                var extractor = new Pm4DatabaseObjectExtractor(databasePath);
                var buildings = await extractor.ExtractBuildingsAsync(pm4FileId);
                
                if (buildings.Any())
                {
                    var objOutputDir = Path.Combine(outputDirectory, "obj_exports");
                    await extractor.ExportBuildingsToOBJAsync(buildings, objOutputDir);
                    await GenerateAnalysisReportAsync(buildings, outputDirectory, sourceFileName);
                    ConsoleLogger.WriteLine($"Successfully exported {buildings.Count} buildings to OBJ files.");
                }
                else
                {
                    ConsoleLogger.WriteLine("No buildings extracted from database.");
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during database-first extraction: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
                throw;
            }
        }
        
        /// <summary>
        /// Synchronous wrapper for backward compatibility.
        /// </summary>
        public void ExportObjects(Pm4Scene scene, string outputDirectory)
        {
            var task = ExportObjectsAsync(scene, outputDirectory);
            task.Wait();
        }

        /// <summary>
        /// Generates an analysis report for the extracted buildings.
        /// </summary>
        private async Task GenerateAnalysisReportAsync(List<ExtractedBuilding> buildings, string outputDirectory, string sourceFileName)
        {
            var reportPath = Path.Combine(outputDirectory, $"{sourceFileName}_analysis_report.txt");
            
            using var writer = new StreamWriter(reportPath);
            
            await writer.WriteLineAsync($"PM4 Database-First Analysis Report");
            await writer.WriteLineAsync($"Generated: {DateTime.Now}");
            await writer.WriteLineAsync($"Source: {sourceFileName}");
            await writer.WriteLineAsync();
            
            await writer.WriteLineAsync($"Total Buildings Extracted: {buildings.Count}");
            await writer.WriteLineAsync();
            
            // Group by extraction method
            var methodGroups = buildings.GroupBy(b => b.ExtractionMethod).ToList();
            
            foreach (var methodGroup in methodGroups)
            {
                await writer.WriteLineAsync($"=== {methodGroup.Key} Method ===");
                await writer.WriteLineAsync($"Buildings: {methodGroup.Count()}");
                await writer.WriteLineAsync($"Total Vertices: {methodGroup.Sum(b => b.VertexCount):N0}");
                await writer.WriteLineAsync($"Total Triangles: {methodGroup.Sum(b => b.TriangleCount):N0}");
                await writer.WriteLineAsync();
                
                foreach (var building in methodGroup.OrderByDescending(b => b.VertexCount))
                {
                    await writer.WriteLineAsync($"  {building.Name}:");
                    await writer.WriteLineAsync($"    Vertices: {building.VertexCount:N0}");
                    await writer.WriteLineAsync($"    Triangles: {building.TriangleCount:N0}");
                    // Note: Bounds fields removed - they don't exist in the actual PM4 data format
                    if (building.PlacementId.HasValue)
                    {
                        await writer.WriteLineAsync($"    Placement ID: {building.PlacementId.Value}");
                    }
                    await writer.WriteLineAsync();
                }
            }
            
            await writer.WriteLineAsync("=== Summary ===");
            await writer.WriteLineAsync($"Largest Building: {buildings.OrderByDescending(b => b.VertexCount).FirstOrDefault()?.Name ?? "None"}");
            await writer.WriteLineAsync($"Most Complex: {buildings.OrderByDescending(b => b.TriangleCount).FirstOrDefault()?.Name ?? "None"}");
            await writer.WriteLineAsync($"Average Vertices per Building: {(buildings.Count > 0 ? buildings.Average(b => b.VertexCount) : 0):F0}");
            await writer.WriteLineAsync($"Average Triangles per Building: {(buildings.Count > 0 ? buildings.Average(b => b.TriangleCount) : 0):F0}");
            
            ConsoleLogger.WriteLine($"Analysis report generated: {reportPath}");
        }
    }

    public class BuildingModel
    {
        public string FileName { get; set; } = string.Empty;
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<int> TriangleIndices { get; set; } = new List<int>();
        public int FaceCount => TriangleIndices.Count / 3;
    }
}
