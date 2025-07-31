using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using ParpToolbox.Services.PM4;
using ParpToolbox.Services.WMO;
using ParpToolbox.Utils;
using ParpToolbox.Formats.P4.Chunks.Common;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command to test PM4-to-WMO spatial correlation and matching
    /// </summary>
    public class Pm4WmoMatchCommand
    {
        public void Execute(string pm4Path, string wmoPath, string outputPath)
        {
            ConsoleLogger.WriteLine("=== PM4-WMO Spatial Correlation and Matching Test ===");
            ConsoleLogger.WriteLine($"PM4 Input: {pm4Path}");
            ConsoleLogger.WriteLine($"WMO Input: {wmoPath}");
            ConsoleLogger.WriteLine($"Output: {outputPath}");

            try
            {
                // Validate input files
                if (!File.Exists(pm4Path))
                {
                    ConsoleLogger.WriteLine($"ERROR: PM4 file not found: {pm4Path}");
                    return;
                }

                if (!File.Exists(wmoPath))
                {
                    ConsoleLogger.WriteLine($"ERROR: WMO file not found: {wmoPath}");
                    return;
                }

                if (!pm4Path.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase))
                {
                    ConsoleLogger.WriteLine($"ERROR: Input file must be a PM4 file: {pm4Path}");
                    return;
                }

                if (!wmoPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                {
                    ConsoleLogger.WriteLine($"ERROR: Input file must be a WMO file: {wmoPath}");
                    return;
                }

                // Create output directory
                Directory.CreateDirectory(outputPath);

                // Step 1: Extract PM4 buildings using spatial clustering
                ConsoleLogger.WriteLine("\n=== Step 1: Extracting PM4 Buildings via Spatial Clustering ===");
                var pm4Buildings = ExtractPm4Buildings(pm4Path);
                
                if (!pm4Buildings.Any())
                {
                    ConsoleLogger.WriteLine("ERROR: No PM4 buildings extracted from spatial clustering");
                    return;
                }

                ConsoleLogger.WriteLine($"Extracted {pm4Buildings.Count} PM4 buildings");

                // Step 2: Perform PM4-WMO correlation
                ConsoleLogger.WriteLine("\n=== Step 2: PM4-WMO Spatial Correlation Analysis ===");
                var correlationResults = PerformCorrelationAnalysis(pm4Buildings, wmoPath);

                // Step 3: Generate results report
                ConsoleLogger.WriteLine("\n=== Step 3: Generating Correlation Report ===");
                GenerateCorrelationReport(correlationResults, outputPath);

                ConsoleLogger.WriteLine($"\n✅ SUCCESS: PM4-WMO matching analysis complete!");
                ConsoleLogger.WriteLine($"Results saved to: {outputPath}");

            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        /// <summary>
        /// Extract PM4 buildings using spatial clustering (reusing existing logic)
        /// </summary>
        private List<Pm4WmoMatcher.Pm4Building> ExtractPm4Buildings(string pm4Path)
        {
            var buildings = new List<Pm4WmoMatcher.Pm4Building>();

            try
            {
                // Load PM4 region (resolves cross-tile vertex references)
                ConsoleLogger.WriteLine("Loading PM4 region to resolve cross-tile references...");
                var adapter = new Pm4Adapter();
                var scene = adapter.LoadRegion(pm4Path);

                if (scene == null)
                {
                    ConsoleLogger.WriteLine("ERROR: Failed to load PM4 scene");
                    return buildings;
                }

                ConsoleLogger.WriteLine($"Scene loaded: {scene.Placements.Count} placements, {scene.Links.Count} links, {scene.Surfaces.Count} surfaces");

                // Use spatial clustering assembler to export buildings
                var assembler = new Pm4SpatialClusteringAssembler();
                var tempOutputDir = Path.Combine(Path.GetTempPath(), "pm4_temp_" + Guid.NewGuid().ToString("N")[0..8]);
                var exportSummary = assembler.ExportBuildingsUsingSpatialClustering(scene, tempOutputDir, "temp_15_37");

                ConsoleLogger.WriteLine($"Spatial clustering exported {exportSummary.Buildings.Count} building clusters");

                // Convert to PM4WMO matcher format
                int buildingIndex = 1;
                foreach (var building in exportSummary.Buildings)
                {
                    var buildingId = $"development_15_37_building_{buildingIndex}";
                    
                    // Generate a synthetic surface key from building metadata or use index
                    var surfaceKey = (uint)(building.Metadata.ContainsKey("SurfaceKey") ?
                        building.Metadata["SurfaceKey"] : (0x40000000 + buildingIndex));
                    
                    var triangleCount = building.TriangleCount;
                    var vertexCount = building.VertexCount;

                    // Extract MSCN vertices if available from ExtraChunks
                    var mscnVertices = new List<Vector3>();
                    var mscnChunk = scene.ExtraChunks?.OfType<MscnChunk>().FirstOrDefault();
                    if (mscnChunk?.Vertices?.Any() == true)
                    {
                        // For now, use sample MSCN vertices - in real implementation,
                        // we'd need to map building surfaces to their MSCN collision vertices
                        mscnVertices = mscnChunk.Vertices.Take(Math.Min(10, mscnChunk.Vertices.Count)).ToList();
                    }

                    var pm4Building = Pm4WmoMatcher.CreatePm4Building(
                        buildingId, surfaceKey, triangleCount, vertexCount, mscnVertices);

                    buildings.Add(pm4Building);
                    buildingIndex++;
                }

                // Clean up temp directory
                try
                {
                    if (Directory.Exists(tempOutputDir))
                        Directory.Delete(tempOutputDir, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }

                return buildings;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR extracting PM4 buildings: {ex.Message}");
                return buildings;
            }
        }

        /// <summary>
        /// Perform PM4-WMO correlation analysis
        /// </summary>
        private List<Pm4WmoMatcher.CorrelationResult> PerformCorrelationAnalysis(
            List<Pm4WmoMatcher.Pm4Building> pm4Buildings, string wmoPath)
        {
            try
            {
                // Create services
                var serviceCollection = new ServiceCollection();
                serviceCollection.AddLogging(builder => builder.AddConsole());
                serviceCollection.AddTransient<IWmoLoader, WowToolsLocalWmoLoader>();
                serviceCollection.AddTransient<Pm4WmoMatcher>();

                var serviceProvider = serviceCollection.BuildServiceProvider();
                var matcher = serviceProvider.GetService<Pm4WmoMatcher>();

                // Perform correlation
                var results = matcher.Correlate(pm4Buildings, wmoPath);

                ConsoleLogger.WriteLine($"Correlation analysis complete: {results.Count} results");
                ConsoleLogger.WriteLine($"Valid matches found: {results.Count(r => r.IsValidMatch)}");

                return results;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR in correlation analysis: {ex.Message}");
                return new List<Pm4WmoMatcher.CorrelationResult>();
            }
        }

        /// <summary>
        /// Generate detailed correlation report
        /// </summary>
        private void GenerateCorrelationReport(List<Pm4WmoMatcher.CorrelationResult> results, string outputPath)
        {
            try
            {
                var reportPath = Path.Combine(outputPath, "pm4_wmo_correlation_report.txt");
                var csvPath = Path.Combine(outputPath, "pm4_wmo_correlation_data.csv");

                // Generate detailed text report
                using (var writer = new StreamWriter(reportPath))
                {
                    writer.WriteLine("PM4-WMO Spatial Correlation Analysis Report");
                    writer.WriteLine("===========================================");
                    writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                    writer.WriteLine();

                    writer.WriteLine("SUMMARY:");
                    writer.WriteLine($"  Total PM4 Buildings Analyzed: {results.Count}");
                    writer.WriteLine($"  Valid Matches Found: {results.Count(r => r.IsValidMatch)}");
                    writer.WriteLine($"  Match Success Rate: {(results.Count > 0 ? results.Count(r => r.IsValidMatch) * 100.0 / results.Count : 0):F1}%");
                    writer.WriteLine();

                    writer.WriteLine("TOP MATCHES (by Overall Score):");
                    writer.WriteLine("================================");
                    var topMatches = results.Where(r => r.IsValidMatch)
                                           .OrderByDescending(r => r.OverallScore)
                                           .Take(10);

                    foreach (var match in topMatches)
                    {
                        writer.WriteLine($"Building: {match.Pm4Building.BuildingId}");
                        writer.WriteLine($"  Surface Key: 0x{match.Pm4Building.SurfaceKey:X8}");
                        writer.WriteLine($"  PM4 Triangles: {match.Pm4Building.TriangleCount:N0}");
                        writer.WriteLine($"  PM4 Vertices: {match.Pm4Building.VertexCount:N0}");
                        writer.WriteLine($"  Spatial Distance: {match.SpatialDistance:F2} units");
                        writer.WriteLine($"  Geometric Similarity: {match.GeometricSimilarity:F3}");
                        writer.WriteLine($"  Overall Score: {match.OverallScore:F3}");
                        writer.WriteLine($"  PM4 Center: ({match.Pm4Building.CenterPoint.X:F1}, {match.Pm4Building.CenterPoint.Y:F1}, {match.Pm4Building.CenterPoint.Z:F1})");
                        writer.WriteLine();
                    }

                    writer.WriteLine("DETAILED ANALYSIS:");
                    writer.WriteLine("==================");
                    foreach (var result in results.OrderByDescending(r => r.OverallScore))
                    {
                        writer.WriteLine($"Building {result.Pm4Building.BuildingId}:");
                        writer.WriteLine($"  SurfaceKey: 0x{result.Pm4Building.SurfaceKey:X8}");
                        writer.WriteLine($"  Triangles: {result.Pm4Building.TriangleCount:N0}");
                        writer.WriteLine($"  Vertices: {result.Pm4Building.VertexCount:N0}");
                        writer.WriteLine($"  Spatial Distance: {result.SpatialDistance:F2}");
                        writer.WriteLine($"  Geometric Similarity: {result.GeometricSimilarity:F3}");
                        writer.WriteLine($"  Overall Score: {result.OverallScore:F3}");
                        writer.WriteLine($"  Valid Match: {(result.IsValidMatch ? "YES" : "NO")}");
                        writer.WriteLine();
                    }
                }

                // Generate CSV data for further analysis
                using (var writer = new StreamWriter(csvPath))
                {
                    writer.WriteLine("BuildingId,SurfaceKey,PM4Triangles,PM4Vertices,SpatialDistance,GeometricSimilarity,OverallScore,IsValidMatch,CenterX,CenterY,CenterZ");
                    
                    foreach (var result in results)
                    {
                        writer.WriteLine($"{result.Pm4Building.BuildingId}," +
                                       $"0x{result.Pm4Building.SurfaceKey:X8}," +
                                       $"{result.Pm4Building.TriangleCount}," +
                                       $"{result.Pm4Building.VertexCount}," +
                                       $"{result.SpatialDistance:F2}," +
                                       $"{result.GeometricSimilarity:F3}," +
                                       $"{result.OverallScore:F3}," +
                                       $"{result.IsValidMatch}," +
                                       $"{result.Pm4Building.CenterPoint.X:F2}," +
                                       $"{result.Pm4Building.CenterPoint.Y:F2}," +
                                       $"{result.Pm4Building.CenterPoint.Z:F2}");
                    }
                }

                ConsoleLogger.WriteLine($"Detailed report: {reportPath}");
                ConsoleLogger.WriteLine($"CSV data: {csvPath}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR generating report: {ex.Message}");
            }
        }
    }
}