using System;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Services.WMO;
using ParpToolbox.Utils;
using ParpToolbox.Formats.P4.Chunks.Common;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using WoWFormatLib.FileProviders;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command to test PM4-to-WMO spatial correlation and matching
    /// </summary>
    public class Pm4WmoMatchCommand
    {
        public void Execute(string pm4Path, string wmoPath, string outputPath, int targetTileX = 0, int targetTileY = 0)
        {
            ConsoleLogger.WriteLine("=== PM4-WMO Spatial Correlation and Matching Test ===");
            ConsoleLogger.WriteLine($"PM4 Input: {pm4Path}");
            ConsoleLogger.WriteLine($"WMO Input: {wmoPath}");
            ConsoleLogger.WriteLine($"Target Tile: {targetTileX:D2}_{targetTileY:D2}");
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

                // Initialize FileProvider with a default build to avoid ArgumentNullException
                // This is required for the wow.tools.local WMO reader to function properly
                var wmoDirectory = Path.GetDirectoryName(wmoPath);
                var localProvider = new LocalFileProvider(wmoDirectory);
                FileProvider.SetProvider(localProvider, "local");
                FileProvider.SetDefaultBuild("local");

                // Create output directory
                Directory.CreateDirectory(outputPath);

                // Step 1: Extract PM4 buildings from the specified tile only
                ConsoleLogger.WriteLine("\n=== Step 1: PM4 Building Extraction (Targeted Tile) ===");
                var pm4Buildings = ExtractPm4BuildingsFromSpecificTile(pm4Path, targetTileX, targetTileY);

                if (!pm4Buildings.Any())
                {
                    ConsoleLogger.WriteLine("ERROR: No PM4 buildings extracted from targeted tile");
                    return;
                }

                ConsoleLogger.WriteLine($"Extracted {pm4Buildings.Count} PM4 buildings from tile {targetTileX:D2}_{targetTileY:D2}");

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

                // Use spatial clustering assembler to export buildings based on correct PM4 architecture
                var assembler = new Pm4SpatialClusteringAssembler();
                // Use ProjectOutput helper to ensure files are written to the timestamped project directory
                var outputDir = ProjectOutput.CreateOutputDirectory("pm4_wmo_match");
                var exportSummary = assembler.ExportBuildingsUsingSpatialClustering(scene, outputDir, Path.GetFileNameWithoutExtension(pm4Path));

                ConsoleLogger.WriteLine($"Spatial clustering exported {exportSummary.Buildings.Count} building clusters");

                // Convert to PM4WMO matcher format
                int buildingIndex = 1;
                foreach (var building in exportSummary.Buildings)
                {
                    var mscnVertices = new List<Vector3>();
                    var mscnChunk = scene.ExtraChunks?.OfType<MscnChunk>().FirstOrDefault();
                    if (mscnChunk?.Vertices?.Any() == true)
                    {
                        foreach (var v in mscnChunk.Vertices)
                        {
                            // Transform Y,X,Z and scale once by 1/4096
                            mscnVertices.Add(new Vector3(v.Y, v.X, v.Z));
                        }
                    }

                    var mscnBoundingBoxMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
                    var mscnBoundingBoxMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

                    foreach (var vertex in mscnVertices)
                    {
                        mscnBoundingBoxMin.X = Math.Min(mscnBoundingBoxMin.X, vertex.X);
                        mscnBoundingBoxMin.Y = Math.Min(mscnBoundingBoxMin.Y, vertex.Y);
                        mscnBoundingBoxMin.Z = Math.Min(mscnBoundingBoxMin.Z, vertex.Z);

                        mscnBoundingBoxMax.X = Math.Max(mscnBoundingBoxMax.X, vertex.X);
                        mscnBoundingBoxMax.Y = Math.Max(mscnBoundingBoxMax.Y, vertex.Y);
                        mscnBoundingBoxMax.Z = Math.Max(mscnBoundingBoxMax.Z, vertex.Z);
                    }

                    var triangleCount = 0; // we no longer rely on surface triangles
                    var vertexCount = mscnVertices.Count;

                    uint surfaceKey = 0u;
                    if (building.Metadata.TryGetValue("SurfaceKey", out var skObj) && skObj is string skStr && skStr.StartsWith("0x"))
                    {
                        _ = uint.TryParse(skStr.Substring(2), System.Globalization.NumberStyles.HexNumber, null, out surfaceKey);
                    }

                    var pm4Building = Pm4WmoMatcher.CreatePm4Building(
                        $"B{buildingIndex}", surfaceKey, triangleCount, vertexCount, mscnVertices);

                    // Set the calculated properties
                    pm4Building.CenterPoint = (mscnBoundingBoxMin + mscnBoundingBoxMax) / 2;
                    pm4Building.BoundingBoxMin = mscnBoundingBoxMin;
                    pm4Building.BoundingBoxMax = mscnBoundingBoxMax;
                    pm4Building.SurfaceArea = 0;

                    buildings.Add(pm4Building);
                    buildingIndex++;
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR during PM4 building extraction: {ex.Message}");
            }

            return buildings;
        }

        /// <summary>
        /// Extract PM4 buildings from a specific tile using spatial clustering
        /// </summary>
        private List<Pm4WmoMatcher.Pm4Building> ExtractPm4BuildingsFromSpecificTile(string pm4Path, int targetTileX, int targetTileY)
        {
            var buildings = new List<Pm4WmoMatcher.Pm4Building>();

            try
            {
                // Load specific PM4 tile only
                ConsoleLogger.WriteLine($"Loading PM4 tile {targetTileX:D2}_{targetTileY:D2} only...");
                var adapter = new Pm4Adapter();
                var dir = Path.GetDirectoryName(pm4Path);
                var name = Path.GetFileNameWithoutExtension(pm4Path);
                var parts = name.Split('_');
                
                // Get the prefix (everything except the last two parts which are coordinates)
                var prefix = string.Join("_", parts.Take(parts.Length - 2));
                
                var scene = adapter.LoadSpecificTile(dir ?? "", prefix, targetTileX, targetTileY);

                if (scene == null)
                {
                    ConsoleLogger.WriteLine("ERROR: Failed to load PM4 scene");
                    return buildings;
                }

                ConsoleLogger.WriteLine($"Scene loaded: {scene.Placements.Count} placements, {scene.Links.Count} links, {scene.Surfaces.Count} surfaces");

                // Use spatial clustering assembler to export buildings based on correct PM4 architecture
                var assembler = new Pm4SpatialClusteringAssembler();
                // Use ProjectOutput helper to ensure files are written to the timestamped project directory
                var outputDir = ProjectOutput.CreateOutputDirectory("pm4_wmo_match");
                var exportSummary = assembler.ExportBuildingsUsingSpatialClustering(scene, outputDir, $"{prefix}_{targetTileX:D2}_{targetTileY:D2}");

                ConsoleLogger.WriteLine($"Spatial clustering exported {exportSummary.Buildings.Count} building clusters");

                // Convert to PM4WMO matcher format
                int buildingIndex = 1;
                foreach (var building in exportSummary.Buildings)
                {
                    // Determine mesh bounding box from the mesh vertices of this building
                    var meshMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
                    var meshMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
                    foreach (var mv in building.Vertices)
                    {
                        meshMin.X = Math.Min(meshMin.X, mv.X);
                        meshMin.Y = Math.Min(meshMin.Y, mv.Y);
                        meshMin.Z = Math.Min(meshMin.Z, mv.Z);

                        meshMax.X = Math.Max(meshMax.X, mv.X);
                        meshMax.Y = Math.Max(meshMax.Y, mv.Y);
                        meshMax.Z = Math.Max(meshMax.Z, mv.Z);
                    }

                    // Extract MSCN vertices that fall inside (or very close to) that bounding box.
                    var mscnVertices = new List<Vector3>();
                    const float bboxMargin = 5.0f; // a little padding to cope with numeric drift
                    if (scene.ExtraChunks.OfType<MscnChunk>().FirstOrDefault() is MscnChunk mscn)
                    {
                        foreach (var pt in mscn.Vertices)
                        {
                            var vec = new Vector3(pt.Y, pt.X, pt.Z);
                            if (vec.X >= meshMin.X - bboxMargin && vec.X <= meshMax.X + bboxMargin &&
                                vec.Y >= meshMin.Y - bboxMargin && vec.Y <= meshMax.Y + bboxMargin &&
                                vec.Z >= meshMin.Z - bboxMargin && vec.Z <= meshMax.Z + bboxMargin)
                            {
                                mscnVertices.Add(vec);
                            }
                        }
                    }

                    // If no MSCN vertices matched the mesh bbox, we still keep the filtered list (which may be empty).

                    
                    // Calculate bounding box from filtered MSCN vertices
                    var mscnBoundingBoxMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
                    var mscnBoundingBoxMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
                    foreach (var mv in mscnVertices)
                    {
                        mscnBoundingBoxMin.X = Math.Min(mscnBoundingBoxMin.X, mv.X);
                        mscnBoundingBoxMin.Y = Math.Min(mscnBoundingBoxMin.Y, mv.Y);
                        mscnBoundingBoxMin.Z = Math.Min(mscnBoundingBoxMin.Z, mv.Z);

                        mscnBoundingBoxMax.X = Math.Max(mscnBoundingBoxMax.X, mv.X);
                        mscnBoundingBoxMax.Y = Math.Max(mscnBoundingBoxMax.Y, mv.Y);
                        mscnBoundingBoxMax.Z = Math.Max(mscnBoundingBoxMax.Z, mv.Z);
                    }

                    var triangleCount = building.TriangleCount;
                    var vertexCount = mscnVertices.Count;

                    uint surfaceKey = 0u;
                    if (building.Metadata.TryGetValue("SurfaceKey", out var skObj) && skObj is string skStr && skStr.StartsWith("0x"))
                    {
                        _ = uint.TryParse(skStr.Substring(2), System.Globalization.NumberStyles.HexNumber, null, out surfaceKey);
                    }

                    // Export MSCN-only OBJ for visual debugging
                    try
                    {
                        var objPath = Path.Combine(outputDir, $"mscnbldg_{buildingIndex}.obj");
                        using var sw = new StreamWriter(objPath);
                        foreach (var v in mscnVertices)
                        {
                            sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
                        }
                    }
                    catch (Exception objEx)
                    {
                        ConsoleLogger.WriteLine($"WARNING: Failed to write MSCN OBJ for building {buildingIndex}: {objEx.Message}");
                    }

                    var pm4Building = Pm4WmoMatcher.CreatePm4Building(
                        $"B{buildingIndex}", surfaceKey, building.TriangleCount, building.VertexCount, mscnVertices);

                    // Set the calculated properties
                    pm4Building.CenterPoint = (mscnBoundingBoxMin + mscnBoundingBoxMax) / 2;
                    pm4Building.BoundingBoxMin = mscnBoundingBoxMin;
                    pm4Building.BoundingBoxMax = mscnBoundingBoxMax;
                    pm4Building.SurfaceArea = 0;
                    buildings.Add(pm4Building);
                    buildingIndex++;
                }

                // No need for cleanup since we're using the ProjectOutput directory
                // Output files will remain in the project_output directory as expected

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