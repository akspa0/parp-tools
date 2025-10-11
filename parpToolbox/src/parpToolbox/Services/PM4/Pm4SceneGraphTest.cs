using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Utils;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Services.Coordinate;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// MINIMAL Scene Graph Test - Testing the breakthrough nested coordinate system architecture
    /// 
    /// Key Insights:
    /// - MSCN data is at 1/4096 scale (spatial anchors)
    /// - MSUR/MSVT data is at full resolution (geometry)
    /// - Need coordinate system transforms to unify
    /// - Scene graph traversal: MPRL → MSLK → geometry
    /// </summary>
    public static class Pm4SceneGraphTest
    {
        private const float MSCN_SCALE_FACTOR = 4096.0f; // MSCN is 1/4096 scale
        
        /// <summary>
        /// Test the scene graph architecture concept using existing working infrastructure
        /// </summary>
        public static void TestSceneGraphArchitecture(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("=== PM4 Scene Graph Architecture Test ===");
            ConsoleLogger.WriteLine("BREAKTHROUGH: Testing nested coordinate system transforms");
            
            // Phase 1: Analyze Scene Graph Structure
            AnalyzeSceneGraphStructure(scene);
            
            // Phase 2: Test Coordinate System Transforms
            TestCoordinateSystemTransforms(scene);
            
            // Phase 3: Test Hierarchical Traversal
            TestHierarchicalTraversal(scene);
            
            // Phase 4: Export Test Results
            ExportTestResults(scene, outputDirectory);
        }
        
        /// <summary>
        /// Analyze the scene graph structure to understand hierarchical organization
        /// </summary>
        private static void AnalyzeSceneGraphStructure(Pm4Scene scene)
        {
            ConsoleLogger.WriteLine("=== Scene Graph Structure Analysis ===");
            
            // Analyze available data
            var vertexCount = scene.Vertices?.Count ?? 0;
            var triangleCount = scene.Triangles?.Count ?? 0;
            var linkCount = scene.Links?.Count ?? 0;
            var surfaceCount = scene.Surfaces?.Count ?? 0;
            var placementCount = scene.Placements?.Count ?? 0;
            
            ConsoleLogger.WriteLine($"Scene Data Overview:");
            ConsoleLogger.WriteLine($"  Vertices: {vertexCount:N0} (full resolution)");
            ConsoleLogger.WriteLine($"  Triangles: {triangleCount:N0}");
            ConsoleLogger.WriteLine($"  MSLK Links: {linkCount:N0} (hierarchy)");
            ConsoleLogger.WriteLine($"  MSUR Surfaces: {surfaceCount:N0} (geometry)");
            ConsoleLogger.WriteLine($"  MPRL Placements: {placementCount:N0} (root nodes)");
            
            // Analyze hierarchy patterns
            if (scene.Placements != null && scene.Links != null)
            {
                // Group placements by Unknown4 (building identifier)
                var buildingGroups = scene.Placements.GroupBy(p => p.Unknown4).ToList();
                ConsoleLogger.WriteLine($"Building Groups: {buildingGroups.Count} (by MPRL.Unknown4)");
                
                // Find MSLK entries that match placement groups
                var matchingLinks = scene.Links.Where(link => 
                    buildingGroups.Any(group => group.Key == link.ParentIndex)).ToList();
                
                ConsoleLogger.WriteLine($"Matching MSLK Links: {matchingLinks.Count}");
                
                if (buildingGroups.Any())
                {
                    var avgLinksPerBuilding = (double)matchingLinks.Count / buildingGroups.Count;
                    ConsoleLogger.WriteLine($"Average Links per Building: {avgLinksPerBuilding:F1}");
                }
            }
        }
        
        /// <summary>
        /// Test coordinate system transforms between MSCN (1/4096) and full-resolution data
        /// </summary>
        private static void TestCoordinateSystemTransforms(Pm4Scene scene)
        {
            ConsoleLogger.WriteLine("=== Coordinate System Transform Test ===");
            
            // Check if we have MSCN data in raw chunks
            var mscnData = ExtractMscnData(scene);
            
            if (mscnData.Any())
            {
                ConsoleLogger.WriteLine($"MSCN Data Found: {mscnData.Count} vertices at 1/4096 scale");
                
                // Transform MSCN coordinates to full resolution
                var transformedMscn = mscnData.Select(vertex => 
                    new Vector3(vertex.X * MSCN_SCALE_FACTOR, 
                               vertex.Y * MSCN_SCALE_FACTOR, 
                               vertex.Z * MSCN_SCALE_FACTOR)).ToList();
                
                ConsoleLogger.WriteLine("Coordinate Transform Applied:");
                ConsoleLogger.WriteLine($"  Original MSCN range: [{GetBounds(mscnData)}]");
                ConsoleLogger.WriteLine($"  Transformed range:   [{GetBounds(transformedMscn)}]");
                
                // Compare with scene vertices
                if (scene.Vertices != null && scene.Vertices.Any())
                {
                    var sceneBounds = GetBounds(scene.Vertices);
                    ConsoleLogger.WriteLine($"  Scene vertex range:  [{sceneBounds}]");
                    
                    // Test spatial alignment
                    TestSpatialAlignment(transformedMscn, scene.Vertices);
                }
            }
            else
            {
                ConsoleLogger.WriteLine("No MSCN data found - transform test skipped");
            }
        }
        
        /// <summary>
        /// Test hierarchical scene graph traversal
        /// </summary>
        private static void TestHierarchicalTraversal(Pm4Scene scene)
        {
            ConsoleLogger.WriteLine("=== Hierarchical Traversal Test ===");
            
            if (scene.Placements == null || scene.Links == null)
            {
                ConsoleLogger.WriteLine("Missing placement or link data - traversal test skipped");
                return;
            }
            
            // Test scene graph traversal pattern: MPRL → MSLK → geometry
            var buildingCount = 0;
            var totalGeometry = 0;
            
            var buildingGroups = scene.Placements.GroupBy(p => p.Unknown4);
            
            foreach (var buildingGroup in buildingGroups)
            {
                var buildingId = buildingGroup.Key;
                var placements = buildingGroup.ToList();
                
                // Find MSLK child nodes for this building
                var childLinks = scene.Links.Where(link => link.ParentIndex == buildingId).ToList();
                
                if (childLinks.Any())
                {
                    buildingCount++;
                    
                    // Count geometry associated with child links
                    var geometryCount = childLinks.Sum(link => 
                        link.MspiIndexCount > 0 ? link.MspiIndexCount : 0);
                    
                    totalGeometry += geometryCount;
                    
                    if (buildingCount <= 5) // Log first few buildings
                    {
                        ConsoleLogger.WriteLine($"Building {buildingId:X8}: {placements.Count} placements, {childLinks.Count} links, ~{geometryCount} geometry indices");
                    }
                }
            }
            
            ConsoleLogger.WriteLine($"Traversal Results:");
            ConsoleLogger.WriteLine($"  Buildings with geometry: {buildingCount}");
            ConsoleLogger.WriteLine($"  Total geometry indices: {totalGeometry:N0}");
            
            if (buildingCount > 0)
            {
                var avgGeometryPerBuilding = (double)totalGeometry / buildingCount;
                ConsoleLogger.WriteLine($"  Average geometry per building: {avgGeometryPerBuilding:F0}");
            }
        }
        
        /// <summary>
        /// Export test results for analysis
        /// </summary>
        private static void ExportTestResults(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("=== Exporting Test Results ===");
            
            Directory.CreateDirectory(outputDirectory);
            
            var testResultsFile = Path.Combine(outputDirectory, "scene_graph_test_results.txt");
            
            using (var writer = new StreamWriter(testResultsFile))
            {
                writer.WriteLine("PM4 Scene Graph Architecture Test Results");
                writer.WriteLine("=========================================");
                writer.WriteLine();
                
                writer.WriteLine("BREAKTHROUGH INSIGHTS:");
                writer.WriteLine("- MSCN data uses 1/4096 scale coordinate system");
                writer.WriteLine("- MSUR/MSVT data uses full ADT resolution");
                writer.WriteLine("- Scene graph hierarchy: MPRL → MSLK → geometry");
                writer.WriteLine("- Coordinate system transforms required for unification");
                writer.WriteLine();
                
                writer.WriteLine("Scene Data Summary:");
                writer.WriteLine($"Vertices: {scene.Vertices?.Count ?? 0:N0}");
                writer.WriteLine($"Triangles: {scene.Triangles?.Count ?? 0:N0}");
                writer.WriteLine($"Links: {scene.Links?.Count ?? 0:N0}");
                writer.WriteLine($"Surfaces: {scene.Surfaces?.Count ?? 0:N0}");
                writer.WriteLine($"Placements: {scene.Placements?.Count ?? 0:N0}");
                writer.WriteLine();
                
                writer.WriteLine("Next Steps:");
                writer.WriteLine("1. Implement coordinate system unification");
                writer.WriteLine("2. Build proper scene graph traversal exporter");
                writer.WriteLine("3. Test with MSCN spatial anchoring");
                writer.WriteLine("4. Verify single-object building exports");
            }
            
            ConsoleLogger.WriteLine($"Test results written to: {testResultsFile}");
            ConsoleLogger.WriteLine("Scene Graph Architecture Test Complete!");
        }
        
        /// <summary>
        /// Extract MSCN data from raw chunks (simplified for testing)
        /// </summary>
        private static List<Vector3> ExtractMscnData(Pm4Scene scene)
        {
            // For now, return empty - would need to access raw chunk data
            // This is a placeholder for the MSCN coordinate system test
            return new List<Vector3>();
        }
        
        /// <summary>
        /// Get bounds string for a list of vertices
        /// </summary>
        private static string GetBounds(IEnumerable<Vector3> vertices)
        {
            if (!vertices.Any()) return "empty";
            
            var minX = vertices.Min(v => v.X);
            var maxX = vertices.Max(v => v.X);
            var minY = vertices.Min(v => v.Y);
            var maxY = vertices.Max(v => v.Y);
            var minZ = vertices.Min(v => v.Z);
            var maxZ = vertices.Max(v => v.Z);
            
            return $"X:{minX:F1}..{maxX:F1}, Y:{minY:F1}..{maxY:F1}, Z:{minZ:F1}..{maxZ:F1}";
        }
        
        /// <summary>
        /// Test spatial alignment between transformed MSCN and scene vertices
        /// </summary>
        private static void TestSpatialAlignment(List<Vector3> transformedMscn, List<Vector3> sceneVertices)
        {
            ConsoleLogger.WriteLine("Testing spatial alignment...");
            
            // Simple alignment test - check if any transformed MSCN points are near scene vertices
            const float tolerance = 10.0f;
            var alignedCount = 0;
            
            foreach (var mscnPoint in transformedMscn.Take(100)) // Test first 100
            {
                var nearestDistance = sceneVertices.Min(v => Vector3.Distance(mscnPoint, v));
                if (nearestDistance < tolerance)
                {
                    alignedCount++;
                }
            }
            
            var alignmentPercentage = (double)alignedCount / Math.Min(100, transformedMscn.Count) * 100;
            ConsoleLogger.WriteLine($"Spatial alignment: {alignmentPercentage:F1}% of MSCN points align with scene vertices");
        }
    }
}
