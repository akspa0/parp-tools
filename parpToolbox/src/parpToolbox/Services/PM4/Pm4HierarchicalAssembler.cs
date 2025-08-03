using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Utils;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// PM4 Hierarchical Object Assembler - Implements the correct PM4 object assembly logic
    /// based on MPRL.Unknown4 to MSLK.ParentIndex hierarchical relationship discovery
    /// </summary>
    public class Pm4HierarchicalAssembler
    {
        public class CompleteBuilding
        {
            public string FileName { get; set; } = "";
            public string Category { get; set; } = "";
            public string MaterialName { get; set; } = "";
            public List<Vector3> Vertices { get; set; } = new();
            public List<int> TriangleIndices { get; set; } = new();
            public Dictionary<string, object> Metadata { get; set; } = new();
            
            public int VertexCount => Vertices.Count;
            public int TriangleCount => TriangleIndices.Count / 3;
            public bool HasGeometry => Vertices.Count > 0 && TriangleIndices.Count > 0;
        }

        public class ExportSummary
        {
            public int TotalBuildings { get; set; }
            public int SuccessfulExports { get; set; }
            public List<CompleteBuilding> Buildings { get; set; } = new();
            public string OutputDirectory { get; set; } = "";
            public TimeSpan ExportDuration { get; set; }
        }

        /// <summary>
        /// Export buildings using the correct hierarchical assembly logic
        /// Groups MPRL entries by Unknown4, then collects all MSLK entries with matching ParentIndex
        /// </summary>
        public ExportSummary ExportBuildingsUsingHierarchicalAssembly(ParpToolbox.Formats.PM4.Pm4Scene scene, string outputDirectory, string baseFileName)
        {
            var startTime = DateTime.Now;
            var summary = new ExportSummary
            {
                OutputDirectory = outputDirectory
            };

            Directory.CreateDirectory(outputDirectory);
            ConsoleLogger.WriteLine("=== PM4 Hierarchical Assembly - Building Export ===");

            if (scene?.Placements == null || !scene.Placements.Any())
            {
                ConsoleLogger.WriteLine("ERROR: No MPRL entries found in scene");
                return summary;
            }

            ConsoleLogger.WriteLine($"Total vertices: {scene.Vertices?.Count ?? 0}");
            ConsoleLogger.WriteLine($"Total triangles: {scene.Triangles?.Count ?? 0}");
            ConsoleLogger.WriteLine($"Total placements: {scene.Placements?.Count ?? 0}");
            ConsoleLogger.WriteLine($"Total links: {scene.Links?.Count ?? 0}");
            ConsoleLogger.WriteLine($"Total surfaces: {scene.Surfaces?.Count ?? 0}");
            
            try
            {
                // === STEP 1: GROUP MPRL ENTRIES BY UNKNOWN4 (BUILDING ID) ===
                ConsoleLogger.WriteLine("Starting MPRL grouping by Unknown4...");
                var buildingGroups = GroupMprlByUnknown4(scene);
                ConsoleLogger.WriteLine($"Found {buildingGroups.Count} unique building groups (MPRL.Unknown4)");

                // === STEP 2: FOR EACH BUILDING GROUP, COLLECT ALL MSLK ENTRIES WITH MATCHING PARENTINDEX ===
                ConsoleLogger.WriteLine("Starting building assembly from MSLK entries...");
                ConsoleLogger.WriteLine("This may take a few moments for large scenes with many buildings...");
                var buildings = AssembleBuildingsFromGroups(scene, buildingGroups);
                ConsoleLogger.WriteLine($"Assembled {buildings.Count} complete buildings");

                // === STEP 3: EXPORT BUILDINGS ===
                ConsoleLogger.WriteLine($"Starting export of buildings with geometry...");
                var exportedCount = 0;
                int totalToExport = buildings.Count(b => b.HasGeometry);
                int exportBatchSize = Math.Max(1, totalToExport / 10); // Report every 10%
                int exportedIndex = 0;
                DateTime exportStartTime = DateTime.Now;
                
                foreach (var building in buildings)
                {
                    // Skip empty buildings
                    if (!building.HasGeometry)
                    {
                        continue;
                    }
                    
                    try
                    {
                        // Generate output file name
                        string objFileName = $"{baseFileName}_building_{building.Metadata["BuildingId"]}.obj";
                        string objFilePath = Path.Combine(outputDirectory, objFileName);

                        // Export as OBJ
                        ExportObjFile(building, objFilePath);
                        exportedCount++;
                        
                        // Report periodically
                        if (exportedCount % exportBatchSize == 0 || exportedCount == 1 || exportedCount == totalToExport)
                        {
                            double exportPercent = (double)exportedCount / totalToExport * 100;
                            ConsoleLogger.WriteLine($"Export progress: {exportedCount}/{totalToExport} ({exportPercent:F1}%)");
                        }
                        
                        // Add to summary
                        summary.Buildings.Add(building);
                    }
                    catch (Exception ex)
                    {
                        ConsoleLogger.WriteLine($"ERROR exporting building {building.Metadata["BuildingId"]}: {ex.Message}");
                    }
                    
                    exportedIndex++;
                }

                summary.TotalBuildings = buildings.Count;
                summary.SuccessfulExports = exportedCount;
                summary.ExportDuration = DateTime.Now - startTime;

                ConsoleLogger.WriteLine($"=== Hierarchical assembly export complete ===");
                ConsoleLogger.WriteLine($"Found {buildings.Count} buildings, exported {exportedCount} with geometry");
                ConsoleLogger.WriteLine($"Output directory: {outputDirectory}");
                ConsoleLogger.WriteLine($"Export took {summary.ExportDuration.TotalSeconds:F2} seconds");
                
                return summary;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR during hierarchical assembly: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
                return summary;
            }
        }

        /// <summary>
        /// Group MPRL entries by Unknown4 (building ID)
        /// </summary>
        private Dictionary<uint, List<dynamic>> GroupMprlByUnknown4(Pm4Scene scene)
        {
            var groupedMprl = new Dictionary<uint, List<dynamic>>();
            
            foreach (var placement in scene.Placements)
            {
                uint unknown4 = GetUnknown4(placement);
                
                if (!groupedMprl.ContainsKey(unknown4))
                {
                    groupedMprl[unknown4] = new List<dynamic>();
                }
                
                groupedMprl[unknown4].Add(placement);
            }
            
            return groupedMprl;
        }
        
        /// <summary>
        /// Get Unknown4 value from an MPRL entry using reflection
        /// </summary>
        private uint GetUnknown4(dynamic mprl)
        {
            var type = mprl.GetType();
            var prop = type.GetProperty("Unknown4");
            
            if (prop != null)
            {
                return (uint)prop.GetValue(mprl);
            }
            
            ConsoleLogger.WriteLine("WARNING: Could not find Unknown4 property in MPRL entry");
            return 0;
        }
        
        /// <summary>
        /// Get ParentIndex value from an MSLK entry using reflection
        /// </summary>
        private uint GetParentIndex(dynamic mslk)
        {
            var type = mslk.GetType();
            var prop = type.GetProperty("ParentIndex");
            
            if (prop != null)
            {
                return (uint)prop.GetValue(mslk);
            }
            
            // Try Unknown4 as fallback
            var unknown4Prop = type.GetProperty("Unknown4");
            if (unknown4Prop != null)
            {
                return (uint)unknown4Prop.GetValue(mslk);
            }
            
            // Try field-based access
            var parentIndexField = type.GetField("ParentIndex");
            if (parentIndexField != null)
            {
                return (uint)parentIndexField.GetValue(mslk);
            }
            
            ConsoleLogger.WriteLine("WARNING: Could not find ParentIndex field in MSLK entry");
            return 0;
        }
        
        /// <summary>
        /// Check if the MSLK entry is a container node (MspiFirstIndex = -1)
        /// </summary>
        private bool IsContainerNode(dynamic mslk)
        {
            var type = mslk.GetType();
            var prop = type.GetProperty("MspiFirstIndex");
            
            if (prop != null)
            {
                var value = prop.GetValue(mslk);
                return (int)value == -1;
            }
            
            return false;
        }
        
        /// <summary>
        /// Assemble complete buildings from MPRL groups and related MSLK entries
        /// </summary>
        private List<CompleteBuilding> AssembleBuildingsFromGroups(
            Pm4Scene scene, 
            Dictionary<uint, List<dynamic>> buildingGroups)
        {
            var buildings = new List<CompleteBuilding>();
            int buildingIndex = 1;
            int totalGroups = buildingGroups.Count;
            int batchSize = Math.Max(1, totalGroups / 20); // Report progress every 5%
            int processedCount = 0;
            int withGeometryCount = 0;
            DateTime lastReport = DateTime.Now;
            DateTime startTime = DateTime.Now;
            
            ConsoleLogger.WriteLine($"Processing {totalGroups} building groups...");
            
            foreach (var group in buildingGroups)
            {
                var buildingId = group.Key;
                var mprlEntries = group.Value;
                
                // Find all MSLK entries with matching ParentIndex
                var relatedMslkEntries = new List<dynamic>();
                
                if (scene.Links != null)
                {
                    foreach (var link in scene.Links)
                    {
                        uint parentIndex = GetParentIndex(link);
                        
                        if (parentIndex == buildingId)
                        {
                            relatedMslkEntries.Add(link);
                        }
                    }
                }
                
                // Create complete building
                var building = CreateCompleteBuilding(
                    scene, 
                    buildingId, 
                    mprlEntries, 
                    relatedMslkEntries, 
                    buildingIndex);
                
                if (building != null)
                {
                    buildings.Add(building);
                    
                    if (building.HasGeometry)
                    {
                        withGeometryCount++;
                    }
                }
                
                buildingIndex++;
                processedCount++;
                
                // Report progress periodically
                TimeSpan elapsed = DateTime.Now - lastReport;
                if (processedCount % batchSize == 0 || elapsed.TotalSeconds >= 5)
                {
                    double percentComplete = (double)processedCount / totalGroups * 100;
                    TimeSpan totalElapsed = DateTime.Now - startTime;
                    double estimatedTotal = totalElapsed.TotalSeconds / percentComplete * 100;
                    double remaining = estimatedTotal - totalElapsed.TotalSeconds;
                    
                    ConsoleLogger.WriteLine($"Progress: {processedCount}/{totalGroups} buildings processed ({percentComplete:F1}%), {withGeometryCount} with geometry");
                    ConsoleLogger.WriteLine($"Time elapsed: {totalElapsed.TotalSeconds:F1}s, estimated remaining: {remaining:F1}s");
                    
                    lastReport = DateTime.Now;
                }
            }
            
            TimeSpan totalTime = DateTime.Now - startTime;
            ConsoleLogger.WriteLine($"Building assembly complete: {buildings.Count} total, {withGeometryCount} with geometry, in {totalTime.TotalSeconds:F1} seconds");
            
            return buildings;
        }
        
        /// <summary>
        /// Create a complete building from related MPRL and MSLK entries
        /// </summary>
        private CompleteBuilding CreateCompleteBuilding(
            Pm4Scene scene,
            uint buildingId,
            List<dynamic> mprlEntries,
            List<dynamic> mslkEntries,
            int buildingIndex)
        {
            var building = new CompleteBuilding
            {
                FileName = $"building_{buildingId}",
                Category = "HierarchicalBuilding",
                Vertices = new List<Vector3>(),
                TriangleIndices = new List<int>(),
                Metadata = new Dictionary<string, object>
                {
                    { "BuildingId", buildingId },
                    { "SurfaceKey", buildingId.ToString() }
                }
            };
            
            // Create vertex lookup to avoid duplicates
            var vertexLookup = new Dictionary<Vector3, int>();
            
            // Process all MSLK entries to collect geometry
            foreach (var mslk in mslkEntries)
            {
                // Skip container nodes
                if (IsContainerNode(mslk))
                {
                    continue;
                }
                
                // Get geometry from MSPI indices via MSLK entry
                CollectGeometryFromMslk(scene, mslk, building, vertexLookup);
            }
            
            return building;
        }
        
        /// <summary>
        /// Collect geometry from MSLK entry and add to building
        /// </summary>
        private void CollectGeometryFromMslk(
            Pm4Scene scene, 
            dynamic mslk, 
            CompleteBuilding building, 
            Dictionary<Vector3, int> vertexLookup)
        {
            try
            {
                var type = mslk.GetType();
                
                // Get MspiFirstIndex and MspiIndexCount to find geometry
                var firstIndexProp = type.GetProperty("MspiFirstIndex");
                var indexCountProp = type.GetProperty("MspiIndexCount");
                
                if (firstIndexProp == null || indexCountProp == null)
                {
                    return;
                }
                
                int firstIndex = (int)firstIndexProp.GetValue(mslk);
                int indexCount = (int)indexCountProp.GetValue(mslk);
                
                // Skip entries without geometry
                if (firstIndex < 0 || indexCount <= 0 || scene.Indices == null)
                {
                    return;
                }
                
                // Add triangles
                for (int i = 0; i + 2 < indexCount && firstIndex + i + 2 < scene.Indices.Count; i += 3)
                {
                    int v1Idx = scene.Indices[firstIndex + i];
                    int v2Idx = scene.Indices[firstIndex + i + 1];
                    int v3Idx = scene.Indices[firstIndex + i + 2];
                    
                    // Skip invalid indices
                    if (v1Idx < 0 || v1Idx >= scene.Vertices.Count ||
                        v2Idx < 0 || v2Idx >= scene.Vertices.Count ||
                        v3Idx < 0 || v3Idx >= scene.Vertices.Count)
                    {
                        continue;
                    }
                    
                    // Add vertices and indices
                    building.TriangleIndices.Add(GetOrAddVertex(building.Vertices, vertexLookup, scene.Vertices[v1Idx]));
                    building.TriangleIndices.Add(GetOrAddVertex(building.Vertices, vertexLookup, scene.Vertices[v2Idx]));
                    building.TriangleIndices.Add(GetOrAddVertex(building.Vertices, vertexLookup, scene.Vertices[v3Idx]));
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR collecting geometry from MSLK: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Get or add vertex to lookup table to avoid duplicates
        /// </summary>
        private int GetOrAddVertex(List<Vector3> vertices, Dictionary<Vector3, int> lookup, Vector3 vertex)
        {
            // Apply X-axis inversion for OBJ format
            Vector3 transformedVertex = new Vector3(-vertex.X, vertex.Y, vertex.Z);
            
            if (lookup.TryGetValue(transformedVertex, out int index))
            {
                return index;
            }
            
            vertices.Add(transformedVertex);
            lookup[transformedVertex] = vertices.Count - 1;
            return vertices.Count - 1;
        }
        
        /// <summary>
        /// Export building as OBJ file
        /// </summary>
        private void ExportObjFile(CompleteBuilding building, string filePath)
        {
            using (var writer = new StreamWriter(filePath))
            {
                // Write OBJ header
                writer.WriteLine($"# PM4 Building: {building.Metadata["BuildingId"]}");
                writer.WriteLine($"# Vertices: {building.VertexCount}");
                writer.WriteLine($"# Triangles: {building.TriangleCount}");
                writer.WriteLine($"# Exported by parpToolbox PM4 Hierarchical Assembler");
                writer.WriteLine();
                
                // Write material (placeholder)
                writer.WriteLine($"mtllib materials.mtl");
                writer.WriteLine($"o {building.FileName}");
                writer.WriteLine($"g {building.Category}");
                writer.WriteLine($"usemtl {building.MaterialName}");
                writer.WriteLine();
                
                // Write vertices
                foreach (var vertex in building.Vertices)
                {
                    writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }
                writer.WriteLine();
                
                // Write faces (adjusting for OBJ's 1-indexed format)
                for (int i = 0; i < building.TriangleIndices.Count; i += 3)
                {
                    writer.WriteLine($"f {building.TriangleIndices[i] + 1} {building.TriangleIndices[i + 1] + 1} {building.TriangleIndices[i + 2] + 1}");
                }
            }
        }
    }
}
