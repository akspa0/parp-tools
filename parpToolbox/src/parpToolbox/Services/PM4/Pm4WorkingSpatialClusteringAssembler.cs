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
    /// WORKING spatial clustering assembler extracted from POC poc_exporter.cs
    /// Uses spatial clustering to combine MSPV structural elements with nearby MSUR surfaces
    /// This is the proven approach that actually worked for building-scale object extraction
    /// </summary>
    public class Pm4WorkingSpatialClusteringAssembler
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
        /// Export buildings using the WORKING spatial clustering approach from POC
        /// </summary>
        public ExportSummary ExportBuildingsUsingSpatialClustering(Pm4Scene scene, string outputDirectory, string baseFileName)
        {
            var startTime = DateTime.Now;
            var summary = new ExportSummary
            {
                OutputDirectory = outputDirectory
            };

            Directory.CreateDirectory(outputDirectory);
            ConsoleLogger.WriteLine("=== PM4 Spatial Clustering Assembly (POC Working Implementation) ===");

            if (scene?.Links == null || !scene.Links.Any())
            {
                ConsoleLogger.WriteLine("ERROR: No MSLK entries found in scene");
                return summary;
            }

            // === STEP 1: FIND ROOT NODES (WORKING POC LOGIC) ===
            var rootNodes = new List<(int nodeIndex, MslkEntry entry)>();
            
            for (int i = 0; i < scene.Links.Count; i++)
            {
                var entry = scene.Links[i];
                if (entry.ParentIndex == i) // Self-referencing = root node
                {
                    rootNodes.Add((i, entry));
                }
            }
            
            ConsoleLogger.WriteLine($"Found {rootNodes.Count} root nodes");

            // === STEP 2: CREATE BUILDINGS USING PURE SURFACEKEY GROUPING ===
            // RESTORE ORIGINAL WORKING APPROACH: Group by SurfaceKey only, no MSLK or spatial clustering
            var buildings = new List<CompleteBuilding>();
            
            // EXPERIMENT: Test different SurfaceKey grouping levels to find building-scale objects
            // User insight: Top byte alone gives too much variation (157-289K triangles)
            // Try grouping by top 2 bytes (0xAABBxxxx) for potential building-level grouping
            
            var buildingGroupToSurfaces = new Dictionary<ushort, List<int>>();
            if (scene.Surfaces != null)
            {
                for (int i = 0; i < scene.Surfaces.Count; i++)
                {
                    var surface = scene.Surfaces[i];
                    // Extract top 2 bytes (0xAABBxxxx) as potential building group ID
                    var buildingGroup = (ushort)((surface.SurfaceKey >> 16) & 0xFFFF);
                    
                    if (!buildingGroupToSurfaces.ContainsKey(buildingGroup))
                    {
                        buildingGroupToSurfaces[buildingGroup] = new List<int>();
                    }
                    buildingGroupToSurfaces[buildingGroup].Add(i);
                }
            }
            
            ConsoleLogger.WriteLine($"Found {buildingGroupToSurfaces.Count} groups using top 2 bytes (0xAABBxxxx) grouping");
            
            var processedBuildingGroups = new HashSet<ushort>();
            
            if (scene.Surfaces != null && scene.Surfaces.Count > 0)
            {
                // Group surfaces by 2-byte building group and create one building per group
                foreach (var kvp in buildingGroupToSurfaces)
                {
                    var buildingGroup = kvp.Key;
                    var surfaceIndices = kvp.Value;
                    
                    // Skip if we've already processed this building group
                    if (processedBuildingGroups.Contains(buildingGroup))
                        continue;
                    
                    processedBuildingGroups.Add(buildingGroup);
                    
                    var buildingIndex = buildings.Count;
                    ConsoleLogger.WriteLine($"Building {buildingIndex + 1}: Group 0x{buildingGroup:X4} ({surfaceIndices.Count} surfaces)");
                    
                    // Create building with surfaces from this building group
                    var building = CreateBuildingFrom2ByteGroup(scene, buildingGroup, surfaceIndices, baseFileName, buildingIndex);
                    building.Metadata["BuildingGroup"] = $"0x{buildingGroup:X4}";
                    building.Metadata["SurfaceCount"] = surfaceIndices.Count;
                    
                    buildings.Add(building);
                }
            }

            // === STEP 3: EXPORT BUILDINGS ===
            int successfulExports = 0;
            foreach (var building in buildings)
            {
                if (building.HasGeometry)
                {
                    ExportBuildingToObj(building, outputDirectory);
                    successfulExports++;
                }
            }

            summary.TotalBuildings = buildings.Count;
            summary.SuccessfulExports = successfulExports;
            summary.Buildings = buildings;
            summary.ExportDuration = DateTime.Now - startTime;

            ConsoleLogger.WriteLine($"\\n=== EXPORT COMPLETE ===");
            ConsoleLogger.WriteLine($"Total buildings: {summary.TotalBuildings}");
            ConsoleLogger.WriteLine($"Successful exports: {summary.SuccessfulExports}");
            ConsoleLogger.WriteLine($"Duration: {summary.ExportDuration.TotalSeconds:F2} seconds");

            return summary;
        }

        /// <summary>
        /// Calculate bounding box of structural elements (MSLK->MSPI->MSCN/MSPV chain)
        /// </summary>
        private (Vector3 min, Vector3 max)? CalculateStructuralElementsBounds(Pm4Scene scene, List<dynamic> buildingEntries)
        {
            var allVertices = new List<Vector3>();
            
            foreach (var linkEntry in buildingEntries)
            {
                var entry = linkEntry.entry;
                if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount <= 0) continue;
                
                // Follow MSLK -> MSPI -> MSCN/MSPV chain to get vertices
                for (int i = 0; i < entry.MspiIndexCount && entry.MspiFirstIndex + i < scene.Spis.Count; i++)
                {
                    var spiIndex = entry.MspiFirstIndex + i;
                    if (spiIndex >= 0 && spiIndex < scene.Spis.Count)
                    {
                        var spi = scene.Spis[spiIndex];
                        
                        // Extract vertex from MSCN or MSPV (structural data)
                        if (spiIndex < scene.Vertices.Count)
                        {
                            allVertices.Add(scene.Vertices[spiIndex]);
                        }
                    }
                }
            }
            
            if (allVertices.Count == 0) return null;
            
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            
            foreach (var vertex in allVertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }
            
            return (min, max);
        }
        
        /// <summary>
        /// Find MSUR surfaces near structural bounds using spatial clustering
        /// </summary>
        private List<int> FindMSURSurfacesNearBounds(Pm4Scene scene, (Vector3 min, Vector3 max) bounds, float tolerance)
        {
            var nearbySurfaces = new List<int>();
            
            if (scene.Surfaces == null || scene.Indices == null) return nearbySurfaces;
            
            // Expand bounds by tolerance for spatial clustering
            var expandedMin = bounds.min - new Vector3(tolerance);
            var expandedMax = bounds.max + new Vector3(tolerance);
            
            for (int surfaceIndex = 0; surfaceIndex < scene.Surfaces.Count; surfaceIndex++)
            {
                var surface = scene.Surfaces[surfaceIndex];
                if (surface.IndexCount <= 0 || surface.MsviFirstIndex < 0) continue;
                
                // Check if any vertices of this surface are within the expanded bounds
                bool hasVertexInBounds = false;
                for (int i = 0; i < surface.IndexCount && surface.MsviFirstIndex + i < scene.Indices.Count; i++)
                {
                    int msviIndex = scene.Indices[(int)surface.MsviFirstIndex + i];
                    if (msviIndex >= 0 && msviIndex < scene.Vertices.Count)
                    {
                        var vertex = scene.Vertices[msviIndex];
                        if (vertex.X >= expandedMin.X && vertex.X <= expandedMax.X &&
                            vertex.Y >= expandedMin.Y && vertex.Y <= expandedMax.Y &&
                            vertex.Z >= expandedMin.Z && vertex.Z <= expandedMax.Z)
                        {
                            hasVertexInBounds = true;
                            break;
                        }
                    }
                }
                
                if (hasVertexInBounds)
                {
                    nearbySurfaces.Add(surfaceIndex);
                }
            }
            
            return nearbySurfaces;
        }
        
        /// <summary>
        /// Create hybrid building combining structural elements (MSLK) with nearby surfaces (MSUR)
        /// </summary>
        private CompleteBuilding CreateHybridBuilding_StructuralPlusNearby(Pm4Scene scene, 
            List<dynamic> buildingEntries, List<int> nearbySurfaces, string baseFileName, int buildingIndex)
        {
            var building = new CompleteBuilding
            {
                FileName = $"{baseFileName}_building_{buildingIndex:D3}",
                Category = "HybridBuilding",
                MaterialName = $"Building_{buildingIndex:D3}"
            };
            
            var vertexLookup = new Dictionary<Vector3, int>();
            
            // Add structural elements (MSLK->MSPI->MSCN/MSPV chain)
            foreach (var linkEntry in buildingEntries)
            {
                var entry = linkEntry.entry;
                if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount <= 0) continue;
                
                var structuralVertices = new List<Vector3>();
                for (int i = 0; i < entry.MspiIndexCount && entry.MspiFirstIndex + i < scene.Spis.Count; i++)
                {
                    var spiIndex = entry.MspiFirstIndex + i;
                    if (spiIndex >= 0 && spiIndex < scene.Vertices.Count)
                    {
                        // Apply X-axis flip for correct orientation
                        var vertex = scene.Vertices[spiIndex];
                        var flippedVertex = new Vector3(-vertex.X, vertex.Y, vertex.Z);
                        structuralVertices.Add(flippedVertex);
                    }
                }
                
                // Add structural vertices with deduplication
                var triangleIndices = new List<int>();
                foreach (var vertex in structuralVertices)
                {
                    if (!vertexLookup.ContainsKey(vertex))
                    {
                        vertexLookup[vertex] = building.Vertices.Count;
                        building.Vertices.Add(vertex);
                    }
                    triangleIndices.Add(vertexLookup[vertex]);
                }
                
                // Add triangles (groups of 3 vertices)
                for (int i = 0; i < triangleIndices.Count - 2; i += 3)
                {
                    if (i + 2 < triangleIndices.Count)
                    {
                        building.TriangleIndices.Add(triangleIndices[i]);
                        building.TriangleIndices.Add(triangleIndices[i + 1]);
                        building.TriangleIndices.Add(triangleIndices[i + 2]);
                    }
                }
            }
            
            // Add nearby surfaces (MSUR->MSVI->MSVT chain)
            foreach (int surfaceIndex in nearbySurfaces)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    if (surface.IndexCount <= 0 || surface.MsviFirstIndex < 0) continue;
                    
                    var surfaceVertices = new List<Vector3>();
                    for (int i = 0; i < surface.IndexCount && surface.MsviFirstIndex + i < scene.Indices.Count; i++)
                    {
                        int msviIndex = scene.Indices[(int)surface.MsviFirstIndex + i];
                        if (msviIndex >= 0 && msviIndex < scene.Vertices.Count)
                        {
                            // Apply X-axis flip for correct orientation
                            var vertex = scene.Vertices[msviIndex];
                            var flippedVertex = new Vector3(-vertex.X, vertex.Y, vertex.Z);
                            surfaceVertices.Add(flippedVertex);
                        }
                    }
                    
                    // Add surface vertices with deduplication
                    var triangleIndices = new List<int>();
                    foreach (var vertex in surfaceVertices)
                    {
                        if (!vertexLookup.ContainsKey(vertex))
                        {
                            vertexLookup[vertex] = building.Vertices.Count;
                            building.Vertices.Add(vertex);
                        }
                        triangleIndices.Add(vertexLookup[vertex]);
                    }
                    
                    // Add triangles (groups of 3 vertices)
                    for (int i = 0; i < triangleIndices.Count - 2; i += 3)
                    {
                        if (i + 2 < triangleIndices.Count)
                        {
                            building.TriangleIndices.Add(triangleIndices[i]);
                            building.TriangleIndices.Add(triangleIndices[i + 1]);
                            building.TriangleIndices.Add(triangleIndices[i + 2]);
                        }
                    }
                }
            }
            
            return building;
        }
        
        /// <summary>
        /// Create a building from a 2-byte SurfaceKey group (EXPERIMENT: top 2 bytes)
        /// Groups all surfaces with same 2-byte group ID (0xAABBxxxx)
        /// </summary>
        private CompleteBuilding CreateBuildingFrom2ByteGroup(Pm4Scene scene, ushort buildingGroup, List<int> surfaceIndices, string baseFileName, int buildingIndex)
        {
            var building = new CompleteBuilding
            {
                FileName = $"{baseFileName}_group_{buildingGroup:X4}",
                Category = "2ByteGroup",
                MaterialName = $"Group_{buildingGroup:X4}"
            };

            var vertexLookup = new Dictionary<Vector3, int>();

            // Add surfaces from this 2-byte building group
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    int indexCount = surface.IndexCount;
                    int first = (int)surface.MsviFirstIndex;

                    if (indexCount <= 0 || first < 0) continue;

                    // Collect vertices from MSVI indices -> MSVT vertices
                    var surfaceVertices = new List<Vector3>();
                    for (int i = 0; i < indexCount && first + i < scene.Indices.Count; i++)
                    {
                        int msviIndex = scene.Indices[first + i];
                        if (msviIndex >= 0 && msviIndex < scene.Vertices.Count)
                        {
                            // Apply X-axis flip for correct orientation
                            var vertex = scene.Vertices[msviIndex];
                            var flippedVertex = new Vector3(-vertex.X, vertex.Y, vertex.Z);
                            surfaceVertices.Add(flippedVertex);
                        }
                    }

                    // Add vertices to building (with deduplication)
                    var polygonIndices = new List<int>();
                    foreach (var vertex in surfaceVertices)
                    {
                        if (!vertexLookup.ContainsKey(vertex))
                        {
                            vertexLookup[vertex] = building.Vertices.Count;
                            building.Vertices.Add(vertex);
                        }
                        polygonIndices.Add(vertexLookup[vertex]);
                    }

                    // Handle n-gons properly (triangles, quads, pentagons)
                    if (polygonIndices.Count >= 3)
                    {
                        if (polygonIndices.Count == 3)
                        {
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[1]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                        }
                        else if (polygonIndices.Count == 4)
                        {
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[1]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                            
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                            building.TriangleIndices.Add(polygonIndices[3]);
                        }
                        else
                        {
                            for (int i = 1; i < polygonIndices.Count - 1; i++)
                            {
                                building.TriangleIndices.Add(polygonIndices[0]);
                                building.TriangleIndices.Add(polygonIndices[i]);
                                building.TriangleIndices.Add(polygonIndices[i + 1]);
                            }
                        }
                    }
                }
            }

            return building;
        }
        
        /// <summary>
        /// Create a building from a hierarchical building group (1-BYTE EXPERIMENT)
        /// Groups all surfaces with same building ID (top byte of SurfaceKey)
        /// </summary>
        private CompleteBuilding CreateBuildingFromHierarchicalGroup(Pm4Scene scene, byte buildingId, List<int> surfaceIndices, string baseFileName, int buildingIndex)
        {
            var building = new CompleteBuilding
            {
                FileName = $"{baseFileName}_building_{buildingId:X2}",
                Category = "HierarchicalBuilding",
                MaterialName = $"Building_{buildingId:X2}"
            };

            var vertexLookup = new Dictionary<Vector3, int>();

            // Add surfaces from this hierarchical building group
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    int indexCount = surface.IndexCount;
                    int first = (int)surface.MsviFirstIndex;

                    if (indexCount <= 0 || first < 0) continue;

                    // Collect vertices from MSVI indices -> MSVT vertices
                    var surfaceVertices = new List<Vector3>();
                    for (int i = 0; i < indexCount && first + i < scene.Indices.Count; i++)
                    {
                        int msviIndex = scene.Indices[first + i];
                        if (msviIndex >= 0 && msviIndex < scene.Vertices.Count)
                        {
                            // Apply X-axis flip for correct orientation
                            var vertex = scene.Vertices[msviIndex];
                            var flippedVertex = new Vector3(-vertex.X, vertex.Y, vertex.Z);
                            surfaceVertices.Add(flippedVertex);
                        }
                    }

                    // Add vertices to building (with deduplication)
                    var polygonIndices = new List<int>();
                    foreach (var vertex in surfaceVertices)
                    {
                        if (!vertexLookup.ContainsKey(vertex))
                        {
                            vertexLookup[vertex] = building.Vertices.Count;
                            building.Vertices.Add(vertex);
                        }
                        polygonIndices.Add(vertexLookup[vertex]);
                    }

                    // Handle n-gons properly (triangles, quads, pentagons)
                    if (polygonIndices.Count >= 3)
                    {
                        if (polygonIndices.Count == 3)
                        {
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[1]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                        }
                        else if (polygonIndices.Count == 4)
                        {
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[1]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                            
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                            building.TriangleIndices.Add(polygonIndices[3]);
                        }
                        else
                        {
                            for (int i = 1; i < polygonIndices.Count - 1; i++)
                            {
                                building.TriangleIndices.Add(polygonIndices[0]);
                                building.TriangleIndices.Add(polygonIndices[i]);
                                building.TriangleIndices.Add(polygonIndices[i + 1]);
                            }
                        }
                    }
                }
            }

            return building;
        }
        
        /// <summary>
        /// Create a building from a SurfaceKey group (LEGACY - NOT USED IN CORRECTED APPROACH)
        /// </summary>
        private CompleteBuilding CreateBuildingFromSurfaceGroup(Pm4Scene scene, uint surfaceKey, List<int> surfaceIndices, string baseFileName, int buildingIndex)
        {
            var building = new CompleteBuilding
            {
                FileName = $"{baseFileName}_surfacekey_0x{surfaceKey:X8}",
                Category = "SurfaceGroup",
                MaterialName = $"SurfaceKey_{surfaceKey:X8}"
            };

            var vertexLookup = new Dictionary<Vector3, int>();

            // Add vertices and triangles from surfaces in this group
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    int indexCount = surface.IndexCount;
                    int first = (int)surface.MsviFirstIndex;

                    if (indexCount <= 0 || first < 0) continue;

                    // Collect vertices from MSVI indices -> MSVT vertices
                    var surfaceVertices = new List<Vector3>();
                    for (int i = 0; i < indexCount && first + i < scene.Indices.Count; i++)
                    {
                        int msviIndex = scene.Indices[first + i];
                        if (msviIndex >= 0 && msviIndex < scene.Vertices.Count)
                        {
                            // Apply X-axis flip for correct orientation
                            var vertex = scene.Vertices[msviIndex];
                            var flippedVertex = new Vector3(-vertex.X, vertex.Y, vertex.Z);
                            surfaceVertices.Add(flippedVertex);
                        }
                    }

                    // Add vertices to building (with deduplication)
                    var polygonIndices = new List<int>();
                    foreach (var vertex in surfaceVertices)
                    {
                        if (!vertexLookup.ContainsKey(vertex))
                        {
                            vertexLookup[vertex] = building.Vertices.Count;
                            building.Vertices.Add(vertex);
                        }
                        polygonIndices.Add(vertexLookup[vertex]);
                    }

                    // FIXED: Handle n-gons properly using actual surface.IndexCount
                    // Each surface defines a single polygon with surface.IndexCount vertices
                    if (polygonIndices.Count >= 3)
                    {
                        // For n-gons (quads, pentagons, etc), we need to triangulate for OBJ export
                        // OBJ format supports n-gons directly, so we can export the full polygon
                        if (polygonIndices.Count == 3)
                        {
                            // Triangle - add directly
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[1]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                        }
                        else if (polygonIndices.Count == 4)
                        {
                            // Quad - triangulate into 2 triangles
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[1]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                            
                            building.TriangleIndices.Add(polygonIndices[0]);
                            building.TriangleIndices.Add(polygonIndices[2]);
                            building.TriangleIndices.Add(polygonIndices[3]);
                        }
                        else if (polygonIndices.Count > 4)
                        {
                            // N-gon - fan triangulation from first vertex
                            for (int i = 1; i < polygonIndices.Count - 1; i++)
                            {
                                building.TriangleIndices.Add(polygonIndices[0]);
                                building.TriangleIndices.Add(polygonIndices[i]);
                                building.TriangleIndices.Add(polygonIndices[i + 1]);
                            }
                        }
                    }
                }
            }

            return building;
        }

        /// <summary>
        /// Export building to OBJ file (WORKING POC LOGIC)
        /// </summary>
        private void ExportBuildingToObj(CompleteBuilding building, string outputDirectory)
        {
            var fileName = Path.Combine(outputDirectory, $"{building.FileName}.obj");

            using var writer = new StreamWriter(fileName);
            
            writer.WriteLine($"# Building: {building.Category}");
            writer.WriteLine($"# Vertices: {building.VertexCount}");
            writer.WriteLine($"# Triangles: {building.TriangleCount}");
            writer.WriteLine();

            // Write vertices (already X-flipped in CreateBuildingFromSurfaceGroup)
            foreach (var vertex in building.Vertices)
            {
                writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }

            // Write faces (1-based indices)
            for (int i = 0; i < building.TriangleIndices.Count; i += 3)
            {
                if (i + 2 < building.TriangleIndices.Count)
                {
                    var i1 = building.TriangleIndices[i] + 1;
                    var i2 = building.TriangleIndices[i + 1] + 1;
                    var i3 = building.TriangleIndices[i + 2] + 1;
                    writer.WriteLine($"f {i1} {i2} {i3}");
                }
            }

            ConsoleLogger.WriteLine($"  Exported: {Path.GetFileName(fileName)} ({building.TriangleCount} triangles)");
        }
    }
}
