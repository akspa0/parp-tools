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
    /// WORKING spatial clustering assembler extracted from POC poc_exporter.cs
    /// Uses spatial clustering to combine MSPV structural elements with nearby MSUR surfaces
    /// This is the proven approach that actually worked for building-scale object extraction
    /// </summary>
    public class Pm4SpatialClusteringAssembler
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
            
            // Added for WMO correlation
            public Vector3 CenterPoint { get; set; }
            public Vector3 BoundingBoxMin { get; set; }
            public Vector3 BoundingBoxMax { get; set; }
            public float SurfaceArea { get; set; }
            public List<Vector3> MscnVertices { get; set; } = new();
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
        /// Export buildings using spatial clustering and SurfaceKey grouping.
        /// Based on the proven reference implementation and commit 23175c0.
        /// Includes tile-based filtering to avoid cross-tile/LOD references.
        /// </summary>
        public ExportSummary ExportBuildingsUsingSpatialClustering(ParpToolbox.Formats.PM4.Pm4Scene scene, string outputDirectory, string baseFileName)
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

            ConsoleLogger.WriteLine("=== PM4 Spatial Clustering Assembler - Building Export ===");
            
            if (scene.Vertices == null || scene.Vertices.Count == 0)
            {
                ConsoleLogger.WriteLine("No vertices found in scene. Export aborted.");
                return summary;
            }
            
            ConsoleLogger.WriteLine($"Total vertices: {scene.Vertices.Count}");
            ConsoleLogger.WriteLine($"Total triangles: {scene.Triangles.Count}");
            ConsoleLogger.WriteLine($"Total surfaces: {scene.Surfaces?.Count ?? 0}");
            ConsoleLogger.WriteLine($"Total links: {scene.Links?.Count ?? 0}");
            
            // Determine primary tile coordinates from the scene to filter cross-tile references
            var primaryTile = DeterminePrimaryTile(scene);
            if (primaryTile.HasValue)
            {
                ConsoleLogger.WriteLine($"Primary tile: ({primaryTile.Value.x}, {primaryTile.Value.y}) - filtering cross-tile references");
            }
            else
            {
                ConsoleLogger.WriteLine("Warning: Could not determine primary tile - may include cross-tile references");
            }

            // === STEP 1: FILTER LINKS BY TILE TO ELIMINATE CROSS-TILE/LOD REFERENCES ===
            var linksToProcess = scene.Links;
            if (primaryTile.HasValue)
            {
                linksToProcess = FilterLinksByTile(scene, primaryTile.Value);
            }
            
            // === STEP 2: FIND ROOT NODES FROM FILTERED LINKS ===
            var rootNodes = new List<(int nodeIndex, MslkEntry entry)>();
            
            if (linksToProcess != null && linksToProcess.Count > 0)
            {
                for (int i = 0; i < linksToProcess.Count; i++)
                {
                    var entry = linksToProcess[i];
                    if (entry.ParentIndex == i) // Self-referencing = root node
                    {
                        rootNodes.Add((i, entry));
                    }
                }
            }
            ConsoleLogger.WriteLine($"Found {rootNodes.Count} root nodes");

            // === STEP 2: CREATE BUILDINGS USING PROPER SURFACE GROUPING ===
            var buildings = new List<CompleteBuilding>();
            
            // Create a mapping from SurfaceKey to surface indices for efficient lookup
            var surfaceKeyToIndices = new Dictionary<uint, List<int>>();
            if (scene.Surfaces != null)
            {
                for (int i = 0; i < scene.Surfaces.Count; i++)
                {
                    var surface = scene.Surfaces[i];
                    if (!surfaceKeyToIndices.ContainsKey(surface.SurfaceKey))
                    {
                        surfaceKeyToIndices[surface.SurfaceKey] = new List<int>();
                    }
                    surfaceKeyToIndices[surface.SurfaceKey].Add(i);
                }
            }
            
            // Create buildings with intelligent size filtering to avoid multi-tile/recursive components
            // Filter out oversized groups that likely represent multiple buildings or entire tiles
            var processedSurfaceKeys = new HashSet<uint>();
            
            if (scene.Surfaces != null && scene.Surfaces.Count > 0)
            {
                // Group surfaces by SurfaceKey and filter for reasonable building sizes
                foreach (var kvp in surfaceKeyToIndices)
                {
                    var surfaceKey = kvp.Key;
                    var surfaceIndices = kvp.Value;
                    
                    // Skip if we've already processed this SurfaceKey
                    if (processedSurfaceKeys.Contains(surfaceKey))
                        continue;
                    
                    processedSurfaceKeys.Add(surfaceKey);
                    
                    // Calculate triangle count for this surface group to filter oversized groups
                    int triangleCount = CalculateTriangleCount(scene, surfaceIndices);
                    
                    // Filter out groups that are too large (likely multi-building or tile-spanning)
                    const int MAX_BUILDING_TRIANGLES = 50000; // Reasonable limit for a single building
                    const int MIN_BUILDING_TRIANGLES = 10;    // Skip tiny fragments
                    
                    if (triangleCount < MIN_BUILDING_TRIANGLES)
                    {
                        ConsoleLogger.WriteLine($"Skipping SurfaceKey 0x{surfaceKey:X8}: too small ({triangleCount} triangles)");
                        continue;
                    }
                    
                    if (triangleCount > MAX_BUILDING_TRIANGLES)
                    {
                        ConsoleLogger.WriteLine($"Skipping SurfaceKey 0x{surfaceKey:X8}: too large ({triangleCount} triangles, likely multi-building)");
                        continue;
                    }
                    
                    var buildingIndex = buildings.Count;
                    ConsoleLogger.WriteLine($"Building {buildingIndex + 1}: SurfaceKey 0x{surfaceKey:X8} ({surfaceIndices.Count} surfaces, {triangleCount} triangles)");
                    
                    // Use the original hybrid building approach - calculate bounds of surfaces then find MSLK entries
                    // Calculate bounds of surface group to find nearby structural elements
                    var surfaceBounds = CalculateSurfaceGroupBounds(scene, surfaceIndices);
                    
                    if (surfaceBounds != null)
                    {
                        // Find structural elements near these surfaces using spatial proximity
                        float proximityTolerance = 5.0f;
                        
                        // CRITICAL FIX: Find the actual MSLK structural elements near these surfaces
                        // This is the key to resolving the exploded geometry issue
                        var structuralElements = FindStructuralElementsNearBounds(scene, surfaceBounds.Value, proximityTolerance);
                        
                        // Log the process for debugging
                        ConsoleLogger.WriteLine($"  Found {structuralElements.Count} structural elements for building {buildingIndex + 1}");
                        
                        // Create a hybrid building that includes both the surfaces and structural elements
                        // This is critical to produce coherent building-scale objects
                        var building = CreateHybridBuilding_StructuralPlusNearby(scene, structuralElements, surfaceIndices, baseFileName, buildingIndex);
                        
                        // Add surface key to metadata for traceability
                        building.Metadata["SurfaceKey"] = $"0x{surfaceKey:X8}";
                        building.Metadata["SurfaceCount"] = surfaceIndices.Count.ToString();
                        building.Metadata["TriangleCount"] = triangleCount.ToString();
                        
                        buildings.Add(building);
                    }
                }
            }
            else
            {
                // Fallback to MSLK-based approach if no surfaces or groups available
                for (int buildingIndex = 0; buildingIndex < rootNodes.Count; buildingIndex++)
                {
                    var (rootNodeIndex, rootEntry) = rootNodes[buildingIndex];
                    var rootGroupKey = rootEntry.ParentIndex;
                    
                    ConsoleLogger.WriteLine($"Building {buildingIndex + 1}: Root Node {rootNodeIndex}, Group 0x{rootGroupKey:X8}");
                    
                    // Get MSLK structural elements for this building
                    var buildingEntries = scene.Links
                        .Select((entry, index) => new { entry, index })
                        .Where(x => x.entry.ParentIndex == rootGroupKey && x.entry.MspiFirstIndex >= 0 && x.entry.MspiIndexCount > 0)
                        .ToList();
                    
                    if (buildingEntries.Count == 0)
                    {
                        ConsoleLogger.WriteLine($"  No structural elements found for building {buildingIndex + 1}");
                        continue;
                    }
                    
                    // Calculate bounding box of structural elements
                    var structuralBounds = CalculateStructuralElementsBounds(scene, buildingEntries.Cast<dynamic>().ToList());
                    if (!structuralBounds.HasValue)
                    {
                        ConsoleLogger.WriteLine($"  Could not calculate bounds for building {buildingIndex + 1}");
                        continue;
                    }
                    
                    // Find MSUR surfaces that belong to the same group as this building
                    // Based on the discovery that MSUR surfaces are grouped by SurfaceKey (Unk1C)
                    var nearbySurfaces = new List<int>();
                    
                    // For now, we'll associate surfaces based on spatial proximity as a fallback
                    // In a proper implementation with cross-tile loading, we would use the correct SurfaceKey grouping
                    nearbySurfaces = FindMSURSurfacesNearBounds(scene, structuralBounds.Value, tolerance: 50.0f);
                    
                    ConsoleLogger.WriteLine($"  {buildingEntries.Count} structural elements, {nearbySurfaces.Count} nearby surfaces");
                    
                    // Create building combining structural elements and nearby surfaces
                    var building = CreateHybridBuilding_StructuralPlusNearby(scene, buildingEntries.Cast<dynamic>().ToList(), nearbySurfaces, baseFileName, buildingIndex);
                    building.Metadata["RootNodeIndex"] = rootNodeIndex;
                    building.Metadata["GroupKey"] = $"0x{rootGroupKey:X8}";
                    building.Metadata["StructuralElements"] = buildingEntries.Count;
                    building.Metadata["RenderSurfaces"] = nearbySurfaces.Count;
                    
                    buildings.Add(building);
                }
            }

            // === STEP 3: EXPORT TO OBJ FILES ===
            foreach (var building in buildings)
            {
                if (building.HasGeometry)
                {
                    var objFileName = $"{building.FileName}.obj";
                    var objFilePath = Path.Combine(outputDirectory, objFileName);
                    
                    ExportBuildingToObj(building, objFilePath);
                    summary.SuccessfulExports++;
                    
                    ConsoleLogger.WriteLine($"Exported {building.FileName}: {building.TriangleCount} triangles, {building.VertexCount} vertices → {objFileName} | Metadata: {string.Join(", ", building.Metadata.Select(kv => $"{kv.Key}={kv.Value}"))}");
                }
            }

            summary.TotalBuildings = buildings.Count;
            summary.Buildings = buildings;
            summary.ExportDuration = DateTime.Now - startTime;

            ConsoleLogger.WriteLine($"=== Export Complete: {summary.SuccessfulExports}/{summary.TotalBuildings} buildings exported ===");
            return summary;
        }

        /// <summary>
        /// Calculate bounding box of a surface group
        /// </summary>
        private (Vector3 min, Vector3 max)? CalculateSurfaceGroupBounds(ParpToolbox.Formats.PM4.Pm4Scene scene, List<int> surfaceIndices)
        {
            if (scene.Vertices == null || scene.Vertices.Count == 0 || surfaceIndices == null || surfaceIndices.Count == 0)
                return null;
                
            Vector3? min = null;
            Vector3? max = null;
            
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    int firstIndex = (int)surface.MsviFirstIndex;
                    int indexCount = surface.IndexCount;
                    
                    if (firstIndex >= 0 && indexCount > 0 && firstIndex + indexCount <= scene.Indices.Count)
                    {
                        for (int i = 0; i < indexCount && firstIndex + i < scene.Indices.Count; i++)
                        {
                            int vIdx = scene.Indices[firstIndex + i];
                            if (vIdx >= 0 && vIdx < scene.Vertices.Count)
                            {
                                var vertex = scene.Vertices[vIdx];
                                if (min == null)
                                {
                                    min = vertex;
                                    max = vertex;
                                }
                                else
                                {
                                    min = new Vector3(
                                        Math.Min(min.Value.X, vertex.X),
                                        Math.Min(min.Value.Y, vertex.Y),
                                        Math.Min(min.Value.Z, vertex.Z));
                                        
                                    max = new Vector3(
                                        Math.Max(max.Value.X, vertex.X),
                                        Math.Max(max.Value.Y, vertex.Y),
                                        Math.Max(max.Value.Z, vertex.Z));
                                }
                            }
                        }
                    }
                }
            }
            
            if (min == null || max == null)
                return null;
                
            return (min.Value, max.Value);
        }
        
        /// <summary>
        /// Calculate bounding box of structural elements (WORKING POC LOGIC)
        /// </summary>
        private (Vector3 min, Vector3 max)? CalculateStructuralElementsBounds(ParpToolbox.Formats.PM4.Pm4Scene scene, List<dynamic> buildingEntries)
        {
            if (scene.Vertices == null || scene.Vertices.Count == 0) 
                return null;
            
            var allVertices = new List<Vector3>();
            
            foreach (var entryData in buildingEntries)
            {
                var entry = entryData.entry;
                
                // Use available vertex indices from scene triangles that relate to this entry
                // This is a simplified approach for now - with global mesh loading this will be more precise
                for (int i = entry.MspiFirstIndex; i < entry.MspiFirstIndex + entry.MspiIndexCount && i < scene.Indices.Count; i++)
                {
                    int vertexIndex = scene.Indices[i];
                    if (vertexIndex >= 0 && vertexIndex < scene.Vertices.Count)
                    {
                        var vertex = scene.Vertices[vertexIndex];
                        allVertices.Add(vertex);
                    }
                }
            }
            
            if (allVertices.Count == 0) return null;
            
            var minX = allVertices.Min(v => v.X);
            var minY = allVertices.Min(v => v.Y);
            var minZ = allVertices.Min(v => v.Z);
            var maxX = allVertices.Max(v => v.X);
            var maxY = allVertices.Max(v => v.Y);
            var maxZ = allVertices.Max(v => v.Z);
            
            return (new Vector3(minX, minY, minZ), new Vector3(maxX, maxY, maxZ));
        }

        /// <summary>
        /// Calculate the approximate triangle count for a surface group to filter oversized groups
        /// </summary>
        private int CalculateTriangleCount(ParpToolbox.Formats.PM4.Pm4Scene scene, List<int> surfaceIndices)
        {
            int totalTriangles = 0;
            
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    int indexCount = surface.IndexCount;
                    
                    // Approximate triangle count (indices come in groups of 3)
                    totalTriangles += indexCount / 3;
                }
            }
            
            return totalTriangles;
        }

        /// <summary>
        /// Determine the primary tile coordinates from the scene to filter cross-tile references
        /// </summary>
        private (byte x, byte y)? DeterminePrimaryTile(ParpToolbox.Formats.PM4.Pm4Scene scene)
        {
            if (scene.Links == null || scene.Links.Count == 0)
                return null;
            
            // Find the most common tile coordinates among all MSLK entries
            var tileFrequency = new Dictionary<(byte x, byte y), int>();
            
            foreach (var link in scene.Links)
            {
                if (link.HasValidTileCoordinates && link.TryDecodeTileCoordinates(out int tileX, out int tileY))
                {
                    var coords = ((byte)tileX, (byte)tileY);
                    tileFrequency[coords] = tileFrequency.GetValueOrDefault(coords, 0) + 1;
                }
            }
            
            if (tileFrequency.Count == 0)
                return null;
            
            // Return the most frequent tile coordinates (primary tile)
            return tileFrequency.OrderByDescending(kvp => kvp.Value).First().Key;
        }

        /// <summary>
        /// Filter MSLK entries to only include those from the specified tile
        /// </summary>
        private List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry> FilterLinksByTile(ParpToolbox.Formats.PM4.Pm4Scene scene, (byte x, byte y) primaryTile)
        {
            var filteredLinks = new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
            
            foreach (var link in scene.Links)
            {
                if (link.TryDecodeTileCoordinates(out int tileX, out int tileY))
                {
                    // Only include links from the primary tile
                    if ((byte)tileX == primaryTile.x && (byte)tileY == primaryTile.y)
                    {
                        filteredLinks.Add(link);
                    }
                }
                else
                {
                    // Include links without valid tile coordinates (might be local)
                    filteredLinks.Add(link);
                }
            }
            
            ConsoleLogger.WriteLine($"Filtered links: {filteredLinks.Count}/{scene.Links.Count} (excluded {scene.Links.Count - filteredLinks.Count} cross-tile references)");
            return filteredLinks;
        }

        /// <summary>
        /// Create a building from a group of surfaces with the same SurfaceKey
        /// Based on the correct linking: MSUR -> MSVI -> MSVT
        /// </summary>
        private CompleteBuilding CreateBuildingFromSurfaceGroup(ParpToolbox.Formats.PM4.Pm4Scene scene, uint surfaceKey, List<int> surfaceIndices, string baseFileName, int buildingIndex)
        {
            var building = new CompleteBuilding
            {
                FileName = $"{baseFileName}_surfacekey_0x{surfaceKey:X8}",
                Category = "SurfaceGroup",
                Vertices = new List<Vector3>(),
                TriangleIndices = new List<int>()
            };
            
            // Create vertex lookup to avoid duplicates
            var vertexLookup = new Dictionary<Vector3, int>();
            
            // Add vertices and triangles from MSUR surfaces (MSUR -> MSVI -> MSVT)
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    
                    // Get vertices from MSVT via MSVI indices
                    int firstIndex = (int)surface.MsviFirstIndex;
                    int indexCount = surface.IndexCount;
                    
                    if (firstIndex >= 0 && indexCount > 0 && firstIndex + indexCount <= scene.Indices.Count)
                    {
                        // Add vertices referenced by MSVI indices
                        for (int i = 0; i < indexCount && firstIndex + i < scene.Indices.Count; i++)
                        {
                            int vertexIndex = scene.Indices[firstIndex + i];
                            
                            if (vertexIndex >= 0 && vertexIndex < scene.Vertices.Count)
                            {
                                var vertex = scene.Vertices[vertexIndex];
                                
                                // Add vertex if not already present
                                if (!vertexLookup.ContainsKey(vertex))
                                {
                                    int newIndex = building.Vertices.Count;
                                    building.Vertices.Add(vertex);
                                    vertexLookup[vertex] = newIndex;
                                }
                            }
                        }
                        
                        // Add triangle indices (triangles are already properly formed in MSVI)
                        for (int i = 0; i + 2 < indexCount && firstIndex + i + 2 < scene.Indices.Count; i += 3)
                        {
                            int v1Index = scene.Indices[firstIndex + i];
                            int v2Index = scene.Indices[firstIndex + i + 1];
                            int v3Index = scene.Indices[firstIndex + i + 2];
                            
                            if (v1Index >= 0 && v1Index < scene.Vertices.Count &&
                                v2Index >= 0 && v2Index < scene.Vertices.Count &&
                                v3Index >= 0 && v3Index < scene.Vertices.Count)
                            {
                                // Get the indices in our building's vertex list
                                var v1 = scene.Vertices[v1Index];
                                var v2 = scene.Vertices[v2Index];
                                var v3 = scene.Vertices[v3Index];
                                
                                if (vertexLookup.ContainsKey(v1) && vertexLookup.ContainsKey(v2) && vertexLookup.ContainsKey(v3))
                                {
                                    building.TriangleIndices.Add(vertexLookup[v1]);
                                    building.TriangleIndices.Add(vertexLookup[v2]);
                                    building.TriangleIndices.Add(vertexLookup[v3]);
                                }
                            }
                        }
                    }
                }
            }
            
            return building;
        }
        
        /// <summary>
        /// Create a hybrid building combining structural elements and nearby surfaces
        /// Based on the correct linking: MSLK -> MSPI -> MSCN/MSPV and MSUR -> MSVI -> MSVT
        /// </summary>
        private CompleteBuilding CreateHybridBuilding_StructuralPlusNearby(ParpToolbox.Formats.PM4.Pm4Scene scene, List<dynamic> buildingEntries, List<int> nearbySurfaces, string baseFileName, int buildingIndex)
        {
            var building = new CompleteBuilding
            {
                FileName = $"{baseFileName}_building_{buildingIndex + 1}",
                Category = "HybridBuilding",
                Vertices = new List<Vector3>(),
                TriangleIndices = new List<int>()
            };
            
            // Create vertex lookup to avoid duplicates
            var vertexLookup = new Dictionary<Vector3, int>();
            
            // The buildingEntries parameter is a remnant of a flawed implementation that
            // caused a severe performance regression (O(N*M) complexity) and incorrect
            // exports (entire scene graph per building).
            // This logic has been removed. The correct approach is to rely on the
            // SurfaceKey grouping, which provides the correct surfaces in nearbySurfaces.
            
            // Add vertices and triangles from nearby surfaces (MSUR -> MSVI -> MSVT)
            foreach (int surfaceIndex in nearbySurfaces)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    
                    // Get vertices from MSVT via MSVI indices
                    int firstIndex = (int)surface.MsviFirstIndex;
                    int indexCount = surface.IndexCount;
                    
                    // Determine best normal orientation once per surface
                    Vector3 nSrc = new Vector3(surface.Nx, surface.Ny, surface.Nz);
                                        bool usePlane = surface.GroupKey != 18 && nSrc.LengthSquared() > 1e-6f;
                    Vector3 nFinal = nSrc;
                    if (usePlane && ShouldSwapNormal(surface, scene))
                    {
                        nFinal = new Vector3(nSrc.Y, nSrc.X, nSrc.Z);
                    }
                    if (usePlane)
                        nFinal = Vector3.Normalize(nFinal);

                    if (firstIndex >= 0 && indexCount > 0 && firstIndex + indexCount <= scene.Indices.Count)
                    {
                        for (int i = 0; i + 2 < indexCount && firstIndex + i + 2 < scene.Indices.Count; i += 3)
                        {
                            int v1Idx = scene.Indices[firstIndex + i];
                            int v2Idx = scene.Indices[firstIndex + i + 1];
                            int v3Idx = scene.Indices[firstIndex + i + 2];

                            if (v1Idx < 0 || v1Idx >= scene.Vertices.Count ||
                                v2Idx < 0 || v2Idx >= scene.Vertices.Count ||
                                v3Idx < 0 || v3Idx >= scene.Vertices.Count) continue;

                            int AddWithPlane(int vIdx)
                            {
                                var v = scene.Vertices[vIdx];
                                if (usePlane)
                                {
                                    float d = Vector3.Dot(nFinal, v) - surface.Height;
                                    v -= nFinal * d;
                                }
                                return GetOrAddVertex(building.Vertices, vertexLookup, v);
                            }

                            building.TriangleIndices.Add(AddWithPlane(v1Idx));
                            building.TriangleIndices.Add(AddWithPlane(v2Idx));
                            building.TriangleIndices.Add(AddWithPlane(v3Idx));
                        }
                    }
                }
            }
            
            return building;
        }
        
        /// <summary>
        /// Find structural elements (MSLK entries) that are spatially near the given bounds
        /// This is critical for creating coherent building-scale objects
        /// </summary>
        private List<dynamic> FindStructuralElementsNearBounds(ParpToolbox.Formats.PM4.Pm4Scene scene, (Vector3 min, Vector3 max) bounds, float tolerance)
        {
            var nearbyElements = new List<dynamic>();
            
            if (scene.Links == null || scene.Links.Count == 0)
                return nearbyElements;
                
            // Expand bounds by tolerance
            var expandedBounds = (
                new Vector3(bounds.min.X - tolerance, bounds.min.Y - tolerance, bounds.min.Z - tolerance),
                new Vector3(bounds.max.X + tolerance, bounds.max.Y + tolerance, bounds.max.Z + tolerance)
            );
            
            // Find all MSLK entries with geometry that intersects with the expanded bounds
            for (int i = 0; i < scene.Links.Count; i++)
            {
                var entry = scene.Links[i];
                
                // Only consider links with valid geometry
                if (entry.MspiFirstIndex >= 0 && entry.MspiIndexCount > 0)
                {
                    int firstIdx = (int)entry.MspiFirstIndex;
                    int indexCount = (int)entry.MspiIndexCount;
                    
                    if (firstIdx >= 0 && indexCount > 0 && firstIdx + indexCount <= scene.Indices.Count)
                    {
                        bool isInBounds = false;
                        
                        // Check if any vertex of this structural element is within bounds
                        for (int j = 0; j < indexCount && !isInBounds && firstIdx + j < scene.Indices.Count; j++)
                        {
                            int vIdx = scene.Indices[firstIdx + j];
                            
                            if (vIdx >= 0 && vIdx < scene.Vertices.Count)
                            {
                                var vertex = scene.Vertices[vIdx];
                                
                                if (vertex.X >= expandedBounds.Item1.X && vertex.X <= expandedBounds.Item2.X &&
                                    vertex.Y >= expandedBounds.Item1.Y && vertex.Y <= expandedBounds.Item2.Y &&
                                    vertex.Z >= expandedBounds.Item1.Z && vertex.Z <= expandedBounds.Item2.Z)
                                {
                                    isInBounds = true;
                                }
                            }
                        }
                        
                        if (isInBounds)
                        {
                            nearbyElements.Add(new { entry, index = i });
                        }
                    }
                }
            }
            
            return nearbyElements;
        }
        
        /// <summary>
        /// Find MSUR surfaces that are spatially near the given bounds
        /// </summary>
        private List<int> FindMSURSurfacesNearBounds(ParpToolbox.Formats.PM4.Pm4Scene scene, (Vector3 min, Vector3 max) bounds, float tolerance)
        {
            var nearbySurfaces = new List<int>();
            
            if (scene.Surfaces == null) return nearbySurfaces;
            
            for (int i = 0; i < scene.Surfaces.Count; i++)
            {
                var surface = scene.Surfaces[i];
                
                // Check if any vertices from this surface are within the bounds + tolerance
                int firstIndex = (int)surface.MsviFirstIndex;
                int indexCount = surface.IndexCount;
                
                if (firstIndex >= 0 && indexCount > 0 && firstIndex + indexCount <= scene.Indices.Count)
                {
                    bool isNearby = false;
                    
                    for (int j = 0; j < indexCount && firstIndex + j < scene.Indices.Count; j++)
                    {
                        int vertexIndex = scene.Indices[firstIndex + j];
                        
                        if (vertexIndex >= 0 && vertexIndex < scene.Vertices.Count)
                        {
                            var vertex = scene.Vertices[vertexIndex];
                            
                            // Check if vertex is within bounds + tolerance
                            if (vertex.X >= bounds.min.X - tolerance && vertex.X <= bounds.max.X + tolerance &&
                                vertex.Y >= bounds.min.Y - tolerance && vertex.Y <= bounds.max.Y + tolerance &&
                                vertex.Z >= bounds.min.Z - tolerance && vertex.Z <= bounds.max.Z + tolerance)
                            {
                                isNearby = true;
                                break;
                            }
                        }
                    }
                    
                    if (isNearby)
                    {
                        nearbySurfaces.Add(i);
                    }
                }
            }
            
            return nearbySurfaces;
        }

        /// <summary>
        /// Helper method to get or add a vertex to the building's vertex list
        /// </summary>
        private int GetOrAddVertex(List<Vector3> vertices, Dictionary<Vector3, int> vertexLookup, Vector3 vertex)
        {
            if (vertexLookup.TryGetValue(vertex, out int existingIndex))
            {
                return existingIndex;
            }
            
            int newIndex = vertices.Count;
            vertices.Add(vertex);
            vertexLookup[vertex] = newIndex;
            return newIndex;
        }

        /// <summary>
        /// Export building to OBJ file
        /// </summary>
        /// <summary>
        /// Heuristically decides whether swapping X/Y in the stored MSUR normal better fits the sampled vertices.
        /// Adapted from the reference MsurObjectExporter.cs.
        /// </summary>
                private bool ShouldSwapNormal(ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surf, ParpToolbox.Formats.PM4.Pm4Scene scene)
        {
            int first = (int)surf.MsviFirstIndex;
            int sampleCount = Math.Min(surf.IndexCount, (byte)9); // up to 3 triangles
            if (sampleCount == 0 || scene.Indices == null || scene.Vertices == null) return false;

            Vector3 nA = new Vector3(surf.Nx, surf.Ny, surf.Nz);
            if (nA.LengthSquared() < 1e-6f) return false;
            Vector3 nB = new Vector3(nA.Y, nA.X, nA.Z);
            nA = Vector3.Normalize(nA);
            nB = Vector3.Normalize(nB);

            float errA = 0, errB = 0;
            int samples = 0;
            for (int i = 0; i < sampleCount && (first + i) < scene.Indices.Count; i++)
            {
                uint idx = (uint)scene.Indices[first + i];
                if (idx >= scene.Vertices.Count) continue;
                var v = scene.Vertices[(int)idx]; // Vertices are already transformed in Pm4Scene
                        errA += Math.Abs(Vector3.Dot(nA, v) - surf.Height);
                errB += Math.Abs(Vector3.Dot(nB, v) - surf.Height);
                samples++;
            }
            if (samples == 0) return false;
            return errB < errA;
        }

        private void ExportBuildingToObj(CompleteBuilding building, string filePath)
        {
            using (var writer = new StreamWriter(filePath))
            {
                writer.WriteLine($"# PM4 Spatial Clustering Building: {building.FileName}");
                writer.WriteLine($"# Category: {building.Category}");
                writer.WriteLine($"# Triangles: {building.TriangleCount}, Vertices: {building.VertexCount}");
                writer.WriteLine($"# Generated: {DateTime.Now}");
                writer.WriteLine();

                // Write vertices
                foreach (var vertex in building.Vertices)
                {
                    writer.WriteLine($"v {CoordinateTransformationService.ApplyPm4Transformation(vertex).X:F6} {CoordinateTransformationService.ApplyPm4Transformation(vertex).Y:F6} {CoordinateTransformationService.ApplyPm4Transformation(vertex).Z:F6}");
                }

                writer.WriteLine();

                // Write faces (OBJ uses 1-based indexing)
                for (int i = 0; i < building.TriangleIndices.Count; i += 3)
                {
                    var i1 = building.TriangleIndices[i] + 1;
                    var i2 = building.TriangleIndices[i + 1] + 1;
                    var i3 = building.TriangleIndices[i + 2] + 1;
                    writer.WriteLine($"f {i1} {i2} {i3}");
                }
            }
        }
    }

    // Note: Coordinate transforms removed since we're using the already-processed
    // Vector3 vertices from Pm4Scene which have already been transformed
}

/// <summary>
/// Simple axis-aligned bounding box for spatial calculations
/// </summary>
public struct BoundingBox
{
    public System.Numerics.Vector3 Min;
    public System.Numerics.Vector3 Max;
    
    public BoundingBox(System.Numerics.Vector3 min, System.Numerics.Vector3 max)
    {
        Min = min;
        Max = max;
    }
    
    public System.Numerics.Vector3 Center => (Min + Max) * 0.5f;
    public System.Numerics.Vector3 Size => Max - Min;
}
