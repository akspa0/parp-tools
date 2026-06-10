using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder;

/// <summary>
/// Hierarchical PM4 to OBJ exporter that properly groups objects by MPRL.Unknown4/MSLK.ParentIndex.
/// Uses the correct grouping logic to assemble building-scale objects with complete geometry.
/// </summary>
public static class Pm4HierarchicalExporter
{
    /// <summary>
    /// Export complete buildings directly from PM4 files to OBJ format using hierarchical grouping.
    /// Uses MPRL.Unknown4 as the primary building identifier and aggregates all MSLK entries with matching ParentIndex.
    /// </summary>
    /// <param name="pm4Path">Path to PM4 file or directory containing PM4 files</param>
    /// <param name="outputDir">Output directory for OBJ files</param>
    /// <returns>0 on success, 1 on error</returns>
    public static int ExportBuildings(string pm4Path, string outputDir)
    {
        try
        {
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Starting hierarchical PM4 → OBJ export from: {pm4Path}");
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Output directory: {outputDir}");
            
            Directory.CreateDirectory(outputDir);
            
            // Load PM4 scene directly
            var scene = LoadPm4Scene(pm4Path);
            if (scene == null)
            {
                Console.WriteLine("[HIERARCHICAL EXPORTER ERROR] Failed to load PM4 scene");
                return 1;
            }
            
            // Extract buildings using correct hierarchical structure
            var buildings = ExtractBuildingsFromScene(scene);
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Found {buildings.Count} buildings using hierarchical grouping");
            
            // Export each building as separate OBJ file
            int successCount = 0;
            foreach (var building in buildings)
            {
                try
                {
                    var objPath = ExportBuildingToObj(building, outputDir);
                    Console.WriteLine($"[HIERARCHICAL EXPORTER] Exported building {building.BuildingId} → {Path.GetFileName(objPath)} ({building.TriangleCount} triangles)");
                    successCount++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[HIERARCHICAL EXPORTER ERROR] Failed to export building {building.BuildingId}: {ex.Message}");
                }
            }
            
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Export complete: {successCount}/{buildings.Count} building OBJ files created");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HIERARCHICAL EXPORTER ERROR] {ex.Message}");
            return 1;
        }
    }
    
    /// <summary>
    /// Load PM4 scene directly using existing LoadSceneAsync.
    /// </summary>
    private static Pm4Scene? LoadPm4Scene(string pm4Path)
    {
        try
        {
            // Use LoadSceneAsync from SceneLoaderHelper class
            var task = SceneLoaderHelper.LoadSceneAsync(
                pm4Path, 
                includeAdjacent: false, // For now, don't try to load adjacent tiles
                applyTransform: false, 
                altTransform: false
            );
            
            return task.GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HIERARCHICAL EXPORTER ERROR] Failed to load PM4 scene: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Extract buildings using correct hierarchical grouping:
    /// 1. Group MPRL entries by Unknown4 (building identifier)
    /// 2. For each group, find all MSLK entries with matching ParentIndex
    /// 3. Aggregate geometry from all matching MSLK entries
    /// </summary>
    private static List<DirectBuildingDefinition> ExtractBuildingsFromScene(Pm4Scene scene)
    {
        var buildings = new List<DirectBuildingDefinition>();
        
        try
        {
            // Check if we have all required chunks
            if (scene.Placements == null || scene.Placements.Count == 0)
            {
                Console.WriteLine("[HIERARCHICAL EXPORTER WARNING] No MPRL placements found, falling back to container-driven assembly");
                return BuildFromContainerGroups(scene);
            }

            Console.WriteLine($"[HIERARCHICAL EXPORTER] Scene has {scene.Vertices.Count} vertices, {scene.Triangles.Count} triangles");
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Found {scene.Placements.Count} MPRL placements for root building objects");
            
            // Step 1: Group MPRL entries by Unknown4 (building identifier)
            // Each unique Unknown4 value represents a separate building/object
            var buildingPlacements = scene.Placements
                .GroupBy(p => p.Unknown4)
                .ToList();
            
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Identified {buildingPlacements.Count} unique buildings by Unknown4");
            
            // Step 2: Process each building group
            int buildingId = 0;
            foreach (var placementGroup in buildingPlacements)
            {
                uint unknown4 = placementGroup.Key;
                var placements = placementGroup.ToList();
                
                // Step 3: Find all MSLK entries with matching ParentIndex
                var linkEntries = scene.Links
                    .Where(link => link.ParentId == unknown4)
                    .ToList();
                
                if (linkEntries.Count > 0)
                {
                    Console.WriteLine($"[HIERARCHICAL EXPORTER] Building {buildingId} (Unknown4={unknown4:X}) has {linkEntries.Count} MSLK child entries");
                    
                    // Step 4: Assemble complete building from all matching MSLK entries
                    var building = AssembleBuildingFromMslkEntries(scene, buildingId, unknown4, linkEntries);
                    if (building != null && building.TriangleCount > 0)
                    {
                        buildings.Add(building);
                        buildingId++;
                    }
                }
                else
                {
                    Console.WriteLine($"[HIERARCHICAL EXPORTER WARNING] Building with Unknown4={unknown4:X} has no matching MSLK entries, skipping");
                }
            }
            
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Extracted {buildings.Count} buildings using correct hierarchical grouping");
            return buildings;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HIERARCHICAL EXPORTER ERROR] Failed to extract buildings: {ex.Message}");
            return buildings;
        }
    }
    
    /// <summary>
    /// Fallback method to build from container groups when MPRL is not available.
    /// </summary>
    private static List<DirectBuildingDefinition> BuildFromContainerGroups(Pm4Scene scene)
    {
        var buildings = new List<DirectBuildingDefinition>();
        
        try
        {
            // Group MSLK entries by ParentId (container identifier)
            var linksByParent = scene.Links
                .GroupBy(link => link.ParentId)
                .Where(group => group.Any())
                .ToList();
            
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Found {linksByParent.Count} unique container groups by ParentId");
            
            int buildingId = 0;
            foreach (var parentGroup in linksByParent)
            {
                var building = AssembleBuildingFromMslkEntries(scene, buildingId, parentGroup.Key, parentGroup.ToList());
                if (building != null && building.TriangleCount > 0)
                {
                    buildings.Add(building);
                    buildingId++;
                }
            }
            
            return buildings;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HIERARCHICAL EXPORTER ERROR] Failed to build from containers: {ex.Message}");
            return buildings;
        }
    }
    
    /// <summary>
    /// Assemble a complete building from a group of MSLK entries with the same ParentId.
    /// Correctly handles container nodes (MspiFirstIndex = -1) and aggregates geometry from all children.
    /// </summary>
    private static DirectBuildingDefinition? AssembleBuildingFromMslkEntries(
        Pm4Scene scene, int buildingId, uint parentId, List<MslkEntry> mslkEntries)
    {
        try
        {
            var building = new DirectBuildingDefinition
            {
                BuildingId = buildingId,
                StartPropertyIndex = (int)parentId,
                EndPropertyIndex = (int)parentId,
                Vertices = new List<(float X, float Y, float Z)>(),
                Triangles = new List<(int V1, int V2, int V3)>()
            };
            
            var usedVertexIndices = new HashSet<int>();
            var aggregatedTriangles = new List<(int A, int B, int C)>();
            var surfaceGroupsUsed = new List<int>();
            var vertexRangeTracker = new Dictionary<string, int>(); // Track which vertex pools are used
            
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Assembling building {buildingId} from ParentId {parentId:X} ({mslkEntries.Count} MSLK entries)");
            
            // Log container node vs. geometry node breakdown
            int containerCount = mslkEntries.Count(l => l.MspiFirstIndex == -1);
            int geometryCount = mslkEntries.Count - containerCount;
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Building {buildingId}: {containerCount} container nodes, {geometryCount} geometry nodes");
            
            // Detailed audit of child entries
            Console.WriteLine($"[AUDIT] ---- Child MSLK entry list for building {buildingId} ----");
            int childIdx = 0;
            foreach (var link in mslkEntries)
            {
                // Get surface reference (SurfaceRefIndex points to MSUR)
                int surfaceRefIndex = link.SurfaceRefIndex;
                
                // Detailed audit of MSLK child entry
                string childType = link.MspiFirstIndex == -1 ? "CONTAINER" : "GEOMETRY";
                string vertexRange = link.MspiFirstIndex == -1 ? "N/A" : $"{link.MspiFirstIndex}..{link.MspiFirstIndex + link.MspiIndexCount - 1}";
                string surfaceRef = surfaceRefIndex >= 0 && surfaceRefIndex < scene.Groups.Count 
                    ? $"valid ({scene.Groups[surfaceRefIndex].Faces?.Count ?? 0} faces)" 
                    : $"INVALID (out of range 0-{scene.Groups.Count-1})";
                
                Console.WriteLine($"[AUDIT] Child #{childIdx}: Type={childType}, SurfaceRefIndex={surfaceRefIndex} ({surfaceRef}), MspiFirstIndex={link.MspiFirstIndex}, MspiIndexCount={link.MspiIndexCount}, VertexRange={vertexRange}");
                childIdx++;
                
                // Process geometry from surface reference
                if (surfaceRefIndex >= 0 && surfaceRefIndex < scene.Groups.Count)
                {
                    if (!surfaceGroupsUsed.Contains(surfaceRefIndex))
                    {
                        surfaceGroupsUsed.Add(surfaceRefIndex);
                        var surfaceGroup = scene.Groups[surfaceRefIndex];
                        var groupFaces = surfaceGroup.Faces;
                        
                        Console.WriteLine($"[HIERARCHICAL EXPORTER] Building {buildingId}: Adding Surface {surfaceRefIndex} ('{surfaceGroup.Name}') with {groupFaces.Count} faces");
                        
                        if (groupFaces != null)
                        {
                            foreach (var face in groupFaces)
                            {
                                // Check if this is a triangle or a quad/n-gon
                                // For now, we know all faces have at least A, B, C properties for 3 vertices
                                int a = face.A;
                                int b = face.B;
                                int c = face.C;
                                
                                // Track vertex pool usage for debugging
                                TrackVertexIndex(a, scene, vertexRangeTracker);
                                TrackVertexIndex(b, scene, vertexRangeTracker);
                                TrackVertexIndex(c, scene, vertexRangeTracker);
                                
                                // Add all vertices to used set
                                usedVertexIndices.Add(a);
                                usedVertexIndices.Add(b);
                                usedVertexIndices.Add(c);
                                
                                // Check for D property (quad face) using reflection
                                var faceType = face.GetType();
                                var dProperty = faceType.GetProperty("D");
                                
                                if (dProperty != null)
                                {
                                    // This is a quad face, triangulate it as two triangles
                                    int d = (int)dProperty.GetValue(face);
                                    TrackVertexIndex(d, scene, vertexRangeTracker);
                                    usedVertexIndices.Add(d);
                                    
                                    // First triangle (A, B, C)
                                    aggregatedTriangles.Add((a, b, c));
                                    
                                    // Second triangle (A, C, D) - proper quad triangulation
                                    aggregatedTriangles.Add((a, c, d));
                                    
                                    Console.WriteLine($"[HIERARCHICAL EXPORTER] Found quad face with vertices {a},{b},{c},{d}");
                                }
                                else
                                {
                                    // Regular triangle - add directly
                                    aggregatedTriangles.Add((a, b, c));
                                    
                                    // Check for additional properties E, F, etc. for n-gons (5+ sides)
                                    var eProperty = faceType.GetProperty("E");
                                    if (eProperty != null)
                                    {
                                        // This is a pentagon or higher n-gon
                                        var polygonVertices = new List<int> { a, b, c };
                                        
                                        // Collect all vertices from the n-gon
                                        int vertexIndex = 4; // Start with 'D'
                                        char propName = 'D';
                                        var nextProp = faceType.GetProperty(propName.ToString());
                                        
                                        while (nextProp != null)
                                        {
                                            int vertexValue = (int)nextProp.GetValue(face);
                                            polygonVertices.Add(vertexValue);
                                            TrackVertexIndex(vertexValue, scene, vertexRangeTracker);
                                            usedVertexIndices.Add(vertexValue);
                                            
                                            // Move to next potential vertex
                                            propName++;
                                            nextProp = faceType.GetProperty(propName.ToString());
                                        }
                                        
                                        Console.WriteLine($"[HIERARCHICAL EXPORTER] Found n-gon with {polygonVertices.Count} vertices");
                                        
                                        // Fan triangulation for n-gons (already added first triangle A,B,C)
                                        for (int i = 2; i < polygonVertices.Count - 1; i++)
                                        {
                                            aggregatedTriangles.Add((polygonVertices[0], polygonVertices[i], polygonVertices[i + 1]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    Console.WriteLine($"[HIERARCHICAL EXPORTER WARNING] Building {buildingId}: Invalid surface reference {surfaceRefIndex} (out of range 0-{scene.Groups.Count-1})");
                }
            }
            
            // Building assembly audit summary
            Console.WriteLine($"[AUDIT] ==== Building {buildingId} (ParentId {parentId:X}) Summary ====");
            Console.WriteLine($"[AUDIT] MSLK Entries: {mslkEntries.Count} total");
            Console.WriteLine($"[AUDIT] Container Entries: {containerCount}");
            Console.WriteLine($"[AUDIT] Geometry Entries: {geometryCount}");
            Console.WriteLine($"[AUDIT] Surface Groups Used: {surfaceGroupsUsed.Count}");
            Console.WriteLine($"[AUDIT] Vertex Indices: {usedVertexIndices.Count} (before mapping)");
            Console.WriteLine($"[AUDIT] Triangles Collected: {aggregatedTriangles.Count}");
            Console.WriteLine($"[AUDIT] Vertex Pools: {string.Join(", ", vertexRangeTracker.Select(kv => $"{kv.Key}={kv.Value}"))}");
            
            // Create vertex index mapping using proper vertex pool resolution
            var vertexIndexMap = new Dictionary<int, int>();
            int outOfRangeIndices = 0;
            
            foreach (var vertexIndex in usedVertexIndices.OrderBy(x => x))
            {
                var vertex = ResolveVertexFromIndex(scene, vertexIndex);
                if (vertex.HasValue)
                {
                    // Apply coordinate system fix: flip X-axis for proper orientation
                    building.Vertices.Add((-vertex.Value.X, vertex.Value.Y, vertex.Value.Z));
                    vertexIndexMap[vertexIndex] = building.Vertices.Count - 1;
                }
                else
                {
                    outOfRangeIndices++;
                }
            }
            
            if (outOfRangeIndices > 0)
            {
                Console.WriteLine($"[HIERARCHICAL EXPORTER WARNING] Building {buildingId}: {outOfRangeIndices} vertex indices out of range (cross-tile references)");
            }
            
            // Map triangles to building's vertex indices
            foreach (var (A, B, C) in aggregatedTriangles)
            {
                if (vertexIndexMap.ContainsKey(A) && vertexIndexMap.ContainsKey(B) && vertexIndexMap.ContainsKey(C))
                {
                    building.Triangles.Add((vertexIndexMap[A], vertexIndexMap[B], vertexIndexMap[C]));
                }
            }
            
            building.TriangleCount = building.Triangles.Count;
            Console.WriteLine($"[HIERARCHICAL EXPORTER] Building {buildingId}: Mapped {building.TriangleCount} triangles using {building.Vertices.Count} vertices");
            
            return building;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HIERARCHICAL EXPORTER ERROR] Failed to assemble building {buildingId}: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Track vertex pool usage for audit and debugging
    /// </summary>
    private static void TrackVertexIndex(int vertexIndex, Pm4Scene scene, Dictionary<string, int> vertexRangeTracker)
    {
        string pool;
        if (vertexIndex < 0)
        {
            pool = "NEGATIVE";
        }
        else if (vertexIndex < scene.Vertices.Count)
        {
            pool = "MSVT";
        }
        else if (vertexIndex < scene.Vertices.Count + scene.MscnVertices.Count)
        {
            pool = "MSCN";
        }
        else
        {
            // Out of range, potentially cross-tile reference
            pool = "XREF";
        }
        
        if (!vertexRangeTracker.ContainsKey(pool))
        {
            vertexRangeTracker[pool] = 0;
        }
        vertexRangeTracker[pool]++;
    }
    
    /// <summary>
    /// Resolve vertex from index, handling different vertex pools (MSVT, MSCN)
    /// </summary>
    private static (float X, float Y, float Z)? ResolveVertexFromIndex(Pm4Scene scene, int vertexIndex)
    {
        try
        {
            // Handle MSVT vertices (regular)
            if (vertexIndex >= 0 && vertexIndex < scene.Vertices.Count)
            {
                var vertex = scene.Vertices[vertexIndex];
                return (vertex.X, vertex.Y, vertex.Z);
            }
            // Handle MSCN vertices (connecting)
            else if (vertexIndex >= scene.Vertices.Count && 
                     vertexIndex < scene.Vertices.Count + scene.MscnVertices.Count)
            {
                int mscnIndex = vertexIndex - scene.Vertices.Count;
                var mscnVertex = scene.MscnVertices[mscnIndex];
                return (mscnVertex.X, mscnVertex.Y, mscnVertex.Z);
            }
            // Handle out-of-range (cross-tile) vertex references
            else
            {
                // TODO: Implement cross-tile vertex resolution
                // For now, this is out of range and we can't resolve it
                return null;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[HIERARCHICAL EXPORTER ERROR] Failed to resolve vertex {vertexIndex}: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Export building to OBJ file.
    /// </summary>
    private static string ExportBuildingToObj(DirectBuildingDefinition building, string outputDir)
    {
        // Create filename with triangle count for diagnostic purposes
        string filename = $"building_{building.BuildingId}_triangles_{building.TriangleCount}.obj";
        string objPath = Path.Combine(outputDir, filename);
        
        using (var writer = new StreamWriter(objPath))
        {
            // OBJ header
            writer.WriteLine($"# Building {building.BuildingId}");
            writer.WriteLine($"# Triangle count: {building.TriangleCount}");
            writer.WriteLine($"# Vertex count: {building.Vertices.Count}");
            writer.WriteLine();
            
            // Write vertices
            foreach (var vertex in building.Vertices)
            {
                writer.WriteLine($"v {vertex.X} {vertex.Y} {vertex.Z}");
            }
            writer.WriteLine();
            
            // Write faces (OBJ is 1-indexed)
            foreach (var triangle in building.Triangles)
            {
                writer.WriteLine($"f {triangle.V1 + 1} {triangle.V2 + 1} {triangle.V3 + 1}");
            }
        }
        
        return objPath;
    }
}
