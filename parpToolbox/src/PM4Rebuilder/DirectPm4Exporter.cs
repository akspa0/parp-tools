using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder;

/// <summary>
/// Direct PM4 to OBJ exporter that skips the database entirely.
/// Uses actual Pm4Scene structure to export all geometry as a single building.
/// </summary>
public static class DirectPm4Exporter
{
    /// <summary>
    /// Export complete buildings directly from PM4 files to OBJ format.
    /// </summary>
    /// <param name="pm4Path">Path to PM4 file or directory containing PM4 files</param>
    /// <param name="outputDir">Output directory for OBJ files</param>
    /// <returns>0 on success, 1 on error</returns>
    public static int ExportBuildings(string pm4Path, string outputDir)
    {
        try
        {
            Console.WriteLine($"[DIRECT EXPORTER] Starting direct PM4 → OBJ export from: {pm4Path}");
            Console.WriteLine($"[DIRECT EXPORTER] Output directory: {outputDir}");
            
            Directory.CreateDirectory(outputDir);
            
            // Load PM4 scene directly
            var scene = LoadPm4Scene(pm4Path);
            if (scene == null)
            {
                Console.WriteLine("[DIRECT EXPORTER ERROR] Failed to load PM4 scene");
                return 1;
            }
            
            // Extract individual buildings/objects from PM4 scene
            var buildings = ExtractBuildingsFromScene(scene);
            Console.WriteLine($"[DIRECT EXPORTER] Found {buildings.Count} buildings/objects in scene");
            
            // Export each building as separate OBJ file
            int successCount = 0;
            foreach (var building in buildings)
            {
                try
                {
                    var objPath = ExportBuildingToObj(building, outputDir);
                    Console.WriteLine($"[DIRECT EXPORTER] Exported building {building.BuildingId} → {Path.GetFileName(objPath)} ({building.TriangleCount} triangles)");
                    successCount++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[DIRECT EXPORTER ERROR] Failed to export building {building.BuildingId}: {ex.Message}");
                }
            }
            
            Console.WriteLine($"[DIRECT EXPORTER] Export complete: {successCount}/{buildings.Count} building OBJ files created");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DIRECT EXPORTER ERROR] {ex.Message}");
            return 1;
        }
    }
    
    /// <summary>
    /// Load PM4 scene directly using existing LoadSceneAsync from Program.cs.
    /// </summary>
    private static Pm4Scene? LoadPm4Scene(string pm4Path)
    {
        try
        {
            // Use LoadSceneAsync from SceneLoaderHelper class
            var task = SceneLoaderHelper.LoadSceneAsync(
                pm4Path, 
                includeAdjacent: false, 
                applyTransform: false, 
                altTransform: false
            );
            
            return task.GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DIRECT EXPORTER ERROR] Failed to load PM4 scene: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Extract individual buildings/objects from PM4 scene using real object boundaries.
    /// Uses PM4 surface groups (MSUR) for proper object-based grouping.
    /// </summary>
    private static List<DirectBuildingDefinition> ExtractBuildingsFromScene(Pm4Scene scene)
    {
        var buildings = new List<DirectBuildingDefinition>();
        
        try
        {
            Console.WriteLine($"[DIRECT EXPORTER] Scene has {scene.Vertices.Count} vertices, {scene.Triangles.Count} triangles");
            Console.WriteLine($"[DIRECT EXPORTER] Scene has {scene.Groups.Count} surface groups for object-based grouping");
            
            // Use PM4 surface groups for real object boundaries
            if (scene.Groups.Count > 0)
            {
                for (int groupIndex = 0; groupIndex < scene.Groups.Count; groupIndex++)
                {
                    var group = scene.Groups[groupIndex];
                    var building = CreateBuildingFromSurfaceGroup(scene, groupIndex, group);
                    if (building != null && building.TriangleCount > 0)
                    {
                        buildings.Add(building);
                    }
                }
                Console.WriteLine($"[DIRECT EXPORTER] Created {buildings.Count} buildings from PM4 surface groups");
            }
            else
            {
                // Fallback to triangle chunking if no surface groups available
                Console.WriteLine($"[DIRECT EXPORTER] No surface groups found, falling back to triangle chunking");
                buildings = ExtractBuildingsFromTriangleChunking(scene);
            }
            
            return buildings;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DIRECT EXPORTER ERROR] Failed to extract buildings: {ex.Message}");
            return buildings;
        }
    }
    
    /// <summary>
    /// Create a building from a PM4 surface group (real object boundary).
    /// </summary>
    private static DirectBuildingDefinition? CreateBuildingFromSurfaceGroup(Pm4Scene scene, int groupIndex, dynamic group)
    {
        try
        {
            var building = new DirectBuildingDefinition
            {
                BuildingId = groupIndex,
                StartPropertyIndex = 0, // Surface groups don't use property indices
                EndPropertyIndex = 0,
                Vertices = new List<(float X, float Y, float Z)>(),
                Triangles = new List<(int V1, int V2, int V3)>()
            };
            
            // Extract triangles from the surface group's Faces collection
            var groupFaces = group.Faces as System.Collections.IEnumerable;
            if (groupFaces != null)
            {
                var usedVertexIndices = new HashSet<int>();
                var groupTriangles = new List<(int A, int B, int C)>();
                
                foreach (dynamic face in groupFaces)
                {
                    int a = face.Item1;  // Use .Item1 instead of .A for ValueTuple
                    int b = face.Item2;  // Use .Item2 instead of .B for ValueTuple
                    int c = face.Item3;  // Use .Item3 instead of .C for ValueTuple
                    
                    groupTriangles.Add((a, b, c));
                    usedVertexIndices.Add(a);
                    usedVertexIndices.Add(b);
                    usedVertexIndices.Add(c);
                }
                
                // Create vertex index mapping using combined vertex sources
                var vertexIndexMap = new Dictionary<int, int>();
                var allVertices = CombineVertexSources(scene);
                Console.WriteLine($"[DIRECT EXPORTER] Combined vertices: {scene.Vertices.Count} regular + {scene.MscnVertices.Count} MSCN = {allVertices.Count} total");
                
                foreach (var vertexIndex in usedVertexIndices.OrderBy(x => x))
                {
                    if (vertexIndex >= 0 && vertexIndex < allVertices.Count)
                    {
                        var vertex = allVertices[vertexIndex];
                        building.Vertices.Add((vertex.X, vertex.Y, vertex.Z));
                        vertexIndexMap[vertexIndex] = building.Vertices.Count - 1;
                    }
                }
                
                // Map triangles to building's vertex indices
                foreach (var (A, B, C) in groupTriangles)
                {
                    if (vertexIndexMap.ContainsKey(A) && vertexIndexMap.ContainsKey(B) && vertexIndexMap.ContainsKey(C))
                    {
                        building.Triangles.Add((vertexIndexMap[A], vertexIndexMap[B], vertexIndexMap[C]));
                    }
                }
                
                building.TriangleCount = building.Triangles.Count;
                Console.WriteLine($"[DIRECT EXPORTER] Surface group {groupIndex} ('{group.Name}', key={group.GroupKey}): {building.TriangleCount} triangles");
            }
            
            return building;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DIRECT EXPORTER ERROR] Failed to create building from surface group {groupIndex}: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Fallback method: Extract buildings using triangle chunking when surface groups aren't available.
    /// </summary>
    private static List<DirectBuildingDefinition> ExtractBuildingsFromTriangleChunking(Pm4Scene scene)
    {
        var buildings = new List<DirectBuildingDefinition>();
        const int trianglesPerBuilding = 1000;
        int buildingId = 0;
        
        for (int startTriangle = 0; startTriangle < scene.Triangles.Count; startTriangle += trianglesPerBuilding)
        {
            int endTriangle = Math.Min(startTriangle + trianglesPerBuilding, scene.Triangles.Count);
            int triangleCount = endTriangle - startTriangle;
            
            if (triangleCount > 0)
            {
                var building = CreateBuildingFromTriangleRange(scene, buildingId, startTriangle, endTriangle);
                if (building != null && building.TriangleCount > 0)
                {
                    buildings.Add(building);
                    buildingId++;
                }
            }
        }
        
        return buildings;
    }
    
    /// <summary>
    /// Create a building from a range of triangles (fallback method).
    /// </summary>
    private static DirectBuildingDefinition? CreateBuildingFromTriangleRange(Pm4Scene scene, int buildingId, int startTriangle, int endTriangle)
    {
        try
        {
            var building = new DirectBuildingDefinition
            {
                BuildingId = buildingId,
                StartPropertyIndex = startTriangle,
                EndPropertyIndex = endTriangle - 1,
                Vertices = new List<(float X, float Y, float Z)>(),
                Triangles = new List<(int V1, int V2, int V3)>()
            };
            
            // Extract vertices used by triangles in this range
            var usedVertexIndices = new HashSet<int>();
            var triangleRange = scene.Triangles.Skip(startTriangle).Take(endTriangle - startTriangle);
            
            foreach (var (A, B, C) in triangleRange)
            {
                usedVertexIndices.Add(A);
                usedVertexIndices.Add(B);
                usedVertexIndices.Add(C);
            }
            
            // Create vertex index mapping using combined vertex sources
            var vertexIndexMap = new Dictionary<int, int>();
            var allVertices = CombineVertexSources(scene);
            foreach (var vertexIndex in usedVertexIndices.OrderBy(x => x))
            {
                if (vertexIndex >= 0 && vertexIndex < allVertices.Count)
                {
                    var vertex = allVertices[vertexIndex];
                    building.Vertices.Add((vertex.X, vertex.Y, vertex.Z));
                    vertexIndexMap[vertexIndex] = building.Vertices.Count - 1;
                }
            }
            
            // Map triangles to building's vertex indices
            foreach (var (A, B, C) in triangleRange)
            {
                if (vertexIndexMap.ContainsKey(A) && vertexIndexMap.ContainsKey(B) && vertexIndexMap.ContainsKey(C))
                {
                    building.Triangles.Add((vertexIndexMap[A], vertexIndexMap[B], vertexIndexMap[C]));
                }
            }
            
            building.TriangleCount = building.Triangles.Count;
            return building;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DIRECT EXPORTER ERROR] Failed to create building {buildingId}: {ex.Message}");
            return null;
        }
    }
    
    /// <summary>
    /// Combine regular scene vertices with MSCN vertices for complete geometry.
    /// </summary>
    private static List<System.Numerics.Vector3> CombineVertexSources(Pm4Scene scene)
    {
        var allVertices = new List<System.Numerics.Vector3>();
        
        // Add regular scene vertices first
        allVertices.AddRange(scene.Vertices);
        
        // Add MSCN vertices (if any)
        if (scene.MscnVertices.Count > 0)
        {
            allVertices.AddRange(scene.MscnVertices);
        }
        
        return allVertices;
    }
    
    /// <summary>
    /// Export a building definition to OBJ format.
    /// </summary>
    private static string ExportBuildingToObj(DirectBuildingDefinition building, string outputDir)
    {
        string fileName = $"building_{building.BuildingId:D3}_triangles_{building.TriangleCount}.obj";
        string objPath = Path.Combine(outputDir, fileName);
        
        using var writer = new StreamWriter(objPath);
        
        // Write header
        writer.WriteLine($"# PM4 Building Export - Building ID {building.BuildingId}");
        writer.WriteLine($"# Triangle Count: {building.TriangleCount}");
        writer.WriteLine($"# Triangle Range: {building.StartPropertyIndex}-{building.EndPropertyIndex}");
        writer.WriteLine();
        
        // Write vertices
        foreach (var vertex in building.Vertices)
        {
            writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
        }
        
        writer.WriteLine();
        
        // Write faces (triangles)
        foreach (var triangle in building.Triangles)
        {
            // OBJ indices are 1-based
            writer.WriteLine($"f {triangle.V1 + 1} {triangle.V2 + 1} {triangle.V3 + 1}");
        }
        
        return objPath;
    }
}

/// <summary>
/// Simple building definition for direct export (no database dependencies).
/// </summary>
public class DirectBuildingDefinition
{
    public int BuildingId { get; set; }
    public int StartPropertyIndex { get; set; }
    public int EndPropertyIndex { get; set; }
    public int TriangleCount { get; set; }
    public List<(float X, float Y, float Z)> Vertices { get; set; } = new();
    public List<(int V1, int V2, int V3)> Triangles { get; set; } = new();
}
