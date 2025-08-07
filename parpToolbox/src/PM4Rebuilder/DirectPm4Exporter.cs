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
            
            // Use MSLK-driven building assembly for complete buildings
            if (scene.Links.Count > 0 && scene.Groups.Count > 0)
            {
                buildings = AssembleBuildingsFromMslkLinkage(scene);
                Console.WriteLine($"[DIRECT EXPORTER] Created {buildings.Count} buildings from MSLK-driven assembly");
            }
            else if (scene.Groups.Count > 0)
            {
                // Fallback: individual surface groups (fragments)
                Console.WriteLine($"[DIRECT EXPORTER] No MSLK data found, using surface group fragments");
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
                // Final fallback to triangle chunking
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
    /// Assemble complete buildings using MSLK linkage data to aggregate MSUR surface fragments.
    /// Groups MSLK entries by ParentId (building container) and collects all referenced surface fragments.
    /// </summary>
    private static List<DirectBuildingDefinition> AssembleBuildingsFromMslkLinkage(Pm4Scene scene)
    {
        var buildings = new List<DirectBuildingDefinition>();
        
        try
        {
            Console.WriteLine($"[DIRECT EXPORTER] MSLK-driven assembly: {scene.Links.Count} links, {scene.Groups.Count} surface groups");
            
            // Group MSLK entries by ParentId (building container)
            var linksByParent = scene.Links
                .Where(link => link.HasGeometry) // Only links with actual geometry
                .GroupBy(link => link.ParentId)
                .Where(group => group.Any()) // Skip empty groups
                .ToList();
            
            Console.WriteLine($"[DIRECT EXPORTER] Found {linksByParent.Count} unique building containers (ParentId groups)");
            
            // Create surface group lookup for fast access
            var surfaceGroupLookup = scene.Groups
                .Select((group, index) => new { group, index })
                .ToDictionary(x => x.index, x => x.group);
            
            int buildingId = 0;
            foreach (var parentGroup in linksByParent)
            {
                var building = AssembleBuildingFromParentGroup(scene, buildingId, parentGroup.Key, parentGroup.ToList(), surfaceGroupLookup);
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
            Console.WriteLine($"[DIRECT EXPORTER ERROR] MSLK assembly failed: {ex.Message}");
            return buildings;
        }
    }
    
    /// <summary>
    /// Assemble a complete building from a group of MSLK entries sharing the same ParentId.
    /// </summary>
    private static DirectBuildingDefinition? AssembleBuildingFromParentGroup(Pm4Scene scene, int buildingId, uint parentId, List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry> mslkEntries, Dictionary<int, ParpToolbox.Formats.PM4.SurfaceGroup> surfaceGroupLookup)
    {
        try
        {
            var building = new DirectBuildingDefinition
            {
                BuildingId = buildingId,
                StartPropertyIndex = (int)parentId, // Use ParentId as identifier
                EndPropertyIndex = (int)parentId,
                Vertices = new List<(float X, float Y, float Z)>(),
                Triangles = new List<(int V1, int V2, int V3)>()
            };
            
            var usedVertexIndices = new HashSet<int>();
            var aggregatedTriangles = new List<(int A, int B, int C)>();
            
            Console.WriteLine($"[DIRECT EXPORTER] Assembling building {buildingId} from ParentId {parentId} ({mslkEntries.Count} MSLK entries)");
            
            var surfaceGroupsUsed = new List<int>();
            var vertexRangeTracker = new Dictionary<string, int>(); // Track which vertex pools are used
            
            // Process each MSLK entry to collect referenced surface fragments
            foreach (var link in mslkEntries)
            {
                // Get surface reference (SurfaceRefIndex points to MSUR)
                int surfaceRefIndex = link.SurfaceRefIndex;
                
                Console.WriteLine($"[DIAGNOSTIC] Building {buildingId}: MSLK entry references surface {surfaceRefIndex}");
                
                // Find the corresponding surface group
                if (surfaceRefIndex >= 0 && surfaceRefIndex < scene.Groups.Count)
                {
                    surfaceGroupsUsed.Add(surfaceRefIndex);
                    var surfaceGroup = scene.Groups[surfaceRefIndex];
                    var groupFaces = surfaceGroup.Faces;
                    
                    Console.WriteLine($"[DIAGNOSTIC] Building {buildingId}: Surface {surfaceRefIndex} ('{surfaceGroup.Name}') has {groupFaces.Count} faces");
                    
                    if (groupFaces != null)
                    {
                        foreach (var face in groupFaces)
                        {
                            int a = face.A;
                            int b = face.B;
                            int c = face.C;
                            
                            // Track vertex pool usage
                            TrackVertexPoolUsage(a, scene, vertexRangeTracker);
                            TrackVertexPoolUsage(b, scene, vertexRangeTracker);
                            TrackVertexPoolUsage(c, scene, vertexRangeTracker);
                            
                            aggregatedTriangles.Add((a, b, c));
                            usedVertexIndices.Add(a);
                            usedVertexIndices.Add(b);
                            usedVertexIndices.Add(c);
                        }
                    }
                }
                else
                {
                    Console.WriteLine($"[DIAGNOSTIC WARNING] Building {buildingId}: Invalid surface reference {surfaceRefIndex} (out of range 0-{scene.Groups.Count-1})");
                }
            }
            
            // Log data source summary for this building
            Console.WriteLine($"[DIAGNOSTIC] Building {buildingId} data sources:");
            Console.WriteLine($"  - Surface groups used: [{string.Join(", ", surfaceGroupsUsed)}]");
            Console.WriteLine($"  - Vertex pool usage: {string.Join(", ", vertexRangeTracker.Select(kv => $"{kv.Key}={kv.Value}"))}");
            Console.WriteLine($"  - Total triangles: {aggregatedTriangles.Count}, vertices: {usedVertexIndices.Count}");
            
            // Check for potential cross-contamination indicators
            if (surfaceGroupsUsed.Count > 10)
            {
                Console.WriteLine($"[CROSS-CONTAMINATION WARNING] Building {buildingId} uses {surfaceGroupsUsed.Count} surface groups - may be over-aggregating!");
            }
            
            // Create vertex index mapping using proper vertex pool resolution
            var vertexIndexMap = new Dictionary<int, int>();
            Console.WriteLine($"[DIRECT EXPORTER] Building {buildingId}: aggregated {aggregatedTriangles.Count} triangles from {mslkEntries.Count} surface fragments");
            
            foreach (var vertexIndex in usedVertexIndices.OrderBy(x => x))
            {
                var vertex = ResolveVertexFromIndex(scene, vertexIndex);
                if (vertex.HasValue)
                {
                    // Apply coordinate system fix: flip X-axis for proper orientation
                    building.Vertices.Add((-vertex.Value.X, vertex.Value.Y, vertex.Value.Z));
                    vertexIndexMap[vertexIndex] = building.Vertices.Count - 1;
                }
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
            return building;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DIRECT EXPORTER ERROR] Failed to assemble building {buildingId}: {ex.Message}");
            return null;
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
                
                // Create vertex index mapping using proper vertex pool resolution
                var vertexIndexMap = new Dictionary<int, int>();
                Console.WriteLine($"[DIRECT EXPORTER] Vertex pools: {scene.Vertices.Count} regular + {scene.MscnVertices.Count} MSCN");
                Console.WriteLine($"[DIRECT EXPORTER] Index range: 0-{scene.Vertices.Count-1} = regular, {scene.Vertices.Count}+ = MSCN");
                
                foreach (var vertexIndex in usedVertexIndices.OrderBy(x => x))
                {
                    var vertex = ResolveVertexFromIndex(scene, vertexIndex);
                    if (vertex.HasValue)
                    {
                        building.Vertices.Add((vertex.Value.X, vertex.Value.Y, vertex.Value.Z));
                        vertexIndexMap[vertexIndex] = building.Vertices.Count - 1;
                    }
                    else
                    {
                        Console.WriteLine($"[DIRECT EXPORTER WARNING] Invalid vertex index {vertexIndex} (out of range)");
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
            
            // Create vertex index mapping using proper vertex pool resolution
            var vertexIndexMap = new Dictionary<int, int>();
            foreach (var vertexIndex in usedVertexIndices.OrderBy(x => x))
            {
                var vertex = ResolveVertexFromIndex(scene, vertexIndex);
                if (vertex.HasValue)
                {
                    // Apply coordinate system fix: flip X-axis for proper orientation
                    building.Vertices.Add((-vertex.Value.X, vertex.Value.Y, vertex.Value.Z));
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
    /// Resolve a vertex index to the correct vertex from the appropriate pool (regular or MSCN).
    /// Indices 0 to scene.Vertices.Count-1 reference regular vertices.
    /// Indices scene.Vertices.Count+ reference MSCN vertices.
    /// </summary>
    private static System.Numerics.Vector3? ResolveVertexFromIndex(Pm4Scene scene, int index)
    {
        // Regular vertex range: 0 to scene.Vertices.Count-1
        if (index >= 0 && index < scene.Vertices.Count)
        {
            return scene.Vertices[index];
        }
        
        // MSCN vertex range: scene.Vertices.Count to scene.Vertices.Count + scene.MscnVertices.Count-1
        int mscnIndex = index - scene.Vertices.Count;
        if (mscnIndex >= 0 && mscnIndex < scene.MscnVertices.Count)
        {
            return scene.MscnVertices[mscnIndex];
        }
        
        // Index out of range
        return null;
    }
    
    /// <summary>
    /// Track vertex pool usage for diagnostic purposes.
    /// Helps identify cross-contamination between buildings.
    /// </summary>
    private static void TrackVertexPoolUsage(int vertexIndex, Pm4Scene scene, Dictionary<string, int> tracker)
    {
        if (vertexIndex >= 0 && vertexIndex < scene.Vertices.Count)
        {
            // Regular vertex pool
            tracker["regular"] = tracker.GetValueOrDefault("regular", 0) + 1;
        }
        else if (vertexIndex >= scene.Vertices.Count && vertexIndex < scene.Vertices.Count + scene.MscnVertices.Count)
        {
            // MSCN vertex pool
            tracker["mscn"] = tracker.GetValueOrDefault("mscn", 0) + 1;
        }
        else
        {
            // Out of range
            tracker["invalid"] = tracker.GetValueOrDefault("invalid", 0) + 1;
        }
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
