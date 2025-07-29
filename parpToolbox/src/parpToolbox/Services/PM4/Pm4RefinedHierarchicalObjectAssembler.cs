using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Refined version of Pm4HierarchicalObjectAssembler that uses insights from chunk combination testing
/// to better identify and assemble true object containers.
/// </summary>
internal static class Pm4RefinedHierarchicalObjectAssembler
{
    /// <summary>
    /// Represents a complete building object assembled using refined hierarchical grouping.
    /// </summary>
    public record RefinedHierarchicalObject(
        int ObjectId,                   // Object group ID
        List<ushort> ComponentTypes,    // MPRL object types included in this building object
        List<(int A, int B, int C)> Triangles, // All triangles for this object
        Vector3 BoundingCenter,         // Calculated center point
        int VertexCount,
        string ObjectType,              // Descriptive name based on object ID
        int ParentContainerId = -1,     // Parent container ID if applicable
        List<int>? ChildContainerIds = null // Child container IDs if applicable
    );

    /// <summary>
    /// Assembles PM4 objects using refined hierarchical grouping in a streaming fashion.
    /// This avoids building all objects in memory and writes each object to disk as it's assembled.
    /// </summary>
    public static void StreamRefinedHierarchicalObjects(Pm4Scene scene, string outputPath)
    {
        Directory.CreateDirectory(outputPath);
        
        Console.WriteLine($"Streaming refined hierarchical PM4 export to {outputPath}...");
        
        // Group placements by tile
        var tileGroups = GroupPlacementsByTile(scene);
        Console.WriteLine($"Found {tileGroups.Count} tile groups");
        
        int exportedCount = 0;
        
        // Process each tile group
        int tileIndex = 0;
        foreach (var (tileX, tileY, tilePlacements) in tileGroups)
        {
            tileIndex++;
            Console.WriteLine($"Processing tile ({tileX}, {tileY}) with {tilePlacements.Count} placements ({tileIndex}/{tileGroups.Count})");
            
            // Create output directory for this tile
            string tileOutputPath = Path.Combine(outputPath, $"tile_{tileX}_{tileY}");
            Directory.CreateDirectory(tileOutputPath);
            
            // Use hierarchy analyzer to find proper object segments
            var hierarchyAnalyzer = new MslkHierarchyAnalyzer();
            var hierarchyResult = hierarchyAnalyzer.AnalyzeHierarchy(scene.Links);
            var objectSegments = hierarchyAnalyzer.SegmentObjectsByHierarchy(hierarchyResult);
            
            Console.WriteLine($"Hierarchy Analysis Results:");
            foreach (var pattern in hierarchyResult.DiscoveredPatterns)
            {
                Console.WriteLine($"  {pattern}");
            }
            Console.WriteLine($"Found {objectSegments.Count} object segments from {hierarchyResult.GroupLeaders.Count} group leaders");

            // Process each object segment
            int processedContainers = 0;
            foreach (var segment in objectSegments)
            {
                processedContainers++;
                Console.WriteLine($"Processing object segment {segment.RootIndex} with {segment.TotalNodes} nodes ({processedContainers}/{objectSegments.Count})");
                Console.WriteLine($"  Object Segment {segment.RootIndex}: {segment.TotalGeometryNodes} geometry nodes, {segment.TotalDoodadNodes} doodad nodes, type: {segment.SegmentationType}");
                
                // Extract geometry for all nodes in this object segment
                var allTriangles = new List<(int A, int B, int C)>();
                var allVertexIndices = new HashSet<int>();
                var processedNodes = new HashSet<int>();
                
                // Process all geometry nodes in this segment
                foreach (var nodeIndex in segment.GeometryNodeIndices)
                {
                    if (processedNodes.Contains(nodeIndex))
                        continue;
                        
                    processedNodes.Add(nodeIndex);
                    
                    try
                    {
                        // Extract geometry for this MSLK node
                        ExtractGeometryForMslkNode(nodeIndex, scene, allTriangles, allVertexIndices);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"    Warning: Failed to extract geometry for MSLK node {nodeIndex}: {ex.Message}");
                    }
                }
                
                // Create and export object if it has geometry
                if (allTriangles.Count > 0)
                {
                    var boundingCenter = CalculateBoundingCenter(allVertexIndices, scene);
                    var objectType = GetObjectTypeName(segment.RootIndex, segment.TotalNodes);
                    
                    var obj = new RefinedHierarchicalObject(
                        segment.RootIndex,
                        new List<ushort> { (ushort)segment.RootIndex },
                        allTriangles,
                        boundingCenter,
                        allVertexIndices.Count,
                        objectType
                    );
                    
                    ExportObjectToObj(obj, tileOutputPath, scene);
                    exportedCount++;
                    Console.WriteLine($"    Exported object segment {segment.RootIndex} ({obj.Triangles.Count} triangles, {obj.VertexCount} vertices)");
                }
                else
                {
                    Console.WriteLine($"    Skipping object segment {segment.RootIndex} - no geometry found");
                }
            }
        }
        
        Console.WriteLine($"Streaming export completed. Exported {exportedCount} objects to {outputPath}");
    }
    
    /// <summary>
    /// Groups placements by tile using MSLK tile coordinates.
    /// </summary>
    private static List<(int tileX, int tileY, List<MprlChunk.Entry> placements)> GroupPlacementsByTile(Pm4Scene scene)
    {
        var tileGroups = new Dictionary<(int, int), List<MprlChunk.Entry>>();
        
        Console.WriteLine($"Grouping {scene.Placements.Count} placements by tile...");
        
        // Create a dictionary for faster link lookups
        var linkByParentIndex = scene.Links
            .GroupBy(link => link.ParentIndex)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        int placementCount = 0;
        
        // Group placements by their associated MSLK links' tile coordinates
        foreach (var placement in scene.Placements)
        {
            placementCount++;
            if (placementCount % 10000 == 0)
            {
                Console.WriteLine($"  Processed {placementCount}/{scene.Placements.Count} placements");
            }
            
            ushort placementId = placement.Unknown4;
            
            // Find MSLK links that reference this placement
            if (linkByParentIndex.TryGetValue(placementId, out var associatedLinks))
            {
                // Use the first link's tile coordinates for grouping
                var firstLink = associatedLinks.First();
                if (firstLink.TryDecodeTileCoordinates(out int tileX, out int tileY))
                {
                    var key = (tileX, tileY);
                    if (!tileGroups.ContainsKey(key))
                    {
                        tileGroups[key] = new List<MprlChunk.Entry>();
                    }
                    tileGroups[key].Add(placement);
                }
                else
                {
                    // Fallback to tile (0, 0) if we can't decode tile coordinates
                    var key = (0, 0);
                    if (!tileGroups.ContainsKey(key))
                    {
                        tileGroups[key] = new List<MprlChunk.Entry>();
                    }
                    tileGroups[key].Add(placement);
                }
            }
            else
            {
                // Fallback to tile (0, 0) if no associated links
                var key = (0, 0);
                if (!tileGroups.ContainsKey(key))
                {
                    tileGroups[key] = new List<MprlChunk.Entry>();
                }
                tileGroups[key].Add(placement);
            }
        }
        
        Console.WriteLine($"Grouped placements into {tileGroups.Count} tiles");
        return tileGroups.Select(kvp => (kvp.Key.Item1, kvp.Key.Item2, kvp.Value)).ToList();
    }
    
    /// <summary>
    /// Groups placements by container ID within a tile.
    /// Uses a combination of Unknown4 type and sequential grouping to ensure multiple separate objects.
    /// </summary>
    private static Dictionary<int, List<MprlChunk.Entry>> GroupPlacementsByContainer(List<MprlChunk.Entry> tilePlacements, Pm4Scene scene)
    {
        var containerGroups = new Dictionary<int, List<MprlChunk.Entry>>();
        
        Console.WriteLine($"  GROUPING_DEBUG: Analyzing {tilePlacements.Count} placements for container grouping");
        
        // First, analyze the distribution of Unknown4 values
        var unknown4Distribution = tilePlacements
            .GroupBy(p => p.Unknown4)
            .ToDictionary(g => g.Key, g => g.Count());
        
        Console.WriteLine($"  GROUPING_DEBUG: Unknown4 distribution: {string.Join(", ", unknown4Distribution.Select(kvp => $"{kvp.Key}:{kvp.Value}"))}");
        
        // If all placements have the same Unknown4, use sequential grouping to split them
        if (unknown4Distribution.Count == 1)
        {
            Console.WriteLine($"  GROUPING_DEBUG: All placements have same Unknown4 ({unknown4Distribution.Keys.First()}), using sequential grouping");
            
            // Split placements into smaller groups of max 50 placements each
            const int maxPlacementsPerGroup = 50;
            int groupId = 0;
            
            for (int i = 0; i < tilePlacements.Count; i += maxPlacementsPerGroup)
            {
                var groupPlacements = tilePlacements.Skip(i).Take(maxPlacementsPerGroup).ToList();
                containerGroups[groupId] = groupPlacements;
                Console.WriteLine($"  GROUPING_DEBUG: Created sequential group {groupId} with {groupPlacements.Count} placements");
                groupId++;
            }
        }
        else
        {
            Console.WriteLine($"  GROUPING_DEBUG: Multiple Unknown4 values found, using Unknown4-based grouping");
            
            // Use Unknown4 values as container IDs since they're actually unique
            foreach (var placement in tilePlacements)
            {
                int containerId = placement.Unknown4;
                
                if (!containerGroups.TryGetValue(containerId, out var containerList))
                {
                    containerList = new List<MprlChunk.Entry>();
                    containerGroups[containerId] = containerList;
                }
                containerList.Add(placement);
            }
        }
        
        Console.WriteLine($"  GROUPING_DEBUG: Created {containerGroups.Count} container groups");
        return containerGroups;
    }
    
    /// <summary>
    /// Extracts geometry for a single MPRL placement by finding associated MSLK links.
    /// </summary>
    private static void ExtractGeometryForPlacement(MprlChunk.Entry placement, Pm4Scene scene, 
        Dictionary<uint, List<MslkEntry>> linkByParentIndex,
        List<(int A, int B, int C)> triangles, HashSet<int> vertexIndices, HashSet<(uint parentIndex, ushort referenceIndex)> visitedLinks, HashSet<ushort>? visitedPlacements = null, int depth = 0)
    {
        // Initialize visitedPlacements set if not provided
        visitedPlacements = visitedPlacements ?? new HashSet<ushort>();
        
        ushort placementId = placement.Unknown4;
        
        // Prevent infinite recursion by checking if we've already processed this placement
        if (visitedPlacements.Contains(placementId))
        {
            //Console.WriteLine($"Skipping already visited placement: {placementId}");
            return;
        }
        
        // Mark this placement as visited
        visitedPlacements.Add(placementId);
        
        // Limit recursion depth to 1 as requested by user
        // Only process direct children, no deeper recursion
        if (depth >= 1)
        {
            Console.WriteLine($"RECURSION_GUARD: Stopping recursion for placement {placementId} at depth {depth} (limit reached)");
            return;
        }
        
        // Find MSLK links that reference this placement using the pre-built dictionary for efficiency
        if (!linkByParentIndex.TryGetValue(placementId, out var associatedLinks))
        {
            // No associated links found
            return;
        }
        
        // Log diagnostic information
        if (depth == 0)
        {
            Console.WriteLine($"  Extracting geometry for placement {placementId} (depth {depth}), found {associatedLinks.Count()} associated links");
        }
        
        foreach (var link in associatedLinks)
        {
            // Create a key for this link to prevent infinite recursion
            var linkKey = (link.ParentIndex, link.ReferenceIndex);
            
            // Prevent infinite recursion by checking if we've already processed this link
            if (visitedLinks.Contains(linkKey))
            {
                //Console.WriteLine($"Skipping already visited link: ParentIndex={link.ParentIndex}, ReferenceIndex={link.ReferenceIndex}");
                continue;
            }
            
            // Mark this link as visited
            visitedLinks.Add(linkKey);
            
            // Also check if we've already processed the reference index to prevent placement cycles
            if (link.ReferenceIndex != placementId && visitedLinks.Any(k => k.Item2 == link.ReferenceIndex))
            {
                //Console.WriteLine($"Skipping already visited placement: ReferenceIndex={link.ReferenceIndex}");
                continue;
            }
            
            //Console.WriteLine($"Processing link: ParentIndex={link.ParentIndex}, ReferenceIndex={link.ReferenceIndex}, MspiFirstIndex={link.MspiFirstIndex}");
            
            if (link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
            {
                // Direct geometry link - extract triangles
                ExtractTrianglesFromLink(link, scene, triangles, vertexIndices);
            }
            else if (link.MspiFirstIndex == -1)
            {
                // Container node - traverse child placements referenced by ReferenceIndex
                // SIMPLIFIED: Only look for direct MPRL placements matching ReferenceIndex
                var childPlacements = scene.Placements
                    .Where(p => p.Unknown4 == link.ReferenceIndex && p.Unknown4 != placementId) // Prevent self-reference
                    .ToList();
                
                if (childPlacements.Any())
                {
                    Console.WriteLine($"RECURSION_DEBUG: Container node {placementId} -> ReferenceIndex {link.ReferenceIndex}, found {childPlacements.Count} child placements");
                    
                    foreach (var childPlacement in childPlacements)
                    {
                        // Increment depth for child placement processing
                        int nextDepth = depth + 1;
                        
                        Console.WriteLine($"RECURSION_DEBUG: Processing child placement {childPlacement.Unknown4} at depth {nextDepth}");
                        ExtractGeometryForPlacement(childPlacement, scene, linkByParentIndex, triangles, vertexIndices, visitedLinks, visitedPlacements, nextDepth);
                    }
                }
                else
                {
                    Console.WriteLine($"RECURSION_DEBUG: Container node {placementId} -> ReferenceIndex {link.ReferenceIndex}, no child placements found");
                }
            }
        }
    }
    
    /// <summary>
    /// Extracts geometry for a specific MSLK node by index.
    /// </summary>
    private static void ExtractGeometryForMslkNode(int nodeIndex, Pm4Scene scene,
        List<(int A, int B, int C)> triangles, HashSet<int> vertexIndices)
    {
        if (nodeIndex < 0 || nodeIndex >= scene.Links.Count)
            return;
            
        var link = scene.Links[nodeIndex];
        
        // Only extract geometry if this node has valid geometry data
        if (link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
        {
            ExtractTrianglesFromLink(link, scene, triangles, vertexIndices);
        }
    }
    
    /// <summary>
    /// Extracts triangles from a single MSLK link.
    /// </summary>
    private static void ExtractTrianglesFromLink(dynamic link, Pm4Scene scene, 
        List<(int A, int B, int C)> triangles, HashSet<int> vertexIndices)
    {
        int startIndex = link.MspiFirstIndex;
        int count = link.MspiIndexCount;
        
        // Cache scene properties for better performance
        var sceneIndices = scene.Indices;
        var sceneVertices = scene.Vertices;
        int verticesCount = sceneVertices.Count;
        
        if (startIndex >= 0 && startIndex + count <= sceneIndices.Count)
        {
            // Pre-allocate capacity for better performance
            int triangleCount = count / 3;
            if (triangles.Capacity < triangles.Count + triangleCount)
            {
                triangles.Capacity = triangles.Count + triangleCount;
            }
            
            for (int i = 0; i < count; i += 3)
            {
                if (i + 2 < count)
                {
                    int idxA = sceneIndices[startIndex + i];
                    int idxB = sceneIndices[startIndex + i + 1];
                    int idxC = sceneIndices[startIndex + i + 2];
                    
                    if (idxA < verticesCount && 
                        idxB < verticesCount && 
                        idxC < verticesCount)
                    {
                        triangles.Add((idxA, idxB, idxC));
                        vertexIndices.Add(idxA);
                        vertexIndices.Add(idxB);
                        vertexIndices.Add(idxC);
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// Calculates the bounding center from a set of vertex indices.
    /// </summary>
    private static Vector3 CalculateBoundingCenter(HashSet<int> vertexIndices, Pm4Scene scene)
    {
        if (vertexIndices.Count == 0)
            return Vector3.Zero;
        
        // Cache scene vertices for better performance
        var sceneVertices = scene.Vertices;
        int verticesCount = sceneVertices.Count;
        
        var sum = Vector3.Zero;
        foreach (int index in vertexIndices)
        {
            if (index < verticesCount)
            {
                sum += sceneVertices[index];
            }
        }
        
        return sum / vertexIndices.Count;
    }
    
    /// <summary>
    /// Gets a descriptive object type name based on the object ID and component count.
    /// </summary>
    private static string GetObjectTypeName(int objectId, int componentCount)
    {
        // Simple naming scheme - can be enhanced based on object characteristics
        if (componentCount > 10)
            return $"LargeBuilding_{objectId}";
        else if (componentCount > 5)
            return $"MediumBuilding_{objectId}";
        else if (componentCount > 1)
            return $"SmallBuilding_{objectId}";
        else
            return $"Object_{objectId}";
    }
    
    /// <summary>
    /// Exports a refined hierarchical object to an OBJ file.
    /// </summary>
    private static void ExportObjectToObj(RefinedHierarchicalObject obj, string outputPath, Pm4Scene scene)
    {
        try
        {
            string filename = Path.Combine(outputPath, $"{obj.ObjectType}_{obj.ObjectId}.obj");
            
            using var writer = new StreamWriter(filename);
            
            // Write header
            writer.WriteLine($"# PM4 Refined Hierarchical Object Export");
            writer.WriteLine($"# Object ID: {obj.ObjectId}");
            writer.WriteLine($"# Component Types: {string.Join(", ", obj.ComponentTypes)}");
            writer.WriteLine($"# Triangles: {obj.Triangles.Count}");
            writer.WriteLine($"# Vertices: {obj.VertexCount}");
            writer.WriteLine();
            
            // Collect all unique vertex indices used by this object
            var usedVertexIndices = new HashSet<int>();
            foreach (var (a, b, c) in obj.Triangles)
            {
                usedVertexIndices.Add(a);
                usedVertexIndices.Add(b);
                usedVertexIndices.Add(c);
            }
            
            // Create a mapping from original vertex indices to remapped indices (1-based)
            var vertexIndexMap = new Dictionary<int, int>();
            int remappedIndex = 1;
            
            foreach (int originalIndex in usedVertexIndices.OrderBy(x => x))
            {
                vertexIndexMap[originalIndex] = remappedIndex++;
            }
            
            // Write vertices using the remapped indices
            foreach (int originalIndex in usedVertexIndices.OrderBy(x => x))
            {
                if (originalIndex < 0 || originalIndex >= scene.Vertices.Count) continue;
                
                var vertex = scene.Vertices[originalIndex];
                // Apply coordinate system correction
                writer.WriteLine($"v {-vertex.X} {vertex.Y} {vertex.Z}");
            }
            
            writer.WriteLine();
            
            // Write faces using the remapped indices
            foreach (var (a, b, c) in obj.Triangles)
            {
                int remappedA = vertexIndexMap[a];
                int remappedB = vertexIndexMap[b];
                int remappedC = vertexIndexMap[c];
                writer.WriteLine($"f {remappedA} {remappedB} {remappedC}");
            }
            
            Console.WriteLine($"  Exported {obj.ObjectType}_{obj.ObjectId}.obj ({obj.Triangles.Count} triangles, {obj.VertexCount} vertices)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error exporting {obj.ObjectType}_{obj.ObjectId}.obj: {ex.Message}");
        }
    }
}
