using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Groups PM4 geometry by MPRL placements (actual building objects) rather than surface subdivision levels.
/// Based on the confirmed relationship: MPRL.Unknown4 = MSLK.ParentIndex
/// </summary>
internal static class Pm4MprlObjectGrouper
{
    /// <summary>
    /// Represents a building object with its placement and associated geometry.
    /// </summary>
    public sealed record BuildingObject(
        uint PlacementId,
        string Name,
        Vector3 Position,
        Vector3 Rotation,
        List<(int A, int B, int C)> Triangles,
        List<MslkEntry> Links,
        int VertexCount
    );

    /// <summary>
    /// Groups PM4 scene geometry by MPRL placements to identify actual building objects.
    /// </summary>
    public static List<BuildingObject> GroupByMprlPlacements(Pm4Scene scene)
    {
        ConsoleLogger.WriteLine($"Grouping PM4 geometry by MPRL placements...");
        ConsoleLogger.WriteLine($"  MPRL Placements: {scene.Placements.Count}");
        ConsoleLogger.WriteLine($"  MSLK Links: {scene.Links.Count}");
        
        var buildingObjects = new List<BuildingObject>();
        
        // Create mapping from MPRL.Unknown4 to MSLK entries (confirmed relationship)
        var linksByParentIndex = scene.Links
            .Where(link => link.ParentIndex > 0) // uint comparison
            .GroupBy(link => link.ParentIndex)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        ConsoleLogger.WriteLine($"  Links grouped by ParentIndex: {linksByParentIndex.Count} groups");
        
        // Process each MPRL placement
        foreach (var placement in scene.Placements)
        {
            uint placementId = placement.Unknown4; // This should match MSLK.ParentIndex
            
            if (!linksByParentIndex.TryGetValue(placementId, out var linkedGeometry))
            {
                // No geometry linked to this placement
                continue;
            }
            
            // Filter out container/grouping nodes (MspiFirstIndex = -1)
            var geometryLinks = linkedGeometry.Where(link => link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0).ToList();
            
            if (geometryLinks.Count == 0)
            {
                continue; // No actual geometry, just container nodes
            }
            
            // Extract triangles for this object
            var objectTriangles = new List<(int A, int B, int C)>();
            var usedVertices = new HashSet<int>();
            
            foreach (var link in geometryLinks)
            {
                int startIndex = link.MspiFirstIndex;
                int count = link.MspiIndexCount;
                
                if (startIndex + count <= scene.Indices.Count)
                {
                    // Extract triangles from this link
                    for (int i = 0; i < count; i += 3)
                    {
                        if (i + 2 < count)
                        {
                            int idxA = scene.Indices[startIndex + i];
                            int idxB = scene.Indices[startIndex + i + 1];
                            int idxC = scene.Indices[startIndex + i + 2];
                            
                            if (idxA < scene.Vertices.Count && idxB < scene.Vertices.Count && idxC < scene.Vertices.Count)
                            {
                                objectTriangles.Add((idxA + 1, idxB + 1, idxC + 1)); // OBJ uses 1-based indexing
                                usedVertices.Add(idxA);
                                usedVertices.Add(idxB);
                                usedVertices.Add(idxC);
                            }
                        }
                    }
                }
            }
            
            if (objectTriangles.Count > 0)
            {
                // Create building object
                var position = placement.Position;
                var rotation = Vector3.Zero; // MPRL doesn't have rotation fields - using zero for now
                string objectName = $"Building_Object_{placementId:X4}";
                
                var buildingObject = new BuildingObject(
                    placementId,
                    objectName,
                    position,
                    rotation,
                    objectTriangles,
                    geometryLinks,
                    usedVertices.Count
                );
                
                buildingObjects.Add(buildingObject);
            }
        }
        
        ConsoleLogger.WriteLine($"Grouped into {buildingObjects.Count} building objects");
        ConsoleLogger.WriteLine($"  Total triangles: {buildingObjects.Sum(obj => obj.Triangles.Count)}");
        ConsoleLogger.WriteLine($"  Average triangles per object: {(buildingObjects.Count > 0 ? buildingObjects.Average(obj => obj.Triangles.Count) : 0):F1}");
        
        return buildingObjects.OrderBy(obj => obj.PlacementId).ToList();
    }

    /// <summary>
    /// Exports building objects as separate OBJ files.
    /// </summary>
    public static void ExportBuildingObjects(List<BuildingObject> buildingObjects, Pm4Scene scene, string outputRoot)
    {
        ConsoleLogger.WriteLine($"Exporting {buildingObjects.Count} building objects to {outputRoot}");
        
        foreach (var obj in buildingObjects)
        {
            if (obj.Triangles.Count == 0) continue;
            
            string objPath = Path.Combine(outputRoot, $"{obj.Name}.obj");
            
            // Create vertex mapping for this object to only include used vertices
            var usedVertices = new HashSet<int>();
            foreach (var (a, b, c) in obj.Triangles)
            {
                usedVertices.Add(a - 1); // Convert from 1-based to 0-based
                usedVertices.Add(b - 1);
                usedVertices.Add(c - 1);
            }
            
            var vertexMap = new Dictionary<int, int>();
            var mappedVertices = new List<Vector3>();
            int newIndex = 1;
            
            foreach (int oldIndex in usedVertices.OrderBy(x => x))
            {
                if (oldIndex >= 0 && oldIndex < scene.Vertices.Count)
                {
                    vertexMap[oldIndex] = newIndex++;
                    mappedVertices.Add(scene.Vertices[oldIndex]);
                }
            }
            
            using var writer = new StreamWriter(objPath);
            
            // Write header
            writer.WriteLine($"# PM4 Building Object: {obj.Name}");
            writer.WriteLine($"# Placement ID: 0x{obj.PlacementId:X8}");
            writer.WriteLine($"# Position: {obj.Position.X:F3}, {obj.Position.Y:F3}, {obj.Position.Z:F3}");
            writer.WriteLine($"# Rotation: {obj.Rotation.X:F3}, {obj.Rotation.Y:F3}, {obj.Rotation.Z:F3}");
            writer.WriteLine($"# Triangles: {obj.Triangles.Count}");
            writer.WriteLine($"# Vertices: {mappedVertices.Count}");
            writer.WriteLine($"# Links: {obj.Links.Count}");
            writer.WriteLine();
            
            // Write vertices with coordinate system correction (flip X-axis)
            foreach (var vertex in mappedVertices)
            {
                var transformedVertex = CoordinateTransformationService.ApplyPm4Transformation(vertex);
                writer.WriteLine($"v {transformedVertex.X:F6} {transformedVertex.Y:F6} {transformedVertex.Z:F6}");
            }
            writer.WriteLine();
            
            // Write object group
            writer.WriteLine($"g {obj.Name}");
            
            // Write faces with remapped indices
            foreach (var (a, b, c) in obj.Triangles)
            {
                if (vertexMap.TryGetValue(a - 1, out int newA) &&
                    vertexMap.TryGetValue(b - 1, out int newB) &&
                    vertexMap.TryGetValue(c - 1, out int newC))
                {
                    writer.WriteLine($"f {newA} {newB} {newC}");
                }
            }
            
            ConsoleLogger.WriteLine($"  {obj.Name}: {obj.Triangles.Count} triangles, {mappedVertices.Count} vertices -> {objPath}");
        }
        
        ConsoleLogger.WriteLine($"Exported {buildingObjects.Count} building objects as separate OBJ files");
    }
}
