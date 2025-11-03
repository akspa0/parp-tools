namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;

/// <summary>
/// Assembles PM4 objects using MPRR hierarchical grouping to combine component types into complete building objects.
/// Uses MPRR separators (Value1=65535) to identify object boundaries and group related components.
/// </summary>
internal static class Pm4HierarchicalObjectAssembler
{
    /// <summary>
    /// Represents a complete building object assembled using MPRR hierarchical grouping.
    /// </summary>
    public record HierarchicalObject(
        int ObjectId,                   // Object group ID (segment between MPRR separators)
        List<ushort> ComponentTypes,    // MPRL object types included in this building object
        List<(int A, int B, int C)> Triangles, // All triangles for this object
        Vector3 BoundingCenter,         // Calculated center point
        int VertexCount,
        string ObjectType               // Descriptive name based on object ID
    );

    /// <summary>
    /// Assembles PM4 objects by using MPRR separators to group related component types into complete building objects.
    /// </summary>
    public static List<HierarchicalObject> AssembleHierarchicalObjects(Pm4Scene scene)
    {
        var assembledObjects = new List<HierarchicalObject>();
        
        Console.WriteLine($"Assembling PM4 objects using MPRR hierarchical grouping...");
        Console.WriteLine($"  MPRR properties: {scene.Properties.Count}");
        Console.WriteLine($"  MPRL placements: {scene.Placements.Count}");
        
        // Parse MPRR segments separated by Value1=65535
        var objectGroups = ParseMprrObjectGroups(scene);
        Console.WriteLine($"  Found {objectGroups.Count} MPRR object groups (building objects)");
        
        // Map MPRL placements by object type for quick lookup
        var placementsByType = scene.Placements
            .GroupBy(p => p.Unknown4)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        // Assemble each object group by combining all component types in the group
        foreach (var (objectId, componentTypes) in objectGroups)
        {
            var allTriangles = new List<(int A, int B, int C)>();
            var allVertexIndices = new HashSet<int>();
            var includedComponentTypes = new List<ushort>();
            
            Console.WriteLine($"    Processing object group {objectId} with {componentTypes.Count} component types");
            
            foreach (var componentType in componentTypes)
            {
                if (placementsByType.TryGetValue(componentType, out var placements))
                {
                    includedComponentTypes.Add(componentType);
                    
                    // Extract geometry for all placements of this component type
                    foreach (var placement in placements)
                    {
                        ExtractGeometryForPlacement(placement, scene, allTriangles, allVertexIndices);
                    }
                }
            }
            
            if (allTriangles.Count > 0)
            {
                // Calculate bounding center from all vertices used by this object
                var boundingCenter = CalculateBoundingCenter(allVertexIndices, scene);
                var objectType = GetObjectTypeName(objectId, includedComponentTypes.Count);
                
                var hierarchicalObject = new HierarchicalObject(
                    objectId,
                    includedComponentTypes,
                    allTriangles,
                    boundingCenter,
                    allVertexIndices.Count,
                    objectType
                );
                
                assembledObjects.Add(hierarchicalObject);
                
                Console.WriteLine($"      Building Object {objectId}: {allTriangles.Count} triangles, {allVertexIndices.Count} vertices, {includedComponentTypes.Count} component types");
            }
        }
        
        Console.WriteLine($"Assembled {assembledObjects.Count} complete building objects using hierarchical grouping");
        return assembledObjects;
    }
    
    /// <summary>
    /// Parses MPRR data to identify object groups separated by Value1=65535 markers.
    /// </summary>
    private static Dictionary<int, List<ushort>> ParseMprrObjectGroups(Pm4Scene scene)
    {
        var objectGroups = new Dictionary<int, List<ushort>>();
        var currentGroup = new List<ushort>();
        int objectId = 0;
        
        foreach (var property in scene.Properties)
        {
            if (property.Value1 == 65535)
            {
                // Separator found - finalize current group and start new one
                if (currentGroup.Count > 0)
                {
                    objectGroups[objectId] = new List<ushort>(currentGroup);
                    currentGroup.Clear();
                    objectId++;
                }
            }
            else
            {
                // Add component type to current group
                // Use Value1 as the component type identifier
                if (property.Value1 > 0 && property.Value1 <= ushort.MaxValue)
                {
                    currentGroup.Add((ushort)property.Value1);
                }
            }
        }
        
        // Add final group if it has content
        if (currentGroup.Count > 0)
        {
            objectGroups[objectId] = currentGroup;
        }
        
        return objectGroups;
    }
    
    /// <summary>
    /// Extracts geometry for a single MPRL placement by finding associated MSLK links.
    /// </summary>
    private static void ExtractGeometryForPlacement(dynamic placement, Pm4Scene scene, 
        List<(int A, int B, int C)> triangles, HashSet<int> vertexIndices)
    {
        ushort placementId = placement.Unknown4;
        
        // Find MSLK links that reference this placement
        var associatedLinks = scene.Links.Where(link => link.ParentIndex == placementId);
        
        foreach (var link in associatedLinks)
        {
            if (link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
            {
                ExtractTrianglesFromLink(link, scene, triangles, vertexIndices);
            }
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
        
        if (startIndex >= 0 && startIndex + count <= scene.Indices.Count)
        {
            for (int i = 0; i < count; i += 3)
            {
                if (i + 2 < count)
                {
                    int idxA = scene.Indices[startIndex + i];
                    int idxB = scene.Indices[startIndex + i + 1];
                    int idxC = scene.Indices[startIndex + i + 2];
                    
                    if (idxA < scene.Vertices.Count && 
                        idxB < scene.Vertices.Count && 
                        idxC < scene.Vertices.Count)
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
        
        var sum = Vector3.Zero;
        foreach (int index in vertexIndices)
        {
            sum += scene.Vertices[index];
        }
        
        return sum / vertexIndices.Count;
    }
    
    /// <summary>
    /// Generates a descriptive name for a building object based on its ID and component count.
    /// </summary>
    private static string GetObjectTypeName(int objectId, int componentCount)
    {
        string baseName = objectId switch
        {
            0 => "Main_Building_Structure",
            1 => "Foundation_Assembly",
            2 => "Wall_System",
            3 => "Floor_Assembly", 
            4 => "Ceiling_Structure",
            5 => "Door_Assembly",
            6 => "Window_Assembly",
            7 => "Staircase_System",
            8 => "Railing_Assembly",
            9 => "Roof_Structure",
            10 => "Interior_Fixtures",
            11 => "Exterior_Details",
            12 => "Support_Structures",
            13 => "Decorative_Elements",
            14 => "Utility_Systems",
            15 => "Secondary_Structures",
            16 => "Landscape_Elements",
            17 => "Miscellaneous_Objects",
            _ => $"Building_Object_{objectId:D2}"
        };
        
        return $"{baseName}_{componentCount}Components";
    }
    
    /// <summary>
    /// Exports hierarchical objects to individual OBJ files.
    /// </summary>
    public static void ExportHierarchicalObjects(List<HierarchicalObject> objects, Pm4Scene scene, string outputRoot)
    {
        var objDir = Path.Combine(outputRoot, "hierarchical_objects");
        Directory.CreateDirectory(objDir);
        
        Console.WriteLine($"Exporting {objects.Count} hierarchical building objects to {objDir}");
        
        foreach (var obj in objects)
        {
            var objPath = Path.Combine(objDir, $"{obj.ObjectType}.obj");
            
            using var writer = new StreamWriter(objPath);
            writer.WriteLine($"# PM4 Hierarchical Building Object - ID: {obj.ObjectId}");
            writer.WriteLine($"# Component Types: {string.Join(", ", obj.ComponentTypes.Select(t => $"0x{t:X4}"))}");
            writer.WriteLine($"# Center: {obj.BoundingCenter}");
            writer.WriteLine($"# Triangles: {obj.Triangles.Count}");
            writer.WriteLine($"# Vertices: {obj.VertexCount}");
            writer.WriteLine();
            
            // Write vertices used by this object (with coordinate fix)
            var vertexMapping = new Dictionary<int, int>();
            var usedVertices = obj.Triangles
                .SelectMany(t => new[] { t.A, t.B, t.C })
                .Distinct()
                .OrderBy(i => i)
                .ToList();
            
            for (int i = 0; i < usedVertices.Count; i++)
            {
                int originalIndex = usedVertices[i];
                vertexMapping[originalIndex] = i + 1; // OBJ uses 1-based indexing
                
                Vector3 vertex;
                if (originalIndex < scene.Vertices.Count)
                {
                    vertex = scene.Vertices[originalIndex];
                }
                else
                {
                    Console.WriteLine($"Warning: Invalid vertex index {originalIndex}, max: {scene.Vertices.Count - 1}");
                    vertex = Vector3.Zero;
                }
                
                var transformedVertex = CoordinateTransformationService.ApplyPm4Transformation(vertex);
                writer.WriteLine($"v {transformedVertex.X:F6} {transformedVertex.Y:F6} {transformedVertex.Z:F6}");
            }
            
            writer.WriteLine();
            writer.WriteLine($"g {obj.ObjectType}");
            
            // Write faces
            foreach (var (a, b, c) in obj.Triangles)
            {
                writer.WriteLine($"f {vertexMapping[a]} {vertexMapping[b]} {vertexMapping[c]}");
            }
            
            writer.WriteLine();
        }
        
        Console.WriteLine($"Exported {objects.Count} hierarchical building objects");
    }
}
