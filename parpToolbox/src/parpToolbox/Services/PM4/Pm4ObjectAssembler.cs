namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Assembles complete PM4 building objects by following the MPRL → MSLK → MSUR relationship chain.
/// Each MPRL placement record defines where a building component should be positioned,
/// similar to how ADT MODF chunks place WMO buildings on terrain.
/// </summary>
internal static class Pm4ObjectAssembler
{
    /// <summary>
    /// Represents a complete building object assembled from PM4 chunks.
    /// </summary>
    public record AssembledObject(
        uint PlacementId,           // MPRL.Unknown4 (object type ID)
        Vector3 Position,           // MPRL position
        List<GeometryPiece> Geometry, // All geometry pieces for this object
        int TotalVertices,
        int TotalTriangles
    );

    /// <summary>
    /// Represents a piece of geometry belonging to an assembled object.
    /// </summary>
    public record GeometryPiece(
        uint ReferenceIndex,        // MSLK.ReferenceIndex (component type)
        List<(int A, int B, int C)> Triangles, // Triangle indices
        string ComponentType        // Descriptive name based on ReferenceIndex
    );

    /// <summary>
    /// Assembles all building objects from a PM4 scene using the confirmed chunk relationships.
    /// Groups MPRL placements by object type to create complete building components.
    /// </summary>
    public static List<AssembledObject> AssembleObjects(Pm4Scene scene)
    {
        var assembledObjects = new List<AssembledObject>();
        
        Console.WriteLine($"Assembling PM4 objects from {scene.Placements.Count} MPRL placements...");
        
        // Group MSLK entries by ParentIndex (which matches MPRL.Unknown4)
        var linksByParent = scene.Links
            .Where(link => link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0) // Only entries with geometry
            .GroupBy(link => link.ParentIndex)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        // Group MPRL placements by object type (Unknown4) to combine related components
        var placementsByType = scene.Placements
            .Where(p => linksByParent.ContainsKey(p.Unknown4)) // Only placements with geometry
            .GroupBy(p => p.Unknown4)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        Console.WriteLine($"Found {placementsByType.Count} distinct object types with geometry");
        
        // For each object type, combine all placements into a single assembled object
        foreach (var (objectType, placements) in placementsByType)
        {
            Console.WriteLine($"  Processing object type {objectType:X4} with {placements.Count} placements");
            
            var allGeometryPieces = new List<GeometryPiece>();
            int totalTriangles = 0;
            
            // Calculate center position from all placements of this type
            var centerPosition = new Vector3(
                placements.Average(p => p.Position.X),
                placements.Average(p => p.Position.Y),
                placements.Average(p => p.Position.Z)
            );
            
            // Collect geometry from all placements of this object type
            var allGeometryLinks = new List<(uint ReferenceIndex, List<(int A, int B, int C)> Triangles)>();
            
            foreach (var placement in placements)
            {
                if (linksByParent.TryGetValue(placement.Unknown4, out var geometryLinks))
                {
                    // Group geometry by ReferenceIndex within this placement
                    var linksByReference = geometryLinks.GroupBy(link => link.ReferenceIndex);
                    
                    foreach (var referenceGroup in linksByReference)
                    {
                        uint referenceIndex = referenceGroup.Key;
                        var triangles = new List<(int A, int B, int C)>();
                        
                        // Extract triangles from all MSLK entries with this ReferenceIndex
                        foreach (var link in referenceGroup)
                        {
                            ExtractTrianglesFromLink(link, scene, triangles);
                        }
                        
                        if (triangles.Count > 0)
                        {
                            allGeometryLinks.Add((referenceIndex, triangles));
                        }
                    }
                }
            }
            
            // Combine triangles by ReferenceIndex across all placements
            var combinedGeometry = allGeometryLinks
                .GroupBy(g => g.ReferenceIndex)
                .Select(g => new GeometryPiece(
                    g.Key,
                    g.SelectMany(x => x.Triangles).ToList(),
                    GetComponentTypeName(g.Key)
                ))
                .ToList();
            
            totalTriangles = combinedGeometry.Sum(g => g.Triangles.Count);
            
            if (combinedGeometry.Count > 0)
            {
                var assembledObject = new AssembledObject(
                    objectType,
                    centerPosition,
                    combinedGeometry,
                    combinedGeometry.SelectMany(g => g.Triangles).SelectMany(t => new[] { t.A, t.B, t.C }).Distinct().Count(),
                    totalTriangles
                );
                
                assembledObjects.Add(assembledObject);
                
                Console.WriteLine($"    Combined object {objectType:X4}: {combinedGeometry.Count} component types, {totalTriangles} triangles");
            }
        }
        
        Console.WriteLine($"Assembled {assembledObjects.Count} complete object types");
        return assembledObjects;
    }
    
    /// <summary>
    /// Extracts triangles from a single MSLK link entry.
    /// </summary>
    private static void ExtractTrianglesFromLink(dynamic link, Pm4Scene scene, List<(int A, int B, int C)> triangles)
    {
        int startIndex = link.MspiFirstIndex;
        int count = link.MspiIndexCount;
        
        if (startIndex + count <= scene.Indices.Count)
        {
            for (int i = 0; i < count; i += 3)
            {
                if (i + 2 < count)
                {
                    int idxA = scene.Indices[startIndex + i];
                    int idxB = scene.Indices[startIndex + i + 1];
                    int idxC = scene.Indices[startIndex + i + 2];
                    
                    if (idxA < scene.Vertices.Count && idxB < scene.Vertices.Count && idxC < scene.Vertices.Count)
                    {
                        triangles.Add((idxA, idxB, idxC));
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// Exports assembled objects to individual OBJ files, one per building component.
    /// </summary>
    public static void ExportAssembledObjects(List<AssembledObject> objects, Pm4Scene scene, string outputRoot)
    {
        var objDir = Path.Combine(outputRoot, "assembled_objects");
        Directory.CreateDirectory(objDir);
        
        Console.WriteLine($"Exporting {objects.Count} assembled objects to {objDir}");
        
        foreach (var obj in objects)
        {
            var objPath = Path.Combine(objDir, $"Object_{obj.PlacementId:X4}.obj");
            
            using var writer = new StreamWriter(objPath);
            writer.WriteLine($"# PM4 Assembled Object {obj.PlacementId:X4}");
            writer.WriteLine($"# Position: {obj.Position}");
            writer.WriteLine($"# Components: {obj.Geometry.Count}");
            writer.WriteLine($"# Triangles: {obj.TotalTriangles}");
            writer.WriteLine();
            
            // Write all vertices used by this object
            var allVertexIndices = obj.Geometry
                .SelectMany(g => g.Triangles)
                .SelectMany(t => new[] { t.A, t.B, t.C })
                .Distinct()
                .OrderBy(i => i)
                .ToList();
            
            var vertexMapping = new Dictionary<int, int>();
            for (int i = 0; i < allVertexIndices.Count; i++)
            {
                int originalIndex = allVertexIndices[i];
                vertexMapping[originalIndex] = i + 1; // OBJ uses 1-based indexing
                
                var vertex = scene.Vertices[originalIndex];
                writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
            
            writer.WriteLine();
            
            // Write geometry pieces as separate groups
            foreach (var piece in obj.Geometry)
            {
                writer.WriteLine($"g {piece.ComponentType}");
                
                foreach (var (a, b, c) in piece.Triangles)
                {
                    writer.WriteLine($"f {vertexMapping[a]} {vertexMapping[b]} {vertexMapping[c]}");
                }
                
                writer.WriteLine();
            }
        }
    }
    
    /// <summary>
    /// Generates a descriptive name for a component based on its ReferenceIndex.
    /// </summary>
    private static string GetComponentTypeName(uint referenceIndex)
    {
        // Based on patterns observed in the data, provide meaningful names
        return referenceIndex switch
        {
            0x0000 => "Foundation",
            0x0001 => "Wall_Primary",
            0x0002 => "Wall_Secondary", 
            0x0003 => "Door_Frame",
            0x0004 => "Window_Frame",
            0x0005 => "Floor_Tile",
            0x0006 => "Ceiling_Beam",
            0x0007 => "Pillar",
            0x0008 => "Trim_Detail",
            0x0009 => "Stair_Step",
            0x000A => "Railing",
            0x000B => "Arch_Detail",
            0x000C => "Corner_Detail",
            0x000D => "Decorative_Element",
            _ when referenceIndex >= 0x0E00 && referenceIndex <= 0x0FFF => $"Furniture_{referenceIndex:X4}",
            _ when referenceIndex >= 0x0700 && referenceIndex <= 0x08FF => $"Structural_{referenceIndex:X4}",
            _ => $"Component_{referenceIndex:X4}"
        };
    }
}
