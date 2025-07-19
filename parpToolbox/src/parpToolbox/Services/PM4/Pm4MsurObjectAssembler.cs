namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Assembles PM4 objects using MSUR IndexCount (0x01 field) to identify individual objects,
/// as documented in legacy notes: "individual objects are identified by the MSUR index (0x01 in the chunk)".
/// </summary>
internal static class Pm4MsurObjectAssembler
{
    /// <summary>
    /// Represents a complete building object assembled using MSUR SurfaceKey grouping.
    /// </summary>
    public record MsurObject(
        uint SurfaceKey,               // MSUR.SurfaceKey - the actual object identifier
        int SurfaceCount,              // Number of surfaces in this object
        List<(int A, int B, int C)> Triangles, // All triangles for this object
        Vector3 BoundingCenter,        // Calculated center point
        int VertexCount,
        string ObjectType              // Descriptive name based on surface properties
    );

    /// <summary>
    /// Assembles PM4 objects by grouping geometry using MSUR SurfaceKey as the object identifier.
    /// </summary>
    public static List<MsurObject> AssembleObjectsByMsurIndex(Pm4Scene scene)
    {
        var assembledObjects = new List<MsurObject>();
        
        Console.WriteLine($"Assembling PM4 objects using MSUR SurfaceKey grouping (corrected method)...");
        Console.WriteLine($"  MSUR surfaces: {scene.Surfaces.Count}");
        Console.WriteLine($"  MSLK links: {scene.Links.Count}");
        
        // Group surfaces by SurfaceKey (the actual object identifier) - use ALL surfaces, not just those with MSLK links
        var surfacesBySurfaceKey = scene.Surfaces
            .Select((surface, index) => new { Surface = surface, Index = index })
            .Where(x => x.Surface.IndexCount > 0) // Only surfaces with geometry
            .GroupBy(x => x.Surface.SurfaceKey)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        Console.WriteLine($"  Found {surfacesBySurfaceKey.Count} distinct SurfaceKey groups (complete objects)");
        
        // Assemble each object by combining all surfaces with the same SurfaceKey
        foreach (var (surfaceKey, surfaceGroup) in surfacesBySurfaceKey)
        {
            var allTriangles = new List<(int A, int B, int C)>();
            var allVertexIndices = new HashSet<int>();
            
            Console.WriteLine($"    Processing SurfaceKey 0x{surfaceKey:X8} with {surfaceGroup.Count} surfaces");
            
            foreach (var surfaceInfo in surfaceGroup)
            {
                var surface = surfaceInfo.Surface;
                
                // Extract geometry directly from MSUR using MSVI (secondary indices)
                ExtractTrianglesFromMsur(surface, scene, allTriangles, allVertexIndices);
            }
            
            if (allTriangles.Count > 0)
            {
                // Calculate bounding center from all vertices used by this object
                var boundingCenter = CalculateBoundingCenter(allVertexIndices, scene);
                var objectType = GetObjectTypeName(surfaceKey);
                
                var msurObject = new MsurObject(
                    surfaceKey,
                    surfaceGroup.Count,
                    allTriangles,
                    boundingCenter,
                    allVertexIndices.Count,
                    objectType
                );
                
                assembledObjects.Add(msurObject);
                
                Console.WriteLine($"      Object SurfaceKey=0x{surfaceKey:X8}: {allTriangles.Count} triangles, {allVertexIndices.Count} vertices, {surfaceGroup.Count} surfaces");
            }
        }
        
        Console.WriteLine($"Assembled {assembledObjects.Count} complete building objects");
        return assembledObjects;
    }
    
    /// <summary>
    /// Extracts triangles directly from a MSUR surface using MSVI indices.
    /// The PM4 adapter already combines vertex pools, so we use the unified Vertices and Indices arrays.
    /// </summary>
    private static void ExtractTrianglesFromMsur(dynamic surface, Pm4Scene scene, 
        List<(int A, int B, int C)> triangles, HashSet<int> vertexIndices)
    {
        int startIndex = (int)surface.MsviFirstIndex;
        int count = surface.IndexCount;
        
        // MSUR surfaces reference MSVI indices (scene.Indices contains the MSVI data)
        if (startIndex >= 0 && startIndex + count <= scene.Indices.Count)
        {
            for (int i = 0; i < count; i += 3)
            {
                if (i + 2 < count)
                {
                    int idxA = scene.Indices[startIndex + i];
                    int idxB = scene.Indices[startIndex + i + 1];
                    int idxC = scene.Indices[startIndex + i + 2];
                    
                    // Validate indices against the unified vertex array
                    if (idxA < scene.Vertices.Count && 
                        idxB < scene.Vertices.Count && 
                        idxC < scene.Vertices.Count)
                    {
                        triangles.Add((idxA, idxB, idxC));
                        vertexIndices.Add(idxA);
                        vertexIndices.Add(idxB);
                        vertexIndices.Add(idxC);
                    }
                    else
                    {
                        Console.WriteLine($"      Warning: Invalid triangle indices {idxA},{idxB},{idxC} for surface 0x{surface.SurfaceKey:X8} (max vertex: {scene.Vertices.Count - 1})");
                    }
                }
            }
        }
        else if (count > 0)
        {
            Console.WriteLine($"      Warning: MSUR surface 0x{surface.SurfaceKey:X8} has invalid MSVI range: start={startIndex}, count={count}, available={scene.Indices.Count}");
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
    /// Generates a descriptive name for an object based on its SurfaceKey.
    /// </summary>
    private static string GetObjectTypeName(uint surfaceKey)
    {
        // Use SurfaceKey patterns to identify object types
        string baseName = surfaceKey switch
        {
            0x40AA0A7E => "Building_Structure_A",
            0x418D9F7C => "Building_Structure_B", 
            0x4218098A => "Building_Structure_C",
            0x41D4B116 => "Stair_Component",
            0x410C11B5 => "Floor_Element",
            0x412CDCD9 => "Wall_Section",
            0x4158810A => "Ceiling_Structure",
            0x40B02353 => "Door_Frame",
            0x409E75E0 => "Window_Frame",
            0x40DB09E1 => "Pillar_Support",
            0x407F219F => "Arch_Feature",
            0x41796495 => "Trim_Detail",
            0x00000000 => "M2_Props",
            _ => $"Building_Object_{surfaceKey:X8}"
        };
        
        return baseName;
    }
    
    /// <summary>
    /// Exports MSUR-based objects to individual OBJ files.
    /// </summary>
    public static void ExportMsurObjects(List<MsurObject> objects, Pm4Scene scene, string outputRoot)
    {
        var objDir = Path.Combine(outputRoot, "msur_objects");
        Directory.CreateDirectory(objDir);
        
        Console.WriteLine($"Exporting {objects.Count} MSUR-based objects to {objDir}");
        
        foreach (var obj in objects)
        {
            var objPath = Path.Combine(objDir, $"{obj.ObjectType}.obj");
            
            using var writer = new StreamWriter(objPath);
            writer.WriteLine($"# PM4 MSUR Object - SurfaceKey: 0x{obj.SurfaceKey:X8}");
            writer.WriteLine($"# Surface Count: {obj.SurfaceCount}");
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
                
                // The PM4 adapter already provides unified vertex data in scene.Vertices
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
                
                writer.WriteLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}"); // Fix X-axis
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
        
        Console.WriteLine($"Exported {objects.Count} MSUR-based objects");
    }
}
