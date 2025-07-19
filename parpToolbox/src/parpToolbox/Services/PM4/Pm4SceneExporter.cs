namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Exports the complete PM4 scene as a single cohesive building interior,
/// rather than trying to separate individual objects.
/// </summary>
internal static class Pm4SceneExporter
{
    /// <summary>
    /// Exports the complete PM4 scene as a single OBJ file with proper material grouping.
    /// </summary>
    public static void ExportCompleteScene(Pm4Scene scene, string outputRoot)
    {
        var objPath = Path.Combine(outputRoot, "complete_scene.obj");
        
        Console.WriteLine($"Exporting complete PM4 scene to {objPath}");
        Console.WriteLine($"  Vertices: {scene.Vertices.Count}");
        Console.WriteLine($"  Indices: {scene.Indices.Count}");
        Console.WriteLine($"  Surfaces: {scene.Surfaces.Count}");
        Console.WriteLine($"  Links: {scene.Links.Count}");
        
        using var writer = new StreamWriter(objPath);
        
        // Write header
        writer.WriteLine("# Complete PM4 Scene Export");
        writer.WriteLine($"# Vertices: {scene.Vertices.Count}");
        writer.WriteLine($"# Surfaces: {scene.Surfaces.Count}");
        writer.WriteLine($"# MPRL Placements: {scene.Placements.Count}");
        writer.WriteLine($"# MSLK Links: {scene.Links.Count}");
        writer.WriteLine();
        
        // Write all vertices with coordinate system correction (flip X-axis)
        foreach (var vertex in scene.Vertices)
        {
            writer.WriteLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
        }
        writer.WriteLine();
        
        // Group geometry by surface properties for better organization
        var geometryBySurface = new Dictionary<uint, List<(int A, int B, int C)>>();
        
        // Process all MSLK links that have geometry
        foreach (var link in scene.Links.Where(l => l.MspiFirstIndex >= 0 && l.MspiIndexCount > 0))
        {
            int startIndex = link.MspiFirstIndex;
            int count = link.MspiIndexCount;
            
            if (startIndex + count <= scene.Indices.Count)
            {
                // Find corresponding surface for this geometry
                uint surfaceKey = GetSurfaceKeyForLink(link, scene);
                
                if (!geometryBySurface.TryGetValue(surfaceKey, out var triangles))
                {
                    triangles = new List<(int A, int B, int C)>();
                    geometryBySurface[surfaceKey] = triangles;
                }
                
                // Extract triangles
                for (int i = 0; i < count; i += 3)
                {
                    if (i + 2 < count)
                    {
                        int idxA = scene.Indices[startIndex + i];
                        int idxB = scene.Indices[startIndex + i + 1];
                        int idxC = scene.Indices[startIndex + i + 2];
                        
                        if (idxA < scene.Vertices.Count && idxB < scene.Vertices.Count && idxC < scene.Vertices.Count)
                        {
                            triangles.Add((idxA + 1, idxB + 1, idxC + 1)); // OBJ uses 1-based indexing
                        }
                    }
                }
            }
        }
        
        // Write geometry groups by surface type
        foreach (var (surfaceKey, triangles) in geometryBySurface.OrderBy(kvp => kvp.Key))
        {
            if (triangles.Count == 0) continue;
            
            string groupName = GetSurfaceGroupName(surfaceKey, scene);
            writer.WriteLine($"g {groupName}");
            
            foreach (var (a, b, c) in triangles)
            {
                writer.WriteLine($"f {a} {b} {c}");
            }
            
            writer.WriteLine();
            Console.WriteLine($"  Group {groupName}: {triangles.Count} triangles");
        }
        
        Console.WriteLine($"Exported complete scene with {geometryBySurface.Count} surface groups");
    }
    
    /// <summary>
    /// Gets the surface key for a given MSLK link by finding the corresponding MSUR entry.
    /// </summary>
    private static uint GetSurfaceKeyForLink(dynamic link, Pm4Scene scene)
    {
        // Try to find the MSUR entry that contains this geometry range
        foreach (var surface in scene.Surfaces)
        {
            if (surface.MsviFirstIndex <= link.MspiFirstIndex && 
                surface.MsviFirstIndex + surface.IndexCount > link.MspiFirstIndex)
            {
                return surface.SurfaceKey;
            }
        }
        
        // Fallback: use ReferenceIndex as a grouping key
        return link.ReferenceIndex;
    }
    
    /// <summary>
    /// Gets a descriptive name for a surface group.
    /// </summary>
    private static string GetSurfaceGroupName(uint surfaceKey, Pm4Scene scene)
    {
        // Find the surface entry for this key
        var surface = scene.Surfaces.FirstOrDefault(s => s.SurfaceKey == surfaceKey);
        
        if (surface != null)
        {
            // Use surface properties to generate meaningful names
            if (surface.SurfaceKey == 0x00000000)
                return "M2_Props";
            
            return $"Surface_{surfaceKey:X8}_Group_{surface.SurfaceGroupKey}";
        }
        
        // Fallback for reference-based grouping
        return surfaceKey switch
        {
            0x0000 => "Foundation",
            0x0001 => "Walls",
            0x0002 => "Floors", 
            0x0003 => "Doors",
            0x0004 => "Windows",
            0x0005 => "Ceiling",
            0x0006 => "Trim",
            0x0007 => "Pillars",
            0x0008 => "Details",
            _ when surfaceKey >= 0x0E00 && surfaceKey <= 0x0FFF => $"Furniture_{surfaceKey:X4}",
            _ => $"Component_{surfaceKey:X4}"
        };
    }
}
