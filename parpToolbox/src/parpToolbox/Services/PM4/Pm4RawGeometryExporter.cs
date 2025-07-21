using System;
using System.IO;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Exports raw PM4 geometry without any grouping to see what we're actually dealing with.
/// </summary>
internal static class Pm4RawGeometryExporter
{
    /// <summary>
    /// Export all PM4 geometry as a single OBJ file to understand what the data actually contains.
    /// </summary>
    public static void ExportRawGeometry(string pm4FilePath, string outputRoot)
    {
        var startTime = DateTime.Now;
        var fileInfo = new FileInfo(pm4FilePath);
        
        ConsoleLogger.WriteLine($"=== RAW PM4 GEOMETRY EXPORT ===");
        ConsoleLogger.WriteLine($"Input: {fileInfo.FullName}");
        ConsoleLogger.WriteLine();
        
        // Load single tile (fast)
        ConsoleLogger.WriteLine("Loading single tile...");
        var adapter = new Pm4Adapter();
        var scene = adapter.Load(pm4FilePath);
        
        ConsoleLogger.WriteLine($"Raw data: {scene.Vertices.Count:N0} vertices, {scene.Indices.Count:N0} indices");
        ConsoleLogger.WriteLine($"Surfaces: {scene.Surfaces.Count:N0}, Links: {scene.Links.Count:N0}");
        ConsoleLogger.WriteLine($"Placements: {scene.Placements.Count:N0}, Properties: {scene.Properties.Count:N0}");
        ConsoleLogger.WriteLine();
        
        // Create output directory
        var objDir = Path.Combine(outputRoot, "raw_geometry");
        Directory.CreateDirectory(objDir);
        
        // Export all geometry as single OBJ
        var objPath = Path.Combine(objDir, "PM4_Raw_All_Geometry.obj");
        ExportAllGeometry(scene, objPath);
        
        // Export geometry by surface groups (to see subdivision levels)
        ExportBySurfaceGroups(scene, objDir);
        
        var elapsed = DateTime.Now - startTime;
        ConsoleLogger.WriteLine($"Raw geometry export completed in {elapsed.TotalSeconds:F1} seconds");
        
        // Write analysis
        WriteGeometryAnalysis(scene, objDir, elapsed);
    }
    
    /// <summary>
    /// Export all geometry as a single OBJ file.
    /// </summary>
    private static void ExportAllGeometry(Pm4Scene scene, string objPath)
    {
        ConsoleLogger.WriteLine("Exporting all geometry as single OBJ...");
        
        using var writer = new StreamWriter(objPath);
        
        // Write header
        writer.WriteLine($"# PM4 Raw Geometry Export - ALL TRIANGLES");
        writer.WriteLine($"# Vertices: {scene.Vertices.Count:N0}");
        writer.WriteLine($"# Indices: {scene.Indices.Count:N0}");
        writer.WriteLine($"# Triangles: {scene.Indices.Count / 3:N0}");
        writer.WriteLine();
        
        // Write all vertices (with coordinate system fix)
        foreach (var vertex in scene.Vertices)
        {
            writer.WriteLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}"); // Fix X-axis
        }
        
        writer.WriteLine();
        writer.WriteLine("g PM4_All_Geometry");
        
        // Write all faces (convert indices to triangles)
        for (int i = 0; i < scene.Indices.Count; i += 3)
        {
            if (i + 2 < scene.Indices.Count)
            {
                int a = scene.Indices[i] + 1;     // OBJ uses 1-based indexing
                int b = scene.Indices[i + 1] + 1;
                int c = scene.Indices[i + 2] + 1;
                
                // Validate indices are within bounds
                if (a > 0 && a <= scene.Vertices.Count &&
                    b > 0 && b <= scene.Vertices.Count &&
                    c > 0 && c <= scene.Vertices.Count)
                {
                    writer.WriteLine($"f {a} {b} {c}");
                }
            }
        }
        
        ConsoleLogger.WriteLine($"Exported {scene.Indices.Count / 3:N0} triangles to: {objPath}");
    }
    
    /// <summary>
    /// Export geometry grouped by MSUR surface groups to see subdivision levels.
    /// </summary>
    private static void ExportBySurfaceGroups(Pm4Scene scene, string objDir)
    {
        ConsoleLogger.WriteLine("Exporting geometry by surface groups...");
        
        if (scene.Surfaces.Count == 0)
        {
            ConsoleLogger.WriteLine("No MSUR surfaces found - skipping surface group export");
            return;
        }
        
        var surfaceGroupDir = Path.Combine(objDir, "surface_groups");
        Directory.CreateDirectory(surfaceGroupDir);
        
        // Group surfaces by SurfaceGroupKey
        var surfaceGroups = scene.Surfaces
            .GroupBy(s => s.SurfaceGroupKey)
            .OrderBy(g => g.Key)
            .ToList();
        
        ConsoleLogger.WriteLine($"Found {surfaceGroups.Count} surface groups");
        
        foreach (var group in surfaceGroups.Take(20)) // Limit to first 20 for performance
        {
            var groupKey = group.Key;
            var surfaces = group.ToList();
            
            var objPath = Path.Combine(surfaceGroupDir, $"Surface_Group_{groupKey:X4}.obj");
            
            using var writer = new StreamWriter(objPath);
            
            writer.WriteLine($"# PM4 Surface Group Export");
            writer.WriteLine($"# Surface Group Key: 0x{groupKey:X4}");
            writer.WriteLine($"# Surfaces in group: {surfaces.Count}");
            writer.WriteLine();
            
            // Collect all vertices used by this surface group
            var usedVertexIndices = new HashSet<int>();
            var triangles = new List<(int A, int B, int C)>();
            
            foreach (var surface in surfaces)
            {
                int startIndex = (int)surface.MsviFirstIndex;
                int count = surface.IndexCount * 3; // IndexCount is triangle count, need index count
                
                for (int i = 0; i < count; i += 3)
                {
                    if (startIndex + i + 2 < scene.Indices.Count)
                    {
                        int a = scene.Indices[startIndex + i];
                        int b = scene.Indices[startIndex + i + 1];
                        int c = scene.Indices[startIndex + i + 2];
                        
                        if (a >= 0 && a < scene.Vertices.Count &&
                            b >= 0 && b < scene.Vertices.Count &&
                            c >= 0 && c < scene.Vertices.Count)
                        {
                            usedVertexIndices.Add(a);
                            usedVertexIndices.Add(b);
                            usedVertexIndices.Add(c);
                            triangles.Add((a, b, c));
                        }
                    }
                }
            }
            
            // Create vertex mapping
            var sortedVertices = usedVertexIndices.OrderBy(i => i).ToList();
            var vertexMapping = new Dictionary<int, int>();
            for (int i = 0; i < sortedVertices.Count; i++)
            {
                vertexMapping[sortedVertices[i]] = i + 1; // OBJ uses 1-based indexing
            }
            
            // Write vertices
            foreach (var originalIndex in sortedVertices)
            {
                var vertex = scene.Vertices[originalIndex];
                writer.WriteLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}"); // Fix X-axis
            }
            
            writer.WriteLine();
            writer.WriteLine($"g Surface_Group_{groupKey:X4}");
            
            // Write faces
            foreach (var (a, b, c) in triangles)
            {
                writer.WriteLine($"f {vertexMapping[a]} {vertexMapping[b]} {vertexMapping[c]}");
            }
        }
        
        ConsoleLogger.WriteLine($"Exported {Math.Min(surfaceGroups.Count, 20)} surface groups");
    }
    
    /// <summary>
    /// Write detailed analysis of the PM4 geometry data.
    /// </summary>
    private static void WriteGeometryAnalysis(Pm4Scene scene, string objDir, TimeSpan elapsed)
    {
        var analysisPath = Path.Combine(objDir, "geometry_analysis.txt");
        
        using var writer = new StreamWriter(analysisPath);
        
        writer.WriteLine("PM4 Raw Geometry Analysis");
        writer.WriteLine("=========================");
        writer.WriteLine($"Export Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine($"Export Duration: {elapsed.TotalSeconds:F1} seconds");
        writer.WriteLine();
        
        writer.WriteLine("Raw Data Counts:");
        writer.WriteLine($"  Vertices (MSVT): {scene.Vertices.Count:N0}");
        writer.WriteLine($"  Indices (MSVI): {scene.Indices.Count:N0}");
        writer.WriteLine($"  Triangles: {scene.Indices.Count / 3:N0}");
        writer.WriteLine($"  Surfaces (MSUR): {scene.Surfaces.Count:N0}");
        writer.WriteLine($"  Links (MSLK): {scene.Links.Count:N0}");
        writer.WriteLine($"  Placements (MPRL): {scene.Placements.Count:N0}");
        writer.WriteLine($"  Properties (MPRR): {scene.Properties.Count:N0}");
        writer.WriteLine();
        
        if (scene.Vertices.Count > 0)
        {
            var minX = scene.Vertices.Min(v => v.X);
            var maxX = scene.Vertices.Max(v => v.X);
            var minY = scene.Vertices.Min(v => v.Y);
            var maxY = scene.Vertices.Max(v => v.Y);
            var minZ = scene.Vertices.Min(v => v.Z);
            var maxZ = scene.Vertices.Max(v => v.Z);
            
            writer.WriteLine("Bounding Box:");
            writer.WriteLine($"  X: {minX:F2} to {maxX:F2} (range: {maxX - minX:F2})");
            writer.WriteLine($"  Y: {minY:F2} to {maxY:F2} (range: {maxY - minY:F2})");
            writer.WriteLine($"  Z: {minZ:F2} to {maxZ:F2} (range: {maxZ - minZ:F2})");
            writer.WriteLine();
        }
        
        if (scene.Surfaces.Count > 0)
        {
            var surfaceGroups = scene.Surfaces.GroupBy(s => s.SurfaceGroupKey).ToList();
            writer.WriteLine("Surface Group Analysis:");
            writer.WriteLine($"  Unique surface groups: {surfaceGroups.Count}");
            writer.WriteLine($"  Surfaces per group - Min: {surfaceGroups.Min(g => g.Count())}, Max: {surfaceGroups.Max(g => g.Count())}, Avg: {surfaceGroups.Average(g => g.Count()):F1}");
            
            writer.WriteLine();
            writer.WriteLine("Top 10 Surface Groups by Triangle Count:");
            var topGroups = surfaceGroups
                .Select(g => new { Key = g.Key, Triangles = g.Sum(s => s.IndexCount) / 3 })
                .OrderByDescending(g => g.Triangles)
                .Take(10)
                .ToList();
            
            foreach (var group in topGroups)
            {
                writer.WriteLine($"    Group 0x{group.Key:X4}: {group.Triangles:N0} triangles");
            }
        }
        
        ConsoleLogger.WriteLine($"Geometry analysis written to: {analysisPath}");
    }
}
