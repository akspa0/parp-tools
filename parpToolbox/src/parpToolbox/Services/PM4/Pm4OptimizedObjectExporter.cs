using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Optimized exporter for PM4 hierarchical objects with performance improvements and limits.
/// </summary>
internal static class Pm4OptimizedObjectExporter
{
    /// <summary>
    /// Export hierarchical objects with performance optimizations and configurable limits.
    /// </summary>
    public static void ExportOptimized(
        List<Pm4HierarchicalObjectAssembler.HierarchicalObject> objects, 
        Pm4Scene scene, 
        string outputRoot,
        int maxObjects = 100,
        int maxTrianglesPerObject = 100000,
        bool useParallelExport = true)
    {
        var objDir = Path.Combine(outputRoot, "hierarchical_objects");
        Directory.CreateDirectory(objDir);
        
        ConsoleLogger.WriteLine($"=== OPTIMIZED PM4 OBJECT EXPORT ===");
        ConsoleLogger.WriteLine($"Total objects available: {objects.Count}");
        ConsoleLogger.WriteLine($"Export limit: {maxObjects} objects");
        ConsoleLogger.WriteLine($"Triangle limit per object: {maxTrianglesPerObject:N0}");
        ConsoleLogger.WriteLine($"Parallel export: {useParallelExport}");
        
        // Filter objects by size and limit count
        var filteredObjects = objects
            .Where(obj => obj.Triangles.Count <= maxTrianglesPerObject)
            .Take(maxObjects)
            .ToList();
        
        var skippedLarge = objects.Count(obj => obj.Triangles.Count > maxTrianglesPerObject);
        var skippedLimit = Math.Max(0, objects.Count - skippedLarge - filteredObjects.Count);
        
        ConsoleLogger.WriteLine($"Objects to export: {filteredObjects.Count}");
        ConsoleLogger.WriteLine($"Skipped (too large): {skippedLarge}");
        ConsoleLogger.WriteLine($"Skipped (limit): {skippedLimit}");
        ConsoleLogger.WriteLine();
        
        if (filteredObjects.Count == 0)
        {
            ConsoleLogger.WriteLine("No objects to export after filtering!");
            return;
        }
        
        // Export objects
        var startTime = DateTime.Now;
        
        if (useParallelExport && filteredObjects.Count > 1)
        {
            ExportParallel(filteredObjects, scene, objDir);
        }
        else
        {
            ExportSequential(filteredObjects, scene, objDir);
        }
        
        var elapsed = DateTime.Now - startTime;
        ConsoleLogger.WriteLine($"Export completed in {elapsed.TotalSeconds:F1} seconds");
        ConsoleLogger.WriteLine($"Average: {elapsed.TotalMilliseconds / filteredObjects.Count:F0}ms per object");
        
        // Write summary
        WriteSummaryReport(filteredObjects, objects, objDir, elapsed);
    }
    
    private static void ExportSequential(
        List<Pm4HierarchicalObjectAssembler.HierarchicalObject> objects,
        Pm4Scene scene,
        string objDir)
    {
        ConsoleLogger.WriteLine("Sequential export...");
        
        for (int i = 0; i < objects.Count; i++)
        {
            var obj = objects[i];
            
            if (i % 10 == 0 || i == objects.Count - 1)
            {
                ConsoleLogger.WriteLine($"  Progress: {i + 1}/{objects.Count} ({(i + 1) * 100.0 / objects.Count:F1}%)");
            }
            
            ExportSingleObject(obj, scene, objDir, i);
        }
    }
    
    private static void ExportParallel(
        List<Pm4HierarchicalObjectAssembler.HierarchicalObject> objects,
        Pm4Scene scene,
        string objDir)
    {
        ConsoleLogger.WriteLine("Parallel export...");
        
        var progress = 0;
        var lockObj = new object();
        
        Parallel.ForEach(objects, (obj, state, index) =>
        {
            ExportSingleObject(obj, scene, objDir, (int)index);
            
            lock (lockObj)
            {
                progress++;
                if (progress % 10 == 0 || progress == objects.Count)
                {
                    ConsoleLogger.WriteLine($"  Progress: {progress}/{objects.Count} ({progress * 100.0 / objects.Count:F1}%)");
                }
            }
        });
    }
    
    private static void ExportSingleObject(
        Pm4HierarchicalObjectAssembler.HierarchicalObject obj,
        Pm4Scene scene,
        string objDir,
        int index)
    {
        try
        {
            var objPath = Path.Combine(objDir, $"Building_Object_{index:D4}_{obj.ObjectId:X4}.obj");
            
            using var writer = new StreamWriter(objPath);
            
            // Write header
            writer.WriteLine($"# PM4 Hierarchical Building Object");
            writer.WriteLine($"# Export Index: {index}");
            writer.WriteLine($"# Object ID: 0x{obj.ObjectId:X4}");
            writer.WriteLine($"# Component Types: {string.Join(", ", obj.ComponentTypes.Select(t => $"0x{t:X4}"))}");
            writer.WriteLine($"# Center: {obj.BoundingCenter.X:F2}, {obj.BoundingCenter.Y:F2}, {obj.BoundingCenter.Z:F2}");
            writer.WriteLine($"# Triangles: {obj.Triangles.Count:N0}");
            writer.WriteLine($"# Vertices: {obj.VertexCount:N0}");
            writer.WriteLine();
            
            // Build vertex mapping for this object only
            var usedVertexIndices = obj.Triangles
                .SelectMany(t => new[] { t.A, t.B, t.C })
                .Where(idx => idx >= 0 && idx < scene.Vertices.Count)
                .Distinct()
                .OrderBy(idx => idx)
                .ToList();
            
            var vertexMapping = new Dictionary<int, int>();
            for (int i = 0; i < usedVertexIndices.Count; i++)
            {
                vertexMapping[usedVertexIndices[i]] = i + 1; // OBJ uses 1-based indexing
            }
            
            // Write vertices (with coordinate system fix)
            foreach (var originalIndex in usedVertexIndices)
            {
                var vertex = scene.Vertices[originalIndex];
                writer.WriteLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}"); // Fix X-axis
            }
            
            writer.WriteLine();
            writer.WriteLine($"g Building_Object_{index:D4}");
            
            // Write faces with remapped indices
            foreach (var (a, b, c) in obj.Triangles)
            {
                if (vertexMapping.TryGetValue(a, out int newA) &&
                    vertexMapping.TryGetValue(b, out int newB) &&
                    vertexMapping.TryGetValue(c, out int newC))
                {
                    writer.WriteLine($"f {newA} {newB} {newC}");
                }
            }
        }
        catch (Exception ex)
        {
            ConsoleLogger.WriteLine($"Error exporting object {index} (ID: 0x{obj.ObjectId:X4}): {ex.Message}");
        }
    }
    
    private static void WriteSummaryReport(
        List<Pm4HierarchicalObjectAssembler.HierarchicalObject> exportedObjects,
        List<Pm4HierarchicalObjectAssembler.HierarchicalObject> allObjects,
        string objDir,
        TimeSpan elapsed)
    {
        var summaryPath = Path.Combine(objDir, "export_summary.txt");
        
        using var writer = new StreamWriter(summaryPath);
        
        writer.WriteLine("PM4 Hierarchical Object Export Summary");
        writer.WriteLine("=====================================");
        writer.WriteLine($"Export Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine($"Export Duration: {elapsed.TotalSeconds:F1} seconds");
        writer.WriteLine();
        
        writer.WriteLine("Statistics:");
        writer.WriteLine($"  Total objects available: {allObjects.Count:N0}");
        writer.WriteLine($"  Objects exported: {exportedObjects.Count:N0}");
        writer.WriteLine($"  Export rate: {exportedObjects.Count / elapsed.TotalSeconds:F1} objects/second");
        writer.WriteLine();
        
        if (exportedObjects.Count > 0)
        {
            var triangleCounts = exportedObjects.Select(obj => obj.Triangles.Count).ToList();
            var vertexCounts = exportedObjects.Select(obj => obj.VertexCount).ToList();
            var componentCounts = exportedObjects.Select(obj => obj.ComponentTypes.Count).ToList();
            
            writer.WriteLine("Exported Object Statistics:");
            writer.WriteLine($"  Triangle count - Min: {triangleCounts.Min():N0}, Max: {triangleCounts.Max():N0}, Avg: {triangleCounts.Average():F0}");
            writer.WriteLine($"  Vertex count - Min: {vertexCounts.Min():N0}, Max: {vertexCounts.Max():N0}, Avg: {vertexCounts.Average():F0}");
            writer.WriteLine($"  Component count - Min: {componentCounts.Min()}, Max: {componentCounts.Max()}, Avg: {componentCounts.Average():F1}");
            writer.WriteLine();
        }
        
        writer.WriteLine("Exported Objects:");
        writer.WriteLine("Index | Object ID | Triangles | Vertices | Components | File");
        writer.WriteLine("------|-----------|-----------|----------|------------|-----");
        
        for (int i = 0; i < exportedObjects.Count; i++)
        {
            var obj = exportedObjects[i];
            var fileName = $"Building_Object_{i:D4}_{obj.ObjectId:X4}.obj";
            writer.WriteLine($"{i,5} | 0x{obj.ObjectId:X6} | {obj.Triangles.Count,9:N0} | {obj.VertexCount,8:N0} | {obj.ComponentTypes.Count,10} | {fileName}");
        }
        
        ConsoleLogger.WriteLine($"Export summary written to: {summaryPath}");
    }
}
