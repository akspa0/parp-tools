using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Tile-based PM4 exporter that processes objects per-tile for better performance and memory usage.
/// </summary>
internal static class Pm4TileBasedExporter
{
    /// <summary>
    /// Export PM4 objects using per-tile processing for optimal performance.
    /// </summary>
    public static void ExportTileBased(string pm4FilePath, string outputRoot, int maxObjectsPerTile = 50)
    {
        var startTime = DateTime.Now;
        var fileInfo = new FileInfo(pm4FilePath);
        
        ConsoleLogger.WriteLine($"=== TILE-BASED PM4 OBJECT EXPORT ===");
        ConsoleLogger.WriteLine($"Input: {fileInfo.FullName}");
        ConsoleLogger.WriteLine($"Max objects per tile: {maxObjectsPerTile}");
        ConsoleLogger.WriteLine();
        
        // Load single tile (much faster)
        ConsoleLogger.WriteLine("Loading single tile (fast mode)...");
        var adapter = new Pm4Adapter();
        var scene = adapter.Load(pm4FilePath);
        
        ConsoleLogger.WriteLine($"Loaded: {scene.Vertices.Count:N0} vertices, {scene.Indices.Count:N0} indices");
        ConsoleLogger.WriteLine($"Properties: {scene.Properties.Count:N0}, Placements: {scene.Placements.Count:N0}");
        
        if (scene.Properties.Count == 0)
        {
            ConsoleLogger.WriteLine("No MPRR properties found - cannot perform object grouping");
            return;
        }
        
        // Parse MPRR object groups (lightweight)
        var objectGroups = ParseMprrObjectGroupsLightweight(scene);
        ConsoleLogger.WriteLine($"Found {objectGroups.Count} object groups in this tile");
        
        if (objectGroups.Count == 0)
        {
            ConsoleLogger.WriteLine("No object groups found");
            return;
        }
        
        // Limit objects for performance
        var limitedGroups = objectGroups.Take(maxObjectsPerTile).ToList();
        ConsoleLogger.WriteLine($"Processing {limitedGroups.Count} object groups (limited for performance)");
        ConsoleLogger.WriteLine();
        
        // Create output directory
        var objDir = Path.Combine(outputRoot, "tile_objects");
        Directory.CreateDirectory(objDir);
        
        // Export objects with streaming approach
        int exportedCount = 0;
        var totalTriangles = 0L;
        
        foreach (var (objectId, componentTypes) in limitedGroups)
        {
            try
            {
                var triangles = ExtractObjectTriangles(scene, componentTypes);
                if (triangles.Count == 0) continue;
                
                var objPath = Path.Combine(objDir, $"Tile_Object_{exportedCount:D3}_{objectId:X4}.obj");
                ExportObjectStreaming(scene, triangles, objPath, objectId, componentTypes, exportedCount);
                
                exportedCount++;
                totalTriangles += triangles.Count;
                
                if (exportedCount % 10 == 0)
                {
                    ConsoleLogger.WriteLine($"  Exported {exportedCount}/{limitedGroups.Count} objects...");
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error exporting object {objectId:X4}: {ex.Message}");
            }
        }
        
        var elapsed = DateTime.Now - startTime;
        
        ConsoleLogger.WriteLine();
        ConsoleLogger.WriteLine($"Export completed in {elapsed.TotalSeconds:F1} seconds");
        ConsoleLogger.WriteLine($"Exported {exportedCount} objects with {totalTriangles:N0} total triangles");
        ConsoleLogger.WriteLine($"Average: {elapsed.TotalMilliseconds / Math.Max(exportedCount, 1):F0}ms per object");
        
        // Write summary
        WriteTileSummary(objDir, exportedCount, totalTriangles, elapsed, fileInfo.Name);
    }
    
    /// <summary>
    /// Parse MPRR object groups without heavy processing - just get the groupings.
    /// </summary>
    private static List<(int ObjectId, List<ushort> ComponentTypes)> ParseMprrObjectGroupsLightweight(Pm4Scene scene)
    {
        var objectGroups = new List<(int, List<ushort>)>();
        var currentComponents = new List<ushort>();
        int objectId = 0;
        
        foreach (var property in scene.Properties)
        {
            if (property.Value1 == 65535) // Sentinel marker
            {
                if (currentComponents.Count > 0)
                {
                    objectGroups.Add((objectId++, currentComponents.ToList()));
                    currentComponents.Clear();
                }
            }
            else
            {
                currentComponents.Add(property.Value2);
            }
        }
        
        // Add final group if exists
        if (currentComponents.Count > 0)
        {
            objectGroups.Add((objectId, currentComponents.ToList()));
        }
        
        return objectGroups;
    }
    
    /// <summary>
    /// Extract triangles for a specific object using component types.
    /// </summary>
    private static List<(int A, int B, int C)> ExtractObjectTriangles(Pm4Scene scene, List<ushort> componentTypes)
    {
        var triangles = new List<(int A, int B, int C)>();
        
        // Map component types to geometry via MSLK links
        var linksByParentIndex = scene.Links
            .Where(link => link.ParentIndex > 0 && link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
            .GroupBy(link => link.ParentIndex)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        foreach (var componentType in componentTypes)
        {
            if (linksByParentIndex.TryGetValue(componentType, out var geometryLinks))
            {
                foreach (var link in geometryLinks)
                {
                    int startIndex = link.MspiFirstIndex;
                    int count = link.MspiIndexCount;
                    
                    // Extract triangles from this geometry link
                    for (int i = 0; i < count; i += 3)
                    {
                        if (startIndex + i + 2 < scene.Indices.Count)
                        {
                            int idxA = scene.Indices[startIndex + i];
                            int idxB = scene.Indices[startIndex + i + 1];
                            int idxC = scene.Indices[startIndex + i + 2];
                            
                            // Validate indices are within bounds
                            if (idxA >= 0 && idxA < scene.Vertices.Count &&
                                idxB >= 0 && idxB < scene.Vertices.Count &&
                                idxC >= 0 && idxC < scene.Vertices.Count)
                            {
                                triangles.Add((idxA, idxB, idxC));
                            }
                        }
                    }
                }
            }
        }
        
        return triangles;
    }
    
    /// <summary>
    /// Export a single object using streaming approach to minimize memory usage.
    /// </summary>
    private static void ExportObjectStreaming(
        Pm4Scene scene, 
        List<(int A, int B, int C)> triangles, 
        string objPath,
        int objectId,
        List<ushort> componentTypes,
        int exportIndex)
    {
        // Build vertex mapping for this object only
        var usedVertexIndices = triangles
            .SelectMany(t => new[] { t.A, t.B, t.C })
            .Distinct()
            .OrderBy(idx => idx)
            .ToList();
        
        var vertexMapping = new Dictionary<int, int>();
        for (int i = 0; i < usedVertexIndices.Count; i++)
        {
            vertexMapping[usedVertexIndices[i]] = i + 1; // OBJ uses 1-based indexing
        }
        
        // Stream write to OBJ file
        using var writer = new StreamWriter(objPath);
        
        // Write header
        writer.WriteLine($"# PM4 Tile-Based Object Export");
        writer.WriteLine($"# Export Index: {exportIndex}");
        writer.WriteLine($"# Object ID: 0x{objectId:X4}");
        writer.WriteLine($"# Component Types: {string.Join(", ", componentTypes.Select(t => $"0x{t:X4}"))}");
        writer.WriteLine($"# Triangles: {triangles.Count:N0}");
        writer.WriteLine($"# Vertices: {usedVertexIndices.Count:N0}");
        writer.WriteLine();
        
        // Stream vertices (with coordinate system fix)
        foreach (var originalIndex in usedVertexIndices)
        {
            var vertex = scene.Vertices[originalIndex];
            var transformedVertex = CoordinateTransformationService.ApplyPm4Transformation(vertex);
            writer.WriteLine($"v {transformedVertex.X:F6} {transformedVertex.Y:F6} {transformedVertex.Z:F6}");
        }
        
        writer.WriteLine();
        writer.WriteLine($"g Tile_Object_{exportIndex:D3}");
        
        // Stream faces with remapped indices
        foreach (var (a, b, c) in triangles)
        {
            if (vertexMapping.TryGetValue(a, out int newA) &&
                vertexMapping.TryGetValue(b, out int newB) &&
                vertexMapping.TryGetValue(c, out int newC))
            {
                writer.WriteLine($"f {newA} {newB} {newC}");
            }
        }
    }
    
    /// <summary>
    /// Write export summary for tile-based export.
    /// </summary>
    private static void WriteTileSummary(string objDir, int exportedCount, long totalTriangles, TimeSpan elapsed, string tileFileName)
    {
        var summaryPath = Path.Combine(objDir, "tile_export_summary.txt");
        
        using var writer = new StreamWriter(summaryPath);
        
        writer.WriteLine("PM4 Tile-Based Object Export Summary");
        writer.WriteLine("===================================");
        writer.WriteLine($"Source Tile: {tileFileName}");
        writer.WriteLine($"Export Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine($"Export Duration: {elapsed.TotalSeconds:F1} seconds");
        writer.WriteLine();
        
        writer.WriteLine("Performance:");
        writer.WriteLine($"  Objects exported: {exportedCount:N0}");
        writer.WriteLine($"  Total triangles: {totalTriangles:N0}");
        writer.WriteLine($"  Export rate: {exportedCount / elapsed.TotalSeconds:F1} objects/second");
        writer.WriteLine($"  Triangle rate: {totalTriangles / elapsed.TotalSeconds:F0} triangles/second");
        writer.WriteLine($"  Average per object: {elapsed.TotalMilliseconds / Math.Max(exportedCount, 1):F0}ms");
        
        ConsoleLogger.WriteLine($"Tile export summary written to: {summaryPath}");
    }
}
