using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Clean, simple PM4 exporter that exports each surface group as a separate building object.
/// Surface groups (MSUR.SurfaceGroupKey) are the correct object boundaries.
/// </summary>
internal static class Pm4SurfaceGroupExporter
{
    /// <summary>
    /// Export PM4 surface groups as individual building objects.
    /// </summary>
    public static void ExportSurfaceGroups(string pm4FilePath, string outputRoot)
    {
        var startTime = DateTime.Now;
        var fileInfo = new FileInfo(pm4FilePath);
        
        ConsoleLogger.WriteLine($"=== PM4 SURFACE GROUP EXPORT (CORRECT OBJECT GROUPING) ===");
        ConsoleLogger.WriteLine($"Input: {fileInfo.FullName}");
        ConsoleLogger.WriteLine();
        
        // Load PM4 data
        ConsoleLogger.WriteLine("Loading PM4 data...");
        var adapter = new Pm4Adapter();
        var scene = adapter.Load(pm4FilePath);
        
        ExportSurfaceGroupsFromScene(scene, outputRoot, fileInfo.Name);
    }

    /// <summary>
    /// Export PM4 surface groups as individual building objects from an existing Pm4Scene.
    /// </summary>
    public static void ExportSurfaceGroupsFromScene(Pm4Scene scene, string outputRoot, string fileName = "scene")
    {
        var startTime = DateTime.Now;
        
        ConsoleLogger.WriteLine($"=== PM4 SURFACE GROUP EXPORT (CORRECT OBJECT GROUPING) ===");
        ConsoleLogger.WriteLine($"Input: {fileName}");
        ConsoleLogger.WriteLine();
        
        ConsoleLogger.WriteLine($"Data: {scene.Vertices.Count:N0} vertices, {scene.Indices.Count:N0} indices");
        ConsoleLogger.WriteLine($"Surfaces: {scene.Surfaces.Count:N0}");
        ConsoleLogger.WriteLine();
        
        // Group surfaces by SurfaceGroupKey (this IS the correct object grouping)
        var surfaceGroups = scene.Surfaces
            .GroupBy(s => s.SurfaceGroupKey)
            .OrderByDescending(g => g.Sum(s => s.IndexCount)) // Largest objects first
            .ToList();
        
        ConsoleLogger.WriteLine($"Found {surfaceGroups.Count} building objects (surface groups):");
        foreach (var group in surfaceGroups)
        {
            var triangleCount = group.Sum(s => s.IndexCount) / 3;
            ConsoleLogger.WriteLine($"  Building 0x{group.Key:X4}: {triangleCount:N0} triangles ({group.Count()} surfaces)");
        }
        ConsoleLogger.WriteLine();
        
        // Create output directory
        var objDir = Path.Combine(outputRoot, "buildings");
        Directory.CreateDirectory(objDir);
        
        // Export each surface group as a separate building
        int buildingIndex = 0;
        foreach (var group in surfaceGroups)
        {
            try
            {
                ExportSingleBuilding(scene, group.ToList(), objDir, (byte)group.Key, buildingIndex);
                buildingIndex++;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error exporting building 0x{group.Key:X4}: {ex.Message}");
            }
        }
        
        var elapsed = DateTime.Now - startTime;
        ConsoleLogger.WriteLine();
        ConsoleLogger.WriteLine($"=== EXPORT COMPLETE ({elapsed.TotalSeconds:F1}s) ===");
        ConsoleLogger.WriteLine($"Exported {surfaceGroups.Count} building objects to {objDir}");
        
        // Write summary
        WriteBuildingSummary(surfaceGroups, objDir, elapsed, fileName);
    }
    
    /// <summary>
    /// Export a single building (surface group) as an OBJ file.
    /// </summary>
    private static void ExportSingleBuilding(
        Pm4Scene scene, 
        List<MsurChunk.Entry> surfaces, 
        string objDir, 
        byte groupKey, 
        int buildingIndex)
    {
        // Collect all vertices and triangles for this building
        var usedVertexIndices = new HashSet<int>();
        var triangles = new List<(int A, int B, int C)>();
        
        foreach (var surface in surfaces)
        {
            int startIndex = (int)surface.MsviFirstIndex;
            int indexCount = surface.IndexCount * 3; // IndexCount is triangle count, need index count
            
            for (int i = 0; i < indexCount; i += 3)
            {
                if (startIndex + i + 2 < scene.Indices.Count)
                {
                    int a = scene.Indices[startIndex + i];
                    int b = scene.Indices[startIndex + i + 1];
                    int c = scene.Indices[startIndex + i + 2];
                    
                    // Validate indices are within bounds
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
        
        if (triangles.Count == 0)
        {
            ConsoleLogger.WriteLine($"Warning: Building 0x{groupKey:X4} has no valid triangles");
            return;
        }
        
        // Create vertex mapping for this building
        var sortedVertices = usedVertexIndices.OrderBy(i => i).ToList();
        var vertexMapping = new Dictionary<int, int>();
        for (int i = 0; i < sortedVertices.Count; i++)
        {
            vertexMapping[sortedVertices[i]] = i + 1; // OBJ uses 1-based indexing
        }
        
        // Calculate building center for naming (with coordinate system fix)
        var buildingCenter = Vector3.Zero;
        foreach (var vertexIndex in sortedVertices)
        {
            var vertex = scene.Vertices[vertexIndex];
            buildingCenter += new Vector3(-vertex.X, vertex.Y, -vertex.Z);
        }
        buildingCenter /= sortedVertices.Count;
        
        // Export to OBJ file
        var objPath = Path.Combine(objDir, $"Building_{buildingIndex:D2}_Group_{groupKey:X4}.obj");
        
        using var writer = new StreamWriter(objPath);
        
        // Write header
        writer.WriteLine($"# PM4 Building Object Export");
        writer.WriteLine($"# Building Index: {buildingIndex}");
        writer.WriteLine($"# Surface Group: 0x{groupKey:X4}");
        writer.WriteLine($"# Surfaces: {surfaces.Count}");
        writer.WriteLine($"# Triangles: {triangles.Count:N0}");
        writer.WriteLine($"# Vertices: {sortedVertices.Count:N0}");
        writer.WriteLine($"# Center: ({buildingCenter.X:F1}, {buildingCenter.Y:F1}, {buildingCenter.Z:F1})");
        writer.WriteLine();
        
        // Write vertices (with coordinate system fix)
        foreach (var originalIndex in sortedVertices)
        {
            var vertex = scene.Vertices[originalIndex];
            writer.WriteLine($"v {-vertex.X:F6} {vertex.Y:F6} {-vertex.Z:F6}"); // Fix X and Z axes
        }
        
        writer.WriteLine();
        writer.WriteLine($"g Building_{buildingIndex:D2}_Group_{groupKey:X4}");
        
        // Write faces with remapped indices
        foreach (var (a, b, c) in triangles)
        {
            writer.WriteLine($"f {vertexMapping[a]} {vertexMapping[b]} {vertexMapping[c]}");
        }
        
        ConsoleLogger.WriteLine($"  Exported Building_{buildingIndex:D2}_Group_{groupKey:X4}.obj ({triangles.Count:N0} triangles)");
    }
    
    /// <summary>
    /// Write building export summary.
    /// </summary>
    private static void WriteBuildingSummary(
        List<IGrouping<byte, MsurChunk.Entry>> surfaceGroups, 
        string objDir, 
        TimeSpan elapsed, 
        string fileName)
    {
        var summaryPath = Path.Combine(objDir, "building_export_summary.txt");
        
        using var writer = new StreamWriter(summaryPath);
        
        writer.WriteLine("PM4 Building Export Summary");
        writer.WriteLine("==========================");
        writer.WriteLine($"Source File: {fileName}");
        writer.WriteLine($"Export Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        writer.WriteLine($"Export Duration: {elapsed.TotalSeconds:F1} seconds");
        writer.WriteLine();
        
        writer.WriteLine($"Buildings Exported: {surfaceGroups.Count}");
        writer.WriteLine();
        
        writer.WriteLine("Building Details:");
        writer.WriteLine("Index | Group ID | Triangles | Surfaces | File Name");
        writer.WriteLine("------|----------|-----------|----------|----------");
        
        int buildingIndex = 0;
        foreach (var group in surfaceGroups)
        {
            var triangleCount = group.Sum(s => s.IndexCount) / 3;
            var fileName2 = $"Building_{buildingIndex:D2}_Group_{group.Key:X4}.obj";
            writer.WriteLine($"{buildingIndex,5} | 0x{group.Key:X6} | {triangleCount,9:N0} | {group.Count(),8} | {fileName2}");
            buildingIndex++;
        }
        
        writer.WriteLine();
        writer.WriteLine("CONCLUSION:");
        writer.WriteLine("Surface groups (MSUR.SurfaceGroupKey) represent complete building objects.");
        writer.WriteLine("Each surface group should be exported as a separate building OBJ file.");
        
        ConsoleLogger.WriteLine($"Building export summary written to: {summaryPath}");
    }
}
