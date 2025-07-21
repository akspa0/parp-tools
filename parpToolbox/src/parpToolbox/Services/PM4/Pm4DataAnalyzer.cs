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
/// Analyzes PM4 data structures to understand object relationships and grouping patterns.
/// </summary>
internal static class Pm4DataAnalyzer
{
    /// <summary>
    /// Analyzes PM4 scene data to understand chunk relationships and potential object grouping strategies.
    /// </summary>
    public static void AnalyzeDataStructure(Pm4Scene scene, string outputPath)
    {
        ConsoleLogger.WriteLine("=== PM4 Data Structure Analysis ===");
        ConsoleLogger.WriteLine($"Vertices: {scene.Vertices.Count:N0}");
        ConsoleLogger.WriteLine($"Indices: {scene.Indices.Count:N0}");
        ConsoleLogger.WriteLine($"Triangles: {scene.Triangles.Count:N0}");
        ConsoleLogger.WriteLine($"Surfaces: {scene.Surfaces.Count:N0}");
        ConsoleLogger.WriteLine($"Links: {scene.Links.Count:N0}");
        ConsoleLogger.WriteLine($"Placements: {scene.Placements.Count:N0}");
        
        using var writer = new StreamWriter(Path.Combine(outputPath, "pm4_data_analysis.csv"));
        
        // Write CSV header
        writer.WriteLine("Analysis,Category,Key,Value,Description");
        
        // Analyze MPRL placements
        AnalyzeMprlPlacements(scene, writer);
        
        // Analyze MSLK links
        AnalyzeMslkLinks(scene, writer);
        
        // Analyze MSUR surfaces
        AnalyzeMsurSurfaces(scene, writer);
        
        // Analyze MPRR properties
        AnalyzeMprrProperties(scene, writer);
        
        // Analyze relationships between chunks
        AnalyzeChunkRelationships(scene, writer);
        
        // Analyze geometry distribution
        AnalyzeGeometryDistribution(scene, writer);
        
        ConsoleLogger.WriteLine($"Data analysis written to: {Path.Combine(outputPath, "pm4_data_analysis.csv")}");
    }
    
    private static void AnalyzeMprlPlacements(Pm4Scene scene, StreamWriter writer)
    {
        ConsoleLogger.WriteLine("\n--- MPRL Placement Analysis ---");
        
        var unknown4Values = scene.Placements.Select(p => p.Unknown4).ToList();
        var unique4Values = unknown4Values.Distinct().ToList();
        
        writer.WriteLine($"MPRL,Count,Total,{scene.Placements.Count},Total MPRL placements");
        writer.WriteLine($"MPRL,Unique,Unknown4,{unique4Values.Count},Unique Unknown4 values");
        writer.WriteLine($"MPRL,Range,Unknown4_Min,{unknown4Values.Min()},Minimum Unknown4 value");
        writer.WriteLine($"MPRL,Range,Unknown4_Max,{unknown4Values.Max()},Maximum Unknown4 value");
        
        // Position analysis
        var positions = scene.Placements.Select(p => p.Position).ToList();
        var minPos = new Vector3(positions.Min(p => p.X), positions.Min(p => p.Y), positions.Min(p => p.Z));
        var maxPos = new Vector3(positions.Max(p => p.X), positions.Max(p => p.Y), positions.Max(p => p.Z));
        
        writer.WriteLine($"MPRL,Bounds,Position_Min,\"{minPos.X:F2},{minPos.Y:F2},{minPos.Z:F2}\",Minimum position bounds");
        writer.WriteLine($"MPRL,Bounds,Position_Max,\"{maxPos.X:F2},{maxPos.Y:F2},{maxPos.Z:F2}\",Maximum position bounds");
        
        ConsoleLogger.WriteLine($"  MPRL Placements: {scene.Placements.Count}");
        ConsoleLogger.WriteLine($"  Unique Unknown4 values: {unique4Values.Count}");
        ConsoleLogger.WriteLine($"  Unknown4 range: {unknown4Values.Min()} - {unknown4Values.Max()}");
    }
    
    private static void AnalyzeMslkLinks(Pm4Scene scene, StreamWriter writer)
    {
        ConsoleLogger.WriteLine("\n--- MSLK Link Analysis ---");
        
        var parentIndices = scene.Links.Select(l => l.ParentIndex).ToList();
        var uniqueParents = parentIndices.Distinct().ToList();
        var validGeometry = scene.Links.Where(l => l.MspiFirstIndex >= 0 && l.MspiIndexCount > 0).ToList();
        
        writer.WriteLine($"MSLK,Count,Total,{scene.Links.Count},Total MSLK links");
        writer.WriteLine($"MSLK,Count,ValidGeometry,{validGeometry.Count},Links with valid geometry indices");
        writer.WriteLine($"MSLK,Unique,ParentIndex,{uniqueParents.Count},Unique ParentIndex values");
        writer.WriteLine($"MSLK,Range,ParentIndex_Min,{parentIndices.Min()},Minimum ParentIndex");
        writer.WriteLine($"MSLK,Range,ParentIndex_Max,{parentIndices.Max()},Maximum ParentIndex");
        
        // Analyze index count distribution
        var indexCounts = validGeometry.Select(l => (int)l.MspiIndexCount).ToList();
        if (indexCounts.Count > 0)
        {
            writer.WriteLine($"MSLK,Stats,IndexCount_Min,{indexCounts.Min()},Minimum index count");
            writer.WriteLine($"MSLK,Stats,IndexCount_Max,{indexCounts.Max()},Maximum index count");
            writer.WriteLine($"MSLK,Stats,IndexCount_Avg,{indexCounts.Average():F1},Average index count");
        }
        
        ConsoleLogger.WriteLine($"  MSLK Links: {scene.Links.Count}");
        ConsoleLogger.WriteLine($"  Valid geometry links: {validGeometry.Count}");
        ConsoleLogger.WriteLine($"  Unique ParentIndex values: {uniqueParents.Count}");
    }
    
    private static void AnalyzeMsurSurfaces(Pm4Scene scene, StreamWriter writer)
    {
        ConsoleLogger.WriteLine("\n--- MSUR Surface Analysis ---");
        
        var surfaceKeys = scene.Surfaces.Select(s => s.SurfaceKey).ToList();
        var uniqueSurfaceKeys = surfaceKeys.Distinct().ToList();
        var indexCounts = scene.Surfaces.Select(s => (int)s.IndexCount).ToList();
        
        writer.WriteLine($"MSUR,Count,Total,{scene.Surfaces.Count},Total MSUR surfaces");
        writer.WriteLine($"MSUR,Unique,SurfaceKey,{uniqueSurfaceKeys.Count},Unique SurfaceKey values");
        writer.WriteLine($"MSUR,Range,SurfaceKey_Min,{surfaceKeys.Min()},Minimum SurfaceKey");
        writer.WriteLine($"MSUR,Range,SurfaceKey_Max,{surfaceKeys.Max()},Maximum SurfaceKey");
        
        if (indexCounts.Count > 0)
        {
            writer.WriteLine($"MSUR,Stats,IndexCount_Min,{indexCounts.Min()},Minimum index count");
            writer.WriteLine($"MSUR,Stats,IndexCount_Max,{indexCounts.Max()},Maximum index count");
            writer.WriteLine($"MSUR,Stats,IndexCount_Avg,{indexCounts.Average():F1},Average index count");
            writer.WriteLine($"MSUR,Stats,IndexCount_Total,{indexCounts.Sum()},Total indices in surfaces");
        }
        
        ConsoleLogger.WriteLine($"  MSUR Surfaces: {scene.Surfaces.Count}");
        ConsoleLogger.WriteLine($"  Unique SurfaceKey values: {uniqueSurfaceKeys.Count}");
    }
    
    private static void AnalyzeMprrProperties(Pm4Scene scene, StreamWriter writer)
    {
        ConsoleLogger.WriteLine("\n--- MPRR Property Analysis ---");
        
        if (scene.Properties.Count == 0)
        {
            ConsoleLogger.WriteLine("  No MPRR properties found");
            writer.WriteLine($"MPRR,Count,Total,0,No MPRR properties found");
            return;
        }
        
        var value1List = scene.Properties.Select(p => p.Value1).ToList();
        var value2List = scene.Properties.Select(p => p.Value2).ToList();
        var uniqueValue1 = value1List.Distinct().ToList();
        var uniqueValue2 = value2List.Distinct().ToList();
        
        writer.WriteLine($"MPRR,Count,Total,{scene.Properties.Count},Total MPRR properties");
        writer.WriteLine($"MPRR,Unique,Value1,{uniqueValue1.Count},Unique Value1 values");
        writer.WriteLine($"MPRR,Unique,Value2,{uniqueValue2.Count},Unique Value2 values");
        writer.WriteLine($"MPRR,Range,Value1_Min,{value1List.Min()},Minimum Value1");
        writer.WriteLine($"MPRR,Range,Value1_Max,{value1List.Max()},Maximum Value1");
        writer.WriteLine($"MPRR,Range,Value2_Min,{value2List.Min()},Minimum Value2");
        writer.WriteLine($"MPRR,Range,Value2_Max,{value2List.Max()},Maximum Value2");
        
        // Analyze sentinel values (65535 = 0xFFFF) that mark object boundaries
        var sentinelCount = scene.Properties.Count(p => p.Value1 == 65535 || p.Value2 == 65535);
        var value1Sentinels = scene.Properties.Count(p => p.Value1 == 65535);
        var value2Sentinels = scene.Properties.Count(p => p.Value2 == 65535);
        
        writer.WriteLine($"MPRR,Sentinels,Total,{sentinelCount},Properties with sentinel values (65535)");
        writer.WriteLine($"MPRR,Sentinels,Value1,{value1Sentinels},Value1 sentinel count");
        writer.WriteLine($"MPRR,Sentinels,Value2,{value2Sentinels},Value2 sentinel count");
        
        // Count object groups separated by sentinels
        int objectGroups = 0;
        bool inGroup = false;
        foreach (var prop in scene.Properties)
        {
            if (prop.Value1 == 65535)
            {
                if (inGroup) objectGroups++;
                inGroup = false;
            }
            else
            {
                inGroup = true;
            }
        }
        if (inGroup) objectGroups++; // Final group
        
        writer.WriteLine($"MPRR,ObjectGroups,Count,{objectGroups},Object groups separated by Value1 sentinels");
        
        ConsoleLogger.WriteLine($"  MPRR Properties: {scene.Properties.Count}");
        ConsoleLogger.WriteLine($"  Unique Value1/Value2: {uniqueValue1.Count}/{uniqueValue2.Count}");
        ConsoleLogger.WriteLine($"  Sentinel markers (65535): {sentinelCount} total, {value1Sentinels} in Value1, {value2Sentinels} in Value2");
        ConsoleLogger.WriteLine($"  Object groups (Value1 sentinel-separated): {objectGroups}");
    }
    
    private static void AnalyzeChunkRelationships(Pm4Scene scene, StreamWriter writer)
    {
        ConsoleLogger.WriteLine("\n--- Chunk Relationship Analysis ---");
        
        // MPRL.Unknown4 vs MSLK.ParentIndex relationship
        var mprlIds = scene.Placements.Select(p => (uint)p.Unknown4).ToHashSet();
        var mslkParents = scene.Links.Select(l => l.ParentIndex).ToHashSet();
        var intersection = mprlIds.Intersect(mslkParents).ToList();
        
        writer.WriteLine($"Relationships,MPRL_MSLK,MPRL_Unknown4_Count,{mprlIds.Count},Unique MPRL Unknown4 values");
        writer.WriteLine($"Relationships,MPRL_MSLK,MSLK_ParentIndex_Count,{mslkParents.Count},Unique MSLK ParentIndex values");
        writer.WriteLine($"Relationships,MPRL_MSLK,Intersection_Count,{intersection.Count},Values present in both");
        writer.WriteLine($"Relationships,MPRL_MSLK,Intersection_Percent,{(intersection.Count * 100.0 / Math.Max(mprlIds.Count, 1)):F1},Percentage overlap");
        
        ConsoleLogger.WriteLine($"  MPRL.Unknown4 ↔ MSLK.ParentIndex overlap: {intersection.Count}/{mprlIds.Count} ({intersection.Count * 100.0 / Math.Max(mprlIds.Count, 1):F1}%)");
        
        // Analyze geometry coverage by different grouping strategies
        var totalIndices = scene.Indices.Count;
        
        // By MSUR surfaces
        var msurIndices = scene.Surfaces.Sum(s => (int)s.IndexCount);
        writer.WriteLine($"Coverage,MSUR,Total_Indices,{msurIndices},Indices covered by MSUR surfaces");
        writer.WriteLine($"Coverage,MSUR,Coverage_Percent,{(msurIndices * 100.0 / Math.Max(totalIndices, 1)):F1},Percentage of total indices");
        
        // By MSLK links with valid geometry
        var mslkIndices = scene.Links.Where(l => l.MspiFirstIndex >= 0 && l.MspiIndexCount > 0).Sum(l => (int)l.MspiIndexCount);
        writer.WriteLine($"Coverage,MSLK,Total_Indices,{mslkIndices},Indices covered by MSLK links");
        writer.WriteLine($"Coverage,MSLK,Coverage_Percent,{(mslkIndices * 100.0 / Math.Max(totalIndices, 1)):F1},Percentage of total indices");
        
        ConsoleLogger.WriteLine($"  Index coverage - MSUR: {msurIndices:N0}/{totalIndices:N0} ({msurIndices * 100.0 / Math.Max(totalIndices, 1):F1}%)");
        ConsoleLogger.WriteLine($"  Index coverage - MSLK: {mslkIndices:N0}/{totalIndices:N0} ({mslkIndices * 100.0 / Math.Max(totalIndices, 1):F1}%)");
    }
    
    private static void AnalyzeGeometryDistribution(Pm4Scene scene, StreamWriter writer)
    {
        ConsoleLogger.WriteLine("\n--- Geometry Distribution Analysis ---");
        
        // Analyze vertex usage patterns
        var usedVertices = new HashSet<int>();
        for (int i = 0; i < scene.Indices.Count; i++)
        {
            if (scene.Indices[i] < scene.Vertices.Count)
                usedVertices.Add(scene.Indices[i]);
        }
        
        writer.WriteLine($"Geometry,Vertices,Total,{scene.Vertices.Count},Total vertices");
        writer.WriteLine($"Geometry,Vertices,Used,{usedVertices.Count},Vertices referenced by indices");
        writer.WriteLine($"Geometry,Vertices,Usage_Percent,{(usedVertices.Count * 100.0 / Math.Max(scene.Vertices.Count, 1)):F1},Percentage of vertices used");
        
        // Analyze triangle size distribution
        var triangleSizes = new List<double>();
        for (int i = 0; i < scene.Indices.Count; i += 3)
        {
            if (i + 2 < scene.Indices.Count)
            {
                var a = scene.Indices[i];
                var b = scene.Indices[i + 1];
                var c = scene.Indices[i + 2];
                
                if (a < scene.Vertices.Count && b < scene.Vertices.Count && c < scene.Vertices.Count)
                {
                    var va = scene.Vertices[a];
                    var vb = scene.Vertices[b];
                    var vc = scene.Vertices[c];
                    
                    var area = Vector3.Cross(vb - va, vc - va).Length() * 0.5f;
                    triangleSizes.Add(area);
                }
            }
        }
        
        if (triangleSizes.Count > 0)
        {
            writer.WriteLine($"Geometry,Triangles,Count,{triangleSizes.Count},Total triangles analyzed");
            writer.WriteLine($"Geometry,Triangles,Area_Min,{triangleSizes.Min():F6},Minimum triangle area");
            writer.WriteLine($"Geometry,Triangles,Area_Max,{triangleSizes.Max():F6},Maximum triangle area");
            writer.WriteLine($"Geometry,Triangles,Area_Avg,{triangleSizes.Average():F6},Average triangle area");
        }
        
        ConsoleLogger.WriteLine($"  Vertex usage: {usedVertices.Count:N0}/{scene.Vertices.Count:N0} ({usedVertices.Count * 100.0 / Math.Max(scene.Vertices.Count, 1):F1}%)");
        ConsoleLogger.WriteLine($"  Triangle analysis: {triangleSizes.Count:N0} triangles");
    }
}
