using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder
{
    /// <summary>
    /// Exports the entire PM4 tile as a single OBJ with all geometry properly linked.
    /// No clustering, no grouping - just raw geometry from MSVI indices and MSLK direct references.
    /// </summary>
    internal static class SingleTileExporter
    {
        public static void ExportSingleTile(Pm4Scene scene, string outputDir, string fileName = "pm4_tile_complete")
        {
            Directory.CreateDirectory(outputDir);
            string objPath = Path.Combine(outputDir, $"{fileName}.obj");
            
            Console.WriteLine($"[SINGLE TILE] Exporting complete PM4 tile geometry...");
            Console.WriteLine($"[SINGLE TILE] MSVT vertices: {scene.Vertices.Count}, MSCN vertices: {scene.MscnVertices.Count}");
            Console.WriteLine($"[SINGLE TILE] MSVI indices: {scene.Indices.Count}, MSLK links: {scene.Links.Count}");

            using var sw = new StreamWriter(objPath);
            
            // Header
            sw.WriteLine("# Complete PM4 Tile Export - All Geometry");
            sw.WriteLine($"# MSVT vertices: {scene.Vertices.Count}");
            sw.WriteLine($"# MSCN vertices: {scene.MscnVertices.Count}");
            sw.WriteLine($"# MSVI indices: {scene.Indices.Count}");
            sw.WriteLine($"# MSLK links: {scene.Links.Count}");
            sw.WriteLine();

            // Write ALL vertices (MSVT first, then MSCN)
            sw.WriteLine("# MSVT Vertices (0-based indices 0 to " + (scene.Vertices.Count - 1) + ")");
            foreach (var vertex in scene.Vertices)
            {
                sw.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
            
            sw.WriteLine($"# MSCN Vertices (0-based indices {scene.Vertices.Count} to {scene.Vertices.Count + scene.MscnVertices.Count - 1})");
            foreach (var vertex in scene.MscnVertices)
            {
                sw.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
            
            sw.WriteLine();

            int trianglesFromMsvi = 0;
            int trianglesFromMscn = 0;
            int totalTriangles = 0;

            // Export MSVI triangles (traditional triangle buffer)
            sw.WriteLine("# MSVI Triangle Data");
            sw.WriteLine("g msvi_geometry");
            
            for (int i = 0; i + 2 < scene.Indices.Count; i += 3)
            {
                int a = scene.Indices[i];
                int b = scene.Indices[i + 1];
                int c = scene.Indices[i + 2];
                
                // Validate indices are in range
                int totalVertexCount = scene.Vertices.Count + scene.MscnVertices.Count;
                if (a >= 0 && a < totalVertexCount && b >= 0 && b < totalVertexCount && c >= 0 && c < totalVertexCount)
                {
                    // Convert to 1-based OBJ indices
                    sw.WriteLine($"f {a + 1} {b + 1} {c + 1}");
                    trianglesFromMsvi++;
                    totalTriangles++;
                }
            }

            sw.WriteLine();

            // Export MSLK-referenced MSCN geometry (direct vertex references)
            sw.WriteLine("# MSLK-Referenced MSCN Geometry");
            sw.WriteLine("g mslk_mscn_geometry");
            
            int mscnStart = scene.Vertices.Count;
            int mscnEnd = scene.Vertices.Count + scene.MscnVertices.Count;
            
            foreach (var link in scene.Links.Where(l => l.MspiFirstIndex >= 0 && l.MspiIndexCount > 0))
            {
                int start = link.MspiFirstIndex;
                int count = link.MspiIndexCount;
                
                // Check if this references MSCN vertices directly
                if (start >= mscnStart && start < mscnEnd)
                {
                    // Generate triangles from sequential MSCN vertices
                    int availableVertices = Math.Min(count, mscnEnd - start);
                    
                    for (int i = 0; i + 2 < availableVertices; i += 3)
                    {
                        int a = start + i;
                        int b = start + i + 1;
                        int c = start + i + 2;
                        
                        if (a < mscnEnd && b < mscnEnd && c < mscnEnd)
                        {
                            // Convert to 1-based OBJ indices
                            sw.WriteLine($"f {a + 1} {b + 1} {c + 1}");
                            trianglesFromMscn++;
                            totalTriangles++;
                        }
                    }
                }
            }

            sw.WriteLine();
            sw.WriteLine($"# Export Summary:");
            sw.WriteLine($"# Total vertices exported: {scene.Vertices.Count + scene.MscnVertices.Count}");
            sw.WriteLine($"# MSVI triangles: {trianglesFromMsvi}");
            sw.WriteLine($"# MSCN triangles: {trianglesFromMscn}");
            sw.WriteLine($"# Total triangles: {totalTriangles}");

            Console.WriteLine($"[SINGLE TILE] Export complete:");
            Console.WriteLine($"[SINGLE TILE]   Vertices: {scene.Vertices.Count + scene.MscnVertices.Count}");
            Console.WriteLine($"[SINGLE TILE]   MSVI triangles: {trianglesFromMsvi}");
            Console.WriteLine($"[SINGLE TILE]   MSCN triangles: {trianglesFromMscn}");
            Console.WriteLine($"[SINGLE TILE]   Total triangles: {totalTriangles}");
            Console.WriteLine($"[SINGLE TILE]   File: {objPath}");
        }
    }
}
