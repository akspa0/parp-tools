using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Enhanced diagnostic spatial clustering with N-GON support and improved object detection
    /// </summary>
    public class EnhancedDiagnosticSpatialCommand
    {
        public void Execute(string inputPath, string outputPath)
        {
            ConsoleLogger.WriteLine("=== ENHANCED Diagnostic Spatial Clustering Export (N-GON Support) ===");
            ConsoleLogger.WriteLine($"Input: {inputPath}");
            ConsoleLogger.WriteLine($"Output: {outputPath}");

            try
            {
                // Validate input file
                if (!File.Exists(inputPath))
                {
                    ConsoleLogger.WriteLine($"ERROR: Input file not found: {inputPath}");
                    return;
                }

                // Create output directory
                Directory.CreateDirectory(outputPath);

                // Load PM4 scene
                ConsoleLogger.WriteLine("Loading PM4 scene...");
                var adapter = new Pm4Adapter();
                var scene = adapter.LoadRegion(inputPath);

                if (scene == null)
                {
                    ConsoleLogger.WriteLine("ERROR: Failed to load PM4 scene");
                    return;
                }

                ConsoleLogger.WriteLine($"Scene loaded: {scene.Surfaces?.Count ?? 0} surfaces, {scene.Vertices?.Count ?? 0} vertices");

                // Group surfaces by MULTIPLE criteria to find more objects
                AnalyzeGroupingOptions(scene);

                // Group surfaces by SurfaceKey
                var surfaceKeyToIndices = new Dictionary<uint, List<int>>();
                if (scene.Surfaces != null)
                {
                    for (int i = 0; i < scene.Surfaces.Count; i++)
                    {
                        var surface = scene.Surfaces[i];
                        if (!surfaceKeyToIndices.ContainsKey(surface.SurfaceKey))
                        {
                            surfaceKeyToIndices[surface.SurfaceKey] = new List<int>();
                        }
                        surfaceKeyToIndices[surface.SurfaceKey].Add(i);
                    }
                }

                ConsoleLogger.WriteLine($"Found {surfaceKeyToIndices.Count} unique SurfaceKeys");

                // Export with RELAXED filtering to find more objects
                int exportCount = 0;
                int maxExports = 15; // Increased to find 12-18 expected objects

                foreach (var kvp in surfaceKeyToIndices.OrderByDescending(x => x.Value.Count).Take(maxExports))
                {
                    var surfaceKey = kvp.Key;
                    var surfaceIndices = kvp.Value;

                    // Calculate triangle count for filtering
                    int polygonCount = CalculatePolygonCount(scene, surfaceIndices);

                    // RELAXED size filtering to catch more objects
                    const int MIN_POLYGONS = 5;   // Reduced from 10
                    const int MAX_POLYGONS = 25000; // Reduced from 50000

                    if (polygonCount < MIN_POLYGONS || polygonCount > MAX_POLYGONS)
                    {
                        ConsoleLogger.WriteLine($"Skipping SurfaceKey 0x{surfaceKey:X8}: {polygonCount} polygons (out of range {MIN_POLYGONS}-{MAX_POLYGONS})");
                        continue;
                    }

                    ConsoleLogger.WriteLine($"Exporting SurfaceKey 0x{surfaceKey:X8}: {surfaceIndices.Count} surfaces, ~{polygonCount} polygons");

                    // Export WITH N-GON support
                    ExportSurfaceGroupWithNGons(scene, surfaceIndices, surfaceKey, outputPath, exportCount);
                    exportCount++;
                }

                ConsoleLogger.WriteLine($"Exported {exportCount} enhanced diagnostic objects to: {outputPath}");
                ConsoleLogger.WriteLine("=== Enhanced Diagnostic Export Complete ===");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        private void AnalyzeGroupingOptions(ParpToolbox.Formats.PM4.Pm4Scene scene)
        {
            ConsoleLogger.WriteLine("\n=== Grouping Analysis ===");

            if (scene.Surfaces == null) return;

            // Analyze GroupKey distribution
            var groupKeyDistribution = new Dictionary<byte, int>();
            var surfaceKeyDistribution = new Dictionary<uint, int>();
            var compositeKeyDistribution = new Dictionary<uint, int>();

            foreach (var surface in scene.Surfaces)
            {
                groupKeyDistribution[surface.GroupKey] = groupKeyDistribution.GetValueOrDefault(surface.GroupKey, 0) + 1;
                surfaceKeyDistribution[surface.SurfaceKey] = surfaceKeyDistribution.GetValueOrDefault(surface.SurfaceKey, 0) + 1;
                compositeKeyDistribution[surface.CompositeKey] = compositeKeyDistribution.GetValueOrDefault(surface.CompositeKey, 0) + 1;
            }

            ConsoleLogger.WriteLine($"GroupKey unique values: {groupKeyDistribution.Count}");
            ConsoleLogger.WriteLine($"SurfaceKey unique values: {surfaceKeyDistribution.Count}");
            ConsoleLogger.WriteLine($"CompositeKey unique values: {compositeKeyDistribution.Count}");

            // Show top GroupKeys
            ConsoleLogger.WriteLine("Top GroupKeys:");
            foreach (var kvp in groupKeyDistribution.OrderByDescending(x => x.Value).Take(10))
            {
                ConsoleLogger.WriteLine($"  GroupKey {kvp.Key}: {kvp.Value} surfaces");
            }

            // Show top SurfaceKeys with reasonable object sizes
            ConsoleLogger.WriteLine("SurfaceKeys with reasonable object sizes (100-10000 surfaces):");
            int reasonableCount = 0;
            foreach (var kvp in surfaceKeyDistribution.Where(x => x.Value >= 100 && x.Value <= 10000).OrderByDescending(x => x.Value).Take(20))
            {
                ConsoleLogger.WriteLine($"  SurfaceKey 0x{kvp.Key:X8}: {kvp.Value} surfaces");
                reasonableCount++;
            }
            ConsoleLogger.WriteLine($"Found {reasonableCount} SurfaceKeys with reasonable object sizes");
            ConsoleLogger.WriteLine();
        }

        private void ExportSurfaceGroupWithNGons(ParpToolbox.Formats.PM4.Pm4Scene scene, List<int> surfaceIndices, uint surfaceKey, string outputPath, int exportIndex)
        {
            var vertices = new List<Vector3>();
            var faces = new List<List<int>>(); // Support for n-gons
            var vertexLookup = new Dictionary<Vector3, int>();

            ConsoleLogger.WriteLine($"  Processing {surfaceIndices.Count} surfaces with N-GON support...");

            int totalPolygons = 0;
            int triangles = 0, quads = 0, nGons = 0;

            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];

                    int firstIndex = (int)surface.MsviFirstIndex;
                    int indexCount = surface.IndexCount;

                    if (firstIndex >= 0 && indexCount >= 3 && firstIndex + indexCount <= scene.Indices.Count)
                    {
                        // READ THE ACTUAL POLYGON VERTEX COUNT!
                        var polygonVertices = new List<int>();

                        for (int i = 0; i < indexCount; i++)
                        {
                            int vertexIndex = scene.Indices[firstIndex + i];
                            if (vertexIndex >= 0 && vertexIndex < scene.Vertices.Count)
                            {
                                var vertex = scene.Vertices[vertexIndex];
                                int localVertexIndex = GetOrAddVertex(vertices, vertexLookup, vertex);
                                polygonVertices.Add(localVertexIndex);
                            }
                        }

                        if (polygonVertices.Count >= 3)
                        {
                            faces.Add(polygonVertices);
                            totalPolygons++;

                            // Count polygon types
                            if (polygonVertices.Count == 3) triangles++;
                            else if (polygonVertices.Count == 4) quads++;
                            else nGons++;
                        }
                    }
                }
            }

            // Export to OBJ with proper n-gon support
            string objPath = Path.Combine(outputPath, $"enhanced_surface_0x{surfaceKey:X8}_{exportIndex}.obj");
            using (var writer = new StreamWriter(objPath))
            {
                writer.WriteLine($"# Enhanced export - SurfaceKey 0x{surfaceKey:X8}");
                writer.WriteLine($"# Vertices: {vertices.Count}, Polygons: {totalPolygons}");
                writer.WriteLine($"# Triangles: {triangles}, Quads: {quads}, N-gons: {nGons}");
                writer.WriteLine($"# Plane projection: DISABLED, N-gons: ENABLED");
                writer.WriteLine();

                // Write vertices
                foreach (var vertex in vertices)
                {
                    writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }

                // Write faces (1-indexed, supports n-gons)
                foreach (var face in faces)
                {
                    writer.Write("f");
                    foreach (int vertexIndex in face)
                    {
                        writer.Write($" {vertexIndex + 1}");
                    }
                    writer.WriteLine();
                }
            }

            ConsoleLogger.WriteLine($"  Exported: {objPath}");
            ConsoleLogger.WriteLine($"    {vertices.Count} vertices, {totalPolygons} polygons (tri:{triangles}, quad:{quads}, n-gon:{nGons})");
        }

        private int GetOrAddVertex(List<Vector3> vertices, Dictionary<Vector3, int> lookup, Vector3 vertex)
        {
            if (lookup.TryGetValue(vertex, out int index))
            {
                return index;
            }

            index = vertices.Count;
            vertices.Add(vertex);
            lookup[vertex] = index;
            return index;
        }

        private int CalculatePolygonCount(ParpToolbox.Formats.PM4.Pm4Scene scene, List<int> surfaceIndices)
        {
            int totalPolygons = 0;
            foreach (int surfaceIndex in surfaceIndices)
            {
                if (surfaceIndex >= 0 && surfaceIndex < scene.Surfaces.Count)
                {
                    var surface = scene.Surfaces[surfaceIndex];
                    if (surface.IndexCount >= 3)
                    {
                        totalPolygons++; // Each surface = one polygon (regardless of triangle/quad)
                    }
                }
            }
            return totalPolygons;
        }
    }
}
