using System.CommandLine;
using System.CommandLine.Invocation;
using System.Numerics;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.CliCommands;

public class TestSurfaceRefBuildingGroupingCommand
{
    private static readonly Argument<string> InputPathArgument = new("input-path", "Path to the input PM4 file.");
    private static readonly Argument<string> OutputPathArgument = new("output-path", "Output directory path.");

    public static Command CreateCommand()
    {
        var command = new Command("test-surfaceref-building-grouping", "Test building-scale grouping using SurfaceRefIndex upper 8-bit as discovered in MSLK analysis.");
        command.AddArgument(InputPathArgument);
        command.AddArgument(OutputPathArgument);
        command.SetHandler(Run);
        return command;
    }

    public static async Task<int> Run(InvocationContext invocationContext)
    {
        var inputPath = invocationContext.ParseResult.GetValueForArgument(InputPathArgument);
        var outputPath = invocationContext.ParseResult.GetValueForArgument(OutputPathArgument);

        Console.WriteLine("=== SurfaceRefIndex Building-Scale Grouping Test ===");
        Console.WriteLine($"Input: {inputPath}");
        Console.WriteLine($"Output: {outputPath}");
        Console.WriteLine();

        try
        {
            var adapter = new Pm4Adapter();
            var scene = adapter.Load(inputPath);
            
            Console.WriteLine($"Loaded PM4 file: {Path.GetFileName(inputPath)}");
            Console.WriteLine($"Data: {scene.Vertices.Count} vertices, {scene.Triangles.Count} triangles");
            Console.WriteLine($"Chunks: MPRL={scene.Placements.Count}, MSLK={scene.Links.Count}, MSUR={scene.Surfaces.Count}");
            Console.WriteLine();

            // Create output directory
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"{outputPath}_{timestamp}");
            Directory.CreateDirectory(outputDir);
            
            Console.WriteLine($"Output directory: {outputDir}");
            Console.WriteLine("=== Testing SurfaceRefIndex Upper 8-Bit Building Grouping ===");

            // Group MSLK entries by SurfaceRefIndex upper 8-bit
            var buildingGroups = scene.Links.GroupBy(l => (byte)((l.SurfaceRefIndex >> 8) & 0xFF)).ToList();
            Console.WriteLine($"Found {buildingGroups.Count} building groups.");

            foreach (var buildingGroup in buildingGroups.OrderBy(g => g.Key))
            {
                var buildingID = buildingGroup.Key;
                var links = buildingGroup.ToList();
                
                Console.WriteLine($"Building {buildingID}: {links.Count} MSLK entries");
                
                // Collect geometry for this building using complete Data Web linkage
                var buildingVertices = new List<Vector3>();
                var buildingTriangles = new List<(int A, int B, int C)>();

                // PHASE 1: Add geometry from MSLK → MSUR linkage
                foreach (var link in links)
                {
                    var surface = scene.Surfaces.FirstOrDefault(s => (s.SurfaceKey & 0xFF) == (link.SurfaceRefIndex & 0xFF));
                    if (surface != null)
                    {
                        AddSurfaceGeometry(scene, surface, buildingVertices, buildingTriangles);
                    }
                }

                // PHASE 2: Add geometry from MPRL → MSUR linkage (if any MPRL entries correlate)
                var correlatedPlacements = scene.Placements.Where(p => 
                    links.Any(l => (l.ParentIndex & 0xFF) == (p.Unknown4 & 0xFF))).ToList();
                
                foreach (var placement in correlatedPlacements)
                {
                    var mprlSurfaces = scene.Surfaces.Where(s => (s.SurfaceKey & 0xFF) == (placement.Unknown16 & 0xFF)).ToList();
                    foreach (var surface in mprlSurfaces)
                    {
                        if (!IsGeometryAlreadyAdded(surface, buildingVertices))
                        {
                            AddSurfaceGeometry(scene, surface, buildingVertices, buildingTriangles);
                        }
                    }
                }

                // Write OBJ file for this building
                if (buildingTriangles.Any())
                {
                    var filename = Path.Combine(outputDir, $"building_{buildingID:D2}.obj");
                    WriteObjFile(filename, buildingVertices, buildingTriangles);
                    Console.WriteLine($"  → building_{buildingID:D2}.obj ({buildingVertices.Count} vertices, {buildingTriangles.Count} triangles)");
                }
                else
                {
                    Console.WriteLine($"  → No geometry found for building {buildingID}");
                }
            }

            Console.WriteLine("Export complete!");
            await Task.CompletedTask;
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static void AddSurfaceGeometry(Pm4Scene scene, ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface, 
        List<Vector3> vertices, List<(int A, int B, int C)> triangles)
    {
        // Add vertices and triangles from this surface
        var surfaceVertices = new List<Vector3>();
        var surfaceTriangles = new List<(int A, int B, int C)>();

        // Extract geometry from surface using MSVI indices
        for (int i = 0; i < surface.IndexCount && (surface.MsviFirstIndex + i) < scene.Indices.Count; i += 3)
        {
            if (i + 2 >= surface.IndexCount) break; // Ensure we have complete triangles
            
            var idx1 = scene.Indices[(int)surface.MsviFirstIndex + i];
            var idx2 = scene.Indices[(int)surface.MsviFirstIndex + i + 1];
            var idx3 = scene.Indices[(int)surface.MsviFirstIndex + i + 2];
            
            if (idx1 >= scene.Vertices.Count || idx2 >= scene.Vertices.Count || idx3 >= scene.Vertices.Count) continue;
            
            var v1 = scene.Vertices[idx1];
            var v2 = scene.Vertices[idx2];
            var v3 = scene.Vertices[idx3];

            var vertexOffset = vertices.Count;
            vertices.Add(new Vector3(-v1.X, v1.Z, v1.Y)); // Apply coordinate flip
            vertices.Add(new Vector3(-v2.X, v2.Z, v2.Y));
            vertices.Add(new Vector3(-v3.X, v3.Z, v3.Y));

            triangles.Add((vertexOffset, vertexOffset + 1, vertexOffset + 2));
        }
    }

    private static bool IsGeometryAlreadyAdded(ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface, List<Vector3> existingVertices)
    {
        // Simple deduplication to avoid duplicate geometry
        return false; // Allow all for now to maximize geometry collection
    }

    private static void WriteObjFile(string filename, List<Vector3> vertices, List<(int A, int B, int C)> triangles)
    {
        using var writer = new StreamWriter(filename);
        
        writer.WriteLine("# PM4 Building-Scale Object Export");
        writer.WriteLine($"# Generated: {DateTime.Now}");
        writer.WriteLine($"# Vertices: {vertices.Count}, Triangles: {triangles.Count}");
        writer.WriteLine();

        // Write vertices
        foreach (var vertex in vertices)
        {
            writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
        }

        writer.WriteLine();

        // Write faces
        foreach (var triangle in triangles)
        {
            writer.WriteLine($"f {triangle.A + 1} {triangle.B + 1} {triangle.C + 1}");
        }
    }
}
