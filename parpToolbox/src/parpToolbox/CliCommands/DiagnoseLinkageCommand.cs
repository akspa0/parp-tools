using System.CommandLine;
using System.CommandLine.Invocation;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.CliCommands;

public class DiagnoseLinkageCommand
{
    private static readonly Argument<string> InputPathArgument = new("input-path", "Path to the input PM4 file.");

    public static Command CreateCommand()
    {
        var command = new Command("diagnose-linkage", "Diagnose MSLK → MSUR linkage issues in PM4 files.");
        command.AddArgument(InputPathArgument);
        command.SetHandler(Run);
        return command;
    }

    public static async Task<int> Run(InvocationContext invocationContext)
    {
        var inputPath = invocationContext.ParseResult.GetValueForArgument(InputPathArgument);

        Console.WriteLine("=== PM4 Linkage Diagnosis ===");
        Console.WriteLine($"Input: {inputPath}");
        Console.WriteLine();

        try
        {
            var adapter = new Pm4Adapter();
            var scene = adapter.Load(inputPath);
            
            Console.WriteLine($"Data: {scene.Vertices.Count} vertices, {scene.Triangles.Count} triangles");
            Console.WriteLine($"Chunks: MSLK={scene.Links.Count}, MSUR={scene.Surfaces.Count}");
            Console.WriteLine();

            // Analyze SurfaceRefIndex → SurfaceKey linkage
            Console.WriteLine("=== MSLK → MSUR Linkage Analysis ===");
            
            var surfaceKeySet = scene.Surfaces.Select(s => (uint)s.SurfaceKey).ToHashSet();
            var surfaceRefIndexSet = scene.Links.Select(l => (uint)l.SurfaceRefIndex).ToHashSet();
            
            Console.WriteLine($"MSUR SurfaceKeys: {surfaceKeySet.Count} unique values (range {surfaceKeySet.Min()}-{surfaceKeySet.Max()})");
            Console.WriteLine($"MSLK SurfaceRefIndex: {surfaceRefIndexSet.Count} unique values (range {surfaceRefIndexSet.Min()}-{surfaceRefIndexSet.Max()})");
            
            var matchingKeys = surfaceKeySet.Intersect(surfaceRefIndexSet).Count();
            var orphanedMslk = surfaceRefIndexSet.Except(surfaceKeySet).Count();
            var orphanedMsur = surfaceKeySet.Except(surfaceRefIndexSet).Count();
            
            Console.WriteLine($"Matching keys: {matchingKeys}");
            Console.WriteLine($"Orphaned MSLK (no matching MSUR): {orphanedMslk}");
            Console.WriteLine($"Orphaned MSUR (no matching MSLK): {orphanedMsur}");
            Console.WriteLine();

            // Container-by-container analysis
            Console.WriteLine("=== Container Linkage Breakdown ===");
            var containerGroups = scene.Links
                .GroupBy(link => GetContainerID(link.ParentIndex))
                .OrderBy(g => g.Key)
                .ToList();

            foreach (var containerGroup in containerGroups)
            {
                var containerID = containerGroup.Key;
                var links = containerGroup.ToList();
                
                var linkedSurfaces = links
                    .Select(l => (uint)l.SurfaceRefIndex)
                    .Where(surfaceKeySet.Contains)
                    .Count();
                
                var validGeometry = links
                    .Where(l => surfaceKeySet.Contains((uint)l.SurfaceRefIndex))
                    .SelectMany(l => {
                        var surface = scene.Surfaces.FirstOrDefault(s => s.SurfaceKey == l.SurfaceRefIndex);
                        return surface != null ? new[] { surface } : new ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry[0];
                    })
                    .Where(s => s.IndexCount > 0)
                    .Count();

                Console.WriteLine($"Container {containerID}: {links.Count} MSLK → {linkedSurfaces} linked MSUR → {validGeometry} with geometry");
            }
            
            await Task.CompletedTask;
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static byte GetContainerID(uint parentIndex)
    {
        return (byte)((parentIndex >> 8) & 0xFF);
    }
}
