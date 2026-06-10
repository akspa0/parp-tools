using System;
using System.Collections.Generic;
using System.IO;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder.Pipeline
{
    /// <summary>
    /// Step A – Audits PM4 chunks and writes summary information to the log file.
    /// Very lightweight: counts each extra chunk type and records total vertices/triangles.
    /// </summary>
    internal static class ChunkAuditor
    {
        public static void Audit(Pm4Scene scene, string outputDir)
        {
            Console.WriteLine("[CHUNK AUDIT] Starting chunk audit …");

            // Count extra chunk types
            var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            foreach (var chunk in scene.ExtraChunks)
            {
                var name = chunk.GetType().Name;
                counts.TryGetValue(name, out int c);
                counts[name] = c + 1;
            }

            foreach (var kvp in counts)
            {
                Console.WriteLine($"{kvp.Key}: {kvp.Value}");
            }

            Console.WriteLine($"Vertices: {scene.Vertices.Count}, Triangles: {scene.Triangles.Count}");
            Console.WriteLine("[CHUNK AUDIT] Completed.");
        }
    }
}
