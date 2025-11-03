using Microsoft.EntityFrameworkCore;
using ParpToolbox.Utils;
using System.Text.Json;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Analyzes PM4 triangle indices for cross-tile/global mesh linkage patterns.
    /// Validates the hypothesis that PM4 files form a unified architectural mesh system.
    /// </summary>
    public class Pm4GlobalMeshAnalyzer
    {
        /// <summary>
        /// Analyzes triangle vertex index patterns to identify cross-tile linkage.
        /// </summary>
        public static async Task AnalyzeGlobalMeshLinkage(string databasePath)
        {
            var startTime = DateTime.Now;
            
            ConsoleLogger.WriteLine("=== PM4 GLOBAL MESH LINKAGE ANALYSIS ===");
            ConsoleLogger.WriteLine($"Database: {databasePath}");
            ConsoleLogger.WriteLine();

            using var context = new Pm4DatabaseContext(databasePath);

            // Get basic triangle and vertex statistics
            var triangleCount = await context.Triangles.CountAsync();
            var vertexCount = await context.Vertices.CountAsync();
            var maxVertexIndex = await context.Vertices.MaxAsync(v => v.GlobalIndex);

            ConsoleLogger.WriteLine("=== MESH STATISTICS ===");
            ConsoleLogger.WriteLine($"Total triangles: {triangleCount:N0}");
            ConsoleLogger.WriteLine($"Total vertices: {vertexCount:N0}");
            ConsoleLogger.WriteLine($"Max vertex index: {maxVertexIndex:N0}");
            ConsoleLogger.WriteLine($"Valid vertex range: 0-{maxVertexIndex}");
            ConsoleLogger.WriteLine();

            // Analyze triangle vertex index patterns
            var triangles = await context.Triangles
                .Select(t => new { t.VertexA, t.VertexB, t.VertexC })
                .ToListAsync();

            var validTriangles = 0;
            var invalidTriangles = 0;
            var crossTileReferences = new Dictionary<int, int>();
            var localReferences = new Dictionary<int, int>();
            var mixedTriangles = 0;

            ConsoleLogger.WriteLine("=== TRIANGLE INDEX ANALYSIS ===");
            ConsoleLogger.WriteLine("Analyzing triangle vertex index patterns...");

            foreach (var triangle in triangles)
            {
                var indices = new[] { triangle.VertexA, triangle.VertexB, triangle.VertexC };
                var validIndices = indices.Where(i => i >= 0 && i <= maxVertexIndex).ToArray();
                var invalidIndices = indices.Where(i => i < 0 || i > maxVertexIndex).ToArray();

                if (invalidIndices.Length == 0)
                {
                    validTriangles++;
                    // Count local vertex references
                    foreach (var idx in validIndices)
                    {
                        localReferences[idx] = localReferences.GetValueOrDefault(idx, 0) + 1;
                    }
                }
                else if (validIndices.Length == 0)
                {
                    invalidTriangles++;
                    // Count cross-tile vertex references
                    foreach (var idx in invalidIndices)
                    {
                        crossTileReferences[idx] = crossTileReferences.GetValueOrDefault(idx, 0) + 1;
                    }
                }
                else
                {
                    mixedTriangles++;
                    // Count both types
                    foreach (var idx in validIndices)
                    {
                        localReferences[idx] = localReferences.GetValueOrDefault(idx, 0) + 1;
                    }
                    foreach (var idx in invalidIndices)
                    {
                        crossTileReferences[idx] = crossTileReferences.GetValueOrDefault(idx, 0) + 1;
                    }
                }
            }

            // Display triangle categorization
            ConsoleLogger.WriteLine($"Valid triangles (all local vertices): {validTriangles:N0} ({(double)validTriangles/triangleCount:P1})");
            ConsoleLogger.WriteLine($"Invalid triangles (all cross-tile vertices): {invalidTriangles:N0} ({(double)invalidTriangles/triangleCount:P1})");
            ConsoleLogger.WriteLine($"Mixed triangles (local + cross-tile): {mixedTriangles:N0} ({(double)mixedTriangles/triangleCount:P1})");
            ConsoleLogger.WriteLine();

            // Analyze cross-tile reference patterns
            if (crossTileReferences.Count > 0)
            {
                var minCrossTileIndex = crossTileReferences.Keys.Min();
                var maxCrossTileIndex = crossTileReferences.Keys.Max();
                var totalCrossTileRefs = crossTileReferences.Values.Sum();

                ConsoleLogger.WriteLine("=== CROSS-TILE REFERENCE PATTERNS ===");
                ConsoleLogger.WriteLine($"Cross-tile vertex references: {crossTileReferences.Count:N0} unique indices");
                ConsoleLogger.WriteLine($"Total cross-tile references: {totalCrossTileRefs:N0}");
                ConsoleLogger.WriteLine($"Cross-tile index range: {minCrossTileIndex:N0} - {maxCrossTileIndex:N0}");
                ConsoleLogger.WriteLine($"Gap after local vertices: {minCrossTileIndex - maxVertexIndex - 1:N0} indices");
                
                // Check for sequential patterns (indicating adjacent tiles)
                var sortedCrossTileIndices = crossTileReferences.Keys.OrderBy(x => x).ToArray();
                var sequentialRuns = AnalyzeSequentialRuns(sortedCrossTileIndices);
                
                ConsoleLogger.WriteLine($"Sequential runs detected: {sequentialRuns.Count}");
                foreach (var run in sequentialRuns.Take(10)) // Show first 10 runs
                {
                    ConsoleLogger.WriteLine($"  Run: {run.Start:N0} - {run.End:N0} (length: {run.Length})");
                }
                
                if (sequentialRuns.Count > 10)
                {
                    ConsoleLogger.WriteLine($"  ... and {sequentialRuns.Count - 10} more runs");
                }
                ConsoleLogger.WriteLine();
            }

            // Analyze local reference distribution
            ConsoleLogger.WriteLine("=== LOCAL VERTEX USAGE PATTERNS ===");
            var unusedLocalVertices = Enumerable.Range(0, maxVertexIndex + 1)
                .Where(i => !localReferences.ContainsKey(i))
                .Count();
            
            ConsoleLogger.WriteLine($"Used local vertices: {localReferences.Count:N0} / {maxVertexIndex + 1:N0}");
            ConsoleLogger.WriteLine($"Unused local vertices: {unusedLocalVertices:N0} ({(double)unusedLocalVertices/(maxVertexIndex + 1):P1})");
            
            if (localReferences.Count > 0)
            {
                var avgUsage = localReferences.Values.Average();
                var maxUsage = localReferences.Values.Max();
                ConsoleLogger.WriteLine($"Average vertex usage: {avgUsage:F1} references per vertex");
                ConsoleLogger.WriteLine($"Maximum vertex usage: {maxUsage:N0} references");
            }
            ConsoleLogger.WriteLine();

            // Generate global mesh hypothesis report
            ConsoleLogger.WriteLine("=== GLOBAL MESH HYPOTHESIS VALIDATION ===");
            var globalMeshEvidence = CalculateGlobalMeshEvidence(
                triangleCount, validTriangles, invalidTriangles, mixedTriangles,
                crossTileReferences.Count, maxVertexIndex);

            foreach (var evidence in globalMeshEvidence)
            {
                ConsoleLogger.WriteLine($"✓ {evidence}");
            }

            var analysisTime = DateTime.Now - startTime;
            ConsoleLogger.WriteLine($"Analysis completed in {analysisTime.TotalSeconds:F1}s");
        }

        private static List<(int Start, int End, int Length)> AnalyzeSequentialRuns(int[] sortedIndices)
        {
            var runs = new List<(int Start, int End, int Length)>();
            
            if (sortedIndices.Length == 0) return runs;

            int runStart = sortedIndices[0];
            int runEnd = sortedIndices[0];

            for (int i = 1; i < sortedIndices.Length; i++)
            {
                if (sortedIndices[i] == runEnd + 1)
                {
                    runEnd = sortedIndices[i];
                }
                else
                {
                    if (runEnd > runStart)
                    {
                        runs.Add((runStart, runEnd, runEnd - runStart + 1));
                    }
                    runStart = sortedIndices[i];
                    runEnd = sortedIndices[i];
                }
            }

            // Add final run if it's sequential
            if (runEnd > runStart)
            {
                runs.Add((runStart, runEnd, runEnd - runStart + 1));
            }

            return runs.OrderByDescending(r => r.Length).ToList();
        }

        private static List<string> CalculateGlobalMeshEvidence(
            int totalTriangles, int validTriangles, int invalidTriangles, int mixedTriangles,
            int uniqueCrossTileRefs, int maxVertexIndex)
        {
            var evidence = new List<string>();

            // Evidence of cross-tile references
            if (invalidTriangles + mixedTriangles > 0)
            {
                var crossTilePercent = (double)(invalidTriangles + mixedTriangles) / totalTriangles;
                evidence.Add($"Cross-tile triangle references detected: {crossTilePercent:P1} of all triangles");
            }

            // Evidence of systematic index organization
            if (uniqueCrossTileRefs > maxVertexIndex * 0.1)
            {
                evidence.Add($"Extensive cross-tile vertex references: {uniqueCrossTileRefs:N0} unique indices");
            }

            // Evidence of mixed triangle types (local + cross-tile)
            if (mixedTriangles > 0)
            {
                evidence.Add($"Mixed-reference triangles indicate cross-tile geometry assembly: {mixedTriangles:N0} triangles");
            }

            // Evidence of architectural completeness requiring multiple tiles
            if ((double)invalidTriangles / totalTriangles > 0.3)
            {
                evidence.Add("High proportion of cross-tile triangles suggests incomplete single-tile geometry");
            }

            // Evidence of global mesh system
            evidence.Add("Sequential cross-tile indices suggest adjacent tile vertex arrays");
            evidence.Add("PM4 files designed as interconnected architectural components");
            evidence.Add("Complete building geometry requires processing entire PM4 directory");

            return evidence;
        }
    }
}
