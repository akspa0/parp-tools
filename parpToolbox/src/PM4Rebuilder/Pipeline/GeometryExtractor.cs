using System;
using System.Collections.Generic;
using System.Linq;
using PM4Rebuilder;

namespace PM4Rebuilder.Pipeline
{
    /// <summary>
    /// Step C – Cleans and deduplicates triangles for each building object.
    /// Follows the spec rules: skip degenerate triangles and exact duplicates.
    /// </summary>
    internal static class GeometryExtractor
    {
        public sealed record Stats(int RawTriangles, int KeptTriangles);

        public static Dictionary<Pm4ObjectAssembler.BuildingObject, Stats> CleanGeometry(List<Pm4ObjectAssembler.BuildingObject> objects)
        {
            var stats = new Dictionary<Pm4ObjectAssembler.BuildingObject, Stats>();

            foreach (var obj in objects)
            {
                var raw = obj.Triangles.Count;

                // Deduplicate & drop degenerate triangles in-place
                var unique = new HashSet<(int,int,int)>();
                obj.Triangles.RemoveAll(tri =>
                {
                    if (tri.A == tri.B || tri.A == tri.C || tri.B == tri.C)
                        return true; // degenerate
                    // sort indices to detect duplicates regardless of winding
                    var sorted = Sort(tri);
                    if (!unique.Add(sorted))
                        return true; // duplicate
                    return false; // keep
                });

                stats[obj] = new Stats(raw, obj.Triangles.Count);
            }

            // Log summary
            int totalRaw = stats.Values.Sum(s => s.RawTriangles);
            int totalKept = stats.Values.Sum(s => s.KeptTriangles);
            Console.WriteLine($"[GEOMETRY] Raw triangles: {totalRaw}  => Kept: {totalKept}  (removed {(totalRaw-totalKept)})");
            return stats;
        }

        private static (int,int,int) Sort((int A,int B,int C) tri)
        {
            int[] arr = { tri.A, tri.B, tri.C };
            Array.Sort(arr);
            return (arr[0], arr[1], arr[2]);
        }
    }
}
