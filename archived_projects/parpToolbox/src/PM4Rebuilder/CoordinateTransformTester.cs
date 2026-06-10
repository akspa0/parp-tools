using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder
{
    /// <summary>
    /// Diagnostic tool to test different MSCN coordinate transformations
    /// and find the correct one that unifies MSCN with MSVT geometry.
    /// </summary>
    internal static class CoordinateTransformTester
    {
        public static void TestTransforms(Pm4Scene scene, string outputDir)
        {
            Console.WriteLine($"Testing MSCN coordinate transformations...");
            Console.WriteLine($"MSVT vertices: {scene.Vertices.Count}, MSCN vertices: {scene.MscnVertices.Count}");

            Directory.CreateDirectory(outputDir);

            // Test different MSCN transformations
            var transforms = new (string name, Func<Vector3, Vector3> transform)[]
            {
                ("identity", (Vector3 v) => v),
                ("flip_x", (Vector3 v) => new Vector3(-v.X, v.Y, v.Z)),
                ("flip_y", (Vector3 v) => new Vector3(v.X, -v.Y, v.Z)),
                ("flip_z", (Vector3 v) => new Vector3(v.X, v.Y, -v.Z)),
                ("flip_xy", (Vector3 v) => new Vector3(-v.X, -v.Y, v.Z)),
                ("flip_xz", (Vector3 v) => new Vector3(-v.X, v.Y, -v.Z)),
                ("flip_yz", (Vector3 v) => new Vector3(v.X, -v.Y, -v.Z)),
                ("flip_xyz", (Vector3 v) => new Vector3(-v.X, -v.Y, -v.Z)),
                ("swap_yz", (Vector3 v) => new Vector3(v.X, v.Z, v.Y)),
                ("swap_yz_flip_x", (Vector3 v) => new Vector3(-v.X, v.Z, v.Y)),
                ("swap_yz_flip_y", (Vector3 v) => new Vector3(v.X, -v.Z, v.Y)),
                ("swap_yz_flip_z", (Vector3 v) => new Vector3(v.X, v.Z, -v.Y)),
                ("current_msvt", (Vector3 v) => { var t = new Vector3(v.X, v.Z, v.Y); t.X = -t.X; t.Y = -t.Y; return t; }),
                ("inverse_msvt", (Vector3 v) => { var t = new Vector3(-v.X, -v.Y, v.Z); return new Vector3(t.X, t.Z, t.Y); }),
            };

            foreach (var transformPair in transforms)
            {
                string name = transformPair.name;
                var transform = transformPair.transform;
                
                string objPath = Path.Combine(outputDir, $"mscn_transform_{name}.obj");
                
                using var sw = new StreamWriter(objPath);
                sw.WriteLine($"# MSCN vertices with {name} transformation");
                sw.WriteLine($"# Original MSVT transform: swap Y/Z, flip X, flip Y");
                sw.WriteLine();

                // Write MSVT vertices first (as reference)
                sw.WriteLine("# MSVT vertices (reference geometry)");
                foreach (var v in scene.Vertices)
                {
                    // Apply current MSVT transformation
                    var transformed = new Vector3(v.X, v.Z, v.Y);
                    transformed.X = -transformed.X;
                    transformed.Y = -transformed.Y;
                    sw.WriteLine($"v {transformed.X:F6} {transformed.Y:F6} {transformed.Z:F6}");
                }

                // Write MSCN vertices with test transformation
                sw.WriteLine("# MSCN vertices (test transformation)");
                foreach (var v in scene.MscnVertices)
                {
                    var transformed = transform(v);
                    sw.WriteLine($"v {transformed.X:F6} {transformed.Y:F6} {transformed.Z:F6}");
                }

                Console.WriteLine($"Generated: {objPath}");
            }

            // Generate bounds analysis
            string boundsPath = Path.Combine(outputDir, "coordinate_bounds_analysis.txt");
            using var bw = new StreamWriter(boundsPath);
            
            bw.WriteLine("MSVT Bounds Analysis:");
            AnalyzeBounds(scene.Vertices, "MSVT", bw);
            
            bw.WriteLine("\nMSCN Bounds Analysis:");
            AnalyzeBounds(scene.MscnVertices, "MSCN", bw);

            Console.WriteLine($"Bounds analysis: {boundsPath}");
        }

        private static void AnalyzeBounds(IList<Vector3> vertices, string label, StreamWriter writer)
        {
            if (vertices.Count == 0)
            {
                writer.WriteLine($"{label}: No vertices");
                return;
            }

            var min = vertices[0];
            var max = vertices[0];

            foreach (var v in vertices)
            {
                if (v.X < min.X) min = new Vector3(v.X, min.Y, min.Z);
                if (v.Y < min.Y) min = new Vector3(min.X, v.Y, min.Z);
                if (v.Z < min.Z) min = new Vector3(min.X, min.Y, v.Z);
                
                if (v.X > max.X) max = new Vector3(v.X, max.Y, max.Z);
                if (v.Y > max.Y) max = new Vector3(max.X, v.Y, max.Z);
                if (v.Z > max.Z) max = new Vector3(max.X, max.Y, v.Z);
            }

            writer.WriteLine($"{label} Min: {min.X:F2}, {min.Y:F2}, {min.Z:F2}");
            writer.WriteLine($"{label} Max: {max.X:F2}, {max.Y:F2}, {max.Z:F2}");
            writer.WriteLine($"{label} Size: {max.X - min.X:F2} x {max.Y - min.Y:F2} x {max.Z - min.Z:F2}");
            writer.WriteLine($"{label} Center: {(min.X + max.X) / 2:F2}, {(min.Y + max.Y) / 2:F2}, {(min.Z + max.Z) / 2:F2}");
            writer.WriteLine($"{label} Count: {vertices.Count}");
        }
    }
}
