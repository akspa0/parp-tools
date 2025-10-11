using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder
{
    /// <summary>
    /// Writes a single Wavefront OBJ file that contains all assembled PM4 objects as separate groups.
    /// Each group name matches <see cref="Pm4ObjectAssembler.BuildingObject.Name"/>. Vertices are de-duplicated
    /// globally so that faces reference a single shared vertex list, allowing easy import as one scene in MeshLab/Blender.
    /// </summary>
    internal static class CombinedObjectExporter
    {
        public static void Export(IEnumerable<Pm4ObjectAssembler.BuildingObject> objects, Pm4Scene scene, string filePath)
        {
            // Collect unique vertex indices across all triangles first so ordering is stable.
            var unique = new HashSet<int>();
            foreach (var obj in objects)
            {
                foreach (var (a, b, c) in obj.Triangles)
                {
                    unique.Add(a - 1);
                    unique.Add(b - 1);
                    unique.Add(c - 1);
                }
            }
            var ordered = unique.OrderBy(i => i).ToList();
            var globalRemap = new Dictionary<int, int>(ordered.Count);
            int nextIdx = 1;
            foreach (int old in ordered)
                globalRemap[old] = nextIdx++;

            using var sw = new StreamWriter(filePath);
            sw.WriteLine("# PM4Rebuilder combined OBJ");
            sw.WriteLine($"# Objects: {objects.Count()}  Vertices: {ordered.Count}");
            sw.WriteLine();

            // Write vertices in ascending order so remap is deterministic
            foreach (var oldIdx in ordered)
            {
                Vector3 v;
                if (oldIdx < scene.Vertices.Count)
                {
                    v = scene.Vertices[oldIdx];
                }
                else
                {
                    int mscnIdx = oldIdx - scene.Vertices.Count;
                    v = (mscnIdx >= 0 && mscnIdx < scene.MscnVertices.Count) ? scene.MscnVertices[mscnIdx] : Vector3.Zero;
                }
                sw.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");
            }
            sw.WriteLine();

            // Write groups sequentially
            foreach (var obj in objects)
            {
                if (obj.Triangles.Count == 0) continue;
                sw.WriteLine($"g {obj.Name}");
                foreach (var (a, b, c) in obj.Triangles)
                {
                    sw.WriteLine($"f {globalRemap[a - 1]} {globalRemap[b - 1]} {globalRemap[c - 1]}");
                }
            }
        }
    }
}
