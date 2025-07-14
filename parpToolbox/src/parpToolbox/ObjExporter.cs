using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ParpToolbox.Formats.WMO;

namespace ParpToolbox
{
    public static class ObjExporter
    { 
        public static void Export(IReadOnlyList<WmoGroup> groups, string filename, bool includeInvisible = false)
        {
            ExportInternal(groups, filename, includeInvisible);
        }

        public static void ExportPerGroup(IReadOnlyList<WmoGroup> groups, string outputDirectory, bool includeInvisible = false)
        {
            Directory.CreateDirectory(outputDirectory);
            foreach (var group in groups)
            {
                if (!includeInvisible && !WmoVisibility.IsRenderable(group.RawFlags))
                    continue;

                var safeName = group.Name.Replace("/", "_").Replace("\\", "_");
                var filePath = Path.Combine(outputDirectory, $"{safeName}.obj");
                ExportInternal([group], filePath, true); // single group, no further filtering
            }
        }

        private static void ExportInternal(IReadOnlyList<WmoGroup> groups, string filename, bool includeInvisible)
        {
            var dir = Path.GetDirectoryName(filename) ?? ".";
            Directory.CreateDirectory(dir);
            var modelName = Path.GetFileNameWithoutExtension(filename);
            var objPath = Path.Combine(dir, modelName + ".obj");
            var mtlPath = Path.Combine(dir, modelName + ".mtl");

            using var objWriter = new StreamWriter(objPath);
            objWriter.WriteLine("# parpToolbox OBJ export");
            objWriter.WriteLine($"mtllib {modelName}.mtl");

            int vertexOffset = 1;
            foreach (var group in groups)
            {
                if (!includeInvisible && !WmoVisibility.IsRenderable(group.RawFlags))
                    continue;

                objWriter.WriteLine($"\ng {group.Name}");

                // Vertices
                foreach (var v in group.Vertices)
                    objWriter.WriteLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");

                // Faces (triangles)
                foreach (var (i1, i2, i3) in group.Faces)
                    objWriter.WriteLine($"f {i1 + vertexOffset} {i2 + vertexOffset} {i3 + vertexOffset}");

                vertexOffset += group.Vertices.Count;
            }

            // rudimentary MTL
            File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
        }

        // (no-op helper removed)
    }
}
