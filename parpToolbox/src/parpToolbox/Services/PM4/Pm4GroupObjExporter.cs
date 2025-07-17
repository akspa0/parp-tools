namespace ParpToolbox.Services.PM4;

using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Exports each <see cref="SurfaceGroup"/> in a <see cref="Pm4Scene"/> to a separate OBJ file.
/// The file names will be &lt;baseName&gt;_&lt;group.Name&gt;.obj inside the specified output directory.
/// </summary>
internal static class Pm4GroupObjExporter
{
    /// <param name="scene">Scene to export – must contain populated <see cref="Pm4Scene.Groups"/>.</param>
    /// <param name="outputDir">Directory to place OBJ/MTL files in (will be created).</param>
    /// <param name="writeFaces">Write face definitions instead of point cloud when true.</param>
    /// <param name="flipX">Invert X coordinate to correct mirroring.</param>
    public static void Export(Pm4Scene scene, string outputDir, bool writeFaces = false, bool flipX = true)
    {
        Directory.CreateDirectory(outputDir);
        foreach (var group in scene.Groups)
        {
            if (group.Faces.Count == 0 && !writeFaces)
            {
                // nothing to export – skip empty groups unless point cloud desired
                if (scene.Vertices.Count == 0) continue;
            }

            string baseName = Path.Combine(outputDir, group.Name);
            string objPath = baseName + ".obj";
            string mtlPath = baseName + ".mtl";

            using var objWriter = new StreamWriter(objPath);
            objWriter.WriteLine("# parpToolbox PM4 group OBJ export");
            objWriter.WriteLine($"mtllib {Path.GetFileName(mtlPath)}");

            // Build a list of used vertex indices to minimise file size and remap indices.
            var vCount = scene.Vertices.Count;
            var used = new HashSet<int>();
            if (writeFaces)
            {
                foreach (var (a, b, c) in group.Faces)
                {
                    if (a < vCount) used.Add(a);
                    if (b < vCount) used.Add(b);
                    if (c < vCount) used.Add(c);
                }
            }
            else
            {
                // For point clouds, include *all* vertices referenced by indices (or fall back to global?)
                for (int i = 0; i < scene.Vertices.Count; i++)
                    used.Add(i);
            }

            // Provide a stable mapping originalIndex -> sequential OBJ index starting at 1
            var remap = new Dictionary<int, int>(used.Count);
            int nextIdx = 1;
            foreach (var idx in used)
            {
                remap[idx] = nextIdx++;
                var v = scene.Vertices[idx];
                float x = flipX ? -v.X : v.X;
                objWriter.WriteLine($"v {x.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
            }

            objWriter.WriteLine("usemtl default");

            if (writeFaces && group.Faces.Count > 0)
            {
                foreach (var (a, b, c) in group.Faces)
                {
                    if (remap.TryGetValue(a, out var ra) &&
                        remap.TryGetValue(b, out var rb) &&
                        remap.TryGetValue(c, out var rc))
                    {
                        objWriter.WriteLine($"f {ra} {rb} {rc}");
                    }
                }
            }
            else
            {
                foreach (var original in used)
                    objWriter.WriteLine($"p {remap[original]}");
            }

            File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
        }
    }
}
