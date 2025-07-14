namespace ParpToolbox.Services.PM4;

using System.Globalization;
using System.IO;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Simple OBJ exporter for <see cref="Pm4Scene"/>. Writes single OBJ with default material.
/// </summary>
internal static class Pm4ObjExporter
{
    /// <summary>
    /// Exports the given scene to Wavefront OBJ.
    /// </summary>
    /// <param name="scene">Scene to export.</param>
    /// <param name="filename">Target OBJ (or any path without extension).</param>
    /// <param name="writeFaces">If true, writes face definitions; otherwise points only.</param>
    public static void Export(Pm4Scene scene, string filename, bool writeFaces = false, bool flipX = true)
    {
        var dir = Path.GetDirectoryName(filename) ?? ".";
        Directory.CreateDirectory(dir);
        var modelName = Path.GetFileNameWithoutExtension(filename);
        var objPath = Path.Combine(dir, modelName + ".obj");
        var mtlPath = Path.Combine(dir, modelName + ".mtl");

        using var objWriter = new StreamWriter(objPath);
        objWriter.WriteLine("# parpToolbox PM4 OBJ export");
        objWriter.WriteLine($"mtllib {modelName}.mtl");

        // vertices (optional X flip)
        foreach (var v in scene.Vertices)
        {
            float x = flipX ? -v.X : v.X;
            objWriter.WriteLine($"v {x.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
        }

        // default material for now
        objWriter.WriteLine("usemtl default");

        if (writeFaces && scene.Triangles.Count > 0)
        {
            // faces (OBJ uses 1-based indices)
            foreach (var (a, b, c) in scene.Triangles)
                objWriter.WriteLine($"f {a + 1} {b + 1} {c + 1}");
        }
        else
        {
            // Write as point cloud (Meshlab-friendly)
            for (int i = 0; i < scene.Vertices.Count; i++)
                objWriter.WriteLine($"p {i + 1}");
        }

        File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
    }
}
