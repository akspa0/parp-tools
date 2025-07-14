namespace ParpToolbox.Services.PM4;

using System.Globalization;
using System.IO;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Simple OBJ exporter for <see cref="Pm4Scene"/>. Writes single OBJ with default material.
/// </summary>
internal static class Pm4ObjExporter
{
    public static void Export(Pm4Scene scene, string filename)
    {
        var dir = Path.GetDirectoryName(filename) ?? ".";
        Directory.CreateDirectory(dir);
        var modelName = Path.GetFileNameWithoutExtension(filename);
        var objPath = Path.Combine(dir, modelName + ".obj");
        var mtlPath = Path.Combine(dir, modelName + ".mtl");

        using var objWriter = new StreamWriter(objPath);
        objWriter.WriteLine("# parpToolbox PM4 OBJ export");
        objWriter.WriteLine($"mtllib {modelName}.mtl");

        // vertices
        foreach (var v in scene.Vertices)
            objWriter.WriteLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");



        File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
    }
}
