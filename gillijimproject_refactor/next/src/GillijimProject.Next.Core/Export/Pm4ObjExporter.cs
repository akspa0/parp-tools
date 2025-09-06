using System;
using System.Globalization;
using System.IO;
using System.Text;
using GillijimProject.Next.Core.PM4;

namespace GillijimProject.Next.Core.Export;

public static class Pm4ObjExporter
{
    /// <summary>
    /// Write a simple OBJ file from a PM4 scene.
    /// If invertX is true, multiplies X by -1 and swaps face winding (a c b) to preserve orientation.
    /// </summary>
    public static void Write(Pm4Scene scene, string path, bool invertX = false)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        using var sw = new StreamWriter(path, false, Encoding.ASCII);
        sw.NewLine = "\n";
        sw.WriteLine("# PM4 OBJ (Next)");

        // Vertices
        foreach (var v in scene.Vertices)
        {
            var x = invertX ? -v.X : v.X;
            sw.Write("v ");
            sw.Write(x.ToString("G9", CultureInfo.InvariantCulture));
            sw.Write(' ');
            sw.Write(v.Y.ToString("G9", CultureInfo.InvariantCulture));
            sw.Write(' ');
            sw.Write(v.Z.ToString("G9", CultureInfo.InvariantCulture));
            sw.WriteLine();
        }

        // Faces from explicit triangles if present
        if (scene.Triangles.Count > 0)
        {
            foreach (var t in scene.Triangles)
            {
                if (invertX)
                {
                    sw.WriteLine($"f {t.A + 1} {t.C + 1} {t.B + 1}");
                }
                else
                {
                    sw.WriteLine($"f {t.A + 1} {t.B + 1} {t.C + 1}");
                }
            }
            return;
        }

        // Fallback: faces from flat index buffer in triplets
        for (int i = 0; i + 2 < scene.Indices.Count; i += 3)
        {
            int a = scene.Indices[i];
            int b = scene.Indices[i + 1];
            int c = scene.Indices[i + 2];
            if (invertX)
            {
                sw.WriteLine($"f {a + 1} {c + 1} {b + 1}");
            }
            else
            {
                sw.WriteLine($"f {a + 1} {b + 1} {c + 1}");
            }
        }
    }
}
