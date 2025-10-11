using System.Globalization;
using System.IO;
using System.Text;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.Export
{
    public class ObjExporter : IObjExporter
    {
        public void Export(RenderMesh mesh, string filePath)
        {
            if (mesh == null || mesh.Vertices.Count == 0 || mesh.Faces.Count == 0)
            {
                return; // Do not create an empty file.
            }

            var sb = new StringBuilder();
            var culture = CultureInfo.InvariantCulture; // Ensures '.' is used as the decimal separator.

            // Write Vertices
            foreach (var vertex in mesh.Vertices)
            {
                sb.AppendLine($"v {vertex.X.ToString(culture)} {vertex.Y.ToString(culture)} {vertex.Z.ToString(culture)}");
            }

            // Write Normals
            foreach (var normal in mesh.Normals)
            {
                sb.AppendLine($"vn {normal.X.ToString(culture)} {normal.Y.ToString(culture)} {normal.Z.ToString(culture)}");
            }

            // Write Faces (OBJ format is 1-based)
            for (int i = 0; i < mesh.Faces.Count; i += 3)
            {
                var i1 = mesh.Faces[i] + 1;
                var i2 = mesh.Faces[i + 1] + 1;
                var i3 = mesh.Faces[i + 2] + 1;

                // Format: f v1//vn1 v2//vn2 v3//vn3 (assuming vertex and normal indices are the same)
                sb.AppendLine($"f {i1}//{i1} {i2}//{i2} {i3}//{i3}");
            }

            File.WriteAllText(filePath, sb.ToString());
        }
    }
}
