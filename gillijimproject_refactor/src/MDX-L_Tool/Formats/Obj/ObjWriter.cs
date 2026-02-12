using System.Globalization;
using System.Text;
using MdxLTool.Formats.Mdx;

namespace MdxLTool.Formats.Obj;

/// <summary>
/// Writes an MdxFile's geometry to Wavefront OBJ format.
/// Supports per-geoset splitting and MTL material references.
/// </summary>
public class ObjWriter
{
    /// <summary>
    /// Map of texture ID → exported texture file path (for MTL generation).
    /// </summary>
    public Dictionary<int, string> ExportedTextures { get; set; } = new();

    /// <summary>
    /// Write an MdxFile to OBJ (and optional MTL).
    /// </summary>
    /// <param name="mdx">Source MDX file.</param>
    /// <param name="path">Output .obj file path.</param>
    /// <param name="split">If true, write one OBJ per geoset.</param>
    public void Write(MdxFile mdx, string path, bool split = false)
    {
        if (split)
        {
            for (int g = 0; g < mdx.Geosets.Count; g++)
            {
                var ext = Path.GetExtension(path);
                var stem = Path.ChangeExtension(path, null);
                var geoPath = $"{stem}_geo{g}{ext}";
                WriteGeosets(mdx, geoPath, new[] { g });
            }
        }
        else
        {
            WriteGeosets(mdx, path, Enumerable.Range(0, mdx.Geosets.Count).ToArray());
        }
    }

    private void WriteGeosets(MdxFile mdx, string path, int[] geosetIndices)
    {
        var mtlPath = Path.ChangeExtension(path, ".mtl");
        var mtlName = Path.GetFileName(mtlPath);

        using var sw = new StreamWriter(path, false, Encoding.ASCII);
        sw.WriteLine($"# MDX-L_Tool OBJ export");
        sw.WriteLine($"# Geosets: {string.Join(", ", geosetIndices)}");

        if (ExportedTextures.Count > 0)
            sw.WriteLine($"mtllib {mtlName}");

        int vertexOffset = 1; // OBJ is 1-indexed

        foreach (var gi in geosetIndices)
        {
            if (gi < 0 || gi >= mdx.Geosets.Count) continue;
            var geo = mdx.Geosets[gi];

            sw.WriteLine($"g geoset_{gi}");

            // Assign material if available
            if (gi < mdx.Materials.Count && mdx.Materials[gi].Layers.Count > 0)
            {
                var texId = mdx.Materials[gi].Layers[0].TextureId;
                if (ExportedTextures.ContainsKey(texId))
                    sw.WriteLine($"usemtl mat_{texId}");
            }

            // Vertices
            foreach (var v in geo.Vertices)
                sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "v {0:F6} {1:F6} {2:F6}", v.X, v.Y, v.Z));

            // Normals
            foreach (var n in geo.Normals)
                sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "vn {0:F6} {1:F6} {2:F6}", n.X, n.Y, n.Z));

            // UVs
            foreach (var uv in geo.TexCoords)
                sw.WriteLine(string.Format(CultureInfo.InvariantCulture, "vt {0:F6} {1:F6}", uv.U, uv.V));

            // Faces
            for (int i = 0; i + 2 < geo.Indices.Count; i += 3)
            {
                int a = geo.Indices[i] + vertexOffset;
                int b = geo.Indices[i + 1] + vertexOffset;
                int c = geo.Indices[i + 2] + vertexOffset;
                sw.WriteLine($"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
            }

            vertexOffset += geo.Vertices.Count;
        }

        // Write MTL file if textures are available
        if (ExportedTextures.Count > 0)
        {
            using var mtlSw = new StreamWriter(mtlPath, false, Encoding.ASCII);
            foreach (var (texId, texPath) in ExportedTextures)
            {
                mtlSw.WriteLine($"newmtl mat_{texId}");
                mtlSw.WriteLine($"map_Kd {texPath}");
                mtlSw.WriteLine();
            }
        }
    }
}
